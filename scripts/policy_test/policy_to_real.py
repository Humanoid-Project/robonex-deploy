#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "sim_to_real"))
sys.path.insert(0, str(THIS_FILE.parent))

try:
    import can
    import numpy as np
    import onnxruntime as ort
except ImportError as error:
    raise SystemExit(f"Missing required package: {error}")

try:
    import n100
except ImportError as error:
    raise SystemExit(
        f"N100 IMU extension not importable: {error}\n"
        "Build it first:\n"
        f"  cd {THIS_FILE.parent}\n"
        "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build -j"
    )

from robonex_can import (
    DEFAULT_INTERFACE,
    HOST_ID,
    JOINT_LIMITS_RAD,
    MECH_POS_INDEX,
    MOTOR_MODELS,
    Motor,
    SPECS,
    clamp,
)
from robonex_common.imu import DEFAULT_IMU_BAUDRATE, DEFAULT_IMU_PORT, MOUNT_ROLL_DEG
from robonex_common.joints import CHANNEL_MOTOR_IDS, JOINT_BY_MODEL_NAME
from robonex_common.motors import MOTOR_CONTROL_KD, MOTOR_CONTROL_KP
from robonex_common.paths import COMMON_REPO_NAMES, git_commit, resolve_repo
from robonex_common.policy import PolicyContract, python_source_sha256
from robonex_common.protocol import MECHANICAL_VELOCITY_INDEX
from robonex_common.runtime import ActionPipeline, assemble_observation
from safety import (
    AxisLimiter,
    align_angle,
    brake_and_stop,
    enable_with_runtime_feedback,
    inspect_zero_positions,
    open_hardware,
    runtime_safety_reason,
    wrap_to_pi,
)

DEG = math.pi / 180.0
CLEAR_SCREEN = "\033[2J\033[3J\033[H"


class TeeStream:
    def __init__(self, terminal, log, lock):
        self.terminal = terminal
        self.log = log
        self.lock = lock

    def write(self, value):
        with self.lock:
            written = self.terminal.write(value)
            self.log.write(value.replace(CLEAR_SCREEN, ""))
            self.log.flush()
        return written

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@dataclass(frozen=True)
class Settings:
    interface: str = DEFAULT_INTERFACE
    host_id: int = HOST_ID
    kp: float = MOTOR_CONTROL_KP
    kd: float = MOTOR_CONTROL_KD

    read_poll_timeout: float = 0.02
    read_print_hz: float = 10.0

    status_hz: float = 10.0
    feedback_timeout: float = 0.10
    imu_stale_timeout: float = 0.10
    overspeed: float = 10.0
    max_temp: float = 70.0
    max_error_deg: float = 25.0
    max_raw_action: float = 20.0

    approach_max_speed: float = 0.30
    approach_max_accel: float = 0.60
    approach_tolerance_deg: float = 6.0
    approach_settle_timeout: float = 5.0

    policy_max_speed: float = 6.0
    policy_max_accel: float = 120.0
    ramp_seconds: float = 1.0

    imu_calibration_seconds: float = 2.0
    brake_time: float = 0.20


SETTINGS = Settings()


def joint_row_order(contract):
    return [(name, JOINT_BY_MODEL_NAME[name].motor_id) for name in contract.joint_order]


class MechPosReader(threading.Thread):
    def __init__(self, channel, motor_ids, settings, notes):
        super().__init__(daemon=True)
        self.channel = channel
        self.motor_ids = tuple(motor_ids)
        self.settings = settings
        self.notes = notes
        self.rate_hz = 0.0
        self._state = {motor_id: (None, None) for motor_id in motor_ids}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def run(self):
        try:
            bus = can.Bus(channel=self.channel, interface=self.settings.interface)
        except OSError as error:
            self.notes.append(
                f"[{self.channel}] open failed: {error}  "
                f"(sudo ip link set {self.channel} up type can bitrate 1000000)"
            )
            return
        motors = {
            motor_id: Motor(bus, motor_id, SPECS[MOTOR_MODELS[motor_id]], host_id=self.settings.host_id)
            for motor_id in self.motor_ids
        }
        started, cycles = time.monotonic(), 0
        try:
            while not self._stop_event.is_set():
                for motor_id in self.motor_ids:
                    motor = motors[motor_id]
                    position = motor.read_parameter(MECH_POS_INDEX, timeout=self.settings.read_poll_timeout)
                    velocity = motor.read_parameter(
                        MECHANICAL_VELOCITY_INDEX, timeout=self.settings.read_poll_timeout
                    )
                    with self._lock:
                        self._state[motor_id] = (position, velocity)
                cycles += 1
                now = time.monotonic()
                if now - started >= 0.5:
                    self.rate_hz = cycles / (now - started)
                    started, cycles = now, 0
        except can.CanError as error:
            self.notes.append(f"[{self.channel}] CAN error: {error}")
        finally:
            bus.shutdown()


class ReadOnlyJointSource:
    label = "mechPos poll (type 0x11, read-only)"

    def __init__(self, settings, notes):
        self.readers = [
            MechPosReader(channel, motor_ids, settings, notes)
            for channel, motor_ids in CHANNEL_MOTOR_IDS.items()
        ]

    def start(self):
        for reader in self.readers:
            reader.start()

    def stop(self):
        for reader in self.readers:
            reader.stop()
        for reader in self.readers:
            reader.join(timeout=2.0)

    def poll(self):
        return None

    def snapshot(self):
        merged = {}
        for reader in self.readers:
            merged.update(reader.snapshot())
        return merged

    def rate_text(self):
        return "  ".join(f"{reader.channel} {reader.rate_hz:5.1f} Hz" for reader in self.readers)


class RuntimeJointSource:
    label = "type 0x02 runtime feedback"

    def __init__(self, motors, hubs):
        self.motors = motors
        self.hubs = hubs
        self._cycles = 0
        self._started = time.monotonic()
        self.rate_hz = 0.0

    def start(self):
        return None

    def stop(self):
        return None

    def poll(self):
        for hub in self.hubs.values():
            hub.pump()
        self._cycles += 1
        now = time.monotonic()
        if now - self._started >= 0.5:
            self.rate_hz = self._cycles / (now - self._started)
            self._started, self._cycles = now, 0

    def snapshot(self):
        return {
            motor_id: (motor.last_position, motor.last_velocity)
            for motor_id, motor in self.motors.items()
        }

    def rate_text(self):
        return f"loop {self.rate_hz:5.1f} Hz"


class ImuSource:
    def __init__(self, settings, notes):
        self.settings = settings
        self.notes = notes
        self.status = "not started"
        self.driver = None
        self.bias_raw = None
        self._last_seq = None
        self._last_seq_time = 0.0

    def start(self, calibrate):
        self.driver = n100.ImuDriver(
            n100.DriverConfig(
                port=DEFAULT_IMU_PORT,
                baudrate=DEFAULT_IMU_BAUDRATE,
                mount_rotation=n100.Quat.from_axis_angle_x(MOUNT_ROLL_DEG * DEG),
            )
        )
        try:
            self.driver.start()
        except RuntimeError as error:
            self.status = "start failed"
            self.notes.append(f"[IMU] {error}")
            self.notes.append(
                f"      ls /dev/ttyUSB* /dev/ttyACM*  (permissions: sudo chmod 666 {DEFAULT_IMU_PORT})"
            )
            return False
        if self.driver.wait_for_sample(timeout=3.0) is None:
            self.status = "no sample in 3 s"
            self.notes.append(f"[IMU] {self.driver.last_error() or 'unknown error'}")
            return False
        if calibrate:
            print(
                f"Calibrating the gyro bias for {self.settings.imu_calibration_seconds:.1f} s. "
                "Keep the robot completely still."
            )
            self.driver.calibrate_gyro_bias(self.settings.imu_calibration_seconds)
            self.bias_raw = self.driver.gyro_bias_raw
            print(
                f"  raw gyro bias  x {self.bias_raw.x:+.6f}  y {self.bias_raw.y:+.6f}  "
                f"z {self.bias_raw.z:+.6f}  [rad/s]"
            )
        self.status = "ready"
        self._last_seq = None
        self._last_seq_time = time.monotonic()
        return True

    def stop(self):
        if self.driver is not None:
            self.driver.stop()

    def read(self, now):
        sample = None if self.driver is None else self.driver.latest()
        if sample is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), None
        if sample.seq != self._last_seq:
            self._last_seq = sample.seq
            self._last_seq_time = now
        angular_velocity = sample.angular_velocity_raw
        gravity = sample.projected_gravity
        return (
            (angular_velocity.x, angular_velocity.y, angular_velocity.z),
            (gravity.x, gravity.y, gravity.z),
            now - self._last_seq_time,
        )

    def failure_reason(self, age):
        if self.driver is None or not self.driver.is_running:
            return f"IMU reader stopped ({self.driver.last_error() if self.driver else 'not started'})"
        if age is None:
            return "IMU produced no sample"
        if age > self.settings.imu_stale_timeout:
            return f"IMU sample is stale ({age:.3f} s > {self.settings.imu_stale_timeout:.3f} s)"
        return None


class PolicyRunner:
    def __init__(self, policy_path, contract):
        self.contract = contract
        self.pipeline = ActionPipeline(contract)
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[-1] not in (None, "None", contract.observation_size):
            raise RuntimeError(f"Policy input size does not match the manifest: {inputs[0].shape}")
        if len(outputs) != 1 or outputs[0].shape[-1] not in (None, "None", contract.action_size):
            raise RuntimeError(f"Policy output size does not match the manifest: {outputs[0].shape}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
        self.offsets = np.asarray(contract.action_offsets, dtype=np.float32)
        self.last_action = np.zeros(contract.action_size, dtype=np.float32)
        self.inference_ms = 0.0

    def reset(self):
        self.last_action.fill(0.0)

    def joint_state(self, snapshot):
        size = self.contract.action_size
        positions = np.zeros(size, dtype=np.float32)
        velocities = np.zeros(size, dtype=np.float32)
        missing = []
        for index, motor_id in enumerate(self.motor_ids):
            position, velocity = snapshot.get(motor_id, (None, None))
            if position is None or not math.isfinite(position):
                missing.append(motor_id)
                continue
            positions[index] = wrap_to_pi(position)
            velocities[index] = velocity if velocity is not None and math.isfinite(velocity) else 0.0
        return positions, velocities, missing

    def observation(self, positions, velocities, angular_velocity, gravity):
        return assemble_observation(
            positions - self.offsets,
            velocities,
            angular_velocity,
            gravity,
            self.last_action,
            expected_size=self.contract.observation_size,
        )

    def step(self, observation, commit):
        started = time.perf_counter()
        raw_action = self.session.run(
            [self.output_name], {self.input_name: observation.reshape(1, -1)}
        )[0][0]
        self.inference_ms = (time.perf_counter() - started) * 1000.0
        action, targets = self.pipeline.apply(raw_action, max_raw_action=SETTINGS.max_raw_action)
        if commit:
            self.last_action[:] = action
        return np.asarray(raw_action, dtype=np.float32), action, targets


class TargetCommander:
    def __init__(self, motors, start_positions, settings):
        self.motors = motors
        self.settings = settings
        self.limiters = {
            motor_id: AxisLimiter(start_positions[motor_id]) for motor_id in motors
        }
        self.commands = dict(start_positions)
        self.velocities = {motor_id: 0.0 for motor_id in motors}
        self.slew_limited_count = 0
        self.command_count = 0

    def send(self, motor_ids_targets, dt, max_speed, max_accel):
        for motor_id, target in motor_ids_targets.items():
            limiter = self.limiters[motor_id]
            aligned = align_angle(limiter.position, target)
            position, velocity = limiter.step(aligned, dt, max_speed, max_accel)
            if abs(position - aligned) > 1.0e-9:
                self.slew_limited_count += 1
            self.command_count += 1
            self.commands[motor_id] = position
            self.velocities[motor_id] = velocity
        for motor_id, motor in self.motors.items():
            motor.control(
                pos=self.commands[motor_id],
                vel=self.velocities[motor_id],
                kp=self.settings.kp,
                kd=self.settings.kd,
                torque=0.0,
            )

    def at_rest(self, targets, tolerance):
        return all(
            abs(self.commands[motor_id] - align_angle(self.commands[motor_id], target)) <= 1.0e-12
            and abs(self.velocities[motor_id]) <= 1.0e-12
            for motor_id, target in targets.items()
        ) and all(
            self.motors[motor_id].last_position is not None
            and abs(wrap_to_pi(self.motors[motor_id].last_position) - target) <= tolerance
            for motor_id, target in targets.items()
        )


@dataclass
class LoopStats:
    steps: int = 0
    period_max: float = 0.0
    inference_ms_max: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def record(self, period, inference_ms):
        self.steps += 1
        self.period_max = max(self.period_max, period)
        self.inference_ms_max = max(self.inference_ms_max, inference_ms)

    def elapsed(self):
        return time.monotonic() - self.started


def resolve_contract(policy_path):
    manifest_path = policy_path.with_name("policy_manifest.json")
    contract = PolicyContract.load(manifest_path)
    manifest_policy = contract.verify_policy(manifest_path)
    if manifest_policy.resolve() != policy_path:
        raise ValueError(f"policy_manifest.json selects {manifest_policy.name}, not {policy_path.name}")
    return contract


def verify_common_source(contract, notes):
    try:
        common_root = resolve_repo(COMMON_REPO_NAMES, "ROBONEX_COMMON_ROOT")
    except FileNotFoundError:
        notes.append(
            "[env] robonex-common checkout not found; the manifest's source fingerprint was not verified"
        )
        return
    actual_commit = git_commit(common_root)
    if actual_commit != contract.common_commit:
        raise RuntimeError(
            f"robonex-common commit mismatch: manifest={contract.common_commit}, checkout={actual_commit}"
        )
    actual_sha256 = python_source_sha256(common_root, ("src/robonex_common",))
    if actual_sha256 != contract.common_sha256:
        raise RuntimeError(
            f"robonex-common source mismatch: manifest={contract.common_sha256}, checkout={actual_sha256}"
        )


def format_joint_table(contract, positions, velocities, raw_action, targets, commands):
    lines = [
        f"  {'ID':>3}  {'joint':<20}  {'pos':>9}  {'rel':>9}  {'vel':>10}  "
        f"{'raw':>8}  {'target':>9}  {'command':>9}"
    ]
    lines.append("  " + "-" * 92)
    for index, (name, motor_id) in enumerate(joint_row_order(contract)):
        offset = contract.action_offsets[index]
        command_text = (
            f"{math.degrees(commands[motor_id]):+8.2f}d" if commands is not None else f"{'--':>9}"
        )
        lines.append(
            f"  {motor_id:>3}  {name:<20}  "
            f"{math.degrees(positions[index]):+8.2f}d  "
            f"{math.degrees(positions[index] - offset):+8.2f}d  "
            f"{velocities[index]:+9.3f}  "
            f"{raw_action[index]:+8.3f}  "
            f"{math.degrees(targets[index]):+8.2f}d  "
            f"{command_text}"
        )
    return lines


def format_imu(imu, angular_velocity, gravity, age):
    age_text = "--" if age is None else f"{age * 1000.0:.1f} ms"
    return [
        f"IMU [{imu.status}]  port {DEFAULT_IMU_PORT}  sample age {age_text}",
        f"  raw angular velocity x {angular_velocity[0]:+8.4f}  y {angular_velocity[1]:+8.4f}  "
        f"z {angular_velocity[2]:+8.4f}  [rad/s]",
        f"  projected gravity    x {gravity[0]:+8.4f}  y {gravity[1]:+8.4f}  z {gravity[2]:+8.4f}",
    ]


def format_pipeline(runner):
    pipeline = runner.pipeline
    calls = max(1, pipeline.policy_call_count * pipeline.action_size)
    return (
        f"clip: runner |a|>{pipeline.runner_clip:.1f} {pipeline.runner_clip_count / calls * 100:5.2f}%   "
        f"target {pipeline.target_clip_count / calls * 100:5.2f}%   "
        f"inference {runner.inference_ms:5.2f} ms"
    )


def confirm(prompt):
    if not sys.stdin.isatty():
        raise RuntimeError("Confirmation requires an interactive terminal")
    answer = input(prompt)
    if answer.strip():
        raise RuntimeError("Cancelled because the input was not empty")


def run_read(policy_path, contract, args):
    notes = []
    runner = PolicyRunner(policy_path, contract)
    joints = ReadOnlyJointSource(SETTINGS, notes)
    imu = ImuSource(SETTINGS, notes)

    print(f"Policy      : {policy_path}")
    print(f"Task        : {contract.task}   {contract.policy_hz:.0f} Hz")
    print(f"Joint source: {joints.label}")
    print("Read-only: no enable, no run-mode write, and no type 0x01 control frame is sent.")
    confirm("Press Enter to start reading CAN and IMU, or Ctrl-C to cancel: ")

    joints.start()
    imu.start(calibrate=True)
    period = 1.0 / SETTINGS.read_print_hz
    deadline = None if args.duration is None else time.monotonic() + args.duration
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            snapshot = joints.snapshot()
            positions, velocities, missing = runner.joint_state(snapshot)
            angular_velocity, gravity, age = imu.read(now)
            observation = runner.observation(positions, velocities, angular_velocity, gravity)
            raw_action, _, targets = runner.step(observation, commit=True)

            lines = [CLEAR_SCREEN]
            lines.append(f"Policy preview (read-only)   {joints.rate_text()}   (Ctrl-C to stop)\n")
            lines.extend(format_joint_table(contract, positions, velocities, raw_action, targets, None))
            if missing:
                lines.append(f"  no response: {sorted(missing)} (reported as 0.0)")
            lines.append("")
            lines.extend(format_imu(imu, angular_velocity, gravity, age))
            lines.append(f"\nObservation vector ({observation.size})")
            lines.append("  " + "  ".join(f"{value:+.3f}" for value in observation))
            lines.append("")
            lines.append(format_pipeline(runner))
            lines.append("No CAN control command is sent in read mode.")
            if notes:
                lines.append("")
                lines.extend(notes)
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            sleep = period - (time.monotonic() - now)
            if sleep > 0.0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        joints.stop()
        imu.stop()
        print("Stopped. No motor was enabled or commanded.")
    return 0


def approach_pose(commander, joints, targets, motors, limits, settings, label):
    print(f"\nSpeed-limited move to the {label} pose ({settings.approach_max_speed:.2f} rad/s cap).")
    tolerance = settings.approach_tolerance_deg * DEG
    period = 1.0 / 100.0
    next_tick = time.monotonic()
    last_tick = next_tick
    last_print = 0.0
    settled_at = None
    while True:
        now = time.monotonic()
        joints.poll()
        reason = runtime_safety_reason(motors, commander.commands, limits, now, settings)
        if reason:
            raise RuntimeError(f"{label} move stopped for safety: {reason}")

        dt = clamp(now - last_tick, period * 0.25, period * 2.0)
        last_tick = now
        commander.send(targets, dt, settings.approach_max_speed, settings.approach_max_accel)

        if now - last_print >= 1.0:
            last_print = now
            worst = max(
                (
                    abs(wrap_to_pi(motors[motor_id].last_position or 0.0) - target)
                    for motor_id, target in targets.items()
                ),
                default=0.0,
            )
            print(f"  [{time.strftime('%H:%M:%S')}] worst error {math.degrees(worst):+6.2f} deg")

        if commander.at_rest(targets, tolerance):
            print(f"{label.capitalize()} pose reached (within {settings.approach_tolerance_deg:g} deg).")
            return
        commanded = all(abs(commander.velocities[motor_id]) <= 1.0e-12 for motor_id in targets)
        if commanded:
            if settled_at is None:
                settled_at = now
            elif now - settled_at > settings.approach_settle_timeout:
                outside = [
                    f"ID {motor_id} {math.degrees(wrap_to_pi(motors[motor_id].last_position) - target):+.2f} deg off"
                    for motor_id, target in sorted(targets.items())
                    if motors[motor_id].last_position is not None
                    and abs(wrap_to_pi(motors[motor_id].last_position) - target) > tolerance
                ]
                raise RuntimeError(
                    f"Failed to reach the {label} pose within "
                    f"{settings.approach_settle_timeout:.1f} s: " + ", ".join(outside)
                )
        else:
            settled_at = None

        next_tick += period
        sleep = next_tick - time.monotonic()
        if sleep > 0.0:
            time.sleep(sleep)
        elif time.monotonic() - next_tick > period:
            next_tick = time.monotonic()


def policy_loop(runner, commander, joints, imu, motors, limits, contract, args, notes, stats):
    period = 1.0 / contract.policy_hz
    stats.started = time.monotonic()
    next_tick = time.monotonic()
    last_tick = next_tick
    last_status = 0.0
    ramp_started = next_tick
    home = {motor_id: commander.commands[motor_id] for motor_id in motors}
    deadline = None if args.duration is None else next_tick + args.duration

    print(
        f"\nPolicy control is active at {contract.policy_hz:.0f} Hz "
        f"(ramp-in {SETTINGS.ramp_seconds:.1f} s). Ctrl-C stops and brakes."
    )
    while deadline is None or time.monotonic() < deadline:
        now = time.monotonic()
        joints.poll()
        reason = runtime_safety_reason(motors, commander.commands, limits, now, SETTINGS)
        if reason:
            raise RuntimeError("Safety stop: " + reason)

        positions, velocities, missing = runner.joint_state(joints.snapshot())
        if missing:
            raise RuntimeError(f"Safety stop: no feedback for motor IDs {sorted(missing)}")
        angular_velocity, gravity, age = imu.read(now)
        imu_reason = imu.failure_reason(age)
        if imu_reason:
            raise RuntimeError("Safety stop: " + imu_reason)

        try:
            observation = runner.observation(positions, velocities, angular_velocity, gravity)
            raw_action, action, targets = runner.step(observation, commit=True)
        except ValueError as error:
            raise RuntimeError(f"Safety stop: {error}") from error

        blend = clamp((now - ramp_started) / SETTINGS.ramp_seconds, 0.0, 1.0)
        commanded = {}
        for index, (_, motor_id) in enumerate(joint_row_order(contract)):
            target = float(targets[index])
            commanded[motor_id] = home[motor_id] + blend * (target - home[motor_id])

        dt = clamp(now - last_tick, period * 0.25, period * 2.0)
        last_tick = now
        commander.send(commanded, dt, SETTINGS.policy_max_speed, SETTINGS.policy_max_accel)
        stats.record(dt, runner.inference_ms)

        if now - last_status >= 1.0 / SETTINGS.status_hz:
            last_status = now
            lines = [CLEAR_SCREEN]
            lines.append(
                f"Policy control   {joints.rate_text()}   ramp {blend * 100:5.1f}%   "
                f"elapsed {stats.elapsed():6.2f} s   (Ctrl-C to stop)\n"
            )
            lines.extend(
                format_joint_table(
                    contract, positions, velocities, raw_action, targets, commander.commands
                )
            )
            lines.append("")
            lines.extend(format_imu(imu, angular_velocity, gravity, age))
            lines.append("")
            lines.append(format_pipeline(runner))
            lines.append(
                f"slew-limited {commander.slew_limited_count / max(1, commander.command_count) * 100:5.2f}%   "
                f"worst period {stats.period_max * 1000.0:6.2f} ms   "
                f"worst inference {stats.inference_ms_max:5.2f} ms"
            )
            if notes:
                lines.append("")
                lines.extend(notes)
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

        next_tick += period
        sleep = next_tick - time.monotonic()
        if sleep > 0.0:
            time.sleep(sleep)
        elif time.monotonic() - next_tick > period:
            next_tick = time.monotonic()


def run_deploy(policy_path, contract, args):
    notes = []
    verify_common_source(contract, notes)
    runner = PolicyRunner(policy_path, contract)
    imu = ImuSource(SETTINGS, notes)

    motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
    hard_limits = {motor_id: JOINT_LIMITS_RAD[motor_id] for motor_id in motor_ids}
    home_targets = {
        JOINT_BY_MODEL_NAME[name].motor_id: float(contract.action_offsets[index])
        for index, name in enumerate(contract.joint_order)
    }

    buses = {}
    motors = {}
    hubs = {}
    enabled_ids = []
    stop_ids = []
    stats = LoopStats()
    commander = None
    try:
        buses, motors, hubs = open_hardware(motor_ids, SETTINGS.interface, SETTINGS.host_id)
        _, blocking = inspect_zero_positions(motors, SETTINGS.approach_tolerance_deg * DEG, hard_limits)
        if blocking:
            raise RuntimeError(
                "Preflight safety check failed; motors will not be enabled:\n  " + "\n  ".join(blocking)
            )
        if not imu.start(calibrate=True):
            raise RuntimeError("IMU is not usable; motors will not be enabled:\n  " + "\n  ".join(notes))

        print("\nThe real motors will move under policy control.")
        print(f"  policy      : {policy_path}")
        print(f"  task        : {contract.task}")
        print(f"  rate        : {contract.policy_hz:.0f} Hz   kp {SETTINGS.kp:g}  kd {SETTINGS.kd:g}")
        print(f"  motor IDs   : {sorted(motor_ids)}")
        print(f"  duration    : {'until Ctrl-C' if args.duration is None else f'{args.duration:.1f} s'}")
        print("  The robot must hang on the stand or be held; this tool cannot catch a fall.")
        print("  Keep the emergency stop within reach.")
        confirm("Press Enter to enable the motors and start, or Ctrl-C to cancel: ")

        stop_ids = list(motor_ids)
        try:
            starts, enabled_ids = enable_with_runtime_feedback(
                motors, hubs, SETTINGS.kp, SETTINGS.kd, hard_limits
            )
        except RuntimeError as error:
            enabled_ids = getattr(error, "enabled_ids", enabled_ids)
            raise

        joints = RuntimeJointSource(motors, hubs)
        commander = TargetCommander(motors, starts, SETTINGS)
        approach_pose(commander, joints, home_targets, motors, hard_limits, SETTINGS, "default")
        runner.reset()
        policy_loop(
            runner, commander, joints, imu, motors, hard_limits, contract, args, notes, stats
        )
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        brake_and_stop(motors, buses, enabled_ids, stop_ids, SETTINGS.brake_time, SETTINGS.kd)
        imu.stop()
        if stop_ids:
            print("Active damping and stop/disable shutdown completed.")
        else:
            print("CAN buses closed. No motor control command was sent.")
        if commander is not None and stats.steps:
            pipeline = runner.pipeline
            calls = max(1, pipeline.policy_call_count * pipeline.action_size)
            print(
                f"Ran {stats.steps} policy steps in {stats.elapsed():.2f} s; "
                f"worst period {stats.period_max * 1000.0:.2f} ms, "
                f"worst inference {stats.inference_ms_max:.2f} ms, "
                f"slew-limited {commander.slew_limited_count / max(1, commander.command_count) * 100:.2f}%, "
                f"runner clip {pipeline.runner_clip_count / calls * 100:.2f}%, "
                f"target clip {pipeline.target_clip_count / calls * 100:.2f}%."
            )
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a trained RoboNex policy on the real motors, or preview its values read-only."
    )
    parser.add_argument("--policy", type=Path, required=True, help="ONNX policy path")
    parser.add_argument(
        "--read",
        action="store_true",
        help="Read-only preview: print observation, action and targets without commanding any motor",
    )
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument(
        "--log",
        nargs="?",
        const="",
        metavar="PATH",
        help="Save terminal output; omit PATH for an automatic file under results/policy_to_real",
    )
    args = parser.parse_args(argv)
    args.policy = args.policy.expanduser().resolve()
    if not args.policy.is_file():
        parser.error(f"Policy not found: {args.policy}")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0.0):
        parser.error("--duration must be finite and positive")
    if args.log == "":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.log = THIS_FILE.parents[2] / "results" / "policy_to_real" / f"{timestamp}_{os.getpid()}.log"
    elif args.log is not None:
        args.log = Path(args.log).expanduser().resolve()
    return args


def run(args):
    try:
        contract = resolve_contract(args.policy)
    except (FileNotFoundError, ValueError) as error:
        print(f"Policy error: {error}")
        return 1
    try:
        if args.read:
            return run_read(args.policy, contract, args)
        return run_deploy(args.policy, contract, args)
    except KeyboardInterrupt:
        print("\nStop requested.")
        return 0
    except (RuntimeError, ValueError, OSError, can.CanError, subprocess.CalledProcessError) as error:
        print(f"\nStopped: {error}")
        return 1


def main(argv=None):
    args = parse_args(argv)
    if args.log is None:
        return run(args)
    try:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("x", encoding="utf-8", buffering=1) as log:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            lock = threading.Lock()
            sys.stdout = TeeStream(original_stdout, log, lock)
            sys.stderr = TeeStream(original_stderr, log, lock)
            try:
                print(f"Log file    : {args.log}")
                return run(args)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    except OSError as error:
        print(f"Log error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
