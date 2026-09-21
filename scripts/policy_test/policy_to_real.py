#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import json
from datetime import datetime, timezone
import math
import os
import select
import shutil
import sys
import tempfile
import termios
import threading
import time
import tty
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path

PROGRAM_STARTED = time.monotonic()

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
import robonex_common
from robonex_common.imu import DEFAULT_IMU_BAUDRATE, DEFAULT_IMU_PORT, MOUNT_ROLL_DEG
from robonex_common.joints import CHANNEL_MOTOR_IDS, JOINT_BY_MODEL_NAME, JOINT_BY_ID
from robonex_common.actuators import CONTROL_GAINS_BY_JOINT
from robonex_common.motors import MOTOR_CONTROL_KD, MOTOR_CONTROL_KP, RATED_TORQUE
from robonex_common.policy import PolicyContract, python_source_sha256
from robonex_common.protocol import MECHANICAL_VELOCITY_INDEX
from robonex_common.runtime import (
    OBSERVATION_TERM_SIZES,
    ActionPipeline,
    ObservationHistory,
    assemble_observation,
    gait_phase_at,
)
from safety import (
    AxisLimiter,
    ThermalLoad,
    align_angle,
    brake_and_stop,
    enable_with_runtime_feedback,
    inspect_zero_positions,
    open_hardware,
    runtime_safety_reason,
    shutdown_report_lines,
    tilt_reason,
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


def resolve_gains(scale):
    """Per-motor (kp, kd) from the shared per-joint table, scaled.

    The policy is trained against CONTROL_GAINS_BY_JOINT (hip 100/2, knee 150/4,
    ankle 40/2); commanding one scalar 40/2 drives the knee at 27% of the stiffness
    the policy assumes, so the joint does not reach the target the policy chose.
    """
    kp_by_motor = {}
    kd_by_motor = {}
    for name, spec in JOINT_BY_MODEL_NAME.items():
        kp, kd = CONTROL_GAINS_BY_JOINT.get(name, (MOTOR_CONTROL_KP, MOTOR_CONTROL_KD))
        kp_by_motor[spec.motor_id] = kp * scale
        kd_by_motor[spec.motor_id] = kd * scale
    return kp_by_motor, kd_by_motor


@dataclass(frozen=True)
class Settings:
    interface: str = DEFAULT_INTERFACE
    host_id: int = HOST_ID
    kp: float = MOTOR_CONTROL_KP
    kd: float = MOTOR_CONTROL_KD
    # Fraction of the per-joint gains in CONTROL_GAINS_BY_JOINT actually commanded.
    # The trained policy assumes those gains; the measured-hardware scalars above are
    # only the fallback for a joint the table does not cover. Ramp this up from a low
    # value on the stand before running at 1.0.
    gain_scale: float = 1.0

    read_poll_timeout: float = 0.02
    read_print_hz: float = 10.0

    status_hz: float = 10.0
    feedback_timeout: float = 0.10
    imu_stale_timeout: float = 0.10
    overspeed: float = 10.0
    max_temp: float = 70.0
    max_error_deg: float = 25.0
    max_tilt_deg: float = 40.0
    max_raw_action: float = 20.0

    approach_max_speed: float = 0.30
    approach_max_accel: float = 0.60
    approach_tolerance_deg: float = 1.0
    approach_settle_timeout: float = 5.0

    policy_max_speed: float = 6.0
    policy_max_accel: float = 120.0
    ramp_seconds: float = 1.0

    imu_calibration_seconds: float = 2.0
    brake_time: float = 0.20


SETTINGS = Settings()

GAIT_COMMAND_DEADBAND = 0.05
ENABLE_KP, ENABLE_KD = resolve_gains(SETTINGS.gain_scale)


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
        self.policy_path = policy_path
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
        self.velocity_command = np.zeros(3, dtype=np.float32)
        self.history = ObservationHistory.from_contract(contract)
        self.gait_step = 0
        self.mask_gait_on_standing = True
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

    def gait_phase_observation(self):
        phase = gait_phase_at(self.gait_step, step_dt=1.0 / self.contract.policy_hz)
        if not self.mask_gait_on_standing:
            return phase
        if float(np.linalg.norm(self.velocity_command)) <= GAIT_COMMAND_DEADBAND:
            return np.zeros_like(phase)
        return phase

    def observation(self, positions, velocities, angular_velocity, gravity):
        return assemble_observation(
            positions - self.offsets,
            velocities,
            angular_velocity,
            gravity,
            self.velocity_command,
            self.gait_phase_observation(),
            self.last_action,
            history=self.history,
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

    def commit_policy_action(self, action, commanded_targets):
        """Feed back the policy's own runner-clipped action, as training defines it.

        Training observes `action_manager.action`, the runner-clipped value *before* the
        target fence, and both the MuJoCo harness and this script's read mode already feed
        that back. Reconstructing it from the commanded motor position instead made the live
        path the only one of the three that disagreed: ramp blending, the slew limiter and
        the target fence all change the commanded position, and the fence is many-to-one so
        the original request cannot be recovered from it. The commanded vector is still
        validated here because a non-finite command must stop the run.
        """
        commanded = np.asarray(commanded_targets, dtype=np.float32).reshape(-1)
        if commanded.shape != (self.contract.action_size,):
            raise ValueError(
                f"commanded targets must have {self.contract.action_size} values, got {commanded.shape[0]}"
            )
        if not np.isfinite(commanded).all():
            raise ValueError("commanded targets are not finite")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.contract.action_size,):
            raise ValueError(
                f"policy action must have {self.contract.action_size} values, got {action.shape[0]}"
            )
        if not np.isfinite(action).all():
            raise ValueError("policy action is not finite")
        self.last_action[:] = action
        self.gait_step += 1


class TargetCommander:
    def __init__(self, motors, start_positions, settings):
        self.motors = motors
        self.settings = settings
        self.limiters = {
            motor_id: AxisLimiter(start_positions[motor_id]) for motor_id in motors
        }
        self.commands = dict(start_positions)
        self.velocities = {motor_id: 0.0 for motor_id in motors}
        self.kp_by_motor, self.kd_by_motor = resolve_gains(settings.gain_scale)
        self.slew_limited_count = 0
        self.command_count = 0
        self.slew_limited_by_motor = {motor_id: 0 for motor_id in motors}
        self.command_count_by_motor = {motor_id: 0 for motor_id in motors}
        self.slew_lag_step_max = 0.0
        self.slew_lag_run_max = 0.0
        self.slew_lag_max_by_motor = {motor_id: 0.0 for motor_id in motors}

    def reset_stats(self):
        self.slew_limited_count = 0
        self.command_count = 0
        self.slew_limited_by_motor = {motor_id: 0 for motor_id in self.motors}
        self.command_count_by_motor = {motor_id: 0 for motor_id in self.motors}
        self.slew_lag_step_max = 0.0
        self.slew_lag_run_max = 0.0
        self.slew_lag_max_by_motor = {motor_id: 0.0 for motor_id in self.motors}

    def send(self, motor_ids_targets, dt, max_speed, max_accel, use_velocity_target):
        step_lag_max = 0.0
        for motor_id, target in motor_ids_targets.items():
            limiter = self.limiters[motor_id]
            aligned = align_angle(limiter.position, target)
            position, velocity = limiter.step(aligned, dt, max_speed, max_accel)
            lag = abs(position - aligned)
            if lag > 1.0e-9:
                self.slew_limited_count += 1
                self.slew_limited_by_motor[motor_id] += 1
            if lag > step_lag_max:
                step_lag_max = lag
            if lag > self.slew_lag_max_by_motor[motor_id]:
                self.slew_lag_max_by_motor[motor_id] = lag
            self.command_count += 1
            self.command_count_by_motor[motor_id] += 1
            self.commands[motor_id] = position
            self.velocities[motor_id] = velocity
        self.slew_lag_step_max = step_lag_max
        if step_lag_max > self.slew_lag_run_max:
            self.slew_lag_run_max = step_lag_max
        for motor_id, motor in self.motors.items():
            motor.control(
                pos=self.commands[motor_id],
                vel=self.velocities[motor_id] if use_velocity_target else 0.0,
                kp=self.kp_by_motor[motor_id],
                kd=self.kd_by_motor[motor_id],
                torque=0.0,
            )

    def slew_rate_by_motor(self):
        return {
            motor_id: self.slew_limited_by_motor[motor_id]
            / max(1, self.command_count_by_motor[motor_id])
            for motor_id in self.motors
        }

    def worst_slew_lag_motor(self):
        if not self.slew_lag_max_by_motor:
            return None
        return max(self.slew_lag_max_by_motor, key=self.slew_lag_max_by_motor.get)

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


# Type-0x02 feedback fault flags (Motor.last_fault bit n = 29-bit ID bit 16+n),
# from the RobStride RS02/RS03 manuals, "Communication Type 2: motor feedback data".
FAULT_FLAG_NAMES = (
    "undervoltage",
    "overcurrent",
    "overtemperature",
    "magnetic encoder fault",
    "stall overload",
    "uncalibrated",
)


def fault_flag_names(flags):
    return [name for bit, name in enumerate(FAULT_FLAG_NAMES) if flags & (1 << bit)]


class FaultMonitor:
    """Record reported motor faults, and stop once one of them persists.

    All six bits mean the motor firmware has decided something is wrong, and every one of
    them makes the next position command either useless or harmful -- a magnetic encoder
    fault means the position being fed back is not trustworthy, and `uncalibrated` appearing
    mid-run means the motor rebooted. So none of them is display-only. What they are not is
    instantaneous: a single frame could be a glitch on a bus that has dropped frames before,
    and dropping a 20 kg robot has its own cost. A bit therefore has to hold for
    BLOCKING_FRAMES consecutive feedback frames before it stops anything.
    """

    BLOCKING_FRAMES = 3

    def __init__(self):
        self.previous = {}
        self.history = {}  # (motor_id, bit) -> [first seconds since start, phase, onset count]
        self.streak = {}   # (motor_id, bit) -> consecutive frames the bit has been set

    def update(self, motors, now, phase):
        for motor_id, motor in motors.items():
            flags = motor.last_fault
            rising = flags & ~self.previous.get(motor_id, 0)
            self.previous[motor_id] = flags
            for bit in range(len(FAULT_FLAG_NAMES)):
                if flags & (1 << bit):
                    self.streak[(motor_id, bit)] = self.streak.get((motor_id, bit), 0) + 1
                else:
                    self.streak.pop((motor_id, bit), None)
                if rising & (1 << bit):
                    record = self.history.setdefault(
                        (motor_id, bit), [now - PROGRAM_STARTED, phase, 0]
                    )
                    record[2] += 1

    def blocking_reason(self):
        for (motor_id, bit), frames in sorted(self.streak.items()):
            if frames >= self.BLOCKING_FRAMES:
                return (f"ID {motor_id} reports {FAULT_FLAG_NAMES[bit]} "
                        f"on {frames} consecutive feedback frames")
        return None

    def current_line(self, motors):
        active = [
            f"ID {motor_id} {', '.join(fault_flag_names(motors[motor_id].last_fault))} "
            f"(0x{motors[motor_id].last_fault:02X})"
            for motor_id in sorted(motors)
            if motors[motor_id].last_fault
        ]
        return "motor faults : " + ("   ".join(active) if active else "none")

    def history_line(self):
        by_motor = {}
        for (motor_id, bit), (seconds, phase, count) in sorted(self.history.items()):
            by_motor.setdefault(motor_id, []).append(
                f"{FAULT_FLAG_NAMES[bit]} @ {seconds:.2f} s ({phase}) x{count}"
            )
        entries = [f"ID {motor_id} " + ", ".join(items) for motor_id, items in by_motor.items()]
        return "fault history: " + ("   ".join(entries) if entries else "none")


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
    # PolicyRunner always assembles OBSERVATION_TERM_SIZES in order; the size check alone
    # accepts a manifest whose terms are reordered, so compare names and sizes before any CAN I/O.
    layout = ObservationHistory.from_contract(contract)
    if layout.term_sizes != OBSERVATION_TERM_SIZES:
        raise ValueError(
            f"observation_terms {list(contract.observation_terms)} do not match the deploy "
            f"layout {[f'{name}:{size}' for name, size in OBSERVATION_TERM_SIZES]}"
        )
    # gait_phase_at() advances the clock by 1/50 s per policy step.
    if contract.policy_hz != 50.0:
        raise ValueError(f"policy_hz={contract.policy_hz:g} but the gait clock assumes 50 Hz")
    return contract


def installed_common_commit():
    try:
        text = metadata.distribution("robonex-common").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id") if text else None


def verify_common_source(contract):
    # Verify the robonex_common that is actually imported (the .venv install), not a sibling checkout.
    package = Path(robonex_common.__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        # The manifest hashed the files as src/robonex_common/*.py; pip installs them without src/.
        shutil.copytree(package, Path(tmp, "src", "robonex_common"), ignore=shutil.ignore_patterns("__pycache__"))
        actual_sha256 = python_source_sha256(tmp, ("src/robonex_common",))
    if actual_sha256 != contract.common_sha256:
        raise RuntimeError(
            f"installed robonex-common differs from the policy's training source: "
            f"manifest={contract.common_sha256}, installed={actual_sha256} ({package})"
        )
    print(
        f"Common      : source matches the manifest  "
        f"(installed commit {installed_common_commit() or 'unknown'}, manifest commit {contract.common_commit})"
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
    runner.velocity_command[:] = (args.vx, args.vy, args.wz)
    runner.mask_gait_on_standing = not args.no_gait_mask
    joints = ReadOnlyJointSource(SETTINGS, notes)
    imu = ImuSource(SETTINGS, notes)

    print(f"Policy      : {policy_path}")
    print(f"Task        : {contract.task}   {contract.policy_hz:.0f} Hz")
    print(f"Joint source: {joints.label}")
    print("Read-only: no enable, no run-mode write, and no type 0x01 control frame is sent.")
    confirm("Press Enter to start reading CAN and IMU, or Ctrl-C to cancel: ")

    joints.start()
    imu.start(calibrate=True)
    read_log = None
    if args.telemetry:
        read_log = ReadRecorder(args.telemetry, contract)
        print(f"Recording   : {read_log.path}")
    period = 1.0 / SETTINGS.read_print_hz
    deadline = None if args.duration is None else time.monotonic() + args.duration
    # The preview prints at read_print_hz, not the policy rate, and nothing here commits a
    # command, so the gait clock has no step counter to ride on. Derive it from elapsed time
    # instead: otherwise the preview shows a frozen phase and the policy it displays is not
    # the one the robot would run.
    started_at = time.monotonic()
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            runner.gait_step = int((now - started_at) * contract.policy_hz)
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
            if read_log is not None:
                read_log.record(now, observation, gravity, angular_velocity, age, positions, missing)

            sleep = period - (time.monotonic() - now)
            if sleep > 0.0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        joints.stop()
        imu.stop()
        if read_log is not None:
            read_log.close()
            print(f"Recorded to {read_log.path}")
        print("Stopped. No motor was enabled or commanded.")
    return 0


def approach_pose(commander, joints, targets, motors, limits, settings, label, faults):
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
        faults.update(motors, now, "approach")
        fault_reason = faults.blocking_reason()
        if fault_reason:
            raise RuntimeError("Safety stop: " + fault_reason)
        reason = runtime_safety_reason(motors, commander.commands, limits, now, settings)
        if reason:
            raise RuntimeError(f"{label} move stopped for safety: {reason}")

        dt = clamp(now - last_tick, period * 0.25, period * 2.0)
        last_tick = now
        commander.send(
            targets,
            dt,
            settings.approach_max_speed,
            settings.approach_max_accel,
            use_velocity_target=True,
        )

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
            if faults.history:
                print("    " + faults.current_line(motors))
                print("    " + faults.history_line())

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


class ReadRecorder:
    """Per-frame record of the read-only preview.

    Read mode has no torque, temperature or type 0x02 feedback age — it polls `mechPos` — so
    this is deliberately not the live `TelemetryRecorder`. It captures what the read check
    actually needs to be verified off the screen: the gait-phase pair across the whole run,
    the IMU vectors, and which motors went silent.
    """

    def __init__(self, path, contract):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise SystemExit(
                f"{self.path} already exists. Recordings of a real robot are not reproducible, "
                "so this refuses to overwrite one. Pass --telemetry with no path for an "
                "automatic timestamped file, or name a new one."
            )
        self.handle = self.path.open("w", newline="")
        self.writer = csv.writer(self.handle)
        header = ["t_s", "gravity_x", "gravity_y", "gravity_z",
                  "gyro_x", "gyro_y", "gyro_z", "imu_age_ms", "missing_ids"]
        header += [f"gait_{i}" for i in range(10)]
        header += [f"{n.replace('_joint','')}.pos" for n in contract.joint_order]
        self.writer.writerow(header)
        self.handle.flush()
        self.started = None

    def record(self, now, observation, gravity, gyro, age, positions, missing):
        if self.started is None:
            self.started = now
        row = [round(now - self.started, 3)]
        row += [round(float(v), 5) for v in gravity]
        row += [round(float(v), 5) for v in gyro]
        row += ["" if age is None else round(age * 1000.0, 2)]
        row += ["|".join(str(m) for m in sorted(missing)) if missing else ""]
        row += [round(float(v), 5) for v in observation[165:175]]
        row += [round(float(v), 5) for v in positions]
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        try:
            self.handle.close()
        except OSError:
            pass


class TelemetryRecorder:
    """Per-policy-step record of everything needed to diagnose a hardware stop.

    The loop has ~20 ms of budget and the previous hardware run already measured a
    20.43 ms worst period, so nothing here touches the filesystem inside the step: rows
    accumulate in memory and are written in batches between steps, and whatever is left
    is flushed by the caller's finally.
    """

    SLACK_S = 0.005
    MAX_ROWS = 500

    def __init__(self, path, contract, motors):
        self.path = Path(path)
        self.rows = []
        self.order = joint_row_order(contract)
        self.motors = motors
        self.dropped = 0
        header = ["t_s", "step", "dt_ms", "inference_ms", "ramp",
                  "cmd_vx", "cmd_vy", "cmd_wz",
                  "gravity_x", "gravity_y", "gravity_z",
                  "gyro_x", "gyro_y", "gyro_z", "imu_age_ms"]
        for name, motor_id in self.order:
            short = name.replace("_joint", "")
            header += [
                f"{short}.age_ms", f"{short}.pos", f"{short}.vel", f"{short}.torque",
                f"{short}.temp", f"{short}.fault", f"{short}.raw_action",
                f"{short}.target", f"{short}.commanded",
            ]
        header.append("stop_reason")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise SystemExit(
                f"{self.path} already exists. Recordings of a real robot are not reproducible, "
                "so this refuses to overwrite one. Pass --telemetry with no path for an "
                "automatic timestamped file, or name a new one."
            )
        self.handle = self.path.open("w", newline="")
        try:
            self.writer = csv.writer(self.handle)
            self.writer.writerow(header)
            self.handle.flush()
        except BaseException:
            self.handle.close()
            raise
        self.step = 0

    def record(self, now, started, dt, inference_ms, ramp, raw_action, targets, commands,
               gravity=None, gyro=None, imu_age=None, velocity_command=None, stop_reason=""):
        row = [
            round(now - started, 4), self.step, round(dt * 1000.0, 3),
            round(inference_ms, 3), round(ramp, 4),
        ]
        row += [_round(v) for v in (velocity_command if velocity_command is not None else (None,) * 3)]
        row += [_round(v) for v in (gravity if gravity is not None else (None,) * 3)]
        row += [_round(v) for v in (gyro if gyro is not None else (None,) * 3)]
        row.append(round(imu_age * 1000.0, 3) if imu_age is not None else "")
        # Age is measured against a fresh clock read, not the `now` the caller captured at the
        # top of its loop: that timestamp predates `joints.poll()`, so the feedback it is being
        # compared against is stamped later and every age came out negative.
        sampled_at = time.monotonic()
        for index, (_, motor_id) in enumerate(self.order):
            motor = self.motors.get(motor_id)
            stamp = getattr(motor, "last_feedback_time", None)
            age = (sampled_at - stamp) * 1000.0 if stamp else float("nan")
            row += [
                round(age, 3),
                _round(getattr(motor, "last_position", None)),
                _round(getattr(motor, "last_velocity", None)),
                _round(getattr(motor, "last_torque", None)),
                _round(getattr(motor, "last_temp", None)),
                getattr(motor, "last_fault", ""),
                round(float(raw_action[index]), 5),
                round(float(targets[index]), 5),
                _round(commands.get(motor_id)),
            ]
        row.append(stop_reason)
        self.rows.append(row)
        self.step += 1
        if len(self.rows) >= self.MAX_ROWS:
            self.flush()

    def write_sidecar(self, policy_path, contract, settings, args):
        """Record which policy produced this file, next to it.

        The CSV carries no policy identity, so a recording cannot be traced back to the
        weights that made it -- and a run analysed under the wrong assumption is worse than
        one that is skipped. This is the same idea as the exporter's `export_receipt.json`.
        """
        meta = {
            "telemetry_file": self.path.name,
            "written_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "policy_file": str(policy_path),
            "policy_sha256": contract.policy_sha256,
            "task": contract.task,
            "policy_hz": contract.policy_hz,
            "training_commit": contract.training_commit,
            "common_commit": contract.common_commit,
            "description_commit": contract.description_commit,
            "deploy_source_sha256": python_source_sha256(
                Path(__file__).resolve().parents[2], ("scripts",)
            ),
            "command": {"vx": args.vx, "vy": args.vy, "wz": args.wz,
                        "keyboard": bool(getattr(args, "keyboard", False))},
            "settings": {
                "gain_scale": settings.gain_scale,
                "max_error_deg": settings.max_error_deg,
                "max_tilt_deg": settings.max_tilt_deg,
                "max_temp": settings.max_temp,
                "overspeed": settings.overspeed,
                "approach_tolerance_deg": settings.approach_tolerance_deg,
                "policy_max_speed": settings.policy_max_speed,
                "policy_max_accel": settings.policy_max_accel,
            },
        }
        path = self.path.with_name(self.path.stem + "_meta.json")
        path.write_text(json.dumps(meta, indent=2) + "\n")
        return path

    def record_stop(self, now, started, reason, commands, gravity=None, gyro=None, imu_age=None,
                    velocity_command=None):
        """Write the sample that tripped a safety stop.

        The checks run at the top of the loop and raise, while `record` is called near the
        bottom, so the sample that actually breached a limit was never written -- the one
        row an operator most wants to see. This captures the motor state as the check saw
        it, against the commands it compared them to. The policy columns are blank because
        this step never reached inference.
        """
        self.record(
            now, started, float("nan"), float("nan"), float("nan"),
            [float("nan")] * len(self.order), [float("nan")] * len(self.order), commands,
            gravity=gravity, gyro=gyro, imu_age=imu_age, velocity_command=velocity_command,
            stop_reason=reason,
        )
        self.flush()

    def flush_if_idle(self, slack_s):
        """Write only while the loop is waiting for its next tick.

        A batched write of 100 rows measured 1.7 ms, and the previous hardware run's worst
        period was already 20.43 ms against a 20 ms budget, so the write must not land
        inside the step.
        """
        if self.rows and slack_s > self.SLACK_S:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        try:
            self.writer.writerows(self.rows)
            self.handle.flush()
        except OSError:
            self.dropped += len(self.rows)
        finally:
            self.rows = []

    def close(self):
        try:
            self.flush()
        finally:
            try:
                self.handle.close()
            except OSError:
                pass


def _round(value, digits=5):
    if value is None:
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


def policy_loop(runner, commander, joints, imu, motors, limits, contract, args, notes, stats,
                faults, thermal, keyboard=None):
    period = 1.0 / contract.policy_hz
    stats.started = time.monotonic()
    next_tick = time.monotonic()
    last_tick = next_tick
    last_status = 0.0
    ramp_started = next_tick
    home = {motor_id: commander.commands[motor_id] for motor_id in motors}
    deadline = None if args.duration is None else next_tick + args.duration
    commander.reset_stats()
    telemetry = (
        TelemetryRecorder(args.telemetry, contract, motors)
        if getattr(args, "telemetry", None)
        else None
    )
    if telemetry is not None:
        print(f"Telemetry   : {telemetry.path}  ({contract.policy_hz:.0f} Hz per-motor record)")
        print(f"Provenance  : {telemetry.write_sidecar(runner.policy_path, contract, SETTINGS, args).name}")

    print(
        f"\nPolicy control is active at {contract.policy_hz:.0f} Hz "
        f"(ramp-in {SETTINGS.ramp_seconds:.1f} s). Ctrl-C stops and brakes."
    )
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            joints.poll()
            faults.update(motors, now, "policy")
            thermal.update(motors, now - last_tick)
            def stop(reason, gravity=None, gyro=None, imu_age=None):
                if telemetry is not None:
                    telemetry.record_stop(
                        now, stats.started, reason, commander.commands,
                        gravity=gravity, gyro=gyro, imu_age=imu_age,
                        velocity_command=tuple(float(v) for v in runner.velocity_command),
                    )
                return RuntimeError("Safety stop: " + reason)

            fault_reason = faults.blocking_reason()
            if fault_reason:
                raise stop(fault_reason)
            reason = runtime_safety_reason(motors, commander.commands, limits, now, SETTINGS)
            if reason:
                raise stop(reason)

            positions, velocities, missing = runner.joint_state(joints.snapshot())
            if missing:
                raise stop(f"no feedback for motor IDs {sorted(missing)}")
            angular_velocity, gravity, age = imu.read(now)
            imu_reason = imu.failure_reason(age)
            if imu_reason:
                raise stop(imu_reason)
            tilt = tilt_reason(gravity, SETTINGS.max_tilt_deg)
            if tilt:
                raise stop(tilt, gravity=gravity, gyro=angular_velocity, imu_age=age)

            try:
                observation = runner.observation(positions, velocities, angular_velocity, gravity)
                raw_action, policy_action, targets = runner.step(observation, commit=False)
            except ValueError as error:
                raise RuntimeError(f"Safety stop: {error}") from error

            blend = clamp((now - ramp_started) / SETTINGS.ramp_seconds, 0.0, 1.0)
            commanded = {}
            for index, (_, motor_id) in enumerate(joint_row_order(contract)):
                target = float(targets[index])
                commanded[motor_id] = home[motor_id] + blend * (target - home[motor_id])

            dt = clamp(now - last_tick, period * 0.25, period * 2.0)
            last_tick = now
            commander.send(
                commanded,
                dt,
                SETTINGS.policy_max_speed,
                SETTINGS.policy_max_accel,
                use_velocity_target=False,
            )
            commanded_targets = [
                commander.commands[motor_id] for _, motor_id in joint_row_order(contract)
            ]
            runner.commit_policy_action(policy_action, commanded_targets)
            stats.record(dt, runner.inference_ms)
            if telemetry is not None:
                telemetry.record(
                    now, stats.started, dt, runner.inference_ms, blend,
                    raw_action, targets, commander.commands,
                    gravity=gravity, gyro=angular_velocity, imu_age=age,
                    velocity_command=tuple(float(v) for v in runner.velocity_command),
                )

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
                worst_lag_id = commander.worst_slew_lag_motor()
                lines.append(
                    f"slew lag now {math.degrees(commander.slew_lag_step_max):6.2f} deg   "
                    f"run max {math.degrees(commander.slew_lag_run_max):6.2f} deg"
                    + (f" (ID {worst_lag_id})" if worst_lag_id is not None else "")
                    + f"   worst period {stats.period_max * 1000.0:6.2f} ms   "
                    f"worst inference {stats.inference_ms_max:5.2f} ms"
                )
                lags = commander.slew_lag_max_by_motor
                lines.append(
                    "slew lag max by motor "
                    + "  ".join(f"ID {motor_id} {math.degrees(lags[motor_id]):5.2f}" for motor_id in sorted(lags))
                    + " deg"
                )
                lines.append("")
                if keyboard is not None:
                    lines.append(
                        f"command  vx {runner.velocity_command[0]:+.2f}  "
                        f"vy {runner.velocity_command[1]:+.2f}  "
                        f"wz {runner.velocity_command[2]:+.2f}    {keyboard.legend()}"
                    )
                lines.append(thermal.worst_line())
                lines.append(faults.current_line(motors))
                lines.append(faults.history_line())
                if notes:
                    lines.append("")
                    lines.extend(notes)
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

            if keyboard is not None:
                note = keyboard.poll(runner.velocity_command)
                if note:
                    print(f"  [command] {note}   "
                          f"vx {runner.velocity_command[0]:+.2f} "
                          f"vy {runner.velocity_command[1]:+.2f} "
                          f"wz {runner.velocity_command[2]:+.2f}", flush=True)

            next_tick += period
            sleep = next_tick - time.monotonic()
            if telemetry is not None:
                telemetry.flush_if_idle(sleep)
                sleep = next_tick - time.monotonic()
            if sleep > 0.0:
                time.sleep(sleep)
            elif time.monotonic() - next_tick > period:
                next_tick = time.monotonic()

    finally:
        if telemetry is not None:
            telemetry.close()


def run_deploy(policy_path, contract, args):
    notes = []
    verify_common_source(contract)
    runner = PolicyRunner(policy_path, contract)
    runner.velocity_command[:] = (args.vx, args.vy, args.wz)
    runner.mask_gait_on_standing = not args.no_gait_mask
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
    faults = FaultMonitor()
    keyboard = KeyboardCommand(COMMAND_LIMITS) if args.keyboard else None
    thermal = ThermalLoad({
        motor_id: RATED_TORQUE[JOINT_BY_ID[motor_id].motor_model] for motor_id in motor_ids
    })
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
        print(f"  rate        : {contract.policy_hz:.0f} Hz")
        print(f"  gain scale  : {SETTINGS.gain_scale:g}  (1.0 = the gains the policy was trained at)")
        for name, spec in JOINT_BY_MODEL_NAME.items():
            print(
                f"      {name:22s} kp {ENABLE_KP[spec.motor_id]:6.1f}  kd {ENABLE_KD[spec.motor_id]:5.2f}"
            )
        print(f"  motor IDs   : {sorted(motor_ids)}")
        print(f"  duration    : {'until Ctrl-C' if args.duration is None else f'{args.duration:.1f} s'}")
        print(f"  tilt stop   : {SETTINGS.max_tilt_deg:g} deg from vertical")
        print("  The robot must hang on the stand or be held; this tool cannot catch a fall.")
        print("  Keep the emergency stop within reach.")
        confirm("Press Enter to enable the motors and start, or Ctrl-C to cancel: ")
        if keyboard is not None and keyboard.start():
            print(f"  {keyboard.legend()}")

        stop_ids = list(motor_ids)
        starts, enabled_ids = enable_with_runtime_feedback(
            motors, hubs, ENABLE_KP, ENABLE_KD, hard_limits, enabled_out=enabled_ids
        )

        joints = RuntimeJointSource(motors, hubs)
        commander = TargetCommander(motors, starts, SETTINGS)
        approach_pose(commander, joints, home_targets, motors, hard_limits, SETTINGS, "default", faults)
        runner.reset()
        policy_loop(
            runner, commander, joints, imu, motors, hard_limits, contract, args, notes, stats,
            faults, thermal, keyboard
        )
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        shutdown_report = brake_and_stop(
            motors, buses, enabled_ids, stop_ids, SETTINGS.brake_time, ENABLE_KD
        )
        imu.stop()
        if stop_ids:
            for line in shutdown_report_lines(shutdown_report):
                print(line)
            print(faults.history_line())
        else:
            print("CAN buses closed. No motor control command was sent.")
        if commander is not None and stats.steps:
            pipeline = runner.pipeline
            calls = max(1, pipeline.policy_call_count * pipeline.action_size)
            print(
                f"Ran {stats.steps} policy steps in {stats.elapsed():.2f} s; "
                f"worst period {stats.period_max * 1000.0:.2f} ms, "
                f"worst inference {stats.inference_ms_max:.2f} ms, "
                f"slew lag max {math.degrees(commander.slew_lag_run_max):.2f} deg, "
                f"{thermal.worst_line()}, "
                f"runner clip {pipeline.runner_clip_count / calls * 100:.2f}%, "
                f"target clip {pipeline.target_clip_count / calls * 100:.2f}%."
            )
        if keyboard is not None:
            keyboard.stop()
    return 0


class KeyboardCommand:
    """Drive the velocity command from the terminal while the policy runs.

    cbreak, not raw: `tty.setraw` clears ISIG, which would stop Ctrl-C from raising
    SIGINT and take away the operator's primary way to stop a moving robot.
    `tty.setcbreak` clears only ECHO and ICANON, so keys arrive one at a time and
    Ctrl-C still works.

    Losing the terminal means losing control, so stdin reaching EOF -- an SSH session
    dropping, most likely -- zeroes the command and disables further input rather than
    leaving the robot walking on the last thing it was told.
    """

    KEYS = {
        "w": (0, +0.05), "s": (0, -0.05),
        "q": (1, +0.05), "e": (1, -0.05),
        "a": (2, +0.05), "d": (2, -0.05),
    }

    def __init__(self, limits):
        self.limits = limits
        self.fd = None
        self.saved = None
        self.active = False
        self.lost = False

    def start(self):
        if not sys.stdin.isatty():
            print("Keyboard control needs an interactive terminal; the command stays fixed.")
            return False
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        atexit.register(self.stop)
        tty.setcbreak(self.fd)
        self.active = True
        termios.tcflush(self.fd, termios.TCIFLUSH)
        return True

    def stop(self):
        if self.active and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.active = False

    def poll(self, command):
        """Apply whatever has been typed. Returns a note to display, or None."""
        if not self.active or self.lost:
            return None
        note = None
        while select.select([sys.stdin], [], [], 0.0)[0]:
            data = os.read(self.fd, 64)
            if not data:
                self.lost = True
                command[:] = (0.0, 0.0, 0.0)
                return "stdin closed; command zeroed and keyboard disabled"
            for byte in data:
                key = chr(byte).lower()
                if key == " ":
                    command[:] = (0.0, 0.0, 0.0)
                    note = "command zeroed"
                elif key in self.KEYS:
                    axis, delta = self.KEYS[key]
                    low, high = self.limits[axis]
                    command[axis] = max(low, min(high, float(command[axis]) + delta))
        return note

    def legend(self):
        return ("keys: w/s forward  q/e strafe  a/d turn  SPACE zero  Ctrl-C stop"
                if self.active else "keyboard: off")


# The policy is only trained inside this envelope; a command outside it is out of
# distribution and the response is undefined, so refuse it rather than clamp silently.
# These are the ranges the velocity curriculum actually reached, not its starting
# ranges: it widens both ways from (0.1, 0.1) toward limit_ranges, and S30, S33 and S34
# all hit the full -0.2..0.5 on x by iteration ~160, spending 84% of training there.
COMMAND_LIMITS = ((-0.2, 0.5), (-0.2, 0.2), (-0.2, 0.2))


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
        "--keyboard",
        action="store_true",
        help="Steer the velocity command from the terminal while the policy runs: w/s "
             "forward, q/e strafe, a/d turn, SPACE to zero every axis. Values are clamped "
             "to the same trained envelope as --vx/--vy/--wz. Needs an interactive "
             "terminal; Ctrl-C keeps working.",
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=None,
        help="Stop if the trunk goes further than this from vertical. The recorded runs "
             "reach 6.9 deg walking and 26.1 deg standing under a push that the robot then "
             "recovered from, so the 40 deg default leaves recovery alone while still firing "
             "long before the robot is flat. Lower it to catch a fall sooner, at the risk of "
             "cutting a recovery short.",
    )
    parser.add_argument(
        "--approach-tolerance-deg",
        type=float,
        default=None,
        help="How close to the default pose counts as reached before policy control starts. "
             "The 1.0 deg default suits a suspended robot; standing on the ground the legs "
             "carry body weight and PD control leaves a steady-state error of roughly "
             "(holding torque / kp), which is 3-5 deg at the ankles. Raise it deliberately, "
             "and only when you know why the robot cannot close the gap",
    )
    parser.add_argument(
        "--telemetry",
        nargs="?",
        const="",
        metavar="PATH",
        help="Write a per-policy-step CSV: per-motor feedback age, position, velocity, torque, "
             "temperature and fault, plus raw action, clipped target and commanded position. "
             "Omit PATH for an automatic timestamped file under results/policy_to_real. "
             "An existing file is never overwritten",
    )
    parser.add_argument(
        "--no-gait-mask",
        action="store_true",
        help="Feed the raw gait clock even on a standing command, for policies trained before the mask",
    )
    parser.add_argument(
        "--vx", type=float, default=0.0, help="Forward velocity command (m/s)"
    )
    parser.add_argument(
        "--vy", type=float, default=0.0, help="Lateral velocity command (m/s)"
    )
    parser.add_argument(
        "--wz", type=float, default=0.0, help="Yaw rate command (rad/s)"
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=Settings.gain_scale,
        metavar="F",
        help=(
            "Fraction of the per-joint gains to command (default 1.0 = the gains the "
            "policy was trained at). Ramp up from a low value on the stand."
        ),
    )
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
    if not math.isfinite(args.gain_scale) or not 0.0 < args.gain_scale <= 1.0:
        parser.error("--gain-scale must be in (0, 1]")
    for name, value, limit in (
        ("--vx", args.vx, COMMAND_LIMITS[0]),
        ("--vy", args.vy, COMMAND_LIMITS[1]),
        ("--wz", args.wz, COMMAND_LIMITS[2]),
    ):
        if not math.isfinite(value):
            parser.error(f"{name} must be finite")
        if not limit[0] <= value <= limit[1]:
            parser.error(
                f"{name}={value} is outside the trained envelope {limit}; the policy has never "
                "seen that command and its response is undefined"
            )
    if args.telemetry == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        kind = "read" if args.read else "live"
        args.telemetry = (
            THIS_FILE.parents[2] / "results" / "policy_to_real" / f"{stamp}_{kind}_telemetry.csv"
        )
    elif args.telemetry is not None:
        args.telemetry = Path(args.telemetry).expanduser().resolve()
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
    except (RuntimeError, ValueError, OSError, can.CanError) as error:
        print(f"\nStopped: {error}")
        return 1


def main(argv=None):
    args = parse_args(argv)
    global SETTINGS, ENABLE_KP, ENABLE_KD
    SETTINGS = replace(SETTINGS, gain_scale=args.gain_scale)
    if args.max_tilt_deg is not None:
        if not 5.0 <= args.max_tilt_deg <= 90.0:
            raise SystemExit("--max-tilt-deg must be between 5 and 90 degrees")
        SETTINGS = replace(SETTINGS, max_tilt_deg=args.max_tilt_deg)
        print(
            f"Tilt stop set to {args.max_tilt_deg:g} deg "
            f"(default {Settings().max_tilt_deg:g})"
        )
    if args.approach_tolerance_deg is not None:
        if not 0.0 < args.approach_tolerance_deg <= 10.0:
            raise SystemExit("--approach-tolerance-deg must be between 0 and 10 degrees")
        SETTINGS = replace(SETTINGS, approach_tolerance_deg=args.approach_tolerance_deg)
        print(
            f"Approach tolerance widened to {args.approach_tolerance_deg:g} deg "
            f"(default {Settings().approach_tolerance_deg:g})"
        )
    ENABLE_KP, ENABLE_KD = resolve_gains(SETTINGS.gain_scale)
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
