#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import can
import mujoco
import mujoco.viewer
import numpy as np

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]

sys.path.insert(0, str(SCRIPTS_DIR))

from robonex_paths import description_model
from robonex_can import (
    JOINT_LIMITS_RAD,
    DEFAULT_INTERFACE,
    FeedbackHub,
    HOST_ID,
    JOINT_MAP,
    Motor,
    SPECS,
    build_arb,
    channel_for_id,
    clamp,
    parse_arb,
)

from robonex_can import MOTOR_ACTUATORS, MOTOR_MODELS

DEFAULT_MODEL_PATH = description_model("mujoco/scene_fixed.xml")

MAX_CONFIG_SPEED = 0.5
MAX_CONFIG_ACCEL = 2.0
MAX_CONFIG_RATE = 200.0
ZERO_SETTLE_TIMEOUT = 5.0


@dataclass
class AxisLimiter:

    position: float
    velocity: float = 0.0

    def step(self, target: float, dt: float, max_speed: float, max_accel: float):
        if not all(math.isfinite(v) for v in (target, dt, max_speed, max_accel)):
            raise ValueError("rate limiter received a non-finite value")
        if dt <= 0.0 or max_speed <= 0.0 or max_accel <= 0.0:
            raise ValueError("dt, max_speed and max_accel must be positive")

        error = target - self.position
        if abs(error) <= 1e-12 and abs(self.velocity) <= 1e-12:
            self.position = target
            self.velocity = 0.0
            return self.position, self.velocity

        braking_speed = max(
            0.0,
            math.sqrt((max_accel * dt) ** 2 + 2.0 * max_accel * abs(error))
            - max_accel * dt,
        )
        desired_velocity = math.copysign(min(max_speed, braking_speed), error)
        dv = clamp(desired_velocity - self.velocity, -max_accel * dt, max_accel * dt)
        next_velocity = clamp(self.velocity + dv, -max_speed, max_speed)
        next_position = self.position + next_velocity * dt

        if error * (target - next_position) <= 0.0:
            next_position = target
            next_velocity = 0.0

        self.position = next_position
        self.velocity = next_velocity
        return self.position, self.velocity


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def align_angle(current, target):
    return current + wrap_to_pi(target - current)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Track fixed-base MuJoCo targets with RoboNex motors. "
            "Simulation-only unless --hardware is set."
        )
    )
    parser.add_argument("--hardware", action="store_true",
                        help="Enable real CAN motor control")
    parser.add_argument("--motor-id", "--motor-ids", dest="motor_id",
                        nargs="+", type=lambda v: int(v, 0),
                        default=list(range(1, 13)),
                        help="Motor IDs to control. Default: 1 through 12")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"Fixed-base MJCF scene. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--interface", default=DEFAULT_INTERFACE,
                        help="python-can interface. Default: socketcan")
    parser.add_argument("--host-id", type=lambda v: int(v, 0), default=HOST_ID)
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Command rate in Hz. Default: 100")
    parser.add_argument("--max-speed", type=float, default=0.10,
                        help="Maximum target speed in rad/s. Default: 0.10")
    parser.add_argument("--max-accel", type=float, default=0.25,
                        help="Maximum target acceleration in rad/s^2. Default: 0.25")
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--zero-tolerance-deg", type=float, default=3.0,
                        help="Position tolerance for reaching zero in degrees. Default: 3")
    parser.add_argument("--limit-margin-deg", type=float, default=3.0,
                        help="Margin inside joint limits in degrees. Default: 3")
    parser.add_argument("--feedback-timeout", type=float, default=0.30,
                        help="Type-0x02 freshness timeout in seconds. Default: 0.30")
    parser.add_argument("--overspeed", type=float, default=2.0,
                        help="Measured-speed stop limit in rad/s. Default: 2.0")
    parser.add_argument("--max-error-deg", type=float, default=25.0,
                        help="Tracking-error stop limit in degrees. Default: 25")
    parser.add_argument("--max-temp", type=float, default=70.0,
                        help="Motor temperature stop limit in degrees C. Default: 70")
    parser.add_argument("--brake-time", type=float, default=0.20,
                        help="Active damping time before shutdown. Default: 0.20 s")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the hardware confirmation prompt")
    parser.add_argument("--headless", action="store_true",
                        help="Run without a viewer; requires --duration")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds")
    return parser.parse_args(argv)


def validate_args(args):
    problems = []
    motor_ids = sorted(set(args.motor_id))
    unknown = [mid for mid in motor_ids if mid not in MOTOR_MODELS]
    if unknown:
        problems.append(f"Unsupported motor ID: {unknown}")

    numeric_positive = (
        "rate", "max_speed", "max_accel", "feedback_timeout", "overspeed",
        "max_error_deg", "max_temp",
    )
    for name in numeric_positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            problems.append(f"--{name.replace('_', '-')} must be finite and positive ({value})")
    for name in ("kp", "kd", "limit_margin_deg", "brake_time"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            problems.append(f"--{name.replace('_', '-')} must be finite and non-negative ({value})")
    if not math.isfinite(args.zero_tolerance_deg) or args.zero_tolerance_deg <= 0.0:
        problems.append(
            f"--zero-tolerance-deg must be finite and positive ({args.zero_tolerance_deg})"
        )
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0.0):
        problems.append(f"--duration must be finite and positive ({args.duration})")
    if args.headless and args.duration is None:
        problems.append("--headless requires --duration")
    if args.rate > MAX_CONFIG_RATE:
        problems.append(f"--rate must be at most {MAX_CONFIG_RATE:g} Hz")
    if args.max_speed > MAX_CONFIG_SPEED:
        problems.append(f"--max-speed must be at most {MAX_CONFIG_SPEED:g} rad/s")
    if args.max_accel > MAX_CONFIG_ACCEL:
        problems.append(f"--max-accel must be at most {MAX_CONFIG_ACCEL:g} rad/s^2")
    if args.overspeed <= args.max_speed:
        problems.append("--overspeed must be greater than --max-speed")
    if motor_ids and not unknown:
        min_kp_max = min(SPECS[MOTOR_MODELS[mid]].kp_max for mid in motor_ids)
        min_kd_max = min(SPECS[MOTOR_MODELS[mid]].kd_max for mid in motor_ids)
        if args.kp > min_kp_max:
            problems.append(f"--kp {args.kp} exceeds the selected motor limit {min_kp_max}")
        if args.kd > min_kd_max:
            problems.append(f"--kd {args.kd} exceeds the selected motor limit {min_kd_max}")
    return motor_ids, problems


def safe_limits(motor_ids, margin_rad):
    result = {}
    for mid in motor_ids:
        lower, upper = JOINT_LIMITS_RAD[mid]
        inner = (lower + margin_rad, upper - margin_rad)
        if inner[0] >= inner[1]:
            raise ValueError(f"ID {mid}: limit margin is larger than the joint range")
        result[mid] = inner
    return result


def load_fixed_model(path, motor_ids):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(
            f"Fixed-base model not found: {path}\n"
            "Build it in robonex-description before running this tool."
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root") >= 0:
        raise RuntimeError(f"Model has a free root joint; use a fixed-base scene: {path}")

    actuator_ids = {}
    for mid in motor_ids:
        actuator_name = MOTOR_ACTUATORS[mid]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if aid < 0:
            raise RuntimeError(f"MuJoCo actuator not found: ID {mid} -> {actuator_name}")
        joint_name = actuator_name + "_joint"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0 or int(model.actuator_trnid[aid, 0]) != jid:
            raise RuntimeError(f"MuJoCo actuator-joint mismatch: {actuator_name} -> {joint_name}")
        actuator_ids[mid] = aid
    return path, model, actuator_ids


def verify_model_limits(model, actuator_ids, motor_ids):
    for mid in motor_ids:
        got = tuple(float(v) for v in model.actuator_ctrlrange[actuator_ids[mid]])
        want = JOINT_LIMITS_RAD[mid]
        if not np.allclose(got, want, atol=5e-6, rtol=0.0):
            raise RuntimeError(
                f"ID {mid} limit mismatch: MuJoCo={got}, common={want}. "
                "Use matching source and generated models."
            )


def open_hardware(motor_ids, interface, host_id):
    buses = {}
    motors = {}
    channels = sorted({channel_for_id(mid) for mid in motor_ids})
    try:
        for channel in channels:
            buses[channel] = can.Bus(channel=channel, interface=interface)
        for mid in motor_ids:
            motors[mid] = Motor(
                buses[channel_for_id(mid)], mid, SPECS[MOTOR_MODELS[mid]], host_id=host_id
            )
    except Exception:
        for bus in buses.values():
            try:
                bus.shutdown()
            except Exception:
                pass
        raise
    hubs = {
        channel: FeedbackHub(
            bus,
            [motor for mid, motor in motors.items() if channel_for_id(mid) == channel],
            host_id,
        )
        for channel, bus in buses.items()
    }
    return buses, motors, hubs


def inspect_zero_positions(motors, tolerance_rad, limits):
    print("\nPreflight zero-position check using read-only mechPos:")
    blocking_failures = []
    measured = {}
    print(f"  {'ID':>2} {'joint':<18} {'raw/wrap':>18} {'move to zero':>15}  status")
    for mid in sorted(motors):
        position = motors[mid].read_mech_position(timeout=0.3)
        measured[mid] = position
        if position is None:
            blocking_failures.append(f"ID {mid} did not respond")
            print(f"  {mid:>2} {JOINT_MAP[mid]:<18} {'no response':>15} {'--':>15}  BLOCK")
            continue
        if not math.isfinite(position):
            blocking_failures.append(f"ID {mid} mechPos is NaN or infinite")
            print(f"  {mid:>2} {JOINT_MAP[mid]:<18} {'NaN/inf':>15} {'--':>15}  BLOCK")
            continue
        wrapped = wrap_to_pi(position)
        degrees = math.degrees(position)
        wrapped_deg = math.degrees(wrapped)
        lower, upper = limits[mid]
        if not lower <= wrapped <= upper:
            status = "BLOCK"
            blocking_failures.append(
                f"ID {mid} {degrees:+.3f} deg (wrapped {wrapped_deg:+.3f} deg) is outside "
                f"{math.degrees(lower):+.3f}..{math.degrees(upper):+.3f} deg"
            )
        elif abs(wrapped) <= tolerance_rad:
            status = "near zero"
        else:
            status = "will move to zero"
        print(
            f"  {mid:>2} {JOINT_MAP[mid]:<18} "
            f"{degrees:+7.2f}/{wrapped_deg:+7.2f}deg "
            f"{(-wrapped_deg):+11.3f} deg  {status}"
        )
    return measured, blocking_failures


def confirm_hardware(args, motor_ids, model_path):
    print("\nThe real motors will move.")
    print(f"  model       : {model_path}")
    print(f"  motor IDs   : {motor_ids}")
    print(f"  max speed   : {args.max_speed:.3f} rad/s ({math.degrees(args.max_speed):.2f} deg/s)")
    print(f"  max accel   : {args.max_accel:.3f} rad/s^2")
    print(f"  kp / kd     : {args.kp:g} / {args.kd:g}")
    print("  Startup      : Move selected motors slowly to zero, then start tracking")
    print("  Required     : Fixed stand, clear workspace, and ready physical E-stop")
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Hardware confirmation requires an interactive terminal or --yes")
    answer = input("Press Enter to move to zero and start tracking, or Ctrl-C to cancel: ")
    if answer.strip():
        raise RuntimeError("Cancelled because the input was not empty")


def enable_with_runtime_feedback(motors, hubs, kp, kd, limits):
    starts = {}
    enabled_ids = []

    stop_errors = []
    for mid, motor in motors.items():
        try:
            motor.stop()
        except (OSError, can.CanError) as error:
            stop_errors.append(f"ID {mid}: {error}")
    if stop_errors:
        raise RuntimeError("Preflight stop failed: " + "; ".join(stop_errors))
    time.sleep(0.05)

    for mid in sorted(motors):
        motor = motors[mid]
        motor.write_run_mode_operation()
        time.sleep(0.005)
        motor.enable()
        enabled_ids.append(mid)
        hub = hubs[channel_for_id(mid)]
        start = hub.wait_for(mid, timeout=0.3)
        if start is None:
            error = RuntimeError(f"ID {mid} has no type-0x02 feedback after enable")
            error.enabled_ids = enabled_ids
            raise error
        lower, upper = limits[mid]
        wrapped = wrap_to_pi(start) if math.isfinite(start) else start
        if not math.isfinite(start) or not lower <= wrapped <= upper:
            error = RuntimeError(
                f"ID {mid} feedback is non-finite or outside the safe range after enable: "
                f"{math.degrees(start):+.3f}deg (wrap {math.degrees(wrapped):+.3f}deg)"
            )
            error.enabled_ids = enabled_ids
            raise error
        starts[mid] = start
        motor.control(pos=start, vel=0.0, kp=kp, kd=kd, torque=0.0)
    return starts, enabled_ids


def move_to_zero(motors, hubs, starts, limits, args):
    tolerance_rad = math.radians(args.zero_tolerance_deg)
    period = 1.0 / args.rate
    commands = dict(starts)
    zero_targets = {mid: align_angle(starts[mid], 0.0) for mid in starts}
    limiters = {mid: AxisLimiter(starts[mid]) for mid in starts}
    next_tick = time.monotonic()
    last_tick = next_tick
    last_print = 0.0
    commanded_zero_at = None

    print("\nStarting the speed-limited move to zero.")
    while True:
        now = time.monotonic()

        for hub in hubs.values():
            hub.pump()
        reason = runtime_safety_reason(motors, commands, limits, now, args)
        if reason:
            raise RuntimeError("Zero move stopped for safety: " + reason)

        dt = clamp(now - last_tick, period * 0.25, period * 2.0)
        last_tick = now
        velocities = {}
        for mid in sorted(motors):
            position, velocity = limiters[mid].step(
                zero_targets[mid], dt, args.max_speed, args.max_accel
            )
            commands[mid] = position
            velocities[mid] = velocity

        for mid, motor in motors.items():
            motor.control(
                pos=commands[mid], vel=velocities[mid],
                kp=args.kp, kd=args.kd, torque=0.0,
            )

        if now - last_print >= 1.0:
            last_print = now
            print(f"[{time.strftime('%H:%M:%S')}] Moving to zero")
            print(f"  {'ID':>2} {'joint':<18} {'limited cmd':>12} {'actual':>11}")
            for mid in sorted(motors):
                actual = motors[mid].last_position
                actual_text = (
                    "--" if actual is None
                    else f"{math.degrees(wrap_to_pi(actual)):+8.2f}deg"
                )
                print(
                    f"  {mid:>2} {JOINT_MAP[mid]:<18} "
                    f"{math.degrees(wrap_to_pi(commands[mid])):+9.2f}deg {actual_text:>11}"
                )

        command_done = all(
            abs(commands[mid] - zero_targets[mid]) <= 1e-12
            and abs(velocities[mid]) <= 1e-12
            for mid in motors
        )
        actual_done = all(
            motors[mid].last_position is not None
            and abs(wrap_to_pi(motors[mid].last_position)) <= tolerance_rad
            for mid in motors
        )
        if command_done and actual_done:
            print(
                "Zero position reached "
                f"(measured position within ±{args.zero_tolerance_deg:g} deg)."
            )
            return dict(commands)
        if command_done:
            if commanded_zero_at is None:
                commanded_zero_at = now
            elif now - commanded_zero_at > ZERO_SETTLE_TIMEOUT:
                outside = [
                    f"ID {mid} {math.degrees(motors[mid].last_position):+.2f}deg "
                    f"(wrap {math.degrees(wrap_to_pi(motors[mid].last_position)):+.2f}deg)"
                    for mid in sorted(motors)
                    if motors[mid].last_position is not None
                    and abs(wrap_to_pi(motors[mid].last_position)) > tolerance_rad
                ]
                raise RuntimeError(
                    f"Failed to reach zero within {ZERO_SETTLE_TIMEOUT:.1f} s: "
                    + ", ".join(outside)
                )
        else:
            commanded_zero_at = None

        next_tick += period
        sleep = next_tick - time.monotonic()
        if sleep > 0.0:
            time.sleep(sleep)
        elif time.monotonic() - next_tick > period:
            next_tick = time.monotonic()


def runtime_safety_reason(motors, commands, limits, now, args):
    for mid, motor in motors.items():
        if motor.last_feedback_time <= 0.0 or motor.last_position is None:
            return f"ID {mid} never received type-0x02 feedback"
        age = now - motor.last_feedback_time
        if age > args.feedback_timeout:
            return f"ID {mid} feedback timeout ({age:.3f} s > {args.feedback_timeout:.3f} s)"
        if not all(math.isfinite(v) for v in (
            motor.last_position, motor.last_velocity, motor.last_torque, motor.last_temp
        )):
            return f"ID {mid} feedback contains NaN or infinity"
        if abs(motor.last_velocity) > args.overspeed:
            return f"ID {mid} overspeed ({motor.last_velocity:+.3f} rad/s)"
        if motor.last_temp > args.max_temp:
            return f"ID {mid} overtemperature ({motor.last_temp:.1f} degC)"
        lower, upper = limits[mid]
        wrapped = wrap_to_pi(motor.last_position)
        if not lower <= wrapped <= upper:
            return (
                f"ID {mid} left the safe joint range ("
                f"{math.degrees(motor.last_position):+.2f}deg "
                f"wrap {math.degrees(wrapped):+.2f}deg, "
                f"allowed {math.degrees(lower):+.2f}..{math.degrees(upper):+.2f} deg)"
            )
        if mid in commands:
            error = wrap_to_pi(commands[mid] - motor.last_position)
            if abs(error) > math.radians(args.max_error_deg):
                return f"ID {mid} tracking error is too large ({math.degrees(error):+.2f} deg)"
    return None


def brake_and_stop(motors, buses, enabled_ids, stop_ids, duration, kd):
    enabled = [mid for mid in sorted(set(enabled_ids)) if mid in motors]
    if enabled and duration > 0.0:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            for mid in enabled:
                try:
                    motor = motors[mid]
                    motor.control(
                        pos=0.0, vel=0.0, kp=0.0,
                        kd=min(kd, motor.spec.kd_max), torque=0.0,
                    )
                except (OSError, can.CanError):
                    pass
            time.sleep(0.01)

    for mid in sorted(set(stop_ids)):
        motor = motors.get(mid)
        if motor is None:
            continue
        try:
            motor.stop()
        except (OSError, can.CanError):
            pass
    for bus in buses.values():
        try:
            bus.shutdown()
        except Exception:
            pass


def print_status(motors, commands, targets, hardware):
    print(f"[{time.strftime('%H:%M:%S')}] " + ("hardware following" if hardware else "dry-run"))
    print(f"  {'ID':>2} {'joint':<18} {'sim target':>11} {'limited cmd':>12} {'actual':>11}")
    for mid in sorted(commands):
        actual = motors[mid].last_position if hardware else None
        actual_text = (
            "--" if actual is None
            else f"{math.degrees(wrap_to_pi(actual)):+8.2f}deg"
        )
        print(
            f"  {mid:>2} {JOINT_MAP[mid]:<18} "
            f"{math.degrees(targets[mid]):+8.2f}deg "
            f"{math.degrees(wrap_to_pi(commands[mid])):+9.2f}deg {actual_text:>11}"
        )


def run(args):
    motor_ids, problems = validate_args(args)
    if problems:
        raise RuntimeError("Argument error:\n  " + "\n  ".join(problems))

    margin_rad = math.radians(args.limit_margin_deg)
    hard_limits = {mid: JOINT_LIMITS_RAD[mid] for mid in motor_ids}
    command_limits = safe_limits(motor_ids, margin_rad)
    model_path, model, actuator_ids = load_fixed_model(args.model, motor_ids)
    verify_model_limits(model, actuator_ids, motor_ids)
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    buses = {}
    motors = {}
    hubs = {}
    enabled_ids = []
    stop_ids = []
    commands = {mid: 0.0 for mid in motor_ids}
    limiters = {mid: AxisLimiter(0.0) for mid in motor_ids}

    try:
        if args.hardware:
            buses, motors, hubs = open_hardware(motor_ids, args.interface, args.host_id)
            _, zero_failures = inspect_zero_positions(
                motors, math.radians(args.zero_tolerance_deg), hard_limits
            )
            if zero_failures:
                raise RuntimeError(
                    "Preflight safety check failed; motors will not be enabled:\n  "
                    + "\n  ".join(zero_failures)
                )
            confirm_hardware(args, motor_ids, model_path)
            stop_ids = list(motor_ids)
            try:
                starts, enabled_ids = enable_with_runtime_feedback(
                    motors, hubs, args.kp, args.kd, hard_limits,
                )
            except RuntimeError as error:
                enabled_ids = getattr(error, "enabled_ids", enabled_ids)
                raise
            commands.update(starts)
            commands.update(move_to_zero(motors, hubs, starts, hard_limits, args))
            limiters = {mid: AxisLimiter(commands[mid]) for mid in motor_ids}
        else:
            print("Simulation-only dry run. No CAN bus is opened and no motor command is sent.")
            print("Use --hardware only after the physical safety setup is ready.")

        viewer_context = (
            contextlib.nullcontext(None)
            if args.headless else mujoco.viewer.launch_passive(model, data)
        )
        with viewer_context as viewer:
            if args.hardware:
                print("\nHardware tracking is active. Use only the MuJoCo Control sliders.")

            period = 1.0 / args.rate
            sim_steps = max(1, int(round(period / model.opt.timestep)))
            next_tick = time.monotonic()
            last_tick = next_tick
            start_time = next_tick
            last_print = 0.0

            while viewer is None or viewer.is_running():
                now = time.monotonic()
                if args.duration is not None and now - start_time >= args.duration:
                    break

                if args.hardware:
                    for hub in hubs.values():
                        hub.pump()
                    reason = runtime_safety_reason(motors, commands, hard_limits, now, args)
                    if reason:
                        raise RuntimeError("Safety stop: " + reason)

                lock = viewer.lock() if viewer is not None else contextlib.nullcontext()
                with lock:
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise RuntimeError("MuJoCo state contains NaN or infinity; motor commands stopped")
                    targets = {}
                    for mid in motor_ids:
                        raw_target = float(data.ctrl[actuator_ids[mid]])
                        if not math.isfinite(raw_target):
                            raise RuntimeError(f"ID {mid} MuJoCo target is NaN or infinite")
                        lower, upper = command_limits[mid]
                        targets[mid] = clamp(raw_target, lower, upper)

                dt = clamp(now - last_tick, period * 0.25, period * 2.0)
                last_tick = now
                command_velocities = {}
                for mid in motor_ids:
                    position, velocity = limiters[mid].step(
                        align_angle(limiters[mid].position, targets[mid]),
                        dt, args.max_speed, args.max_accel,
                    )
                    commands[mid] = position
                    command_velocities[mid] = velocity

                if args.hardware:
                    for mid, motor in motors.items():
                        motor.control(
                            pos=commands[mid], vel=command_velocities[mid],
                            kp=args.kp, kd=args.kd, torque=0.0,
                        )

                lock = viewer.lock() if viewer is not None else contextlib.nullcontext()
                with lock:
                    mujoco.mj_step(model, data, nstep=sim_steps)
                if viewer is not None:
                    viewer.sync()

                if now - last_print >= 1.0:
                    last_print = now
                    print_status(motors, commands, targets, args.hardware)

                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0.0:
                    time.sleep(sleep)
                elif time.monotonic() - next_tick > period:
                    next_tick = time.monotonic()
    finally:
        if args.hardware:
            brake_and_stop(
                motors, buses, enabled_ids, stop_ids, args.brake_time, args.kd
            )
            if stop_ids:
                print("Active damping and stop/disable shutdown completed.")
            else:
                print("CAN buses closed. No motor control command was sent during the preflight check.")


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStop requested.")
        return 0
    except (RuntimeError, ValueError, OSError, can.CanError) as error:
        print(f"\nStopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
