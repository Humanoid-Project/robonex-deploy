#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(THIS_FILE.parent))

from robonex_common.paths import description_model
from robonex_can import (
    JOINT_LIMITS_RAD,
    DEFAULT_INTERFACE,
    HOST_ID,
    JOINT_MAP,
    clamp,
)

from robonex_can import MOTOR_MODELS
from safety import (
    AxisLimiter,
    align_angle,
    brake_and_stop,
    enable_with_runtime_feedback,
    inspect_zero_positions,
    load_fixed_model,
    open_hardware,
    runtime_safety_reason,
    safe_limits,
    verify_model_limits,
    wrap_to_pi,
)

DEFAULT_MODEL_PATH = description_model("mujoco/robot/scene_fixed.xml", anchors=(__file__,))

ZERO_SETTLE_TIMEOUT = 5.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Track fixed-base MuJoCo targets with RoboNex motors."
    )
    parser.add_argument("--motor-id", dest="motor_id",
                        nargs="+", type=lambda v: int(v, 0),
                        default=list(range(1, 13)),
                        help="Motor IDs to control. Default: 1 through 12")
    args = parser.parse_args(argv)
    args.model = DEFAULT_MODEL_PATH
    args.max_speed = 0.10
    args.max_accel = 0.25
    args.interface = DEFAULT_INTERFACE
    args.host_id = HOST_ID
    args.rate = 100.0
    args.kp = 40.0
    args.kd = 2.0
    args.zero_tolerance_deg = 3.0
    args.limit_margin_deg = 3.0
    args.feedback_timeout = 0.30
    args.overspeed = 2.0
    args.max_error_deg = 25.0
    args.max_temp = 70.0
    args.brake_time = 0.20
    return args


def validate_args(args):
    problems = []
    motor_ids = sorted(set(args.motor_id))
    unknown = [mid for mid in motor_ids if mid not in MOTOR_MODELS]
    if unknown:
        problems.append(f"Unsupported motor ID: {unknown}")

    numeric_positive = ("max_speed", "max_accel")
    for name in numeric_positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            problems.append(f"--{name.replace('_', '-')} must be finite and positive ({value})")
    return motor_ids, problems







def confirm_hardware(args, motor_ids, model_path):
    print("\nThe real motors will move.")
    print(f"  model       : {model_path}")
    print(f"  motor IDs   : {motor_ids}")
    print(f"  max speed   : {args.max_speed:.3f} rad/s ({math.degrees(args.max_speed):.2f} deg/s)")
    print(f"  max accel   : {args.max_accel:.3f} rad/s^2")
    print("  Keep the robot fixed and the emergency stop ready.")
    if not sys.stdin.isatty():
        raise RuntimeError("Hardware confirmation requires an interactive terminal")
    answer = input("Press Enter to move to zero and start tracking, or Ctrl-C to cancel: ")
    if answer.strip():
        raise RuntimeError("Cancelled because the input was not empty")



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




def print_status(motors, commands, targets):
    print(f"[{time.strftime('%H:%M:%S')}] hardware tracking")
    print(f"  {'ID':>2} {'joint':<18} {'sim target':>11} {'limited cmd':>12} {'actual':>11}")
    for mid in sorted(commands):
        actual = motors[mid].last_position
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

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\nHardware tracking is active. Use only the MuJoCo Control sliders.")

            period = 1.0 / args.rate
            sim_steps = max(1, int(round(period / model.opt.timestep)))
            next_tick = time.monotonic()
            last_tick = next_tick
            last_print = 0.0

            while viewer.is_running():
                now = time.monotonic()

                for hub in hubs.values():
                    hub.pump()
                reason = runtime_safety_reason(motors, commands, hard_limits, now, args)
                if reason:
                    raise RuntimeError("Safety stop: " + reason)

                with viewer.lock():
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

                for mid, motor in motors.items():
                    motor.control(
                        pos=commands[mid], vel=command_velocities[mid],
                        kp=args.kp, kd=args.kd, torque=0.0,
                    )

                with viewer.lock():
                    mujoco.mj_step(model, data, nstep=sim_steps)
                viewer.sync()

                if now - last_print >= 1.0:
                    last_print = now
                    print_status(motors, commands, targets)

                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0.0:
                    time.sleep(sleep)
                elif time.monotonic() - next_tick > period:
                    next_tick = time.monotonic()
    finally:
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
