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

from robonex_can import (
    DEFAULT_INTERFACE,
    HOST_ID,
    JOINT_LIMITS_RAD,
    JOINT_MAP,
    clamp,
    stop_idle_motors,
)

from robonex_can import MOTOR_MODELS
from robonex_common.joints import MOTOR_BY_ID, VARIANT_MOTOR_IDS
from bench import (
    ROBOT_VARIANTS,
    attached_robot_model,
    bench_gains,
    channel_problems,
    check_gain_table,
    deg_text,
    format_ids,
    identity_path,
    open_checked,
    placeholder_ids,
    print_banner,
    print_table,
    resolve_motor_ids,
    resolve_variant,
    selection_help,
    sim_state,
)
from safety import (
    AxisLimiter,
    align_angle,
    brake_and_stop,
    enable_with_runtime_feedback,
    inspect_zero_positions,
    fixed_model_path,
    load_fixed_model,
    open_hardware,
    runtime_safety_reason,
    shutdown_report_lines,
    clip_roll_targets,
    require_robot_model,
    roll_pairs_for,
    safe_limits,
    verify_model_limits,
    wrap_to_pi,
)

ZERO_SETTLE_TIMEOUT = 5.0
ROBOT_VARIANTS_SHORT = {name: short for short, name in ROBOT_VARIANTS.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Track fixed-base MuJoCo targets with RoboNex motors."
    )
    parser.add_argument("--motor-id", dest="motor_id", nargs="+", metavar="MOTOR",
                        help=selection_help() + ". A motor without a MuJoCo actuator on the attached variant is moved to zero and held there")
    parser.add_argument("--robot", choices=(*ROBOT_VARIANTS, *ROBOT_VARIANTS.values()),
                        help="Check only: must match the robot identity file, which sets the variant")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve variant, motors, gains, limits and buses and print the start table; no CAN bus is opened")
    args = parser.parse_args(argv)
    try:
        args.variant, args.variant_source = resolve_variant(args.robot, need_identity=not args.dry_run)
        args.motor_ids = resolve_motor_ids(args.motor_id, args.variant)
    except ValueError as error:
        parser.error(str(error))
    args.model = fixed_model_path(ROBOT_VARIANTS_SHORT[args.variant])
    args.max_speed = 0.10
    args.max_accel = 0.25
    args.interface = DEFAULT_INTERFACE
    args.host_id = HOST_ID
    args.rate = 100.0
    args.gain_scale = 1.0
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
    motor_ids = sorted(set(args.motor_ids))
    unknown = [mid for mid in motor_ids if mid not in MOTOR_MODELS]
    if unknown:
        problems.append(f"Unsupported motor ID: {unknown}")

    numeric_positive = ("max_speed", "max_accel")
    for name in numeric_positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            problems.append(f"--{name.replace('_', '-')} must be finite and positive ({value})")
    return motor_ids, problems







def print_motor_table(args, motor_ids, hard_limits, actuator_ids, capped):
    placeholders = set(placeholder_ids(motor_ids))
    rows = []
    for mid in motor_ids:
        joint = MOTOR_BY_ID[mid]
        lower, upper = hard_limits[mid]
        notes = [text for flag, text in ((mid in placeholders, "PLACEHOLDER"), (mid in capped, "capped")) if flag]
        rows.append([
            mid, JOINT_MAP[mid], joint.motor_model, joint.channel,
            f"{args.kp[mid]:g}", f"{args.kd[mid]:g}",
            f"{math.degrees(lower):+.1f}..{math.degrees(upper):+.1f}",
            sim_state(mid, actuator_ids), " ".join(notes),
        ])
    print_table(("ID", "joint", "motor", "bus", "kp", "kd", "limit deg", "sim", "note"), rows)


def confirm_hardware(args, motor_ids, model_path, hard_limits, actuator_ids, capped):
    print("\nThe real motors will move.")
    print(f"  model       : {model_path}")
    print(f"  max speed   : {args.max_speed:.3f} rad/s ({math.degrees(args.max_speed):.2f} deg/s)")
    print(f"  max accel   : {args.max_accel:.3f} rad/s^2")
    print_motor_table(args, motor_ids, hard_limits, actuator_ids, capped)
    print("  Keep the robot fixed and the emergency stop ready.")
    if not sys.stdin.isatty():
        raise RuntimeError("Hardware confirmation requires an interactive terminal")
    answer = input("Press Enter to move to zero and start tracking, or Ctrl-C to cancel: ")
    if answer.strip():
        raise RuntimeError("Cancelled because the input was not empty")


def dry_run_report(args, motor_ids, model_path, hard_limits, actuator_ids, capped):
    print("\nDry run: nothing below is sent to a motor.")
    print(f"  model       : {model_path}")
    print_motor_table(args, motor_ids, hard_limits, actuator_ids, capped)
    blockers = channel_problems(motor_ids, args.interface)
    if attached_robot_model(identity_path()) is None:
        blockers.append(f"no robot identity file at {identity_path()} (variant came from --robot)")
    if blockers:
        print("A real run would stop before enabling:\n  " + "\n  ".join(blockers))
    print(f"Would enable ID {format_ids(motor_ids)} after the preflight zero check and the Enter confirmation.")


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
                kp=args.kp[mid], kd=args.kd[mid], torque=0.0,
            )

        if now - last_print >= 1.0:
            last_print = now
            print(f"[{time.strftime('%H:%M:%S')}] Moving to zero")
            print_table(("ID", "joint", "limited cmd", "actual"), [
                [mid, JOINT_MAP[mid], deg_text(wrap_to_pi(commands[mid])), actual_deg(motors[mid])]
                for mid in sorted(motors)
            ])

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




def actual_deg(motor):
    actual = motor.last_position
    return deg_text(None if actual is None else wrap_to_pi(actual))


def print_status(motors, commands, targets, actuator_ids):
    print(f"[{time.strftime('%H:%M:%S')}] hardware tracking")
    print_table(("ID", "joint", "sim target", "limited cmd", "actual", "sim"), [
        [
            mid, JOINT_MAP[mid], deg_text(targets[mid]), deg_text(wrap_to_pi(commands[mid])),
            actual_deg(motors[mid]), sim_state(mid, actuator_ids),
        ]
        for mid in sorted(commands)
    ])


def run(args):
    motor_ids, problems = validate_args(args)
    if problems:
        raise RuntimeError("Argument error:\n  " + "\n  ".join(problems))

    check_gain_table()
    margin_rad = math.radians(args.limit_margin_deg)
    args.kp, args.kd, capped = bench_gains(motor_ids, args.gain_scale)
    model_path, model, variant_actuators = load_fixed_model(args.model, VARIANT_MOTOR_IDS[args.variant])
    actuator_ids = {mid: aid for mid, aid in variant_actuators.items() if mid in motor_ids}
    profile = verify_model_limits(model, actuator_ids, motor_ids)
    model_limits = profile.joint_limits_by_id()
    hard_limits = {mid: model_limits[mid] if mid in model_limits else JOINT_LIMITS_RAD[mid] for mid in motor_ids}
    command_limits = safe_limits(motor_ids, margin_rad, hard_limits)
    print_banner(
        args.variant, args.variant_source, motor_ids, variant_actuators, model_path.name,
        "moved to zero and held there", args.interface,
    )
    print(f"Leg limits  : {profile.name} profile")
    roll_pairs = roll_pairs_for(profile, motor_ids)
    if args.dry_run:
        dry_run_report(args, motor_ids, model_path, hard_limits, actuator_ids, capped)
        return
    require_robot_model(args.variant, identity_path())
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
        buses, motors, hubs = open_checked(open_hardware, motor_ids, args.interface, args.host_id)
        idle = stop_idle_motors(buses, motor_ids, args.host_id, VARIANT_MOTOR_IDS[args.variant])
        if idle:
            print(f"Stop sent to the other motors on the open buses: {idle}")
        _, zero_failures = inspect_zero_positions(
            motors, math.radians(args.zero_tolerance_deg), hard_limits
        )
        if zero_failures:
            raise RuntimeError(
                "Preflight safety check failed; motors will not be enabled:\n  "
                + "\n  ".join(zero_failures)
            )
        confirm_hardware(args, motor_ids, model_path, hard_limits, actuator_ids, capped)
        stop_ids = list(motor_ids)
        try:
            starts, enabled_ids = enable_with_runtime_feedback(
                motors, hubs, args.kp, args.kd, hard_limits, enabled_out=enabled_ids,
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
                        if mid not in actuator_ids:
                            targets[mid] = commands[mid]
                            continue
                        raw_target = float(data.ctrl[actuator_ids[mid]])
                        if not math.isfinite(raw_target):
                            raise RuntimeError(f"ID {mid} MuJoCo target is NaN or infinite")
                        lower, upper = command_limits[mid]
                        targets[mid] = clamp(raw_target, lower, upper)
                if roll_pairs:
                    targets = clip_roll_targets(targets, roll_pairs, command_limits, profile.foot_roll)

                dt = clamp(now - last_tick, period * 0.25, period * 2.0)
                last_tick = now
                command_velocities = {}
                previous = {mid: limiters[mid].position for mid in motor_ids}
                for mid in motor_ids:
                    position, velocity = limiters[mid].step(
                        targets[mid],
                        dt, args.max_speed, args.max_accel,
                    )
                    commands[mid] = position
                    command_velocities[mid] = velocity
                if roll_pairs:
                    projected = clip_roll_targets(commands, roll_pairs, command_limits, profile.foot_roll)
                    for mid in motor_ids:
                        limiters[mid].position = projected[mid]
                        limiters[mid].velocity = (projected[mid] - previous[mid]) / dt
                        commands[mid] = projected[mid]
                        command_velocities[mid] = limiters[mid].velocity

                for mid, motor in motors.items():
                    motor.control(
                        pos=commands[mid], vel=command_velocities[mid],
                        kp=args.kp[mid], kd=args.kd[mid], torque=0.0,
                    )

                with viewer.lock():
                    mujoco.mj_step(model, data, nstep=sim_steps)
                viewer.sync()

                if now - last_print >= 1.0:
                    last_print = now
                    print_status(motors, commands, targets, actuator_ids)

                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0.0:
                    time.sleep(sleep)
                elif time.monotonic() - next_tick > period:
                    next_tick = time.monotonic()
    finally:
        shutdown_report = brake_and_stop(
            motors, buses, enabled_ids, stop_ids, args.brake_time, args.kd
        )
        if stop_ids:
            for line in shutdown_report_lines(shutdown_report):
                print(line)
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
