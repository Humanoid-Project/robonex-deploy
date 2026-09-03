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

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(THIS_FILE.parent))

from robonex_common.paths import description_model
from robonex_can import (
    JOINT_LIMITS_RAD,
    DEFAULT_INTERFACE,
    FeedbackHub,
    HOST_ID,
    JOINT_MAP,
    Motor,
    SPECS,
    channel_for_id,
    clamp,
)

from robonex_can import MOTOR_ACTUATORS, MOTOR_MODELS
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

DEFAULT_MODEL_PATH = description_model("mujoco/scene_fixed.xml", anchors=(__file__,))

MAX_CONFIG_SPEED = 2.0
MAX_CONFIG_ACCEL = 2.0
MAX_CONFIG_RATE = 200.0
ZERO_SETTLE_TIMEOUT = 5.0
MIN_SEGMENT_TIME = 0.05

SEQUENCE = [
    (0.35, {1: +0.0000, 2: +0.0000, 3: +0.0000, 4: -0.1200, 5: +0.0500, 6: -0.0500,
            7: +0.0000, 8: -0.0000, 9: +0.0000, 10: +0.1200, 11: -0.0500, 12: +0.0500}),
    (0.35, {1: +0.0000, 2: +0.1646, 3: +0.0000, 4: -0.4433, 5: +0.1088, 6: -0.1088,
            7: +0.0000, 8: -0.0000, 9: +0.0000, 10: +0.1200, 11: -0.0500, 12: +0.0500}),
    (0.35, {1: +0.0000, 2: +0.2663, 3: +0.0000, 4: -0.6431, 5: +0.1451, 6: -0.1451,
            7: +0.0000, 8: -0.0000, 9: +0.0000, 10: +0.1200, 11: -0.0500, 12: +0.0500}),
    (0.35, {1: +0.0000, 2: +0.2663, 3: +0.0000, 4: -0.6431, 5: +0.1451, 6: -0.1451,
            7: +0.0000, 8: -0.0000, 9: +0.0000, 10: +0.1200, 11: -0.0500, 12: +0.0500}),
    (0.35, {1: +0.0000, 2: +0.1646, 3: +0.0000, 4: -0.4433, 5: +0.1088, 6: -0.1088,
            7: +0.0000, 8: -0.0000, 9: +0.0000, 10: +0.1200, 11: -0.0500, 12: +0.0500}),
    (0.35, {1: +0.0000, 2: +0.0000, 3: +0.0000, 4: -0.1200, 5: +0.0500, 6: -0.0500,
            7: +0.0000, 8: -0.0000, 9: +0.0000, 10: +0.1200, 11: -0.0500, 12: +0.0500}),
    (0.35, {1: +0.0000, 2: +0.0000, 3: +0.0000, 4: -0.1200, 5: +0.0500, 6: -0.0500,
            7: +0.0000, 8: -0.1646, 9: +0.0000, 10: +0.4433, 11: -0.1088, 12: +0.1088}),
    (0.35, {1: +0.0000, 2: +0.0000, 3: +0.0000, 4: -0.1200, 5: +0.0500, 6: -0.0500,
            7: +0.0000, 8: -0.2663, 9: +0.0000, 10: +0.6431, 11: -0.1451, 12: +0.1451}),
    (0.35, {1: +0.0000, 2: +0.0000, 3: +0.0000, 4: -0.1200, 5: +0.0500, 6: -0.0500,
            7: +0.0000, 8: -0.2663, 9: +0.0000, 10: +0.6431, 11: -0.1451, 12: +0.1451}),
    (0.35, {1: +0.0000, 2: +0.0000, 3: +0.0000, 4: -0.1200, 5: +0.0500, 6: -0.0500,
            7: +0.0000, 8: -0.1646, 9: +0.0000, 10: +0.4433, 11: -0.1088, 12: +0.1088}),
    (0.35, {1: +0.0000, 2: +0.0000, 3: +0.0000, 4: -0.1200, 5: +0.0500, 6: -0.0500,
            7: +0.0000, 8: -0.0000, 9: +0.0000, 10: +0.1200, 11: -0.0500, 12: +0.0500}),
]





def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Play a timed keyframe sequence on RoboNex. MuJoCo is visualisation only; "
            "simulation-only unless --hardware is set."
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
    parser.add_argument("--max-speed", type=float, default=1.0,
                        help="Maximum joint speed the sequence may demand in rad/s. Default: 1.0")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Multiply every segment duration by this factor before playback "
                             "(>1 = slower, <1 = faster; SEQUENCE itself is untouched). Default: 1.0")
    parser.add_argument("--approach-speed", type=float, default=0.10,
                        help="Speed for the zero move and the first-keyframe approach. Default: 0.10")
    parser.add_argument("--approach-accel", type=float, default=0.25,
                        help="Acceleration for the zero move and the approach. Default: 0.25")
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--zero-tolerance-deg", type=float, default=3.0,
                        help="Position tolerance for reaching zero in degrees. Default: 3")
    parser.add_argument("--limit-margin-deg", type=float, default=2.0,
                        help="Margin inside joint limits in degrees. Default: 2")
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
                        help="Run without a viewer")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the sequence and exit without running")
    return parser.parse_args(argv)


def validate_args(args):
    problems = []
    motor_ids = sorted(set(args.motor_id))
    unknown = [mid for mid in motor_ids if mid not in MOTOR_MODELS]
    if unknown:
        problems.append(f"Unsupported motor ID: {unknown}")

    numeric_positive = (
        "rate", "max_speed", "approach_speed", "approach_accel", "feedback_timeout",
        "overspeed", "max_error_deg", "max_temp", "time_scale",
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
    if args.rate > MAX_CONFIG_RATE:
        problems.append(f"--rate must be at most {MAX_CONFIG_RATE:g} Hz")
    if args.max_speed > MAX_CONFIG_SPEED:
        problems.append(f"--max-speed must be at most {MAX_CONFIG_SPEED:g} rad/s")
    if args.approach_speed > args.max_speed:
        problems.append("--approach-speed must not exceed --max-speed")
    if args.approach_accel > MAX_CONFIG_ACCEL:
        problems.append(f"--approach-accel must be at most {MAX_CONFIG_ACCEL:g} rad/s^2")
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



def validate_sequence(sequence, motor_ids, command_limits, max_speed, time_scale):
    problems = []
    if not sequence:
        problems.append("SEQUENCE is empty")
        return problems, []

    for index, entry in enumerate(sequence):
        if not isinstance(entry, tuple) or len(entry) != 2:
            problems.append(f"keyframe {index}: expected a (duration, targets) tuple")
            continue
        duration, targets = entry
        if not isinstance(duration, (int, float)) or not math.isfinite(duration):
            problems.append(f"keyframe {index}: duration is not a finite number ({duration})")
        else:
            effective_duration = duration * time_scale
            if effective_duration < MIN_SEGMENT_TIME:
                problems.append(
                    f"keyframe {index}: duration {duration:g} s x time-scale {time_scale:g} = "
                    f"{effective_duration:g} s is below the {MIN_SEGMENT_TIME:g} s minimum"
                )
        if not isinstance(targets, dict):
            problems.append(f"keyframe {index}: targets is not a dict")
            continue
        missing = [mid for mid in motor_ids if mid not in targets]
        if missing:
            problems.append(f"keyframe {index}: missing motor IDs {missing}")
        extra = [mid for mid in targets if mid not in motor_ids]
        if extra:
            problems.append(f"keyframe {index}: unselected motor IDs {extra}")
        for mid in motor_ids:
            if mid not in targets:
                continue
            value = targets[mid]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"keyframe {index} ID {mid}: target is not a finite number ({value})")
                continue
            lower, upper = command_limits[mid]
            if not lower <= value <= upper:
                problems.append(
                    f"keyframe {index} ID {mid} ({JOINT_MAP[mid]}): "
                    f"{math.degrees(value):+.3f} deg is outside the safe range "
                    f"{math.degrees(lower):+.3f}..{math.degrees(upper):+.3f} deg"
                )
    if problems:
        return problems, []

    segments = []
    for index in range(1, len(sequence)):
        duration = float(sequence[index][0]) * time_scale
        previous = sequence[index - 1][1]
        targets = sequence[index][1]
        worst_id = None
        worst_speed = 0.0
        for mid in motor_ids:
            speed = abs(targets[mid] - previous[mid]) / duration
            if speed > worst_speed:
                worst_speed = speed
                worst_id = mid
        segments.append((index, duration, worst_id, worst_speed))
        if worst_speed > max_speed:
            problems.append(
                f"segment {index - 1}->{index} needs {worst_speed:.3f} rad/s on ID {worst_id} "
                f"({JOINT_MAP[worst_id]}) but --max-speed is {max_speed:.3f} rad/s"
            )
    return problems, segments


def report_sequence(sequence, motor_ids, segments, command_limits, args):
    total = sum(float(sequence[i][0]) for i in range(1, len(sequence))) * args.time_scale
    print("\nSequence preflight")
    print(f"  keyframes        : {len(sequence)}")
    scale_note = "" if args.time_scale == 1.0 else f" (time-scale x{args.time_scale:g})"
    print(f"  timed segments   : {len(segments)} ({total:.2f} s total{scale_note})")
    print(f"  motor IDs        : {motor_ids}")
    print(f"  first keyframe   : reached by a speed-limited approach at "
          f"{args.approach_speed:.3f} rad/s; its duration field is not used")
    if segments:
        peak = max(segments, key=lambda s: s[3])
        print(f"  peak demand      : {peak[3]:.3f} rad/s on ID {peak[2]} "
              f"({JOINT_MAP[peak[2]]}) in segment {peak[0] - 1}->{peak[0]}")
    tight = []
    for index, (_, targets) in enumerate(sequence):
        for mid in motor_ids:
            lower, upper = command_limits[mid]
            margin = min(targets[mid] - lower, upper - targets[mid])
            tight.append((margin, index, mid))
    tight.sort()
    print("  closest approach to the safe limit:")
    for margin, index, mid in tight[:3]:
        print(f"    ID {mid:>2} {JOINT_MAP[mid]:<18} {math.degrees(margin):5.2f} deg at keyframe {index}")






def confirm_hardware(args, motor_ids, model_path, sequence, segments):
    total = sum(float(sequence[i][0]) for i in range(1, len(sequence))) * args.time_scale
    peak = max(segments, key=lambda s: s[3])[3] if segments else 0.0
    print("\nThe real motors will move through the whole sequence.")
    print(f"  model         : {model_path}")
    print(f"  motor IDs     : {motor_ids}")
    print(f"  keyframes     : {len(sequence)} ({total:.2f} s of timed playback)")
    print(f"  peak demand   : {peak:.3f} rad/s ({math.degrees(peak):.1f} deg/s)")
    print(f"  approach      : {args.approach_speed:.3f} rad/s to zero, then to keyframe 0")
    print(f"  kp / kd       : {args.kp:g} / {args.kd:g}")
    print("  Startup        : move to zero, approach keyframe 0, then play the sequence")
    print("  Required       : Fixed stand, clear workspace, and ready physical E-stop")
    if args.yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Hardware confirmation requires an interactive terminal or --yes")
    answer = input("Press Enter to start, or Ctrl-C to cancel: ")
    if answer.strip():
        raise RuntimeError("Cancelled because the input was not empty")



def move_speed_limited(motors, hubs, starts, goals, limits, args, label, tolerance_rad):
    period = 1.0 / args.rate
    commands = dict(starts)
    limiters = {mid: AxisLimiter(starts[mid]) for mid in starts}
    next_tick = time.monotonic()
    last_tick = next_tick
    last_print = 0.0
    commanded_at = None

    print(f"\nStarting the speed-limited move: {label}")
    while True:
        now = time.monotonic()

        for hub in hubs.values():
            hub.pump()
        reason = runtime_safety_reason(motors, commands, limits, now, args)
        if reason:
            raise RuntimeError(f"{label} stopped for safety: " + reason)

        dt = clamp(now - last_tick, period * 0.25, period * 2.0)
        last_tick = now
        velocities = {}
        for mid in sorted(motors):
            position, velocity = limiters[mid].step(
                goals[mid], dt, args.approach_speed, args.approach_accel
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
            print(f"[{time.strftime('%H:%M:%S')}] {label}")
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
            abs(commands[mid] - goals[mid]) <= 1e-12 and abs(velocities[mid]) <= 1e-12
            for mid in motors
        )
        actual_done = all(
            motors[mid].last_position is not None
            and abs(motors[mid].last_position - goals[mid]) <= tolerance_rad
            for mid in motors
        )
        if command_done and actual_done:
            print(f"{label} reached (measured within ±{math.degrees(tolerance_rad):g} deg).")
            return dict(commands)
        if command_done:
            if commanded_at is None:
                commanded_at = now
            elif now - commanded_at > ZERO_SETTLE_TIMEOUT:
                outside = [
                    f"ID {mid} {math.degrees(motors[mid].last_position - goals[mid]):+.2f}deg off"
                    for mid in sorted(motors)
                    if motors[mid].last_position is not None
                    and abs(motors[mid].last_position - goals[mid]) > tolerance_rad
                ]
                raise RuntimeError(
                    f"{label} did not settle within {ZERO_SETTLE_TIMEOUT:.1f} s: "
                    + ", ".join(outside)
                )
        else:
            commanded_at = None

        next_tick += period
        sleep = next_tick - time.monotonic()
        if sleep > 0.0:
            time.sleep(sleep)
        elif time.monotonic() - next_tick > period:
            next_tick = time.monotonic()


def animate_sim_approach(model, data, actuator_ids, motor_ids, goals, args, viewer, period, sim_steps, label):
    limiters = {mid: AxisLimiter(0.0) for mid in motor_ids}
    commands = {mid: 0.0 for mid in motor_ids}
    next_tick = time.monotonic()
    last_tick = next_tick
    print(f"\n{label}")
    while True:
        now = time.monotonic()
        if viewer is not None and not viewer.is_running():
            raise RuntimeError("Viewer closed; sequence aborted")

        dt = clamp(now - last_tick, period * 0.25, period * 2.0)
        last_tick = now
        done = True
        for mid in motor_ids:
            position, velocity = limiters[mid].step(
                goals[mid], dt, args.approach_speed, args.approach_accel
            )
            commands[mid] = position
            data.ctrl[actuator_ids[mid]] = position
            if abs(position - goals[mid]) > 1e-9 or abs(velocity) > 1e-9:
                done = False

        lock = viewer.lock() if viewer is not None else contextlib.nullcontext()
        with lock:
            mujoco.mj_step(model, data, nstep=sim_steps)
        if viewer is not None:
            viewer.sync()
        if done:
            return commands

        next_tick += period
        sleep = next_tick - time.monotonic()
        if sleep > 0.0:
            time.sleep(sleep)
        elif time.monotonic() - next_tick > period:
            next_tick = time.monotonic()




def print_status(motors, commands, targets, index, elapsed, duration, hardware):
    print(f"[{time.strftime('%H:%M:%S')}] segment {index - 1}->{index} "
          f"{elapsed:.2f}/{duration:.2f} s " + ("hardware" if hardware else "dry-run"))
    print(f"  {'ID':>2} {'joint':<18} {'keyframe':>11} {'command':>11} {'actual':>11}")
    for mid in sorted(targets):
        actual = motors[mid].last_position if hardware else None
        actual_text = (
            "--" if actual is None
            else f"{math.degrees(wrap_to_pi(actual)):+8.2f}deg"
        )
        print(
            f"  {mid:>2} {JOINT_MAP[mid]:<18} "
            f"{math.degrees(targets[mid]):+8.2f}deg "
            f"{math.degrees(wrap_to_pi(commands[mid])):+8.2f}deg {actual_text:>11}"
        )


def run(args):
    motor_ids, problems = validate_args(args)
    if problems:
        raise RuntimeError("Argument error:\n  " + "\n  ".join(problems))

    margin_rad = math.radians(args.limit_margin_deg)
    hard_limits = {mid: JOINT_LIMITS_RAD[mid] for mid in motor_ids}
    command_limits = safe_limits(motor_ids, margin_rad)

    seq_problems, segments = validate_sequence(
        SEQUENCE, motor_ids, command_limits, args.max_speed, args.time_scale
    )
    if seq_problems:
        raise RuntimeError("Sequence error:\n  " + "\n  ".join(seq_problems))
    report_sequence(SEQUENCE, motor_ids, segments, command_limits, args)

    model_path, model, actuator_ids = load_fixed_model(args.model, motor_ids)
    verify_model_limits(model, actuator_ids, motor_ids)
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    if args.dry_run:
        print("\nDry run only. The sequence and the model are valid; nothing was executed.")
        return

    buses = {}
    motors = {}
    hubs = {}
    enabled_ids = []
    stop_ids = []
    zero_reference = {mid: 0.0 for mid in motor_ids}
    commands = {mid: 0.0 for mid in motor_ids}
    completed = False

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
            confirm_hardware(args, motor_ids, model_path, SEQUENCE, segments)
            stop_ids = list(motor_ids)
            try:
                starts, enabled_ids = enable_with_runtime_feedback(
                    motors, hubs, args.kp, args.kd, hard_limits,
                )
            except RuntimeError as error:
                enabled_ids = getattr(error, "enabled_ids", enabled_ids)
                raise
            zero_goals = {mid: align_angle(starts[mid], 0.0) for mid in motor_ids}
            commands = move_speed_limited(
                motors, hubs, starts, zero_goals, hard_limits, args,
                "move to zero", math.radians(args.zero_tolerance_deg),
            )
            zero_reference = dict(zero_goals)
            first_goals = {
                mid: zero_reference[mid] + SEQUENCE[0][1][mid] for mid in motor_ids
            }
            commands = move_speed_limited(
                motors, hubs, commands, first_goals, hard_limits, args,
                "approach keyframe 0", math.radians(args.zero_tolerance_deg),
            )
        else:
            print("\nSimulation-only dry run. No CAN bus is opened and no motor command is sent.")
            print("Use --hardware only after the physical safety setup is ready.")

        for mid in motor_ids:
            data.ctrl[actuator_ids[mid]] = commands[mid] - zero_reference[mid]

        viewer_context = (
            contextlib.nullcontext(None)
            if args.headless else mujoco.viewer.launch_passive(model, data)
        )
        with viewer_context as viewer:
            period = 1.0 / args.rate
            sim_steps = max(1, int(round(period / model.opt.timestep)))

            if not args.hardware:
                sim_goals = {mid: SEQUENCE[0][1][mid] for mid in motor_ids}
                commands = animate_sim_approach(
                    model, data, actuator_ids, motor_ids, sim_goals, args, viewer,
                    period, sim_steps, "Approaching keyframe 0 at --approach-speed.",
                )

            print(f"\nPlaying {len(segments)} timed segments.")

            for index in range(1, len(SEQUENCE)):
                duration = float(SEQUENCE[index][0]) * args.time_scale
                previous = SEQUENCE[index - 1][1]
                targets = SEQUENCE[index][1]
                velocities = {
                    mid: (targets[mid] - previous[mid]) / duration for mid in motor_ids
                }
                segment_start = time.monotonic()
                next_tick = segment_start
                last_print = 0.0

                while True:
                    now = time.monotonic()
                    elapsed = now - segment_start
                    alpha = 1.0 if elapsed >= duration else elapsed / duration

                    if viewer is not None and not viewer.is_running():
                        raise RuntimeError("Viewer closed; sequence aborted")

                    if args.hardware:
                        for hub in hubs.values():
                            hub.pump()
                        reason = runtime_safety_reason(
                            motors, commands, hard_limits, now, args
                        )
                        if reason:
                            raise RuntimeError("Safety stop: " + reason)

                    for mid in motor_ids:
                        joint_target = previous[mid] + alpha * (targets[mid] - previous[mid])
                        lower, upper = command_limits[mid]
                        joint_target = clamp(joint_target, lower, upper)
                        commands[mid] = zero_reference[mid] + joint_target
                        data.ctrl[actuator_ids[mid]] = joint_target

                    if args.hardware:
                        for mid, motor in motors.items():
                            motor.control(
                                pos=commands[mid],
                                vel=0.0 if alpha >= 1.0 else velocities[mid],
                                kp=args.kp, kd=args.kd, torque=0.0,
                            )

                    lock = viewer.lock() if viewer is not None else contextlib.nullcontext()
                    with lock:
                        mujoco.mj_step(model, data, nstep=sim_steps)
                    if viewer is not None:
                        viewer.sync()

                    if now - last_print >= 1.0:
                        last_print = now
                        print_status(
                            motors, commands, targets, index, elapsed, duration, args.hardware
                        )

                    if alpha >= 1.0:
                        break

                    next_tick += period
                    sleep = next_tick - time.monotonic()
                    if sleep > 0.0:
                        time.sleep(sleep)
                    elif time.monotonic() - next_tick > period:
                        next_tick = time.monotonic()

            print("\nSequence finished.")
            completed = True

        if args.hardware and completed:
            zero_goals = {mid: zero_reference[mid] for mid in motor_ids}
            commands = move_speed_limited(
                motors, hubs, commands, zero_goals, hard_limits, args,
                "return to zero", math.radians(args.zero_tolerance_deg),
            )
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
