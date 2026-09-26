#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import can

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "sim_to_real"))

from robonex_common.actuators import CONTROL_GAINS_BY_JOINT
from robonex_common.joints import JOINT_BY_ID
from robonex_can import DEFAULT_INTERFACE, HOST_ID, JOINT_MAP, MOTOR_MODELS, SPECS, Motor
from safety import (
    brake_and_stop,
    enable_with_runtime_feedback,
    open_hardware,
    runtime_safety_reason,
    safe_limits,
    shutdown_report_lines,
    wrap_to_pi,
)

MAX_AMPLITUDE_RAD = 0.15
CONTINUOUS_TORQUE = {"rs02": 6.0, "rs03": 13.0}
MAX_TARGET_SPEED = 3.0
MAX_DURATION_S = 120.0
LIMIT_MARGIN_RAD = math.radians(3.0)


def profile_offset(args, t):
    a = args.amplitude
    if args.profile == "step":
        cycle = 4.0 * args.hold
        phase = (t % cycle) / args.hold
        return (a, 0.0, -a, 0.0)[int(phase) % 4]
    if args.profile == "triangle":
        phase = (t / args.period + 0.25) % 1.0
        return a * (4.0 * phase - 1.0 if phase < 0.5 else 3.0 - 4.0 * phase)
    k = math.log(args.f1 / args.f0) / args.duration
    phase = 2.0 * math.pi * args.f0 * (math.exp(k * t) - 1.0) / k
    return a * math.sin(phase)


def peak_target_speed(args):
    if args.profile == "step":
        return 0.0
    if args.profile == "triangle":
        return 4.0 * args.amplitude / args.period
    return 2.0 * math.pi * args.f1 * args.amplitude


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Drive ONE motor of a hung robot through a small step, triangle or chirp around its "
                    "current position and log feedback at the command rate (system identification)."
    )
    parser.add_argument("--motor-id", type=lambda v: int(v, 0), required=True)
    parser.add_argument("--profile", choices=("step", "triangle", "chirp"), required=True)
    parser.add_argument("--amplitude", type=float, default=0.05, help="rad, at most %.2f" % MAX_AMPLITUDE_RAD)
    parser.add_argument("--duration", type=float, default=20.0, help="seconds, at most %.0f" % MAX_DURATION_S)
    parser.add_argument("--hold", type=float, default=1.0, help="step: seconds per level")
    parser.add_argument("--period", type=float, default=10.0, help="triangle: seconds per cycle")
    parser.add_argument("--f0", type=float, default=0.2, help="chirp: start frequency (Hz)")
    parser.add_argument("--f1", type=float, default=5.0, help="chirp: end frequency (Hz)")
    parser.add_argument("--rate", type=float, default=200.0, help="command/feedback rate (Hz), 50..250")
    parser.add_argument("--gain-scale", type=float, default=1.0,
                        help="fraction of the per-joint walking gains (kp, kd)")
    parser.add_argument("--output", type=Path, help="CSV path; default results/sysid/<stamp>_<id>_<profile>.csv")
    args = parser.parse_args(argv)
    if args.motor_id not in JOINT_BY_ID:
        parser.error(f"unknown motor id {args.motor_id}")
    for name in ("amplitude", "duration", "hold", "period", "f0", "f1", "rate", "gain_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.amplitude > MAX_AMPLITUDE_RAD:
        parser.error(f"--amplitude is capped at {MAX_AMPLITUDE_RAD} rad")
    if args.duration > MAX_DURATION_S:
        parser.error(f"--duration is capped at {MAX_DURATION_S:.0f} s")
    if args.gain_scale > 1.0:
        parser.error("--gain-scale must be at most 1.0")
    if args.f1 <= args.f0:
        parser.error("--f1 must be above --f0")
    if not 50.0 <= args.rate <= 250.0:
        parser.error("--rate must be 50..250 Hz (the measured per-motor CAN ceiling is ~250 Hz)")
    if args.profile == "step" and args.hold < 0.2:
        parser.error("--hold must be at least 0.2 s so each step settles before the next")
    if peak_target_speed(args) > MAX_TARGET_SPEED:
        parser.error(
            f"the profile's peak target speed {peak_target_speed(args):.2f} rad/s exceeds {MAX_TARGET_SPEED} rad/s; "
            "lower the amplitude or the frequency"
        )
    if not sys.stdin.isatty():
        parser.error("needs an interactive terminal for the confirmation")
    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = REPO_ROOT / "results" / "sysid" / f"{stamp}_id{args.motor_id}_{args.profile}.csv"
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(f"{args.output} exists; recordings are never overwritten")
    args.feedback_timeout = 0.30
    args.overspeed = 4.0
    args.max_error_deg = 15.0
    args.max_temp = 70.0
    return args


def run(args):
    mid = args.motor_id
    spec = JOINT_BY_ID[mid]
    kp_full, kd_full = CONTROL_GAINS_BY_JOINT[spec.model_name]
    kp, kd = kp_full * args.gain_scale, kd_full * args.gain_scale
    step_torque = kp * args.amplitude
    limit = CONTINUOUS_TORQUE[spec.motor_model]
    if step_torque > limit:
        raise SystemExit(
            f"kp {kp:g} x amplitude {args.amplitude:g} rad can demand {step_torque:.1f} N·m, above the "
            f"{spec.motor_model.upper()} continuous {limit:g} N·m; lower --amplitude or --gain-scale"
        )
    limits = safe_limits([mid], LIMIT_MARGIN_RAD)
    print(f"\nSystem-identification probe: ONE motor, ID {mid} ({JOINT_MAP[mid]}, {spec.model_name}).")
    print(f"  profile {args.profile}, amplitude {args.amplitude:.3f} rad ({math.degrees(args.amplitude):.1f} deg) "
          f"around the position at enable, {args.duration:.1f} s, {args.rate:.0f} Hz, kp {kp:g} kd {kd:g}")
    print(f"  worst-case PD torque demand {step_torque:.1f} N·m (continuous rating {limit:g} N·m)")
    print("  The robot must hang so that this joint moves freely; all other motors stay disabled.")
    print("  Ctrl-C brakes and stops. Keep the emergency stop within reach.")
    input("Press Enter to enable the motor and start, or Ctrl-C to cancel: ")

    buses, motors, hubs, enabled = {}, {}, {}, []
    rows = []
    status = "completed"
    try:
        buses, motors, hubs = open_hardware([mid], DEFAULT_INTERFACE, HOST_ID)
        bus = next(iter(buses.values()))
        for other in sorted(JOINT_BY_ID):
            if other != mid and JOINT_BY_ID[other].channel == spec.channel:
                Motor(bus, other, SPECS[MOTOR_MODELS[other]], host_id=HOST_ID).stop()
        starts, enabled = enable_with_runtime_feedback(motors, hubs, {mid: kp}, {mid: kd}, limits, enabled_out=enabled)
        center = starts[mid]
        lower, upper = limits[mid]
        if not (lower <= wrap_to_pi(center) - args.amplitude and wrap_to_pi(center) + args.amplitude <= upper):
            raise RuntimeError("the probe range around the current position leaves the safe joint range")
        motor = motors[mid]
        hub = next(iter(hubs.values()))
        period = 1.0 / args.rate
        started = time.monotonic()
        next_tick = started
        last_rx = None
        while True:
            now = time.monotonic()
            t = now - started
            if t >= args.duration:
                break
            hub.pump()
            target = center + profile_offset(args, t)
            reason = runtime_safety_reason(motors, {mid: target}, limits, now, args)
            if not reason and motor.last_fault:
                reason = f"ID {mid} reports fault flags 0x{int(motor.last_fault):02x} in its type-0x02 feedback"
            if reason:
                raise RuntimeError("Safety stop: " + reason)
            rx = getattr(motor, "last_rx_kernel_time", None)
            wall = time.time()
            motor.control(pos=target, vel=0.0, kp=kp, kd=kd, torque=0.0)
            sent = time.monotonic()
            rows.append([round(t, 6), round(wall, 6), round((sent - now) * 1000.0, 3), round(target, 6),
                         motor.last_position, motor.last_velocity, motor.last_torque, motor.last_temp,
                         motor.last_fault, "" if rx is None else round(rx, 6), int(rx is not None and rx != last_rx)])
            last_rx = rx
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0.0:
                time.sleep(sleep)
            elif time.monotonic() - next_tick > period:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        status = "stopped by operator"
        print("\nStop requested.")
    except (RuntimeError, OSError, can.CanError) as error:
        status = str(error)
        print(f"\nStopped: {error}")
    finally:
        report = brake_and_stop(motors, buses, enabled, list(motors), 0.3, {mid: kd_full} if motors else 0.0)
        if motors:
            for line in shutdown_report_lines(report):
                print(line)
            if report["stop_errors"]:
                status = "stop frame failed: " + "; ".join(f"ID {k}: {v}" for k, v in report["stop_errors"].items())
        if rows:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["t_s", "wall_time", "send_ms", "target", "pos", "vel", "torque", "temp",
                                 "fault", "rx_kernel_time", "new_frame"])
                writer.writerows(rows)
            meta = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
            meta.update({"joint": spec.model_name, "kp": kp, "kd": kd, "status": status, "rows": len(rows)})
            args.output.with_name(args.output.stem + "_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"Wrote {len(rows)} rows to {args.output}")
    return 0 if status == "completed" else 1


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
