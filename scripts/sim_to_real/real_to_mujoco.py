from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys
import time

import can
import mujoco
import mujoco.viewer
import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(THIS_FILE.parent))
sys.path.insert(0, str(THIS_FILE.parents[1]))

from robonex_common.paths import description_model

DEFAULT_MODEL_PATH = description_model("mujoco/robot/scene_fixed.xml", anchors=(__file__,))

from robonex_can import (
    DEFAULT_INTERFACE,
    HOST_ID,
    JOINT_MAP,
    MOTOR_MODELS,
    channel_for_id,
    clamp,
)
from safety import load_fixed_model, open_hardware

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Mirror hand-moved RoboNex mechPos values into fixed-base MuJoCo. "
            "Only stop messages are sent at startup and shutdown."
        )
    )
    parser.add_argument(
        "--motor-id",
        dest="motor_id",
        nargs="+",
        type=lambda value: int(value, 0),
        default=list(range(1, 13)),
        help="Motor IDs to read. Default: 1 through 12",
    )
    args = parser.parse_args(argv)
    args.model = DEFAULT_MODEL_PATH
    args.interface = DEFAULT_INTERFACE
    args.host_id = HOST_ID
    args.rate = 30.0
    args.read_timeout = 0.03
    args.startup_timeout = 2.0
    args.stale_timeout = 0.5
    args.limit_tolerance_deg = 1.0
    return args


def validate_args(args):
    problems = []
    motor_ids = sorted(set(args.motor_id))
    unknown = [motor_id for motor_id in motor_ids if motor_id not in MOTOR_MODELS]
    if unknown:
        problems.append(f"Unsupported motor ID: {unknown}")
    return motor_ids, problems


def confirm_hardware(args, motor_ids, model_path):
    print("\nThis tool does not drive motors. It only reads hand-moved positions.")
    print(f"  model       : {model_path}")
    print(f"  motor IDs   : {motor_ids}")
    print("  Motors remain disabled; only stop and position-read messages are sent.")
    print("  Keep the robot fixed and joints clear.")
    if not sys.stdin.isatty():
        raise RuntimeError("Start confirmation requires an interactive terminal")
    answer = input("Press Enter to send stop and start mirroring, or Ctrl-C to cancel: ")
    if answer.strip():
        raise RuntimeError("Cancelled because the input was not empty")


def stop_motors(motors, required):
    failures = []
    for motor_id in sorted(motors):
        try:
            motors[motor_id].stop()
        except (OSError, can.CanError) as error:
            failures.append(f"ID {motor_id}: {error}")
    if required and failures:
        raise RuntimeError("Failed to send stop: " + "; ".join(failures))


def motors_by_channel(motors):
    groups = {}
    for motor_id, motor in motors.items():
        groups.setdefault(channel_for_id(motor_id), {})[motor_id] = motor
    return groups


def read_group_positions(group, read_timeout):
    positions = {}
    for motor_id in sorted(group):
        value = group[motor_id].read_mech_position(timeout=read_timeout)
        if value is not None:
            positions[motor_id] = value
    return positions


def read_all_positions(groups, executor, read_timeout):
    if not groups:
        return {}
    if len(groups) == 1:
        return read_group_positions(next(iter(groups.values())), read_timeout)
    positions = {}
    futures = [
        executor.submit(read_group_positions, group, read_timeout)
        for group in groups.values()
    ]
    for future in futures:
        positions.update(future.result())
    return positions


def collect_initial_positions(motors, groups, executor, timeout, read_timeout):
    deadline = time.monotonic() + timeout
    positions = {}
    while len(positions) < len(motors) and time.monotonic() < deadline:
        remaining = {
            channel: {
                motor_id: motor
                for motor_id, motor in group.items()
                if motor_id not in positions
            }
            for channel, group in groups.items()
        }
        remaining = {channel: group for channel, group in remaining.items() if group}
        positions.update(read_all_positions(remaining, executor, read_timeout))
    missing = sorted(set(motors) - set(positions))
    if missing:
        raise RuntimeError(f"No initial mechPos response: {missing}")
    return positions


def model_limits_rad(model, actuator_ids):
    return {
        motor_id: tuple(float(v) for v in model.actuator_ctrlrange[actuator_id])
        for motor_id, actuator_id in actuator_ids.items()
    }


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def validate_positions(positions, limits, tolerance_rad):
    targets = {}
    clamped = set()
    for motor_id, value in positions.items():
        if not math.isfinite(value):
            raise RuntimeError(f"ID {motor_id} mechPos is NaN or infinite")
        wrapped = wrap_to_pi(value)
        lower, upper = limits[motor_id]
        if wrapped < lower - tolerance_rad or wrapped > upper + tolerance_rad:
            raise RuntimeError(
                f"ID {motor_id} mechPos {math.degrees(value):+.2f}deg "
                f"(wrapped {math.degrees(wrapped):+.2f} deg) is outside the model range "
                f"{math.degrees(lower):+.2f}..{math.degrees(upper):+.2f} deg"
            )
        target = clamp(wrapped, lower, upper)
        if target != wrapped:
            clamped.add(motor_id)
        targets[motor_id] = target
    return targets, clamped


def poll_positions(motors, groups, executor, positions, last_seen, read_timeout, stale_timeout):
    got = read_all_positions(groups, executor, read_timeout)
    now = time.monotonic()
    for motor_id, value in got.items():
        positions[motor_id] = value
        last_seen[motor_id] = now
    stale = [
        f"ID {motor_id} {now - last_seen[motor_id]:.3f}s"
        for motor_id in sorted(motors)
        if now - last_seen[motor_id] > stale_timeout
    ]
    if stale:
        raise RuntimeError("Stale mechPos feedback: " + ", ".join(stale))


def initialize_simulation(model, data, actuator_ids, targets):
    qpos_addresses = {}
    dof_addresses = {}
    for motor_id, actuator_id in actuator_ids.items():
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        qpos_addresses[motor_id] = qpos_address
        dof_addresses[motor_id] = dof_address
        data.ctrl[actuator_id] = targets[motor_id]
        data.qpos[qpos_address] = targets[motor_id]
        data.qvel[dof_address] = 0.0
    mujoco.mj_forward(model, data)
    warmup_steps = max(1, int(round(0.2 / model.opt.timestep)))
    mujoco.mj_step(model, data, nstep=warmup_steps)
    return qpos_addresses


def print_status(data, qpos_addresses, positions, targets, clamped):
    print(f"[{time.strftime('%H:%M:%S')}] real -> MuJoCo")
    print(f"  {'ID':>2} {'joint':<18} {'real':>11} {'sim target':>11} {'sim qpos':>11} {'state':>7}")
    for motor_id in sorted(targets):
        state = "CLAMP" if motor_id in clamped else "OK"
        print(
            f"  {motor_id:>2} {JOINT_MAP[motor_id]:<18} "
            f"{math.degrees(positions[motor_id]):+8.2f}deg "
            f"{math.degrees(targets[motor_id]):+8.2f}deg "
            f"{math.degrees(data.qpos[qpos_addresses[motor_id]]):+8.2f}deg "
            f"{state:>7}"
        )


def run(args):
    motor_ids, problems = validate_args(args)
    if problems:
        raise RuntimeError("Argument error:\n  " + "\n  ".join(problems))
    model_path, model, actuator_ids = load_fixed_model(args.model, motor_ids)
    limits = model_limits_rad(model, actuator_ids)
    model.opt.gravity[:] = 0.0
    confirm_hardware(args, motor_ids, model_path)

    buses = {}
    motors = {}
    positions = {}
    try:
        buses, motors, _ = open_hardware(
            motor_ids, args.interface, args.host_id
        )
        stop_motors(motors, required=True)
        time.sleep(0.05)
        groups = motors_by_channel(motors)
        with ThreadPoolExecutor(max_workers=max(1, len(groups))) as executor:
            positions = collect_initial_positions(
                motors, groups, executor, args.startup_timeout, args.read_timeout
            )
            targets, clamped = validate_positions(
                positions, limits, math.radians(args.limit_tolerance_deg)
            )
            last_seen = {motor_id: time.monotonic() for motor_id in motor_ids}

            data = mujoco.MjData(model)
            data.ctrl[:] = 0.0
            qpos_addresses = initialize_simulation(
                model, data, actuator_ids, targets
            )
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Initial MuJoCo state contains NaN or infinity")

            period = 1.0 / args.rate
            sim_steps = max(1, int(round(period / model.opt.timestep)))
            start_time = time.monotonic()
            next_tick = start_time
            last_print = 0.0

            print("\nPosition mirroring started. Close the viewer or press Ctrl-C to stop.")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    now = time.monotonic()
                    poll_positions(
                        motors,
                        groups,
                        executor,
                        positions,
                        last_seen,
                        args.read_timeout,
                        args.stale_timeout,
                    )
                    targets, clamped = validate_positions(
                        positions, limits, math.radians(args.limit_tolerance_deg)
                    )
                    with viewer.lock():
                        for motor_id, target in targets.items():
                            data.ctrl[actuator_ids[motor_id]] = target
                        mujoco.mj_step(model, data, nstep=sim_steps)
                        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                            raise RuntimeError("MuJoCo state contains NaN or infinity")
                    viewer.sync()
                    if now - last_print >= 1.0:
                        last_print = now
                        print_status(data, qpos_addresses, positions, targets, clamped)
                    next_tick += period
                    sleep = next_tick - time.monotonic()
                    if sleep > 0.0:
                        time.sleep(sleep)
                    elif time.monotonic() - next_tick > period:
                        next_tick = time.monotonic()
    finally:
        if motors:
            stop_motors(motors, required=False)
        for bus in buses.values():
            try:
                bus.shutdown()
            except Exception:
                pass
    print("Stop state held and CAN shutdown completed.")


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
