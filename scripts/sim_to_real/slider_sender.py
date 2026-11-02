#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import mujoco

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parents[1]))
sys.path.insert(0, str(THIS_FILE.parent))

from bench import ROBOT_VARIANTS, deg_text, print_table
from safety import fixed_model_path
from slider_link import PORT, TargetSender, viewer_loop

ROBOT_VARIANTS_SHORT = {name: short for short, name in ROBOT_VARIANTS.items()}
STATUS_STALE = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Send MuJoCo Control slider targets over UDP to mujoco_to_real.py --remote on the robot. Opens no CAN bus."
    )
    parser.add_argument("--robot-host", required=True, metavar="HOST",
                        help="Robot computer running mujoco_to_real.py --remote <this computer>")
    parser.add_argument("--robot", required=True, choices=(*ROBOT_VARIANTS, *ROBOT_VARIANTS.values()),
                        help="Robot variant; must match the robot identity file on the robot computer")
    parser.add_argument("--rate", type=float, default=50.0, help="Send rate in Hz")
    args = parser.parse_args(argv)
    args.variant = ROBOT_VARIANTS.get(args.robot, args.robot)
    if not math.isfinite(args.rate) or not 10.0 <= args.rate <= 200.0:
        parser.error("--rate must be between 10 and 200 Hz")
    return args


def print_robot_status(status, age):
    stamp = time.strftime("%H:%M:%S")
    if status is None:
        print(f"[{stamp}] no status from the robot yet")
        return
    if age > STATUS_STALE:
        print(f"[{stamp}] no status from the robot for {age:.1f} s (check the robot terminal and the network)")
        return
    print(f"[{stamp}] robot {status.get('state')}: {status.get('note')}")
    rows = status.get("rows") or []
    if rows:
        print_table(("ID", "joint", "target", "limited cmd", "actual", "temp C"), [
            [mid, joint, deg_text(target), deg_text(command), deg_text(actual),
             "--" if temp is None else f"{temp:.1f}"]
            for mid, joint, _name, target, command, actual, temp in rows
        ])


def main(argv=None):
    args = parse_args(argv)
    model_path = fixed_model_path(ROBOT_VARIANTS_SHORT[args.variant])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    actuators = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid): aid for aid in range(model.nu)}
    sender = TargetSender(args.robot_host, args.variant)
    print(f"Model  : {model_path}")
    print(f"Robot  : {args.variant} at {sender.address[0]} (UDP port {PORT}), session {sender.session}")
    print("Move only the MuJoCo Control sliders. Closing this window makes the robot hold its position; "
          "it does not stop the robot program.")

    latest = {"status": None, "stamp": None, "printed": 0.0}

    def on_targets(targets):
        sender.send(targets)
        status = sender.poll_status()
        now = time.monotonic()
        if status is not None:
            latest["status"], latest["stamp"] = status, now
        if now - latest["printed"] >= 1.0:
            latest["printed"] = now
            age = math.inf if latest["stamp"] is None else now - latest["stamp"]
            print_robot_status(latest["status"], age)

    def sync():
        status, stamp = latest["status"], latest["stamp"]
        if status is None or time.monotonic() - stamp > 0.5:
            return None
        return status.get("sync")

    try:
        viewer_loop(model, data, actuators, on_targets, args.rate, sync=sync)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
    print("Slider sender closed; the robot holds its position until a sender reconnects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
