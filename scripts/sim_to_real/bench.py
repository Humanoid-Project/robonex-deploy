from __future__ import annotations

import math
import os
from pathlib import Path

import can
from robonex_common.actuators import CONTROL_GAINS_BY_JOINT
from robonex_common.buses import bus_map_path
from robonex_common.joints import ACTUATED_JOINTS, ALL_MOTORS, MOTOR_BY_ID, VARIANT_MOTOR_IDS
from robonex_common.motors import MOTOR_SPECS

ROBOT_IDENTITY_ENV = "ROBONEX_ROBOT_MODEL_FILE"
ROBOT_IDENTITY_FILE = Path.home() / ".config" / "robonex" / "robot_model"
ROBOT_VARIANTS = {name.removeprefix("ver2_"): name for name in VARIANT_MOTOR_IDS}
LEG_MOTOR_IDS = tuple(joint.motor_id for joint in ACTUATED_JOINTS)
GROUP_ORDER = ("left_leg", "right_leg", "head", "left_arm", "right_arm")
GROUP_ALIASES = {
    **{group: (group,) for group in GROUP_ORDER},
    "legs": ("left_leg", "right_leg"),
    "arms": ("left_arm", "right_arm"),
}
MOTOR_BY_NAME = {
    name: joint
    for joint in ALL_MOTORS
    for name in (joint.hardware_name, joint.model_name, joint.model_name.removesuffix("_joint"))
}
PLACEHOLDER_LIMITS_DEG = (30.0, 45.0)
SYS_NET = Path("/sys/class/net")
CAN_BITRATE = 1000000


def identity_path():
    configured = os.environ.get(ROBOT_IDENTITY_ENV)
    return Path(configured).expanduser() if configured else ROBOT_IDENTITY_FILE


def attached_robot_model(path=None):
    path = ROBOT_IDENTITY_FILE if path is None else Path(path)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"cannot read the robot identity file {path}: {error.strerror or error}") from None
    return value or None


def variant_name(value):
    name = ROBOT_VARIANTS.get(value, value)
    if name not in VARIANT_MOTOR_IDS:
        known = ", ".join([*ROBOT_VARIANTS, *VARIANT_MOTOR_IDS])
        raise ValueError(f"unknown robot variant {value!r} (known: {known})")
    return name


def resolve_variant(override=None, need_identity=True):
    path = identity_path()
    attached = attached_robot_model(path)
    requested = variant_name(override) if override else None
    if attached is None:
        if requested and not need_identity:
            return requested, f"--robot {override} (no identity file at {path})"
        raise ValueError(
            f"no robot identity file: write the attached robot's variant once with "
            f"`mkdir -p {path.parent} && echo ver2_edu > {path}` (ver2_edu, ver2_pro or ver2_max)"
        )
    if attached not in VARIANT_MOTOR_IDS:
        raise ValueError(f"{path} names an unknown robot variant {attached!r} (known: {', '.join(VARIANT_MOTOR_IDS)})")
    if requested and requested != attached:
        raise ValueError(f"--robot {override} disagrees with the attached robot {attached} ({path})")
    return attached, str(path)


def format_ids(motor_ids):
    ids = sorted(set(motor_ids))
    if not ids:
        return "none"
    runs = []
    start = previous = ids[0]
    for mid in ids[1:]:
        if mid == previous + 1:
            previous = mid
            continue
        runs.append((start, previous))
        start = previous = mid
    runs.append((start, previous))
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in runs)


SELECTION_WORDS = f"motor IDs, groups ({', '.join([*GROUP_ALIASES, 'all'])}) or joint names (e.g. left_knee_pitch)"


def selection_help():
    return f"{SELECTION_WORDS[0].upper()}{SELECTION_WORDS[1:]}; mix freely. Default: the 12 leg motors"


def _token_ids(token, variant, variant_ids):
    word = token.strip().lower()
    if word == "all":
        return set(variant_ids)
    if word in GROUP_ALIASES:
        groups = GROUP_ALIASES[word]
        ids = {joint.motor_id for joint in ALL_MOTORS if joint.group in groups and joint.motor_id in variant_ids}
        if not ids:
            raise ValueError(f"{variant} has no {word} motors (its motors: {format_ids(variant_ids)})")
        return ids
    if word in MOTOR_BY_NAME:
        return {MOTOR_BY_NAME[word].motor_id}
    try:
        return {int(word, 0)}
    except ValueError:
        raise ValueError(f"unknown motor selection {token!r}; use {SELECTION_WORDS}") from None


def resolve_motor_ids(tokens, variant):
    variant_ids = set(VARIANT_MOTOR_IDS[variant])
    if not tokens:
        return list(LEG_MOTOR_IDS)
    ids = set()
    for token in tokens:
        ids |= _token_ids(token, variant, variant_ids)
    outside = sorted(ids - variant_ids)
    if outside:
        raise ValueError(
            f"motor ID {format_ids(outside)} is not on {variant}; {variant} has {format_ids(variant_ids)}"
        )
    return sorted(ids)


def check_gain_table():
    missing = [joint.motor_id for joint in ALL_MOTORS if joint.model_name not in CONTROL_GAINS_BY_JOINT]
    if missing:
        raise RuntimeError(f"robonex-common has no gain entry for motor ID {format_ids(missing)}")


def bench_gains(motor_ids, scale=1.0):
    kp_by_motor, kd_by_motor, capped = {}, {}, []
    for mid in motor_ids:
        joint = MOTOR_BY_ID[mid]
        spec = MOTOR_SPECS[joint.motor_model]
        kp, kd = CONTROL_GAINS_BY_JOINT[joint.model_name]
        kp_by_motor[mid] = min(kp * scale, spec.kp_max)
        kd_by_motor[mid] = min(kd * scale, spec.kd_max)
        if kp * scale > spec.kp_max or kd * scale > spec.kd_max:
            capped.append(mid)
    return kp_by_motor, kd_by_motor, capped


def placeholder_ids(motor_ids):
    found = []
    for mid in motor_ids:
        joint = MOTOR_BY_ID[mid]
        if mid in LEG_MOTOR_IDS:
            continue
        for degrees in PLACEHOLDER_LIMITS_DEG:
            bound = math.radians(degrees)
            if math.isclose(joint.lower, -bound, abs_tol=1e-5) and math.isclose(joint.upper, bound, abs_tol=1e-5):
                found.append(mid)
                break
    return found


def bus_plan(motor_ids):
    plan = {}
    for mid in sorted(motor_ids):
        joint = MOTOR_BY_ID[mid]
        entry = plan.setdefault(joint.channel, {})
        entry.setdefault(joint.group, []).append(mid)
    return dict(sorted(plan.items()))


def channel_state(channel, interface="socketcan", sys_net=SYS_NET):
    if interface != "socketcan":
        return "ok"
    device = Path(sys_net) / channel
    if not device.exists():
        return "missing"
    try:
        flags = int((device / "flags").read_text().strip(), 16)
    except (OSError, ValueError):
        return "ok"
    return "ok" if flags & 0x1 else "down"


def up_command(channel):
    return f"sudo ip link set {channel} up type can bitrate {CAN_BITRATE}"


def channel_fix_lines(channel, groups, state):
    users = ", ".join(f"{group} {format_ids(ids)}" for group, ids in groups.items())
    reason = {
        "missing": f"no such interface on this machine (adapter unplugged, or wrong name in {bus_map_path()})",
        "down": "interface is down",
    }.get(state, state)
    return [f"{channel}: {reason}; needed by {users}", f"    {up_command(channel)}"]


def channel_problems(motor_ids, interface="socketcan", sys_net=SYS_NET):
    lines = []
    for channel, groups in bus_plan(motor_ids).items():
        state = channel_state(channel, interface, sys_net)
        if state != "ok":
            lines.extend(channel_fix_lines(channel, groups, state))
    return lines


def open_checked(open_hardware, motor_ids, interface, host_id, sys_net=SYS_NET):
    problems = channel_problems(motor_ids, interface, sys_net)
    if problems:
        raise RuntimeError("CAN channel not ready:\n  " + "\n  ".join(problems))
    try:
        return open_hardware(motor_ids, interface, host_id)
    except (OSError, can.CanError) as error:
        lines = []
        for channel, groups in bus_plan(motor_ids).items():
            lines.extend(channel_fix_lines(channel, groups, "could not be opened"))
        raise RuntimeError(f"Could not open CAN ({error}):\n  " + "\n  ".join(lines)) from None


def sim_state(motor_id, actuator_ids):
    return "ok" if motor_id in actuator_ids else "not in sim"


def print_banner(variant, source, motor_ids, actuator_ids, model_name, not_in_sim="not simulated",
                 interface="socketcan", sys_net=SYS_NET):
    variant_ids = VARIANT_MOTOR_IDS[variant]
    print(f"Robot       : {variant} (from {source})")
    print(f"Variant has : {format_ids(variant_ids)}")
    print(f"Selected    : {format_ids(motor_ids)} ({len(motor_ids)} motors)")
    print(f"CAN buses   : (map {bus_map_path()})")
    for channel, groups in bus_plan(motor_ids).items():
        users = ", ".join(f"{group} {format_ids(ids)}" for group, ids in groups.items())
        state = channel_state(channel, interface, sys_net)
        print(f"  {channel:<6} ← {users}" + ("" if state == "ok" else f"  [{state}]"))
    unmodelled = sorted(mid for mid in variant_ids if mid not in actuator_ids)
    if unmodelled:
        print(f"WARNING     : {model_name} has no actuator for {len(unmodelled)} {variant} motor(s): "
              f"{format_ids(unmodelled)}")
    not_simulated = [mid for mid in motor_ids if mid not in actuator_ids]
    if not_simulated:
        print(f"Not in sim  : {len(not_simulated)} selected — {format_ids(not_simulated)} ({not_in_sim})")
    placeholders = placeholder_ids(motor_ids)
    if placeholders:
        print(f"PLACEHOLDER : limits of ID {format_ids(placeholders)} equal the common placeholder values "
              "(±30° head, ±45° arms); kp/kd from robonex-common")


def print_table(headers, rows, indent="  "):
    widths = [max([len(str(header))] + [len(str(row[i])) for row in rows]) for i, header in enumerate(headers)]
    for line in (headers, *rows):
        cells = [
            f"{str(cell):<{widths[i]}}" if i == 1 else f"{str(cell):>{widths[i]}}"
            for i, cell in enumerate(line)
        ]
        print(indent + " ".join(cells).rstrip())


def deg_text(radians, width=8):
    if radians is None:
        return "--"
    return f"{math.degrees(radians):+{width}.2f}deg"
