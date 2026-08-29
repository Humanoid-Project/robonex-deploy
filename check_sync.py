#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

SOURCES = {
    "robonex_can": [
        (PROJECT / "Robstride-Motor-Test/scripts/motor_control/motor_test/set_motor_pose.py",
         ["HOST_ID", "DEFAULT_INTERFACE", "CHANNEL_ID_RANGES", "JOINT_MAP",
          "MECH_POS_INDEX", "SPECS", "clamp", "float_to_uint",
          "build_arb", "parse_arb", "channel_for_id"]),
        (PROJECT / "Robstride-Motor-Test/scripts/measurements/common.py",
         ["JOINT_LIMITS_RAD", "DEFAULT_LIMIT_MARGIN_RAD"]),
    ],
}

ASSETS = {
    "assets/mujoco/robonex.xml": PROJECT / "robonex_description/mujoco/robonex.xml",
    "assets/mujoco/scene.xml": PROJECT / "robonex_description/mujoco/scene.xml",
    "assets/mujoco/robonex_fixed.xml": PROJECT / "robonex_description/mujoco/robonex_fixed.xml",
    "assets/mujoco/scene_fixed.xml": PROJECT / "robonex_description/mujoco/scene_fixed.xml",
    "assets/mujoco/full_limit/robonex_full_limit.xml": PROJECT / "robonex_description/mujoco/full_limit/robonex_full_limit.xml",
    "assets/mujoco/full_limit/robonex_fixed_full_limit.xml": PROJECT / "robonex_description/mujoco/full_limit/robonex_fixed_full_limit.xml",
    "assets/mujoco/full_limit/scene_full_limit.xml": PROJECT / "robonex_description/mujoco/full_limit/scene_full_limit.xml",
    "assets/mujoco/full_limit/scene_fixed_full_limit.xml": PROJECT / "robonex_description/mujoco/full_limit/scene_fixed_full_limit.xml",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spec_tuple(value):
    if hasattr(value, "p_min"):
        return (value.p_min, value.p_max, value.v_min, value.v_max,
                value.t_min, value.t_max, value.kp_max, value.kd_max)
    return value


def compare(local, origin, names):
    bad = []
    for name in names:
        if not hasattr(origin, name):
            bad.append("%s: 원본에 없음" % name)
            continue
        if not hasattr(local, name):
            bad.append("%s: 사본에 없음" % name)
            continue
        a, b = getattr(origin, name), getattr(local, name)
        if callable(a) and not isinstance(a, dict):
            continue
        if isinstance(a, dict):
            a = {k: spec_tuple(v) for k, v in a.items()}
            b = {k: spec_tuple(v) for k, v in b.items()}
        if str(a) != str(b):
            bad.append("%s: %r != %r" % (name, b, a))
    return bad


def main():
    parser = argparse.ArgumentParser(
        description="Check copied code and assets against their originals.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    failures = 0

    for package, entries in SOURCES.items():
        local = __import__(package)
        for origin_path, names in entries:
            if not origin_path.is_file():
                print("SKIP  %s (원본 없음: %s)" % (package, origin_path))
                continue
            origin = load(origin_path, "origin_%d" % id(origin_path))
            bad = compare(local, origin, names)
            failures += len(bad)
            if bad:
                print("DRIFT %s vs %s" % (package, origin_path.name))
                for line in bad:
                    print("        " + line)
            elif not args.quiet:
                print("OK    %s vs %s  (%d symbols)" % (package, origin_path.name, len(names)))

    for local_rel, origin_path in ASSETS.items():
        local_path = ROOT / local_rel
        if not origin_path.is_file():
            print("SKIP  %s (원본 없음)" % local_rel)
            continue
        a = hashlib.sha256(local_path.read_bytes()).hexdigest()
        b = hashlib.sha256(origin_path.read_bytes()).hexdigest()
        if a != b:
            failures += 1
            print("DRIFT %s\n        local  %s\n        origin %s" % (local_rel, a[:16], b[:16]))
        elif not args.quiet:
            print("OK    %s" % local_rel)

    print("\n%s (%d 불일치)" % ("모두 동기화됨" if failures == 0 else "동기화 필요", failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
