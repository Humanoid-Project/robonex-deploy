#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from robonex_common.policy import PolicyContract
from robonex_common.runtime import (
    OBSERVATION_TERM_SIZES,
    ActionPipeline,
    ObservationHistory,
    assemble_observation_frame,
)


def infer(session, input_name, output_name, observation):
    return np.asarray(
        session.run([output_name], {input_name: observation.reshape(1, -1)})[0][0],
        dtype=np.float32,
    )


def apply(pipeline, raw_action):
    action, target = pipeline.apply(raw_action)
    scaled = action * pipeline.scales + pipeline.offsets
    clipped = bool(np.any(action != raw_action) or np.any(target != scaled))
    return target, clipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reject an ONNX policy whose local position-command gain is too high."
    )
    parser.add_argument("--policy", type=Path, required=True, help="ONNX policy path")
    parser.add_argument("--max-gain", type=float, default=5.0, help="Maximum allowed target gain")
    parser.add_argument("--perturb-deg", type=float, default=1.0, help="Joint-position perturbation")
    args = parser.parse_args(argv)
    args.policy = args.policy.expanduser().resolve()
    if not args.policy.is_file():
        parser.error(f"Policy not found: {args.policy}")
    if not math.isfinite(args.max_gain) or args.max_gain <= 0.0:
        parser.error("--max-gain must be finite and positive")
    if not math.isfinite(args.perturb_deg) or args.perturb_deg <= 0.0:
        parser.error("--perturb-deg must be finite and positive")
    return args


def run(args):
    manifest_path = args.policy.with_name("policy_manifest.json")
    contract = PolicyContract.load(manifest_path)
    selected = contract.verify_policy(manifest_path)
    if selected.resolve() != args.policy:
        raise ValueError(f"policy_manifest.json selects {selected.name}, not {args.policy.name}")
    layout = ObservationHistory.from_contract(contract)
    if layout.term_sizes != OBSERVATION_TERM_SIZES:
        raise ValueError(f"unsupported observation layout: {contract.observation_terms}")

    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("policy must have exactly one input and one output")
    input_name = inputs[0].name
    output_name = outputs[0].name

    def observe(joint_pos):
        frame = assemble_observation_frame(
            joint_pos,
            np.zeros(contract.action_size),
            np.zeros(3),
            (0.0, 0.0, -1.0),
            np.zeros(3),
            np.zeros(2),
            np.zeros(contract.action_size),
            frame_size=layout.frame_size,
        )
        layout.reset()
        return layout.append(frame).observation()

    observation = observe(np.zeros(contract.action_size))
    pipeline = ActionPipeline(contract)
    baseline_raw = infer(session, input_name, output_name, observation)
    baseline_target, baseline_clipped = apply(pipeline, baseline_raw)
    offsets = np.asarray(contract.action_offsets, dtype=np.float32)
    baseline_deviation = math.degrees(float(np.max(np.abs(baseline_target - offsets))))

    perturb = math.radians(args.perturb_deg)
    rows = []
    any_clipped = baseline_clipped
    for input_index, input_joint in enumerate(contract.joint_order):
        best_gain = -1.0
        best_output = ""
        for sign in (-1.0, 1.0):
            joint_pos = np.zeros(contract.action_size)
            joint_pos[input_index] = sign * perturb
            sample = observe(joint_pos)
            raw_action = infer(session, input_name, output_name, sample)
            target, clipped = apply(pipeline, raw_action)
            any_clipped |= clipped
            delta = np.abs(target - baseline_target) / perturb
            output_index = int(np.argmax(delta))
            if float(delta[output_index]) > best_gain:
                best_gain = float(delta[output_index])
                best_output = contract.joint_order[output_index]
        rows.append((input_joint, best_gain, best_output))

    print(f"Policy: {args.policy}")
    print(f"Baseline max target deviation: {baseline_deviation:.3f} deg")
    print(f"Gain gate: {args.perturb_deg:g} deg perturb, limit {args.max_gain:g}x")
    print(f"  {'input joint':<28} {'max gain':>10}  output joint")
    for input_joint, gain, output_joint in rows:
        print(f"  {input_joint:<28} {gain:>9.3f}x  {output_joint}")

    worst = max(rows, key=lambda row: row[1])
    if any_clipped:
        print("FAIL: runner or target clipping occurred during the local gain probe.")
        return 1
    if worst[1] > args.max_gain:
        print(
            f"FAIL: {worst[0]} reached {worst[1]:.3f}x through {worst[2]}, "
            f"above {args.max_gain:g}x."
        )
        return 1
    print(f"PASS: worst local target gain is {worst[1]:.3f}x.")
    return 0


def main(argv=None):
    try:
        return run(parse_args(argv))
    except Exception as error:
        print(f"Policy gain check failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
