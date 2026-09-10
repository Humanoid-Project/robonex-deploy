import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    import onnxruntime as ort
except ImportError as error:
    raise SystemExit(f"Missing required package: {error}")

from robonex_common.joints import PASSIVE_CLOSED_LOOP_JOINTS
from robonex_common.policy import (
    PolicyContract,
    mujoco_bundle_sha256,
    python_source_sha256,
    sha256_file,
)
from robonex_common.runtime import ActionPipeline, assemble_observation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from robonex_common.paths import DESCRIPTION_REPO_NAMES, description_model, git_commit, resolve_repo

file_hash = sha256_file


def sibling_robot_hash(model_path):
    robot_path = model_path.parent / "robonex.xml"
    return file_hash(robot_path) if robot_path.is_file() else None


def required_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Required MuJoCo name not found: {name}")
    return object_id


def tensor_last_dim(value):
    if not value.shape:
        return None
    return value.shape[-1]


def quaternion_to_rpy(quaternion):
    w, x, y, z = quaternion
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


class PolicyAdapter:
    def __init__(self, model, policy_path, contract):
        self.model = model
        self.policy_path = policy_path
        self.contract = contract
        self.joint_names = contract.joint_order
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or tensor_last_dim(inputs[0]) != contract.observation_size:
            raise RuntimeError(f"Policy input size does not match the manifest: {[item.shape for item in inputs]}")
        if len(outputs) != 1 or tensor_last_dim(outputs[0]) != contract.action_size:
            raise RuntimeError(f"Policy output size does not match the manifest: {[item.shape for item in outputs]}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.joint_ids = np.array(
            [required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dof_addresses = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.default_positions = model.qpos0[self.qpos_addresses].copy()
        actuator_ids = []
        for name, joint_id in zip(self.joint_names, self.joint_ids):
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                raise RuntimeError(f"Expected one actuator for {name}: {matches.tolist()}")
            actuator_ids.append(int(matches[0]))
        self.actuator_ids = np.array(actuator_ids, dtype=np.int32)
        if model.nu != 12 or set(self.actuator_ids.tolist()) != set(range(model.nu)):
            raise RuntimeError(f"The 12 motor actuators do not match the policy joints: nu={model.nu}")
        passive_actuators = []
        for name in PASSIVE_CLOSED_LOOP_JOINTS:
            joint_id = required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            if matches.size:
                passive_actuators.append((name, matches.tolist()))
        if passive_actuators:
            raise RuntimeError(f"Passive closed-loop joints have actuators: {passive_actuators}")
        self.base_body_id = required_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.root_joint_id = required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if model.jnt_type[self.root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("The root joint is not a free joint")
        self.root_qpos_address = int(model.jnt_qposadr[self.root_joint_id])
        self.floor_geom_id = required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.left_foot_body_id = required_id(model, mujoco.mjtObj.mjOBJ_BODY, "l_foot")
        self.right_foot_body_id = required_id(model, mujoco.mjtObj.mjOBJ_BODY, "r_foot")
        self.pipeline = ActionPipeline(contract)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.object_velocity = np.zeros(6, dtype=np.float64)
        self.gravity_world = np.array((0.0, 0.0, -1.0), dtype=np.float64)

    def reset(self):
        self.last_action.fill(0.0)

    def observation(self, data):
        joint_positions = data.qpos[self.qpos_addresses] - self.default_positions
        joint_velocities = data.qvel[self.dof_addresses]
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_XBODY,
            self.base_body_id,
            self.object_velocity,
            1,
        )
        angular_velocity = self.object_velocity[:3]
        rotation_world_from_base = data.xmat[self.base_body_id].reshape(3, 3)
        projected_gravity = rotation_world_from_base.T @ self.gravity_world
        try:
            return assemble_observation(
                joint_positions,
                joint_velocities,
                angular_velocity,
                projected_gravity,
                self.last_action,
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error

    def apply(self, data, max_raw_action):
        observation = self.observation(data)
        output = self.session.run([self.output_name], {self.input_name: observation.reshape(1, 42)})[0]
        try:
            clipped, targets = self.pipeline.apply(output, max_raw_action=max_raw_action)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        data.ctrl[self.actuator_ids] = targets
        self.last_action[:] = clipped
        return observation, clipped.copy(), targets.copy()

    def mapping(self):
        rows = []
        for policy_index, name in enumerate(self.joint_names):
            actuator_id = int(self.actuator_ids[policy_index])
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            rows.append(
                {
                    "policy_index": policy_index,
                    "joint": name,
                    "qpos_address": int(self.qpos_addresses[policy_index]),
                    "dof_address": int(self.dof_addresses[policy_index]),
                    "actuator_index": actuator_id,
                    "actuator": actuator_name,
                    "scale": float(self.pipeline.scales[policy_index]),
                    "offset": float(self.pipeline.offsets[policy_index]),
                    "target_clip_rad": [
                        float(self.pipeline.target_low[policy_index]),
                        float(self.pipeline.target_high[policy_index]),
                    ],
                }
            )
        return rows


class Diagnostics:
    def __init__(self, model, adapter):
        self.model = model
        self.adapter = adapter
        self.physics_steps = 0
        self.policy_steps = 0
        self.root_z_min = math.inf
        self.root_z_max = -math.inf
        self.roll_abs_max = 0.0
        self.pitch_abs_max = 0.0
        self.angular_speed_max = 0.0
        self.raw_action_abs_max = 0.0
        self.target_abs_max = 0.0
        self.actuator_force_abs_max = 0.0
        self.force_saturation_count = 0
        self.force_sample_count = 0
        self.constraint_error_max = 0.0
        self.joint_limit_overrun_max = 0.0
        self.joint_limit_violation_steps = 0
        self.left_contact_steps = 0
        self.right_contact_steps = 0
        self.both_contact_steps = 0
        self.contact_normal_force_max = 0.0
        self.fall_time_s = None
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_targets = np.zeros(12, dtype=np.float32)

    def record_policy(self, action, targets):
        self.policy_steps += 1
        self.last_action[:] = action
        self.last_targets[:] = targets
        self.raw_action_abs_max = max(self.raw_action_abs_max, float(np.max(np.abs(action))))
        self.target_abs_max = max(self.target_abs_max, float(np.max(np.abs(targets))))

    def equality_error(self, data):
        if data.nefc == 0:
            return 0.0
        types = np.asarray(data.efc_type[: data.nefc])
        mask = types == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(np.asarray(data.efc_pos[: data.nefc])[mask])))

    def joint_limit_overrun(self, data):
        maximum = 0.0
        for joint_id, qpos_address in zip(self.adapter.joint_ids, self.adapter.qpos_addresses):
            if not self.model.jnt_limited[joint_id]:
                continue
            lower, upper = self.model.jnt_range[joint_id]
            position = float(data.qpos[qpos_address])
            maximum = max(maximum, lower - position, position - upper, 0.0)
        return maximum

    def contacts(self, data):
        left = False
        right = False
        normal_force_max = 0.0
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if contact.geom1 == self.adapter.floor_geom_id:
                other_geom = contact.geom2
            elif contact.geom2 == self.adapter.floor_geom_id:
                other_geom = contact.geom1
            else:
                continue
            body_id = int(self.model.geom_bodyid[other_geom])
            if body_id == self.adapter.left_foot_body_id:
                left = True
            if body_id == self.adapter.right_foot_body_id:
                right = True
            mujoco.mj_contactForce(self.model, data, contact_index, contact_force)
            normal_force_max = max(normal_force_max, abs(float(contact_force[0])))
        return left, right, normal_force_max

    def record_physics(self, data):
        self.physics_steps += 1
        root_z = float(data.qpos[self.adapter.root_qpos_address + 2])
        self.root_z_min = min(self.root_z_min, root_z)
        self.root_z_max = max(self.root_z_max, root_z)
        roll, pitch, _ = quaternion_to_rpy(data.xquat[self.adapter.base_body_id])
        self.roll_abs_max = max(self.roll_abs_max, abs(roll))
        self.pitch_abs_max = max(self.pitch_abs_max, abs(pitch))
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_XBODY,
            self.adapter.base_body_id,
            self.adapter.object_velocity,
            1,
        )
        self.angular_speed_max = max(
            self.angular_speed_max,
            float(np.linalg.norm(self.adapter.object_velocity[:3])),
        )
        actuator_force = np.abs(np.asarray(data.actuator_force))
        if actuator_force.size:
            self.actuator_force_abs_max = max(self.actuator_force_abs_max, float(np.max(actuator_force)))
            force_limits = np.max(np.abs(self.model.actuator_forcerange), axis=1)
            limited = np.asarray(self.model.actuator_forcelimited, dtype=bool)
            valid = limited & (force_limits > 0.0)
            self.force_saturation_count += int(np.count_nonzero(actuator_force[valid] >= 0.99 * force_limits[valid]))
            self.force_sample_count += int(np.count_nonzero(valid))
        constraint_error = self.equality_error(data)
        self.constraint_error_max = max(self.constraint_error_max, constraint_error)
        overrun = self.joint_limit_overrun(data)
        self.joint_limit_overrun_max = max(self.joint_limit_overrun_max, overrun)
        if overrun > 1.0e-6:
            self.joint_limit_violation_steps += 1
        left, right, normal_force_max = self.contacts(data)
        self.left_contact_steps += int(left)
        self.right_contact_steps += int(right)
        self.both_contact_steps += int(left and right)
        self.contact_normal_force_max = max(self.contact_normal_force_max, normal_force_max)
        return root_z, constraint_error, overrun

    def result(self, data, status, reason, wall_seconds, model_path, policy_path):
        denominator = max(1, self.physics_steps)
        force_denominator = max(1, self.force_sample_count)
        roll, pitch, yaw = quaternion_to_rpy(data.xquat[self.adapter.base_body_id])
        return {
            "status": status,
            "reason": reason,
            "simulation_time_s": float(data.time),
            "wall_time_s": wall_seconds,
            "physics_steps": self.physics_steps,
            "policy_steps": self.policy_steps,
            "spawn": "isaac",
            "model": str(model_path),
            "model_sha256": file_hash(model_path),
            "included_robot_sha256": sibling_robot_hash(model_path),
            "policy": str(policy_path),
            "policy_sha256": file_hash(policy_path),
            "mujoco_version": mujoco.__version__,
            "onnxruntime_version": ort.__version__,
            "model_total_mass_kg": float(mujoco.mj_getTotalmass(self.model)),
            "model_timestep_s": float(self.model.opt.timestep),
            "root_z_m": {
                "final": float(data.qpos[self.adapter.root_qpos_address + 2]),
                "min": self.root_z_min if math.isfinite(self.root_z_min) else None,
                "max": self.root_z_max if math.isfinite(self.root_z_max) else None,
            },
            "base_rpy_deg_final": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
            "roll_abs_max_deg": math.degrees(self.roll_abs_max),
            "pitch_abs_max_deg": math.degrees(self.pitch_abs_max),
            "angular_speed_max_rad_s": self.angular_speed_max,
            "raw_action_abs_max": self.raw_action_abs_max,
            "target_abs_max_rad": self.target_abs_max,
            "actuator_force_abs_max_nm": self.actuator_force_abs_max,
            "force_saturation_fraction": self.force_saturation_count / force_denominator,
            "constraint_error_abs_max": self.constraint_error_max,
            "joint_limit_overrun_max_rad": self.joint_limit_overrun_max,
            "joint_limit_violation_steps": self.joint_limit_violation_steps,
            "left_foot_contact_fraction": self.left_contact_steps / denominator,
            "right_foot_contact_fraction": self.right_contact_steps / denominator,
            "both_feet_contact_fraction": self.both_contact_steps / denominator,
            "contact_normal_force_abs_max_n": self.contact_normal_force_max,
            "fall_detected": self.fall_time_s is not None,
            "fall_time_s": self.fall_time_s,
            "last_raw_action": self.last_action.tolist(),
            "last_targets_rad_policy_order": self.last_targets.tolist(),
            "runner_action_clip": self.adapter.pipeline.runner_clip,
            "runner_clip_fraction": self.adapter.pipeline.runner_clip_count
            / max(1, self.adapter.pipeline.policy_call_count * 12),
            "target_clip_fraction": self.adapter.pipeline.target_clip_count
            / max(1, self.adapter.pipeline.policy_call_count * 12),
            "policy_mapping": self.adapter.mapping(),
        }


class HeadlessViewer:
    def is_running(self):
        return True

    def sync(self):
        return None


def reset_simulation(model, data, adapter):
    mujoco.mj_resetData(model, data)
    data.qpos[adapter.root_qpos_address + 2] = 1.0789
    adapter.reset()
    mujoco.mj_forward(model, data)


def state_is_finite(data):
    return all(
        np.isfinite(values).all()
        for values in (data.qpos, data.qvel, data.ctrl, data.xpos, data.actuator_force)
    )


def simulate(model, data, adapter, args, viewer_handle):
    reset_simulation(model, data, adapter)
    diagnostics = Diagnostics(model, adapter)
    policy_period = 1.0 / args.policy_hz
    next_policy_time = 0.0
    next_viewer_sync = 0.0
    wall_start = time.perf_counter()
    status = "timeout"
    reason = "duration_reached"
    while True:
        if not viewer_handle.is_running():
            status = "viewer_closed"
            reason = "viewer_closed"
            break
        if args.duration is not None and data.time >= args.duration:
            break
        if data.time + 1.0e-12 >= next_policy_time:
            try:
                _, action, targets = adapter.apply(data, args.max_raw_action)
            except RuntimeError as error:
                status = "policy_error"
                reason = str(error)
                break
            diagnostics.record_policy(action, targets)
            next_policy_time += policy_period
        mujoco.mj_step(model, data)
        if not state_is_finite(data):
            status = "nonfinite"
            reason = "simulation_state_is_not_finite"
            break
        root_z, constraint_error, joint_overrun = diagnostics.record_physics(data)
        if root_z < args.minimum_height:
            if diagnostics.fall_time_s is None:
                diagnostics.fall_time_s = float(data.time)
        if constraint_error > args.max_constraint_error:
            status = "constraint_divergence"
            reason = f"constraint_error_above_{args.max_constraint_error}"
            break
        if joint_overrun > args.max_joint_overrun:
            status = "joint_limit"
            reason = f"joint_limit_overrun_above_{args.max_joint_overrun}"
            break
        if float(np.max(np.abs(data.xpos))) > args.max_body_position:
            status = "position_divergence"
            reason = f"body_position_above_{args.max_body_position}"
            break
        if data.time + 1.0e-12 >= next_viewer_sync:
            viewer_handle.sync()
            next_viewer_sync += policy_period
        if not args.headless:
            sleep_seconds = wall_start + data.time - time.perf_counter()
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
    wall_seconds = time.perf_counter() - wall_start
    return diagnostics.result(
        data,
        status,
        reason,
        wall_seconds,
        args.model,
        args.policy,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--duration", type=float, help="Stop after this many simulated seconds")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer")
    args = parser.parse_args()
    if args.headless and args.duration is None:
        parser.error("--headless requires --duration")
    args.minimum_height = 0.6
    args.max_raw_action = 20.0
    args.max_constraint_error = 0.05
    args.max_joint_overrun = 0.05
    args.max_body_position = 100.0
    args.policy = args.policy.expanduser().resolve()
    if not args.policy.is_file():
        parser.error(f"Policy not found: {args.policy}")
    args.manifest = args.policy.with_name("policy_manifest.json")
    try:
        args.contract = PolicyContract.load(args.manifest)
        manifest_policy = args.contract.verify_policy(args.manifest)
        if manifest_policy.resolve() != args.policy:
            parser.error(
                f"policy_manifest.json selects {manifest_policy.name}, not {args.policy.name}"
            )
        description_root = resolve_repo(
            DESCRIPTION_REPO_NAMES,
            "ROBONEX_DESCRIPTION_ROOT",
            None,
        )
        actual_description_commit = git_commit(description_root)
        if actual_description_commit != args.contract.description_commit:
            parser.error(
                f"robonex-description commit mismatch: manifest={args.contract.description_commit}, "
                f"checkout={actual_description_commit}"
            )
        actual_description_sha256 = mujoco_bundle_sha256(
            description_root, args.contract.description_model
        )
        if actual_description_sha256 != args.contract.description_sha256:
            parser.error(
                f"robonex-description model bundle mismatch: manifest={args.contract.description_sha256}, "
                f"checkout={actual_description_sha256}"
            )
        common_root = resolve_repo("robonex-common", "ROBONEX_COMMON_ROOT")
        actual_common_commit = git_commit(common_root)
        if actual_common_commit != args.contract.common_commit:
            parser.error(
                f"robonex-common commit mismatch: manifest={args.contract.common_commit}, "
                f"checkout={actual_common_commit}"
            )
        actual_common_sha256 = python_source_sha256(common_root, ("src/robonex_common",))
        if actual_common_sha256 != args.contract.common_sha256:
            parser.error(
                f"robonex-common source mismatch: manifest={args.contract.common_sha256}, "
                f"checkout={actual_common_sha256}"
            )
        args.model = description_model(args.contract.description_model, description_root)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    args.policy_hz = args.contract.policy_hz
    if not args.model.is_file():
        parser.error(f"MuJoCo model not found: {args.model}")
    if args.output is not None:
        args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    adapter = PolicyAdapter(model, args.policy, args.contract)
    if args.headless:
        result = simulate(model, data, adapter, args, HeadlessViewer())
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer_handle:
            result = simulate(model, data, adapter, args, viewer_handle)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] in ("viewer_closed", "timeout") else 1


if __name__ == "__main__":
    sys.exit(main())
