import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    import onnxruntime as ort
except ImportError as error:
    raise SystemExit(f"필수 패키지를 불러오지 못했습니다: {error}")


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "assets" / "mujoco" / "scene.xml"
DEFAULT_POLICY = None

POLICY_JOINTS = (
    "l_hip_yaw_joint",
    "r_hip_yaw_joint",
    "l_hip_pitch_joint",
    "r_hip_pitch_joint",
    "l_hip_roll_joint",
    "r_hip_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_lower_joint",
    "l_ankle_upper_joint",
    "r_ankle_lower_joint",
    "r_ankle_upper_joint",
)

PASSIVE_JOINTS = (
    "l_knee_joint",
    "r_knee_joint",
    "l_knee_coupler_joint_a",
    "r_knee_coupler_joint_a",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
)

ACTION_OFFSETS = (
    0.0,
    0.0,
    0.0,
    0.0,
    -0.479966,
    0.479966,
    -0.3926995,
    0.3926995,
    0.0872665,
    -0.0872665,
    -0.0872665,
    0.0872665,
)

ACTION_SCALES = (
    0.688132,
    0.688132,
    0.862665,
    0.862665,
    0.557232,
    0.557232,
    0.4699655,
    0.4699655,
    0.5135985,
    0.5135985,
    0.5135985,
    0.5135985,
)

# PPORunnerCfg.clip_actions
RUNNER_ACTION_CLIP = 3.0

# JointPositionActionCfg.clip, in POLICY_JOINTS order
TARGET_CLIPS = (
    (-0.688132, 0.688132),
    (-0.688132, 0.688132),
    (-0.862665, 0.862665),
    (-0.862665, 0.862665),
    (-1.037198, 0.077266),
    (-0.077266, 1.037198),
    (-0.862665, 0.077266),
    (-0.077266, 0.862665),
    (-0.426332, 0.600865),
    (-0.600865, 0.426332),
    (-0.600865, 0.426332),
    (-0.426332, 0.600865),
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sibling_robot_hash(model_path):
    robot_path = model_path.parent / "robonex.xml"
    return file_hash(robot_path) if robot_path.is_file() else None


def required_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo 모델에 필요한 이름이 없습니다: {name}")
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
    def __init__(self, model, policy_path):
        self.model = model
        self.policy_path = policy_path
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or tensor_last_dim(inputs[0]) != 42:
            raise RuntimeError(f"정책 입력이 [1, 42]가 아닙니다: {[item.shape for item in inputs]}")
        if len(outputs) != 1 or tensor_last_dim(outputs[0]) != 12:
            raise RuntimeError(f"정책 출력이 [1, 12]가 아닙니다: {[item.shape for item in outputs]}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.joint_ids = np.array(
            [required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in POLICY_JOINTS],
            dtype=np.int32,
        )
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dof_addresses = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.default_positions = model.qpos0[self.qpos_addresses].copy()
        actuator_ids = []
        for name, joint_id in zip(POLICY_JOINTS, self.joint_ids):
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                raise RuntimeError(f"{name}의 actuator 수가 1이 아닙니다: {matches.tolist()}")
            actuator_ids.append(int(matches[0]))
        self.actuator_ids = np.array(actuator_ids, dtype=np.int32)
        if model.nu != 12 or set(self.actuator_ids.tolist()) != set(range(model.nu)):
            raise RuntimeError(f"12개 motor actuator가 정책 관절과 정확히 일치하지 않습니다: nu={model.nu}")
        passive_actuators = []
        for name in PASSIVE_JOINTS:
            joint_id = required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            if matches.size:
                passive_actuators.append((name, matches.tolist()))
        if passive_actuators:
            raise RuntimeError(f"passive closed-loop joint에 actuator가 연결되어 있습니다: {passive_actuators}")
        self.base_body_id = required_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.root_joint_id = required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if model.jnt_type[self.root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("root joint가 free joint가 아닙니다")
        self.root_qpos_address = int(model.jnt_qposadr[self.root_joint_id])
        self.floor_geom_id = required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.left_foot_body_id = required_id(model, mujoco.mjtObj.mjOBJ_BODY, "l_foot")
        self.right_foot_body_id = required_id(model, mujoco.mjtObj.mjOBJ_BODY, "r_foot")
        self.scales = np.asarray(ACTION_SCALES, dtype=np.float32)
        self.offsets = np.asarray(ACTION_OFFSETS, dtype=np.float32)
        self.runner_clip = float(RUNNER_ACTION_CLIP)
        self.target_low = np.asarray([pair[0] for pair in TARGET_CLIPS], dtype=np.float32)
        self.target_high = np.asarray([pair[1] for pair in TARGET_CLIPS], dtype=np.float32)
        self.runner_clip_count = 0
        self.target_clip_count = 0
        self.policy_call_count = 0
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
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            self.object_velocity,
            1,
        )
        angular_velocity = self.object_velocity[:3]
        rotation_world_from_base = data.xmat[self.base_body_id].reshape(3, 3)
        projected_gravity = rotation_world_from_base.T @ self.gravity_world
        observation = np.concatenate(
            (
                joint_positions,
                joint_velocities,
                angular_velocity,
                projected_gravity,
                self.last_action,
            )
        ).astype(np.float32)
        if observation.shape != (42,) or not np.isfinite(observation).all():
            raise RuntimeError("42차원 observation이 유한하지 않습니다")
        return observation

    def apply(self, data, max_raw_action):
        observation = self.observation(data)
        output = self.session.run([self.output_name], {self.input_name: observation.reshape(1, 42)})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError("12차원 policy action이 유한하지 않습니다")
        raw_max = float(np.max(np.abs(action)))
        if raw_max > max_raw_action:
            raise RuntimeError(f"raw action 한계를 초과했습니다: {raw_max:.6f} > {max_raw_action:.6f}")
        clipped = np.clip(action, -self.runner_clip, self.runner_clip)
        scaled = clipped * self.scales + self.offsets
        targets = np.clip(scaled, self.target_low, self.target_high)
        self.policy_call_count += 1
        self.runner_clip_count += int(np.count_nonzero(clipped != action))
        self.target_clip_count += int(np.count_nonzero(targets != scaled))
        data.ctrl[self.actuator_ids] = targets
        self.last_action[:] = clipped
        return observation, clipped.copy(), targets.copy()

    def mapping(self):
        rows = []
        for policy_index, name in enumerate(POLICY_JOINTS):
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
                    "scale": float(self.scales[policy_index]),
                    "offset": float(self.offsets[policy_index]),
                    "target_clip_rad": [
                        float(self.target_low[policy_index]),
                        float(self.target_high[policy_index]),
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
            mujoco.mjtObj.mjOBJ_BODY,
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

    def result(self, data, status, reason, wall_seconds, model_path, policy_path, spawn):
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
            "spawn": spawn,
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
            "runner_action_clip": self.adapter.runner_clip,
            "runner_clip_fraction": self.adapter.runner_clip_count
            / max(1, self.adapter.policy_call_count * 12),
            "target_clip_fraction": self.adapter.target_clip_count
            / max(1, self.adapter.policy_call_count * 12),
            "policy_mapping": self.adapter.mapping(),
        }


def reset_simulation(model, data, adapter, spawn):
    mujoco.mj_resetData(model, data)
    if spawn == "isaac":
        data.qpos[adapter.root_qpos_address + 2] = 1.0789
    adapter.reset()
    mujoco.mj_forward(model, data)


def state_is_finite(data):
    return all(
        np.isfinite(values).all()
        for values in (data.qpos, data.qvel, data.ctrl, data.xpos, data.actuator_force)
    )


def simulate(model, data, adapter, args, viewer_handle=None):
    reset_simulation(model, data, adapter, args.spawn)
    diagnostics = Diagnostics(model, adapter)
    policy_period = 1.0 / args.policy_hz
    next_policy_time = 0.0
    next_viewer_sync = 0.0
    wall_start = time.perf_counter()
    status = "timeout"
    reason = "duration_reached"
    while data.time < args.duration:
        if viewer_handle is not None and not viewer_handle.is_running():
            status = "viewer_closed"
            reason = "viewer_closed"
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
            if viewer_handle is None or args.stop_on_fall:
                status = "fall"
                reason = f"root_z_below_{args.minimum_height}"
                break
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
        if viewer_handle is not None and data.time + 1.0e-12 >= next_viewer_sync:
            viewer_handle.sync()
            next_viewer_sync += policy_period
        if args.real_time or viewer_handle is not None:
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
        args.spawn,
    )


def check_result(model, data, adapter, args):
    reset_simulation(model, data, adapter, args.spawn)
    observation, action, targets = adapter.apply(data, args.max_raw_action)
    return {
        "status": "check_ok",
        "model": str(args.model),
        "model_sha256": file_hash(args.model),
        "included_robot_sha256": sibling_robot_hash(args.model),
        "policy": str(args.policy),
        "policy_sha256": file_hash(args.policy),
        "mujoco_version": mujoco.__version__,
        "onnxruntime_version": ort.__version__,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "neq": model.neq,
        "total_mass_kg": float(mujoco.mj_getTotalmass(model)),
        "timestep_s": float(model.opt.timestep),
        "policy_hz": args.policy_hz,
        "runner_action_clip": adapter.runner_clip,
        "spawn": args.spawn,
        "observation_shape": list(observation.shape),
        "action_shape": list(action.shape),
        "initial_raw_action": action.tolist(),
        "initial_targets_rad_policy_order": targets.tolist(),
        "policy_mapping": adapter.mapping(),
        "passive_joint_actuators": [],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY,
                        required=DEFAULT_POLICY is None)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--spawn", choices=("mujoco", "isaac"), default="mujoco")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-height", type=float, default=0.6)
    parser.add_argument("--max-raw-action", type=float, default=20.0)
    parser.add_argument("--max-constraint-error", type=float, default=0.05)
    parser.add_argument("--max-joint-overrun", type=float, default=0.05)
    parser.add_argument("--max-body-position", type=float, default=100.0)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.policy = args.policy.resolve()
    if not args.model.is_file():
        parser.error(f"MuJoCo 모델 파일이 없습니다: {args.model}")
    if not args.policy.is_file():
        parser.error(f"정책 파일이 없습니다: {args.policy}")
    if args.duration is None:
        args.duration = math.inf if args.viewer else 15.0
    elif args.duration == 0.0:
        args.duration = math.inf
    elif args.duration < 0.0:
        parser.error("--duration은 0 이상이어야 합니다")
    if args.policy_hz <= 0.0:
        parser.error("--policy-hz는 양수여야 합니다")
    if args.output is not None:
        args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    adapter = PolicyAdapter(model, args.policy)
    if args.check_only:
        result = check_result(model, data, adapter, args)
    elif args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer_handle:
            result = simulate(model, data, adapter, args, viewer_handle)
    else:
        result = simulate(model, data, adapter, args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] in {"check_ok", "timeout", "viewer_closed"} else 1


if __name__ == "__main__":
    sys.exit(main())
