import csv
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/sim_to_sim/isaac_to_mujoco.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mj():
    return load(SOURCE, "mujoco_strafe_metrics")


def test_blocks_settle_and_command_boundaries(mj):
    metrics = mj.StrafeMetrics(0.2)
    metrics.command((0, 0.2, 0), 0)
    for i in range(1, 101):
        vy = 0.5 if i % 2 else -0.1
        metrics.sample(i * 0.02, (0.1, vy, 0, -0.01 if i % 2 else 0.1, i % 2))
    metrics.command((0.1, 0.2, 0), 2.0)
    metrics.sample(2.3, (0, 0.0, 0, 0.1, 0))
    metrics.command((0.1, 0.2, 0.1), 2.3)
    metrics.finish(2.3, "policy_error")
    first, second, third = metrics.segments
    assert first["completed_duration_s"] == 2
    assert first["measured_duration_s"] == pytest.approx(1.8)
    assert first["got_vx"] == pytest.approx(0.1)
    assert first["got_vy"] == pytest.approx(0.2)
    assert first["err_vy"] == pytest.approx(0.3)
    assert first["block_count"] == 2
    assert first["err_vy_block_mae"] == pytest.approx(0, abs=1e-12)
    assert first["err_vy_block_p95"] == pytest.approx(0, abs=1e-12)
    assert first["discarded_block_duration_s"] == pytest.approx(0.2)
    assert first["min_foot_sep_y"] == -0.01
    assert first["crossing_fraction"] == pytest.approx(0.5)
    assert first["foot_foot_contact_count"] == 45
    assert first["stop_reason"] == "command_change"
    assert second["block_count"] == 0
    assert second["err_vy_block_mae"] is None
    assert third["got_vy"] is None
    assert third["stop_reason"] == "policy_error"


def test_block_absolute_mean_and_p95(mj):
    metrics = mj.StrafeMetrics(0)
    metrics.command((0, 0, 0), 0)
    for i in range(1, 81):
        metrics.sample(i * 0.02, (0, 0.1 if i <= 40 else -0.3, 0, 1, 0))
    metrics.finish(1.6, "duration_reached")
    result = metrics.segments[0]
    assert result["err_vy_block_mae"] == pytest.approx(0.2)
    assert result["err_vy_block_p95"] == pytest.approx(0.29)


def test_yaw_frame_and_contact_filter(mj, monkeypatch):
    def velocity(model, data, kind, body, output, local):
        assert local == 0
        output[:] = [0, 0, 0, 1, 2, 3]
    monkeypatch.setattr(mj.mujoco, "mj_objectVelocity", velocity)
    yaw, roll = math.pi / 2, math.pi / 3
    cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
    data = SimpleNamespace(xquat=np.array([[cr*cy, sr*cy, sr*sy, cr*sy]]),
                           xpos=np.array([[0, 0, 0], [-0.2, 0, 0], [0, 0, 0]]),
                           contact=[SimpleNamespace(geom1=0, geom2=1, dist=-0.01)], ncon=1)
    model = SimpleNamespace(geom_bodyid=[1, 2, 0])
    adapter = SimpleNamespace(base_body_id=0, left_foot_body_id=1, right_foot_body_id=2)
    assert mj.strafe_sample(model, data, adapter) == pytest.approx((2, -1, yaw, 0.2, 1))
    data.contact[0].dist = 0.01
    assert mj.strafe_sample(model, data, adapter)[-1] == 0
    data.contact[0] = SimpleNamespace(geom1=0, geom2=2, dist=-0.01)
    assert mj.strafe_sample(model, data, adapter)[-1] == 0


def rollout(module, tmp_path, name, duration=3.6, **overrides):
    deploy = Path(os.environ.get("ROBONEX_DEPLOY_TEST_ROOT", "/home/polygon/humanoid_project/robonex-deploy"))
    policy = deploy / "policies/2026-10-08_W113s43E_v2_2_edu_base_iter999/policy.onnx"
    scene = Path(os.environ.get("ROBONEX_DESCRIPTION_ROOT", "/home/polygon/humanoid_project/robonex-description")) / "ver2-2/mujoco/robot/edu/scene.xml"
    contract = module.PolicyContract.load(policy.with_name("policy_manifest.json"))
    model = module.mujoco.MjModel.from_xml_path(str(scene))
    data = module.mujoco.MjData(model)
    adapter = module.PolicyAdapter(model, policy, contract)
    args = SimpleNamespace(policy=policy, model=scene, policy_hz=contract.policy_hz,
                           duration=duration, headless=True, scenario="0:0,0,0;1.2:0,0.1,0;2.4:0,-0.1,0",
                           trace=tmp_path / (name + ".csv"), minimum_height=0.6,
                           max_raw_action=20.0, max_constraint_error=0.05,
                           max_joint_overrun=0.05, max_body_position=100.0,
                           slew_limit=True, metrics_settle_s=0.2, joint_friction=[0.14, 0.47])
    for key, value in overrides.items():
        setattr(args, key, value)
    result = module.simulate(model, data, adapter, args, module.HeadlessViewer())
    with args.trace.open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    return result, fields, rows


def test_cpu_rollout_segments_and_trace(mj, tmp_path):
    result, fields, rows = rollout(mj, tmp_path, "new")
    assert result["status"] == "timeout", result
    assert len(result["command_segments"]) == 3
    assert fields[-4:] == ["root_vy_yaw", "yaw", "foot_sep_y", "foot_foot_contact"]
    assert rows and all(math.isfinite(float(row[key])) for row in rows for key in fields[-4:])
    for segment, vy in zip(result["command_segments"], (0, 0.1, -0.1)):
        assert segment["command"] == pytest.approx([0, vy, 0])
        assert segment["completed_duration_s"] == pytest.approx(1.2, abs=0.003)
        assert segment["block_count"] == 1
        assert segment["measured_duration_s"] == pytest.approx(1, abs=0.003)
        assert segment["err_vy"] >= abs(segment["got_vy"] - vy) - 1e-10
        assert 0 <= segment["crossing_fraction"] <= 1
    assert result["command_segments"][-1]["stop_reason"] == "duration_reached"


def test_early_stop(mj, tmp_path):
    result, _, _ = rollout(mj, tmp_path, "stop", max_constraint_error=-1)
    assert result["status"] == "constraint_divergence"
    segment = result["command_segments"][0]
    assert segment["completed_duration_s"] > 0
    assert segment["completed_duration_s"] < 0.2
    assert segment["got_vy"] is None
    assert segment["stop_reason"] == result["reason"]


def test_append_only_compatibility(mj, tmp_path):
    baseline_path = os.environ.get("ROBONEX_METRICS_BASELINE")
    if not baseline_path:
        pytest.skip("Set ROBONEX_METRICS_BASELINE to an unmodified HEAD script")
    baseline = load(Path(baseline_path), "mujoco_strafe_baseline")
    old, old_fields, old_rows = rollout(baseline, tmp_path, "old", duration=1)
    new, fields, rows = rollout(mj, tmp_path, "new", duration=1)
    assert fields[:len(old_fields)] == old_fields
    assert [{key: row[key] for key in old_fields} for row in rows] == old_rows
    assert {key: new[key] for key in old if key != "wall_time_s"} == {
        key: value for key, value in old.items() if key != "wall_time_s"}
