import os
import signal
from pathlib import Path

import numpy as np
import pytest

from test_policy_to_real_recording import load_meta, run_full_deploy
from test_policy_to_real_timing import POLICY_TO_REAL, FakeSession, load_module, make_contract, read_csv


@pytest.fixture
def module():
    return load_module("policy_to_real_review_fixes", POLICY_TO_REAL)


def recordings(path):
    return (
        read_csv(path),
        path.with_name(path.stem + "_arrays.npz"),
        read_csv(path.with_name(path.stem + "_phases.csv")),
        load_meta(path),
    )


def test_second_ctrl_c_during_shutdown_keeps_every_recording(module, monkeypatch, tmp_path, capsys):
    before = signal.getsignal(signal.SIGINT)
    sent = []

    def interrupt(where):
        sent.append(where)
        os.kill(os.getpid(), signal.SIGINT)

    original_stats = module.imu_driver_stats
    stats_calls = []

    def stats(imu):
        stats_calls.append(1)
        if len(stats_calls) >= 2:
            interrupt("imu_driver_stats")
        return original_stats(imu)

    original_close = module.TelemetryRecorder.close

    def close(self):
        interrupt("telemetry.close")
        return original_close(self)

    original_finalize = module.TelemetryRecorder.finalize_sidecar

    def finalize(self, end):
        interrupt("finalize_sidecar")
        return original_finalize(self, end)

    monkeypatch.setattr(module, "imu_driver_stats", stats)
    monkeypatch.setattr(module.TelemetryRecorder, "close", close)
    monkeypatch.setattr(module.TelemetryRecorder, "finalize_sidecar", finalize)
    escaped = None
    run = None
    try:
        run = run_full_deploy(module, monkeypatch, tmp_path)
    except KeyboardInterrupt as error:
        escaped = error
    capsys.readouterr()
    assert escaped is None
    assert sent == ["imu_driver_stats", "telemetry.close", "finalize_sidecar"]
    assert run.error is None
    assert run.events.index("brake") < run.events.index("close")
    assert all(motor.stopped for motor in run.motors.values())
    rows, arrays, phases, meta = recordings(run.path)
    assert len(rows) >= 90
    assert arrays.exists() and np.load(arrays)["obs"].shape == (len(rows), 235)
    assert sum(row["phase"] == "brake" for row in phases) >= 10
    assert meta["end"]["exit"] == "completed"
    assert meta["end"]["rows_recorded"] == len(rows)
    assert signal.getsignal(signal.SIGINT) is before


def test_raw_action_limit_stop_records_observation_and_raw_action(module, monkeypatch, tmp_path, capsys):
    original = FakeSession.run
    calls = []

    def run_session(self, outputs, feed):
        result = original(self, outputs, feed)
        if len(self.observations) == 31 and not calls:
            calls.append(1)
            result = [np.full_like(result[0], 25.0)]
        return result

    monkeypatch.setattr(FakeSession, "run", run_session)
    run = run_full_deploy(module, monkeypatch, tmp_path)
    capsys.readouterr()
    assert calls == [1]
    assert isinstance(run.error, RuntimeError)
    assert str(run.error) == "Safety stop: Raw action limit exceeded: 25.000000 > 20.000000"
    rows, arrays_path, _, meta = recordings(run.path)
    assert len(rows) == 31
    assert all(row["stop_reason"] == "" for row in rows[:30])
    assert rows[30]["stop_reason"] == "Raw action limit exceeded: 25.000000 > 20.000000"
    first = run.contract.joint_order[0].replace("_joint", "")
    assert float(rows[30][f"{first}.raw_action"]) == 25.0
    arrays = np.load(arrays_path)
    assert arrays["obs"].shape == (31, 235)
    assert arrays["obs"].tobytes() == np.stack(run.observations).astype(np.float32).tobytes()
    assert (arrays["raw_action"][30] == 25.0).all()
    assert np.isnan(arrays["policy_action"][30]).all() and np.isnan(arrays["targets"][30]).all()
    assert "Raw action limit exceeded" in meta["end"]["exit"]


def test_send_error_in_policy_loop_records_one_stop_row(module, monkeypatch, tmp_path, capsys):
    original = module.TargetCommander.send
    policy_sends = []

    def send(self, targets, dt, max_speed, max_accel, use_velocity_target, reproject=None, align=True):
        if align is False:
            policy_sends.append(1)
            if len(policy_sends) == 21:
                raise OSError("synthetic bus failure")
        return original(self, targets, dt, max_speed, max_accel, use_velocity_target,
                        reproject=reproject, align=align)

    monkeypatch.setattr(module.TargetCommander, "send", send)
    with pytest.raises(OSError, match="^synthetic bus failure$"):
        run_full_deploy(module, monkeypatch, tmp_path)
    capsys.readouterr()
    rows, arrays_path, phases, meta = recordings(tmp_path / "live.csv")
    assert len(rows) == 21
    assert [row["stop_reason"] for row in rows].count("") == 20
    assert rows[20]["stop_reason"] == "OSError: synthetic bus failure"
    arrays = np.load(arrays_path)
    assert arrays["obs"].shape == (21, 235)
    observations = FakeSession.instances[-1].observations
    assert arrays["obs"].tobytes() == np.stack(observations).astype(np.float32).tobytes()
    assert np.isfinite(arrays["raw_action"][20]).all()
    assert np.isfinite(arrays["policy_action"][20]).all() and np.isfinite(arrays["targets"][20]).all()
    assert meta["end"]["exit"] == "OSError: synthetic bus failure"
    assert sum(row["phase"] == "brake" for row in phases) >= 10


def test_safety_stop_keeps_a_single_stop_row(module, monkeypatch, tmp_path, capsys):
    original = module.PolicyRunner.joint_state
    calls = []

    def joint_state(self, snapshot):
        calls.append(1)
        positions, velocities, missing = original(self, snapshot)
        if len(calls) == 26:
            missing = [self.motor_ids[0]]
        return positions, velocities, missing

    monkeypatch.setattr(module.PolicyRunner, "joint_state", joint_state)
    run = run_full_deploy(module, monkeypatch, tmp_path)
    capsys.readouterr()
    rows, arrays_path, _, _ = recordings(run.path)
    stops = [row["stop_reason"] for row in rows if row["stop_reason"]]
    assert len(stops) == 1 and stops[0].startswith("no feedback for motor IDs [")
    assert str(run.error) == "Safety stop: " + stops[0]
    assert len(rows) == 26 and rows[-1]["stop_reason"] == stops[0]
    assert np.load(arrays_path)["obs"].shape == (25, 235)


def test_warm_up_runs_once_and_leaves_runner_state_alone(module):
    contract = make_contract()
    FakeSession.instances.clear()
    runner = module.PolicyRunner(Path("policy.onnx"), contract)
    session = FakeSession.instances[-1]
    assert len(session.warmups) == 1 and session.observations == []
    assert session.warmups[0].shape == (contract.observation_size,) and not session.warmups[0].any()
    assert runner.gait_step == 0 and runner.inference_ms == 0.0
    assert not runner.last_action.any()
    assert runner.pipeline.policy_call_count == 0
    assert runner.pipeline.runner_clip_count == 0 and runner.pipeline.target_clip_count == 0
    fresh = module.ObservationHistory.from_contract(contract)
    assert vars(runner.history).keys() == vars(fresh).keys()
    for key, value in vars(fresh).items():
        other = vars(runner.history)[key]
        if isinstance(value, np.ndarray):
            assert value.tobytes() == other.tobytes()
        else:
            assert repr(value) == repr(other)


def test_slew_lag_threshold_counts_only_lags_over_half_degree(module):
    class Motor:
        def control(self, **kwargs):
            pass

    commander = module.TargetCommander({1: Motor(), 2: Motor()}, {1: 0.0, 2: 0.0}, module.SETTINGS)
    targets = {1: np.radians(0.3), 2: np.radians(3.0)}
    commander.send(targets, 0.02, 1.0, 1.0e12, False, align=False)
    assert commander.slew_limited_by_motor == {1: 1, 2: 1}
    assert commander.slew_lag_over_by_motor == {1: 0, 2: 1}
    assert commander.slew_lag_over_fraction_by_motor() == {1: 0.0, 2: 1.0}
    commander.reset_stats()
    assert commander.slew_lag_over_by_motor == {1: 0, 2: 0}
    assert commander.slew_lag_over_fraction_by_motor() == {1: 0.0, 2: 0.0}


def test_full_deploy_summary_has_lag_fraction(module, monkeypatch, tmp_path, capsys):
    run = run_full_deploy(module, monkeypatch, tmp_path)
    capsys.readouterr()
    slew = load_meta(run.path)["end"]["slew"]
    assert "limited_fraction_by_motor" in slew
    fractions = slew["lag_over_0p5deg_fraction_by_motor"]
    assert sorted(fractions) == sorted(slew["limited_fraction_by_motor"])
    assert all(0.0 <= value <= 1.0 for value in fractions.values())
    assert all(fractions[key] <= slew["limited_fraction_by_motor"][key] for key in fractions)


def test_zero_head_help_names_the_neck_motor_id(module, capsys):
    from robonex_common.joints import ALL_MOTORS

    neck = next(spec.motor_id for spec in ALL_MOTORS if spec.model_name == "neck_pitch_joint")
    assert neck == 26
    with pytest.raises(SystemExit):
        module.parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert f"neck motor (ID {neck})" in text and "ID 13" not in text


def test_settings_have_no_unused_scalar_gains(module):
    from dataclasses import fields

    names = {field.name for field in fields(module.Settings)}
    assert "kp" not in names and "kd" not in names
    assert "kp" not in module.asdict(module.SETTINGS)
