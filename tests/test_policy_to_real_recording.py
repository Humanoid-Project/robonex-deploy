import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_policy_to_real_timing import (
    FakeClock,
    FakeImuDriver,
    FakeJoints,
    FakeMotor,
    FakeSession,
    make_contract,
    modules,
    read_csv,
    run_loop,
)


class BrakeMotor(FakeMotor):
    def __init__(self, motor_id, position, log, mode_running):
        super().__init__(motor_id, position, log, mode_running)
        self.spec = SimpleNamespace(kd_max=5.0)
        self.stopped = False
        self.parameters = {}

    def stop(self):
        self.stopped = True

    def read_parameter(self, index, fmt="<f", timeout=0.2):
        value = self.parameters.get(index)
        if isinstance(value, Exception):
            raise value
        return value


class FakeHub:
    def __init__(self, joints):
        self.joints = joints
        self.last_pump_frames = 0
        self.pumps = 0

    def pump(self):
        self.joints.poll()
        self.pumps += 1
        self.last_pump_frames = len(self.joints.motors)


class RichImuDriver(FakeImuDriver):
    def latest(self):
        sample = super().latest()
        sample.linear_acceleration = SimpleNamespace(x=0.1, y=-0.2, z=9.8)
        sample.orientation = SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)
        sample.imu_temperature = 31.5
        return sample


def run_full_deploy(module, monkeypatch, tmp_path, fail_in_policy=False):
    import safety
    from robonex_common.joints import JOINT_BY_MODEL_NAME

    contract = make_contract()
    clock = FakeClock()
    monkeypatch.setattr(module, "time", clock)
    FakeSession.instances.clear()
    events = []
    control_log = []
    motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
    offsets = {JOINT_BY_MODEL_NAME[name].motor_id: float(contract.action_offsets[i])
               for i, name in enumerate(contract.joint_order)}
    starts = {mid: offsets[mid] + 0.05 for mid in motor_ids}
    motors = {mid: BrakeMotor(mid, starts[mid], control_log, safety.MODE_RUNNING) for mid in motor_ids}
    motors[motor_ids[0]].parameters = {0x700B: 17.0, 0x7018: 23.0, 0x701C: 48.2, 0x7005: 0, 0x7029: 1}
    motors[motor_ids[1]].parameters = {0x700B: OSError("bus")}
    joints = FakeJoints(motors, clock)
    for motor in motors.values():
        motor.last_feedback_time = clock.now
    hubs = {"can0": FakeHub(joints)}

    def start(self, calibrate):
        self.driver = RichImuDriver(clock)
        self.status = "ready"
        self.bias_raw = SimpleNamespace(x=0.001, y=0.002, z=0.003)
        return True

    def enable(motors_, hubs_, kp, kd, limits, enabled_out=None):
        events.append("enable")
        enabled_out.extend(sorted(motors_))
        return {mid: motors_[mid].last_position for mid in motors_}, enabled_out

    original_brake = module.brake_and_stop

    def brake(*args, **kwargs):
        events.append("brake")
        return original_brake(*args, **kwargs)

    original_close = module.TelemetryRecorder.close

    def close(self):
        events.append("close")
        return original_close(self)

    monkeypatch.setattr(module, "verify_common_source", lambda contract: None)
    monkeypatch.setattr(module, "require_robot_model", lambda name: None)
    monkeypatch.setattr(module, "open_hardware", lambda ids, interface, host_id: ({}, motors, hubs))
    monkeypatch.setattr(module, "stop_idle_motors", lambda buses, ids, host_id: [])
    monkeypatch.setattr(module, "inspect_zero_positions",
                        lambda motors_, tolerance, limits: ({mid: starts[mid] for mid in motors_}, []))
    monkeypatch.setattr(module, "roll_pairs_for", lambda profile, ids: [])
    monkeypatch.setattr(module, "confirm", lambda prompt: events.append("confirm"))
    monkeypatch.setattr(module, "enable_with_runtime_feedback", enable)
    monkeypatch.setattr(module, "brake_and_stop", brake)
    monkeypatch.setattr(module.TelemetryRecorder, "close", close)
    monkeypatch.setattr(module.ImuSource, "start", start)
    monkeypatch.setattr(module.ImuSource, "stop", lambda self: events.append("imu_stop"))
    if fail_in_policy:
        original_step = module.PolicyRunner.step
        calls = []

        def failing_step(self, observation, commit):
            calls.append(1)
            if len(calls) > 40:
                raise ValueError("synthetic pipeline failure")
            return original_step(self, observation, commit)

        monkeypatch.setattr(module.PolicyRunner, "step", failing_step)
    path = tmp_path / "live.csv"
    args = SimpleNamespace(vx=0.0, vy=0.0, wz=0.0, scenario=None, scenario_text="", keyboard=False,
                           duration=2.0, telemetry=path)
    error = None
    try:
        module.run_deploy(Path("policy.onnx"), contract, args)
    except RuntimeError as caught:
        error = caught
    return SimpleNamespace(path=path, events=events, motors=motors, hubs=hubs, error=error,
                           observations=FakeSession.instances[-1].observations, contract=contract)


def load_meta(path):
    return json.loads(path.with_name(path.stem + "_meta.json").read_text())


def test_full_deploy_records_every_phase_and_closes_after_brake(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    run = run_full_deploy(new, monkeypatch, tmp_path)
    capsys.readouterr()
    assert run.error is None
    assert run.events.index("confirm") < run.events.index("enable")
    assert run.events.index("brake") < run.events.index("close")
    assert all(motor.stopped for motor in run.motors.values())

    rows = read_csv(run.path)
    assert len(rows) == len(run.observations) >= 90
    assert all(row["rx_frames.can0"] == "12" for row in rows)
    assert all(row["acc_z"] == "9.8" and row["imu_temp_c"] == "31.5" for row in rows)
    assert all(row["late_ms"] != "" and row["poll_ms"] != "" for row in rows)
    assert rows[0]["prev_work_ms"] == "" and all(row["prev_work_ms"] != "" for row in rows[1:])

    arrays = np.load(run.path.with_name(run.path.stem + "_arrays.npz"))
    assert arrays["obs"].shape == (len(rows), 235)
    assert arrays["obs"].tobytes() == np.stack(run.observations).astype(np.float32).tobytes()
    for key in ("raw_action", "policy_action", "targets", "commanded", "pos", "vel", "torque"):
        assert arrays[key].shape == (len(rows), 12)
    assert arrays["step"].tolist() == list(range(len(rows)))
    first_action = [float(rows[0][f"{name.replace('_joint', '')}.action"]) for name in run.contract.joint_order]
    assert first_action == pytest.approx(arrays["policy_action"][0].tolist(), abs=1e-5)

    phases = read_csv(run.path.with_name(run.path.stem + "_phases.csv"))
    kinds = [row["phase"] for row in phases]
    assert kinds[0] == "enabled"
    assert kinds.count("default") > 10
    assert kinds.count("brake") >= 10
    assert kinds.index("brake") > max(i for i, kind in enumerate(kinds) if kind == "default")
    assert phases[1]["acc_x"] == "0.1"

    meta = load_meta(run.path)
    assert meta["files"]["arrays"].endswith("_arrays.npz")
    first, second = sorted(run.motors)[:2]
    params = meta["preflight_parameters"]
    assert params[str(first)] == {"limit_torque_nm": 17.0, "limit_cur_a": 23.0, "vbus_v": 48.2,
                                  "run_mode": 0, "zero_sta": 1}
    assert params[str(second)]["limit_torque_nm"] is None
    assert meta["imu_gyro_bias_raw"] == [0.001, 0.002, 0.003]
    assert meta["host"]["git"]["deploy"]["head"]
    end = meta["end"]
    assert end["exit"] == "completed"
    assert end["policy_steps"] == len(rows) and end["rows_recorded"] == len(rows)
    assert end["rows_dropped"] == 0 and end["recording_errors"] == {}
    assert end["shutdown"]["observe_errors"] == []
    assert end["approach"]["seconds"] > 0.0
    assert all(abs(v) < 1.0 for v in end["approach"]["final_error_deg"].values())


def test_full_deploy_failure_still_brakes_then_saves(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    run = run_full_deploy(new, monkeypatch, tmp_path, fail_in_policy=True)
    capsys.readouterr()
    assert run.error is not None and "synthetic pipeline failure" in str(run.error)
    assert run.events.index("brake") < run.events.index("close")
    rows = read_csv(run.path)
    assert len(rows) == 40
    arrays = np.load(run.path.with_name(run.path.stem + "_arrays.npz"))
    assert arrays["obs"].shape == (40, 235)
    meta = load_meta(run.path)
    assert "synthetic pipeline failure" in meta["end"]["exit"]
    phases = read_csv(run.path.with_name(run.path.stem + "_phases.csv"))
    assert sum(row["phase"] == "brake" for row in phases) >= 10


@pytest.mark.parametrize("existing", ["live.csv", "live_phases.csv", "live_meta.json", "live_arrays.npz"])
def test_existing_recording_file_refused_before_enable_and_kept(modules, monkeypatch, tmp_path, capsys, existing):
    _, new = modules
    original = b"ORIGINAL DO NOT OVERWRITE\n"
    (tmp_path / existing).write_bytes(original)
    events = []
    monkeypatch.setattr(new, "confirm", lambda prompt: events.append("confirm"))
    with pytest.raises(SystemExit):
        run_full_deploy(new, monkeypatch, tmp_path)
    capsys.readouterr()
    assert (tmp_path / existing).read_bytes() == original
    assert "enable" not in events and "confirm" not in events
    if existing != "live.csv":
        assert not (tmp_path / "live_meta.json").exists() or existing == "live_meta.json"


def test_stop_row_with_full_buffer_does_not_flush(modules, tmp_path, monkeypatch):
    _, new = modules
    contract = make_contract()
    recorder = new.TelemetryRecorder(tmp_path / "t.csv", contract, {})
    recorder.rows = [["x"]] * (recorder.MAX_ROWS - 1)
    flushed = []
    monkeypatch.setattr(recorder, "flush", lambda: flushed.append(len(recorder.rows)))
    recorder.record_stop(1.0, 0.0, "synthetic stop", {})
    assert flushed == [] and len(recorder.rows) == recorder.MAX_ROWS


def test_stop_row_is_buffered_not_written_before_brake(modules, tmp_path):
    _, new = modules
    contract = make_contract()
    recorder = new.TelemetryRecorder(tmp_path / "t.csv", contract, {})
    recorder.record_stop(1.0, 0.0, "synthetic stop", {})
    assert (tmp_path / "t.csv").read_text().count("\n") == 1
    recorder.close()
    rows = read_csv(tmp_path / "t.csv")
    assert len(rows) == 1 and rows[0]["stop_reason"] == "synthetic stop"
    assert not (tmp_path / "t_arrays.npz").exists()


def test_policy_loop_arrays_match_session(modules, tmp_path, capsys):
    _, new = modules
    contract = make_contract()
    _, observations = run_loop(new, contract, tmp_path / "loop.csv")
    capsys.readouterr()
    arrays = np.load(tmp_path / "loop_arrays.npz")
    assert arrays["obs"].tobytes() == np.stack(observations).astype(np.float32).tobytes()
    assert arrays["velocity_command"].shape == (len(observations), 3)


def test_brake_observer_runs_and_failures_do_not_stop_braking(modules):
    import safety

    log = []
    motors = {mid: BrakeMotor(mid, 0.0, log, safety.MODE_RUNNING) for mid in (1, 2)}
    calls = []

    def observe():
        calls.append(1)
        if len(calls) > 3:
            raise RuntimeError("observer broke")

    report = safety.brake_and_stop(motors, {}, [1, 2], [1, 2], 0.1, 2.0, observe=observe)
    assert len(calls) >= 5
    assert len(report["observe_errors"]) == len(calls) - 3
    assert all(entry[3] == 0.0 and entry[4] == 2.0 for entry in log)
    assert len(log) >= 2 * (len(calls) - 1)
    assert all(motor.stopped for motor in motors.values())
    assert report["stop_sent"] == [1, 2]

    plain = safety.brake_and_stop(motors, {}, [1, 2], [1, 2], 0.02, 2.0)
    assert "observe_errors" not in plain


def test_timestamped_hub_counts_frames(modules):
    import safety

    class Bus:
        def __init__(self, frames):
            self.frames = list(frames)

        def recv(self, timeout=0.0):
            return self.frames.pop(0) if self.frames else None

    hub = safety.TimestampedFeedbackHub.__new__(safety.TimestampedFeedbackHub)
    hub.bus = Bus([SimpleNamespace(timestamp=1.0)] * 3)
    hub.route = lambda msg, now: None
    hub.pump()
    assert hub.last_pump_frames == 3 and hub.total_frames == 3
    hub.bus = Bus([])
    hub.pump()
    assert hub.last_pump_frames == 0 and hub.total_frames == 3


def test_preflight_parameter_reader_and_table(modules):
    _, new = modules
    import safety

    good = BrakeMotor(5, 0.0, [], safety.MODE_RUNNING)
    good.parameters = {0x700B: 14.0, 0x7018: 23.0, 0x701C: float("nan"), 0x7005: 0, 0x7029: 1}
    silent = BrakeMotor(3, 0.0, [], safety.MODE_RUNNING)
    values = new.read_preflight_parameters({5: good, 3: silent})
    assert values[5] == {"limit_torque_nm": 14.0, "limit_cur_a": 23.0, "vbus_v": None,
                         "run_mode": 0, "zero_sta": 1}
    assert all(v is None for v in values[3].values())
    lines = new.preflight_parameter_lines(values)
    assert len(lines) == 4 and "14.000" in lines[-1] and lines[2].split()[0] == "3"


def test_phase_recorder_survives_bad_motor(modules, tmp_path):
    _, new = modules
    contract = make_contract()

    class Broken:
        @property
        def last_position(self):
            raise RuntimeError("bad")

    recorder = new.PhaseRecorder(tmp_path / "p.csv", contract, {1: Broken()})
    recorder.record("brake")
    recorder.close()
    assert recorder.error and "bad" in recorder.error
    assert len(read_csv(tmp_path / "p.csv")) == 0


def test_brake_damps_before_first_observation(modules):
    import safety

    log = []
    motors = {1: BrakeMotor(1, 0.0, log, safety.MODE_RUNNING)}
    order = []

    def observe():
        order.append(("observe", len(log)))

    safety.brake_and_stop(motors, {}, [1], [1], 0.05, 2.0, observe=observe)
    assert order and order[0][1] >= 1
    assert motors[1].stopped


def test_step_arrays_capacity(modules, tmp_path, monkeypatch):
    _, new = modules
    arrays = new.StepArrays(tmp_path / "a.npz")
    monkeypatch.setattr(arrays, "MAX_STEPS", 3)
    for step in range(5):
        arrays.append(step=step, obs=np.zeros(4), t_s=0.5 * step)
    assert arrays.steps == 3 and arrays.error.startswith("capacity")
    arrays.save()
    data = np.load(tmp_path / "a.npz")
    assert data["step"].dtype == np.int64 and data["step"].tolist() == [0, 1, 2]
    assert data["t_s"].dtype == np.float64 and data["obs"].dtype == np.float32
