import csv
import importlib.util
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "7d6c9e0"
POLICY_TO_REAL = DEPLOY / "scripts" / "policy_test" / "policy_to_real.py"
TIMING_REPORT = DEPLOY / "scripts" / "analysis" / "timing_report.py"


class FakeSession:
    instances = []

    def __init__(self, path, providers=None):
        rng = np.random.default_rng(7)
        self.weights = rng.normal(0.0, 0.05, size=(12, 235)).astype(np.float32)
        self.observations = []
        FakeSession.instances.append(self)

    def get_inputs(self):
        return [SimpleNamespace(name="obs", shape=[1, 235])]

    def get_outputs(self):
        return [SimpleNamespace(name="actions", shape=[1, 12])]

    def run(self, outputs, feed):
        observation = np.array(feed["obs"], dtype=np.float32, copy=True)
        self.observations.append(observation.reshape(-1))
        return [np.sin(observation @ self.weights.T).astype(np.float32) * 2.0]


def install_stubs():
    if "onnxruntime" not in sys.modules and importlib.util.find_spec("onnxruntime") is None:
        sys.modules["onnxruntime"] = types.SimpleNamespace(InferenceSession=FakeSession)
    if "n100" not in sys.modules and importlib.util.find_spec("n100") is None:
        sys.modules["n100"] = types.ModuleType("n100")


def load_module(name, path):
    install_stubs()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ort = types.SimpleNamespace(InferenceSession=FakeSession)
    return module


@pytest.fixture(scope="module")
def modules(tmp_path_factory):
    try:
        source = subprocess.run(
            ["git", "-C", str(DEPLOY), "show", f"{BASELINE_COMMIT}:scripts/policy_test/policy_to_real.py"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(f"baseline {BASELINE_COMMIT} not readable: {error}")
    base_dir = tmp_path_factory.mktemp("baseline") / "scripts" / "policy_test"
    base_dir.mkdir(parents=True)
    base_path = base_dir / "policy_to_real_baseline.py"
    base_path.write_text(source)
    return load_module("policy_to_real_baseline", base_path), load_module("policy_to_real_new", POLICY_TO_REAL)


def make_contract():
    from robonex_common.models import robot_model
    from robonex_common.policy import PolicyContract

    roll = robot_model("ver2_edu").foot_roll
    return PolicyContract.from_dict({
        "schema_version": 3,
        "task": "RoboNex-Walking-V2-Edu-v0",
        "policy_file": "policy.onnx",
        "policy_sha256": "0" * 64,
        "description_sha256": "0" * 64,
        "common_sha256": "0" * 64,
        "training_sha256": "0" * 64,
        "description_model": "ver2/mujoco/robot/edu/scene.xml",
        "joint_order": [
            "l_hip_yaw_joint", "r_hip_yaw_joint", "l_hip_pitch_joint", "r_hip_pitch_joint",
            "l_hip_roll_joint", "r_hip_roll_joint", "l_knee_pitch_joint", "r_knee_pitch_joint",
            "l_ankle_lower_joint", "l_ankle_upper_joint", "r_ankle_lower_joint", "r_ankle_upper_joint",
        ],
        "observation_terms": [
            "joint_pos_rel:12x5", "joint_vel_rel:12x5", "imu_ang_vel:3x5", "projected_gravity:3x5",
            "velocity_commands:3x5", "gait_phase:2x5", "last_action:12x5",
        ],
        "action_offsets": [0.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.3298656951565011, 0.3298656951565011,
                           -0.2038048914121279, 0.21255773798848565, 0.2038048914121279, -0.21255773798848565],
        "action_scales": [0.25, 0.25, 0.25, 0.25, 0.148886, 0.148886, 0.1648, 0.1648,
                          0.219621, 0.160604, 0.219621, 0.160604],
        "target_clips": [[-0.827758, 0.827758], [-0.827758, 0.827758], [-1.648063, 1.648063],
                         [-1.648063, 1.648063], [-2.084395, 0.164533], [-0.164533, 2.084395],
                         [-1.21173, 0.164533], [-0.164533, 1.21173], [-0.862665, 0.513599],
                         [-0.269253, 0.862665], [-0.513599, 0.862665], [-0.862665, 0.269253]],
        "runner_action_clip": 14.0,
        "observation_size": 235,
        "action_size": 12,
        "policy_hz": 50.0,
        "description_commit": "0" * 40,
        "common_commit": "0" * 40,
        "training_commit": "0" * 40,
        "robot_model": "ver2_edu",
        "foot_roll_limit": 0.0 if roll is None else roll.limit,
        "foot_roll_coeffs": [] if roll is None else list(roll.coeffs),
        "foot_roll_pairs": [] if roll is None else [list(p) for p in roll.pairs],
    })


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(round(self.now * 1.0e9))

    def perf_counter(self):
        return self.now

    def time(self):
        return 1.7e9 + self.now

    def sleep(self, seconds):
        self.now += seconds

    def strftime(self, fmt, *args):
        return "00:00:00"


class FakeMotor:
    def __init__(self, motor_id, position, log, mode_running):
        self.motor_id = motor_id
        self.last_position = position
        self.last_velocity = 0.0
        self.last_torque = 0.5
        self.last_temp = 30.0
        self.last_fault = 0
        self.last_mode_status = mode_running
        self.last_feedback_time = 0.0
        self.last_command = position
        self.log = log

    def control(self, pos, vel, kp, kd, torque=0.0):
        self.last_command = pos
        self.log.append((self.motor_id, pos, vel, kp, kd, torque))


class FakeJoints:
    def __init__(self, motors, clock):
        self.motors = motors
        self.clock = clock

    def poll(self):
        t = self.clock.now
        for motor_id, motor in self.motors.items():
            previous = motor.last_position
            motor.last_position = (previous + 0.3 * (motor.last_command - previous)
                                   + 0.002 * math.sin(2.0 * math.pi * 1.3 * t + motor_id))
            motor.last_velocity = (motor.last_position - previous) / 0.02
            motor.last_torque = 0.5 + 0.1 * math.sin(t + motor_id)
            motor.last_feedback_time = t - 0.0004 * motor_id
            motor.last_rx_kernel_time = self.clock.time() - 0.0005

    def snapshot(self):
        return {motor_id: (m.last_position, m.last_velocity) for motor_id, m in self.motors.items()}

    def rate_text(self):
        return "fake"


class FakeVec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeImuDriver:
    PERIOD = 0.01
    PHASE = 0.0037

    def __init__(self, clock, stall_after=None):
        self.clock = clock
        self.stall_after = stall_after
        self.is_running = True

    def index(self):
        t = self.clock.now
        if self.stall_after is not None:
            t = min(t, self.stall_after)
        return int(math.floor((t - self.PHASE) / self.PERIOD))

    def latest(self):
        k = self.index()
        return SimpleNamespace(
            seq=k,
            host_timestamp_ns=int(round((k * self.PERIOD + self.PHASE) * 1.0e9)),
            device_timestamp_us=5_000_000 + k * 10_000 + k // 50,
            has_imu_frame=True,
            angular_velocity_raw=FakeVec(0.01 * math.sin(k), 0.02 * math.cos(0.3 * k), -0.005),
            projected_gravity=FakeVec(0.02 * math.sin(0.1 * k), 0.01, -0.9997),
        )

    def stats(self):
        return SimpleNamespace(imu_frames=self.index())

    def last_error(self):
        return ""


def run_loop(module, contract, telemetry_path, duration=3.0):
    import safety
    from robonex_common.joints import JOINT_BY_ID, JOINT_BY_MODEL_NAME
    from robonex_common.models import robot_model
    from robonex_common.motors import RATED_TORQUE

    clock = FakeClock()
    real_time = module.time
    module.time = clock
    FakeSession.instances.clear()
    try:
        runner = module.PolicyRunner(Path("policy.onnx"), contract)
        session = FakeSession.instances[-1]
        control_log = []
        motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
        starts = {JOINT_BY_MODEL_NAME[name].motor_id: float(contract.action_offsets[i])
                  for i, name in enumerate(contract.joint_order)}
        motors = {mid: FakeMotor(mid, starts[mid], control_log, safety.MODE_RUNNING) for mid in motor_ids}
        joints = FakeJoints(motors, clock)
        imu = module.ImuSource(module.SETTINGS, [])
        imu.driver = FakeImuDriver(clock)
        imu.status = "ready"
        commander = module.TargetCommander(motors, starts, module.SETTINGS)
        limits = {mid: robot_model("ver2_edu").joint_limits_by_id()[mid] for mid in motor_ids}
        thermal = module.ThermalLoad({mid: RATED_TORQUE[JOINT_BY_ID[mid].motor_model] for mid in motor_ids})
        args = SimpleNamespace(
            duration=duration, telemetry=telemetry_path, vx=0.2, vy=0.0, wz=0.1, keyboard=False,
        )
        runner.policy_path = Path("policy.onnx")
        runner.velocity_command[:] = (args.vx, args.vy, args.wz)
        module.policy_loop(runner, commander, joints, imu, motors, limits, contract, args, [],
                           module.LoopStats(), module.FaultMonitor(), thermal, None)
    finally:
        module.time = real_time
    return control_log, session.observations


def command_bytes(commands):
    return np.asarray(commands, dtype=np.float64).tobytes()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("with_telemetry", [False, True])
def test_policy_loop_commands_and_observations_unchanged(modules, tmp_path, capsys, with_telemetry):
    baseline, new = modules
    contract = make_contract()
    base_csv = tmp_path / "base.csv" if with_telemetry else None
    new_csv = tmp_path / "new.csv" if with_telemetry else None
    base_commands, base_obs = run_loop(baseline, contract, base_csv)
    new_commands, new_obs = run_loop(new, contract, new_csv)
    capsys.readouterr()

    steps = len(new_obs)
    assert steps >= 150 and len(base_obs) == steps
    assert len(base_commands) == len(new_commands) == steps * 12
    assert command_bytes(base_commands) == command_bytes(new_commands)
    for a, b in zip(base_obs, new_obs):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    assert any(abs(pos) > 1e-3 for _, pos, *_ in new_commands)

    if with_telemetry:
        base_rows, new_rows = read_csv(base_csv), read_csv(new_csv)
        assert len(base_rows) == len(new_rows) == steps
        joint_extras = [f"{name.replace('_joint', '')}.{field}" for name in contract.joint_order
                        for field in new.JOINT_EXTRA_FIELDS]
        added = list(new.TIMING_COLUMNS) + list(new.STEP_EXTRA_COLUMNS) + joint_extras
        assert list(new_rows[0]) == list(base_rows[0])[:-1] + added + ["stop_reason"]
        for old_row, new_row in zip(base_rows, new_rows):
            assert {k: new_row[k] for k in old_row} == old_row
        base_lines, new_lines = base_csv.read_bytes().splitlines(), new_csv.read_bytes().splitlines()
        assert len(base_lines) == len(new_lines) == steps + 1
        for old_line, new_line in zip(base_lines, new_lines):
            old_prefix, old_stop = old_line.rsplit(b",", 1)
            new_prefix, *added_cells, new_stop = new_line.rsplit(b",", len(added) + 1)
            assert len(added_cells) == len(added)
            assert (old_prefix, old_stop) == (new_prefix, new_stop)
        ages = [float(r["imu_host_age_ms"]) for r in new_rows]
        assert ages == pytest.approx([6.3] * steps, abs=1e-6)
        assert all(float(r["tick_period_ms"]) == pytest.approx(20.0, abs=1e-6) for r in new_rows[1:])
        assert all(r["imu_seq_gap"] == "1" for r in new_rows[1:])
        assert all(float(r["imu_host_dt_ms"]) == pytest.approx(20.0, abs=1e-6) for r in new_rows[1:])
        assert new_rows[0]["tick_period_ms"] == new_rows[0]["imu_seq_gap"] == ""
        header = list(new_rows[0])
        assert header[-1] == "stop_reason"
        assert header[: len(base_rows[0]) - 1] == list(base_rows[0])[:-1]


def test_imu_read_and_stale_check_unchanged(modules):
    baseline, new = modules
    results = []
    for module in (baseline, new):
        clock = FakeClock()
        imu = module.ImuSource(module.SETTINGS, [])
        imu.driver = FakeImuDriver(clock, stall_after=1000.5)
        out = []
        for _ in range(60):
            values = imu.read(clock.now)
            out.append((values, imu.failure_reason(values[2])))
            clock.sleep(0.02)
        results.append(out)
    assert results[0] == results[1]
    assert any(reason for _, reason in results[1])


def test_timing_probe_values(modules):
    _, new = modules
    probe = new.TimingProbe()
    sample = lambda seq, host_ns, dev_us: SimpleNamespace(seq=seq, host_timestamp_ns=host_ns, device_timestamp_us=dev_us)
    first = probe.update(1.0, sample(10, 990_000_000, 100_000), 1_000_000_000)
    assert first == (10.0, None, None, None, None)
    second = probe.update(1.02, sample(12, 1_010_000_000, 120_010), 1_020_500_000)
    assert second[0] == pytest.approx(10.5)
    assert second[1] == pytest.approx(20.01)
    assert second[2] == pytest.approx(20.0)
    assert second[3] == 1
    assert second[4] == pytest.approx(20.0)
    repeated = probe.update(1.04, sample(12, 1_010_000_000, 120_010), 1_040_000_000)
    assert repeated[3] == -1 and repeated[1] == 0.0
    assert probe.update(1.06, None, 0)[:4] == (None, None, None, None)
    assert new.timing_cells(None) == [""] * 5


def test_read_recorder_appends_timing_columns(modules, tmp_path):
    baseline, new = modules
    contract = make_contract()
    paths = []
    for module, name in ((baseline, "base.csv"), (new, "new.csv")):
        recorder = module.ReadRecorder(tmp_path / name, contract)
        observation = np.arange(235, dtype=np.float32) / 100.0
        kwargs = {"timing": (4.2, 20.0, 20.0, 1, 20.1)} if module is new else {}
        recorder.record(5.0, observation, (0.0, 0.0, -1.0), (0.1, 0.2, 0.3), 0.0,
                        np.zeros(12, dtype=np.float32), [], **kwargs)
        recorder.close()
        paths.append(tmp_path / name)
    base_rows, new_rows = read_csv(paths[0]), read_csv(paths[1])
    assert list(new_rows[0])[: len(base_rows[0])] == list(base_rows[0])
    joint_extras = [f"{name.replace('_joint', '')}.{field}" for name in contract.joint_order
                    for field in new.READ_JOINT_EXTRA_FIELDS]
    assert list(new_rows[0])[len(base_rows[0]):] == (
        list(new.TIMING_COLUMNS) + joint_extras + list(new.IMU_EXTRA_COLUMNS))
    assert {k: new_rows[0][k] for k in base_rows[0]} == base_rows[0]
    assert new_rows[0]["imu_host_age_ms"] == "4.2" and new_rows[0]["imu_seq_gap"] == "1"


@pytest.fixture(scope="module")
def timing_report():
    spec = importlib.util.spec_from_file_location("timing_report", TIMING_REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_synthetic(path, n=3000):
    rng = np.random.default_rng(3)
    header = ["t_s", "imu_age_ms", "l_hip_yaw.age_ms", "l_hip_yaw.rx_age_ms",
              "r_hip_yaw.age_ms", "r_hip_yaw.rx_age_ms", *(
                  "imu_host_age_ms", "imu_device_dt_ms", "imu_host_dt_ms", "imu_seq_gap", "tick_period_ms"),
              "stop_reason"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(n):
            t = i * 0.02
            age = 4.0 + 0.5 * t / 60.0 + rng.uniform(0.0, 0.2)
            gap = 1 if i % 100 else 2
            writer.writerow([
                round(t, 4), 0.0, 1.0, 1.5, 2.0, 2.5,
                round(age, 3),
                "" if i == 0 else round(20.0 * (gap + 1) / 2 * (1 + 100e-6), 6),
                "" if i == 0 else 20.0 * (gap + 1) / 2,
                "" if i == 0 else gap,
                "" if i == 0 else 20.0 + (0.3 if i % 50 == 0 else 0.0),
                "",
            ])
        writer.writerow([n * 0.02, "", "", "", "", "", 99.0, "", "", "", "", "IMU sample is stale"])


def test_timing_report_on_synthetic_csv(timing_report, tmp_path, capsys):
    path = tmp_path / "synthetic_live_telemetry.csv"
    write_synthetic(path)
    result = timing_report.report(path, d0_ms=None, window_s=10.0)
    age = result["imu_host_age_ms"]
    assert age["n"] == 3000
    assert 4.0 < age["p50"] < 5.0 and age["max"] < 5.0
    assert result["phase_drift"]["slope_ms_per_min"] == pytest.approx(0.5, abs=0.05)
    assert len(result["phase_drift"]["windows"]) == 6
    assert result["imu_seq_gap"]["histogram"] == {"1": 2970, "2": 29}
    assert result["clock_rate"]["device_minus_host_ppm"] == pytest.approx(100.0, abs=0.5)
    assert result["tick_period_ms"]["max"] == pytest.approx(20.3)
    assert set(result["joint_fb_age_ms"]) == {"l_hip_yaw", "r_hip_yaw"}
    assert result["joint_rx_age_ms"]["r_hip_yaw"]["mean"] == pytest.approx(2.5)
    assert "d0 <=" in result["budget"] and "d0 unknown" in result["budget"]
    inside = timing_report.report(path, d0_ms=3.5)["budget"]
    assert inside.endswith("INSIDE")
    outside = timing_report.report(path, d0_ms=11.0)["budget"]
    assert outside.endswith("OUTSIDE")
    assert timing_report.main([str(path), "--d0-ms", "3.5", "--output", str(tmp_path / "out.json")]) == 0
    printed = capsys.readouterr().out
    assert "host age" in printed and "INSIDE" in printed
    assert (tmp_path / "out.json").exists()


def test_timing_report_rejects_old_csv(timing_report, tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("t_s,imu_age_ms,stop_reason\n0.0,0.0,\n")
    with pytest.raises(ValueError, match="imu_host_age_ms"):
        timing_report.report(path)


def test_timing_report_on_policy_loop_output(modules, timing_report, tmp_path, capsys):
    _, new = modules
    path = tmp_path / "loop_live_telemetry.csv"
    _, observations = run_loop(new, make_contract(), path)
    capsys.readouterr()
    result = timing_report.report(path, d0_ms=None)
    assert result["imu_host_age_ms"]["p50"] == pytest.approx(6.3, abs=1e-6)
    assert result["imu_seq_gap"]["histogram"] == {"1": len(observations) - 1}
    assert result["tick_period_ms"]["mean"] == pytest.approx(20.0, abs=1e-6)
    assert len(result["joint_fb_age_ms"]) == 12 and len(result["joint_rx_age_ms"]) == 12


def test_policy_loop_survives_failing_probe(modules, tmp_path, capsys, monkeypatch):
    _, new = modules
    contract = make_contract()
    clean_commands, clean_obs = run_loop(new, contract, tmp_path / "clean.csv")
    original = new.TimingProbe.update
    calls = []

    def failing(self, tick, sample, read_ns):
        calls.append(tick)
        if len(calls) > 50:
            raise AttributeError("device_timestamp_us")
        return original(self, tick, sample, read_ns)

    monkeypatch.setattr(new.TimingProbe, "update", failing)
    path = tmp_path / "failing.csv"
    commands, observations = run_loop(new, contract, path)
    printed = capsys.readouterr().out
    assert len(observations) == len(clean_obs) >= 150
    assert command_bytes(commands) == command_bytes(clean_commands)
    rows = read_csv(path)
    assert len(rows) == len(observations)
    assert all(rows[i]["imu_host_age_ms"] != "" for i in range(50))
    assert all(all(row[c] == "" for c in new.TIMING_COLUMNS) for row in rows[50:])
    assert all(row["stop_reason"] == "" for row in rows)
    assert printed.count("timing telemetry failed") == 1


class FieldlessImuDriver(FakeImuDriver):
    def latest(self):
        sample = super().latest()
        del sample.device_timestamp_us
        del sample.host_timestamp_ns
        return sample


class ReachedEnable(Exception):
    pass


def run_deploy_until_enable(module, monkeypatch, driver_class, telemetry):
    events = []

    def start(self, calibrate):
        self.driver = driver_class(FakeClock())
        self.status = "ready"
        return True

    def enable(*args, **kwargs):
        events.append("enable")
        raise ReachedEnable()

    monkeypatch.setattr(module, "verify_common_source", lambda contract: None)
    monkeypatch.setattr(module, "require_robot_model", lambda name: None)
    monkeypatch.setattr(module, "open_hardware", lambda ids, interface, host_id: ({}, {}, {}))
    monkeypatch.setattr(module, "stop_idle_motors", lambda buses, ids, host_id: [])
    monkeypatch.setattr(module, "inspect_zero_positions", lambda motors, tolerance, limits: ({}, []))
    monkeypatch.setattr(module, "roll_pairs_for", lambda profile, ids: [])
    monkeypatch.setattr(module, "confirm", lambda prompt: events.append("confirm"))
    monkeypatch.setattr(module, "enable_with_runtime_feedback", enable)
    monkeypatch.setattr(module, "brake_and_stop", lambda *args, **kwargs: events.append("brake"))
    monkeypatch.setattr(module, "shutdown_report_lines", lambda report: [])
    monkeypatch.setattr(module.ImuSource, "start", start)
    monkeypatch.setattr(module.ImuSource, "stop", lambda self: None)
    args = SimpleNamespace(vx=0.0, vy=0.0, wz=0.0, keyboard=False,
                           duration=1.0, telemetry=telemetry)
    try:
        module.run_deploy(Path("policy.onnx"), make_contract(), args)
    except ReachedEnable:
        events.append("reached")
    except RuntimeError as error:
        events.append(str(error))
    return events


def test_telemetry_capability_refused_before_enable(modules, tmp_path, monkeypatch, capsys):
    _, new = modules
    events = run_deploy_until_enable(new, monkeypatch, FieldlessImuDriver, tmp_path / "t.csv")
    assert "confirm" not in events and "enable" not in events and "reached" not in events
    assert any("device_timestamp_us" in e and "host_timestamp_ns" in e and "will not be enabled" in e
               for e in events)
    assert events[-2] == "brake"


def test_telemetry_off_ignores_missing_timing_fields(modules, monkeypatch, capsys):
    _, new = modules
    events = run_deploy_until_enable(new, monkeypatch, FieldlessImuDriver, None)
    assert events == ["confirm", "enable", "brake", "reached"]


def test_telemetry_capability_passes_with_full_sample(modules, tmp_path, monkeypatch, capsys):
    _, new = modules
    events = run_deploy_until_enable(new, monkeypatch, FakeImuDriver, tmp_path / "t.csv")
    assert events == ["confirm", "enable", "brake", "reached"]


def test_missing_timing_fields(modules):
    _, new = modules
    assert new.missing_timing_fields(None) == list(new.TIMING_SAMPLE_FIELDS)
    full = SimpleNamespace(seq=1, device_timestamp_us=2, host_timestamp_ns=3)
    assert new.missing_timing_fields(full) == []
    assert new.missing_timing_fields(SimpleNamespace(seq=1, device_timestamp_us=None,
                                                     host_timestamp_ns=float("nan"))) == [
        "device_timestamp_us", "host_timestamp_ns"]


def test_timing_report_stop_only_csv(modules, timing_report, tmp_path, capsys):
    _, new = modules
    contract = make_contract()
    from robonex_common.joints import JOINT_BY_MODEL_NAME
    import safety

    motors = {}
    for name in contract.joint_order:
        motor_id = JOINT_BY_MODEL_NAME[name].motor_id
        motors[motor_id] = FakeMotor(motor_id, 0.0, [], safety.MODE_RUNNING)
    path = tmp_path / "stop_only_live_telemetry.csv"
    recorder = new.TelemetryRecorder(path, contract, motors)
    recorder.record_stop(1000.0, 1000.0, "IMU produced no sample", {m: 0.0 for m in motors})
    recorder.close()
    rows = read_csv(path)
    assert len(rows) == 1 and rows[0]["stop_reason"] == "IMU produced no sample"
    with pytest.raises(ValueError, match="no non-stop samples"):
        timing_report.report(path)
    assert timing_report.main([str(path)]) == 1
    captured = capsys.readouterr()
    assert "no non-stop samples" in captured.err


@pytest.mark.parametrize("window", [0.0, -1.0, math.nan, math.inf])
def test_timing_report_rejects_bad_window(timing_report, tmp_path, window, capsys):
    path = tmp_path / "synthetic_live_telemetry.csv"
    write_synthetic(path, n=10)
    with pytest.raises(ValueError, match="window_s"):
        timing_report.report(path, window_s=window)
    t = np.arange(10) * 0.02
    with pytest.raises(ValueError, match="window_s"):
        timing_report.phase_drift(t, np.ones(10), window)
    with pytest.raises(SystemExit) as exit_info:
        timing_report.main([str(path), "--window", str(window)])
    assert exit_info.value.code == 2
    assert "--window must be a finite number > 0" in capsys.readouterr().err


def test_timing_report_window_cli_does_not_hang(tmp_path):
    path = tmp_path / "synthetic_live_telemetry.csv"
    write_synthetic(path, n=10)
    for window in ("0", "-1"):
        result = subprocess.run([sys.executable, str(TIMING_REPORT), str(path), "--window", window],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 2 and "--window" in result.stderr


@pytest.mark.parametrize("d0", [-1.0, math.nan, math.inf])
def test_timing_report_rejects_bad_d0(timing_report, tmp_path, d0, capsys):
    path = tmp_path / "synthetic_live_telemetry.csv"
    write_synthetic(path, n=10)
    with pytest.raises(ValueError, match="d0_ms"):
        timing_report.report(path, d0_ms=d0)
    with pytest.raises(SystemExit) as exit_info:
        timing_report.main([str(path), "--d0-ms", str(d0)])
    assert exit_info.value.code == 2
    assert timing_report.report(path, d0_ms=0.0)["budget"].endswith("INSIDE")


def test_timing_report_rejects_partial_schema(timing_report, tmp_path, capsys):
    path = tmp_path / "synthetic_live_telemetry.csv"
    write_synthetic(path, n=10)
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    header = [k for k in rows[0] if k != "tick_period_ms"]
    partial = tmp_path / "partial_live_telemetry.csv"
    with open(partial, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="tick_period_ms"):
        timing_report.report(partial)
    assert timing_report.main([str(partial)]) == 1
    assert "tick_period_ms" in capsys.readouterr().err


def test_joint_table_shows_command_error_torque_and_temperature(modules):
    ptr = modules[1]
    contract = make_contract()
    order = ptr.joint_row_order(contract)
    positions = np.zeros(12)
    velocities = np.zeros(12)
    knee_index = next(i for i, (name, _) in enumerate(order) if name == "l_knee_pitch_joint")
    knee_id = order[knee_index][1]
    positions[knee_index] = math.radians(-20.0)
    commands = {motor_id: 0.0 for _, motor_id in order}
    commands[knee_id] = math.radians(-18.5)
    motors = {motor_id: SimpleNamespace(spec=SimpleNamespace(t_max=60.0), last_torque=0.0, last_temp=30.0)
              for _, motor_id in order}
    motors[knee_id] = SimpleNamespace(spec=SimpleNamespace(t_max=60.0), last_torque=-12.0, last_temp=41.5)
    lines = ptr.format_joint_table(contract, positions, velocities, commands, motors)
    assert "torque" in lines[0] and "temp" in lines[0] and "raw" not in lines[0]
    row = next(line for line in lines[2:] if "l_knee_pitch" in line)
    assert "-20.00d" in row and "-18.50d" in row and "-1.50d" in row
    assert "-12.00 ( 20%)" in row and "41.5C" in row
    preview = ptr.format_joint_table(contract, positions, velocities, None)
    row = next(line for line in preview[2:] if "l_knee_pitch" in line)
    assert row.count("--") == 4
