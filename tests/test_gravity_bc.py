import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_policy_to_real_timing import (
    DEPLOY,
    POLICY_TO_REAL,
    FakeClock,
    FakeImuDriver,
    FakeJoints,
    FakeMotor,
    FakeSession,
    load_module,
    make_contract,
    read_csv,
)
from test_policy_to_real_recording import run_full_deploy, load_meta

BASELINE_COMMIT = "0a5bf3d"
HW_DIR = Path("/home/polygon/humanoid_project/etc/2026-10-11_hw")
E_BC = Path("/home/polygon/humanoid_project/robonex-walking/etc/delegates/2026-10-12_hwlog/E_work/bc.py")
LOGGED_RUNS = ("20261011_195254", "20261011_192811")
TILT = math.radians(3.0)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    try:
        source = subprocess.run(
            ["git", "-C", str(DEPLOY), "show", f"{BASELINE_COMMIT}:scripts/policy_test/policy_to_real.py"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(f"baseline {BASELINE_COMMIT} not readable: {error}")
    base_dir = tmp_path_factory.mktemp("baseline_bc") / "scripts" / "policy_test"
    base_dir.mkdir(parents=True)
    base_path = base_dir / "policy_to_real_baseline_bc.py"
    base_path.write_text(source)
    hold_dir = base_dir.parent / "sim_to_real"
    hold_dir.mkdir()
    hold_dir.joinpath("heading_hold.py").write_text(subprocess.run(
        ["git", "-C", str(DEPLOY), "show", f"{BASELINE_COMMIT}:scripts/sim_to_real/heading_hold.py"],
        check=True, capture_output=True, text=True,
    ).stdout)
    sys.path.insert(0, str(hold_dir))
    try:
        baseline = load_module("policy_to_real_baseline_bc", base_path)
    finally:
        sys.path.remove(str(hold_dir))
    return baseline, load_module("policy_to_real_bc", POLICY_TO_REAL)


@pytest.fixture(scope="module")
def ptr(pair):
    return pair[1]


class TiltedImuDriver(FakeImuDriver):
    def latest(self):
        sample = super().latest()
        sample.linear_acceleration = SimpleNamespace(x=0.0, y=-9.81 * math.sin(TILT), z=9.81 * math.cos(TILT))
        sample.orientation = SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)
        sample.imu_temperature = 31.5
        return sample


def run_loop(module, telemetry_path, duration=3.0, **extra):
    import safety
    from robonex_common.joints import JOINT_BY_ID, JOINT_BY_MODEL_NAME
    from robonex_common.models import robot_model
    from robonex_common.motors import RATED_TORQUE

    contract = make_contract()
    clock = FakeClock()
    real_time = module.time
    module.time = clock
    FakeSession.instances.clear()
    seen = []
    try:
        runner = module.PolicyRunner(Path("policy.onnx"), contract)
        session = FakeSession.instances[-1]
        observe = runner.observation

        def spy(positions, velocities, angular_velocity, gravity):
            seen.append((tuple(angular_velocity), tuple(gravity)))
            return observe(positions, velocities, angular_velocity, gravity)

        runner.observation = spy
        control_log = []
        motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
        starts = {JOINT_BY_MODEL_NAME[name].motor_id: float(contract.action_offsets[i])
                  for i, name in enumerate(contract.joint_order)}
        motors = {mid: FakeMotor(mid, starts[mid], control_log, safety.MODE_RUNNING) for mid in motor_ids}
        imu = module.ImuSource(module.SETTINGS, [])
        imu.driver = TiltedImuDriver(clock)
        imu.status = "ready"
        commander = module.TargetCommander(motors, starts, module.SETTINGS)
        limits = {mid: robot_model("ver2_edu").joint_limits_by_id()[mid] for mid in motor_ids}
        thermal = module.ThermalLoad({mid: RATED_TORQUE[JOINT_BY_ID[mid].motor_model] for mid in motor_ids})
        args = SimpleNamespace(duration=duration, telemetry=telemetry_path, vx=0.2, vy=0.0, wz=0.1, keyboard=False,
                               **extra)
        runner.policy_path = Path("policy.onnx")
        runner.velocity_command[:] = (args.vx, args.vy, args.wz)
        module.policy_loop(runner, commander, FakeJoints(motors, clock), imu, motors, limits, contract, args, [],
                           module.LoopStats(), module.FaultMonitor(), thermal, None)
    finally:
        module.time = real_time
    return control_log, session.observations, seen


SEND_TIMING_COLUMNS = ("obs_ms", "tx_first_ms", "tx_last_ms")
REMOVED_HEADING_COLUMNS = (
    "heading_gyro_deg", "heading_quat_deg", "heading_target_deg", "heading_error_deg",
    "heading_integral", "heading_wz", "heading_engaged", "policy_wz",
)


def read_rows(csv_path):
    with csv_path.open(newline="") as stream:
        return list(csv.reader(stream))


def drop_columns(rows, names):
    keep = [index for index, name in enumerate(rows[0]) if name not in names]
    return [[row[index] for index in keep] for row in rows]


def load_arrays(csv_path):
    with np.load(csv_path.with_name(csv_path.stem + "_arrays.npz")) as data:
        return {key: data[key] for key in data.files}


def test_off_is_identical_to_baseline(pair, tmp_path, capsys):
    baseline, new = pair
    base_csv, off_csv = tmp_path / "base.csv", tmp_path / "off.csv"
    base_commands, base_obs, base_seen = run_loop(baseline, base_csv)
    off_commands, off_obs, off_seen = run_loop(new, off_csv, gravity_bc=False)
    capsys.readouterr()
    assert len(off_obs) == len(base_obs) >= 140
    assert np.asarray(off_commands).tobytes() == np.asarray(base_commands).tobytes()
    assert all(a.tobytes() == b.tobytes() for a, b in zip(base_obs, off_obs))
    assert off_seen == base_seen
    base_rows, off_rows = read_rows(base_csv), read_rows(off_csv)
    assert not set(SEND_TIMING_COLUMNS) & set(base_rows[0])
    assert set(REMOVED_HEADING_COLUMNS) <= set(base_rows[0])
    assert not set(REMOVED_HEADING_COLUMNS) & set(off_rows[0])
    assert drop_columns(off_rows, SEND_TIMING_COLUMNS) == drop_columns(base_rows, REMOVED_HEADING_COLUMNS)
    base_arrays, off_arrays = load_arrays(base_csv), load_arrays(off_csv)
    assert sorted(base_arrays) == sorted(off_arrays)
    assert all(base_arrays[k].tobytes() == off_arrays[k].tobytes() for k in base_arrays)
    for module, name in ((baseline, "base_read.csv"), (new, "off_read.csv")):
        recorder = module.ReadRecorder(tmp_path / name, make_contract())
        recorder.record(5.0, np.arange(235, dtype=np.float32) / 100.0, (0.0, 0.0, -1.0), (0.1, 0.2, 0.3), 0.0,
                        np.zeros(12, dtype=np.float32), [], timing=(4.2, 20.0, 20.0, 1, 20.1))
        recorder.close()
    assert (tmp_path / "off_read.csv").read_bytes() == (tmp_path / "base_read.csv").read_bytes()


def test_on_feeds_corrected_gravity_and_keeps_gyro(ptr, tmp_path, capsys):
    off_csv, on_csv = tmp_path / "off.csv", tmp_path / "on.csv"
    _, off_obs, off_seen = run_loop(ptr, off_csv, gravity_bc=False)
    _, on_obs, on_seen = run_loop(ptr, on_csv, gravity_bc=True)
    capsys.readouterr()
    assert len(on_seen) == len(off_seen)
    assert [gyro for gyro, _ in on_seen] == [gyro for gyro, _ in off_seen]
    off_rows, rows = read_csv(off_csv), read_csv(on_csv)
    header = list(rows[0])
    assert header == list(off_rows[0])[:-1] + list(ptr.GRAVITY_BC_COLUMNS) + ["stop_reason"]
    window = ptr.GravityBC.WINDOW
    replay = ptr.GravityBC()
    force = (0.0, -9.81 * math.sin(TILT), 9.81 * math.cos(TILT))
    for k, (row, (_, gravity)) in enumerate(zip(rows, on_seen)):
        raw = tuple(float(row[f"gravity_{axis}"]) for axis in "xyz")
        assert raw == pytest.approx(off_seen[k][1], abs=1e-5)
        expected = replay.update(off_seen[k][1], force)
        assert gravity == pytest.approx(expected, abs=1e-12)
        assert tuple(float(row[c]) for c in ptr.GRAVITY_BC_COLUMNS[:3]) == pytest.approx(gravity, abs=1e-5)
        assert row["gravity_bc_ready"] == ("1" if k >= window - 1 else "0")
        if k < window - 1:
            assert float(row["gravity_bc_droll_deg"]) == 0.0 and float(row["gravity_bc_dpitch_deg"]) == 0.0
    last_roll = float(rows[-1]["gravity_bc_droll_deg"])
    assert 1.0 < last_roll < 3.0
    assert math.degrees(math.asin(on_seen[-1][1][1])) - math.degrees(math.asin(off_seen[-1][1][1])) == pytest.approx(
        last_roll, abs=0.01)
    assert any(a.tobytes() != b.tobytes() for a, b in zip(off_obs, on_obs))
    arrays = load_arrays(on_csv)
    assert arrays["gravity_bc"].shape == (len(rows), 3) and arrays["gravity_bc_delta_deg"].shape == (len(rows), 2)
    assert np.allclose(arrays["gravity_bc"], np.array([g for _, g in on_seen], dtype=np.float32))
    assert np.allclose(arrays["gravity"], np.array([g for _, g in off_seen], dtype=np.float32))
    assert np.allclose(arrays["gyro"], load_arrays(off_csv)["gyro"])


def test_full_deploy_meta_flag(ptr, monkeypatch, tmp_path, capsys):
    original = ptr.run_deploy

    def with_bc(policy_path, contract, args):
        args.gravity_bc = True
        return original(policy_path, contract, args)

    run = run_full_deploy(ptr, monkeypatch, tmp_path / "off")
    assert "gravity_bc" not in load_meta(run.path)
    assert not set(ptr.GRAVITY_BC_COLUMNS) & set(read_csv(run.path)[0])
    monkeypatch.setattr(ptr, "run_deploy", with_bc)
    run = run_full_deploy(ptr, monkeypatch, tmp_path / "on")
    assert load_meta(run.path)["gravity_bc"] == ptr.GravityBC.config()
    rows = read_csv(run.path)
    assert list(rows[0])[-len(ptr.GRAVITY_BC_COLUMNS) - 1:-1] == list(ptr.GRAVITY_BC_COLUMNS)
    assert rows[-1]["gravity_bc_ready"] == "1"


class ReadJoints:
    label = "fake"

    def __init__(self, settings, notes):
        contract = make_contract()
        from robonex_common.joints import JOINT_BY_MODEL_NAME
        self.positions = {JOINT_BY_MODEL_NAME[name].motor_id: (float(contract.action_offsets[i]), 0.0)
                          for i, name in enumerate(contract.joint_order)}

    def start(self):
        pass

    def stop(self):
        pass

    def snapshot(self):
        return dict(self.positions)

    def rate_text(self):
        return "fake"


def run_read(module, monkeypatch, path, gravity_bc):
    clock = FakeClock()
    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(module, "confirm", lambda prompt: None)
    monkeypatch.setattr(module, "ReadOnlyJointSource", ReadJoints)

    def start(self, calibrate):
        self.driver = TiltedImuDriver(clock)
        self.status = "ready"
        return True

    monkeypatch.setattr(module.ImuSource, "start", start)
    monkeypatch.setattr(module.ImuSource, "stop", lambda self: None)
    FakeSession.instances.clear()
    args = SimpleNamespace(vx=0.0, vy=0.0, wz=0.0, duration=2.0, telemetry=path, gravity_bc=gravity_bc)
    assert module.run_read(Path("policy.onnx"), make_contract(), args) == 0
    return read_csv(path)


def test_read_mode(ptr, monkeypatch, tmp_path, capsys):
    off = run_read(ptr, monkeypatch, tmp_path / "off_read.csv", False)
    assert "BC gravity" not in capsys.readouterr().out
    on = run_read(ptr, monkeypatch, tmp_path / "on_read.csv", True)
    printed = capsys.readouterr().out
    assert "BC gravity (policy)" in printed and "No motor was enabled" in printed
    assert list(on[0]) == list(off[0]) + list(ptr.GRAVITY_BC_COLUMNS)
    assert len(on) == len(off) >= 90
    assert [r["gyro_x"] for r in on] == [r["gyro_x"] for r in off]
    assert [r["gravity_y"] for r in on] == [r["gravity_y"] for r in off]
    window = ptr.GravityBC.WINDOW
    assert [r["gravity_bc_ready"] for r in on] == ["0"] * (window - 1) + ["1"] * (len(on) - window + 1)
    assert float(on[-1]["gravity_bc_droll_deg"]) > 0.5
    arrays = load_arrays(tmp_path / "on_read.csv")
    assert arrays["gravity_bc"].shape == (len(on), 3)
    assert "gravity_bc" not in load_arrays(tmp_path / "off_read.csv")


def reference_bc(g, f, window=40, tau=1.0, dt=0.02):
    cg = np.cumsum(np.r_[np.zeros((1, 3)), g], 0)
    cf = np.cumsum(np.r_[np.zeros((1, 3)), f], 0)
    out = np.empty_like(g)
    dl = np.zeros(3)
    a = dt / (tau + dt)
    for k in range(len(g)):
        if k + 1 >= window:
            j = k + 1 - window
            gm = (cg[k + 1] - cg[j]) / window
            fm = (cf[k + 1] - cf[j]) / window
            dl = dl + a * (np.cross(gm / np.linalg.norm(gm), -fm / np.linalg.norm(fm)) - dl)
        v = g[k] + np.cross(dl, g[k])
        out[k] = v / np.linalg.norm(v)
    return out


def load_logged(run):
    path = HW_DIR / f"{run}_live_telemetry.csv"
    if not path.exists():
        pytest.skip(f"{path} not present")
    names = ("t_s", "gyro_x", "gravity_x", "gravity_y", "gravity_z", "acc_x", "acc_y", "acc_z")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    def value(row, name):
        try:
            return float(row[name])
        except (TypeError, ValueError):
            return math.nan

    data = np.array([[value(row, name) for name in names] for row in rows])
    data = data[np.all(np.isfinite(data), axis=1)]
    return data[:, 2:5], data[:, 5:8]


@pytest.mark.parametrize("run", LOGGED_RUNS)
def test_reproduces_e_bc_on_logged_runs(ptr, run):
    g, f = load_logged(run)
    assert len(g) > 2000
    bc = ptr.GravityBC()
    out = np.array([bc.update(gk, fk) for gk, fk in zip(g, f)])
    assert np.max(np.abs(out - reference_bc(g, f))) <= 1e-6
    if not E_BC.exists():
        pytest.skip(f"{E_BC} not present")
    sys.path.insert(0, str(E_BC.parent))
    try:
        spec = importlib.util.spec_from_file_location("e_bc", E_BC)
        e_bc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(e_bc)
    finally:
        sys.path.remove(str(E_BC.parent))
    d = {"gravity_x": g[:, 0], "gravity_y": g[:, 1], "gravity_z": g[:, 2],
         "acc_x": f[:, 0], "acc_y": f[:, 1], "acc_z": f[:, 2]}
    e_out = e_bc.bc(d, 40, 1.0)
    assert np.max(np.abs(out[600:] - e_out[600:])) <= 1e-6
    deltas = []
    bc.reset()
    for gk, fk in zip(g, f):
        bc.update(gk, fk)
        deltas.append(bc.delta_deg())
    assert np.max(np.abs(deltas)) < ptr.GravityBC.BOUND_DEG


def test_warmup_and_reset(ptr):
    bc = ptr.GravityBC()
    gravity = (0.01, 0.02, -0.9997)
    force = (0.3, -0.5, 9.7)
    n = math.sqrt(sum(v * v for v in gravity))
    for k in range(bc.WINDOW - 1):
        out = bc.update(gravity, force)
        assert bc.delta == (0.0, 0.0, 0.0) and not bc.ready
        assert out == pytest.approx(tuple(v / n for v in gravity), abs=1e-15)
    assert bc.update(gravity, None) == pytest.approx(tuple(v / n for v in gravity), abs=1e-15)
    assert bc.update(gravity, (math.nan, 0.0, 9.8)) and not bc.ready
    bc.update(gravity, force)
    assert bc.ready
    fn = math.sqrt(sum(v * v for v in force))
    u = tuple(-v / fn for v in force)
    gm = tuple(v / n for v in gravity)
    error = (gm[1] * u[2] - gm[2] * u[1], gm[2] * u[0] - gm[0] * u[2], gm[0] * u[1] - gm[1] * u[0])
    assert bc.delta == pytest.approx(tuple(bc.alpha * e for e in error), abs=1e-15)
    assert bc.alpha == pytest.approx(0.02 / 1.02)
    bc.reset()
    assert bc.delta == (0.0, 0.0, 0.0) and not bc.ready and bc.corrected is None
    assert bc.cells() == [""] * len(ptr.GRAVITY_BC_COLUMNS)
    assert ptr.gravity_bc_cells(None) == [""] * len(ptr.GRAVITY_BC_COLUMNS)


def test_bound_per_axis_and_no_windup(ptr):
    bc = ptr.GravityBC()
    level = (0.0, 0.0, -1.0)
    roll, pitch = math.radians(25.0), math.radians(-20.0)
    tilted = (9.81 * math.sin(pitch), -9.81 * math.sin(roll), 9.81)
    bound = math.radians(ptr.GravityBC.BOUND_DEG)
    for _ in range(1000):
        bc.update(level, tilted)
        assert abs(bc.delta[0]) <= bound and abs(bc.delta[1]) <= bound
    assert bc.delta[0] == bound and bc.delta[1] == -bound
    assert bc.delta_deg() == pytest.approx((8.0, -8.0))
    out = bc.corrected
    assert math.degrees(math.atan2(out[1], -out[2])) == pytest.approx(8.0, abs=0.2)
    for _ in range(bc.WINDOW):
        bc.update(level, (0.0, 0.0, 9.81))
    assert abs(bc.delta[0]) <= bound * (1.0 - bc.alpha) + 1e-12
    for _ in range(500):
        bc.update(level, (0.0, 0.0, 9.81))
    assert max(abs(d) for d in bc.delta) < math.radians(0.01)


def test_parse_args(ptr, tmp_path):
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"")
    assert ptr.parse_args(["--policy", str(policy)]).gravity_bc is False
    assert ptr.parse_args(["--policy", str(policy), "--gravity-bc"]).gravity_bc is True
    args = ptr.parse_args(["--policy", str(policy), "--gravity-bc", "--read", "--telemetry"])
    assert args.gravity_bc and args.read and args.telemetry.name.endswith("_read_telemetry.csv")
    args = ptr.parse_args(["--policy", str(policy), "--gravity-bc", "--keyboard"])
    assert args.gravity_bc and args.keyboard


def test_refused_for_a_policy_without_gravity_or_at_another_rate(ptr, monkeypatch, tmp_path, capsys):
    assert ptr.gravity_bc_reason(make_contract()) is None
    no_gravity = SimpleNamespace(observation_terms=("joint_pos_rel:12x5", "imu_ang_vel:3x5"), policy_hz=50.0)
    other_rate = SimpleNamespace(observation_terms=("projected_gravity:3x5",), policy_hz=100.0)
    assert "projected_gravity" in ptr.gravity_bc_reason(no_gravity)
    assert "100 Hz" in ptr.gravity_bc_reason(other_rate)
    called = []
    monkeypatch.setattr(ptr, "run_read", lambda *a: called.append("read") or 0)
    monkeypatch.setattr(ptr, "run_deploy", lambda *a: called.append("deploy") or 0)
    for contract in (no_gravity, other_rate):
        monkeypatch.setattr(ptr, "resolve_contract", lambda path, c=contract: c)
        for read in (True, False):
            assert ptr.run(SimpleNamespace(policy=tmp_path / "p.onnx", read=read, gravity_bc=True)) == 1
            assert ptr.run(SimpleNamespace(policy=tmp_path / "p.onnx", read=read, gravity_bc=False)) == 0
    assert called == ["read", "deploy"] * 2
    assert "--gravity-bc" in capsys.readouterr().out
