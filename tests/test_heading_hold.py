import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_policy_to_real_timing import (
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


@pytest.fixture(scope="module")
def ptr():
    return load_module("policy_to_real_heading", POLICY_TO_REAL)


LEVEL = (0.0, 0.0, -1.0)


def quat_sample(yaw):
    return SimpleNamespace(orientation=SimpleNamespace(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2)))


def simulate(hold, disturbance, seconds, command=(0.2, 0.0, 0.0), dt=0.02):
    heading, now, history = 0.0, 0.0, []
    for _ in range(int(round(seconds / dt))):
        wz = hold.update(now, (0.0, 0.0, disturbance + hold.wz), LEVEL, None, command, active=True)
        history.append((now, heading, wz))
        heading += (disturbance + wz) * dt
        now += dt
    return history


def test_engages_only_on_straight_walking_command(ptr):
    hold = ptr.HeadingHold(1.0, 0.1, "gyro")
    for command, active, expected in (
        ((0.2, 0.0, 0.0), True, True),
        ((0.0, 0.1, 0.0), True, True),
        ((0.2, 0.0, 0.05), True, False),
        ((0.04, 0.0, 0.0), True, False),
        ((0.0, 0.0, 0.0), True, False),
        ((0.2, 0.0, 0.0), False, False),
    ):
        hold.release()
        hold.update(0.0, (0.0, 0.0, 0.0), LEVEL, None, command, active)
        assert hold.engaged is expected, command
        if not expected:
            assert hold.wz == 0.0 and hold.integral == 0.0 and hold.target is None


def test_pi_removes_a_steady_yaw_disturbance(ptr):
    disturbance = math.radians(0.45) + 0.02
    free = simulate(ptr.HeadingHold(0.0, 0.0, "gyro"), disturbance, 20.0)
    held = simulate(ptr.HeadingHold(1.0, 0.1, "gyro"), disturbance, 20.0)
    assert abs(free[-1][1]) > 0.4
    assert abs(held[-1][1]) < math.radians(0.5)
    assert held[-1][2] == pytest.approx(-disturbance, abs=2e-3)


def test_output_and_integral_limits(ptr):
    hold = ptr.HeadingHold(2.0, 0.5, "gyro")
    history = simulate(hold, 1.0, 10.0)
    assert max(abs(wz) for _, _, wz in history) <= hold.OUTPUT_LIMIT + 1e-12
    assert abs(hold.integral) <= hold.INTEGRAL_LIMIT + 1e-12
    assert hold.wz == pytest.approx(-hold.OUTPUT_LIMIT)


def test_integral_does_not_wind_up_while_saturated(ptr):
    hold = ptr.HeadingHold(2.0, 0.1, "gyro")
    simulate(hold, 0.5, 3.0)
    assert abs(hold.integral) < hold.INTEGRAL_LIMIT


def test_release_and_recapture_on_turn(ptr):
    hold = ptr.HeadingHold(1.0, 0.1, "gyro")
    hold.update(0.0, (0.0, 0.0, 0.0), LEVEL, None, (0.2, 0.0, 0.0), True)
    for k in range(1, 51):
        hold.update(0.02 * k, (0.0, 0.0, 0.3), LEVEL, None, (0.2, 0.0, 0.1), True)
        assert not hold.engaged
    hold.update(1.02, (0.0, 0.0, 0.0), LEVEL, None, (0.2, 0.0, 0.0), True)
    assert hold.engaged
    assert hold.target == pytest.approx(0.3)
    assert hold.error == pytest.approx(0.0)


def test_gyro_heading_uses_world_vertical(ptr):
    tilt = math.radians(20.0)
    gravity = (0.0, math.sin(tilt), -math.cos(tilt))
    gyro = (0.0, -0.5 * math.sin(tilt), 0.5 * math.cos(tilt))
    assert ptr.HeadingHold.world_yaw_rate(gyro, gravity) == pytest.approx(0.5)
    assert ptr.HeadingHold.world_yaw_rate((0.0, 0.0, 0.2), (0.0, 0.0, 0.0)) == pytest.approx(0.2)


def test_quat_source(ptr):
    hold = ptr.HeadingHold(1.0, 0.0, "quat")
    hold.update(0.0, (0.0, 0.0, 0.0), LEVEL, quat_sample(0.4), (0.2, 0.0, 0.0), True)
    assert hold.engaged and hold.target == pytest.approx(0.4)
    wz = hold.update(0.02, (0.0, 0.0, 0.0), LEVEL, quat_sample(0.5), (0.2, 0.0, 0.0), True)
    assert wz == pytest.approx(-0.1)
    hold.update(0.04, (0.0, 0.0, 0.0), LEVEL, None, (0.2, 0.0, 0.0), True)
    assert not hold.engaged and hold.heading_quat is None


def test_error_wraps_across_pi(ptr):
    hold = ptr.HeadingHold(1.0, 0.0, "quat")
    hold.update(0.0, (0.0, 0.0, 0.0), LEVEL, quat_sample(math.pi - 0.01), (0.2, 0.0, 0.0), True)
    wz = hold.update(0.02, (0.0, 0.0, 0.0), LEVEL, quat_sample(-math.pi + 0.01), (0.2, 0.0, 0.0), True)
    assert wz == pytest.approx(-0.02, abs=1e-9)


def test_cells_match_columns(ptr):
    hold = ptr.HeadingHold(1.0, 0.1, "gyro")
    assert len(hold.cells()) == len(ptr.HEADING_COLUMNS)
    assert ptr.heading_cells(None) == [""] * len(ptr.HEADING_COLUMNS)
    assert ptr.STEP_EXTRA_COLUMNS[-len(ptr.HEADING_COLUMNS):] == ptr.HEADING_COLUMNS


def test_parse_args(ptr, tmp_path):
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"")
    args = ptr.parse_args(["--policy", str(policy)])
    assert (args.heading_hold, args.heading_kp, args.heading_ki, args.heading_source) == (False, 1.0, 0.1, "gyro")
    args = ptr.parse_args(["--policy", str(policy), "--heading-hold", "--heading-source", "quat"])
    assert args.heading_hold and args.heading_source == "quat"
    for bad in (["--heading-kp", "4"], ["--heading-kp", "-1"], ["--heading-ki", "1"], ["--heading-kp", "nan"],
                ["--heading-hold", "--read"]):
        with pytest.raises(SystemExit):
            ptr.parse_args(["--policy", str(policy)] + bad)


def run_loop(module, telemetry_path, heading_hold, duration=4.0, wz=0.0):
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
        observe = runner.observation

        def spy(*a, **k):
            seen.append(runner.velocity_command.copy())
            return observe(*a, **k)

        runner.observation = spy
        log = []
        motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
        starts = {JOINT_BY_MODEL_NAME[name].motor_id: float(contract.action_offsets[i])
                  for i, name in enumerate(contract.joint_order)}
        motors = {mid: FakeMotor(mid, starts[mid], log, safety.MODE_RUNNING) for mid in motor_ids}
        imu = module.ImuSource(module.SETTINGS, [])
        imu.driver = FakeImuDriver(clock)
        imu.status = "ready"
        commander = module.TargetCommander(motors, starts, module.SETTINGS)
        limits = {mid: robot_model("ver2_edu").joint_limits_by_id()[mid] for mid in motor_ids}
        thermal = module.ThermalLoad({mid: RATED_TORQUE[JOINT_BY_ID[mid].motor_model] for mid in motor_ids})
        args = SimpleNamespace(duration=duration, telemetry=telemetry_path, vx=0.2, vy=0.0, wz=wz, keyboard=False,
                               heading_hold=heading_hold, heading_kp=1.0, heading_ki=0.1, heading_source="gyro")
        runner.policy_path = Path("policy.onnx")
        runner.velocity_command[:] = (args.vx, args.vy, args.wz)
        module.policy_loop(runner, commander, FakeJoints(motors, clock), imu, motors, limits, contract, args, [],
                           module.LoopStats(), module.FaultMonitor(), thermal, None)
        final = runner.velocity_command.copy()
    finally:
        module.time = real_time
    return log, np.array(seen), final


def test_policy_loop_feeds_heading_wz_only_to_the_policy(ptr, tmp_path, capsys):
    path = tmp_path / "held.csv"
    _, seen, final = run_loop(ptr, path, True)
    hold_end = ptr.SETTINGS.ramp_seconds + ptr.SETTINGS.command_hold_seconds
    rows = read_csv(path)
    assert set(ptr.HEADING_COLUMNS) <= set(rows[0])
    times = np.array([float(r["t_s"]) for r in rows])
    engaged = np.array([r["heading_engaged"] for r in rows])
    assert set(engaged[times < hold_end - 0.05]) == {"0"}
    assert set(engaged[times > hold_end + 0.05]) == {"1"}
    logged_wz = np.array([float(r["heading_wz"]) for r in rows])
    assert np.all(seen[: len(rows), 2] == pytest.approx(np.where(engaged == "1", logged_wz, 0.0), abs=1e-5))
    assert np.max(np.abs(seen[:, 2])) > 0.0
    assert np.max(np.abs(seen[:, 2])) <= ptr.HeadingHold.OUTPUT_LIMIT
    assert all(float(r["cmd_wz"]) == 0.0 for r in rows)
    policy_wz = np.array([float(r["policy_wz"]) for r in rows])
    assert np.all(policy_wz == pytest.approx(seen[: len(rows), 2], abs=1e-5))
    assert tuple(final) == (0.2, 0.0, 0.0)


def test_policy_wz_logs_the_operator_turn_while_released(ptr, tmp_path, capsys):
    path = tmp_path / "turn.csv"
    _, seen, final = run_loop(ptr, path, True, wz=0.1)
    rows = read_csv(path)
    hold_end = ptr.SETTINGS.ramp_seconds + ptr.SETTINGS.command_hold_seconds
    late = [r for r in rows if float(r["t_s"]) > hold_end + 0.05]
    assert late and all(r["heading_engaged"] == "0" for r in late)
    assert all(float(r["heading_wz"]) == 0.0 for r in late)
    assert all(float(r["policy_wz"]) == pytest.approx(0.1, abs=1e-6) for r in late)
    assert float(final[2]) == pytest.approx(0.1, abs=1e-6)


def test_keyboard_opposite_turns_return_to_exact_zero(ptr, monkeypatch):
    keys = iter([b"a", b"d", b"aad", b"d"])
    pending = []

    def fake_select(r, w, x, timeout):
        if not pending:
            try:
                pending.append(next(keys))
            except StopIteration:
                return ([], [], [])
        return (r, [], [])

    def fake_read(fd, n):
        return pending.pop(0)

    monkeypatch.setattr(ptr.select, "select", fake_select)
    monkeypatch.setattr(ptr.os, "read", fake_read)
    keyboard = ptr.KeyboardCommand(ptr.COMMAND_LIMITS)
    keyboard.active, keyboard.fd = True, 0
    command = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    keyboard.poll(command)
    assert float(command[2]) == 0.0
    hold = ptr.HeadingHold(1.0, 0.1, "gyro")
    hold.update(0.0, (0.0, 0.0, 0.0), LEVEL, None, command, True)
    assert hold.engaged and hold.integral == 0.0 and hold.wz == 0.0


def test_policy_loop_without_heading_hold_is_unchanged(ptr, tmp_path, capsys):
    off_log, off_seen, _ = run_loop(ptr, tmp_path / "off.csv", False)
    assert np.all(off_seen[:, 2] == 0.0)
    rows = read_csv(tmp_path / "off.csv")
    assert all(r["heading_engaged"] == "" for r in rows)


def test_ankle_gain_scale(ptr, tmp_path):
    kp, kd = ptr.resolve_gains(1.0, 1.5)
    base_kp, base_kd = ptr.resolve_gains(1.0)
    for name, spec in ptr.JOINT_BY_MODEL_NAME.items():
        factor = 1.5 if "ankle" in name else 1.0
        assert kp[spec.motor_id] == pytest.approx(base_kp[spec.motor_id] * factor)
        assert kd[spec.motor_id] == pytest.approx(base_kd[spec.motor_id] * factor)
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"")
    assert ptr.parse_args(["--policy", str(policy)]).ankle_gain_scale == 1.0
    assert ptr.parse_args(["--policy", str(policy), "--ankle-gain-scale", "1.5"]).ankle_gain_scale == 1.5
    for bad in ("0.5", "2.5", "nan"):
        with pytest.raises(SystemExit):
            ptr.parse_args(["--policy", str(policy), "--ankle-gain-scale", bad])
