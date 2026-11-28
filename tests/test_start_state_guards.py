import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

import safety
from robonex_common.joints import JOINT_BY_MODEL_NAME
from test_policy_to_real_recording import load_meta, run_full_deploy
from test_policy_to_real_timing import make_contract, modules

DEG = math.pi / 180.0
CONTRACT = make_contract()
LEG_IDS = [JOINT_BY_MODEL_NAME[name].motor_id for name in CONTRACT.joint_order]


def preflight_position(name):
    return float(CONTRACT.action_offsets[CONTRACT.joint_order.index(name)]) + 0.05


class GateMotor:
    def __init__(self, motor_id, position):
        self.motor_id = motor_id
        self.position = position
        self.last_position = None
        self.last_mode_status = 0
        self.controls = []

    def stop(self):
        pass

    def write_run_mode_operation(self):
        pass

    def enable(self):
        pass

    def control(self, pos, vel, kp, kd, torque=0.0):
        self.controls.append(pos)
        self.last_mode_status = safety.MODE_RUNNING


class GateHub:
    def __init__(self, motors):
        self.motors = motors
        self.bus = SimpleNamespace(recv=lambda timeout=0.0: None)

    def wait_for(self, motor_id, timeout):
        motor = self.motors[motor_id]
        motor.last_position = motor.position
        return motor.position


def gate(positions, **kwargs):
    motors = {mid: GateMotor(mid, value) for mid, value in positions.items()}
    hub = GateHub(motors)
    hubs = {safety.channel_for_id(mid): hub for mid in motors}
    limits = {mid: (-math.pi, math.pi) for mid in motors}
    enabled = []
    try:
        starts, _ = safety.enable_with_runtime_feedback(motors, hubs, 10.0, 1.0, limits, enabled_out=enabled, **kwargs)
    except RuntimeError as error:
        return motors, enabled, error
    return motors, enabled, starts


def test_enable_refuses_type2_more_than_5_deg_from_the_recheck():
    motors, enabled, error = gate({1: 8.0 * DEG}, reference={1: 0.0}, max_deviation=5.0 * DEG)
    assert isinstance(error, RuntimeError)
    assert "ID 1 type-0x02 position at enable +8.00 deg" in str(error) and "by +8.00 deg (limit 5 deg)" in str(error)
    assert error.enabled_ids == [1] and enabled == [1]
    assert motors[1].controls == []


def test_enable_passes_type2_2_deg_from_the_recheck():
    motors, _, starts = gate({1: 2.0 * DEG}, reference={1: 0.0}, max_deviation=5.0 * DEG)
    assert starts == {1: pytest.approx(2.0 * DEG)}
    assert motors[1].controls == [pytest.approx(2.0 * DEG)]


def test_enable_deviation_is_wrap_aware():
    _, _, starts = gate({1: -179.0 * DEG}, reference={1: 179.0 * DEG}, max_deviation=5.0 * DEG)
    assert starts == {1: pytest.approx(-179.0 * DEG)}
    _, _, error = gate({1: -170.0 * DEG}, reference={1: 179.0 * DEG}, max_deviation=5.0 * DEG)
    assert isinstance(error, RuntimeError) and "by +11.00 deg" in str(error)


def test_enable_refusal_on_a_later_motor_keeps_the_enabled_list_for_the_brake():
    motors, _, error = gate({1: 0.0, 2: -9.0 * DEG}, reference={1: 0.0, 2: 0.0}, max_deviation=5.0 * DEG)
    assert isinstance(error, RuntimeError) and str(error).startswith("ID 2 ")
    assert error.enabled_ids == [1, 2]
    assert motors[1].controls and not motors[2].controls


def test_enable_without_reference_keeps_todays_behaviour():
    _, _, starts = gate({1: 30.0 * DEG, 2: -40.0 * DEG})
    assert set(starts) == {1, 2}


def test_enable_refuses_a_motor_missing_from_the_reference():
    _, _, error = gate({1: 0.0, 2: 0.0}, reference={1: 0.0}, max_deviation=5.0 * DEG)
    assert isinstance(error, RuntimeError) and "ID 2 has no start reference" in str(error)


def test_enable_refuses_a_held_motor_outside_its_hold_window():
    motors, _, error = gate({26: 18.7 * DEG}, hold_targets={26: 0.0}, hold_window=10.0 * DEG)
    assert isinstance(error, RuntimeError)
    assert "ID 26 type-0x02 position at enable +18.70 deg is +18.70 deg from its hold pose" in str(error)
    assert "move it by hand to its hold pose" in str(error) and "--zero-head" in str(error)
    assert motors[26].controls == []
    _, _, starts = gate({26: 3.0 * DEG}, hold_targets={26: 0.0}, hold_window=10.0 * DEG)
    assert set(starts) == {26}
    _, _, starts = gate({26: 18.7 * DEG})
    assert set(starts) == {26}


def test_hold_start_blocks_window(capsys):
    blocks = safety.hold_start_blocks({26: 18.7 * DEG}, {26: 0.0}, 10.0 * DEG)
    assert len(blocks) == 1 and blocks[0].startswith("ID 26 start +18.70 deg")
    assert "move it by hand to its hold pose" in blocks[0] and "--zero-head" in blocks[0]
    assert "BLOCK" in capsys.readouterr().out
    assert safety.hold_start_blocks({26: 3.0 * DEG}, {26: 0.0}, 10.0 * DEG) == []
    assert safety.hold_start_blocks({26: -3.0 * DEG}, {26: 0.0}, 10.0 * DEG) == []
    assert safety.hold_start_blocks({26: 18.7 * DEG}, {}, 10.0 * DEG) == []


def _held(module):
    ids = module.held_motor_ids(CONTRACT.robot_model)
    assert ids
    return ids[0]


def test_deploy_refuses_a_held_motor_at_18p7_deg_in_the_preflight(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    monkeypatch.setattr(new, "SETTINGS", replace(new.SETTINGS, hold_head=True))
    held = _held(new)
    run = run_full_deploy(new, monkeypatch, tmp_path, start_overrides={held: 18.7 * DEG})
    out = capsys.readouterr().out
    assert run.error is not None and "Preflight safety check failed" in str(run.error)
    assert f"ID {held} start +18.70 deg" in str(run.error) and "--zero-head" in str(run.error)
    assert "confirm" not in run.events and "enable" not in run.events
    assert "BLOCK" in out


def test_deploy_passes_a_held_motor_at_3_deg_and_gives_enable_the_hold_window(modules, monkeypatch, tmp_path,
                                                                              capsys):
    _, new = modules
    monkeypatch.setattr(new, "SETTINGS", replace(new.SETTINGS, hold_head=True))
    held = _held(new)
    run = run_full_deploy(new, monkeypatch, tmp_path, start_overrides={held: 3.0 * DEG})
    capsys.readouterr()
    assert run.error is None
    kwargs = run.limits["enable_kwargs"]
    assert kwargs["hold_targets"] == {held: 0.0}
    assert kwargs["hold_window"] == pytest.approx(10.0 * DEG)


def test_deploy_refuses_a_held_motor_moved_after_the_preflight(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    monkeypatch.setattr(new, "SETTINGS", replace(new.SETTINGS, hold_head=True))
    held = _held(new)
    run = run_full_deploy(new, monkeypatch, tmp_path, recheck_overrides={held: 18.7 * DEG})
    capsys.readouterr()
    assert run.error is not None and "Start re-check after the confirmation failed" in str(run.error)
    assert f"ID {held} start +18.70 deg" in str(run.error)
    assert run.events.index("confirm") < run.events.index("brake") and "enable" not in run.events


def test_deploy_without_hold_has_no_hold_window(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    monkeypatch.setattr(new, "SETTINGS", replace(new.SETTINGS, hold_head=False))
    run = run_full_deploy(new, monkeypatch, tmp_path, start_overrides={})
    capsys.readouterr()
    assert run.error is None
    assert run.limits["enable_kwargs"]["hold_targets"] == {}


def test_deploy_recheck_refuses_an_ankle_moved_after_the_preflight(modules, monkeypatch, tmp_path, capsys):
    from robonex_common.models import robot_model

    _, new = modules
    profile = robot_model("ver2_edu")
    stops, commands = profile.stop_limits_by_id(), profile.joint_limits_by_id()
    upper = JOINT_BY_MODEL_NAME["l_ankle_upper_joint"].motor_id
    lower = JOINT_BY_MODEL_NAME["l_ankle_lower_joint"].motor_id
    moved = {upper: math.radians(30.0), lower: math.radians(-3.0)}
    assert commands[upper][1] < moved[upper] < stops[upper][1]
    run = run_full_deploy(new, monkeypatch, tmp_path, real_roll_pairs=True, recheck_overrides=moved)
    out = capsys.readouterr().out
    assert run.error is not None and "Start re-check after the confirmation failed" in str(run.error)
    assert f"ID {upper}: start +30.00 deg is outside the ankle command range" in str(run.error)
    assert "Start re-check after the confirmation (fresh mechPos" in out
    assert "confirm" in run.events and "enable" not in run.events
    end = load_meta(run.path)["end"]
    expected = math.degrees(safety.wrap_to_pi(moved[upper] - preflight_position("l_ankle_upper_joint")))
    assert end["start_recheck"]["minus_preflight_deg"][str(upper)] == pytest.approx(expected)
    assert "enable_minus_recheck_deg" not in end["start_recheck"]


def test_deploy_recheck_refuses_a_foot_roll_outside_the_band(modules, monkeypatch, tmp_path, capsys):
    from robonex_common.foot_roll import foot_roll
    from robonex_common.models import robot_model

    _, new = modules
    profile = robot_model("ver2_edu")
    commands = profile.joint_limits_by_id()
    upper = JOINT_BY_MODEL_NAME["l_ankle_upper_joint"].motor_id
    lower = JOINT_BY_MODEL_NAME["l_ankle_lower_joint"].motor_id
    sign = next(s for u, l, s in safety.roll_pairs_for(profile, LEG_IDS) if u == upper)
    moved = None
    for a in range(-40, 41):
        for b in range(-40, 41):
            pa, pb = math.radians(a), math.radians(b)
            inside = (commands[upper][0] <= pa <= commands[upper][1]
                      and commands[lower][0] <= pb <= commands[lower][1])
            if inside and abs(foot_roll(sign * pa, sign * pb, profile.foot_roll.coeffs)) > profile.foot_roll.limit + 3.0 * DEG:
                moved = {upper: pa, lower: pb}
                break
        if moved:
            break
    assert moved is not None
    run = run_full_deploy(new, monkeypatch, tmp_path, real_roll_pairs=True, recheck_overrides=moved)
    capsys.readouterr()
    assert run.error is not None and "Start re-check after the confirmation failed" in str(run.error)
    assert f"IDs {upper}/{lower}: start foot roll" in str(run.error)
    assert "enable" not in run.events


def test_deploy_gives_enable_the_fresh_read_and_records_the_deltas(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    hip = JOINT_BY_MODEL_NAME["l_hip_pitch_joint"].motor_id
    preflight = preflight_position("l_hip_pitch_joint")
    run = run_full_deploy(new, monkeypatch, tmp_path, recheck_overrides={hip: preflight + 2.0 * DEG})
    capsys.readouterr()
    assert run.error is None
    kwargs = run.limits["enable_kwargs"]
    assert kwargs["max_deviation"] == pytest.approx(5.0 * DEG)
    assert kwargs["reference"][hip] == pytest.approx(preflight + 2.0 * DEG)
    assert set(kwargs["reference"]) == set(LEG_IDS)
    assert run.events.index("confirm") < run.events.index("enable")
    recheck = load_meta(run.path)["end"]["start_recheck"]
    assert recheck["mech_pos_rad"][str(hip)] == pytest.approx(preflight + 2.0 * DEG)
    assert recheck["minus_preflight_deg"][str(hip)] == pytest.approx(2.0)
    assert recheck["enable_minus_recheck_deg"][str(hip)] == pytest.approx(-2.0)
    others = [v for k, v in recheck["minus_preflight_deg"].items() if k != str(hip)]
    assert len(others) == len(LEG_IDS) - 1 and all(v == pytest.approx(0.0) for v in others)
