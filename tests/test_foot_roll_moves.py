import math

import pytest

import bench
import mujoco_to_real
import real_to_mujoco
import safety
from robonex_common.foot_roll import foot_roll
from robonex_common.joints import JOINT_BY_MODEL_NAME
from robonex_common.models import ROBOT_MODELS
from test_policy_to_real_timing import POLICY_TO_REAL, FakeClock, FakeJoints, FakeMotor, load_module

PROFILE = ROBOT_MODELS["ver2_edu"]
ROLL = PROFILE.foot_roll
LIMITS = PROFILE.joint_limits_by_id()
PAIRS = [(JOINT_BY_MODEL_NAME[u].motor_id, JOINT_BY_MODEL_NAME[l].motor_id, float(s)) for u, l, s in ROLL.pairs]
ANKLE_IDS = [mid for upper, lower, _ in PAIRS for mid in (upper, lower)]


def leaving_start(upper_deg=49.8, lower_deg=-18.3):
    starts = {}
    for upper, lower, sign in PAIRS:
        starts[upper], starts[lower] = sign * math.radians(upper_deg), sign * math.radians(lower_deg)
    return starts


def roll_deg(positions, pair):
    upper, lower, sign = pair
    return abs(math.degrees(foot_roll(sign * positions[upper], sign * positions[lower], ROLL.coeffs)))


def peak_roll(history):
    return max(roll_deg(positions, pair) for positions in history for pair in PAIRS)


class IdealMotor:
    def __init__(self, motor_id, position, clock, history):
        self.motor_id = motor_id
        self.clock = clock
        self.history = history
        self.last_position = position
        self.last_velocity = 0.0
        self.last_torque = 0.0
        self.last_temp = 30.0
        self.last_fault = 0
        self.last_mode_status = safety.MODE_RUNNING
        self.last_feedback_time = clock.now
        self.velocities = []

    def control(self, pos, vel, kp, kd, torque=0.0):
        self.last_position = pos
        self.last_velocity = vel
        self.last_feedback_time = self.clock.now
        self.velocities.append(vel)
        if self.history and self.motor_id not in self.history[-1]:
            self.history[-1][self.motor_id] = pos
        else:
            self.history.append({self.motor_id: pos})


def zero_move(monkeypatch, starts, roll_pairs):
    clock = FakeClock()
    monkeypatch.setattr(mujoco_to_real, "time", clock)
    history = []
    motors = {mid: IdealMotor(mid, starts[mid], clock, history) for mid in starts}
    args = type("Args", (), {})()
    args.rate, args.max_speed, args.max_accel, args.zero_tolerance_deg = 100.0, 0.10, 0.25, 3.0
    args.feedback_timeout, args.overspeed, args.max_error_deg, args.max_temp = 0.30, 2.0, 25.0, 70.0
    args.kp = {mid: 20.0 for mid in starts}
    args.kd = {mid: 1.0 for mid in starts}
    final = mujoco_to_real.move_to_zero(motors, {}, starts, LIMITS, args, roll_pairs, ROLL)
    return final, history, motors


def test_zero_move_leaves_the_band_without_the_projection(monkeypatch, capsys):
    _, history, _ = zero_move(monkeypatch, leaving_start(), ())
    assert roll_deg(leaving_start(), PAIRS[0]) < 12.0
    assert peak_roll(history) > 13.0


def test_zero_move_stays_in_the_band(monkeypatch, capsys):
    final, history, motors = zero_move(monkeypatch, leaving_start(), PAIRS)
    assert peak_roll(history) <= math.degrees(ROLL.limit) + 1e-6
    assert all(final[mid] == pytest.approx(0.0, abs=1e-12) for mid in ANKLE_IDS)
    assert max(abs(v) for motor in motors.values() for v in motor.velocities) <= 0.10 * 1.15


def test_zero_move_never_raises_an_out_of_band_start(monkeypatch, capsys):
    starts = leaving_start(49.8, -16.7)
    start_roll = roll_deg(starts, PAIRS[0])
    assert 12.0 < start_roll < 13.0
    final, history, motors = zero_move(monkeypatch, starts, PAIRS)
    assert peak_roll(history) <= start_roll + 1e-6
    assert all(final[mid] == pytest.approx(0.0, abs=1e-12) for mid in ANKLE_IDS)
    assert max(abs(v) for motor in motors.values() for v in motor.velocities) <= 0.10 * 1.15


def test_mujoco_to_real_blocks_a_start_roll_outside_the_band(tmp_path, monkeypatch, capsys):
    identity = tmp_path / "robot_model"
    identity.write_text("ver2_edu\n")
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(identity))
    args = mujoco_to_real.parse_args([])
    measured = {mid: 0.0 for mid in args.motor_ids}
    for upper, lower, sign in PAIRS:
        measured[upper], measured[lower] = sign * math.radians(20.0), sign * math.radians(20.0)
    assert roll_deg(measured, PAIRS[0]) > 13.0
    monkeypatch.setattr(mujoco_to_real, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(mujoco_to_real, "open_checked", lambda opener, *a: ({}, {mid: None for mid in args.motor_ids}, {}))
    monkeypatch.setattr(mujoco_to_real, "stop_idle_motors", lambda *a, **k: [])
    monkeypatch.setattr(mujoco_to_real, "inspect_zero_positions", lambda motors, tol, limits: (dict(measured), []))

    def never(*_a, **_k):
        raise AssertionError("must not get past the preflight check")

    monkeypatch.setattr(mujoco_to_real, "confirm_hardware", never)
    monkeypatch.setattr(mujoco_to_real, "enable_with_runtime_feedback", never)
    assert mujoco_to_real.main([]) == 1
    out = capsys.readouterr().out
    assert "start foot roll" in out and "outside the 12 deg band" in out


@pytest.fixture(scope="module")
def ptr():
    return load_module("policy_to_real_roll", POLICY_TO_REAL)


def approach(ptr, roll_pairs):
    clock = FakeClock()
    real_time = ptr.time
    ptr.time = clock
    try:
        log = []
        starts = leaving_start(49.0, -17.5)
        motors = {mid: FakeMotor(mid, starts[mid], log, safety.MODE_RUNNING) for mid in starts}
        joints = FakeJoints(motors, clock)
        joints.poll()
        commander = ptr.TargetCommander(motors, starts, ptr.SETTINGS)
        history = []

        class Recorder:
            def record(self, label, commands):
                history.append(dict(commands))

        ptr.approach_pose(commander, joints, {mid: 0.0 for mid in starts}, motors, LIMITS, ptr.SETTINGS, "default",
                          ptr.FaultMonitor(), recorder=Recorder(), roll_pairs=roll_pairs, roll=ROLL)
    finally:
        ptr.time = real_time
    return history, log, commander


def test_approach_pose_leaves_the_band_without_the_projection(ptr, capsys):
    history, _, _ = approach(ptr, ())
    assert peak_roll(history) > 13.0


def test_approach_pose_stays_in_the_band(ptr, capsys):
    history, log, commander = approach(ptr, PAIRS)
    assert peak_roll(history) <= math.degrees(ROLL.limit) + 1e-6
    assert all(commander.commands[mid] == pytest.approx(0.0, abs=1e-12) for mid in ANKLE_IDS)
    assert max(abs(entry[2]) for entry in log) <= ptr.SETTINGS.approach_max_speed * 1.15


def test_require_robot_model_honours_the_identity_env(tmp_path, monkeypatch, capsys):
    identity = tmp_path / "robot_model"
    identity.write_text("ver2_edu\n")
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(identity))
    safety.require_robot_model("ver2_edu")
    assert str(identity) in capsys.readouterr().out
    assert bench.attached_robot_model() == "ver2_edu"
    identity.write_text("ver2_pro\n")
    with pytest.raises(RuntimeError, match="ver2_pro"):
        safety.require_robot_model("ver2_edu")


def test_real_to_mujoco_flags_multiturn_mechpos():
    limits = {1: (-math.radians(90.0), math.radians(90.0))}
    assert real_to_mujoco.position_state(math.radians(365.6), limits[1], 0.05)[1] == "MULTITURN"
    assert real_to_mujoco.position_state(math.radians(5.6), limits[1], 0.05)[1] == "OK"
    with pytest.raises(RuntimeError, match="zero_sta"):
        real_to_mujoco.validate_positions({1: math.radians(365.6)}, limits, 0.05)
