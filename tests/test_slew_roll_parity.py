import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robonex_common.joints import JOINT_BY_MODEL_NAME
from robonex_common.runtime import ActionPipeline
from test_policy_to_real_timing import POLICY_TO_REAL, FakeMotor, load_module, make_contract

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/sim_to_sim/isaac_to_mujoco.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def itm():
    return load(SOURCE, "slew_roll_parity_itm")


@pytest.fixture(scope="module")
def ptr():
    return load_module("slew_roll_parity_ptr", POLICY_TO_REAL)


@pytest.fixture(scope="module")
def contract():
    return make_contract()


def raw_actions(contract, steps, seed):
    order = list(contract.joint_order)
    pairs = [(order.index(upper), order.index(lower), sign) for upper, lower, sign in contract.foot_roll_pairs]
    rng = np.random.default_rng(seed)
    actions = []
    for step in range(steps):
        action = rng.uniform(-1.0, 1.0, contract.action_size).astype(np.float32)
        if step % 3 == 0:
            for upper, lower, sign in pairs:
                kick = 4.0 if step % 2 else -4.0
                action[upper] = kick
                action[lower] = -kick
        actions.append(action)
    return actions


def judge_targets(itm, contract, actions, period):
    pipeline = ActionPipeline(contract)
    limiters = [itm.AxisLimiter(float(p)) for p in contract.action_offsets]
    out, moved = [], 0
    for action in actions:
        _, targets = pipeline.apply(action, max_raw_action=20.0)
        moved += int(itm.slew_targets(limiters, targets, pipeline, period))
        out.append(np.array(targets, dtype=np.float64))
    return out, moved, limiters, pipeline


def deploy_targets(ptr, contract, actions, period):
    pipeline = ActionPipeline(contract)
    motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
    start = {mid: float(p) for mid, p in zip(motor_ids, contract.action_offsets)}
    log = []
    motors = {mid: FakeMotor(mid, start[mid], log, True) for mid in motor_ids}
    commander = ptr.TargetCommander(motors, start, ptr.SETTINGS)
    reproject = ptr.roll_reprojector(pipeline, motor_ids)
    out = []
    for action in actions:
        _, targets = pipeline.apply(action, max_raw_action=20.0)
        commander.send({mid: float(t) for mid, t in zip(motor_ids, targets)}, period,
                       ptr.SETTINGS.policy_max_speed, ptr.SETTINGS.policy_max_accel,
                       use_velocity_target=False, reproject=reproject, align=False)
        out.append(np.array([commander.commands[mid] for mid in motor_ids], dtype=np.float64))
    return out, commander


def test_judge_slew_constants_match_deploy(itm, ptr):
    assert itm.SLEW_MAX_SPEED == ptr.SETTINGS.policy_max_speed
    assert itm.SLEW_MAX_ACCEL == ptr.SETTINGS.policy_max_accel


def test_judge_slew_and_roll_reprojection_match_deploy(itm, ptr, contract):
    assert contract.foot_roll_limit > 0.0 and contract.foot_roll_pairs
    period = 1.0 / contract.policy_hz
    actions = raw_actions(contract, 300, seed=7)
    judge, moved, _, _ = judge_targets(itm, contract, actions, period)
    deploy, commander = deploy_targets(ptr, contract, actions, period)
    assert moved > 0
    assert moved == commander.roll_reprojected_count
    for step, (a, b) in enumerate(zip(judge, deploy)):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1.0e-6, err_msg=f"step {step}")


def test_reprojected_targets_stay_in_the_roll_band(itm, contract):
    period = 1.0 / contract.policy_hz
    judge, moved, limiters, pipeline = judge_targets(itm, contract, raw_actions(contract, 200, seed=3), period)
    assert moved > 0
    for targets in judge:
        projected, changed = pipeline.clip_roll(targets)
        assert not changed
        np.testing.assert_array_equal(projected, targets)
    for limiter, final in zip(limiters, judge[-1]):
        assert limiter.position == pytest.approx(final, abs=1.0e-6)


def test_limiter_velocity_follows_the_reprojected_step(itm, contract):
    period = 1.0 / contract.policy_hz
    pipeline = ActionPipeline(contract)
    limiters = [itm.AxisLimiter(float(p)) for p in contract.action_offsets]
    for action in raw_actions(contract, 50, seed=11):
        previous = [limiter.position for limiter in limiters]
        _, targets = pipeline.apply(action, max_raw_action=20.0)
        itm.slew_targets(limiters, targets, pipeline, period)
        for limiter, before in zip(limiters, previous):
            assert limiter.velocity == pytest.approx((limiter.position - before) / period, abs=1.0e-9)


def test_without_roll_pairs_only_the_slew_runs(itm):
    class NoRoll:
        roll_pairs = []

        def clip_roll(self, targets):
            raise AssertionError("clip_roll must not run without roll pairs")

    limiters = [itm.AxisLimiter(0.0), itm.AxisLimiter(0.0)]
    targets = np.array([1.0, -1.0], dtype=np.float32)
    assert itm.slew_targets(limiters, targets, NoRoll(), 0.02) is False
    expected = itm.AxisLimiter(0.0).step(1.0, 0.02, itm.SLEW_MAX_SPEED, itm.SLEW_MAX_ACCEL)[0]
    assert targets[0] == pytest.approx(expected)
    assert targets[1] == pytest.approx(-expected)


class SteppingClock:
    def __init__(self, step_ns):
        self.now_ns = 5_000_000_000
        self.step_ns = step_ns

    def monotonic_ns(self):
        self.now_ns += self.step_ns
        return self.now_ns

    def monotonic(self):
        return self.now_ns * 1.0e-9


def test_commander_stamps_first_and_last_frame(ptr, contract, monkeypatch):
    motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
    start = {mid: float(p) for mid, p in zip(motor_ids, contract.action_offsets)}
    clock = SteppingClock(1_000_000)
    log = []

    class StampingMotor(FakeMotor):
        def control(self, pos, vel, kp, kd, torque=0.0):
            clock.now_ns += 250_000
            super().control(pos, vel, kp, kd, torque)

    motors = {mid: StampingMotor(mid, start[mid], log, True) for mid in motor_ids}
    commander = ptr.TargetCommander(motors, start, ptr.SETTINGS)
    assert commander.tx_first_ns is None and commander.tx_last_ns is None
    monkeypatch.setattr(ptr, "time", clock)
    tick = clock.monotonic()
    commander.send(start, 0.02, ptr.SETTINGS.policy_max_speed, ptr.SETTINGS.policy_max_accel,
                   use_velocity_target=False, align=False)
    assert len(log) == len(motor_ids)
    assert commander.tx_last_ns - commander.tx_first_ns == len(motor_ids) * 250_000 + 1_000_000
    first_ms = ptr._since_tick_ms(tick, commander.tx_first_ns)
    last_ms = ptr._since_tick_ms(tick, commander.tx_last_ns)
    assert first_ms == pytest.approx(1.0)
    assert last_ms == pytest.approx(2.0 + len(motor_ids) * 0.25)
    commander.reset_stats()
    assert commander.tx_first_ns is None and commander.tx_last_ns is None


def test_step_extra_cells_append_send_timing(ptr):
    assert ptr.LOOP_EXTRA_COLUMNS[-3:] == ("obs_ms", "tx_first_ms", "tx_last_ms")
    commander = SimpleNamespace(slew_lag_step_max=0.0, roll_reprojected_count=0,
                                tx_first_ns=10_002_000_000, tx_last_ns=10_005_500_000)
    cells = ptr.step_extra_cells(1.0, 2.0, 3.0, commander, SimpleNamespace(), None, None,
                                 tick=10.0, obs_ns=10_001_250_000)
    loop = cells[:len(ptr.LOOP_EXTRA_COLUMNS)]
    assert loop[-3:] == [pytest.approx(1.25), pytest.approx(2.0), pytest.approx(5.5)]
    blank = ptr.step_extra_cells(1.0, 2.0, 3.0, SimpleNamespace(slew_lag_step_max=0.0), SimpleNamespace(),
                                 None, None)
    assert blank[len(ptr.LOOP_EXTRA_COLUMNS) - 3:len(ptr.LOOP_EXTRA_COLUMNS)] == ["", "", ""]


def test_policy_session_is_single_threaded_without_spinning(ptr):
    options = ptr.policy_session_options()
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.config == {"session.intra_op.allow_spinning": "0", "session.inter_op.allow_spinning": "0"}


def test_single_threaded_session_matches_default_bitwise(ptr, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    policies = sorted((ROOT / "policies").glob("*/policy.onnx"))
    if not policies:
        pytest.skip("no exported policy under policies/")
    policy = policies[-1]
    monkeypatch.setattr(ptr, "ort", ort)
    options = ptr.policy_session_options()
    assert options.intra_op_num_threads == 1 and options.inter_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert options.get_session_config_entry("session.inter_op.allow_spinning") == "0"
    default = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    single = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    name = default.get_inputs()[0].name
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.normal(0.0, 1.0, (1, default.get_inputs()[0].shape[1])).astype(np.float32)
        assert default.run(None, {name: x})[0].tobytes() == single.run(None, {name: x})[0].tobytes()
