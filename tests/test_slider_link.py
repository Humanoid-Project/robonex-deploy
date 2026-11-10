import math
import time

import pytest

import bench
import mujoco_to_real
import safety
import slider_link
import slider_sender
from test_foot_roll_moves import IdealMotor
from test_policy_to_real_timing import FakeClock

VARIANT = "ver2_edu"
IDS = [1, 2]
NAMES = {1: "l_hip_yaw", 2: "l_hip_pitch"}
LIMITS = {mid: (-1.0, 1.0) for mid in IDS}


def test_target_packet_round_trip():
    payload = slider_link.encode_targets(VARIANT, "abc", 7, {"l_hip_yaw": 0.25})
    assert slider_link.decode_targets(payload, VARIANT) == ("abc", 7, {"l_hip_yaw": 0.25})


@pytest.mark.parametrize("payload, message", [
    (b"\xff", "not a JSON packet"),
    (b'{"protocol": 1, "kind": "status"}', "not a slider target packet"),
    (slider_link.encode_targets("ver2_pro", "a", 1, {}), "does not match this robot"),
    (b'{"protocol":1,"kind":"targets","variant":"ver2_edu","session":"a","seq":1,"targets":{"x":NaN}}', "not a finite number"),
    (b'{"protocol":1,"kind":"targets","variant":"ver2_edu","session":"a","seq":1,"targets":{"x":true}}', "not a finite number"),
    (b'{"protocol":1,"kind":"targets","variant":"ver2_edu","seq":1,"targets":{}}', "missing session"),
])
def test_target_packet_rejects(payload, message):
    with pytest.raises(ValueError, match=message):
        slider_link.decode_targets(payload, VARIANT)


def receiver(box, clock=time.monotonic):
    return slider_link.TargetReceiver("127.0.0.1", VARIANT, box, port=0, clock=clock)


def test_receiver_filters_peer_order_and_sessions():
    box = slider_link.TargetBox()
    rx = receiver(box, clock=lambda: 5.0)
    try:
        rx.accept(slider_link.encode_targets(VARIANT, "s1", 1, {"a": 1.0}), ("10.0.0.9", 1))
        assert box.latest()[1] is None and "only 127.0.0.1" in rx.last_reject
        rx.accept(slider_link.encode_targets(VARIANT, "s1", 2, {"a": 2.0}), ("127.0.0.1", 9))
        rx.accept(slider_link.encode_targets(VARIANT, "s1", 1, {"a": 1.0}), ("127.0.0.1", 9))
        assert box.latest() == ({"a": 2.0}, 5.0, "s1", False)
        rx.accept(slider_link.encode_targets(VARIANT, "s2", 1, {"a": 3.0}), ("127.0.0.1", 9))
        assert box.latest() == ({"a": 3.0}, 5.0, "s2", False)
    finally:
        rx.close()


def test_udp_loopback_targets_and_status():
    box = slider_link.TargetBox()
    rx = receiver(box)
    rx.start()
    tx = slider_link.TargetSender("127.0.0.1", VARIANT, port=rx.sock.getsockname()[1])
    try:
        tx.send({"l_hip_yaw": 0.1})
        deadline = time.monotonic() + 2.0
        while box.latest()[1] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert box.latest()[0] == {"l_hip_yaw": 0.1}
        rx.send_status("syncing", "note", [[1, "j", "l_hip_yaw", 0.0, 0.0, 0.0, 30.0]], {"l_hip_yaw": 0.2})
        status = None
        while status is None and time.monotonic() < deadline:
            status = tx.poll_status()
            time.sleep(0.01)
        assert status["state"] == "syncing" and status["sync"] == {"l_hip_yaw": 0.2}
        rx.set_idle("stopped", "why")
        status = None
        while (status is None or status["state"] != "stopped") and time.monotonic() < deadline:
            status = tx.poll_status() or status
            time.sleep(0.02)
        assert status["note"] == "why" and status["rows"] == []
    finally:
        tx.close()
        rx.close()


class ScriptedBox(slider_link.TargetBox):
    def __init__(self, clock, script, end):
        super().__init__()
        self.clock = clock
        self.script = script
        self.end = end
        self.syncs = []

    def latest(self):
        now = self.clock.now - self.start
        if now >= self.end:
            return None, None, None, True
        return self.script(now, self.start)

    def set_sync(self, values):
        self.syncs.append((self.clock.now - self.start, values))
        super().set_sync(values)


def run_track(monkeypatch, script, end, starts=None):
    clock = FakeClock()
    monkeypatch.setattr(mujoco_to_real, "time", clock)
    history = []
    starts = starts or {mid: 0.0 for mid in IDS}
    motors = {mid: IdealMotor(mid, starts[mid], clock, history) for mid in IDS}
    args = type("Args", (), {})()
    args.rate, args.max_speed, args.max_accel = 100.0, 0.10, 0.25
    args.feedback_timeout, args.overspeed, args.max_error_deg, args.max_temp = 0.30, 2.0, 25.0, 70.0
    args.kp = {mid: 20.0 for mid in IDS}
    args.kd = {mid: 1.0 for mid in IDS}
    box = ScriptedBox(clock, script, end)
    box.start = clock.now
    commands = dict(starts)
    limiters = {mid: safety.AxisLimiter(starts[mid]) for mid in IDS}
    states = []
    mujoco_to_real.track(
        motors, {}, IDS, NAMES, commands, limiters, LIMITS, LIMITS, (), None, args, box,
        report=lambda state, targets, cmds: states.append((clock.now - box.start, state, dict(targets))),
    )
    return states, commands, motors, box


def fresh(targets):
    return lambda now, start: (targets, start + now, "s", False)


def test_track_arms_on_matching_targets_and_follows(monkeypatch, capsys):
    script = lambda now, start: ({"l_hip_yaw": 0.0 if now < 1.0 else 0.2, "l_hip_pitch": 0.0}, start + now, "s", False)
    states, commands, _, _ = run_track(monkeypatch, script, 8.0)
    assert states[0][1] == "tracking"
    assert commands[1] == pytest.approx(0.2, abs=1e-9)


def test_track_holds_forever_on_link_loss_and_resyncs(monkeypatch, capsys):
    def script(now, start):
        if now < 1.0:
            return {"l_hip_yaw": 0.0 if now < 0.2 else 0.3, "l_hip_pitch": 0.0}, start + now, "s", False
        if now < 3600.0:
            return {"l_hip_yaw": 0.3, "l_hip_pitch": 0.0}, start + 1.0, "s", False
        return {"l_hip_yaw": 0.6, "l_hip_pitch": 0.0}, start + now, "s2", False

    states, commands, motors, box = run_track(monkeypatch, script, 3602.0)
    by_state = {}
    for t, state, _ in states:
        by_state.setdefault(state, []).append(t)
    assert min(by_state["holding"]) == pytest.approx(1.51, abs=0.02)
    assert max(by_state["holding"]) == pytest.approx(3600.0, abs=0.02)
    held = [targets[1] for t, state, targets in states if state == "holding"]
    assert max(held) - min(held) < 1e-12
    assert max(by_state["syncing"]) >= 3600.0
    hold_sync = [values for t, values in box.syncs if values is not None]
    assert hold_sync and hold_sync[-1]["l_hip_yaw"] == pytest.approx(held[0])
    assert commands[1] == pytest.approx(held[0], abs=1e-9)
    assert all(state != "tracking" for t, state, _ in states if t > 1.6)


def test_track_resumes_after_sender_syncs(monkeypatch, capsys):
    synced = {}

    def script(now, start):
        if now < 1.0:
            return {"l_hip_yaw": 0.0, "l_hip_pitch": 0.0}, start + now, "s", False
        if now < 5.0:
            return {"l_hip_yaw": 0.0, "l_hip_pitch": 0.0}, start + 1.0, "s", False
        if now < 5.5:
            return {"l_hip_yaw": 0.5, "l_hip_pitch": 0.5}, start + now, "s2", False
        if now < 6.0:
            return dict(synced), start + now, "s2", False
        return {"l_hip_yaw": 0.1, "l_hip_pitch": 0.0}, start + now, "s2", False

    original = ScriptedBox.set_sync

    def capture(self, values):
        original(self, values)
        if values is not None:
            synced.update(values)
    monkeypatch.setattr(ScriptedBox, "set_sync", capture)
    states, commands, _, _ = run_track(monkeypatch, script, 12.0)
    phases = {}
    for t, state, _ in states:
        phases.setdefault(state, []).append(t)
    assert max(phases["holding"]) < 5.0
    assert 5.0 <= min(phases["syncing"]) and max(phases["syncing"]) < 5.5
    assert min(t for t in phases["tracking"] if t > 2.0) == pytest.approx(5.5, abs=0.02)
    assert commands[1] == pytest.approx(0.1, abs=1e-9)


def test_track_hardware_fault_still_stops(monkeypatch, capsys):
    clock = FakeClock()

    def script(now, start):
        return {"l_hip_yaw": 0.0, "l_hip_pitch": 0.0}, start + now, "s", False

    monkeypatch.setattr(mujoco_to_real, "time", clock)
    motors = {mid: IdealMotor(mid, 0.0, clock, []) for mid in IDS}
    motors[2].last_temp = 75.0
    args = type("Args", (), {})()
    args.rate, args.max_speed, args.max_accel = 100.0, 0.10, 0.25
    args.feedback_timeout, args.overspeed, args.max_error_deg, args.max_temp = 0.30, 2.0, 25.0, 70.0
    args.kp = {mid: 20.0 for mid in IDS}
    args.kd = {mid: 1.0 for mid in IDS}
    box = ScriptedBox(clock, script, 5.0)
    box.start = clock.now
    with pytest.raises(RuntimeError, match="overtemperature"):
        mujoco_to_real.track(
            motors, {}, IDS, NAMES, {1: 0.0, 2: 0.0}, {mid: safety.AxisLimiter(0.0) for mid in IDS},
            LIMITS, LIMITS, (), None, args, box,
        )


def test_run_restarts_after_a_stop_instead_of_exiting(tmp_path, monkeypatch, capsys):
    path = tmp_path / "robot_model"
    path.write_text("ver2_edu\n")
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(path))
    args = mujoco_to_real.parse_args(["--remote", "127.0.0.1"])

    class FakeReceiver:
        def __init__(self, *a, **k):
            self.idle = []
            self.closed = False

        def start(self):
            pass

        def set_idle(self, state, note=""):
            self.idle.append(state)

        def close(self):
            self.closed = True

    made = []
    monkeypatch.setattr(mujoco_to_real, "TargetReceiver", lambda *a, **k: made.append(FakeReceiver()) or made[-1])
    monkeypatch.setattr(mujoco_to_real, "wait_for_sender", lambda rx, box: None)
    outcomes = iter(["Safety stop: ID 2 overtemperature (75.0 degC)", "Zero move stopped", None])
    monkeypatch.setattr(mujoco_to_real, "run_session", lambda *a: next(outcomes))
    restarts = []
    monkeypatch.setattr(mujoco_to_real, "wait_for_restart", lambda: restarts.append(1))
    mujoco_to_real.run(args)
    assert len(restarts) == 2
    assert made[0].idle.count("stopped") == 2 and made[0].closed
    assert "The program keeps running" in capsys.readouterr().out


def test_wait_for_restart_needs_a_terminal(monkeypatch):
    class NoTty:
        def isatty(self):
            return False
    monkeypatch.setattr(mujoco_to_real.sys, "stdin", NoTty())
    with pytest.raises(RuntimeError, match="interactive terminal"):
        mujoco_to_real.wait_for_restart()


def test_sender_args():
    args = slider_sender.parse_args(["--robot-host", "orin", "--robot", "edu"])
    assert args.variant == "ver2" and args.rate == 50.0
    with pytest.raises(SystemExit):
        slider_sender.parse_args(["--robot-host", "orin", "--robot", "edu", "--rate", "1000"])


def test_remote_dry_run_opens_nothing(monkeypatch, capsys):
    monkeypatch.setattr(mujoco_to_real, "TargetReceiver", lambda *a, **k: pytest.fail("socket opened"))
    assert mujoco_to_real.main(["--dry-run", "--robot", "edu", "--remote", "127.0.0.1"]) == 0
    assert "slider_sender.py on 127.0.0.1" in capsys.readouterr().out
