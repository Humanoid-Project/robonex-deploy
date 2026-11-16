"""Offline checks of FreshFeedback in policy_to_real.py and of its wiring there. No hardware.

Run: .venv/bin/python scripts/policy_test/test_fresh_feedback.py

The class cases use python-can virtual buses. The wiring cases reuse the fakes in
tests/test_policy_to_real_*.py, which import pytest; without pytest they are skipped.
"""

import contextlib
import csv
import functools
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import can
from robonex_common.can import FeedbackHub, Motor
from robonex_common.protocol import COMM_FEEDBACK, COMM_OPERATION, HOST_ID, build_arbitration_id, parse_arbitration_id

from policy_to_real import FreshFeedback, FreshResult

DEPLOY = Path(__file__).resolve().parents[2]
POLICY_TO_REAL = DEPLOY / "scripts" / "policy_test" / "policy_to_real.py"
OFF_PATH_BASELINE = "HEAD"  # the committed version, without the --fresh-feedback wiring
CHANNELS = {"fake_left": (1, 2, 3), "fake_right": (7, 8, 9)}
RUNNING = 2 << 14  # mode bits of a running motor


def feedback(motor_id, length=8):
    return can.Message(
        arbitration_id=build_arbitration_id(COMM_FEEDBACK, RUNNING | motor_id, HOST_ID),
        data=bytes([0x80, 0x00] * 4)[:length], is_extended_id=True,
    )


class FakeMotors(threading.Thread):
    """Answers every type-0x01 frame with type 0x02, per-motor delay / drop / short payload."""

    def __init__(self, channel, notify=None):
        super().__init__(daemon=True)
        self.bus = can.Bus(channel=channel, interface="virtual")
        self.notify = notify
        self.delay = {}     # motor id -> seconds
        self.drop = set()
        self.short = set()
        self.seen = 0
        self.timers = []
        self.running = True

    def reply(self, motor_id):
        if self.running:
            self.bus.send(feedback(motor_id, 4 if motor_id in self.short else 8))
            if self.notify:
                self.notify()

    def run(self):
        while self.running:
            msg = self.bus.recv(timeout=0.05)
            if msg is None:
                continue
            comm_type, _, motor_id = parse_arbitration_id(msg.arbitration_id)
            if comm_type != COMM_OPERATION:
                continue
            self.seen += 1
            if motor_id in self.drop:
                continue
            delay = self.delay.get(motor_id, 0.0)
            if delay:
                timer = threading.Timer(delay, self.reply, (motor_id,))
                self.timers.append(timer)
                timer.start()
            else:
                self.reply(motor_id)

    def close(self):
        self.running = False
        for timer in self.timers:
            timer.cancel()
            timer.join()
        self.join()
        self.bus.shutdown()


class PipeBus:
    """Virtual bus with a fileno(), so the select() branch used on socketcan runs offline."""

    def __init__(self, bus):
        self.bus = bus
        self.read_fd, self.write_fd = os.pipe()

    def fileno(self):
        return self.read_fd

    def notify(self):
        os.write(self.write_fd, b"x")

    def recv(self, timeout=None):
        msg = self.bus.recv(timeout)
        if msg is not None:
            os.read(self.read_fd, 1)
        return msg

    def send(self, msg, timeout=None):
        self.bus.send(msg, timeout)

    def shutdown(self):
        self.bus.shutdown()
        os.close(self.read_fd)
        os.close(self.write_fd)


@contextmanager
def rig(selectable):
    """Two virtual buses with three fake motors each; everything is closed on the way out."""
    buses, fakes, motors, hubs = {}, {}, {}, {}
    try:
        for channel, motor_ids in CHANNELS.items():
            bus = can.Bus(channel=channel, interface="virtual")
            if selectable:
                bus = PipeBus(bus)
            buses[channel] = bus
            fakes[channel] = FakeMotors(channel, bus.notify if selectable else None)
            fakes[channel].start()
            for motor_id in motor_ids:
                motors[motor_id] = Motor(bus, motor_id, "rs02")
            hubs[channel] = FeedbackHub(bus, [motors[motor_id] for motor_id in motor_ids])
        yield buses, fakes, motors, hubs
    finally:
        for fake in fakes.values():
            fake.close()
        for bus in buses.values():
            bus.shutdown()


def command_all(motors):
    for motor in motors.values():
        motor.control(pos=0.1, vel=0.0, kp=10.0, kd=1.0)


def drain(hubs):
    time.sleep(0.03)  # let every pending reply land first
    for hub in hubs.values():
        hub.pump()


def seen(fakes):
    return sum(fake.seen for fake in fakes.values())


# --- FreshFeedback -----------------------------------------------------------------------

def case_all_fresh(selectable):
    """All motors answer: all fresh, state updated, one repeated frame each, hub counters grow."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        fresh = FreshFeedback(motors, hubs)
        command_all(motors)
        assert sorted(fresh.cache) == sorted(motors)
        drain(hubs)
        before = {motor_id: motor.last_feedback_time for motor_id, motor in motors.items()}
        count = seen(fakes)
        hubs["fake_left"].last_pump_frames = 5   # as if the tick's drain had read 5 frames
        hubs["fake_left"].total_frames = 100
        result = fresh.refresh(time.monotonic())
        assert result.resent == 6 and result.late_ids == [] and result.stop_reason is None, result
        assert sorted(result.rx_age_ms) == sorted(motors)
        assert all(0.0 <= age < 50.0 for age in result.rx_age_ms.values()), result
        assert all(motors[motor_id].last_feedback_time > before[motor_id] for motor_id in motors)
        assert all(motor.last_rx_kernel_time > 0.0 for motor in motors.values())
        assert (hubs["fake_left"].last_pump_frames, hubs["fake_left"].total_frames) == (8, 103)
        # A plain FeedbackHub has no counters of its own: they start from 0.
        assert (hubs["fake_right"].last_pump_frames, hubs["fake_right"].total_frames) == (3, 3)
        assert fresh.summary()["ticks"] == 1 and fresh.summary()["ticks_with_late_motor"] == 0
        time.sleep(0.02)
        assert seen(fakes) == count + 6  # exactly one repeated frame per motor


def case_missing_cache(selectable):
    """A motor without a cached frame stops the run by name; nothing is sent."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        fresh = FreshFeedback(motors, hubs)
        result = fresh.refresh(time.monotonic())
        assert result.resent == 0 and result.stop_reason, result
        assert str(sorted(motors)) in result.stop_reason, result
        for motor_id in (1, 2, 3, 7):
            motors[motor_id].control(pos=0.1, vel=0.0, kp=10.0, kd=1.0)
        drain(hubs)
        count = seen(fakes)
        result = fresh.refresh(time.monotonic())
        assert result.resent == 0 and "[8, 9]" in result.stop_reason, result
        time.sleep(0.02)
        assert seen(fakes) == count
        assert fresh.summary()["ticks"] == 0


def case_stale_short_late_and_stop(selectable):
    """Stale, short and late replies are not fresh; three misses in a row stop the run."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        left, right = fakes["fake_left"], fakes["fake_right"]
        fresh = FreshFeedback(motors, hubs, max_misses=3)
        command_all(motors)
        drain(hubs)

        # A reply from before the re-send is ingested but is not fresh (motor 2 then stays silent).
        left.drop.add(2)
        left.bus.send(feedback(2))
        if selectable:
            buses["fake_left"].notify()
        time.sleep(0.01)
        stamp = motors[2].last_feedback_time
        result = fresh.refresh(time.monotonic())
        assert result.late_ids == [2] and 2 not in result.rx_age_ms, result
        assert motors[2].last_feedback_time > stamp   # the stale frame still updated the state
        assert fresh.misses[2] == 1 and result.stop_reason is None
        assert 4.0 < result.wait_ms < 30.0, result    # waited for the 5 ms deadline

        # Short payload: rejected, previous value kept. Motor 8 answers after the deadline.
        right.short.add(7)
        right.delay[8] = 0.03
        position = motors[7].last_position
        stamp = motors[7].last_feedback_time
        result = fresh.refresh(time.monotonic())
        assert result.late_ids == [2, 7, 8], result
        assert motors[7].last_feedback_time == stamp and motors[7].last_position == position
        assert fresh.misses == {1: 0, 2: 2, 3: 0, 7: 1, 8: 1, 9: 0}, fresh.misses
        assert result.stop_reason is None

        # Motor 8's late reply is now queued: it predates the next re-send, so it must not count.
        # Motor 7 recovers (counter back to 0). Motor 2 misses a third time -> stop, cache cleared.
        time.sleep(0.06)
        right.short.clear()
        right.drop.add(8)
        result = fresh.refresh(time.monotonic())
        assert result.late_ids == [2, 8], result
        assert fresh.misses[7] == 0 and fresh.misses[8] == 2
        assert result.stop_reason and "[2]" in result.stop_reason, result
        assert fresh.cache == {}
        assert fresh.refresh(time.monotonic()).stop_reason   # nothing left to repeat
        assert fresh.summary()["ticks"] == 3 and fresh.summary()["ticks_with_late_motor"] == 3


def case_tick_cap(selectable):
    """7.5 ms of the tick already used -> about 0.5 ms of waiting, not the full 5 ms."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        for fake in fakes.values():
            fake.drop.update(motors)
        fresh = FreshFeedback(motors, hubs)
        command_all(motors)
        result = fresh.refresh(time.monotonic() - 0.0075)
        assert result.late_ids == sorted(motors) and result.wait_ms < 4.5, result


def case_brake_clears_cache(selectable):
    """Damping frames are cached like any MIT frame; the stop frames forget them."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        fresh = FreshFeedback(motors, hubs)
        for motor in motors.values():
            motor.control(pos=0.0, vel=0.0, kp=0.0, kd=1.0)
        assert sorted(fresh.cache) == sorted(motors)
        for motor in motors.values():
            motor.stop()
        assert fresh.cache == {}
        time.sleep(0.02)  # let the fake motors count the damping frames first
        count = seen(fakes)
        assert fresh.refresh(time.monotonic()).resent == 0
        time.sleep(0.02)
        assert seen(fakes) == count


def case_send_failure(selectable):
    """A send that raises clears the cache and propagates."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        fresh = FreshFeedback(motors, hubs)
        command_all(motors)

        def broken(msg, timeout=None):
            raise can.CanOperationError("Transmit buffer full")

        buses["fake_right"].send = broken
        try:
            fresh.refresh(time.monotonic())
        except can.CanError:
            pass
        else:
            raise AssertionError("send failure was swallowed")
        assert fresh.cache == {}


def case_close(selectable):
    """close() hands back the very same bus objects, forgets everything, and can be repeated."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        originals = {motor_id: motor.bus for motor_id, motor in motors.items()}
        fresh = FreshFeedback(motors, hubs)
        assert all(motors[motor_id].bus is not originals[motor_id] for motor_id in motors)
        command_all(motors)
        fresh.misses[1] = 2
        for _ in range(2):
            fresh.close()
            assert all(motors[motor_id].bus is originals[motor_id] for motor_id in motors)
            assert fresh.cache == {} and set(fresh.misses.values()) == {0}
        command_all(motors)            # raw buses again: nothing is recorded
        assert fresh.cache == {}
        assert fresh.refresh(time.monotonic()).stop_reason
        FreshFeedback(motors, hubs).close()   # a new one installs cleanly after close()


def case_double_install(selectable):
    """A second install is refused before any motor is touched."""
    with rig(selectable) as (buses, fakes, motors, hubs):
        subset = {motor_id: motors[motor_id] for motor_id in (7, 8)}   # checked after 1, 2, 3
        first = FreshFeedback(subset, hubs)
        installed = {motor_id: motor.bus for motor_id, motor in motors.items()}
        try:
            FreshFeedback(motors, hubs)
        except RuntimeError:
            pass
        else:
            raise AssertionError("double install was accepted")
        assert all(motors[motor_id].bus is installed[motor_id] for motor_id in motors)
        assert motors[1].bus is buses["fake_left"] and motors[9].bus is buses["fake_right"]
        first.close()
        assert motors[7].bus is buses["fake_right"]


MODULE_CASES = (
    case_all_fresh, case_missing_cache, case_stale_short_late_and_stop, case_tick_cap,
    case_brake_clears_cache, case_send_failure, case_close, case_double_install,
)


# --- wiring in policy_to_real.py ---------------------------------------------------------

def load_harness():
    for sub in ("scripts/sim_to_real", "scripts", "tests"):
        if str(DEPLOY / sub) not in sys.path:
            sys.path.insert(0, str(DEPLOY / sub))
    import test_policy_to_real_timing as harness  # imports pytest
    return harness


def quiet_loop(harness, module, contract, csv_path):
    """One fake-clock policy_loop run -> (commands, observations, CSV bytes, stdout, error)."""
    out = io.StringIO()
    error = None
    commands = observations = None
    with contextlib.redirect_stdout(out):
        try:
            commands, observations = harness.run_loop(module, contract, csv_path, duration=1.0)
        except (RuntimeError, OSError) as caught:
            error = caught
    data = csv_path.read_bytes()
    shutil.rmtree(csv_path.parent)
    csv_path.parent.mkdir()
    return commands, observations, data, out.getvalue(), error


def case_flag_off_is_identical(harness, tmp):
    """Flag off: same frames, same observations, same CSV bytes, same printout as the baseline."""
    source = subprocess.run(
        ["git", "-C", str(DEPLOY), "show", f"{OFF_PATH_BASELINE}:scripts/policy_test/policy_to_real.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    base_path = tmp / "baseline" / "scripts" / "policy_test" / "policy_to_real_before_fresh.py"
    base_path.parent.mkdir(parents=True)
    base_path.write_text(source)
    contract = harness.make_contract()
    csv_path = tmp / "out" / "loop.csv"
    csv_path.parent.mkdir()
    runs = [
        quiet_loop(harness, harness.load_module(name, path), contract, csv_path)
        for name, path in (("policy_to_real_before_fresh", base_path), ("policy_to_real_fresh_off", POLICY_TO_REAL))
    ]
    (old_commands, old_obs, old_csv, old_text, old_error), (commands, obs, csv_bytes, text, error) = runs
    assert old_error is None and error is None
    assert len(commands) == len(old_commands) >= 40 * 12
    assert harness.command_bytes(commands) == harness.command_bytes(old_commands)
    assert [o.tobytes() for o in obs] == [o.tobytes() for o in old_obs]
    assert csv_bytes == old_csv and len(csv_bytes.splitlines()) == len(obs) + 1
    assert text == old_text and "Policy control is active" in text


class ScriptedFresh:
    """Stands in for FreshFeedback inside policy_loop."""

    def __init__(self, events, stop_at=None, raise_at=None):
        self.events, self.stop_at, self.raise_at = events, stop_at, raise_at
        self.calls = 0

    def refresh(self, tick_start):
        self.calls += 1
        self.events.append("refresh")
        if self.calls == self.raise_at:
            raise OSError("synthetic bus failure")
        result = FreshResult(resent=12, resend_ms=0.4, wait_ms=1.5, rx_age_ms={1: 0.7, 2: 0.9},
                             late_ids=[3, 5] if self.calls == 2 else [])
        if self.calls == self.stop_at:
            result.stop_reason = "no fresh feedback from motor IDs [3] on 3 consecutive ticks"
        return result


def case_flag_on_loop(harness, tmp):
    """Flag on: poll -> refresh -> one faults/thermal update per tick, new columns, stop rows."""
    module = harness.load_module("policy_to_real_fresh_on", POLICY_TO_REAL)
    contract = harness.make_contract()
    csv_path = tmp / "on" / "loop.csv"
    csv_path.parent.mkdir()
    events = []

    def traced(original, name):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            events.append(name)
            return original(*args, **kwargs)
        return wrapper

    original_loop = module.policy_loop
    original_poll = harness.FakeJoints.poll
    harness.FakeJoints.poll = traced(original_poll, "poll")
    module.FaultMonitor.update = traced(module.FaultMonitor.update, "faults")
    original_thermal = module.ThermalLoad.update   # safety.ThermalLoad, shared with other modules
    module.ThermalLoad.update = traced(original_thermal, "thermal")

    def run(**scripted):
        events.clear()
        module.policy_loop = functools.partial(original_loop, fresh=ScriptedFresh(events, **scripted))
        _, _, data, _, error = quiet_loop(harness, module, contract, csv_path)
        reader = csv.DictReader(io.StringIO(data.decode(), newline=""))
        return list(reader), reader.fieldnames, error

    try:
        rows, header, error = run()
        assert error is None and len(rows) >= 40
        assert header[-5:] == list(module.FRESH_COLUMNS) + ["stop_reason"]
        assert events[:4] == ["poll", "refresh", "faults", "thermal"]
        assert events == ["poll", "refresh", "faults", "thermal"] * len(rows)
        assert [rows[0][name] for name in module.FRESH_COLUMNS] == ["1.5", "0.4", "", "0.9"]
        assert rows[1]["fresh_late_ids"] == "3 5"

        rows, _, error = run(stop_at=5)
        reason = "no fresh feedback from motor IDs [3] on 3 consecutive ticks"
        assert isinstance(error, RuntimeError) and str(error) == "Safety stop: " + reason
        assert len(rows) == 5 and [row["stop_reason"] for row in rows] == [""] * 4 + [reason]
        assert events[-2:] == ["poll", "refresh"]   # stopped before faults / thermal / send

        rows, _, error = run(raise_at=5)
        assert isinstance(error, OSError) and str(error) == "synthetic bus failure"
        assert len(rows) == 5 and rows[-1]["stop_reason"] == "OSError: synthetic bus failure"
    finally:
        module.policy_loop = original_loop
        module.ThermalLoad.update = original_thermal
        harness.FakeJoints.poll = original_poll


def case_flag_on_deploy(harness, tmp):
    """run_deploy: installed before the first command, closed before the brake, summary saved."""
    import pytest
    import test_policy_to_real_recording as recording

    module = harness.load_module("policy_to_real_fresh_deploy", POLICY_TO_REAL)
    seen_at = {}

    class RecordingFresh(ScriptedFresh):
        def __init__(self, motors, hubs):
            super().__init__([])
            self.motors = motors
            seen_at["install_commands"] = sum(len(motor.log) for motor in motors.values())

        def close(self):
            seen_at.setdefault("close_stopped", [motor.stopped for motor in self.motors.values()])
            seen_at["close_commands"] = len(next(iter(self.motors.values())).log)

        def summary(self):
            return {"ticks": self.calls}

    original_deploy = module.run_deploy

    def run_deploy(policy_path, contract, args):
        args.fresh_feedback = True
        return original_deploy(policy_path, contract, args)

    out = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out):
        patch.setattr(module, "FreshFeedback", RecordingFresh)
        patch.setattr(module, "run_deploy", run_deploy)
        run = recording.run_full_deploy(module, patch, tmp)
    assert run.error is None
    rows = harness.read_csv(run.path)
    assert seen_at["install_commands"] == 0                      # before approach_pose
    assert seen_at["close_stopped"] == [False] * len(run.motors)  # before the stop frames
    assert len(run.control_log) > seen_at["close_commands"]       # and before the damping frames
    assert all(motor.stopped for motor in run.motors.values())
    assert len(rows) >= 40 and all(row["fresh_wait_ms"] == "1.5" for row in rows)
    assert recording.load_meta(run.path)["end"]["fresh_feedback"] == {"ticks": len(rows)}


WIRING_CASES = (case_flag_off_is_identical, case_flag_on_loop, case_flag_on_deploy)


if __name__ == "__main__":
    for selectable in (False, True):
        kind = "select() on fileno" if selectable else "virtual bus polling"
        for case in MODULE_CASES:
            case(selectable)
            print(f"ok  {case.__name__}  ({kind})")
    try:
        wiring_harness = load_harness()
    except ImportError as import_error:
        print(f"skipped  wiring cases ({import_error}); they need pytest on the path")
    else:
        for case in WIRING_CASES:
            with tempfile.TemporaryDirectory() as directory:
                case(wiring_harness, Path(directory))
            print(f"ok  {case.__name__}")
