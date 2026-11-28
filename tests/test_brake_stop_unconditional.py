from types import SimpleNamespace

import can
import pytest

import safety
from robonex_common.can import Motor
from robonex_common.protocol import COMM_STOP, parse_arbitration_id


class RecordingMotor:
    def __init__(self, motor_id, control_error=None, stop_error=None, spec=True):
        self.motor_id = motor_id
        if spec:
            self.spec = SimpleNamespace(kd_max=5.0)
        self.control_error = control_error
        self.stop_error = stop_error
        self.damped = 0
        self.stopped = False

    def control(self, pos, vel, kp, kd, torque=0.0):
        if self.control_error is not None:
            raise self.control_error
        assert kp == 0.0 and torque == 0.0
        self.damped += 1

    def stop(self):
        if self.stop_error is not None:
            raise self.stop_error
        self.stopped = True


def _motors(ids, overrides=None):
    overrides = overrides or {}
    return {mid: overrides.get(mid, RecordingMotor(mid)) for mid in ids}


def test_missing_gain_still_stops_every_motor():
    motors = _motors((1, 2, 16))
    report = safety.brake_and_stop(motors, {}, [1, 2, 16], [1, 2, 16], 0.03, {1: 2.0, 2: 2.0})
    assert report["stop_sent"] == [1, 2, 16]
    assert all(motor.stopped for motor in motors.values())
    assert isinstance(report["damping_errors"][16], KeyError)
    assert motors[1].damped >= 1 and motors[2].damped >= 1
    assert motors[16].damped == 0


@pytest.mark.parametrize("error", [ValueError("non-finite"), RuntimeError("bad motor"), TypeError("bad gain")])
def test_damping_error_of_one_motor_keeps_damping_the_others_and_stops_all(error):
    motors = _motors((1, 2, 3), {2: RecordingMotor(2, control_error=error)})
    report = safety.brake_and_stop(motors, {}, [1, 2, 3], [1, 2, 3], 0.03, 2.0)
    assert report["stop_sent"] == [1, 2, 3]
    assert report["damping_errors"][2] is error
    assert motors[1].damped >= 2 and motors[3].damped >= 2


def test_motor_without_spec_is_still_stopped():
    motors = _motors((1, 7), {7: RecordingMotor(7, spec=False)})
    report = safety.brake_and_stop(motors, {}, [1, 7], [1, 7], 0.02, 2.0)
    assert report["stop_sent"] == [1, 7]
    assert isinstance(report["damping_errors"][7], AttributeError)


def test_failing_stop_frame_does_not_skip_the_rest():
    error = RuntimeError("bus gone")
    motors = _motors((1, 2, 3), {2: RecordingMotor(2, stop_error=error)})
    report = safety.brake_and_stop(motors, {}, [1, 2, 3], [1, 2, 3], 0.0, 2.0)
    assert report["stop_sent"] == [1, 3]
    assert report["stop_errors"] == {2: error}
    assert motors[1].stopped and motors[3].stopped


def test_stop_frames_go_out_even_when_the_damping_phase_raises():
    motors = _motors((1, 2))

    def observe():
        raise SystemExit("observer exits")

    with pytest.raises(SystemExit):
        safety.brake_and_stop(motors, {}, [1, 2], [1, 2], 0.05, 2.0, observe=observe)
    assert all(motor.stopped for motor in motors.values())


def test_stop_ids_without_damping_get_a_stop_frame():
    motors = _motors((1, 2, 3))
    report = safety.brake_and_stop(motors, {}, [1], [1, 2, 3], 0.02, {})
    assert report["stop_sent"] == [1, 2, 3]
    assert isinstance(report["damping_errors"][1], KeyError)


def test_missing_gain_on_a_virtual_bus_sends_every_stop_frame():
    channel = "brake_fix_virtual"
    sender = can.Bus(channel=channel, interface="virtual")
    listener = can.Bus(channel=channel, interface="virtual")
    try:
        motors = {
            1: Motor(sender, 1, "rs02"),
            2: Motor(sender, 2, "rs03"),
            16: Motor(sender, 16, "rs02"),
        }
        report = safety.brake_and_stop(
            motors, {"virtual": sender}, [1, 2, 16], [1, 2, 16], 0.03, {1: 2.0, 2: 2.0}
        )
        stopped = set()
        damped = set()
        while True:
            message = listener.recv(timeout=0.05)
            if message is None:
                break
            comm, _, motor_id = parse_arbitration_id(message.arbitration_id)
            if comm == COMM_STOP:
                stopped.add(motor_id)
            else:
                damped.add(motor_id)
    finally:
        listener.shutdown()
    assert stopped == {1, 2, 16}
    assert damped == {1, 2}
    assert report["stop_sent"] == [1, 2, 16]
    assert isinstance(report["damping_errors"][16], KeyError)
