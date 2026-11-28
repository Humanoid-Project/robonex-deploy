import struct
import uuid
from types import SimpleNamespace

import can
import pytest

import safety
from robonex_common.can import Motor
from robonex_common.joints import JOINT_BY_MODEL_NAME, MOTOR_BY_ID
from robonex_common.motors import MOTOR_SPECS
from robonex_common.protocol import (
    COMM_ENABLE,
    COMM_FEEDBACK,
    COMM_OPERATION,
    COMM_PARAMETER_READ,
    COMM_PARAMETER_WRITE,
    COMM_SET_CAN_ID,
    COMM_SET_ZERO,
    COMM_STOP,
    HOST_ID,
    build_arbitration_id,
)
from test_policy_to_real_recording import FakeHub, load_meta, run_full_deploy
from test_policy_to_real_timing import make_contract, modules

CONTRACT = make_contract()
LEG_IDS = [JOINT_BY_MODEL_NAME[name].motor_id for name in CONTRACT.joint_order]


def _feedback(motor_id):
    data16 = (safety.MODE_RUNNING << 14) | motor_id
    payload = struct.pack(">HHHH", 32767, 32767, 32767, 300)
    return can.Message(arbitration_id=build_arbitration_id(COMM_FEEDBACK, data16, HOST_ID), data=payload,
                       is_extended_id=True)


def _request(comm_type, motor_id, data16=0x00FE):
    return can.Message(arbitration_id=build_arbitration_id(comm_type, data16, motor_id), data=bytes(8),
                       is_extended_id=True)


@pytest.fixture
def virtual_hub():
    channel = f"g2_{uuid.uuid4().hex}"
    deploy = can.Bus(channel=channel, interface="virtual", receive_own_messages=False)
    other = can.Bus(channel=channel, interface="virtual")
    listener = can.Bus(channel=channel, interface="virtual")
    motor = Motor(deploy, 1, MOTOR_SPECS[MOTOR_BY_ID[1].motor_model])
    hub = safety.TimestampedFeedbackHub(deploy, [motor], HOST_ID, channel=channel)
    try:
        yield SimpleNamespace(channel=channel, deploy=deploy, other=other, listener=listener, motor=motor, hub=hub)
    finally:
        for bus in (deploy, other, listener):
            bus.shutdown()


RUNTIME_ARGS = SimpleNamespace(feedback_timeout=1.0, overspeed=10.0, max_temp=70.0, max_error_deg=25.0)


def _reason(v):
    return safety.runtime_safety_reason({1: v.motor}, {}, {1: (-1.0, 1.0)}, safety.time.monotonic(), RUNTIME_ARGS)


def test_foreign_read_warns_once_per_channel_and_the_run_continues(virtual_hub, capsys):
    v = virtual_hub
    v.other.send(_feedback(1))
    for _ in range(3):
        v.other.send(_request(COMM_PARAMETER_READ, 1))
    v.hub.pump()
    for _ in range(2):
        v.other.send(_request(COMM_PARAMETER_READ, 1))
    v.hub.pump()
    out = capsys.readouterr().out
    assert out.count("Warning: foreign parameter read") == 1
    assert f"(comm type 0x11) to motor ID 1 on {v.channel}" in out
    assert v.hub.foreign_frames == {"reads": 5, "control": 0, "other": 0}
    assert v.hub.foreign_control is None
    assert _reason(v) is None


@pytest.mark.parametrize("comm_type, name", [
    (COMM_OPERATION, "MIT"), (COMM_ENABLE, "enable"), (COMM_STOP, "stop"),
    (COMM_SET_ZERO, "set zero"), (COMM_SET_CAN_ID, "set ID"), (COMM_PARAMETER_WRITE, "parameter write"),
])
def test_foreign_control_stops_with_channel_type_and_id(virtual_hub, comm_type, name):
    v = virtual_hub
    v.other.send(_feedback(1))
    v.hub.pump()
    assert _reason(v) is None
    v.other.send(_request(comm_type, 1, data16=0x1234))
    v.hub.pump()
    reason = _reason(v)
    assert reason is not None
    assert reason.startswith(f"foreign {name} frame (comm type 0x{comm_type:02x}) to motor ID 1 on {v.channel}")
    assert v.hub.foreign_frames == {"reads": 0, "control": 1, "other": 0}
    assert v.hub.foreign_control == reason


def test_foreign_control_seen_by_wait_for_is_caught(virtual_hub):
    v = virtual_hub
    v.other.send(_request(COMM_OPERATION, 1))
    v.other.send(_feedback(1))
    assert v.hub.wait_for(1, timeout=0.2) is not None
    assert _reason(v).startswith("foreign MIT frame")


def test_replies_to_the_host_and_frames_for_other_motors_are_no_g2_event(virtual_hub, capsys):
    v = virtual_hub
    v.other.send(_feedback(1))
    v.other.send(can.Message(arbitration_id=build_arbitration_id(COMM_PARAMETER_READ, 1, HOST_ID),
                             data=bytes(8), is_extended_id=True))
    v.other.send(_feedback(7))
    v.other.send(_request(COMM_OPERATION, 7))
    v.other.send(can.Message(arbitration_id=0x123, data=bytes(8), is_extended_id=False))
    v.hub.pump()
    assert v.hub.foreign_frames == {"reads": 0, "control": 0, "other": 4}
    assert v.hub.foreign_control is None and _reason(v) is None
    assert "Warning" not in capsys.readouterr().out


def test_the_deploys_own_frames_never_reach_its_hub(virtual_hub):
    v = virtual_hub
    v.other.send(_feedback(1))
    v.hub.pump()
    frames = v.hub.total_frames
    v.motor.enable()
    v.motor.control(pos=0.0, vel=0.0, kp=10.0, kd=1.0)
    v.motor.write_run_mode_operation()
    v.motor.stop()
    v.hub.pump()
    assert v.hub.total_frames == frames
    assert v.hub.foreign_frames == {"reads": 0, "control": 0, "other": 0}
    assert _reason(v) is None
    seen = []
    while (msg := v.listener.recv(timeout=0.0)) is not None:
        seen.append(msg.arbitration_id >> 24 & 0x1F)
    assert seen[-4:] == [COMM_ENABLE, COMM_OPERATION, COMM_PARAMETER_WRITE, COMM_STOP]


def test_open_hardware_asks_python_can_not_to_return_own_frames(monkeypatch):
    calls = []

    class RecordingBus:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def shutdown(self):
            pass

    monkeypatch.setattr(safety.can, "Bus", RecordingBus)
    buses, motors, hubs = safety.open_hardware(LEG_IDS, "socketcan", HOST_ID)
    assert len(calls) == len(hubs) >= 1
    assert all(call["receive_own_messages"] is False for call in calls)
    assert all(isinstance(hub, safety.TimestampedFeedbackHub) and hub.channel == channel
               for channel, hub in hubs.items())


def test_deploy_approach_stops_on_foreign_control(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    original = new.approach_pose
    target = LEG_IDS[0]

    def approach(commander, joints, targets, motors_, *args, **kwargs):
        motors_[target].foreign_control = f"foreign MIT frame (comm type 0x01) to motor ID {target} on can0: x"
        return original(commander, joints, targets, motors_, *args, **kwargs)

    monkeypatch.setattr(new, "approach_pose", approach)
    run = run_full_deploy(new, monkeypatch, tmp_path)
    capsys.readouterr()
    assert run.error is not None
    assert str(run.error) == (f"default move stopped for safety: foreign MIT frame (comm type 0x01) to motor ID "
                              f"{target} on can0: x")
    assert "brake" in run.events


def test_deploy_policy_loop_stops_on_foreign_control(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    original = new.policy_loop
    target = LEG_IDS[1]

    def loop(runner, commander, joints, imu, motors_, *args, **kwargs):
        motors_[target].foreign_control = f"foreign stop frame (comm type 0x04) to motor ID {target} on can0: x"
        return original(runner, commander, joints, imu, motors_, *args, **kwargs)

    monkeypatch.setattr(new, "policy_loop", loop)
    run = run_full_deploy(new, monkeypatch, tmp_path)
    capsys.readouterr()
    assert run.error is not None
    assert str(run.error) == f"Safety stop: foreign stop frame (comm type 0x04) to motor ID {target} on can0: x"
    assert "brake" in run.events
    assert "foreign stop frame" in load_meta(run.path)["end"]["exit"]


def test_foreign_counters_are_in_meta_and_zero_on_a_clean_run(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    monkeypatch.setattr(FakeHub, "foreign_frames", {"reads": 0, "control": 0, "other": 0}, raising=False)
    run = run_full_deploy(new, monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert run.error is None
    assert load_meta(run.path)["end"]["foreign_frames"] == {"can0": {"reads": 0, "control": 0, "other": 0}}
    assert "Foreign CAN frames" not in out


def test_foreign_counters_nonzero_are_printed(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    monkeypatch.setattr(FakeHub, "foreign_frames", {"reads": 3, "control": 0, "other": 2}, raising=False)
    run = run_full_deploy(new, monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert run.error is None
    assert load_meta(run.path)["end"]["foreign_frames"] == {"can0": {"reads": 3, "control": 0, "other": 2}}
    assert "Foreign CAN frames: can0 reads 3 control 0 other 2" in out


def test_virtual_hub_counters_reach_the_shutdown_line(virtual_hub, modules):
    _, new = modules
    v = virtual_hub
    v.other.send(_request(COMM_PARAMETER_READ, 1))
    v.other.send(_request(COMM_OPERATION, 1))
    v.hub.pump()
    line = new.foreign_frames_line({v.channel: v.hub.foreign_frames})
    assert line == f"Foreign CAN frames: {v.channel} reads 1 control 1 other 0"
    assert new.foreign_frames_line({"can0": {"reads": 0, "control": 0, "other": 0}}) is None
