"""CAN primitives copied from Robstride-Motor-Test.

Sources and hashes are recorded in ../SNAPSHOT.md; run ../check_sync.py to
detect drift against the originals.
"""
import struct
import time

import can

HOST_ID = 0xFD
DEFAULT_INTERFACE = "socketcan"

RUN_MODE_INDEX = 0x7005
OPERATION_RUN_MODE = 0
MECH_POS_INDEX = 0x7019

CHANNEL_ID_RANGES = {
    "can0": range(1, 7),
    "can1": range(7, 13),
}

MOTORS = {
    1:  {"target_rad": +0.0119, "model": "rs02"},
    2:  {"target_rad": +0.4013, "model": "rs03"},
    3:  {"target_rad": -0.1473, "model": "rs03"},
    4:  {"target_rad": -0.5475, "model": "rs03"},
    5:  {"target_rad": +0.0000, "model": "rs02"},
    6:  {"target_rad": +0.0000, "model": "rs02"},
    7:  {"target_rad": +0.0016, "model": "rs02"},
    8:  {"target_rad": -0.6227, "model": "rs03"},
    9:  {"target_rad": +0.0114, "model": "rs03"},
    10: {"target_rad": +0.7454, "model": "rs03"},
    11: {"target_rad": +0.0000, "model": "rs02"},
    12: {"target_rad": +0.0001, "model": "rs02"},
}

MOVE_SPEED = 0.4
MIN_MOVE_TIME = 3.0
RATE = 100.0
HOLD_KP = 40.0
HOLD_KD = 2.0
OVERSPEED_STOP = 2.0
FEEDBACK_TIMEOUT = 0.3

JOINT_MAP = {
    1: "left_hip_yaw", 2: "left_hip_pitch", 3: "left_hip_roll",
    4: "left_knee_pitch", 5: "left_ankle_upper", 6: "left_ankle_lower",
    7: "right_hip_yaw", 8: "right_hip_pitch", 9: "right_hip_roll",
    10: "right_knee_pitch", 11: "right_ankle_upper", 12: "right_ankle_lower",
}


def channel_for_id(motor_id):
    for channel, id_range in CHANNEL_ID_RANGES.items():
        if motor_id in id_range:
            return channel
    raise ValueError(f"모터 ID {motor_id} 에 대한 채널을 찾을 수 없습니다 (CHANNEL_ID_RANGES 확인).")


class MotorSpec:
    def __init__(self, name, p_min, p_max, v_min, v_max, t_min, t_max, kp_max, kd_max):
        self.name = name
        self.p_min, self.p_max = p_min, p_max
        self.v_min, self.v_max = v_min, v_max
        self.t_min, self.t_max = t_min, t_max
        self.kp_max, self.kd_max = kp_max, kd_max


SPECS = {
    "rs03": MotorSpec("RS03", -12.57, 12.57, -20.0, 20.0, -60.0, 60.0, 5000.0, 100.0),
    "rs02": MotorSpec("RS02", -12.57, 12.57, -44.0, 44.0, -17.0, 17.0, 500.0, 5.0),
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def float_to_uint(x, x_min, x_max, bits):
    x = clamp(x, x_min, x_max)
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))


def build_arb(comm_type, data16, target_id):
    return ((comm_type & 0x1F) << 24) | ((data16 & 0xFFFF) << 8) | (target_id & 0xFF)


def parse_arb(arbitration_id):
    comm_type = (arbitration_id >> 24) & 0x1F
    data16 = (arbitration_id >> 8) & 0xFFFF
    destination = arbitration_id & 0xFF
    return comm_type, data16, destination


class Motor:
    def __init__(self, bus, motor_id, spec, host_id=HOST_ID):
        self.bus = bus
        self.motor_id = motor_id
        self.spec = spec
        self.host_id = host_id
        self.last_velocity = 0.0
        self.last_position = None
        self.last_torque = 0.0
        self.last_temp = 0.0
        self.last_feedback_time = 0.0

    def _send(self, comm_type, data16, data):
        self.bus.send(can.Message(arbitration_id=build_arb(comm_type, data16, self.motor_id),
                                  data=bytes(data), is_extended_id=True))

    def read_mech_position(self, timeout=0.2):
        data = bytearray(8)
        struct.pack_into("<H", data, 0, MECH_POS_INDEX)
        self._send(0x11, self.host_id, data)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if msg is None or not msg.is_extended_id:
                continue
            comm_type, data16, destination = parse_arb(msg.arbitration_id)
            if comm_type != 0x11 or destination != self.host_id or (data16 & 0xFF) != self.motor_id:
                continue
            payload = bytes(msg.data)
            if len(payload) >= 8 and int.from_bytes(payload[0:2], "little") == MECH_POS_INDEX:
                return struct.unpack_from("<f", payload, 4)[0]
        return None

    def write_run_mode_operation(self):
        data = bytearray(8)
        struct.pack_into("<H", data, 0, RUN_MODE_INDEX)
        data[4] = OPERATION_RUN_MODE & 0xFF
        self._send(0x12, self.host_id, data)

    def enable(self):
        self._send(0x03, self.host_id, bytes(8))

    def stop(self, clear_fault=False):
        data = bytearray(8)
        if clear_fault:
            data[0] = 1
        self._send(0x04, self.host_id, data)

    def control(self, pos, vel, kp, kd, torque=0.0):
        s = self.spec
        data16 = float_to_uint(torque, s.t_min, s.t_max, 16)
        raw_pos = float_to_uint(pos, s.p_min, s.p_max, 16)
        raw_vel = float_to_uint(vel, s.v_min, s.v_max, 16)
        raw_kp = float_to_uint(kp, 0.0, s.kp_max, 16)
        raw_kd = float_to_uint(kd, 0.0, s.kd_max, 16)
        data = bytes([
            (raw_pos >> 8) & 0xFF, raw_pos & 0xFF,
            (raw_vel >> 8) & 0xFF, raw_vel & 0xFF,
            (raw_kp >> 8) & 0xFF, raw_kp & 0xFF,
            (raw_kd >> 8) & 0xFF, raw_kd & 0xFF,
        ])
        self._send(0x01, data16, data)

    def ingest_feedback(self, data, now=None):
        s = self.spec
        if len(data) < 8:
            return
        raw_pos = (data[0] << 8) | data[1]
        raw_vel = (data[2] << 8) | data[3]
        raw_torque = (data[4] << 8) | data[5]
        raw_temp = (data[6] << 8) | data[7]
        self.last_position = raw_pos / 65535.0 * (s.p_max - s.p_min) + s.p_min
        self.last_velocity = raw_vel / 65535.0 * (s.v_max - s.v_min) + s.v_min
        self.last_torque = raw_torque / 65535.0 * (s.t_max - s.t_min) + s.t_min
        self.last_temp = raw_temp / 10.0
        self.last_feedback_time = time.monotonic() if now is None else now


class FeedbackHub:


    def __init__(self, bus, motors, host_id):
        self.bus = bus
        self.motors = {m.motor_id: m for m in motors}
        self.host_id = host_id

    def _route(self, msg, now):
        if not msg.is_extended_id:
            return None
        comm_type, data16, destination = parse_arb(msg.arbitration_id)
        if comm_type != 0x02 or destination != self.host_id:
            return None
        motor = self.motors.get(data16 & 0xFF)
        if motor is not None:
            motor.ingest_feedback(bytes(msg.data), now)
        return motor

    def pump(self, max_frames=512):
        now = time.monotonic()
        for _ in range(max_frames):
            msg = self.bus.recv(timeout=0.0)
            if msg is None:
                return
            self._route(msg, now)

    def wait_for(self, motor_id, timeout):
        deadline = time.monotonic() + timeout
        target = self.motors.get(motor_id)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            msg = self.bus.recv(timeout=remaining)
            if msg is None:
                return None
            if self._route(msg, time.monotonic()) is target and target is not None:
                return target.last_position


JOINT_LIMITS_RAD = {
    1: (-0.698132, 0.698132),
    2: (-0.872665, 0.872665),
    3: (-1.047198, 0.087266),
    4: (-0.872665, 0.087266),
    5: (-0.610865, 0.436332),
    6: (-0.436332, 0.610865),
    7: (-0.698132, 0.698132),
    8: (-0.872665, 0.872665),
    9: (-0.087266, 1.047198),
    10: (-0.087266, 0.872665),
    11: (-0.436332, 0.610865),
    12: (-0.610865, 0.436332),
}

DEFAULT_LIMIT_MARGIN_RAD = 0.05


def joint_limit_for(motor_id, margin=DEFAULT_LIMIT_MARGIN_RAD):
    lo, hi = JOINT_LIMITS_RAD[motor_id]
    return lo + margin, hi - margin


def exceeds_joint_limit(pos, motor_id, margin=DEFAULT_LIMIT_MARGIN_RAD):
    lo, hi = joint_limit_for(motor_id, margin)
    return pos <= lo or pos >= hi
