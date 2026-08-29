from robonex_common.can import FeedbackHub, Motor
from robonex_common.joints import ACTUATED_JOINTS, JOINT_LIMITS_BY_ID
from robonex_common.limits import DEFAULT_LIMIT_MARGIN_RAD, exceeds_joint_limit, joint_limit_for
from robonex_common.motors import MOTOR_SPECS
from robonex_common.protocol import (
    DEFAULT_INTERFACE,
    HOST_ID,
    MECHANICAL_POSITION_INDEX,
    RUN_MODE_INDEX,
    RUN_MODE_OPERATION,
    build_arbitration_id,
    clamp,
    float_to_uint,
    parse_arbitration_id,
)

OPERATION_RUN_MODE = RUN_MODE_OPERATION
MECH_POS_INDEX = MECHANICAL_POSITION_INDEX
SPECS = MOTOR_SPECS
JOINT_LIMITS_RAD = JOINT_LIMITS_BY_ID
JOINT_MAP = {joint.motor_id: joint.hardware_name for joint in ACTUATED_JOINTS}
MOTOR_MODELS = {joint.motor_id: joint.motor_model for joint in ACTUATED_JOINTS}
MOTOR_ACTUATORS = {joint.motor_id: joint.model_name.removesuffix("_joint") for joint in ACTUATED_JOINTS}
CHANNEL_ID_RANGES = {
    "can0": range(1, 7),
    "can1": range(7, 13),
}

MOTORS = {
    1: {"target_rad": 0.0119, "model": "rs02"},
    2: {"target_rad": 0.4013, "model": "rs03"},
    3: {"target_rad": -0.1473, "model": "rs03"},
    4: {"target_rad": -0.5475, "model": "rs03"},
    5: {"target_rad": 0.0, "model": "rs02"},
    6: {"target_rad": 0.0, "model": "rs02"},
    7: {"target_rad": 0.0016, "model": "rs02"},
    8: {"target_rad": -0.6227, "model": "rs03"},
    9: {"target_rad": 0.0114, "model": "rs03"},
    10: {"target_rad": 0.7454, "model": "rs03"},
    11: {"target_rad": 0.0, "model": "rs02"},
    12: {"target_rad": 0.0001, "model": "rs02"},
}

MOVE_SPEED = 0.4
MIN_MOVE_TIME = 3.0
RATE = 100.0
HOLD_KP = 40.0
HOLD_KD = 2.0
OVERSPEED_STOP = 2.0
FEEDBACK_TIMEOUT = 0.3


def channel_for_id(motor_id):
    for channel, id_range in CHANNEL_ID_RANGES.items():
        if motor_id in id_range:
            return channel
    raise ValueError(f"No CAN channel for motor ID {motor_id}")


build_arb = build_arbitration_id
parse_arb = parse_arbitration_id
