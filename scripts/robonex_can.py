from robonex_common.can import FeedbackHub, Motor
from robonex_common.joints import ALL_MOTORS, JOINT_BY_ID, MOTOR_LIMITS_BY_ID
from robonex_common.joints import channel_for_motor_id as channel_for_id
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
JOINT_LIMITS_RAD = MOTOR_LIMITS_BY_ID
JOINT_MAP = {joint.motor_id: joint.hardware_name for joint in ALL_MOTORS}
MOTOR_MODELS = {joint.motor_id: joint.motor_model for joint in ALL_MOTORS}
MOTOR_ACTUATORS = {joint.motor_id: joint.model_name.removesuffix("_joint") for joint in ALL_MOTORS}
POLICY_MOTOR_IDS = frozenset(JOINT_BY_ID)
build_arb = build_arbitration_id
parse_arb = parse_arbitration_id
