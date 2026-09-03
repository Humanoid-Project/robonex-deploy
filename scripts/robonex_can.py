from robonex_common.can import FeedbackHub, Motor
from robonex_common.joints import ACTUATED_JOINTS, JOINT_LIMITS_BY_ID
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
JOINT_LIMITS_RAD = JOINT_LIMITS_BY_ID
JOINT_MAP = {joint.motor_id: joint.hardware_name for joint in ACTUATED_JOINTS}
MOTOR_MODELS = {joint.motor_id: joint.motor_model for joint in ACTUATED_JOINTS}
MOTOR_ACTUATORS = {joint.motor_id: joint.model_name.removesuffix("_joint") for joint in ACTUATED_JOINTS}
build_arb = build_arbitration_id
parse_arb = parse_arbitration_id
