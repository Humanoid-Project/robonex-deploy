from robonex_common.can import FeedbackHub, Motor
from robonex_common.actuators import CONTROL_GAINS_BY_JOINT
from robonex_common.joints import ALL_MOTORS, JOINT_BY_ID, MOTOR_BY_ID, MOTOR_LIMITS_BY_ID
from robonex_common.joints import channel_for_motor_id as channel_for_id
from robonex_common.limits import DEFAULT_LIMIT_MARGIN_RAD, exceeds_joint_limit, joint_limit_for
from robonex_common.motors import MOTOR_CONTROL_KD, MOTOR_CONTROL_KP, MOTOR_SPECS
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


def motor_gains(motor_ids, scale=1.0, fallback=(MOTOR_CONTROL_KP, MOTOR_CONTROL_KD)):
    """Per-motor {id: kp}, {id: kd} from the shared per-joint table; motors the table does not cover get `fallback`."""
    kp_by_motor = {}
    kd_by_motor = {}
    for mid in motor_ids:
        kp, kd = CONTROL_GAINS_BY_JOINT.get(MOTOR_BY_ID[mid].model_name, fallback)
        kp_by_motor[mid] = kp * scale
        kd_by_motor[mid] = kd * scale
    return kp_by_motor, kd_by_motor


def gains_without_table_entry(motor_ids):
    return [mid for mid in motor_ids if MOTOR_BY_ID[mid].model_name not in CONTROL_GAINS_BY_JOINT]


def stop_idle_motors(buses, active_ids, host_id=HOST_ID):
    """Send one stop frame to every registered motor that shares an open bus but is not being controlled."""
    stopped = []
    for joint in ALL_MOTORS:
        if joint.motor_id in active_ids or joint.channel not in buses:
            continue
        Motor(buses[joint.channel], joint.motor_id, SPECS[joint.motor_model], host_id=host_id).stop()
        stopped.append(joint.motor_id)
    return stopped
build_arb = build_arbitration_id
parse_arb = parse_arbitration_id
