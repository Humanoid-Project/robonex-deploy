"""Shared hardware-facing helpers for the sim_to_real teaching tools.

Both mujoco_to_real.py and process_mujoco_to_real.py drive the real motors through
these; the per-script speed/accel caps stay in the scripts themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import can
import mujoco
import numpy as np

from robonex_can import (
    FeedbackHub,
    JOINT_LIMITS_RAD,
    JOINT_MAP,
    MOTOR_ACTUATORS,
    MOTOR_MODELS,
    Motor,
    SPECS,
    channel_for_id,
    clamp,
)


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def align_angle(current, target):
    return current + wrap_to_pi(target - current)

@dataclass
class AxisLimiter:

    position: float
    velocity: float = 0.0

    def step(self, target: float, dt: float, max_speed: float, max_accel: float):
        if not all(math.isfinite(v) for v in (target, dt, max_speed, max_accel)):
            raise ValueError("rate limiter received a non-finite value")
        if dt <= 0.0 or max_speed <= 0.0 or max_accel <= 0.0:
            raise ValueError("dt, max_speed and max_accel must be positive")

        error = target - self.position
        if abs(error) <= 1e-12 and abs(self.velocity) <= 1e-12:
            self.position = target
            self.velocity = 0.0
            return self.position, self.velocity

        braking_speed = max(
            0.0,
            math.sqrt((max_accel * dt) ** 2 + 2.0 * max_accel * abs(error))
            - max_accel * dt,
        )
        desired_velocity = math.copysign(min(max_speed, braking_speed), error)
        dv = clamp(desired_velocity - self.velocity, -max_accel * dt, max_accel * dt)
        next_velocity = clamp(self.velocity + dv, -max_speed, max_speed)
        next_position = self.position + next_velocity * dt

        if error * (target - next_position) <= 0.0:
            next_position = target
            next_velocity = 0.0

        self.position = next_position
        self.velocity = next_velocity
        return self.position, self.velocity

def safe_limits(motor_ids, margin_rad):
    result = {}
    for mid in motor_ids:
        lower, upper = JOINT_LIMITS_RAD[mid]
        inner = (lower + margin_rad, upper - margin_rad)
        if inner[0] >= inner[1]:
            raise ValueError(f"ID {mid}: limit margin is larger than the joint range")
        result[mid] = inner
    return result

def verify_model_limits(model, actuator_ids, motor_ids):
    for mid in motor_ids:
        got = tuple(float(v) for v in model.actuator_ctrlrange[actuator_ids[mid]])
        want = JOINT_LIMITS_RAD[mid]
        if not np.allclose(got, want, atol=5e-6, rtol=0.0):
            raise RuntimeError(
                f"ID {mid} limit mismatch: MuJoCo={got}, common={want}. "
                "Use matching source and generated models."
            )

def load_fixed_model(path, motor_ids):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(
            f"Fixed-base model not found: {path}\n"
            "Build it in robonex-description before running this tool."
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root") >= 0:
        raise RuntimeError(f"Model has a free root joint; use a fixed-base scene: {path}")

    actuator_ids = {}
    for mid in motor_ids:
        actuator_name = MOTOR_ACTUATORS[mid]
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if aid < 0:
            raise RuntimeError(f"MuJoCo actuator not found: ID {mid} -> {actuator_name}")
        joint_name = actuator_name + "_joint"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0 or int(model.actuator_trnid[aid, 0]) != jid:
            raise RuntimeError(f"MuJoCo actuator-joint mismatch: {actuator_name} -> {joint_name}")
        actuator_ids[mid] = aid
    return path, model, actuator_ids

def open_hardware(motor_ids, interface, host_id):
    buses = {}
    motors = {}
    channels = sorted({channel_for_id(mid) for mid in motor_ids})
    try:
        for channel in channels:
            buses[channel] = can.Bus(channel=channel, interface=interface)
        for mid in motor_ids:
            motors[mid] = Motor(
                buses[channel_for_id(mid)], mid, SPECS[MOTOR_MODELS[mid]], host_id=host_id
            )
    except Exception:
        for bus in buses.values():
            try:
                bus.shutdown()
            except Exception:
                pass
        raise
    hubs = {
        channel: FeedbackHub(
            bus,
            [motor for mid, motor in motors.items() if channel_for_id(mid) == channel],
            host_id,
        )
        for channel, bus in buses.items()
    }
    return buses, motors, hubs

def enable_with_runtime_feedback(motors, hubs, kp, kd, limits):
    starts = {}
    enabled_ids = []

    stop_errors = []
    for mid, motor in motors.items():
        try:
            motor.stop()
        except (OSError, can.CanError) as error:
            stop_errors.append(f"ID {mid}: {error}")
    if stop_errors:
        raise RuntimeError("Preflight stop failed: " + "; ".join(stop_errors))
    time.sleep(0.05)

    for mid in sorted(motors):
        motor = motors[mid]
        motor.write_run_mode_operation()
        time.sleep(0.005)
        motor.enable()
        enabled_ids.append(mid)
        hub = hubs[channel_for_id(mid)]
        start = hub.wait_for(mid, timeout=0.3)
        if start is None:
            error = RuntimeError(f"ID {mid} has no type-0x02 feedback after enable")
            error.enabled_ids = enabled_ids
            raise error
        lower, upper = limits[mid]
        wrapped = wrap_to_pi(start) if math.isfinite(start) else start
        if not math.isfinite(start) or not lower <= wrapped <= upper:
            error = RuntimeError(
                f"ID {mid} feedback is non-finite or outside the safe range after enable: "
                f"{math.degrees(start):+.3f}deg (wrap {math.degrees(wrapped):+.3f}deg)"
            )
            error.enabled_ids = enabled_ids
            raise error
        starts[mid] = start
        motor.control(pos=start, vel=0.0, kp=kp, kd=kd, torque=0.0)
    return starts, enabled_ids

def inspect_zero_positions(motors, tolerance_rad, limits):
    print("\nPreflight zero-position check using read-only mechPos:")
    blocking_failures = []
    measured = {}
    print(f"  {'ID':>2} {'joint':<18} {'raw/wrap':>18} {'move to zero':>15}  status")
    for mid in sorted(motors):
        position = motors[mid].read_mech_position(timeout=0.3)
        measured[mid] = position
        if position is None:
            blocking_failures.append(f"ID {mid} did not respond")
            print(f"  {mid:>2} {JOINT_MAP[mid]:<18} {'no response':>15} {'--':>15}  BLOCK")
            continue
        if not math.isfinite(position):
            blocking_failures.append(f"ID {mid} mechPos is NaN or infinite")
            print(f"  {mid:>2} {JOINT_MAP[mid]:<18} {'NaN/inf':>15} {'--':>15}  BLOCK")
            continue
        wrapped = wrap_to_pi(position)
        degrees = math.degrees(position)
        wrapped_deg = math.degrees(wrapped)
        lower, upper = limits[mid]
        if not lower <= wrapped <= upper:
            status = "BLOCK"
            blocking_failures.append(
                f"ID {mid} {degrees:+.3f} deg (wrapped {wrapped_deg:+.3f} deg) is outside "
                f"{math.degrees(lower):+.3f}..{math.degrees(upper):+.3f} deg"
            )
        elif abs(wrapped) <= tolerance_rad:
            status = "near zero"
        else:
            status = "will move to zero"
        print(
            f"  {mid:>2} {JOINT_MAP[mid]:<18} "
            f"{degrees:+7.2f}/{wrapped_deg:+7.2f}deg "
            f"{(-wrapped_deg):+11.3f} deg  {status}"
        )
    return measured, blocking_failures

def runtime_safety_reason(motors, commands, limits, now, args):
    for mid, motor in motors.items():
        if motor.last_feedback_time <= 0.0 or motor.last_position is None:
            return f"ID {mid} never received type-0x02 feedback"
        age = now - motor.last_feedback_time
        if age > args.feedback_timeout:
            return f"ID {mid} feedback timeout ({age:.3f} s > {args.feedback_timeout:.3f} s)"
        if not all(math.isfinite(v) for v in (
            motor.last_position, motor.last_velocity, motor.last_torque, motor.last_temp
        )):
            return f"ID {mid} feedback contains NaN or infinity"
        if abs(motor.last_velocity) > args.overspeed:
            return f"ID {mid} overspeed ({motor.last_velocity:+.3f} rad/s)"
        if motor.last_temp > args.max_temp:
            return f"ID {mid} overtemperature ({motor.last_temp:.1f} degC)"
        lower, upper = limits[mid]
        wrapped = wrap_to_pi(motor.last_position)
        if not lower <= wrapped <= upper:
            return (
                f"ID {mid} left the safe joint range ("
                f"{math.degrees(motor.last_position):+.2f}deg "
                f"wrap {math.degrees(wrapped):+.2f}deg, "
                f"allowed {math.degrees(lower):+.2f}..{math.degrees(upper):+.2f} deg)"
            )
        if mid in commands:
            error = wrap_to_pi(commands[mid] - motor.last_position)
            if abs(error) > math.radians(args.max_error_deg):
                return f"ID {mid} tracking error is too large ({math.degrees(error):+.2f} deg)"
    return None

def brake_and_stop(motors, buses, enabled_ids, stop_ids, duration, kd):
    enabled = [mid for mid in sorted(set(enabled_ids)) if mid in motors]
    if enabled and duration > 0.0:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            for mid in enabled:
                try:
                    motor = motors[mid]
                    motor.control(
                        pos=0.0, vel=0.0, kp=0.0,
                        kd=min(kd, motor.spec.kd_max), torque=0.0,
                    )
                except (OSError, can.CanError):
                    pass
            time.sleep(0.01)

    for mid in sorted(set(stop_ids)):
        motor = motors.get(mid)
        if motor is None:
            continue
        try:
            motor.stop()
        except (OSError, can.CanError):
            pass
    for bus in buses.values():
        try:
            bus.shutdown()
        except Exception:
            pass
