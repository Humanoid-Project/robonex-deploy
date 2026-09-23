"""Shared hardware-facing helpers for the sim_to_real teaching tools.

Both mujoco_to_real.py and process_mujoco_to_real.py drive the real motors through
these; the per-script speed/accel caps stay in the scripts themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import signal
import threading
import time

import can
import mujoco
import numpy as np
from robonex_common.can import drain

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


MODE_RUNNING = 2


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

def gain_for(gains, motor_id):
    """Accept either a scalar gain or a per-motor {id: value} mapping."""
    if isinstance(gains, dict):
        return gains[motor_id]
    return gains


def enable_with_runtime_feedback(motors, hubs, kp, kd, limits, enabled_out=None):
    starts = {}
    enabled_ids = enabled_out if enabled_out is not None else []

    stop_errors = []
    for mid, motor in motors.items():
        try:
            motor.stop()
        except (OSError, can.CanError) as error:
            stop_errors.append(f"ID {mid}: {error}")
    if stop_errors:
        raise RuntimeError("Preflight stop failed: " + "; ".join(stop_errors))
    time.sleep(0.05)
    for hub in hubs.values():
        drain(hub.bus)

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
        if math.isfinite(start) and abs(start) > math.pi:
            error = RuntimeError(
                f"ID {mid} type-0x02 position {math.degrees(start):+.3f}deg is outside "
                "-180..+180 deg; set zero_sta=1 (0x7029) on this motor"
            )
            error.enabled_ids = enabled_ids
            raise error
        if not math.isfinite(start) or not lower <= wrapped <= upper:
            error = RuntimeError(
                f"ID {mid} feedback is non-finite or outside the safe range after enable: "
                f"{math.degrees(start):+.3f}deg (wrap {math.degrees(wrapped):+.3f}deg)"
            )
            error.enabled_ids = enabled_ids
            raise error
        starts[mid] = start
        motor.control(
            pos=start, vel=0.0,
            kp=gain_for(kp, mid), kd=gain_for(kd, mid), torque=0.0,
        )
        deadline = time.monotonic() + 0.3
        while motor.last_mode_status != MODE_RUNNING and time.monotonic() < deadline:
            if hub.wait_for(mid, timeout=max(0.0, deadline - time.monotonic())) is None:
                break
        if motor.last_mode_status != MODE_RUNNING:
            error = RuntimeError(
                f"ID {mid} is not running after enable "
                f"(mode {motor.last_mode_status}, expected {MODE_RUNNING})"
            )
            error.enabled_ids = enabled_ids
            raise error
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
        if abs(position) > math.pi:
            status = "BLOCK"
            blocking_failures.append(
                f"ID {mid} raw mechPos {degrees:+.3f} deg is outside -180..+180 deg; "
                "set zero_sta=1 (0x7029) on this motor"
            )
        elif not lower <= wrapped <= upper:
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

class ThermalLoad:
    """Accumulate time spent above the continuous torque rating, and leak it back.

    `runtime_safety_reason` checks instantaneous torque for finiteness only, so a motor can sit
    above its continuous rating indefinitely with no software action until it reaches `max_temp`.
    A continuous rating is not an instantaneous limit, so the magnitude alone means nothing --
    what matters is magnitude times duration, which is what this integrates.

    The accumulator is `d/dt A = max((tau/rated)^2 - 1, 0) - A/tau_leak`, in units of
    rated-squared-seconds, evaluated per motor.
    """

    def __init__(self, rated_by_id, budget=30.0, leak_s=30.0):
        self.rated = dict(rated_by_id)
        self.budget = budget
        self.leak_s = leak_s
        self.load = {mid: 0.0 for mid in rated_by_id}
        self.peak = {mid: 0.0 for mid in rated_by_id}

    def update(self, motors, dt):
        if not math.isfinite(dt) or dt <= 0.0:
            return
        for mid, rated in self.rated.items():
            motor = motors.get(mid)
            torque = getattr(motor, "last_torque", None) if motor is not None else None
            if torque is None or not math.isfinite(torque) or rated <= 0.0:
                continue
            excess = max((torque / rated) ** 2 - 1.0, 0.0)
            value = self.load[mid] + (excess - self.load[mid] / self.leak_s) * dt
            self.load[mid] = max(value, 0.0)
            if self.load[mid] > self.peak[mid]:
                self.peak[mid] = self.load[mid]

    def reason(self):
        for mid in sorted(self.load):
            if self.load[mid] >= self.budget:
                return (f"ID {mid} has been over its {self.rated[mid]:.0f} N.m continuous rating "
                        f"for too long (thermal load {self.load[mid]:.1f} of {self.budget:.0f})")
        return None

    def worst_line(self):
        if not self.peak:
            return "thermal load: no motors tracked"
        mid = max(self.peak, key=self.peak.get)
        return (f"thermal load peak {self.peak[mid]:.2f} of {self.budget:.0f} on ID {mid} "
                f"(now {self.load[mid]:.2f})")


def runtime_safety_reason(motors, commands, limits, now, args):
    for mid, motor in motors.items():
        if motor.last_feedback_time <= 0.0 or motor.last_position is None:
            return f"ID {mid} never received type-0x02 feedback"
        age = now - motor.last_feedback_time
        if age > args.feedback_timeout:
            return f"ID {mid} feedback timeout ({age:.3f} s > {args.feedback_timeout:.3f} s)"
        if motor.last_mode_status != MODE_RUNNING:
            return f"ID {mid} left run mode (mode {motor.last_mode_status})"
        if not all(math.isfinite(v) for v in (
            motor.last_position, motor.last_velocity, motor.last_torque, motor.last_temp
        )):
            return f"ID {mid} feedback contains NaN or infinity"
        if abs(motor.last_velocity) > args.overspeed:
            return f"ID {mid} overspeed ({motor.last_velocity:+.3f} rad/s)"
        if motor.last_temp >= args.max_temp:
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

def tilt_reason(gravity, max_tilt_deg):
    """Stop once the trunk is further from vertical than a walking robot ever gets.

    Nothing else in `runtime_safety_reason` looks at attitude: the IMU is read for the
    policy and checked for staleness, but its value is never a stop condition. On the
    stand that did not matter because the ropes caught a fall. Untethered it does -- a
    robot lying on the floor keeps being commanded to its policy targets, which is high
    torque at zero speed, the worst case for motor heating.

    The threshold is on the angle between measured gravity and straight down. The three
    recorded hardware runs reach 6.88 deg while walking and 26.12 deg standing under a
    push, so 30 deg clears normal operation and still fires long before the robot is flat.
    """
    if gravity is None or len(gravity) < 3:
        return "IMU returned no gravity vector"
    if not all(math.isfinite(v) for v in gravity[:3]):
        return "IMU gravity contains NaN or infinity"
    norm = math.sqrt(sum(v * v for v in gravity[:3]))
    if norm < 0.5:
        return f"IMU gravity vector is degenerate (norm {norm:.3f})"
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gravity[2] / norm))))
    if tilt > max_tilt_deg:
        return f"the robot is {tilt:.1f} deg from vertical (limit {max_tilt_deg:.0f} deg)"
    return None


def brake_and_stop(motors, buses, enabled_ids, stop_ids, duration, kd):
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return _brake_and_stop(motors, buses, enabled_ids, stop_ids, duration, kd)
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def _brake_and_stop(motors, buses, enabled_ids, stop_ids, duration, kd):
    damping_errors = {}
    stop_errors = {}
    stop_sent = []
    enabled = [mid for mid in sorted(set(enabled_ids)) if mid in motors]
    if enabled and duration > 0.0:
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline:
                for mid in enabled:
                    try:
                        motor = motors[mid]
                        motor.control(
                            pos=0.0, vel=0.0, kp=0.0,
                            kd=min(gain_for(kd, mid), motor.spec.kd_max), torque=0.0,
                        )
                    except (OSError, can.CanError) as error:
                        damping_errors[mid] = error

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

    for mid in sorted(set(stop_ids)):
        motor = motors.get(mid)
        if motor is None:
            continue
        try:
            motor.stop()
        except (OSError, can.CanError, KeyboardInterrupt) as error:
            stop_errors[mid] = error
        else:
            stop_sent.append(mid)
    for bus in buses.values():
        try:
            bus.shutdown()
        except Exception:
            pass
    return {
        "damped": enabled,
        "damping_errors": damping_errors,
        "stop_sent": stop_sent,
        "stop_errors": stop_errors,
    }


def shutdown_report_lines(report):
    """Describe a `brake_and_stop` result without claiming more than was verified."""
    lines = []
    stop_errors = report["stop_errors"]
    if stop_errors:
        failed = ", ".join(str(mid) for mid in sorted(stop_errors))
        lines.append(
            f"WARNING: the stop frame failed for ID {failed}. Those motors may still be "
            "enabled. Cut power at the supply before approaching the robot."
        )
        for mid in sorted(stop_errors):
            lines.append(f"  ID {mid} stop: {stop_errors[mid]}")
    damping_errors = report["damping_errors"]
    if damping_errors:
        failed = ", ".join(str(mid) for mid in sorted(damping_errors))
        lines.append(f"WARNING: active damping failed for ID {failed}.")
    if report["stop_sent"]:
        sent = ", ".join(str(mid) for mid in report["stop_sent"])
        lines.append(
            f"Stop frame sent to ID {sent}. The motors do not acknowledge it, so delivery "
            "is unconfirmed."
        )
    return lines
