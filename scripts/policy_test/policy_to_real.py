#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import json
from datetime import datetime, timezone
import math
import os
import platform
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path

PROGRAM_STARTED = time.monotonic()

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "sim_to_real"))
sys.path.insert(0, str(THIS_FILE.parent))

try:
    import can
    import numpy as np
    import onnxruntime as ort
except ImportError as error:
    raise SystemExit(f"Missing required package: {error}")

try:
    import n100
except ImportError as error:
    raise SystemExit(
        f"N100 IMU extension not importable: {error}\n"
        "Build it first:\n"
        f"  cd {THIS_FILE.parent}\n"
        "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build -j"
    )

from robonex_can import (
    DEFAULT_INTERFACE,
    HOST_ID,
    MECH_POS_INDEX,
    MOTOR_MODELS,
    stop_idle_motors,
    Motor,
    SPECS,
    clamp,
)
import robonex_common
from robonex_common.imu import DEFAULT_IMU_BAUDRATE, DEFAULT_IMU_PORT, MOUNT_ROLL_DEG
from robonex_common.joints import CHANNEL_MOTOR_IDS, JOINT_BY_MODEL_NAME, JOINT_BY_ID
from robonex_common.models import robot_model
from robonex_common.actuators import CONTROL_GAINS_BY_JOINT
from robonex_common.motors import MOTOR_CONTROL_KD, MOTOR_CONTROL_KP, RATED_TORQUE
from robonex_common.policy import PolicyContract, python_source_sha256
from robonex_common.protocol import (
    CURRENT_LIMIT_INDEX,
    MECHANICAL_VELOCITY_INDEX,
    RUN_MODE_INDEX,
    TORQUE_LIMIT_INDEX,
    VBUS_INDEX,
    ZERO_STATUS_INDEX,
)
from robonex_common.runtime import (
    OBSERVATION_TERM_SIZES,
    ActionPipeline,
    ObservationHistory,
    assemble_observation,
    gait_phase_at,
)
from safety import (
    AxisLimiter,
    ThermalLoad,
    align_angle,
    brake_and_stop,
    enable_with_runtime_feedback,
    inspect_zero_positions,
    open_hardware,
    require_robot_model,
    roll_pairs_for,
    runtime_safety_reason,
    shutdown_report_lines,
    start_roll_blocks,
    tilt_reason,
    wrap_to_pi,
)

DEG = math.pi / 180.0
CLEAR_SCREEN = "\033[2J\033[3J\033[H"


class TeeStream:
    def __init__(self, terminal, log, lock):
        self.terminal = terminal
        self.log = log
        self.lock = lock

    def write(self, value):
        with self.lock:
            written = self.terminal.write(value)
            self.log.write(value.replace(CLEAR_SCREEN, ""))
            self.log.flush()
        return written

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def resolve_gains(scale):
    """Per-motor (kp, kd) from the shared per-joint table, scaled.

    The policy is trained against CONTROL_GAINS_BY_JOINT (hip 100/2, knee 150/4,
    ankle 40/2); commanding one scalar 40/2 drives the knee at 27% of the stiffness
    the policy assumes, so the joint does not reach the target the policy chose.
    """
    kp_by_motor = {}
    kd_by_motor = {}
    for name, spec in JOINT_BY_MODEL_NAME.items():
        kp, kd = CONTROL_GAINS_BY_JOINT.get(name, (MOTOR_CONTROL_KP, MOTOR_CONTROL_KD))
        kp_by_motor[spec.motor_id] = kp * scale
        kd_by_motor[spec.motor_id] = kd * scale
    return kp_by_motor, kd_by_motor


@dataclass(frozen=True)
class Settings:
    interface: str = DEFAULT_INTERFACE
    host_id: int = HOST_ID
    kp: float = MOTOR_CONTROL_KP
    kd: float = MOTOR_CONTROL_KD
    # Fraction of the per-joint gains in CONTROL_GAINS_BY_JOINT actually commanded.
    # The trained policy assumes those gains; the measured-hardware scalars above are
    # only the fallback for a joint the table does not cover. Ramp this up from a low
    # value on the stand before running at 1.0.
    gain_scale: float = 1.0

    read_poll_timeout: float = 0.02
    read_print_hz: float = 10.0

    status_hz: float = 10.0
    feedback_timeout: float = 0.10
    imu_stale_timeout: float = 0.10
    overspeed: float = 10.0
    max_temp: float = 70.0
    max_error_deg: float = 25.0
    max_tilt_deg: float = 40.0
    max_raw_action: float = 20.0

    approach_max_speed: float = 0.30
    approach_max_accel: float = 0.60
    approach_tolerance_deg: float = 1.0
    approach_settle_timeout: float = 5.0

    policy_max_speed: float = 6.0
    policy_max_accel: float = 120.0
    ramp_seconds: float = 1.0
    command_hold_seconds: float = 0.5

    imu_calibration_seconds: float = 2.0
    brake_time: float = 0.20


SETTINGS = Settings()

GAIT_COMMAND_DEADBAND = 0.05
ENABLE_KP, ENABLE_KD = resolve_gains(SETTINGS.gain_scale)


def joint_row_order(contract):
    return [(name, JOINT_BY_MODEL_NAME[name].motor_id) for name in contract.joint_order]


def roll_reprojector(pipeline, motor_ids):
    if not pipeline.roll_pairs:
        return None

    def reproject(commands):
        clipped, _ = pipeline.clip_roll([commands[motor_id] for motor_id in motor_ids])
        return {motor_id: float(clipped[index]) for index, motor_id in enumerate(motor_ids)}

    return reproject


class MechPosReader(threading.Thread):
    def __init__(self, channel, motor_ids, settings, notes):
        super().__init__(daemon=True)
        self.channel = channel
        self.motor_ids = tuple(motor_ids)
        self.settings = settings
        self.notes = notes
        self.rate_hz = 0.0
        self._state = {motor_id: (None, None) for motor_id in motor_ids}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def run(self):
        try:
            bus = can.Bus(channel=self.channel, interface=self.settings.interface)
        except OSError as error:
            self.notes.append(
                f"[{self.channel}] open failed: {error}  "
                f"(sudo ip link set {self.channel} up type can bitrate 1000000)"
            )
            return
        motors = {
            motor_id: Motor(bus, motor_id, SPECS[MOTOR_MODELS[motor_id]], host_id=self.settings.host_id)
            for motor_id in self.motor_ids
        }
        started, cycles = time.monotonic(), 0
        try:
            while not self._stop_event.is_set():
                for motor_id in self.motor_ids:
                    motor = motors[motor_id]
                    position = motor.read_parameter(MECH_POS_INDEX, timeout=self.settings.read_poll_timeout)
                    velocity = motor.read_parameter(
                        MECHANICAL_VELOCITY_INDEX, timeout=self.settings.read_poll_timeout
                    )
                    with self._lock:
                        self._state[motor_id] = (position, velocity)
                cycles += 1
                now = time.monotonic()
                if now - started >= 0.5:
                    self.rate_hz = cycles / (now - started)
                    started, cycles = now, 0
        except can.CanError as error:
            self.notes.append(f"[{self.channel}] CAN error: {error}")
        finally:
            bus.shutdown()


class ReadOnlyJointSource:
    label = "mechPos poll (type 0x11, read-only)"

    def __init__(self, settings, notes):
        self.readers = [
            MechPosReader(channel, motor_ids, settings, notes)
            for channel, motor_ids in CHANNEL_MOTOR_IDS.items()
        ]

    def start(self):
        for reader in self.readers:
            reader.start()

    def stop(self):
        for reader in self.readers:
            reader.stop()
        for reader in self.readers:
            reader.join(timeout=2.0)

    def poll(self):
        return None

    def snapshot(self):
        merged = {}
        for reader in self.readers:
            merged.update(reader.snapshot())
        return merged

    def rate_text(self):
        return "  ".join(f"{reader.channel} {reader.rate_hz:5.1f} Hz" for reader in self.readers)


class RuntimeJointSource:
    label = "type 0x02 runtime feedback"

    def __init__(self, motors, hubs):
        self.motors = motors
        self.hubs = hubs
        self._cycles = 0
        self._started = time.monotonic()
        self.rate_hz = 0.0

    def start(self):
        return None

    def stop(self):
        return None

    def poll(self):
        for hub in self.hubs.values():
            hub.pump()
        self._cycles += 1
        now = time.monotonic()
        if now - self._started >= 0.5:
            self.rate_hz = self._cycles / (now - self._started)
            self._started, self._cycles = now, 0

    def snapshot(self):
        return {
            motor_id: (motor.last_position, motor.last_velocity)
            for motor_id, motor in self.motors.items()
        }

    def rate_text(self):
        return f"loop {self.rate_hz:5.1f} Hz"


class ImuSource:
    def __init__(self, settings, notes):
        self.settings = settings
        self.notes = notes
        self.status = "not started"
        self.driver = None
        self.bias_raw = None
        self._last_seq = None
        self._last_seq_time = 0.0
        self._last_imu_frames = None
        self._last_imu_time = 0.0
        self.last_sample = None

    def start(self, calibrate):
        self.driver = n100.ImuDriver(
            n100.DriverConfig(
                port=DEFAULT_IMU_PORT,
                baudrate=DEFAULT_IMU_BAUDRATE,
                mount_rotation=n100.Quat.from_axis_angle_x(MOUNT_ROLL_DEG * DEG),
            )
        )
        try:
            self.driver.start()
        except RuntimeError as error:
            self.status = "start failed"
            self.notes.append(f"[IMU] {error}")
            self.notes.append(
                f"      ls /dev/ttyUSB* /dev/ttyACM*  (permissions: sudo chmod 666 {DEFAULT_IMU_PORT})"
            )
            return False
        if self.driver.wait_for_sample(timeout=3.0) is None:
            self.status = "no sample in 3 s"
            self.notes.append(f"[IMU] {self.driver.last_error() or 'unknown error'}")
            return False
        deadline = time.monotonic() + 1.0
        while self.driver.stats().imu_frames == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        if self.driver.stats().imu_frames == 0:
            self.status = "no raw IMU frame"
            self.notes.append("[IMU] AHRS samples arrive but no raw IMU frame; the policy needs the raw gyro")
            return False
        if calibrate:
            print(
                f"Calibrating the gyro bias for {self.settings.imu_calibration_seconds:.1f} s. "
                "Keep the robot completely still."
            )
            if not self.driver.calibrate_gyro_bias(self.settings.imu_calibration_seconds):
                self.status = "gyro calibration failed"
                self.notes.append("[IMU] gyro bias calibration received no sample")
                return False
            self.bias_raw = self.driver.gyro_bias_raw
            print(
                f"  raw gyro bias  x {self.bias_raw.x:+.6f}  y {self.bias_raw.y:+.6f}  "
                f"z {self.bias_raw.z:+.6f}  [rad/s]"
            )
        self.status = "ready"
        self._last_seq = None
        self._last_seq_time = time.monotonic()
        self._last_imu_frames = None
        self._last_imu_time = self._last_seq_time
        return True

    def stop(self):
        if self.driver is not None:
            self.driver.stop()

    def read(self, now):
        sample = None if self.driver is None else self.driver.latest()
        self.last_sample = sample
        if sample is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), None
        if sample.seq != self._last_seq:
            self._last_seq = sample.seq
            self._last_seq_time = now
        imu_frames = self.driver.stats().imu_frames
        if sample.has_imu_frame and imu_frames != self._last_imu_frames:
            self._last_imu_frames = imu_frames
            self._last_imu_time = now
        angular_velocity = sample.angular_velocity_raw
        gravity = sample.projected_gravity
        return (
            (angular_velocity.x, angular_velocity.y, angular_velocity.z),
            (gravity.x, gravity.y, gravity.z),
            now - min(self._last_seq_time, self._last_imu_time),
        )

    def failure_reason(self, age):
        if self.driver is None or not self.driver.is_running:
            return f"IMU reader stopped ({self.driver.last_error() if self.driver else 'not started'})"
        if age is None:
            return "IMU produced no sample"
        if age > self.settings.imu_stale_timeout:
            return f"IMU sample is stale ({age:.3f} s > {self.settings.imu_stale_timeout:.3f} s)"
        return None


TIMING_COLUMNS = ("imu_host_age_ms", "imu_device_dt_ms", "imu_host_dt_ms", "imu_seq_gap", "tick_period_ms")
TIMING_SAMPLE_FIELDS = ("seq", "device_timestamp_us", "host_timestamp_ns")


def missing_timing_fields(sample):
    if sample is None:
        return list(TIMING_SAMPLE_FIELDS)
    missing = []
    for name in TIMING_SAMPLE_FIELDS:
        try:
            int(getattr(sample, name))
        except (AttributeError, TypeError, ValueError, OverflowError):
            missing.append(name)
    return missing


class TimingProbe:
    def __init__(self):
        self.previous_tick = None
        self.previous_seq = None
        self.previous_device_us = None
        self.previous_host_ns = None
        self.error = None

    def safe_update(self, tick, sample, read_ns):
        try:
            return self.update(tick, sample, read_ns)
        except Exception as error:
            if self.error is None:
                self.error = f"{type(error).__name__}: {error}"
            return None

    def warning(self):
        if self.error is None:
            return None
        return f"Warning: timing telemetry failed and its cells are blank from the first failure on ({self.error})"

    def update(self, tick, sample, read_ns):
        tick_period_ms = None if self.previous_tick is None else (tick - self.previous_tick) * 1000.0
        self.previous_tick = tick
        if sample is None:
            return (None, None, None, None, tick_period_ms)
        seq = int(sample.seq)
        device_us = int(sample.device_timestamp_us)
        host_ns = int(sample.host_timestamp_ns)
        host_age_ms = (read_ns - host_ns) / 1.0e6
        device_dt_ms = None if self.previous_device_us is None else (device_us - self.previous_device_us) / 1000.0
        host_dt_ms = None if self.previous_host_ns is None else (host_ns - self.previous_host_ns) / 1.0e6
        seq_gap = None if self.previous_seq is None else seq - self.previous_seq - 1
        self.previous_seq = seq
        self.previous_device_us = device_us
        self.previous_host_ns = host_ns
        return (host_age_ms, device_dt_ms, host_dt_ms, seq_gap, tick_period_ms)


def timing_cells(timing):
    if timing is None:
        return [""] * len(TIMING_COLUMNS)
    host_age_ms, device_dt_ms, host_dt_ms, seq_gap, tick_period_ms = timing
    return [_round(host_age_ms, 3), _round(device_dt_ms, 3), _round(host_dt_ms, 3),
            "" if seq_gap is None else int(seq_gap), _round(tick_period_ms, 3)]


IMU_EXTRA_COLUMNS = ("acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z", "imu_temp_c")
HEADING_COLUMNS = (
    "heading_gyro_deg", "heading_quat_deg", "heading_target_deg", "heading_error_deg",
    "heading_integral", "heading_wz", "heading_engaged",
)
LOOP_EXTRA_COLUMNS = (
    "late_ms", "poll_ms", "prev_work_ms", "slew_lag_deg", "runner_clip_total", "target_clip_total",
    "roll_clip_total", "roll_reprojected_total", "thermal_load_max",
)
STEP_EXTRA_COLUMNS = LOOP_EXTRA_COLUMNS + IMU_EXTRA_COLUMNS + HEADING_COLUMNS
JOINT_EXTRA_FIELDS = ("mode", "action", "slew_vel")
READ_JOINT_EXTRA_FIELDS = ("vel", "raw_action", "target")
PHASE_JOINT_FIELDS = ("age_ms", "pos", "vel", "torque", "temp", "fault", "mode", "commanded", "rx_age_ms")


def imu_extra_cells(sample):
    if sample is None:
        return [""] * len(IMU_EXTRA_COLUMNS)
    try:
        acc = sample.linear_acceleration
        quat = sample.orientation
        return [_round(acc.x), _round(acc.y), _round(acc.z),
                _round(quat.w), _round(quat.x), _round(quat.y), _round(quat.z),
                _round(sample.imu_temperature, 3)]
    except Exception:
        return [""] * len(IMU_EXTRA_COLUMNS)


def heading_cells(heading):
    if heading is None:
        return [""] * len(HEADING_COLUMNS)
    try:
        return heading.cells()
    except Exception:
        return [""] * len(HEADING_COLUMNS)


def step_extra_cells(late_ms, poll_ms, prev_work_ms, commander, pipeline, thermal, sample, heading=None):
    try:
        loads = getattr(thermal, "load", None) or {}
        cells = [
            _round(late_ms, 3), _round(poll_ms, 3), _round(prev_work_ms, 3),
            _round(math.degrees(commander.slew_lag_step_max), 4),
            getattr(pipeline, "runner_clip_count", ""),
            getattr(pipeline, "target_clip_count", ""),
            getattr(pipeline, "roll_clip_count", ""),
            getattr(commander, "roll_reprojected_count", ""),
            _round(max(loads.values()) if loads else None, 4),
        ]
    except Exception:
        cells = [""] * len(LOOP_EXTRA_COLUMNS)
    return cells + imu_extra_cells(sample) + heading_cells(heading)


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


PREFLIGHT_PARAMETERS = (
    ("limit_torque_nm", TORQUE_LIMIT_INDEX, "<f"),
    ("limit_cur_a", CURRENT_LIMIT_INDEX, "<f"),
    ("vbus_v", VBUS_INDEX, "<f"),
    ("run_mode", RUN_MODE_INDEX, "<B"),
    ("zero_sta", ZERO_STATUS_INDEX, "<B"),
)


def read_preflight_parameters(motors, timeout=0.05):
    values = {}
    for motor_id in sorted(motors):
        row = {}
        for name, index, fmt in PREFLIGHT_PARAMETERS:
            try:
                value = motors[motor_id].read_parameter(index, fmt=fmt, timeout=timeout)
            except Exception:
                value = None
            if value is None:
                row[name] = None
            elif fmt == "<f":
                row[name] = _finite_or_none(value)
            else:
                row[name] = int(value)
        values[motor_id] = row
    return values


def preflight_parameter_lines(values):
    names = [name for name, _, _ in PREFLIGHT_PARAMETERS]
    lines = ["\nSaved motor parameters (read-only, recorded, not checked):",
             "  " + f"{'ID':>3}  {'model':<5}  " + "  ".join(f"{name:>15}" for name in names)]
    for motor_id, row in sorted(values.items()):
        model = JOINT_BY_ID[motor_id].motor_model if motor_id in JOINT_BY_ID else "?"
        cells = []
        for name in names:
            value = row.get(name)
            cells.append(f"{'--':>15}" if value is None else
                         (f"{value:15.3f}" if isinstance(value, float) else f"{value:15d}"))
        lines.append(f"  {motor_id:>3}  {model:<5}  " + "  ".join(cells))
    return lines


def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def git_state(path):
    try:
        def git(*command):
            return subprocess.run(
                ["git", "-C", str(path), *command], capture_output=True, text=True, timeout=5,
            ).stdout
        return {
            "path": str(path),
            "head": git("rev-parse", "HEAD").strip() or None,
            "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
            "dirty_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"path": str(path), "error": str(error)}


def can_statistics(channels):
    result = {}
    for channel in channels:
        base = Path("/sys/class/net") / channel
        counters = {}
        try:
            for entry in sorted((base / "statistics").iterdir()):
                text = _read_text(entry)
                if text is not None and text.lstrip("-").isdigit():
                    counters[entry.name] = int(text)
        except OSError:
            pass
        result[channel] = {
            "operstate": _read_text(base / "operstate"),
            "tx_queue_len": _read_text(base / "tx_queue_len"),
            "statistics": counters,
        }
    return result


def imu_driver_stats(imu):
    try:
        stats = imu.driver.stats()
    except Exception:
        return None
    names = ("ahrs_frames", "bytes_read", "crc16_errors", "crc8_errors", "dropped_bytes",
             "frame_end_errors", "ground_frames", "imu_frames", "insgps_frames", "samples", "sn_lost")
    return {name: getattr(stats, name, None) for name in names}


def host_snapshot(channels):
    repos = {"deploy": git_state(THIS_FILE.parents[2])}
    try:
        repos["common"] = git_state(Path(robonex_common.__file__).resolve().parents[2])
    except Exception as error:
        repos["common"] = {"error": str(error)}
    try:
        from robonex_common.paths import DESCRIPTION_REPO_NAMES, resolve_repo
        repos["description"] = git_state(resolve_repo(DESCRIPTION_REPO_NAMES, anchors=(THIS_FILE,)))
    except Exception as error:
        repos["description"] = {"error": str(error)}
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "cpu_governor": _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "load_average": load_average,
        "git": repos,
        "can": can_statistics(channels),
    }


class StepArrays:
    DTYPES = {"t_s": np.float64, "wall_time": np.float64, "step": np.int64, "commanded": np.float64}
    MAX_STEPS = 60000

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        self.error = None
        self.steps = 0

    def append(self, **values):
        if self.error is not None:
            return
        if self.steps >= self.MAX_STEPS:
            self.error = f"capacity {self.MAX_STEPS} steps reached; later steps are in the CSV only"
            return
        try:
            converted = {
                key: np.array(value, dtype=self.DTYPES.get(key, np.float32))
                for key, value in values.items()
            }
            for key, value in converted.items():
                self.data.setdefault(key, []).append(value)
            self.steps += 1
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.data = {}

    def save(self):
        if not self.data:
            return None
        if self.error is not None and not self.error.startswith("capacity"):
            return None
        lengths = {len(values) for values in self.data.values()}
        if len(lengths) != 1:
            self.error = f"array lengths differ: {sorted(lengths)}"
            return None
        arrays = {key: np.stack(values) for key, values in self.data.items()}
        with self.path.open("xb") as handle:
            np.savez(handle, **arrays)
        self.data = {}
        return self.path


class PhaseRecorder:
    def __init__(self, path, contract, motors, imu=None):
        self.path = Path(path)
        self.order = joint_row_order(contract)
        self.motors = motors
        self.imu = imu
        self.rows = []
        self.error = None
        self.started = time.monotonic()
        header = ["t_s", "wall_time", "phase", "gravity_x", "gravity_y", "gravity_z",
                  "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
        for name, _ in self.order:
            short = name.replace("_joint", "")
            header += [f"{short}.{field_name}" for field_name in PHASE_JOINT_FIELDS]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", newline="") as handle:
                csv.writer(handle).writerow(header)
        except FileExistsError:
            raise SystemExit(f"{self.path} already exists; recordings are never overwritten.")

    def record(self, phase, commands=None):
        try:
            now = time.monotonic()
            wall = time.time()
            driver = getattr(self.imu, "driver", None)
            sample = driver.latest() if driver is not None else None
            row = [round(now - self.started, 4), round(wall, 6), phase]
            if sample is not None:
                gravity = sample.projected_gravity
                gyro = sample.angular_velocity_raw
                row += [_round(gravity.x), _round(gravity.y), _round(gravity.z),
                        _round(gyro.x), _round(gyro.y), _round(gyro.z)]
            else:
                row += [""] * 6
            row += imu_extra_cells(sample)[:3]
            for _, motor_id in self.order:
                motor = self.motors.get(motor_id)
                stamp = getattr(motor, "last_feedback_time", None)
                rx = getattr(motor, "last_rx_kernel_time", None)
                row += [
                    round((now - stamp) * 1000.0, 3) if stamp else "",
                    _round(getattr(motor, "last_position", None)),
                    _round(getattr(motor, "last_velocity", None)),
                    _round(getattr(motor, "last_torque", None)),
                    _round(getattr(motor, "last_temp", None)),
                    getattr(motor, "last_fault", ""),
                    getattr(motor, "last_mode_status", ""),
                    _round(commands.get(motor_id)) if commands else "",
                    round((wall - rx) * 1000.0, 3) if rx else "",
                ]
            self.rows.append(row)
        except Exception as error:
            if self.error is None:
                self.error = f"{type(error).__name__}: {error}"

    def close(self):
        if not self.rows:
            return
        rows, self.rows = self.rows, []
        try:
            with self.path.open("a", newline="") as handle:
                csv.writer(handle).writerows(rows)
        except Exception as error:
            if self.error is None:
                self.error = f"write failed, {len(rows)} rows lost: {error}"


class PolicyRunner:
    def __init__(self, policy_path, contract):
        self.policy_path = policy_path
        self.contract = contract
        self.pipeline = ActionPipeline(contract)
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[-1] not in (None, "None", contract.observation_size):
            raise RuntimeError(f"Policy input size does not match the manifest: {inputs[0].shape}")
        if len(outputs) != 1 or outputs[0].shape[-1] not in (None, "None", contract.action_size):
            raise RuntimeError(f"Policy output size does not match the manifest: {outputs[0].shape}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
        self.offsets = np.asarray(contract.action_offsets, dtype=np.float32)
        self.last_action = np.zeros(contract.action_size, dtype=np.float32)
        self.velocity_command = np.zeros(3, dtype=np.float32)
        self.history = ObservationHistory.from_contract(contract)
        self.gait_step = 0
        self.inference_ms = 0.0

    def reset(self):
        self.last_action.fill(0.0)

    def joint_state(self, snapshot):
        size = self.contract.action_size
        positions = np.zeros(size, dtype=np.float32)
        velocities = np.zeros(size, dtype=np.float32)
        missing = []
        for index, motor_id in enumerate(self.motor_ids):
            position, velocity = snapshot.get(motor_id, (None, None))
            if position is None or not math.isfinite(position):
                missing.append(motor_id)
                continue
            positions[index] = wrap_to_pi(position)
            velocities[index] = velocity if velocity is not None and math.isfinite(velocity) else 0.0
        return positions, velocities, missing

    def gait_phase_observation(self):
        phase = gait_phase_at(self.gait_step, step_dt=1.0 / self.contract.policy_hz)
        if float(np.linalg.norm(self.velocity_command)) <= GAIT_COMMAND_DEADBAND:
            return np.zeros_like(phase)
        return phase

    def observation(self, positions, velocities, angular_velocity, gravity):
        return assemble_observation(
            positions - self.offsets,
            velocities,
            angular_velocity,
            gravity,
            self.velocity_command,
            self.gait_phase_observation(),
            self.last_action,
            history=self.history,
        )

    def step(self, observation, commit):
        started = time.perf_counter()
        raw_action = self.session.run(
            [self.output_name], {self.input_name: observation.reshape(1, -1)}
        )[0][0]
        self.inference_ms = (time.perf_counter() - started) * 1000.0
        action, targets = self.pipeline.apply(raw_action, max_raw_action=SETTINGS.max_raw_action)
        if commit:
            self.last_action[:] = action
        return np.asarray(raw_action, dtype=np.float32), action, targets

    def commit_policy_action(self, action, commanded_targets):
        """Feed back the policy's own runner-clipped action, as training defines it.

        Training observes `action_manager.action`, the runner-clipped value *before* the
        target fence, and both the MuJoCo harness and this script's read mode already feed
        that back. Reconstructing it from the commanded motor position instead made the live
        path the only one of the three that disagreed: ramp blending, the slew limiter and
        the target fence all change the commanded position, and the fence is many-to-one so
        the original request cannot be recovered from it. The commanded vector is still
        validated here because a non-finite command must stop the run.
        """
        commanded = np.asarray(commanded_targets, dtype=np.float32).reshape(-1)
        if commanded.shape != (self.contract.action_size,):
            raise ValueError(
                f"commanded targets must have {self.contract.action_size} values, got {commanded.shape[0]}"
            )
        if not np.isfinite(commanded).all():
            raise ValueError("commanded targets are not finite")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.contract.action_size,):
            raise ValueError(
                f"policy action must have {self.contract.action_size} values, got {action.shape[0]}"
            )
        if not np.isfinite(action).all():
            raise ValueError("policy action is not finite")
        self.last_action[:] = action
        self.gait_step += 1


class TargetCommander:
    def __init__(self, motors, start_positions, settings):
        self.motors = motors
        self.settings = settings
        self.limiters = {
            motor_id: AxisLimiter(start_positions[motor_id]) for motor_id in motors
        }
        self.commands = dict(start_positions)
        self.velocities = {motor_id: 0.0 for motor_id in motors}
        self.kp_by_motor, self.kd_by_motor = resolve_gains(settings.gain_scale)
        self.slew_limited_count = 0
        self.command_count = 0
        self.slew_limited_by_motor = {motor_id: 0 for motor_id in motors}
        self.command_count_by_motor = {motor_id: 0 for motor_id in motors}
        self.slew_lag_step_max = 0.0
        self.slew_lag_run_max = 0.0
        self.slew_lag_max_by_motor = {motor_id: 0.0 for motor_id in motors}
        self.roll_reprojected_count = 0

    def reset_stats(self):
        self.slew_limited_count = 0
        self.command_count = 0
        self.slew_limited_by_motor = {motor_id: 0 for motor_id in self.motors}
        self.command_count_by_motor = {motor_id: 0 for motor_id in self.motors}
        self.slew_lag_step_max = 0.0
        self.slew_lag_run_max = 0.0
        self.slew_lag_max_by_motor = {motor_id: 0.0 for motor_id in self.motors}
        self.roll_reprojected_count = 0

    def send(self, motor_ids_targets, dt, max_speed, max_accel, use_velocity_target, reproject=None, align=True):
        step_lag_max = 0.0
        previous = {}
        for motor_id, target in motor_ids_targets.items():
            limiter = self.limiters[motor_id]
            previous[motor_id] = limiter.position
            aligned = align_angle(limiter.position, target) if align else target
            position, velocity = limiter.step(aligned, dt, max_speed, max_accel)
            lag = abs(position - aligned)
            if lag > 1.0e-9:
                self.slew_limited_count += 1
                self.slew_limited_by_motor[motor_id] += 1
            if lag > step_lag_max:
                step_lag_max = lag
            if lag > self.slew_lag_max_by_motor[motor_id]:
                self.slew_lag_max_by_motor[motor_id] = lag
            self.command_count += 1
            self.command_count_by_motor[motor_id] += 1
            self.commands[motor_id] = position
            self.velocities[motor_id] = velocity
        self.slew_lag_step_max = step_lag_max
        if step_lag_max > self.slew_lag_run_max:
            self.slew_lag_run_max = step_lag_max
        if reproject is not None:
            projected = reproject(self.commands)
            moved = False
            for motor_id, position in projected.items():
                if motor_id not in previous:
                    continue
                moved |= position != self.commands[motor_id]
                limiter = self.limiters[motor_id]
                limiter.position = position
                limiter.velocity = (position - previous[motor_id]) / dt
                self.commands[motor_id] = position
                self.velocities[motor_id] = limiter.velocity
            self.roll_reprojected_count += int(moved)
        for motor_id, motor in self.motors.items():
            motor.control(
                pos=self.commands[motor_id],
                vel=self.velocities[motor_id] if use_velocity_target else 0.0,
                kp=self.kp_by_motor[motor_id],
                kd=self.kd_by_motor[motor_id],
                torque=0.0,
            )

    def slew_rate_by_motor(self):
        return {
            motor_id: self.slew_limited_by_motor[motor_id]
            / max(1, self.command_count_by_motor[motor_id])
            for motor_id in self.motors
        }

    def worst_slew_lag_motor(self):
        if not self.slew_lag_max_by_motor:
            return None
        return max(self.slew_lag_max_by_motor, key=self.slew_lag_max_by_motor.get)

    def at_rest(self, targets, tolerance):
        return all(
            abs(self.commands[motor_id] - align_angle(self.commands[motor_id], target)) <= 1.0e-12
            and abs(self.velocities[motor_id]) <= 1.0e-12
            for motor_id, target in targets.items()
        ) and all(
            self.motors[motor_id].last_position is not None
            and abs(wrap_to_pi(self.motors[motor_id].last_position) - target) <= tolerance
            for motor_id, target in targets.items()
        )


# Type-0x02 feedback fault flags (Motor.last_fault bit n = 29-bit ID bit 16+n),
# from the RobStride RS02/RS03 manuals, "Communication Type 2: motor feedback data".
FAULT_FLAG_NAMES = (
    "undervoltage",
    "overcurrent",
    "overtemperature",
    "magnetic encoder fault",
    "stall overload",
    "uncalibrated",
)


def fault_flag_names(flags):
    return [name for bit, name in enumerate(FAULT_FLAG_NAMES) if flags & (1 << bit)]


class FaultMonitor:
    """Record reported motor faults, and stop once one of them persists.

    All six bits mean the motor firmware has decided something is wrong, and every one of
    them makes the next position command either useless or harmful -- a magnetic encoder
    fault means the position being fed back is not trustworthy, and `uncalibrated` appearing
    mid-run means the motor rebooted. So none of them is display-only. What they are not is
    instantaneous: a single frame could be a glitch on a bus that has dropped frames before,
    and dropping a 20 kg robot has its own cost. A bit therefore has to hold for
    BLOCKING_FRAMES consecutive feedback frames before it stops anything.
    """

    BLOCKING_FRAMES = 3

    def __init__(self):
        self.previous = {}
        self.history = {}  # (motor_id, bit) -> [first seconds since start, phase, onset count]
        self.streak = {}   # (motor_id, bit) -> consecutive frames the bit has been set

    def update(self, motors, now, phase):
        for motor_id, motor in motors.items():
            flags = motor.last_fault
            rising = flags & ~self.previous.get(motor_id, 0)
            self.previous[motor_id] = flags
            for bit in range(len(FAULT_FLAG_NAMES)):
                if flags & (1 << bit):
                    self.streak[(motor_id, bit)] = self.streak.get((motor_id, bit), 0) + 1
                else:
                    self.streak.pop((motor_id, bit), None)
                if rising & (1 << bit):
                    record = self.history.setdefault(
                        (motor_id, bit), [now - PROGRAM_STARTED, phase, 0]
                    )
                    record[2] += 1

    def blocking_reason(self):
        for (motor_id, bit), frames in sorted(self.streak.items()):
            if frames >= self.BLOCKING_FRAMES:
                return (f"ID {motor_id} reports {FAULT_FLAG_NAMES[bit]} "
                        f"on {frames} consecutive feedback frames")
        return None

    def current_line(self, motors):
        active = [
            f"ID {motor_id} {', '.join(fault_flag_names(motors[motor_id].last_fault))} "
            f"(0x{motors[motor_id].last_fault:02X})"
            for motor_id in sorted(motors)
            if motors[motor_id].last_fault
        ]
        return "motor faults : " + ("   ".join(active) if active else "none")

    def history_line(self):
        by_motor = {}
        for (motor_id, bit), (seconds, phase, count) in sorted(self.history.items()):
            by_motor.setdefault(motor_id, []).append(
                f"{FAULT_FLAG_NAMES[bit]} @ {seconds:.2f} s ({phase}) x{count}"
            )
        entries = [f"ID {motor_id} " + ", ".join(items) for motor_id, items in by_motor.items()]
        return "fault history: " + ("   ".join(entries) if entries else "none")


@dataclass
class LoopStats:
    steps: int = 0
    period_max: float = 0.0
    inference_ms_max: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def record(self, period, inference_ms):
        self.steps += 1
        self.period_max = max(self.period_max, period)
        self.inference_ms_max = max(self.inference_ms_max, inference_ms)

    def elapsed(self):
        return time.monotonic() - self.started


def resolve_contract(policy_path):
    manifest_path = policy_path.with_name("policy_manifest.json")
    contract = PolicyContract.load(manifest_path)
    manifest_policy = contract.verify_policy(manifest_path)
    if manifest_policy.resolve() != policy_path:
        raise ValueError(f"policy_manifest.json selects {manifest_policy.name}, not {policy_path.name}")
    # PolicyRunner always assembles OBSERVATION_TERM_SIZES in order; the size check alone
    # accepts a manifest whose terms are reordered, so compare names and sizes before any CAN I/O.
    layout = ObservationHistory.from_contract(contract)
    if layout.term_sizes != OBSERVATION_TERM_SIZES:
        raise ValueError(
            f"observation_terms {list(contract.observation_terms)} do not match the deploy "
            f"layout {[f'{name}:{size}' for name, size in OBSERVATION_TERM_SIZES]}"
        )
    # gait_phase_at() advances the clock by 1/50 s per policy step.
    if contract.policy_hz != 50.0:
        raise ValueError(f"policy_hz={contract.policy_hz:g} but the gait clock assumes 50 Hz")
    return contract


def installed_common_commit():
    try:
        text = metadata.distribution("robonex-common").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id") if text else None


def verify_common_source(contract):
    # Verify the robonex_common that is actually imported (the .venv install), not a sibling checkout.
    package = Path(robonex_common.__file__).parent
    with tempfile.TemporaryDirectory() as tmp:
        # The manifest hashed the files as src/robonex_common/*.py; pip installs them without src/.
        shutil.copytree(package, Path(tmp, "src", "robonex_common"), ignore=shutil.ignore_patterns("__pycache__"))
        actual_sha256 = python_source_sha256(tmp, ("src/robonex_common",))
    if actual_sha256 != contract.common_sha256:
        raise RuntimeError(
            f"installed robonex-common differs from the policy's training source: "
            f"manifest={contract.common_sha256}, installed={actual_sha256} ({package})"
        )
    print(
        f"Common      : source matches the manifest  "
        f"(installed commit {installed_common_commit() or 'unknown'}, manifest commit {contract.common_commit})"
    )


def format_joint_table(contract, positions, velocities, raw_action, targets, commands):
    lines = [
        f"  {'ID':>3}  {'joint':<20}  {'pos':>9}  {'rel':>9}  {'vel':>10}  "
        f"{'raw':>8}  {'target':>9}  {'command':>9}"
    ]
    lines.append("  " + "-" * 92)
    for index, (name, motor_id) in enumerate(joint_row_order(contract)):
        offset = contract.action_offsets[index]
        command_text = (
            f"{math.degrees(commands[motor_id]):+8.2f}d" if commands is not None else f"{'--':>9}"
        )
        lines.append(
            f"  {motor_id:>3}  {name:<20}  "
            f"{math.degrees(positions[index]):+8.2f}d  "
            f"{math.degrees(positions[index] - offset):+8.2f}d  "
            f"{velocities[index]:+9.3f}  "
            f"{raw_action[index]:+8.3f}  "
            f"{math.degrees(targets[index]):+8.2f}d  "
            f"{command_text}"
        )
    return lines


def format_imu(imu, angular_velocity, gravity, age):
    age_text = "--" if age is None else f"{age * 1000.0:.1f} ms"
    return [
        f"IMU [{imu.status}]  port {DEFAULT_IMU_PORT}  sample age {age_text}",
        f"  raw angular velocity x {angular_velocity[0]:+8.4f}  y {angular_velocity[1]:+8.4f}  "
        f"z {angular_velocity[2]:+8.4f}  [rad/s]",
        f"  projected gravity    x {gravity[0]:+8.4f}  y {gravity[1]:+8.4f}  z {gravity[2]:+8.4f}",
    ]


def format_pipeline(runner):
    pipeline = runner.pipeline
    calls = max(1, pipeline.policy_call_count * pipeline.action_size)
    return (
        f"clip: runner |a|>{pipeline.runner_clip:.1f} {pipeline.runner_clip_count / calls * 100:5.2f}%   "
        f"target {pipeline.target_clip_count / calls * 100:5.2f}%   "
        + (f"roll {pipeline.roll_clip_count / max(1, pipeline.policy_call_count) * 100:5.2f}%   "
           if pipeline.roll_pairs else "")
        + f"inference {runner.inference_ms:5.2f} ms"
    )


def confirm(prompt):
    if not sys.stdin.isatty():
        raise RuntimeError("Confirmation requires an interactive terminal")
    answer = input(prompt)
    if answer.strip():
        raise RuntimeError("Cancelled because the input was not empty")


def run_read(policy_path, contract, args):
    notes = []
    runner = PolicyRunner(policy_path, contract)
    runner.velocity_command[:] = (args.vx, args.vy, args.wz)
    joints = ReadOnlyJointSource(SETTINGS, notes)
    imu = ImuSource(SETTINGS, notes)

    print(f"Policy      : {policy_path}")
    print(f"Task        : {contract.task}   {contract.policy_hz:.0f} Hz")
    print(f"Joint source: {joints.label}")
    print("Read-only: no enable, no run-mode write, and no type 0x01 control frame is sent.")
    confirm("Press Enter to start reading CAN and IMU, or Ctrl-C to cancel: ")

    joints.start()
    imu.start(calibrate=True)
    read_log = None
    if args.telemetry:
        read_log = ReadRecorder(args.telemetry, contract)
        print(f"Recording   : {read_log.path}")
    period = 1.0 / contract.policy_hz
    print_every = max(1, round(contract.policy_hz / SETTINGS.read_print_hz))
    step = 0
    deadline = None if args.duration is None else time.monotonic() + args.duration
    # The preview prints at read_print_hz, not the policy rate, and nothing here commits a
    # command, so the gait clock has no step counter to ride on. Derive it from elapsed time
    # instead: otherwise the preview shows a frozen phase and the policy it displays is not
    # the one the robot would run.
    started_at = time.monotonic()
    timing = TimingProbe() if read_log is not None else None
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            runner.gait_step = int((now - started_at) * contract.policy_hz)
            snapshot = joints.snapshot()
            positions, velocities, missing = runner.joint_state(snapshot)
            angular_velocity, gravity, age = imu.read(now)
            imu_read_ns = time.monotonic_ns()
            observation = runner.observation(positions, velocities, angular_velocity, gravity)
            raw_action, _, targets = runner.step(observation, commit=True)
            if read_log is not None:
                read_log.record(now, observation, gravity, angular_velocity, age, positions, missing,
                                timing=timing.safe_update(now, imu.last_sample, imu_read_ns),
                                velocities=velocities, raw_action=raw_action, targets=targets,
                                sample=imu.last_sample)
            step += 1
            if (step - 1) % print_every:
                sleep = period - (time.monotonic() - now)
                if sleep > 0.0:
                    time.sleep(sleep)
                continue

            lines = [CLEAR_SCREEN]
            lines.append(f"Policy preview (read-only)   {joints.rate_text()}   (Ctrl-C to stop)\n")
            lines.extend(format_joint_table(contract, positions, velocities, raw_action, targets, None))
            if missing:
                lines.append(f"  no response: {sorted(missing)} (reported as 0.0)")
            lines.append("")
            lines.extend(format_imu(imu, angular_velocity, gravity, age))
            lines.append(f"\nObservation vector ({observation.size})")
            lines.append("  " + "  ".join(f"{value:+.3f}" for value in observation))
            lines.append("")
            lines.append(format_pipeline(runner))
            lines.append("No CAN control command is sent in read mode.")
            if notes:
                lines.append("")
                lines.extend(notes)
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            sleep = period - (time.monotonic() - now)
            if sleep > 0.0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        joints.stop()
        imu.stop()
        if read_log is not None:
            read_log.close()
            print(f"Recorded to {read_log.path}")
            if read_log.arrays.error:
                print(f"Warning: arrays not saved ({read_log.arrays.error})")
            if timing.warning():
                print(timing.warning())
        print("Stopped. No motor was enabled or commanded.")
    return 0


def approach_pose(commander, joints, targets, motors, limits, settings, label, faults, recorder=None):
    print(f"\nSpeed-limited move to the {label} pose ({settings.approach_max_speed:.2f} rad/s cap).")
    tolerance = settings.approach_tolerance_deg * DEG
    period = 1.0 / 100.0
    next_tick = time.monotonic()
    last_tick = next_tick
    last_print = 0.0
    settled_at = None
    while True:
        now = time.monotonic()
        joints.poll()
        faults.update(motors, now, "approach")
        fault_reason = faults.blocking_reason()
        if fault_reason:
            raise RuntimeError("Safety stop: " + fault_reason)
        reason = runtime_safety_reason(motors, commander.commands, limits, now, settings)
        if reason:
            raise RuntimeError(f"{label} move stopped for safety: {reason}")

        dt = clamp(now - last_tick, period * 0.25, period * 2.0)
        last_tick = now
        commander.send(
            targets,
            dt,
            settings.approach_max_speed,
            settings.approach_max_accel,
            use_velocity_target=True,
        )
        if recorder is not None:
            recorder.record(label, commander.commands)

        if now - last_print >= 1.0:
            last_print = now
            worst = max(
                (
                    abs(wrap_to_pi(motors[motor_id].last_position or 0.0) - target)
                    for motor_id, target in targets.items()
                ),
                default=0.0,
            )
            print(f"  [{time.strftime('%H:%M:%S')}] worst error {math.degrees(worst):+6.2f} deg")
            if faults.history:
                print("    " + faults.current_line(motors))
                print("    " + faults.history_line())

        if commander.at_rest(targets, tolerance):
            print(f"{label.capitalize()} pose reached (within {settings.approach_tolerance_deg:g} deg).")
            return
        commanded = all(abs(commander.velocities[motor_id]) <= 1.0e-12 for motor_id in targets)
        if commanded:
            if settled_at is None:
                settled_at = now
            elif now - settled_at > settings.approach_settle_timeout:
                outside = [
                    f"ID {motor_id} {math.degrees(wrap_to_pi(motors[motor_id].last_position) - target):+.2f} deg off"
                    for motor_id, target in sorted(targets.items())
                    if motors[motor_id].last_position is not None
                    and abs(wrap_to_pi(motors[motor_id].last_position) - target) > tolerance
                ]
                raise RuntimeError(
                    f"Failed to reach the {label} pose within "
                    f"{settings.approach_settle_timeout:.1f} s: " + ", ".join(outside)
                )
        else:
            settled_at = None

        next_tick += period
        sleep = next_tick - time.monotonic()
        if sleep > 0.0:
            time.sleep(sleep)
        elif time.monotonic() - next_tick > period:
            next_tick = time.monotonic()


class ReadRecorder:
    """Per-frame record of the read-only preview.

    Read mode has no torque, temperature or type 0x02 feedback age — it polls `mechPos` — so
    this is deliberately not the live `TelemetryRecorder`. It captures what the read check
    actually needs to be verified off the screen: the gait-phase pair across the whole run,
    the IMU vectors, and which motors went silent.
    """

    def __init__(self, path, contract):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.handle = self.path.open("x", newline="")
        except FileExistsError:
            raise SystemExit(
                f"{self.path} already exists. Recordings of a real robot are not reproducible, "
                "so this refuses to overwrite one. Pass --telemetry with no path for an "
                "automatic timestamped file, or name a new one."
            )
        self.writer = csv.writer(self.handle)
        header = ["t_s", "gravity_x", "gravity_y", "gravity_z",
                  "gyro_x", "gyro_y", "gyro_z", "imu_age_ms", "missing_ids"]
        header += [f"gait_{i}" for i in range(10)]
        header += [f"{n.replace('_joint','')}.pos" for n in contract.joint_order]
        header += list(TIMING_COLUMNS)
        for name in contract.joint_order:
            header += [f"{name.replace('_joint', '')}.{field_name}" for field_name in READ_JOINT_EXTRA_FIELDS]
        header += list(IMU_EXTRA_COLUMNS)
        self.writer.writerow(header)
        self.handle.flush()
        self.started = None
        self.size = len(contract.joint_order)
        self.arrays = StepArrays(self.path.with_name(self.path.stem + "_arrays.npz"))
        if self.arrays.path.exists():
            self.handle.close()
            raise SystemExit(f"{self.arrays.path} already exists; recordings are never overwritten.")

    def record(self, now, observation, gravity, gyro, age, positions, missing, timing=None,
               velocities=None, raw_action=None, targets=None, sample=None):
        if self.started is None:
            self.started = now
        row = [round(now - self.started, 3)]
        row += [round(float(v), 5) for v in gravity]
        row += [round(float(v), 5) for v in gyro]
        row += ["" if age is None else round(age * 1000.0, 2)]
        row += ["|".join(str(m) for m in sorted(missing)) if missing else ""]
        row += [round(float(v), 5) for v in observation[165:175]]
        row += [round(float(v), 5) for v in positions]
        row += timing_cells(timing)
        for index in range(self.size):
            row += [
                "" if velocities is None else round(float(velocities[index]), 5),
                "" if raw_action is None else round(float(raw_action[index]), 5),
                "" if targets is None else round(float(targets[index]), 5),
            ]
        row += imu_extra_cells(sample)
        self.writer.writerow(row)
        self.handle.flush()
        if raw_action is not None and targets is not None and velocities is not None:
            self.arrays.append(t_s=now - self.started, obs=observation, pos=positions, vel=velocities,
                               raw_action=raw_action, targets=targets, gyro=gyro, gravity=gravity)

    def close(self):
        try:
            self.handle.close()
        except OSError:
            pass
        try:
            self.arrays.save()
        except Exception as error:
            if self.arrays.error is None:
                self.arrays.error = f"save failed: {type(error).__name__}: {error}"


class TelemetryRecorder:
    """Per-policy-step record of everything needed to diagnose a hardware stop.

    The loop has ~20 ms of budget and the previous hardware run already measured a
    20.43 ms worst period, so nothing here touches the filesystem inside the step: rows
    accumulate in memory and are written in batches between steps, and whatever is left
    is flushed by the caller's finally.
    """

    SLACK_S = 0.005
    MAX_ROWS = 500

    def __init__(self, path, contract, motors, hubs=None):
        self.path = Path(path)
        self.rows = []
        self.order = joint_row_order(contract)
        self.motors = motors
        self.hubs = dict(sorted((hubs or {}).items()))
        self.dropped = 0
        self.meta = None
        self.arrays = StepArrays(self.path.with_name(self.path.stem + "_arrays.npz"))
        self.phases_path = self.path.with_name(self.path.stem + "_phases.csv")
        self.meta_path = self.path.with_name(self.path.stem + "_meta.json")
        for reserved in (self.arrays.path, self.phases_path, self.meta_path):
            if reserved.exists():
                raise SystemExit(f"{reserved} already exists; recordings are never overwritten.")
        header = ["t_s", "wall_time", "step", "dt_ms", "inference_ms", "send_ms", "ramp",
                  "cmd_vx", "cmd_vy", "cmd_wz",
                  "gravity_x", "gravity_y", "gravity_z",
                  "gyro_x", "gyro_y", "gyro_z", "imu_age_ms"]
        for name, motor_id in self.order:
            short = name.replace("_joint", "")
            header += [
                f"{short}.age_ms", f"{short}.pos", f"{short}.vel", f"{short}.torque",
                f"{short}.temp", f"{short}.fault", f"{short}.raw_action",
                f"{short}.target", f"{short}.commanded", f"{short}.rx_age_ms",
            ]
        header += list(TIMING_COLUMNS)
        header += list(STEP_EXTRA_COLUMNS)
        header += [f"rx_frames.{channel}" for channel in self.hubs]
        for name, motor_id in self.order:
            short = name.replace("_joint", "")
            header += [f"{short}.{field_name}" for field_name in JOINT_EXTRA_FIELDS]
        header.append("stop_reason")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.handle = self.path.open("x", newline="")
        except FileExistsError:
            raise SystemExit(
                f"{self.path} already exists. Recordings of a real robot are not reproducible, "
                "so this refuses to overwrite one. Pass --telemetry with no path for an "
                "automatic timestamped file, or name a new one."
            )
        try:
            self.writer = csv.writer(self.handle)
            self.writer.writerow(header)
            self.handle.flush()
        except BaseException:
            self.handle.close()
            raise
        self.step = 0

    def record(self, now, started, dt, inference_ms, ramp, raw_action, targets, commands,
               gravity=None, gyro=None, imu_age=None, velocity_command=None, stop_reason="",
               send_ms=None, timing=None, extra=None, policy_action=None, slew_velocities=None,
               observation=None, positions=None, velocities=None, allow_flush=True):
        wall = time.time()
        row = [
            round(now - started, 4), round(wall, 6), self.step, round(dt * 1000.0, 3),
            round(inference_ms, 3), _round(send_ms, 3), round(ramp, 4),
        ]
        row += [_round(v) for v in (velocity_command if velocity_command is not None else (None,) * 3)]
        row += [_round(v) for v in (gravity if gravity is not None else (None,) * 3)]
        row += [_round(v) for v in (gyro if gyro is not None else (None,) * 3)]
        row.append(round(imu_age * 1000.0, 3) if imu_age is not None else "")
        # Age is measured against a fresh clock read, not the `now` the caller captured at the
        # top of its loop: that timestamp predates `joints.poll()`, so the feedback it is being
        # compared against is stamped later and every age came out negative.
        sampled_at = time.monotonic()
        for index, (_, motor_id) in enumerate(self.order):
            motor = self.motors.get(motor_id)
            stamp = getattr(motor, "last_feedback_time", None)
            age = (sampled_at - stamp) * 1000.0 if stamp else float("nan")
            row += [
                round(age, 3),
                _round(getattr(motor, "last_position", None)),
                _round(getattr(motor, "last_velocity", None)),
                _round(getattr(motor, "last_torque", None)),
                _round(getattr(motor, "last_temp", None)),
                getattr(motor, "last_fault", ""),
                round(float(raw_action[index]), 5),
                round(float(targets[index]), 5),
                _round(commands.get(motor_id)),
            ]
            rx = getattr(motor, "last_rx_kernel_time", None)
            row.append(round((wall - rx) * 1000.0, 3) if rx else "")
        row += timing_cells(timing)
        row += list(extra) if extra is not None else [""] * len(STEP_EXTRA_COLUMNS)
        row += [getattr(hub, "last_pump_frames", "") for hub in self.hubs.values()]
        for index, (_, motor_id) in enumerate(self.order):
            motor = self.motors.get(motor_id)
            row += [
                getattr(motor, "last_mode_status", ""),
                "" if policy_action is None else round(float(policy_action[index]), 5),
                "" if slew_velocities is None else _round(slew_velocities.get(motor_id)),
            ]
        row.append(stop_reason)
        self.rows.append(row)
        if observation is not None:
            self.arrays.append(
                t_s=now - started, wall_time=wall, step=self.step, obs=observation,
                raw_action=raw_action, policy_action=policy_action, targets=targets,
                commanded=[commands.get(motor_id, float("nan")) for _, motor_id in self.order],
                pos=positions, vel=velocities,
                torque=[_float_attr(self.motors.get(motor_id), "last_torque") for _, motor_id in self.order],
                gyro=gyro, gravity=gravity, velocity_command=velocity_command,
            )
        self.step += 1
        if allow_flush and len(self.rows) >= self.MAX_ROWS:
            self.flush()

    def write_sidecar(self, policy_path, contract, settings, args, extra=None):
        """Record which policy produced this file, next to it.

        The CSV carries no policy identity, so a recording cannot be traced back to the
        weights that made it -- and a run analysed under the wrong assumption is worse than
        one that is skipped. This is the same idea as the exporter's `export_receipt.json`.
        """
        meta = {
            "telemetry_file": self.path.name,
            "written_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "policy_file": str(policy_path),
            "policy_sha256": contract.policy_sha256,
            "task": contract.task,
            "robot_model": contract.robot_model,
            "policy_hz": contract.policy_hz,
            "training_commit": contract.training_commit,
            "common_commit": contract.common_commit,
            "description_commit": contract.description_commit,
            "deploy_source_sha256": python_source_sha256(
                Path(__file__).resolve().parents[2], ("scripts",)
            ),
            "command": {"vx": args.vx, "vy": args.vy, "wz": args.wz,
                        "keyboard": bool(getattr(args, "keyboard", False))},
            "heading_hold": (
                {"kp": args.heading_kp, "ki": args.heading_ki, "source": args.heading_source,
                 "integral_limit": HeadingHold.INTEGRAL_LIMIT, "output_limit": HeadingHold.OUTPUT_LIMIT}
                if getattr(args, "heading_hold", False) else None
            ),
            "settings": {
                "gain_scale": settings.gain_scale,
                "max_error_deg": settings.max_error_deg,
                "max_tilt_deg": settings.max_tilt_deg,
                "max_temp": settings.max_temp,
                "overspeed": settings.overspeed,
                "approach_tolerance_deg": settings.approach_tolerance_deg,
                "policy_max_speed": settings.policy_max_speed,
                "policy_max_accel": settings.policy_max_accel,
            },
        }
        meta["files"] = {
            "telemetry": self.path.name,
            "arrays": self.arrays.path.name,
            "phases": self.phases_path.name,
        }
        if extra:
            meta.update(extra)
        text = json.dumps(meta, indent=2, default=str) + "\n"
        try:
            with self.meta_path.open("x") as handle:
                handle.write(text)
        except FileExistsError:
            raise SystemExit(f"{self.meta_path} already exists; recordings are never overwritten.")
        self.meta = meta
        return self.meta_path

    def finalize_sidecar(self, end):
        if self.meta is None:
            return None
        self.meta["end"] = end
        text = json.dumps(self.meta, indent=2, default=str) + "\n"
        partial = self.meta_path.with_name(self.meta_path.name + ".partial")
        partial.write_text(text)
        os.replace(partial, self.meta_path)
        return self.meta_path

    def record_stop(self, now, started, reason, commands, gravity=None, gyro=None, imu_age=None,
                    velocity_command=None):
        """Write the sample that tripped a safety stop.

        The checks run at the top of the loop and raise, while `record` is called near the
        bottom, so the sample that actually breached a limit was never written -- the one
        row an operator most wants to see. This captures the motor state as the check saw
        it, against the commands it compared them to. The policy columns are blank because
        this step never reached inference.
        """
        self.record(
            now, started, float("nan"), float("nan"), float("nan"),
            [float("nan")] * len(self.order), [float("nan")] * len(self.order), commands,
            gravity=gravity, gyro=gyro, imu_age=imu_age, velocity_command=velocity_command,
            stop_reason=reason, allow_flush=False,
        )

    def flush_if_idle(self, slack_s):
        """Write only while the loop is waiting for its next tick.

        A batched write of 100 rows measured 1.7 ms, and the previous hardware run's worst
        period was already 20.43 ms against a 20 ms budget, so the write must not land
        inside the step.
        """
        if self.rows and slack_s > self.SLACK_S:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        try:
            self.writer.writerows(self.rows)
            self.handle.flush()
        except OSError:
            self.dropped += len(self.rows)
        finally:
            self.rows = []

    def close(self):
        try:
            self.flush()
        finally:
            try:
                self.handle.close()
            except OSError:
                pass
            try:
                self.arrays.save()
            except Exception as error:
                if self.arrays.error is None:
                    self.arrays.error = f"save failed: {type(error).__name__}: {error}"


def _float_attr(obj, name):
    value = getattr(obj, name, None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _round(value, digits=5):
    if value is None:
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


class HeadingHold:
    INTEGRAL_LIMIT = 0.1
    OUTPUT_LIMIT = 0.2
    MAX_DT = 0.1

    def __init__(self, kp, ki, source):
        self.kp = float(kp)
        self.ki = float(ki)
        self.source = source
        self.heading_gyro = 0.0
        self.heading_quat = None
        self.target = None
        self.error = None
        self.integral = 0.0
        self.wz = 0.0
        self.engaged = False
        self.last_now = None

    @staticmethod
    def quat_yaw(sample):
        if sample is None:
            return None
        try:
            q = sample.orientation
            w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (w, x, y, z)) or w * w + x * x + y * y + z * z < 1e-6:
            return None
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def world_yaw_rate(gyro, gravity):
        norm = math.sqrt(sum(float(g) * float(g) for g in gravity))
        if not math.isfinite(norm) or norm < 1e-6:
            return float(gyro[2])
        return -sum(float(w) * float(g) for w, g in zip(gyro, gravity)) / norm

    def release(self):
        self.engaged = False
        self.target = None
        self.error = None
        self.integral = 0.0
        self.wz = 0.0

    def update(self, now, gyro, gravity, sample, command, active):
        dt = 0.0 if self.last_now is None else min(max(now - self.last_now, 0.0), self.MAX_DT)
        self.last_now = now
        rate = self.world_yaw_rate(gyro, gravity)
        if math.isfinite(rate):
            self.heading_gyro = wrap_to_pi(self.heading_gyro + rate * dt)
        self.heading_quat = self.quat_yaw(sample)
        heading = self.heading_quat if self.source == "quat" else self.heading_gyro
        planar = math.hypot(float(command[0]), float(command[1]))
        if (not active or heading is None or not math.isfinite(heading)
                or float(command[2]) != 0.0 or planar <= GAIT_COMMAND_DEADBAND):
            self.release()
            return 0.0
        if not self.engaged:
            self.engaged = True
            self.target = heading
            self.integral = 0.0
            dt = 0.0
        self.error = wrap_to_pi(self.target - heading)
        integral = clamp(self.integral + self.ki * self.error * dt, -self.INTEGRAL_LIMIT, self.INTEGRAL_LIMIT)
        demand = self.kp * self.error + integral
        if abs(demand) <= self.OUTPUT_LIMIT or demand * self.error < 0.0:
            self.integral = integral
        self.wz = clamp(self.kp * self.error + self.integral, -self.OUTPUT_LIMIT, self.OUTPUT_LIMIT)
        return self.wz

    def cells(self):
        def deg(value):
            return "" if value is None else _round(math.degrees(value), 4)
        return [deg(self.heading_gyro), deg(self.heading_quat), deg(self.target), deg(self.error),
                _round(self.integral, 5), _round(self.wz, 5), int(self.engaged)]

    def status_line(self):
        if not self.engaged:
            return f"heading hold ({self.source}): released"
        return (f"heading hold ({self.source}): error {math.degrees(self.error):+6.2f} deg   "
                f"wz {self.wz:+.3f} rad/s   integral {self.integral:+.3f}")


def policy_loop(runner, commander, joints, imu, motors, limits, contract, args, notes, stats,
                faults, thermal, keyboard=None, telemetry=None):
    period = 1.0 / contract.policy_hz
    stats.started = time.monotonic()
    next_tick = time.monotonic()
    last_tick = next_tick
    last_status = 0.0
    ramp_started = next_tick
    home = {motor_id: wrap_to_pi(commander.commands[motor_id]) for motor_id in motors}
    deadline = None if args.duration is None else next_tick + args.duration
    commander.reset_stats()
    reproject = roll_reprojector(runner.pipeline, runner.motor_ids)
    owns_telemetry = telemetry is None and bool(getattr(args, "telemetry", None))
    if owns_telemetry:
        telemetry = TelemetryRecorder(args.telemetry, contract, motors, hubs=getattr(joints, "hubs", None))
    timing = TimingProbe() if telemetry is not None else None
    prev_work_ms = None
    heading = None
    if getattr(args, "heading_hold", False):
        heading = HeadingHold(args.heading_kp, args.heading_ki, args.heading_source)
    if owns_telemetry:
        print(f"Telemetry   : {telemetry.path}  ({contract.policy_hz:.0f} Hz per-motor record)")
        print(f"Provenance  : {telemetry.write_sidecar(runner.policy_path, contract, SETTINGS, args).name}")

    print(
        f"\nPolicy control is active at {contract.policy_hz:.0f} Hz "
        f"(ramp-in {SETTINGS.ramp_seconds:.1f} s). Ctrl-C stops and brakes."
    )
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            late_ms = (now - next_tick) * 1000.0
            joints.poll()
            poll_ms = (time.monotonic() - now) * 1000.0 if telemetry is not None else None
            faults.update(motors, now, "policy")
            thermal.update(motors, now - last_tick)
            def stop(reason, gravity=None, gyro=None, imu_age=None):
                if telemetry is not None:
                    telemetry.record_stop(
                        now, stats.started, reason, commander.commands,
                        gravity=gravity, gyro=gyro, imu_age=imu_age,
                        velocity_command=tuple(float(v) for v in runner.velocity_command),
                    )
                return RuntimeError("Safety stop: " + reason)

            fault_reason = faults.blocking_reason()
            if fault_reason:
                raise stop(fault_reason)
            reason = runtime_safety_reason(motors, commander.commands, limits, now, SETTINGS)
            if reason:
                raise stop(reason)

            positions, velocities, missing = runner.joint_state(joints.snapshot())
            if missing:
                raise stop(f"no feedback for motor IDs {sorted(missing)}")
            angular_velocity, gravity, age = imu.read(now)
            imu_read_ns = time.monotonic_ns()
            imu_reason = imu.failure_reason(age)
            if imu_reason:
                raise stop(imu_reason)
            tilt = tilt_reason(gravity, SETTINGS.max_tilt_deg)
            if tilt:
                raise stop(tilt, gravity=gravity, gyro=angular_velocity, imu_age=age)

            requested_command = runner.velocity_command.copy()
            holding = now - ramp_started < SETTINGS.ramp_seconds + SETTINGS.command_hold_seconds
            if heading is not None:
                heading_wz = heading.update(now, angular_velocity, gravity, imu.last_sample,
                                            requested_command, active=not holding)
            if holding:
                runner.velocity_command[:] = 0.0
            elif heading is not None and heading.engaged:
                runner.velocity_command[2] = heading_wz
            try:
                observation = runner.observation(positions, velocities, angular_velocity, gravity)
                raw_action, policy_action, targets = runner.step(observation, commit=False)
            except ValueError as error:
                raise RuntimeError(f"Safety stop: {error}") from error
            finally:
                runner.velocity_command[:] = requested_command

            blend = clamp((now - ramp_started) / SETTINGS.ramp_seconds, 0.0, 1.0)
            commanded = {}
            for index, (_, motor_id) in enumerate(joint_row_order(contract)):
                target = float(targets[index])
                commanded[motor_id] = home[motor_id] + blend * (target - home[motor_id])

            dt = clamp(now - last_tick, period * 0.25, period * 2.0)
            last_tick = now
            send_started = time.monotonic()
            commander.send(
                commanded,
                dt,
                SETTINGS.policy_max_speed,
                SETTINGS.policy_max_accel,
                use_velocity_target=False,
                reproject=reproject,
                align=False,
            )
            send_ms = (time.monotonic() - send_started) * 1000.0
            commanded_targets = [
                commander.commands[motor_id] for _, motor_id in joint_row_order(contract)
            ]
            runner.commit_policy_action(policy_action, commanded_targets)
            stats.record(dt, runner.inference_ms)
            if telemetry is not None:
                telemetry.record(
                    now, stats.started, dt, runner.inference_ms, blend,
                    raw_action, targets, commander.commands,
                    gravity=gravity, gyro=angular_velocity, imu_age=age,
                    velocity_command=tuple(float(v) for v in runner.velocity_command),
                    send_ms=send_ms,
                    timing=timing.safe_update(now, imu.last_sample, imu_read_ns),
                    extra=step_extra_cells(late_ms, poll_ms, prev_work_ms, commander, runner.pipeline,
                                           thermal, imu.last_sample, heading),
                    policy_action=policy_action, slew_velocities=commander.velocities,
                    observation=observation, positions=positions, velocities=velocities,
                )

            if now - last_status >= 1.0 / SETTINGS.status_hz:
                last_status = now
                lines = [CLEAR_SCREEN]
                lines.append(
                    f"Policy control   {joints.rate_text()}   ramp {blend * 100:5.1f}%   "
                    f"elapsed {stats.elapsed():6.2f} s   (Ctrl-C to stop)\n"
                )
                lines.extend(
                    format_joint_table(
                        contract, positions, velocities, raw_action, targets, commander.commands
                    )
                )
                lines.append("")
                lines.extend(format_imu(imu, angular_velocity, gravity, age))
                lines.append("")
                lines.append(format_pipeline(runner))
                worst_lag_id = commander.worst_slew_lag_motor()
                lines.append(
                    f"slew lag now {math.degrees(commander.slew_lag_step_max):6.2f} deg   "
                    f"run max {math.degrees(commander.slew_lag_run_max):6.2f} deg"
                    + (f" (ID {worst_lag_id})" if worst_lag_id is not None else "")
                    + f"   worst period {stats.period_max * 1000.0:6.2f} ms   "
                    f"worst inference {stats.inference_ms_max:5.2f} ms"
                )
                lags = commander.slew_lag_max_by_motor
                lines.append(
                    "slew lag max by motor "
                    + "  ".join(f"ID {motor_id} {math.degrees(lags[motor_id]):5.2f}" for motor_id in sorted(lags))
                    + " deg"
                )
                lines.append("")
                if keyboard is not None:
                    lines.append(
                        f"command  vx {runner.velocity_command[0]:+.2f}  "
                        f"vy {runner.velocity_command[1]:+.2f}  "
                        f"wz {runner.velocity_command[2]:+.2f}    {keyboard.legend()}"
                    )
                if heading is not None:
                    lines.append(heading.status_line())
                lines.append(thermal.worst_line())
                lines.append(faults.current_line(motors))
                lines.append(faults.history_line())
                if notes:
                    lines.append("")
                    lines.extend(notes)
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

            if keyboard is not None:
                note = keyboard.poll(runner.velocity_command)
                if note:
                    print(f"  [command] {note}   "
                          f"vx {runner.velocity_command[0]:+.2f} "
                          f"vy {runner.velocity_command[1]:+.2f} "
                          f"wz {runner.velocity_command[2]:+.2f}", flush=True)

            next_tick += period
            sleep = next_tick - time.monotonic()
            if telemetry is not None:
                prev_work_ms = (time.monotonic() - now) * 1000.0
                telemetry.flush_if_idle(sleep)
                sleep = next_tick - time.monotonic()
            if sleep > 0.0:
                time.sleep(sleep)
            elif time.monotonic() - next_tick > period:
                next_tick = time.monotonic()

    finally:
        if owns_telemetry:
            telemetry.close()
        if timing is not None and timing.warning():
            print(timing.warning())


def run_deploy(policy_path, contract, args):
    notes = []
    verify_common_source(contract)
    require_robot_model(contract.robot_model)
    runner = PolicyRunner(policy_path, contract)
    runner.velocity_command[:] = (args.vx, args.vy, args.wz)
    imu = ImuSource(SETTINGS, notes)

    motor_ids = [JOINT_BY_MODEL_NAME[name].motor_id for name in contract.joint_order]
    model_limits = robot_model(contract.robot_model).joint_limits_by_id()
    hard_limits = {motor_id: model_limits[motor_id] for motor_id in motor_ids}
    home_targets = {
        JOINT_BY_MODEL_NAME[name].motor_id: float(contract.action_offsets[index])
        for index, name in enumerate(contract.joint_order)
    }

    buses = {}
    motors = {}
    hubs = {}
    enabled_ids = []
    stop_ids = []
    stats = LoopStats()
    commander = None
    faults = FaultMonitor()
    keyboard = None
    if args.keyboard:
        keyboard = KeyboardCommand(COMMAND_LIMITS)
    thermal = ThermalLoad({
        motor_id: RATED_TORQUE[JOINT_BY_ID[motor_id].motor_model] for motor_id in motor_ids
    })
    telemetry = None
    phases = None
    run_meta = {}
    end_meta = {}
    exit_text = "completed"
    try:
        buses, motors, hubs = open_hardware(motor_ids, SETTINGS.interface, SETTINGS.host_id)
        idle = stop_idle_motors(buses, motor_ids, SETTINGS.host_id)
        if idle:
            print(f"Stop sent to the non-policy motors on the open buses: {idle}")
        measured, blocking = inspect_zero_positions(motors, SETTINGS.approach_tolerance_deg * DEG, hard_limits)
        profile = robot_model(contract.robot_model)
        roll_pairs = roll_pairs_for(profile, motor_ids)
        if roll_pairs:
            blocking += start_roll_blocks(measured, roll_pairs, profile.foot_roll, math.radians(1.0))
        if blocking:
            raise RuntimeError(
                "Preflight safety check failed; motors will not be enabled:\n  " + "\n  ".join(blocking)
            )
        preflight_parameters = read_preflight_parameters(motors)
        for line in preflight_parameter_lines(preflight_parameters):
            print(line)
        run_meta["preflight_parameters"] = preflight_parameters
        run_meta["preflight_mech_pos_rad"] = {
            motor_id: _finite_or_none(value) for motor_id, value in sorted(measured.items())
        }
        if not imu.start(calibrate=True):
            raise RuntimeError("IMU is not usable; motors will not be enabled:\n  " + "\n  ".join(notes))
        if getattr(args, "telemetry", None):
            missing_fields = missing_timing_fields(imu.driver.latest())
            if missing_fields:
                raise RuntimeError(
                    f"IMU sample lacks the timing fields {missing_fields} that --telemetry records; "
                    "motors will not be enabled. Rebuild the n100 binding or run without --telemetry"
                )
            telemetry = TelemetryRecorder(args.telemetry, contract, motors, hubs=hubs)
            phases = PhaseRecorder(telemetry.phases_path, contract, motors, imu=imu)
            bias = imu.bias_raw
            run_meta.update({
                "host": host_snapshot(sorted(hubs)),
                "imu_gyro_bias_raw": None if bias is None else [bias.x, bias.y, bias.z],
                "imu_driver_stats_start": imu_driver_stats(imu),
                "gains": {motor_id: [ENABLE_KP[motor_id], ENABLE_KD[motor_id]] for motor_id in motor_ids},
                "settings_all": asdict(SETTINGS),
            })
            print(f"Telemetry   : {telemetry.path}  ({contract.policy_hz:.0f} Hz per-motor record)")
            print(f"Phases      : {telemetry.phases_path.name}  (enable, approach and brake samples)")
            print(f"Provenance  : {telemetry.write_sidecar(policy_path, contract, SETTINGS, args, run_meta).name}")

        print("\nThe real motors will move under policy control.")
        print(f"  policy      : {policy_path}")
        print(f"  task        : {contract.task}")
        print(f"  rate        : {contract.policy_hz:.0f} Hz")
        print(f"  gain scale  : {SETTINGS.gain_scale:g}  (1.0 = the gains the policy was trained at)")
        for name, spec in JOINT_BY_MODEL_NAME.items():
            print(
                f"      {name:22s} kp {ENABLE_KP[spec.motor_id]:6.1f}  kd {ENABLE_KD[spec.motor_id]:5.2f}"
            )
        print(f"  motor IDs   : {sorted(motor_ids)}")
        print(f"  duration    : {'until Ctrl-C' if args.duration is None else f'{args.duration:.1f} s'}")
        print(f"  tilt stop   : {SETTINGS.max_tilt_deg:g} deg from vertical")
        print("  The robot must hang on the stand or be held; this tool cannot catch a fall.")
        print("  Keep the emergency stop within reach.")
        confirm("Press Enter to enable the motors and start, or Ctrl-C to cancel: ")
        if keyboard is not None and keyboard.start():
            print(f"  {keyboard.legend()}")

        stop_ids = list(motor_ids)
        starts, enabled_ids = enable_with_runtime_feedback(
            motors, hubs, ENABLE_KP, ENABLE_KD, hard_limits, enabled_out=enabled_ids
        )

        end_meta["enable_start_positions_rad"] = {motor_id: _finite_or_none(v) for motor_id, v in starts.items()}
        joints = RuntimeJointSource(motors, hubs)
        commander = TargetCommander(motors, starts, SETTINGS)
        if phases is not None:
            phases.record("enabled", commander.commands)
        approach_started = time.monotonic()
        try:
            approach_pose(commander, joints, home_targets, motors, hard_limits, SETTINGS, "default", faults,
                          recorder=phases)
        finally:
            end_meta["approach"] = {
                "seconds": time.monotonic() - approach_started,
                "final_error_deg": {
                    motor_id: (None if motors[motor_id].last_position is None else
                               math.degrees(wrap_to_pi(motors[motor_id].last_position) - target))
                    for motor_id, target in sorted(home_targets.items())
                },
            }
        runner.reset()
        policy_loop(
            runner, commander, joints, imu, motors, hard_limits, contract, args, notes, stats,
            faults, thermal, keyboard, telemetry=telemetry
        )
    except KeyboardInterrupt:
        exit_text = "stop requested (Ctrl-C)"
        print("\nStop requested.")
    except BaseException as error:
        exit_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        brake_kwargs = {}
        if phases is not None:
            def observe_brake():
                for hub in hubs.values():
                    hub.pump()
                phases.record("brake")
            brake_kwargs["observe"] = observe_brake
        shutdown_report = brake_and_stop(
            motors, buses, enabled_ids, stop_ids, SETTINGS.brake_time, ENABLE_KD, **brake_kwargs
        )
        if telemetry is not None:
            end_meta["imu_driver_stats_end"] = imu_driver_stats(imu)
        imu.stop()
        if stop_ids:
            for line in shutdown_report_lines(shutdown_report):
                print(line)
            print(faults.history_line())
        else:
            print("CAN buses closed. No motor control command was sent.")
        if commander is not None and stats.steps:
            pipeline = runner.pipeline
            calls = max(1, pipeline.policy_call_count * pipeline.action_size)
            print(
                f"Ran {stats.steps} policy steps in {stats.elapsed():.2f} s; "
                f"worst period {stats.period_max * 1000.0:.2f} ms, "
                f"worst inference {stats.inference_ms_max:.2f} ms, "
                f"slew lag max {math.degrees(commander.slew_lag_run_max):.2f} deg, "
                f"{thermal.worst_line()}, "
                f"runner clip {pipeline.runner_clip_count / calls * 100:.2f}%, "
                f"target clip {pipeline.target_clip_count / calls * 100:.2f}%."
            )
        if keyboard is not None:
            keyboard.stop()
        if telemetry is not None:
            previous = None
            if threading.current_thread() is threading.main_thread():
                previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                close_recordings(telemetry, phases, end_meta, exit_text, shutdown_report, stats, commander,
                                 runner, thermal, faults, sorted(hubs))
            finally:
                if previous is not None:
                    signal.signal(signal.SIGINT, previous)
    return 0


def close_recordings(telemetry, phases, end_meta, exit_text, shutdown_report, stats, commander, runner,
                     thermal, faults, channels):
    errors = {}
    for name, action in (("telemetry", telemetry.close), ("phases", phases.close if phases else None)):
        if action is None:
            continue
        try:
            action()
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    try:
        summarize_run(telemetry, phases, end_meta, exit_text, shutdown_report, stats, commander, runner,
                      thermal, faults, channels, errors)
    except Exception as error:
        end_meta["summary_error"] = f"{type(error).__name__}: {error}"
    try:
        path = telemetry.finalize_sidecar(end_meta)
        print(f"Recorded    : {telemetry.path.name}, {telemetry.arrays.path.name}, "
              f"{telemetry.phases_path.name}, {path.name if path else '-'}")
    except Exception as error:
        print(f"Warning: run summary not written ({type(error).__name__}: {error})")
    problems = dict(end_meta.get("recording_errors", {}))
    if "summary_error" in end_meta:
        problems["summary"] = end_meta["summary_error"]
    for name, message in problems.items():
        print(f"Warning: {name} recording problem: {message}")


def summarize_run(telemetry, phases, end_meta, exit_text, shutdown_report, stats, commander, runner,
                  thermal, faults, channels, errors):
    pipeline = runner.pipeline
    end_meta.update({
        "written_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "exit": exit_text,
        "policy_steps": stats.steps,
        "policy_seconds": stats.elapsed() if stats.steps else 0.0,
        "worst_period_ms": stats.period_max * 1000.0,
        "worst_inference_ms": stats.inference_ms_max,
        "rows_recorded": telemetry.step,
        "rows_dropped": telemetry.dropped,
        "shutdown": {
            key: ({str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else value)
            for key, value in (shutdown_report if isinstance(shutdown_report, dict) else {}).items()
        },
        "fault_history": faults.history_line(),
        "thermal": {"line": thermal.worst_line(), "peak": dict(thermal.peak), "load_end": dict(thermal.load)},
        "pipeline": {
            "policy_calls": getattr(pipeline, "policy_call_count", None),
            "runner_clip_count": getattr(pipeline, "runner_clip_count", None),
            "target_clip_count": getattr(pipeline, "target_clip_count", None),
            "roll_clip_count": getattr(pipeline, "roll_clip_count", None),
        },
        "can_after": can_statistics(channels),
        "recording_errors": {
            **errors,
            **({"arrays": telemetry.arrays.error} if telemetry.arrays.error else {}),
            **({"phases": phases.error} if phases is not None and phases.error else {}),
        },
    })
    if commander is not None:
        end_meta["slew"] = {
            "lag_run_max_deg": math.degrees(commander.slew_lag_run_max),
            "limited_fraction_by_motor": commander.slew_rate_by_motor(),
            "roll_reprojected_count": commander.roll_reprojected_count,
        }


class KeyboardCommand:
    """Drive the velocity command from the terminal while the policy runs.

    cbreak, not raw: `tty.setraw` clears ISIG, which would stop Ctrl-C from raising
    SIGINT and take away the operator's primary way to stop a moving robot.
    `tty.setcbreak` clears only ECHO and ICANON, so keys arrive one at a time and
    Ctrl-C still works.

    Losing the terminal means losing control, so stdin reaching EOF -- an SSH session
    dropping, most likely -- zeroes the command and disables further input rather than
    leaving the robot walking on the last thing it was told.
    """

    KEYS = {
        "w": (0, +0.05), "s": (0, -0.05),
        "q": (1, +0.05), "e": (1, -0.05),
        "a": (2, +0.05), "d": (2, -0.05),
    }

    def __init__(self, limits):
        self.limits = limits
        self.escape_state = 0
        self.fd = None
        self.saved = None
        self.active = False
        self.lost = False

    def start(self):
        if not sys.stdin.isatty():
            print("Keyboard control needs an interactive terminal; the command stays fixed.")
            return False
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        atexit.register(self.stop)
        tty.setcbreak(self.fd)
        self.active = True
        termios.tcflush(self.fd, termios.TCIFLUSH)
        return True

    def stop(self):
        if self.active and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.active = False

    def poll(self, command):
        """Apply whatever has been typed. Returns a note to display, or None."""
        if not self.active or self.lost:
            return None
        note = None
        while select.select([sys.stdin], [], [], 0.0)[0]:
            data = os.read(self.fd, 64)
            if not data:
                self.lost = True
                command[:] = (0.0, 0.0, 0.0)
                return "stdin closed; command zeroed and keyboard disabled"
            for byte in data:
                if self.escape_state == 2:
                    if 0x40 <= byte <= 0x7E:
                        self.escape_state = 0
                    continue
                if self.escape_state == 1:
                    self.escape_state = 0
                    if byte in (0x5B, 0x4F):
                        self.escape_state = 2
                        continue
                if byte == 0x1B:
                    self.escape_state = 1
                    continue
                key = chr(byte).lower()
                if key == " ":
                    command[:] = (0.0, 0.0, 0.0)
                    note = "command zeroed"
                elif key in self.KEYS:
                    axis, delta = self.KEYS[key]
                    low, high = self.limits[axis]
                    command[axis] = max(low, min(high, float(command[axis]) + delta))
        return note

    def legend(self):
        if not self.active:
            return "keyboard: off"
        return "keys: w/s forward  q/e strafe  a/d turn  SPACE zero  Ctrl-C stop"


# The policy is only trained inside this envelope; a command outside it is out of
# distribution and the response is undefined, so refuse it rather than clamp silently.
# These are the ranges the velocity curriculum actually reached, not its starting
# ranges: it widens both ways from (0.1, 0.1) toward limit_ranges, and S30, S33 and S34
# all hit the full -0.2..0.5 on x by iteration ~160, spending 84% of training there.
COMMAND_LIMITS = ((-0.2, 0.5), (-0.2, 0.2), (-0.2, 0.2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a trained RoboNex policy on the real motors, or preview its values read-only."
    )
    parser.add_argument("--policy", type=Path, required=True, help="ONNX policy path")
    parser.add_argument(
        "--read",
        action="store_true",
        help="Read-only preview: print observation, action and targets without commanding any motor",
    )
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Steer the velocity command from the terminal while the policy runs: w/s "
             "forward, q/e strafe, a/d turn, SPACE to zero every axis. Values are clamped "
             "to the same trained envelope as --vx/--vy/--wz. Needs an interactive "
             "terminal; Ctrl-C keeps working.",
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=None,
        help="Stop if the trunk goes further than this from vertical. The recorded runs "
             "reach 6.9 deg walking and 26.1 deg standing under a push that the robot then "
             "recovered from, so the 40 deg default leaves recovery alone while still firing "
             "long before the robot is flat. Lower it to catch a fall sooner, at the risk of "
             "cutting a recovery short.",
    )
    parser.add_argument(
        "--approach-tolerance-deg",
        type=float,
        default=None,
        help="How close to the default pose counts as reached before policy control starts. "
             "The 1.0 deg default suits a suspended robot; standing on the ground the legs "
             "carry body weight and PD control leaves a steady-state error of roughly "
             "(holding torque / kp), which is 3-5 deg at the ankles. Raise it deliberately, "
             "and only when you know why the robot cannot close the gap",
    )
    parser.add_argument(
        "--telemetry",
        nargs="?",
        const="",
        metavar="PATH",
        help="Write a per-policy-step CSV: per-motor feedback age, position, velocity, torque, "
             "temperature, fault and mode, raw and fed-back action, clipped target, commanded "
             "position and slew velocity, IMU vectors, loop timing and log-only IMU timing "
             "columns for scripts/analysis/timing_report.py; next to it a full-precision "
             "_arrays.npz (observation and action vectors), a _phases.csv with enable, approach "
             "and brake samples, and a _meta.json with the saved motor parameters and the run "
             "summary, all written after the brake. Omit PATH for an automatic timestamped file "
             "under results/policy_to_real. An existing file is never overwritten",
    )
    parser.add_argument(
        "--vx", type=float, default=0.0, help="Forward velocity command (m/s)"
    )
    parser.add_argument(
        "--vy", type=float, default=0.0, help="Lateral velocity command (m/s)"
    )
    parser.add_argument(
        "--wz", type=float, default=0.0, help="Yaw rate command (rad/s)"
    )
    parser.add_argument(
        "--heading-hold",
        action="store_true",
        help="Hold the heading captured when walking starts by adding a yaw-rate command the "
             "policy sees (PI, output within +/-0.2 rad/s). Engages only while the operator's wz "
             "is 0 and vx/vy is above the gait deadband; turning or zeroing releases it and the "
             "next straight command captures a new heading. Off by default",
    )
    parser.add_argument(
        "--heading-kp", type=float, default=1.0, metavar="K",
        help="Heading-hold proportional gain, (rad/s) per rad of heading error (0-2)",
    )
    parser.add_argument(
        "--heading-ki", type=float, default=0.1, metavar="K",
        help="Heading-hold integral gain, (rad/s) per rad*s; the integral is limited to +/-0.1 rad/s (0-0.5)",
    )
    parser.add_argument(
        "--heading-source", choices=("gyro", "quat"), default="gyro",
        help="Heading the loop holds: the bias-calibrated raw gyro integrated about the world vertical "
             "(gyro), or the N100 AHRS quaternion yaw (quat). Both are recorded either way",
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=Settings.gain_scale,
        metavar="F",
        help=(
            "Fraction of the per-joint gains to command (default 1.0 = the gains the "
            "policy was trained at). Ramp up from a low value on the stand."
        ),
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="",
        metavar="PATH",
        help="Save terminal output; omit PATH for an automatic file under results/policy_to_real",
    )
    args = parser.parse_args(argv)
    args.policy = args.policy.expanduser().resolve()
    if not args.policy.is_file():
        parser.error(f"Policy not found: {args.policy}")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0.0):
        parser.error("--duration must be finite and positive")
    if not math.isfinite(args.gain_scale) or not 0.0 < args.gain_scale <= 1.0:
        parser.error("--gain-scale must be in (0, 1]")
    if not math.isfinite(args.heading_kp) or not 0.0 <= args.heading_kp <= 2.0:
        parser.error("--heading-kp must be in [0, 2]; 4 and above jittered at the step rhythm in simulation")
    if not math.isfinite(args.heading_ki) or not 0.0 <= args.heading_ki <= 0.5:
        parser.error("--heading-ki must be in [0, 0.5]")
    if args.heading_hold and args.read:
        parser.error("--heading-hold has no effect in --read mode")
    for name, value, limit in (
        ("--vx", args.vx, COMMAND_LIMITS[0]),
        ("--vy", args.vy, COMMAND_LIMITS[1]),
        ("--wz", args.wz, COMMAND_LIMITS[2]),
    ):
        if not math.isfinite(value):
            parser.error(f"{name} must be finite")
        if not limit[0] <= value <= limit[1]:
            parser.error(
                f"{name}={value} is outside the trained envelope {limit}; the policy has never "
                "seen that command and its response is undefined"
            )
    if args.telemetry == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        kind = "read" if args.read else "live"
        args.telemetry = (
            THIS_FILE.parents[2] / "results" / "policy_to_real" / f"{stamp}_{kind}_telemetry.csv"
        )
    elif args.telemetry is not None:
        args.telemetry = Path(args.telemetry).expanduser().resolve()
    if args.log == "":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.log = THIS_FILE.parents[2] / "results" / "policy_to_real" / f"{timestamp}_{os.getpid()}.log"
    elif args.log is not None:
        args.log = Path(args.log).expanduser().resolve()
    return args


def run(args):
    try:
        contract = resolve_contract(args.policy)
    except (FileNotFoundError, ValueError) as error:
        print(f"Policy error: {error}")
        return 1
    try:
        if args.read:
            return run_read(args.policy, contract, args)
        return run_deploy(args.policy, contract, args)
    except KeyboardInterrupt:
        print("\nStop requested.")
        return 0
    except (RuntimeError, ValueError, OSError, can.CanError) as error:
        print(f"\nStopped: {error}")
        return 1


def main(argv=None):
    args = parse_args(argv)
    global SETTINGS, ENABLE_KP, ENABLE_KD
    SETTINGS = replace(SETTINGS, gain_scale=args.gain_scale)
    if args.max_tilt_deg is not None:
        if not 5.0 <= args.max_tilt_deg <= 90.0:
            raise SystemExit("--max-tilt-deg must be between 5 and 90 degrees")
        SETTINGS = replace(SETTINGS, max_tilt_deg=args.max_tilt_deg)
        print(
            f"Tilt stop set to {args.max_tilt_deg:g} deg "
            f"(default {Settings().max_tilt_deg:g})"
        )
    if args.approach_tolerance_deg is not None:
        if not 0.0 < args.approach_tolerance_deg <= 10.0:
            raise SystemExit("--approach-tolerance-deg must be between 0 and 10 degrees")
        SETTINGS = replace(SETTINGS, approach_tolerance_deg=args.approach_tolerance_deg)
        print(
            f"Approach tolerance widened to {args.approach_tolerance_deg:g} deg "
            f"(default {Settings().approach_tolerance_deg:g})"
        )
    ENABLE_KP, ENABLE_KD = resolve_gains(SETTINGS.gain_scale)
    if args.log is None:
        return run(args)
    try:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("x", encoding="utf-8", buffering=1) as log:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            lock = threading.Lock()
            sys.stdout = TeeStream(original_stdout, log, lock)
            sys.stderr = TeeStream(original_stderr, log, lock)
            try:
                print(f"Log file    : {args.log}")
                return run(args)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    except OSError as error:
        print(f"Log error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
