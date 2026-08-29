#!/usr/bin/env python3

import argparse
import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import can
import n100
from robonex_common.can import Motor
from robonex_common.joints import CHANNEL_MOTOR_IDS, JOINT_BY_ID, JOINT_BY_MODEL_NAME
from robonex_common.policy import PolicyContract
from robonex_common.protocol import DEFAULT_INTERFACE, MECHANICAL_POSITION_INDEX, MECHANICAL_VELOCITY_INDEX

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime is required: pip install onnxruntime")

DEG = math.pi / 180.0

MOUNT_ROLL_DEG = 180.0

PRINT_HZ = 10.0
CAN_TIMEOUT = 0.02

class CanReader(threading.Thread):

    def __init__(self, channel, motor_ids, interface, timeout, state, lock, notes):
        super().__init__(daemon=True)
        self.channel = channel
        self.motor_ids = motor_ids
        self.interface = interface
        self.timeout = timeout
        self.state = state
        self.lock = lock
        self.notes = notes
        self.rate_hz = 0.0
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            bus = can.Bus(channel=self.channel, interface=self.interface)
        except OSError as error:
            self.notes.append(f"[{self.channel}] Open failed: {error}  "
                              f"(sudo ip link set {self.channel} up type can bitrate 1000000)")
            return
        t0, cycles = time.monotonic(), 0
        motors = {
            motor_id: Motor(bus, motor_id, JOINT_BY_ID[motor_id].motor_model)
            for motor_id in self.motor_ids
        }
        try:
            while not self._stop_event.is_set():
                for motor_id in self.motor_ids:
                    pos = motors[motor_id].read_parameter(MECHANICAL_POSITION_INDEX, timeout=self.timeout)
                    vel = motors[motor_id].read_parameter(MECHANICAL_VELOCITY_INDEX, timeout=self.timeout)
                    with self.lock:
                        self.state[motor_id] = (pos, vel)
                cycles += 1
                now = time.monotonic()
                if now - t0 >= 0.5:
                    self.rate_hz = cycles / (now - t0)
                    t0, cycles = now, 0
        except can.CanError as error:
            self.notes.append(f"[{self.channel}] CAN error: {error}")
        finally:
            bus.shutdown()


def build_observation(snapshot, ang_vel, gravity, prev_action, joint_order):

    pos = np.zeros(12, dtype=np.float32)
    vel = np.zeros(12, dtype=np.float32)
    for i, joint in enumerate(joint_order):
        motor_id = JOINT_BY_MODEL_NAME[joint].motor_id
        p, v = snapshot.get(motor_id, (None, None))
        pos[i] = p if p is not None else 0.0
        vel[i] = v if v is not None else 0.0

    obs = np.concatenate([
        pos, vel,
        [ang_vel.x, ang_vel.y, ang_vel.z],
        [gravity.x, gravity.y, gravity.z],
        prev_action,
    ]).astype(np.float32)
    return obs.reshape(1, -1)


def main():
    parser = argparse.ArgumentParser(
        description="Build a live observation, run the policy, and print targets. Read-only.")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="policy_manifest.json created with policy.onnx")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0", help="IMU serial port")
    parser.add_argument("--channels", nargs="+", default=list(CHANNEL_MOTOR_IDS),
                        choices=list(CHANNEL_MOTOR_IDS), help="CAN channels to use")
    parser.add_argument("--interface", default=DEFAULT_INTERFACE, help="python-can interface")
    parser.add_argument("--timeout", type=float, default=CAN_TIMEOUT,
                        help="Timeout for one motor parameter request in seconds")
    parser.add_argument("--rate", type=float, default=PRINT_HZ, help="Display update rate in Hz")
    args = parser.parse_args()

    if args.rate <= 0:
        print("--rate must be positive.")
        return 1
    try:
        contract = PolicyContract.load(args.manifest)
        policy = contract.verify_policy(args.manifest)
    except (FileNotFoundError, ValueError) as error:
        print(f"Policy manifest error: {error}")
        return 1

    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    if input_shape[-1] not in (None, "None", contract.observation_size):
        print(f"Policy input size does not match the manifest: {input_shape}")
        return 1
    if output_shape[-1] not in (None, "None", contract.action_size):
        print(f"Policy output size does not match the manifest: {output_shape}")
        return 1
    print(f"Policy: {policy}")
    print(f"  Input {input_shape}  Output {output_shape}")
    print("  Contract: 12 active motor joints; passive closed-loop joints are never targeted.\n")

    notes = []
    state = {mid: (None, None) for mid in JOINT_BY_ID}
    lock = threading.Lock()

    readers = [CanReader(channel, CHANNEL_MOTOR_IDS[channel], args.interface, args.timeout,
                         state, lock, notes)
              for channel in args.channels]
    for reader in readers:
        reader.start()

    driver = n100.ImuDriver(n100.DriverConfig(
        port=args.imu_port,
        mount_rotation=n100.Quat.from_axis_angle_x(MOUNT_ROLL_DEG * DEG),
    ))
    imu_status = "Starting..."
    try:
        driver.start()
        if driver.wait_for_sample(timeout=3.0) is None:
            imu_status = "No response in 3 seconds"
            notes.append(f"[IMU] {driver.last_error() or 'Unknown error'}")
        else:
            imu_status = "Ready"
    except RuntimeError as error:
        imu_status = "Start failed"
        notes.append(f"[IMU] {error}")
        notes.append(f"      ls /dev/ttyUSB* /dev/ttyACM*  "
                     f"(permissions: sudo chmod 666 {args.imu_port})")

    time.sleep(0.3)

    prev_action = np.zeros(12, dtype=np.float32)

    print("Press Ctrl-C to stop.\n")
    try:
        while True:
            with lock:
                snapshot = dict(state)
            sample = driver.latest()
            if imu_status == "Ready" and not driver.is_running:
                imu_status = f"Reader stopped: {driver.last_error() or 'Unknown error'}"

            lines = ["\033[2J\033[3J\033[H"]
            can_hz = "  ".join(f"{r.channel} {r.rate_hz:5.1f} Hz" for r in readers)
            lines.append(f"Policy target preview (read-only)   {can_hz}   (Ctrl-C to stop)\n")

            lines.append("Active closed-loop motor joints")
            lines.append(f"  {'joint':<22}  {'ID':>3}  {'pos [rad]':>10}  {'vel [rad/s]':>12}")
            for joint in contract.joint_order:
                motor_id = JOINT_BY_MODEL_NAME[joint].motor_id
                p, v = snapshot.get(motor_id, (None, None))
                pv = f"{p:+10.4f}" if p is not None else f"{'--':>10}"
                vv = f"{v:+12.4f}" if v is not None else f"{'--':>12}"
                lines.append(f"  {joint:<14}  {motor_id:>3}  {pv}  {vv}")

            if sample is None:
                lines.append("\nNo IMU sample yet; using zero angular velocity and gravity (0,0,-1)")
                ang_vel, gravity = n100.Vec3(), n100.Vec3(0.0, 0.0, -1.0)
            else:
                ang_vel, gravity = sample.angular_velocity_raw, sample.projected_gravity
                lines.append(f"\nIMU [{imu_status}]  raw angular velocity x {ang_vel.x:+8.4f}  "
                             f"y {ang_vel.y:+8.4f}  z {ang_vel.z:+8.4f}  [rad/s]")
                lines.append(f"        gravity x {gravity.x:+8.4f}  y {gravity.y:+8.4f}  "
                             f"z {gravity.z:+8.4f}")

            obs = build_observation(snapshot, ang_vel, gravity, prev_action, contract.joint_order)
            t0 = time.perf_counter()
            raw_action = session.run(None, {input_name: obs})[0][0]
            action = np.clip(raw_action, -contract.runner_action_clip, contract.runner_action_clip)
            targets = np.clip(
                action * np.asarray(contract.action_scales) + np.asarray(contract.action_offsets),
                np.asarray(contract.target_clips)[:, 0],
                np.asarray(contract.target_clips)[:, 1],
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0

            lines.append(f"\nRaw policy action ({infer_ms:.2f} ms) and clipped target")
            lines.append(f"  {'joint':<22}  {'raw':>8}  {'target [rad]':>12}  {'[deg]':>8}")
            for joint, a, target in zip(contract.joint_order, raw_action, targets):
                lines.append(f"  {joint:<14}  {a:+8.4f}  {target:+12.4f}  "
                             f"{target / DEG:+7.2f}")

            lines.append("\nRead-only: this tool never sends policy actions over CAN.")

            if notes:
                lines.append("")
                lines.extend(notes)

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            prev_action = action.astype(np.float32)
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        for reader in readers:
            reader.stop()
        for reader in readers:
            reader.join(timeout=2.0)
        driver.stop()
        print("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
