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
from robonex_common.imu import DEFAULT_IMU_PORT, MOUNT_ROLL_DEG
from robonex_common.joints import CHANNEL_MOTOR_IDS, JOINT_BY_ID, JOINT_BY_MODEL_NAME
from robonex_common.policy import PolicyContract
from robonex_common.protocol import DEFAULT_INTERFACE, MECHANICAL_POSITION_INDEX, MECHANICAL_VELOCITY_INDEX

DEG = math.pi / 180.0


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


def main():
    parser = argparse.ArgumentParser(
        description="Print the live policy observation. Read-only.")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="policy_manifest.json that defines observation order")
    args = parser.parse_args()
    try:
        contract = PolicyContract.load(args.manifest)
    except (FileNotFoundError, ValueError) as error:
        print(f"Policy manifest error: {error}")
        return 1

    if not sys.stdin.isatty():
        print("Start confirmation requires an interactive terminal")
        return 1
    try:
        answer = input("Press Enter to start reading CAN and IMU, or Ctrl-C to cancel: ")
    except KeyboardInterrupt:
        print("\nStop requested.")
        return 0
    if answer.strip():
        print("Cancelled because the input was not empty")
        return 1

    notes = []
    state = {mid: (None, None) for mid in JOINT_BY_ID}
    lock = threading.Lock()

    readers = [CanReader(channel, motor_ids, DEFAULT_INTERFACE, CAN_TIMEOUT,
                         state, lock, notes)
              for channel, motor_ids in CHANNEL_MOTOR_IDS.items()]
    for reader in readers:
        reader.start()

    driver = n100.ImuDriver(n100.DriverConfig(
        port=DEFAULT_IMU_PORT,
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
                     f"(permissions: sudo chmod 666 {DEFAULT_IMU_PORT})")

    time.sleep(0.3)

    try:
        while True:
            with lock:
                snapshot = dict(state)
            sample = driver.latest()
            if imu_status == "Ready" and not driver.is_running:
                imu_status = f"Reader stopped: {driver.last_error() or 'Unknown error'}"

            lines = ["\033[2J\033[3J\033[H"]
            can_hz = "  ".join(f"{r.channel} {r.rate_hz:5.1f} Hz" for r in readers)
            lines.append(f"Policy observation   {can_hz}   (Ctrl-C to stop)\n")

            lines.append(f"Joint position and velocity ({contract.action_size}, manifest order)")
            lines.append(f"  {'ID':>3}  {'joint':<18}  {'pos [rad]':>10}  {'vel [rad/s]':>12}")
            lines.append("  " + "-" * 50)
            joint_pos, joint_vel = [], []
            for joint in contract.joint_order:
                motor_id = JOINT_BY_MODEL_NAME[joint].motor_id
                pos, vel = snapshot.get(motor_id, (None, None))
                joint_pos.append(pos if pos is not None else 0.0)
                joint_vel.append(vel if vel is not None else 0.0)
                p = f"{pos:+10.4f}" if pos is not None else f"{'--':>10}"
                v = f"{vel:+12.4f}" if vel is not None else f"{'--':>12}"
                lines.append(f"  {motor_id:>3}  {joint:<18}  {p}  {v}")

            lines.append(f"\nIMU [{imu_status}]  port {DEFAULT_IMU_PORT}")
            if sample is None:
                lines.append("  No sample yet")
                ang_vel, gravity = n100.Vec3(), n100.Vec3(0.0, 0.0, -1.0)
            else:
                ang_vel, gravity = sample.angular_velocity, sample.projected_gravity
                raw = sample.angular_velocity_raw
                lines.append(f"  angular velocity x {ang_vel.x:+8.4f}  y {ang_vel.y:+8.4f}  "
                             f"z {ang_vel.z:+8.4f}  [rad/s, AHRS fused]")
                lines.append(f"  raw angular velocity x {raw.x:+8.4f}  y {raw.y:+8.4f}  z {raw.z:+8.4f}  "
                             f"[rad/s]")
                lines.append(f"  gravity x {gravity.x:+8.4f}  y {gravity.y:+8.4f}  "
                             f"z {gravity.z:+8.4f}")

            lines.append("\nPrevious action")
            previous_action = [0.0] * contract.action_size
            lines.append("  " + "  ".join(f"{v:+.3f}" for v in previous_action))

            obs = [*joint_pos, *joint_vel, ang_vel.x, ang_vel.y, ang_vel.z,
                  gravity.x, gravity.y, gravity.z, *previous_action]
            lines.append(f"\nObservation vector ({len(obs)} = {len(joint_pos)} pos + "
                         f"{len(joint_vel)} vel + 3 ang_vel + 3 gravity + "
                         f"{len(previous_action)} prev_action)")
            lines.append("  [" + ", ".join(f"{v:+.3f}" for v in obs) + "]")

            if notes:
                lines.append("")
                lines.extend(notes)

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / PRINT_HZ)
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
