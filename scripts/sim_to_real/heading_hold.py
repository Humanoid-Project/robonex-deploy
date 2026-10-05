import math

from robonex_common.protocol import clamp

from safety import wrap_to_pi

GAIT_COMMAND_DEADBAND = 0.05


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
        self.policy_wz = None

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
                _round(self.integral, 5), _round(self.wz, 5), int(self.engaged), _round(self.policy_wz, 5)]

    def status_line(self):
        if not self.engaged:
            return f"heading hold ({self.source}): released"
        return (f"heading hold ({self.source}): error {math.degrees(self.error):+6.2f} deg   "
                f"wz {self.wz:+.3f} rad/s   integral {self.integral:+.3f}")
