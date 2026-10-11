import math
from collections import deque


def _round(value, digits=5):
    if value is None:
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


class GravityBC:
    WINDOW = 40
    TAU_S = 1.0
    DT_S = 0.02
    BOUND_DEG = 8.0

    def __init__(self):
        self.alpha = self.DT_S / (self.TAU_S + self.DT_S)
        self.bound = math.radians(self.BOUND_DEG)
        self.reset()

    @classmethod
    def config(cls):
        return {
            "window_ticks": cls.WINDOW,
            "tau_s": cls.TAU_S,
            "dt_s": cls.DT_S,
            "bound_deg": cls.BOUND_DEG,
            "warmup": "delta 0 until the window holds WINDOW samples",
            "gravity_columns": "gravity_x/y/z and arrays 'gravity' are the raw AHRS projected gravity; "
                               "the policy saw gravity_bc_x/y/z (arrays 'gravity_bc')",
        }

    def reset(self):
        self.gravity_window = deque()
        self.force_window = deque()
        self.gravity_sum = [0.0, 0.0, 0.0]
        self.force_sum = [0.0, 0.0, 0.0]
        self.delta = (0.0, 0.0, 0.0)
        self.raw = None
        self.corrected = None

    @property
    def ready(self):
        return len(self.gravity_window) == self.WINDOW

    @staticmethod
    def sample_force(sample):
        if sample is None or not getattr(sample, "has_imu_frame", True):
            return None
        try:
            acc = sample.linear_acceleration
            return (float(acc.x), float(acc.y), float(acc.z))
        except (AttributeError, TypeError, ValueError):
            return None

    def _push(self, gravity, force):
        self.gravity_window.append(gravity)
        self.force_window.append(force)
        for i in range(3):
            self.gravity_sum[i] += gravity[i]
            self.force_sum[i] += force[i]
        if len(self.gravity_window) > self.WINDOW:
            old_gravity = self.gravity_window.popleft()
            old_force = self.force_window.popleft()
            for i in range(3):
                self.gravity_sum[i] -= old_gravity[i]
                self.force_sum[i] -= old_force[i]

    def update(self, gravity, force):
        g = tuple(float(v) for v in gravity)
        self.raw = g
        if force is not None and all(math.isfinite(v) for v in g + tuple(force)):
            self._push(g, tuple(float(v) for v in force))
            if self.ready:
                gm = [s / self.WINDOW for s in self.gravity_sum]
                fm = [s / self.WINDOW for s in self.force_sum]
                gn, fn = _norm(gm), _norm(fm)
                if gn > 0.0 and fn > 0.0:
                    error = _cross([v / gn for v in gm], [-v / fn for v in fm])
                    self.delta = tuple(
                        min(max(d + self.alpha * (e - d), -self.bound), self.bound)
                        for d, e in zip(self.delta, error)
                    )
        self.corrected = self.apply(g)
        return self.corrected

    def apply(self, gravity):
        turn = _cross(self.delta, gravity)
        v = tuple(g + t for g, t in zip(gravity, turn))
        n = _norm(v)
        if not math.isfinite(n) or n < 1e-9:
            return tuple(gravity)
        return tuple(x / n for x in v)

    def delta_deg(self):
        return (math.degrees(self.delta[0]), math.degrees(self.delta[1]))

    def cells(self):
        if self.corrected is None:
            return [""] * 6
        roll, pitch = self.delta_deg()
        return [_round(v) for v in self.corrected] + [_round(roll, 4), _round(pitch, 4), int(self.ready)]

    def status_line(self):
        if self.corrected is None:
            return "gravity BC: no sample"
        roll, pitch = self.delta_deg()
        return (f"  BC gravity (policy) x {self.corrected[0]:+8.4f}  y {self.corrected[1]:+8.4f}  "
                f"z {self.corrected[2]:+8.4f}   delta roll {roll:+5.2f}  pitch {pitch:+5.2f} deg"
                + ("" if self.ready else "   (filling window)"))
