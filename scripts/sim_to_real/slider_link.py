from __future__ import annotations

import json
import math
import secrets
import socket
import threading
import time

PROTOCOL = 1
PORT = 47021
MAX_PACKET = 8192
TARGET_HOLD_AFTER = 0.5
ARM_TOLERANCE_RAD = math.radians(0.5)


class TargetBox:
    def __init__(self):
        self._lock = threading.Lock()
        self._targets = None
        self._stamp = None
        self._session = None
        self._closed = False
        self._sync = None

    def put(self, targets, stamp, session):
        with self._lock:
            self._targets = dict(targets)
            self._stamp = stamp
            self._session = session

    def close(self):
        with self._lock:
            self._closed = True

    def latest(self):
        with self._lock:
            return self._targets, self._stamp, self._session, self._closed

    def set_sync(self, values):
        with self._lock:
            self._sync = None if values is None else dict(values)

    def sync(self):
        with self._lock:
            return None if self._sync is None else dict(self._sync)


def encode_targets(variant, session, seq, targets):
    return json.dumps({
        "protocol": PROTOCOL, "kind": "targets", "variant": variant,
        "session": session, "seq": seq,
        "targets": {name: float(value) for name, value in targets.items()},
    }, separators=(",", ":")).encode()


def decode_targets(payload, variant):
    try:
        message = json.loads(payload.decode())
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"not a JSON packet ({error})") from None
    if not isinstance(message, dict) or message.get("protocol") != PROTOCOL or message.get("kind") != "targets":
        raise ValueError(f"not a slider target packet of protocol {PROTOCOL}")
    if message.get("variant") != variant:
        raise ValueError(f"sender model {message.get('variant')!r} does not match this robot ({variant})")
    session, seq, targets = message.get("session"), message.get("seq"), message.get("targets")
    if not isinstance(session, str) or not isinstance(seq, int) or not isinstance(targets, dict):
        raise ValueError("packet is missing session, seq or targets")
    for name, value in targets.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"target {name} is not a finite number")
    return session, seq, {name: float(value) for name, value in targets.items()}


def encode_status(state, note, rows, sync=None):
    return json.dumps({
        "protocol": PROTOCOL, "kind": "status", "state": state, "note": note,
        "rows": rows, "sync": sync,
    }, separators=(",", ":")).encode()


def decode_status(payload):
    message = json.loads(payload.decode())
    if not isinstance(message, dict) or message.get("protocol") != PROTOCOL or message.get("kind") != "status":
        raise ValueError(f"not a status packet of protocol {PROTOCOL}")
    return message


def resolve_host(host):
    try:
        return socket.gethostbyname(host)
    except OSError as error:
        raise ValueError(f"cannot resolve {host}: {error}") from None


class TargetReceiver:
    def __init__(self, peer, variant, box, port=PORT, clock=time.monotonic):
        self.peer = resolve_host(peer)
        self.variant = variant
        self.box = box
        self.clock = clock
        self.session = None
        self.seq = None
        self.reply_to = None
        self.rejected = 0
        self.last_reject = None
        self._idle = None
        self._idle_sent = 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="slider-receiver", daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.sock.close()

    def accept(self, payload, address):
        if address[0] != self.peer:
            self.rejected += 1
            self.last_reject = f"packet from {address[0]} ignored (only {self.peer} is accepted)"
            return
        try:
            session, seq, targets = decode_targets(payload, self.variant)
        except ValueError as error:
            self.rejected += 1
            self.last_reject = str(error)
            return
        if session != self.session:
            self.session = session
            self.seq = None
        if self.seq is not None and seq <= self.seq:
            return
        self.seq = seq
        self.reply_to = address
        self.box.put(targets, self.clock(), session)

    def set_idle(self, state, note=""):
        self._idle = None if state is None else (state, note)
        self._idle_sent = 0.0

    def _send_idle(self):
        idle = self._idle
        now = time.monotonic()
        if idle is not None and now - self._idle_sent >= 0.25:
            self._idle_sent = now
            self.send_status(idle[0], idle[1], [])

    def send_status(self, state, note, rows, sync=None):
        if self.reply_to is None:
            return
        try:
            self.sock.sendto(encode_status(state, note, rows, sync), self.reply_to)
        except OSError:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                payload, address = self.sock.recvfrom(MAX_PACKET)
            except socket.timeout:
                self._send_idle()
                continue
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(0.5)
                continue
            self.accept(payload, address)
            self._send_idle()


class TargetSender:
    def __init__(self, host, variant, port=PORT):
        self.address = (resolve_host(host), port)
        self.variant = variant
        self.session = secrets.token_hex(8)
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.send_errors = 0

    def send(self, targets):
        self.seq += 1
        try:
            self.sock.sendto(encode_targets(self.variant, self.session, self.seq, targets), self.address)
        except OSError:
            self.send_errors += 1

    def poll_status(self):
        latest = None
        while True:
            try:
                payload, address = self.sock.recvfrom(MAX_PACKET)
            except OSError:
                return latest
            if address[0] != self.address[0]:
                continue
            try:
                latest = decode_status(payload)
            except (UnicodeDecodeError, ValueError):
                continue

    def close(self):
        self.sock.close()


def viewer_loop(model, data, actuator_names, on_targets, rate, should_stop=lambda: False, sync=lambda: None):
    import mujoco
    import mujoco.viewer
    import numpy as np

    period = 1.0 / rate
    sim_steps = max(1, int(round(period / model.opt.timestep)))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_tick = time.monotonic()
        while viewer.is_running() and not should_stop():
            values = sync()
            with viewer.lock():
                if values:
                    for name, value in values.items():
                        if name in actuator_names:
                            data.ctrl[actuator_names[name]] = value
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    ctrl = data.ctrl.copy()
                    mujoco.mj_resetData(model, data)
                    data.ctrl[:] = ctrl
                    mujoco.mj_forward(model, data)
                targets = {name: float(data.ctrl[aid]) for name, aid in actuator_names.items()}
                mujoco.mj_step(model, data, nstep=sim_steps)
            on_targets(targets)
            viewer.sync()
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0.0:
                time.sleep(sleep)
            elif time.monotonic() - next_tick > period:
                next_tick = time.monotonic()
