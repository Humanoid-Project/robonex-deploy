import json
import math
import struct
import threading

import pytest

mujoco = pytest.importorskip("mujoco")
can = pytest.importorskip("can")

import bench
import mujoco_to_real
import real_to_mujoco
import safety
from robonex_common.actuators import CONTROL_GAINS_BY_JOINT
from robonex_common.buses import bus_map
from robonex_common.joints import ALL_MOTORS, MOTOR_BY_ID, MOTOR_LIMITS_BY_ID, VARIANT_MOTOR_IDS
from robonex_common.motors import MOTOR_SPECS
from robonex_common.protocol import (
    COMM_PARAMETER_READ,
    HOST_ID,
    MECHANICAL_POSITION_INDEX,
    build_arbitration_id,
    parse_arbitration_id,
)

VARIANTS = ("edu", "pro", "max")
NOT_IN_SIM = {"edu": [13], "pro": [13], "max": []}
fmt = bench.format_ids


@pytest.fixture(autouse=True)
def default_bus_map(tmp_path, monkeypatch):
    path = tmp_path / "bus_map.json"
    path.write_text("{}")
    monkeypatch.setenv("ROBONEX_BUS_MAP", str(path))
    bus_map.cache_clear()
    yield path
    bus_map.cache_clear()


def _identity(tmp_path, monkeypatch, content):
    path = tmp_path / "robot_model"
    if content is not None:
        path.write_text(content + "\n")
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(path))
    return path


def _limit_text(mid):
    lower, upper = MOTOR_LIMITS_BY_ID[mid]
    return f"{math.degrees(lower):+.1f}..{math.degrees(upper):+.1f}"


def _expected_placeholders(motor_ids):
    found = []
    for mid in motor_ids:
        if mid in bench.LEG_MOTOR_IDS:
            continue
        lower, upper = MOTOR_LIMITS_BY_ID[mid]
        if any(math.isclose(lower, -math.radians(d), abs_tol=1e-5) and math.isclose(upper, math.radians(d), abs_tol=1e-5)
               for d in (30.0, 45.0)):
            found.append(mid)
    return found


def _rows(out, after):
    rows = {}
    for line in out.split(after, 1)[1].splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and line.startswith("  "):
            rows[int(parts[0])] = line
    return rows


def _dry_run(tmp_path, monkeypatch, capsys, variant, selection):
    _identity(tmp_path, monkeypatch, f"ver2_{variant}")
    argv = ["--dry-run"] + (["--motor-id", *selection] if selection else [])
    code = mujoco_to_real.main(argv)
    return code, capsys.readouterr()


@pytest.mark.parametrize("variant", VARIANTS)
def test_mjcf_actuators_match_expected_not_in_sim(variant):
    model = mujoco.MjModel.from_xml_path(str(safety.fixed_model_path(variant)))
    missing = [
        joint.motor_id for joint in ALL_MOTORS
        if joint.motor_id in VARIANT_MOTOR_IDS[f"ver2_{variant}"]
        and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint.model_name.removesuffix("_joint")) < 0
    ]
    assert missing == NOT_IN_SIM[variant]


@pytest.mark.parametrize("variant", VARIANTS)
def test_dry_run_all_banner(variant, tmp_path, monkeypatch, capsys):
    code, captured = _dry_run(tmp_path, monkeypatch, capsys, variant, ["all"])
    out = captured.out
    assert code == 0, out
    assert "KeyError" not in out + captured.err and "Traceback" not in captured.err
    ids = list(VARIANT_MOTOR_IDS[f"ver2_{variant}"])
    assert f"Robot       : ver2_{variant} (from {tmp_path / 'robot_model'})" in out
    assert f"Selected    : {fmt(ids)} ({len(ids)} motors)" in out
    assert "Leg limits  : ver2_edu profile" in out
    assert f"Would enable ID {fmt(ids)} after the preflight zero check" in out
    missing = NOT_IN_SIM[variant]
    if missing:
        assert f"Not in sim  : {len(missing)} selected — {fmt(missing)} (moved to zero and held there)" in out
        assert f"has no actuator for {len(missing)} ver2_{variant} motor(s): {fmt(missing)}" in out
    else:
        assert "Not in sim" not in out and "WARNING" not in out
    placeholders = _expected_placeholders(ids)
    if placeholders:
        assert f"PLACEHOLDER : limits of ID {fmt(placeholders)} equal the common placeholder values" in out
    else:
        assert "PLACEHOLDER" not in out
    rows = _rows(out, "Dry run:")
    assert sorted(rows) == ids
    for mid in ids:
        line = rows[mid]
        assert MOTOR_BY_ID[mid].hardware_name in line
        assert MOTOR_BY_ID[mid].channel in line
        if mid not in bench.LEG_MOTOR_IDS:
            assert _limit_text(mid) in line
        assert ("not in sim" in line) == (mid in missing)
        assert ("PLACEHOLDER" in line) == (mid in placeholders)


def test_dry_run_default_legs_edu(tmp_path, monkeypatch, capsys):
    code, captured = _dry_run(tmp_path, monkeypatch, capsys, "edu", None)
    out = captured.out
    assert code == 0
    assert "Selected    : 1–12 (12 motors)" in out
    assert "Would enable ID 1–12 after" in out
    assert "Not in sim" not in out
    assert sorted(_rows(out, "Dry run:")) == list(range(1, 13))


def test_dry_run_head_name_and_hex_max(tmp_path, monkeypatch, capsys):
    code, captured = _dry_run(tmp_path, monkeypatch, capsys, "max", ["head", "left_knee_pitch", "0x1"])
    out = captured.out
    assert code == 0
    assert "Selected    : 1, 4, 13 (3 motors)" in out
    assert "Would enable ID 1, 4, 13 after" in out
    rows = _rows(out, "Dry run:")
    assert sorted(rows) == [1, 4, 13]
    assert "neck_pitch" in rows[13] and "rs05" in rows[13] and _limit_text(13) in rows[13]
    assert rows[13].split()[-1] in ("ok", "PLACEHOLDER")
    assert "not in sim" not in out


def test_dry_run_head_only_max_uses_only_profile(tmp_path, monkeypatch, capsys):
    code, captured = _dry_run(tmp_path, monkeypatch, capsys, "max", ["head"])
    assert code == 0
    assert "Leg limits  : ver2_edu profile" in captured.out
    assert "Would enable ID 13 after" in captured.out


@pytest.mark.parametrize("variant, selection, message", [
    ("edu", ["15"], "motor ID 15 is not on ver2_edu; ver2_edu has 1–13"),
    ("edu", ["arms"], "ver2_edu has no arms motors (its motors: 1–13)"),
    ("max", ["19"], "motor ID 19 is not on ver2_max"),
    ("max", ["14"], "motor ID 14 is not on ver2_max"),
    ("max", ["left_wrist"], "unknown motor selection 'left_wrist'"),
])
def test_selection_errors(variant, selection, message, tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, f"ver2_{variant}")
    with pytest.raises(SystemExit) as info:
        mujoco_to_real.main(["--dry-run", "--motor-id", *selection])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err


@pytest.mark.parametrize("module, flag", [(mujoco_to_real, "--dry-run"), (real_to_mujoco, "--once")])
def test_robot_mismatch(module, flag, tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_edu")
    with pytest.raises(SystemExit) as info:
        module.main([flag, "--robot", "max"])
    assert info.value.code == 2
    assert "--robot max disagrees with the attached robot ver2_edu" in capsys.readouterr().err


def test_no_identity_file(tmp_path, monkeypatch, capsys):
    path = _identity(tmp_path, monkeypatch, None)
    for argv in ([], ["--dry-run"]):
        with pytest.raises(SystemExit) as info:
            mujoco_to_real.main(argv)
        assert info.value.code == 2
        err = capsys.readouterr().err
        assert "no robot identity file" in err and str(path) in err
    with pytest.raises(SystemExit) as info:
        mujoco_to_real.main(["--robot", "pro"])
    assert info.value.code == 2
    assert "no robot identity file" in capsys.readouterr().err

    assert mujoco_to_real.main(["--dry-run", "--robot", "pro", "--motor-id", "all"]) == 0
    out = capsys.readouterr().out
    assert f"Robot       : ver2_pro (from --robot pro (no identity file at {path}))" in out
    assert f"no robot identity file at {path} (variant came from --robot)" in out
    assert "Would enable ID 1–13, 15–18, 20–23" in out
    assert not path.exists()


def _fake_sys_net(tmp_path):
    root = tmp_path / "net"
    (root / "can0").mkdir(parents=True)
    (root / "can0" / "flags").write_text("0x40c1\n")
    (root / "can1").mkdir()
    (root / "can1" / "flags").write_text("0x40c0\n")
    return root


def test_channel_state_fake_sysfs(tmp_path):
    root = _fake_sys_net(tmp_path)
    assert bench.channel_state("can0", sys_net=root) == "ok"
    assert bench.channel_state("can1", sys_net=root) == "down"
    assert bench.channel_state("can2", sys_net=root) == "missing"
    assert bench.channel_state("can2", interface="virtual", sys_net=root) == "ok"


def test_channel_problems_lines(tmp_path, default_bus_map):
    root = _fake_sys_net(tmp_path)
    lines = bench.channel_problems([1, 2, 7, 8, 15, 20], sys_net=root)
    assert lines == [
        "can1: interface is down; needed by right_leg 7–8",
        "    sudo ip link set can1 up type can bitrate 1000000",
        f"can2: no such interface on this machine (adapter unplugged, or wrong name in {default_bus_map}); "
        "needed by left_arm 15",
        "    sudo ip link set can2 up type can bitrate 1000000",
        f"can3: no such interface on this machine (adapter unplugged, or wrong name in {default_bus_map}); "
        "needed by right_arm 20",
        "    sudo ip link set can3 up type can bitrate 1000000",
    ]
    assert bench.channel_problems([1, 2], sys_net=root) == []
    assert bench.channel_problems([7, 15], interface="virtual", sys_net=root) == []


def test_open_checked_never_opens_on_problem(tmp_path):
    root = _fake_sys_net(tmp_path)
    calls = []

    def opener(*args):
        calls.append(args)
        return "opened"

    with pytest.raises(RuntimeError) as info:
        bench.open_checked(opener, [1, 7], "socketcan", HOST_ID, sys_net=root)
    assert calls == []
    assert str(info.value).startswith("CAN channel not ready:\n  can1: interface is down")
    assert "sudo ip link set can1 up type can bitrate 1000000" in str(info.value)
    assert bench.open_checked(opener, [1, 2], "socketcan", HOST_ID, sys_net=root) == "opened"
    assert calls == [([1, 2], "socketcan", HOST_ID)]


def test_open_checked_translates_can_error(tmp_path):
    root = _fake_sys_net(tmp_path)

    def opener(*_args):
        raise can.CanError("adapter gone")

    with pytest.raises(RuntimeError) as info:
        bench.open_checked(opener, [1, 2], "socketcan", HOST_ID, sys_net=root)
    text = str(info.value)
    assert text.startswith("Could not open CAN (adapter gone):")
    assert "can0: could not be opened; needed by left_leg 1–2" in text
    assert "sudo ip link set can0 up type can bitrate 1000000" in text


def test_default_bus_plan_max():
    plan = bench.bus_plan(VARIANT_MOTOR_IDS["ver2_max"])
    assert plan == {
        "can0": {"left_leg": [1, 2, 3, 4, 5, 6]},
        "can1": {"right_leg": [7, 8, 9, 10, 11, 12]},
        "can2": {"left_arm": [15, 16, 17, 18]},
        "can3": {"right_arm": [20, 21, 22, 23]},
        "can4": {"head": [13]},
    }


def test_bus_map_file_is_honoured(default_bus_map):
    default_bus_map.write_text(json.dumps({"head": "can0", "right_arm": "can2"}))
    bus_map.cache_clear()
    plan = bench.bus_plan([1, 13, 15, 20])
    assert plan == {"can0": {"left_leg": [1], "head": [13]}, "can2": {"left_arm": [15], "right_arm": [20]}}


def test_gain_table_covers_every_motor(monkeypatch):
    bench.check_gain_table()
    for joint in ALL_MOTORS:
        assert joint.model_name in CONTROL_GAINS_BY_JOINT
    partial = {k: v for k, v in CONTROL_GAINS_BY_JOINT.items() if k != "neck_pitch_joint"}
    monkeypatch.setattr(bench, "CONTROL_GAINS_BY_JOINT", partial)
    with pytest.raises(RuntimeError) as info:
        bench.check_gain_table()
    assert "robonex-common has no gain entry for motor ID 13" in str(info.value)


def test_bench_gains_and_cap():
    ids = [joint.motor_id for joint in ALL_MOTORS]
    kp, kd, capped = bench.bench_gains(ids, 1.0)
    assert capped == []
    for mid in ids:
        assert (kp[mid], kd[mid]) == CONTROL_GAINS_BY_JOINT[MOTOR_BY_ID[mid].model_name]
    kp, kd, capped = bench.bench_gains(ids, 10.0)
    expected_capped = []
    for mid in ids:
        spec = MOTOR_SPECS[MOTOR_BY_ID[mid].motor_model]
        base_kp, base_kd = CONTROL_GAINS_BY_JOINT[MOTOR_BY_ID[mid].model_name]
        assert kp[mid] == min(base_kp * 10.0, spec.kp_max) <= spec.kp_max
        assert kd[mid] == min(base_kd * 10.0, spec.kd_max) <= spec.kd_max
        if base_kp * 10.0 > spec.kp_max or base_kd * 10.0 > spec.kd_max:
            expected_capped.append(mid)
    assert capped == expected_capped
    assert 13 in capped and kd[13] == MOTOR_SPECS["rs05"].kd_max


def test_dry_run_table_marks_capped(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_max")
    args = mujoco_to_real.parse_args(["--dry-run", "--motor-id", "head", "1"])
    args.gain_scale = 10.0
    mujoco_to_real.run(args)
    rows = _rows(capsys.readouterr().out, "Dry run:")
    assert "capped" in rows[13] and f"{MOTOR_SPECS['rs05'].kd_max:g}" in rows[13].split()
    assert "capped" in rows[1]


def _patched_loader(monkeypatch, motor_id, ctrlrange):
    original = mujoco_to_real.load_fixed_model

    def loader(path, motor_ids):
        model_path, model, actuators = original(path, motor_ids)
        model.actuator_ctrlrange[actuators[motor_id]] = ctrlrange
        return model_path, model, actuators

    monkeypatch.setattr(mujoco_to_real, "load_fixed_model", loader)


def test_arm_limit_mismatch_is_named(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_pro")
    _patched_loader(monkeypatch, 17, (-1.0, 1.0))
    assert mujoco_to_real.main(["--dry-run", "--motor-id", "legs", "arms"]) == 1
    out = capsys.readouterr().out
    assert "Stopped: MuJoCo actuator limits differ from robonex-common MOTOR_LIMITS_BY_ID:" in out
    assert (f"ID 17 left_shoulder_yaw (actuator l_shoulder_yaw): MuJoCo ctrlrange (-1.0, 1.0) "
            f"!= robonex-common {MOTOR_LIMITS_BY_ID[17]}") in out
    assert "ID 16 " not in out and "Dry run:" not in out
    assert mujoco_to_real.main(["--dry-run", "--motor-id", "legs", "right_arm"]) == 0
    assert "Would enable ID 1–12, 20–23" in capsys.readouterr().out


def test_leg_limit_mismatch_matches_no_profile(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_edu")
    _patched_loader(monkeypatch, 4, (-1.0, 0.1))
    assert mujoco_to_real.main(["--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "match 0 robonex-common robot models" in out and "Dry run:" not in out


class FakeMotors:
    def __init__(self, positions_by_channel):
        self.positions = {mid: value for group in positions_by_channel.values() for mid, value in group.items()}
        self.frames = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._buses = {ch: can.Bus(channel=ch, interface="virtual") for ch in positions_by_channel}
        self._threads = [threading.Thread(target=self._serve, args=(ch, bus), daemon=True)
                         for ch, bus in self._buses.items()]
        for thread in self._threads:
            thread.start()

    def _serve(self, channel, bus):
        while not self._stop.is_set():
            msg = bus.recv(timeout=0.01)
            if msg is None:
                continue
            comm, data16, target = parse_arbitration_id(msg.arbitration_id)
            with self._lock:
                self.frames.append((channel, comm, data16, target, bytes(msg.data)))
            value = self.positions.get(target)
            if comm != COMM_PARAMETER_READ or value is None:
                continue
            index = int.from_bytes(bytes(msg.data[0:2]), "little")
            payload = bytearray(8)
            struct.pack_into("<H", payload, 0, index)
            struct.pack_into("<f", payload, 4, value)
            bus.send(can.Message(arbitration_id=build_arbitration_id(COMM_PARAMETER_READ, target, data16 & 0xFF),
                                 data=bytes(payload), is_extended_id=True))

    def close(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        for bus in self._buses.values():
            bus.shutdown()


def _virtual_args(monkeypatch):
    original = real_to_mujoco.parse_args

    def parse(argv=None):
        args = original(argv)
        args.interface = "virtual"
        args.startup_timeout = 0.5
        return args

    monkeypatch.setattr(real_to_mujoco, "parse_args", parse)

    def guarded_open(motor_ids, interface, host_id):
        assert interface == "virtual"
        return safety.open_hardware(motor_ids, interface, host_id)

    monkeypatch.setattr(real_to_mujoco, "open_hardware", guarded_open)


def _assert_read_only(fake):
    assert fake.frames
    for _channel, comm, data16, _target, data in fake.frames:
        assert comm == COMM_PARAMETER_READ
        assert data16 == HOST_ID
        assert int.from_bytes(data[0:2], "little") == MECHANICAL_POSITION_INDEX


def test_real_to_mujoco_once_virtual_max(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_max")
    _virtual_args(monkeypatch)
    beyond = MOTOR_LIMITS_BY_ID[13][1] + math.radians(10.0)
    fake = FakeMotors({
        "can0": {1: 0.1, 4: -0.2},
        "can2": {15: 0.3},
        "can4": {13: beyond},
    })
    try:
        code = real_to_mujoco.main(["--once", "--motor-id", "1", "4", "13", "15", "16"])
    finally:
        fake.close()
    out = capsys.readouterr().out
    assert code == 1, out
    rows = _rows(out, "(one read)")
    assert sorted(rows) == [1, 4, 13, 15, 16]
    for mid in (1, 4, 15):
        assert rows[mid].split()[-2:] == ["OK", "ok"], rows[mid]
    assert "LIMIT" in rows[13] and _limit_text(13) in rows[13]
    assert "no response" in rows[16] and "NO REPLY" in rows[16]
    assert "-11.46deg" in rows[4] and "+17.19deg" in rows[15]
    _assert_read_only(fake)
    targets = {target for _c, _comm, _d, target, _data in fake.frames}
    assert targets == {1, 4, 13, 15, 16}
    channels = {target: channel for channel, _comm, _d, target, _data in fake.frames}
    assert channels == {1: "can0", 4: "can0", 15: "can2", 16: "can2", 13: "can4"}


def test_real_to_mujoco_once_virtual_pro_head_not_in_sim(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_pro")
    _virtual_args(monkeypatch)
    fake = FakeMotors({"can2": {15: -0.25}, "can4": {13: 0.1}})
    try:
        code = real_to_mujoco.main(["--once", "--motor-id", "head", "15"])
    finally:
        fake.close()
    out = capsys.readouterr().out
    assert code == 0, out
    rows = _rows(out, "(one read)")
    assert rows[13].split()[-4:] == ["OK", "not", "in", "sim"]
    assert _limit_text(13) in rows[13]
    assert rows[15].split()[-2:] == ["OK", "ok"] and _limit_text(15) in rows[15]
    _assert_read_only(fake)


@pytest.mark.parametrize("variant", ["max", "pro"])
def test_mirror_arm_and_neck_qpos(variant):
    model_path, model, actuators = safety.load_fixed_model(
        safety.fixed_model_path(variant), VARIANT_MOTOR_IDS[f"ver2_{variant}"]
    )
    model.opt.gravity[:] = 0.0
    targets = {15: -0.3, 18: 0.4, 22: 0.25, 4: -0.4}
    if variant == "max":
        targets[13] = 0.2
    else:
        assert 13 not in actuators
    selected = {mid: actuators[mid] for mid in targets}
    data = mujoco.MjData(model)
    qpos_addresses = real_to_mujoco.initialize_simulation(model, data, selected, targets)
    assert set(qpos_addresses) == set(targets)
    for mid, target in targets.items():
        joint = model.joint(MOTOR_BY_ID[mid].model_name)
        assert qpos_addresses[mid] == int(joint.qposadr[0])
        assert abs(data.qpos[joint.qposadr[0]] - target) < math.radians(1.0), (mid, data.qpos[joint.qposadr[0]])
        assert data.ctrl[selected[mid]] == pytest.approx(target)
    assert real_to_mujoco.sim_qpos_text(data, qpos_addresses, 16) == "--"
