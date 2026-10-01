import builtins
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import can
import pytest

_MAP_DIR = Path(tempfile.mkdtemp(prefix="busmap_"))
(_MAP_DIR / "bus_map.json").write_text(json.dumps({"right_arm": "can0"}))
os.environ["ROBONEX_BUS_MAP"] = str(_MAP_DIR / "bus_map.json")

DEPLOY = Path("/home/polygon/humanoid_project/robonex-deploy")

import bench
import safety
import mujoco_to_real
import real_to_mujoco
from robonex_common.protocol import COMM_STOP, parse_arbitration_id

_spec = importlib.util.spec_from_file_location("joint_probe", DEPLOY / "scripts" / "sysid" / "joint_probe.py")
joint_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(joint_probe)


class _Tty:
    def isatty(self):
        return True


def _identity(tmp_path, monkeypatch, content):
    path = tmp_path / "robot_model"
    if content is not None:
        path.write_text(content + "\n")
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(path))
    return path


def _probe_args(tmp_path, monkeypatch, motor_id=1, robot="ver2_edu"):
    monkeypatch.setattr(sys, "stdin", _Tty())
    return joint_probe.parse_args([
        "--motor-id", str(motor_id), "--profile", "step", "--robot-model", robot,
        "--output", str(tmp_path / "probe.csv"),
    ])


def _no_hardware(*_args, **_kwargs):
    raise AssertionError("CAN must not be opened")


def test_bus_map_fixture():
    from robonex_common.joints import MOTOR_BY_ID
    assert MOTOR_BY_ID[20].channel == "can0"
    assert MOTOR_BY_ID[13].channel == "can4"


def test_identity_reader_missing_and_directory(tmp_path):
    assert bench.attached_robot_model(tmp_path / "absent") is None
    with pytest.raises(ValueError) as info:
        bench.attached_robot_model(tmp_path)
    assert str(tmp_path) in str(info.value) and "cannot read the robot identity file" in str(info.value)


def test_identity_reader_unreadable(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root reads any file")
    path = tmp_path / "robot_model"
    path.write_text("ver2_edu\n")
    path.chmod(0)
    try:
        with pytest.raises(ValueError) as info:
            bench.attached_robot_model(path)
        assert str(path) in str(info.value) and "Permission denied" in str(info.value)
    finally:
        path.chmod(0o600)


def test_require_robot_model_directory_is_runtime_error(tmp_path):
    with pytest.raises(RuntimeError) as info:
        safety.require_robot_model("ver2_edu", tmp_path)
    assert str(tmp_path) in str(info.value) and "Motors will not be enabled" in str(info.value)


@pytest.mark.parametrize("module", [mujoco_to_real, real_to_mujoco])
def test_mains_identity_directory_is_friendly(module, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(tmp_path))
    with pytest.raises(SystemExit) as info:
        module.main(["--robot", "edu"] + (["--dry-run"] if module is mujoco_to_real else ["--once"]))
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "cannot read the robot identity file" in err and str(tmp_path) in err
    assert "Traceback" not in err


def test_mujoco_to_real_run_identity_unreadable_exit_1(tmp_path, monkeypatch, capsys):
    path = _identity(tmp_path, monkeypatch, "ver2_edu")
    args = mujoco_to_real.parse_args([])
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(tmp_path))
    monkeypatch.setattr(mujoco_to_real, "open_hardware", _no_hardware)
    monkeypatch.setattr(mujoco_to_real, "parse_args", lambda argv=None: args)
    assert mujoco_to_real.main([]) == 1
    out = capsys.readouterr().out
    assert "Stopped: cannot read the robot identity file" in out
    assert path.exists()


@pytest.mark.parametrize("content, expected", [
    (None, "No robot identity"),
    ("ver2_pro", "The attached robot is ver2_pro"),
    ("ver9", "names an unknown robot model 'ver9'"),
])
def test_joint_probe_identity_errors_exit_1(content, expected, tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, content)
    args = _probe_args(tmp_path, monkeypatch)
    monkeypatch.setattr(joint_probe, "open_hardware", _no_hardware)
    monkeypatch.setattr(builtins, "input", lambda *_: pytest.fail("no confirmation before the identity check"))
    assert joint_probe.run(args) == 1
    out = capsys.readouterr().out
    assert "Stopped: " in out and expected in out
    assert not (tmp_path / "probe.csv").exists()


def test_joint_probe_identity_directory_exit_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(tmp_path))
    args = _probe_args(tmp_path, monkeypatch)
    monkeypatch.setattr(joint_probe, "open_hardware", _no_hardware)
    monkeypatch.setattr(builtins, "input", lambda *_: pytest.fail("no confirmation before the identity check"))
    assert joint_probe.run(args) == 1
    assert "Stopped: cannot read the robot identity file" in capsys.readouterr().out


def test_joint_probe_ctrl_c_at_confirmation(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_edu")
    args = _probe_args(tmp_path, monkeypatch)
    monkeypatch.setattr(joint_probe, "open_hardware", _no_hardware)

    def interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupt)
    assert joint_probe.run(args) == 1
    assert "Stop requested." in capsys.readouterr().out


def test_joint_probe_main_identity_error_no_traceback(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, None)
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(joint_probe, "open_hardware", _no_hardware)
    code = joint_probe.main(["--motor-id", "1", "--profile", "step", "--robot-model", "ver2_edu",
                             "--output", str(tmp_path / "probe.csv")])
    assert code == 1
    assert "No robot identity" in capsys.readouterr().out


def test_joint_probe_idle_stop_respects_variant(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_edu")
    args = _probe_args(tmp_path, monkeypatch, motor_id=1, robot="ver2_edu")
    monkeypatch.setattr(joint_probe, "DEFAULT_INTERFACE", "virtual")
    monkeypatch.setattr(builtins, "input", lambda *_: "")

    def stop_before_enable(*_args, **_kwargs):
        raise RuntimeError("test stop before enable")

    monkeypatch.setattr(joint_probe, "enable_with_runtime_feedback", stop_before_enable)
    listener = can.Bus(channel="can0", interface="virtual")
    try:
        assert joint_probe.run(args) == 1
        stopped = set()
        while True:
            message = listener.recv(timeout=0.05)
            if message is None:
                break
            comm, _, motor_id = parse_arbitration_id(message.arbitration_id)
            if comm == COMM_STOP:
                stopped.add(motor_id)
    finally:
        listener.shutdown()
    assert stopped - {1} == {2, 3, 4, 5, 6}
    out = capsys.readouterr().out
    assert "Stop frame first to the other ver2_edu motors on can0: 2–6" in out
    assert "Stopped: test stop before enable" in out


def test_joint_probe_idle_banner_pro(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_edu")
    args = _probe_args(tmp_path, monkeypatch, motor_id=1, robot="ver2_pro")
    monkeypatch.setattr(joint_probe, "open_hardware", _no_hardware)
    assert joint_probe.run(args) == 1
    out = capsys.readouterr().out
    assert "Stop frame first to the other ver2_pro motors on can0: 2–6, 20–23" in out


def test_help_texts(capsys):
    for module in (mujoco_to_real, real_to_mujoco):
        with pytest.raises(SystemExit):
            module.parse_args(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "(head, arms)" not in out
    assert "without a MuJoCo actuator on the attached variant is moved to zero and held there" in out
    assert "without a MuJoCo actuator on the attached variant is read and printed only" in out
    assert "Sends only parameter reads (mechPos) and, except with --once, stop frames" in out


def test_placeholder_banner_wording(capsys):
    ids = [15, 16, 17, 18]
    bench.print_banner("ver2_pro", "test", ids, {mid: mid for mid in ids}, "model.xml", interface="virtual")
    out = capsys.readouterr().out
    assert "PLACEHOLDER : limits of ID 15–18 equal the common placeholder values (±30° head, ±45° arms)" in out
    assert "unmeasured" not in out


def test_dry_run_legs_still_works(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch, "ver2_edu")
    assert mujoco_to_real.main(["--dry-run"]) in (0, None)
    out = capsys.readouterr().out
    assert "Would enable ID 1–12" in out
