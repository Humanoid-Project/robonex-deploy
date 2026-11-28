import builtins
import functools
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import bench
import mujoco_to_real
from robonex_common.joints import JOINT_BY_MODEL_NAME, MOTOR_BY_ID
from test_policy_to_real_timing import make_contract, modules

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = make_contract()
LEG_IDS = [JOINT_BY_MODEL_NAME[name].motor_id for name in CONTRACT.joint_order]


def _fake_sysfs(root, channels, counts=(100, 200)):
    for channel in channels:
        statistics = root / channel / "statistics"
        statistics.mkdir(parents=True)
        (statistics / "tx_packets").write_text(f"{counts[0]}\n")
        (statistics / "rx_packets").write_text(f"{counts[1]}\n")
    return root


def _bump(root, channel, name, by):
    path = root / channel / "statistics" / name
    path.write_text(f"{int(path.read_text()) + by}\n")


def test_foreign_traffic_still_counters_pass(tmp_path):
    channels = list(bench.bus_plan(LEG_IDS))
    root = _fake_sysfs(tmp_path, channels)
    waits = []
    assert bench.check_foreign_traffic(LEG_IDS, "socketcan", sys_net=root, sleep=waits.append) == []
    assert waits == [0.5]


def test_foreign_traffic_moving_tx_is_refused_with_rates_and_hand_check(tmp_path):
    channels = list(bench.bus_plan(LEG_IDS))
    root = _fake_sysfs(tmp_path, channels)
    with pytest.raises(RuntimeError) as info:
        bench.check_foreign_traffic(LEG_IDS, "socketcan", sys_net=root,
                                    sleep=lambda s: _bump(root, channels[0], "tx_packets", 50))
    text = str(info.value)
    assert f"{channels[0]}: 100 tx/s, 0 rx/s" in text
    assert bench.FOREIGN_HAND_CHECK in text and "--allow-foreign-readers" in text
    assert all(f"{channel}:" not in text for channel in channels[1:])


def test_foreign_traffic_moving_rx_is_refused(tmp_path):
    channels = list(bench.bus_plan(LEG_IDS))
    root = _fake_sysfs(tmp_path, channels)
    with pytest.raises(RuntimeError) as info:
        bench.check_foreign_traffic(LEG_IDS, "socketcan", sys_net=root,
                                    sleep=lambda s: _bump(root, channels[-1], "rx_packets", 1))
    assert f"{channels[-1]}: 0 tx/s, 2 rx/s" in str(info.value)


def test_foreign_traffic_allowed_warns(tmp_path, capsys):
    channels = list(bench.bus_plan(LEG_IDS))
    root = _fake_sysfs(tmp_path, channels)
    lines = bench.check_foreign_traffic(LEG_IDS, "socketcan", allow=True, sys_net=root,
                                        sleep=lambda s: _bump(root, channels[0], "tx_packets", 5))
    assert len(lines) == 1
    out = capsys.readouterr().out
    assert out.startswith("Warning: CAN traffic before this program opened the bus")
    assert "Continuing because of --allow-foreign-readers" in out


def test_foreign_traffic_virtual_interface_is_skipped(tmp_path):
    waits = []
    assert bench.check_foreign_traffic(LEG_IDS, "virtual", sys_net=tmp_path / "none", sleep=waits.append) == []
    assert waits == []


def test_foreign_traffic_missing_interface_is_left_to_the_open(tmp_path):
    waits = []
    assert bench.check_foreign_traffic(LEG_IDS, "socketcan", sys_net=tmp_path, sleep=waits.append) == []
    assert waits == []


def test_foreign_traffic_unreadable_counters_refuse(tmp_path):
    channels = list(bench.bus_plan(LEG_IDS))
    root = _fake_sysfs(tmp_path, channels)
    (root / channels[0] / "statistics" / "tx_packets").write_text("garbage\n")
    with pytest.raises(RuntimeError) as info:
        bench.check_foreign_traffic(LEG_IDS, "socketcan", sys_net=root, sleep=lambda s: None)
    assert f"{channels[0]}: cannot read tx/rx packet counters" in str(info.value)


class Opened(Exception):
    pass


def run_deploy_to_open(module, monkeypatch, root, sleep, allow=False):
    opened = []

    def open_hardware(ids, interface, host_id):
        opened.append(list(ids))
        raise Opened()

    monkeypatch.setattr(module, "verify_common_source", lambda contract: None)
    monkeypatch.setattr(module, "require_robot_model", lambda name: None)
    monkeypatch.setattr(module, "check_foreign_traffic",
                        functools.partial(bench.check_foreign_traffic, sys_net=root, sleep=sleep))
    monkeypatch.setattr(module, "open_hardware", open_hardware)
    monkeypatch.setattr(module, "brake_and_stop", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "shutdown_report_lines", lambda report: [])
    monkeypatch.setattr(module.ImuSource, "stop", lambda self: None)
    args = SimpleNamespace(vx=0.0, vy=0.0, wz=0.0, keyboard=False, duration=1.0, telemetry=None,
                           allow_foreign_readers=allow)
    try:
        module.run_deploy(Path("policy.onnx"), make_contract(), args)
    except Opened:
        return opened, None
    except RuntimeError as error:
        return opened, error
    return opened, None


def test_deploy_refuses_foreign_traffic_before_any_bus_opens(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    channels = list(bench.bus_plan(LEG_IDS))
    root = _fake_sysfs(tmp_path, channels)
    opened, error = run_deploy_to_open(new, monkeypatch, root, lambda s: _bump(root, channels[0], "tx_packets", 20))
    assert opened == []
    assert error is not None and "CAN traffic before this program opened the bus" in str(error)
    assert "CAN buses closed. No motor control command was sent." in capsys.readouterr().out


def test_deploy_still_counters_open_the_bus(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    root = _fake_sysfs(tmp_path, list(bench.bus_plan(LEG_IDS)))
    opened, error = run_deploy_to_open(new, monkeypatch, root, lambda s: None)
    assert error is None and opened and sorted(opened[0]) == sorted(LEG_IDS)


def test_deploy_allow_foreign_readers_warns_and_opens(modules, monkeypatch, tmp_path, capsys):
    _, new = modules
    channels = list(bench.bus_plan(LEG_IDS))
    root = _fake_sysfs(tmp_path, channels)
    opened, error = run_deploy_to_open(new, monkeypatch, root, lambda s: _bump(root, channels[0], "rx_packets", 7),
                                       allow=True)
    assert error is None and opened
    assert "Continuing because of --allow-foreign-readers" in capsys.readouterr().out


def test_deploy_allow_foreign_readers_flag(modules, tmp_path):
    _, new = modules
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"")
    assert new.parse_args(["--policy", str(policy)]).allow_foreign_readers is False
    assert new.parse_args(["--policy", str(policy), "--allow-foreign-readers"]).allow_foreign_readers is True


def _identity(tmp_path, monkeypatch):
    identity = tmp_path / "robot_model"
    identity.write_text("ver2_edu\n")
    monkeypatch.setenv(bench.ROBOT_IDENTITY_ENV, str(identity))


def test_mujoco_to_real_refuses_foreign_traffic_before_opening(tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch)
    args = mujoco_to_real.parse_args([])
    assert args.allow_foreign_readers is False
    assert mujoco_to_real.parse_args(["--allow-foreign-readers"]).allow_foreign_readers is True
    channels = list(bench.bus_plan(args.motor_ids))
    root = _fake_sysfs(tmp_path / "net", channels)
    monkeypatch.setattr(mujoco_to_real, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(mujoco_to_real, "check_foreign_traffic", functools.partial(
        bench.check_foreign_traffic, sys_net=root, sleep=lambda s: _bump(root, channels[0], "tx_packets", 3)))
    monkeypatch.setattr(mujoco_to_real, "open_checked", lambda *a, **k: pytest.fail("a bus was opened"))
    assert mujoco_to_real.main([]) == 1
    out = capsys.readouterr().out
    assert "CAN traffic before this program opened the bus" in out and f"{channels[0]}: 6 tx/s" in out


@pytest.fixture
def joint_probe():
    spec = importlib.util.spec_from_file_location("joint_probe_guards", ROOT / "scripts" / "sysid" / "joint_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tty:
    def isatty(self):
        return True


def test_joint_probe_refuses_foreign_traffic_before_opening(joint_probe, tmp_path, monkeypatch, capsys):
    _identity(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "stdin", _Tty())
    argv = ["--motor-id", "1", "--profile", "step", "--robot-model", "ver2_edu", "--output", str(tmp_path / "p.csv")]
    assert joint_probe.parse_args(argv + ["--allow-foreign-readers"]).allow_foreign_readers is True
    args = joint_probe.parse_args(argv)
    assert args.allow_foreign_readers is False
    channel = MOTOR_BY_ID[1].channel
    root = _fake_sysfs(tmp_path / "net", [channel])
    monkeypatch.setattr(joint_probe, "check_foreign_traffic", functools.partial(
        bench.check_foreign_traffic, sys_net=root, sleep=lambda s: _bump(root, channel, "rx_packets", 4)))
    monkeypatch.setattr(joint_probe, "open_hardware", lambda *a, **k: pytest.fail("a bus was opened"))
    monkeypatch.setattr(builtins, "input", lambda *_: "")
    assert joint_probe.run(args) == 1
    out = capsys.readouterr().out
    assert "Stopped: CAN traffic before this program opened the bus" in out and f"{channel}: 0 tx/s, 8 rx/s" in out
