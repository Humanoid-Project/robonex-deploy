import pytest

from test_policy_to_real_timing import modules  # noqa: F401


@pytest.mark.parametrize("version", ["2.4.1", "2.3.0", "1.2.0", "2.4.9", "", "x.y"])
def test_zero_head_refused_before_common_2_5(modules, version):
    _, ptr = modules
    reason = ptr.zero_head_refusal(version)
    assert reason and "CAN ID 26" in reason and ">= 2.5.0" in reason


@pytest.mark.parametrize("version", ["2.5.0", "2.6.0", "2.10.1", "3.0.0"])
def test_zero_head_allowed_from_common_2_5(modules, version):
    _, ptr = modules
    assert ptr.zero_head_refusal(version) is None


def test_main_refuses_zero_head_on_old_common_before_any_setup(modules, tmp_path, monkeypatch):
    _, ptr = modules
    monkeypatch.setattr(ptr.robonex_common, "__version__", "2.4.1")

    def must_not_run(_args):
        raise AssertionError("run_logged must not be reached")

    monkeypatch.setattr(ptr, "run_logged", must_not_run)
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"")
    with pytest.raises(SystemExit) as info:
        ptr.main(["--policy", str(policy), "--zero-head"])
    assert "--zero-head is refused with robonex-common 2.4.1" in str(info.value)


def test_main_without_zero_head_is_not_affected_by_the_guard(modules, tmp_path, monkeypatch):
    _, ptr = modules
    monkeypatch.setattr(ptr.robonex_common, "__version__", "2.4.1")
    reached = []
    monkeypatch.setattr(ptr, "run_logged", lambda args: reached.append(args.zero_head) or 0)
    for name in ("SETTINGS", "ENABLE_KP", "ENABLE_KD"):
        monkeypatch.setattr(ptr, name, getattr(ptr, name))
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"")
    assert ptr.main(["--policy", str(policy)]) == 0
    assert reached == [False]
