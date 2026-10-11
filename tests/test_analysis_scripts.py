import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def metrics():
    return load(ROOT / "scripts/analysis/scenario_metrics.py", "scenario_metrics_under_test")


@pytest.fixture(scope="module")
def probe():
    return load(ROOT / "scripts/sysid/analyze_probe.py", "analyze_probe_under_test")


def test_dominant_frequency_finite_and_nan(metrics):
    dt = 0.02
    x = np.sin(2 * np.pi * 1.25 * np.arange(500) * dt)
    assert metrics.dominant_frequency(x, dt) == pytest.approx(1.25, abs=0.1)
    x[100] = np.nan
    assert metrics.dominant_frequency(x, dt) is None
    assert metrics.dominant_frequency(np.full(500, np.inf), dt) is None


def test_load_header_only_csv(probe, tmp_path):
    path = tmp_path / "probe.csv"
    path.write_text("t_s,wall_time,target,pos,new_frame,rx_kernel_time\n")
    with pytest.raises(SystemExit, match="no samples"):
        probe.load(path)


def chirp_recording(delay_s, f0=0.2, f1=10.0, duration=60.0, rate=1000.0):
    wall = np.arange(0.0, duration, 1.0 / rate)
    k = np.log(f1 / f0) / duration

    def signal(t):
        return 0.1 * np.sin(2 * np.pi * f0 * (np.exp(k * t) - 1.0) / k)

    c = {"wall_time": wall, "target": signal(wall), "pos": signal(wall - delay_s),
         "new_frame": np.ones_like(wall), "rx_kernel_time": wall.copy()}
    return c, {"f0": f0, "f1": f1, "duration": duration}


def test_chirp_delay_is_unwrapped_past_180_degrees(probe):
    c, meta = chirp_recording(0.065)
    response = probe.chirp_response(c, meta, [0.3, 0.5, 1.0, 1.29, 2.0, 3.0, 4.0, 5.0, 8.0])
    by_hz = {r["hz"]: r for r in response}
    assert 8.0 in by_hz and by_hz[8.0]["phase_deg"] < -180.0
    for r in response:
        assert r["delay_ms_equiv"] == pytest.approx(65.0, abs=3.0)
        assert r["gain"] == pytest.approx(1.0, abs=0.05)


def test_chirp_short_delay_unchanged(probe):
    c, meta = chirp_recording(0.020)
    for r in probe.chirp_response(c, meta, [1.0, 4.0, 8.0]):
        assert -180.0 < r["phase_deg"] < 0.0
        assert r["delay_ms_equiv"] == pytest.approx(20.0, abs=3.0)
