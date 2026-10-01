#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

IMU_BUDGET_MS = 15.0


def load(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    if "imu_host_age_ms" not in rows[0]:
        raise ValueError(f"{path} has no imu_host_age_ms column; record it with the current policy_to_real.py")
    rows = [row for row in rows if not row.get("stop_reason")]
    columns = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]) if row[key] not in ("", None) else math.nan)
            except ValueError:
                values.append(math.nan)
        columns[key] = np.array(values)
    return columns


def distribution(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def phase_drift(t, age, window_s):
    mask = np.isfinite(t) & np.isfinite(age)
    t, age = t[mask], age[mask]
    if t.size < 2:
        return None
    windows = []
    start = t[0]
    while start <= t[-1]:
        inside = (t >= start) & (t < start + window_s)
        if inside.any():
            windows.append({
                "t_start_s": float(start),
                "mean_ms": float(np.mean(age[inside])),
                "p95_ms": float(np.percentile(age[inside], 95)),
                "band_ms": float(np.percentile(age[inside], 99) - np.percentile(age[inside], 1)),
            })
        start += window_s
    slope = float(np.polyfit(t, age, 1)[0]) * 60.0 if np.ptp(t) > 0.0 else 0.0
    return {"slope_ms_per_min": slope, "windows": windows}


def clock_rate(device_dt, host_dt):
    mask = np.isfinite(device_dt) & np.isfinite(host_dt)
    host_total = float(np.sum(host_dt[mask]))
    if host_total <= 0.0:
        return None
    device_total = float(np.sum(device_dt[mask]))
    return {"device_minus_host_ppm": (device_total - host_total) / host_total * 1.0e6,
            "host_span_s": host_total / 1000.0}


def seq_gaps(gap):
    gap = gap[np.isfinite(gap)].astype(int)
    if gap.size == 0:
        return None
    values, counts = np.unique(gap, return_counts=True)
    return {
        "ticks": int(gap.size),
        "repeated_sample_ticks": int(np.sum(gap < 0)),
        "histogram": {str(int(v)): int(c) for v, c in zip(values, counts)},
    }


def joint_ages(columns, suffix):
    result = {}
    for key in columns:
        if key.endswith(suffix) and "." in key:
            stats = distribution(columns[key])
            if stats is not None:
                result[key[: -len(suffix)]] = stats
    return result


def budget_line(age_stats, d0_ms, budget_ms=IMU_BUDGET_MS):
    if age_stats is None:
        return "IMU host age: no samples"
    worst, p95 = age_stats["max"], age_stats["p95"]
    if d0_ms is None:
        return (f"IMU host age max {worst:.2f} ms (p95 {p95:.2f} ms) stays inside {budget_ms:g} ms - d0 "
                f"for d0 <= {budget_ms - worst:.2f} ms (p95: d0 <= {budget_ms - p95:.2f} ms); d0 unknown")
    bound = budget_ms - d0_ms
    verdict = "INSIDE" if worst <= bound else "OUTSIDE"
    return (f"IMU host age max {worst:.2f} ms (p95 {p95:.2f} ms) vs bound {budget_ms:g} - {d0_ms:g} = "
            f"{bound:.2f} ms: {verdict}")


def report(path, d0_ms=None, window_s=5.0, budget_ms=IMU_BUDGET_MS):
    columns = load(path)
    t = columns.get("t_s")
    age = columns["imu_host_age_ms"]
    age_stats = distribution(age)
    tick = distribution(columns["tick_period_ms"])
    if tick is not None:
        tick["jitter_ms"] = tick["std"]
    return {
        "file": str(path),
        "imu_host_age_ms": age_stats,
        "imu_legacy_age_ms": distribution(columns.get("imu_age_ms", np.array([]))),
        "phase_drift": phase_drift(t, age, window_s) if t is not None else None,
        "imu_device_dt_ms": distribution(columns["imu_device_dt_ms"]),
        "imu_host_dt_ms": distribution(columns["imu_host_dt_ms"]),
        "clock_rate": clock_rate(columns["imu_device_dt_ms"], columns["imu_host_dt_ms"]),
        "imu_seq_gap": seq_gaps(columns["imu_seq_gap"]),
        "tick_period_ms": tick,
        "joint_fb_age_ms": joint_ages(columns, ".age_ms"),
        "joint_rx_age_ms": joint_ages(columns, ".rx_age_ms"),
        "budget": budget_line(age_stats, d0_ms, budget_ms),
    }


def format_stats(name, stats):
    if stats is None:
        return f"  {name:<22} --"
    return (f"  {name:<22} n {stats['n']:6d}  mean {stats['mean']:7.3f}  p50 {stats['p50']:7.3f}  "
            f"p95 {stats['p95']:7.3f}  max {stats['max']:7.3f}  std {stats['std']:6.3f}")


def print_report(result):
    print(f"File: {result['file']}")
    print("IMU")
    print(format_stats("host age [ms]", result["imu_host_age_ms"]))
    print(format_stats("legacy imu_age [ms]", result["imu_legacy_age_ms"]))
    print(format_stats("device dt [ms]", result["imu_device_dt_ms"]))
    print(format_stats("host dt [ms]", result["imu_host_dt_ms"]))
    rate = result["clock_rate"]
    if rate is not None:
        print(f"  device vs host clock   {rate['device_minus_host_ppm']:+.1f} ppm over {rate['host_span_s']:.1f} s")
    gaps = result["imu_seq_gap"]
    if gaps is not None:
        print(f"  seq gap per tick       {gaps['histogram']}  (repeated sample on {gaps['repeated_sample_ticks']} ticks)")
    drift = result["phase_drift"]
    if drift is not None:
        print(f"  phase drift            host age slope {drift['slope_ms_per_min']:+.3f} ms/min")
        for window in drift["windows"]:
            print(f"    t {window['t_start_s']:7.1f} s  mean {window['mean_ms']:7.3f}  p95 {window['p95_ms']:7.3f}  "
                  f"p1-p99 band {window['band_ms']:6.3f}")
    print("Loop")
    tick = result["tick_period_ms"]
    print(format_stats("tick period [ms]", tick))
    for label, key in (("Joint feedback age at drain [ms]", "joint_fb_age_ms"),
                       ("Joint kernel rx age [ms]", "joint_rx_age_ms")):
        ages = result[key]
        if ages:
            print(label)
            for name, stats in ages.items():
                print(format_stats(name, stats))
    print(result["budget"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="IMU and joint-feedback timing from a policy_to_real telemetry CSV.")
    parser.add_argument("csv", type=Path, nargs="+", help="*_live_telemetry.csv or *_read_telemetry.csv")
    parser.add_argument("--d0-ms", type=float, default=None,
                        help="Fixed IMU delay before the host stamp, ms; omit to print the bound as a function of d0")
    parser.add_argument("--budget-ms", type=float, default=IMU_BUDGET_MS, help="Total IMU age budget, ms")
    parser.add_argument("--window", type=float, default=5.0, help="Phase-drift window, s")
    parser.add_argument("--output", type=Path, help="JSON output path")
    args = parser.parse_args(argv)
    results = []
    for path in args.csv:
        result = report(path, d0_ms=args.d0_ms, window_s=args.window, budget_ms=args.budget_ms)
        print_report(result)
        results.append(result)
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
