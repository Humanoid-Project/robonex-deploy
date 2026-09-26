#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(np.nan)
        columns[key] = np.array(values)
    meta_path = Path(path).with_name(Path(path).stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return columns, meta


def step_latency(c, threshold_rad, window_s=1.0):
    target = c["target"]
    edges = np.flatnonzero(np.abs(np.diff(target)) > 1e-6) + 1
    results = []
    for edge in edges:
        sent_wall = c["wall_time"][edge]
        before = c["pos"][max(0, edge - 5):edge]
        if len(before) == 0 or not np.all(np.isfinite(before)):
            continue
        base = float(np.mean(before))
        direction = np.sign(target[edge] - target[edge - 1])
        later = np.flatnonzero((c["t_s"] > c["t_s"][edge]) & (c["t_s"] <= c["t_s"][edge] + window_s))
        ok = (direction * (c["pos"][later] - base) > threshold_rad) & (c["new_frame"][later] > 0) & (
            c["rx_kernel_time"][later] > sent_wall)
        moved = later[ok]
        if len(moved) == 0:
            continue
        first = moved[0]
        results.append({
            "edge_row": int(edge),
            "step_rad": float(target[edge] - target[edge - 1]),
            "latency_ms_kernel": float((c["rx_kernel_time"][first] - sent_wall) * 1000.0),
            "latency_ms_rows": float((c["t_s"][first] - c["t_s"][edge]) * 1000.0),
        })
    return results


def triangle_hysteresis(c):
    target, pos = c["target"], c["pos"]
    velocity = np.gradient(target, c["t_s"])
    fresh = c["new_frame"] > 0
    up = fresh & (velocity > 1e-4)
    down = fresh & (velocity < -1e-4)
    if up.sum() < 20 or down.sum() < 20:
        return None
    lag_up = float(np.median(target[up] - pos[up]))
    lag_down = float(np.median(target[down] - pos[down]))
    bins = np.linspace(np.nanmin(target), np.nanmax(target), 21)
    widths = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        a = pos[up & (target >= lo) & (target < hi)]
        b = pos[down & (target >= lo) & (target < hi)]
        if len(a) > 3 and len(b) > 3:
            widths.append(float(np.median(b) - np.median(a)))
    return {
        "median_target_minus_pos_rising_rad": lag_up,
        "median_target_minus_pos_falling_rad": lag_down,
        "position_hysteresis_width_rad": float(np.median(widths)) if widths else None,
        "note": "width = pos(falling) - pos(rising) at the same target, from the motor-side encoder: friction, kd and "
                "delay (2 x speed x delay); gearbox play is invisible to this encoder",
    }


def chirp_response(c, meta, frequencies):
    wall, target = c["wall_time"], c["target"]
    fresh = (c["new_frame"] > 0) & np.isfinite(c["pos"]) & np.isfinite(c["rx_kernel_time"])
    rx, pos = c["rx_kernel_time"][fresh], c["pos"][fresh]
    t0 = wall[0]
    f0, f1, duration = float(meta["f0"]), float(meta["f1"]), float(meta["duration"])
    k = np.log(f1 / f0) / duration
    dt = float(np.median(np.diff(wall)))
    out = []
    for f in frequencies:
        if not f0 < f < f1:
            continue
        center = np.log(f / f0) / k
        half = 2.0 / f
        if center - half < 0.0 or center + half > duration:
            continue
        grid = np.arange(center - half, center + half, dt / 4.0) + t0
        x = np.interp(grid, wall, target)
        y = np.interp(grid, rx, pos)
        x -= x.mean()
        y -= y.mean()
        ref = np.exp(-2j * np.pi * f * (grid - t0)) * np.hanning(len(grid))
        xs, ys = np.sum(x * ref), np.sum(y * ref)
        if abs(xs) < 1e-12:
            continue
        h = ys / xs
        out.append({"hz": f, "gain": float(abs(h)), "phase_deg": float(np.degrees(np.angle(h))),
                    "delay_ms_equiv": float(-np.angle(h) / (2 * np.pi * f) * 1000.0)})
    return out


def main():
    parser = argparse.ArgumentParser(description="Latency, hysteresis and frequency response from joint_probe.py recordings.")
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--threshold-deg", type=float, default=0.2, help="step: motion onset threshold")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.csv:
        c, meta = load(path)
        profile = meta.get("profile")
        if profile is None:
            raise SystemExit(f"{path}: the *_meta.json written by joint_probe.py is missing")
        result = {"file": str(path), "profile": profile, "motor_id": meta.get("motor_id"),
                  "kp": meta.get("kp"), "kd": meta.get("kd")}
        if profile == "step":
            steps = step_latency(c, np.radians(args.threshold_deg))
            lat = np.array([s["latency_ms_kernel"] for s in steps])
            result["steps"] = steps
            if len(lat):
                result["latency_ms_kernel"] = {"n": int(len(lat)), "p50": float(np.percentile(lat, 50)),
                                               "p95": float(np.percentile(lat, 95)), "max": float(lat.max())}
        elif profile == "triangle":
            result["hysteresis"] = triangle_hysteresis(c)
        else:
            result["response"] = chirp_response(c, meta, [0.3, 0.5, 1.0, 1.29, 2.0, 3.0, 4.0, 5.0, 8.0])
        results.append(result)
        print(json.dumps(result if profile != "step" else {k: v for k, v in result.items() if k != "steps"}, indent=1))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
