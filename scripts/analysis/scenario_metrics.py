#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

JOINTS = (
    "l_hip_yaw", "r_hip_yaw", "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll",
    "l_knee_pitch", "r_knee_pitch", "l_ankle_upper", "r_ankle_upper", "l_ankle_lower", "r_ankle_lower",
)
PEAK_TORQUE = {"hip_yaw": 17.0, "ankle_upper": 17.0, "ankle_lower": 17.0,
               "hip_pitch": 60.0, "hip_roll": 60.0, "knee_pitch": 60.0}
STANDSTILL_TORQUE = {"hip_yaw": 6.0, "ankle_upper": 6.0, "ankle_lower": 6.0,
                     "hip_pitch": 13.0, "hip_roll": 13.0, "knee_pitch": 13.0}
SPINNING_TORQUE = {"hip_yaw": 6.0, "ankle_upper": 6.0, "ankle_lower": 6.0,
                   "hip_pitch": 20.0, "hip_roll": 20.0, "knee_pitch": 20.0}


def load(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    columns = {}
    for key in rows[0]:
        try:
            columns[key] = np.array([float(r[key]) if r[key] not in ("", None) else math.nan for r in rows])
        except ValueError:
            continue
    return columns


def source_kind(columns):
    if any(k.endswith(".commanded") for k in columns):
        return "hardware"
    if "root_vx_b" in columns:
        return "sim"
    raise ValueError("unknown CSV layout: expected hardware telemetry or a sim trace")


def segments(columns, settle_s, min_ramp):
    cmd = np.stack([columns["cmd_vx"], columns["cmd_vy"], columns["cmd_wz"]], axis=1)
    t = columns["t_s"]
    ok = np.ones(len(t), dtype=bool)
    if "ramp" in columns:
        ok &= columns["ramp"] >= min_ramp
    if "reset" in columns:
        ok &= columns["reset"] < 0.5
    change = np.r_[True, np.any(np.abs(np.diff(cmd, axis=0)) > 1e-9, axis=1)]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(t)]
    out = []
    for start, end in zip(starts, ends):
        index = np.arange(start, end)
        index = index[(t[index] >= t[start] + settle_s) & ok[index]]
        if len(index) >= 50:
            out.append((tuple(float(v) for v in cmd[start]), index))
    return out


def dominant_frequency(x, dt, band=(0.3, 3.0)):
    x = x - np.mean(x)
    if len(x) < 32 or not np.any(x):
        return None
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), dt)
    mask = (freq >= band[0]) & (freq <= band[1])
    if not np.any(mask):
        return None
    return float(freq[mask][np.argmax(spectrum[mask])])


def band_rms(x, dt, low, high):
    x = x - np.mean(x)
    if len(x) < 32:
        return None
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), dt)
    weight = np.full(freq.shape, 2.0)
    weight[0] = 1.0
    if len(x) % 2 == 0:
        weight[-1] = 1.0
    mask = (freq >= low) & (freq < high)
    return float(np.sqrt(np.sum(spectrum[mask] * weight[mask]) / len(x) ** 2))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def segment_metrics(columns, kind, index):
    t = columns["t_s"][index]
    dt = float(np.median(np.diff(t)))
    gx, gy, gz = (columns[f"gravity_{a}"][index] for a in "xyz")
    roll = np.degrees(np.arctan2(-gy, -gz))
    pitch = np.degrees(np.arctan2(gx, np.sqrt(gy ** 2 + gz ** 2)))
    gyro = {a: columns[f"gyro_{a}"][index] for a in "xyz"}
    out = {
        "duration_s": float(t[-1] - t[0] + dt),
        "samples": int(len(index)),
        "dt_s": dt,
        "roll_deg": {"mean": float(np.mean(roll)), "osc_rms": float(np.std(roll)),
                     "peak_hz": dominant_frequency(gyro["x"], dt)},
        "pitch_deg": {"mean": float(np.mean(pitch)), "osc_rms": float(np.std(pitch)),
                      "peak_hz": dominant_frequency(gyro["y"], dt)},
        "yaw_rate": {"mean_rad_s": float(np.mean(gyro["z"])), "osc_rms_rad_s": float(np.std(gyro["z"])),
                     "peak_hz": dominant_frequency(gyro["z"], dt),
                     "gait_band_rms_rad_s": band_rms(gyro["z"], dt, 0.3, 3.0),
                     "above_3hz_rms_rad_s": band_rms(gyro["z"], dt, 3.0, 1.0e9),
                     "heading_drift_deg_per_s": float(np.degrees(np.mean(gyro["z"])))},
        "forward_speed_m_s": float(np.mean(columns["root_vx_b"][index])) if "root_vx_b" in columns else None,
        "lateral_speed_m_s": float(np.mean(columns["root_vy_b"][index])) if "root_vy_b" in columns else None,
    }
    target_key = "commanded" if kind == "hardware" else "target"
    tracking, torque = {}, {}
    for joint in JOINTS:
        pos = columns.get(f"{joint}.pos")
        target = columns.get(f"{joint}.{target_key}")
        if pos is not None and target is not None:
            if kind == "hardware":
                paired = index[index + 1 < len(pos)]
                error = target[paired] - pos[paired + 1]
            else:
                error = target[index] - pos[index]
            tracking[joint] = float(np.degrees(rms(error)))
        tau = columns.get(f"{joint}.torque")
        if tau is not None:
            kind_name = joint[2:]
            values = np.abs(tau[index])
            torque[joint] = {"rms_nm": rms(tau[index]), "max_nm": float(np.max(values)),
                             "above_standstill_rating_frac": float(np.mean(values > STANDSTILL_TORQUE[kind_name])),
                             "above_spinning_rating_frac": float(np.mean(values > SPINNING_TORQUE[kind_name])),
                             "near_peak_frac": float(np.mean(values >= 0.95 * PEAK_TORQUE[kind_name]))}
    out["tracking_rms_deg"] = tracking
    out["torque"] = torque
    if "dt_ms" in columns and kind == "hardware":
        loop = columns["dt_ms"][index]
        out["loop_dt_ms"] = {"median": float(np.median(loop)), "p95": float(np.percentile(loop, 95)),
                             "max": float(np.max(loop))}
    return out


def analyse(path, settle_s, min_ramp):
    columns = load(path)
    kind = source_kind(columns)
    result = {"file": str(path), "source": kind,
              "conventions": {"pitch_deg": "positive = nose down (x toward -z), from projected gravity",
                              "roll_deg": "positive rotation about +x (left side up)",
                              "torque": "type-0x02 feedback estimate" if kind == "hardware" else "simulator actuator torque"},
              "segments": []}
    for cmd, index in segments(columns, settle_s, min_ramp):
        entry = {"command": {"vx": cmd[0], "vy": cmd[1], "wz": cmd[2]}}
        entry.update(segment_metrics(columns, kind, index))
        result["segments"].append(entry)
    return result


def summary_line(result):
    lines = []
    for seg in result["segments"]:
        cmd = seg["command"]
        vx = seg["forward_speed_m_s"]
        track = seg["tracking_rms_deg"]
        worst = max(track, key=track.get) if track else None
        lines.append(
            "%-8s cmd(%.2f,%.2f,%.2f) %5.1fs | vx %s | yaw osc %.3f rad/s @%s Hz, drift %+.2f deg/s | "
            "roll osc %.2f deg @%s Hz, pitch mean %+.2f osc %.2f deg | worst tracking %s %.2f deg" % (
                result["source"], cmd["vx"], cmd["vy"], cmd["wz"], seg["duration_s"],
                "n/a" if vx is None else "%.3f" % vx,
                seg["yaw_rate"]["osc_rms_rad_s"], seg["yaw_rate"]["peak_hz"],
                seg["yaw_rate"]["heading_drift_deg_per_s"],
                seg["roll_deg"]["osc_rms"], seg["roll_deg"]["peak_hz"],
                seg["pitch_deg"]["mean"], seg["pitch_deg"]["osc_rms"],
                worst, track.get(worst, math.nan) if worst else math.nan))
    return lines


def main():
    parser = argparse.ArgumentParser(description="Per-command-segment walking metrics for hardware telemetry and sim traces.")
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--settle", type=float, default=2.0, help="seconds skipped after every command change")
    parser.add_argument("--min-ramp", type=float, default=0.999, help="hardware rows with a lower policy ramp are skipped")
    parser.add_argument("--output", type=Path, help="write all results as one JSON file")
    args = parser.parse_args()
    results = [analyse(path, args.settle, args.min_ramp) for path in args.csv]
    for result in results:
        print(result["file"])
        for line in summary_line(result):
            print("  " + line)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
