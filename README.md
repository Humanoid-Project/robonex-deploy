# robonex-deploy

## Setup
```bash
# Example
cd ~/humanoid_project
git clone https://github.com/Humanoid-Project/robonex-deploy.git
git clone https://github.com/Humanoid-Project/robonex-description.git
git clone https://github.com/Humanoid-Project/imu-n100-test.git
cd robonex-deploy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Robot identity: the model physically attached (ver2_edu, ver2_pro or ver2_max); every motor-enabling script checks it
mkdir -p ~/.config/robonex && echo ver2_edu > ~/.config/robonex/robot_model
```

`robonex-common` is pinned in `requirements.txt` — see [`robonex-common/setup/SETUP.md`](https://github.com/Humanoid-Project/robonex-common/blob/main/setup/SETUP.md).

<br>

## Structure

```text
robonex-deploy/
├── README.md
├── policies/
├── scripts/
│   ├── robonex_can.py
│   ├── analysis/
│   │   ├── scenario_metrics.py
│   │   └── timing_report.py
│   ├── sysid/
│   │   ├── joint_probe.py
│   │   └── analyze_probe.py
│   ├── sim_to_sim/
│   │   └── isaac_to_mujoco.py
│   ├── sim_to_real/
│   │   ├── bench.py
│   │   ├── safety.py
│   │   ├── mujoco_to_real.py
│   │   └── real_to_mujoco.py
│   └── policy_test/
│       ├── CMakeLists.txt
│       ├── n100_binding.cpp
│       ├── print_policy_values.py
│       └── print_policy_action.py
└── requirements.txt
```

<br>

## sim-to-sim

### `isaac_to_mujoco.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--policy` | `Required` | ONNX policy path |
| - | `--output` | Terminal only | Optional JSON output path |
| - | `--duration` | Until the viewer closes | Stop after this many simulated seconds |
| - | `--headless` | Viewer on | Run without a viewer; requires `--duration` |
| - | `--vx` / `--vy` / `--wz` | `0.3` / `0.0` / `0.0` | Constant velocity command (m/s, m/s, rad/s) |
| - | `--scenario` | - | Command schedule `T:VX,VY,WZ;...` in seconds; replaces `--vx/--vy/--wz` |
| - | `--trace` | - | Per-policy-step CSV trace for `scenario_metrics.py` |
| - | `--slew-limit` | Off | Pass targets through the deploy slew limiter (6 rad/s, 120 rad/s²) |
| - | `--heading-hold` | Off | PI heading hold on the policy's yaw-rate command, as `policy_to_real.py --heading-hold` |
| - | `--heading-kp` / `--heading-ki` | `1.0` / `0.1` | Heading-hold gains |
| - | `--heading-source` | `gyro` | Integrated base gyro (`gyro`) or base orientation yaw (`quat`) |
| - | `--heading-start-s` | `1.5` | Simulated time before the heading hold may engage |
| - | `--joint-friction` | Model value | Coulomb friction on the 12 motor joints by model, RS02 then RS03 (N·m) |
| - | `--match-isaac` | Off | Robot-robot collisions off and passive-joint damping 0, as in the Isaac model |
| - | `--hip-yaw-kp` | Model value | Hip-yaw position gain override (diagnostic) |
| - | `--hip-yaw-backlash` | - | Hip-yaw free play in rad, no torque inside it (diagnostic) |

```bash
# Example
cd ~/humanoid_project/robonex-deploy
source .venv/bin/activate

python3 scripts/sim_to_sim/isaac_to_mujoco.py \
  --policy policies/<run>/policy.onnx

# Keyboard walk with the deploy heading hold and measured joint friction
python3 scripts/sim_to_sim/isaac_to_mujoco.py \
  --policy policies/<run>/policy.onnx \
  --vx 0.0 \
  --slew-limit \
  --joint-friction 0.14 0.47 \
  --heading-hold

# Reproducible check without a viewer
python3 scripts/sim_to_sim/isaac_to_mujoco.py \
  --policy policies/<run>/policy.onnx \
  --headless \
  --duration 15 \
  --output /tmp/sim_to_sim.json

# Fixed command scenario with a trace
python3 scripts/sim_to_sim/isaac_to_mujoco.py \
  --policy policies/<run>/policy.onnx \
  --headless \
  --duration 30 \
  --scenario "0:0,0,0;10:0.1,0,0;20:0.2,0,0" \
  --trace /tmp/scenario_trace.csv
```

The required schema-2 manifest verifies the policy file, MuJoCo XML/mesh bundle, action
contract, and `robonex-common` runtime source before simulation starts.

<br>

## analysis

### `scenario_metrics.py`

Per-command-segment metrics for hardware telemetry and Isaac/MuJoCo traces: speed (sim only), heading drift, yaw-rate oscillation (gait band and above 3 Hz), roll/pitch, joint tracking, torque.

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `csv` | `Required` | One or more `*_live_telemetry.csv` or sim trace CSVs |
| - | `--settle` | `2.0` | Seconds skipped after every command change |
| - | `--min-ramp` | `0.999` | Hardware rows with a lower policy ramp are skipped |
| - | `--output` | - | JSON output path |

```bash
# Example
python3 scripts/analysis/scenario_metrics.py \
  results/policy_to_real/<stamp>_live_telemetry.csv \
  --output /tmp/metrics.json
```

<br>

### `timing_report.py`

IMU host age (mean, p50, p95, max), host age per window and its least-squares "host age slope" in ms/min (a trend fit over the raw ages, not an unwrapped phase drift), device-vs-host clock rate, seq gaps, tick period jitter, and per-joint feedback age from a `policy_to_real.py --telemetry` CSV; stop rows are skipped, and a file with only stop rows or missing timing columns exits 1. Timing columns only; they do not show that the loop's wall-clock timing matches a run without `--telemetry`.

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `csv` | `Required` | One or more `*_live_telemetry.csv` or `*_read_telemetry.csv` |
| - | `--d0-ms` | Unknown | Fixed IMU delay before the host stamp (ms, finite, >= 0); unknown prints the bound as a function of d0 |
| - | `--budget-ms` | `15.0` | Trained IMU age range upper end (ms, finite, > 0); the check is host age max <= budget - d0 |
| - | `--window` | `5.0` | Host age window (s, finite, > 0) |
| - | `--output` | - | JSON output path |

```bash
# Example
python3 scripts/analysis/timing_report.py \
  results/policy_to_real/<stamp>_read_telemetry.csv

python3 scripts/analysis/timing_report.py \
  results/policy_to_real/<stamp>_live_telemetry.csv \
  --d0-ms 3.5 \
  --output /tmp/timing.json
```

<br>

## sysid

### `joint_probe.py`

Drives ONE motor of a hung robot around its current position and logs feedback at the command rate; refuses a PD torque demand (kp × amplitude) above the motor's continuous rating.

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--motor-id` | `Required` | The single motor to move; must be on `--robot-model` |
| - | `--profile` | `Required` | `step` (latency), `triangle` (hysteresis), `chirp` (frequency response) |
| - | `--robot-model` | `Required` | `ver2_edu`, `ver2_pro` or `ver2_max`; must match the robot identity |
| - | `--amplitude` | `0.05` | rad, at most 0.15 |
| - | `--duration` | `20.0` | s, at most 120 |
| - | `--hold` | `1.0` | step: seconds per level |
| - | `--period` | `10.0` | triangle: seconds per cycle |
| - | `--f0` / `--f1` | `0.2` / `5.0` | chirp: start / end frequency (Hz) |
| - | `--rate` | `200.0` | Command and feedback rate (Hz), 50–250 |
| - | `--gain-scale` | `1.0` | Fraction of the walking gains |
| - | `--output` | `results/sysid/<stamp>_id<ID>_<profile>.csv` | CSV path (never overwritten) |

```bash
# Example
python3 scripts/sysid/joint_probe.py --motor-id 1 --robot-model ver2_edu --profile step --amplitude 0.05
python3 scripts/sysid/joint_probe.py --motor-id 1 --robot-model ver2_edu --profile triangle --amplitude 0.05 --period 10 --duration 30
python3 scripts/sysid/joint_probe.py --motor-id 1 --robot-model ver2_edu --profile chirp --amplitude 0.03 --f0 0.2 --f1 5 --duration 40
python3 scripts/sysid/joint_probe.py --motor-id 13 --robot-model ver2_edu --profile step --amplitude 0.05
```

<br>

### `analyze_probe.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `csv` | `Required` | One or more `joint_probe.py` recordings |
| - | `--threshold-deg` | `0.2` | Step: motion-onset threshold |
| - | `--output` | - | JSON output path |

```bash
# Example
python3 scripts/sysid/analyze_probe.py results/sysid/*_id1_*.csv --output /tmp/probe.json
```

<br>

## sim-to-real

The variant comes from `~/.config/robonex/robot_model`; CAN channels from `~/.config/robonex/bus_map.json` (default below).

| Group | IDs | Motor | Default CAN | edu | pro | max |
| --- | --- | --- | --- | --- | --- | --- |
| `left_leg` | `1`–`6` | rs02 / rs03 | `can0` | ✓ | ✓ | ✓ |
| `right_leg` | `7`–`12` | rs02 / rs03 | `can1` | ✓ | ✓ | ✓ |
| `head` | `13` neck_pitch (`14` reserved for neck yaw, no motor yet) | rs05 | `can4` | `13` | `13` | `13` |
| `left_arm` | `15`–`18` shoulder_pitch/roll/yaw, elbow | rs02 | `can2` | - | ✓ | ✓ |
| `right_arm` | `20`–`23` shoulder_pitch/roll/yaw, elbow | rs02 | `can3` | - | ✓ | ✓ |

Head limit ±73° and shoulder pitch (15, 20) ±100° (user, 2026-10-01). PLACEHOLDER until measured: head kp 20 / kd 1, other arm limits ±45°, arm kp 40 / kd 2. Motors without a MuJoCo actuator (head `13` on edu and pro) show `not in sim`.

### `mujoco_to_real.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--motor-id` | `legs` (`1`–`12`) | IDs, groups (`left_leg`, `right_leg`, `head`, `left_arm`, `right_arm`, `legs`, `arms`, `all`) or joint names (`left_knee_pitch`); `not in sim` motors are moved to zero and held |
| - | `--robot` | Identity file | `edu`, `pro`, `max` (or `ver2_*`); must match the identity file |
| - | `--dry-run` | Off | Print variant, buses and the per-motor kp/kd/limit table, then exit; no CAN bus is opened |

```bash
# Example
cd ~/humanoid_project/robonex-deploy
source .venv/bin/activate

python3 scripts/sim_to_real/mujoco_to_real.py --dry-run
python3 scripts/sim_to_real/mujoco_to_real.py --dry-run --motor-id all

python3 scripts/sim_to_real/mujoco_to_real.py
python3 scripts/sim_to_real/mujoco_to_real.py --motor-id left_leg
python3 scripts/sim_to_real/mujoco_to_real.py --motor-id 4
python3 scripts/sim_to_real/mujoco_to_real.py --motor-id head

# pro / max identity file
python3 scripts/sim_to_real/mujoco_to_real.py --motor-id legs arms
python3 scripts/sim_to_real/mujoco_to_real.py --robot max --motor-id all
```

<br>

### `real_to_mujoco.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--motor-id` | `legs` (`1`–`12`) | Same selection as `mujoco_to_real.py`; `not in sim` motors are read and printed only |
| - | `--robot` | Identity file | `edu`, `pro`, `max` (or `ver2_*`); must match the identity file, sets the variant when there is none |
| - | `--once` | Off | Read once, print one table, exit; no viewer, no stop frame |

```bash
# Example
python3 scripts/sim_to_real/real_to_mujoco.py
python3 scripts/sim_to_real/real_to_mujoco.py --motor-id 4
python3 scripts/sim_to_real/real_to_mujoco.py --motor-id head

python3 scripts/sim_to_real/real_to_mujoco.py --once
python3 scripts/sim_to_real/real_to_mujoco.py --once --motor-id all

# Without an identity file
python3 scripts/sim_to_real/real_to_mujoco.py --robot edu
python3 scripts/sim_to_real/real_to_mujoco.py --robot pro --motor-id arms
python3 scripts/sim_to_real/real_to_mujoco.py --robot max --once --motor-id head
```

<br>

## policy-test

### `policy_to_real.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--policy` | `Required` | ONNX policy path (manifest next to it) |
| - | `--read` | Off | Read-only preview; no motor is commanded |
| - | `--duration` | Until Ctrl-C | Stop after this many seconds |
| - | `--vx` / `--vy` / `--wz` | `0.0` | Constant velocity command inside the trained envelope |
| - | `--keyboard` | Off | Steer the command with w/s, q/e, a/d, SPACE |
| - | `--heading-hold` | Off | PI heading hold on the yaw-rate command the policy sees (±0.2 rad/s); engages with wz 0 and vx/vy above the gait deadband |
| - | `--heading-kp` / `--heading-ki` | `1.0` / `0.1` | Heading-hold gains (kp 0–2, ki 0–0.5; integral ±0.1 rad/s) |
| - | `--heading-source` | `gyro` | Held heading: integrated bias-calibrated gyro (`gyro`) or N100 AHRS yaw (`quat`); both are recorded |
| - | `--telemetry` | Off | Per-step CSV plus `_arrays.npz` and, live only, `_phases.csv` and `_meta.json`; no path = timestamped file under `results/policy_to_real`; live mode refuses before motor enable when any of these files exists or the IMU sample lacks `seq`, `device_timestamp_us` or `host_timestamp_ns`, and a timing-column failure mid-run leaves those cells blank and warns at exit |
| - | `--log` | Off | Save terminal output; no path = timestamped file |
| - | `--gain-scale` | `1.0` | Fraction of the trained per-joint gains |
| - | `--ankle-gain-scale` | `1.0` | Extra factor 1–2 on the four ankle motors' kp and kd (diagnostic; trained at 1.0) |
| - | `--max-tilt-deg` | `40` | Trunk tilt stop |
| - | `--approach-tolerance-deg` | `1.0` | Default-pose tolerance before policy control |

```bash
# Example
python3 scripts/policy_test/policy_to_real.py \
  --policy policies/<run>/policy.onnx \
  --duration 60 \
  --keyboard \
  --approach-tolerance-deg 5 \
  --telemetry \
  --log

# Straight walk with heading hold
python3 scripts/policy_test/policy_to_real.py \
  --policy policies/<run>/policy.onnx \
  --duration 30 \
  --vx 0.1 \
  --keyboard \
  --heading-hold \
  --approach-tolerance-deg 5 \
  --telemetry \
  --log

# Read-only timing capture, no motor commanded
python3 scripts/policy_test/policy_to_real.py \
  --policy policies/<run>/policy.onnx \
  --read \
  --duration 60 \
  --telemetry
```

| Output | Description |
| --- | --- |
| `heading_*`, `policy_wz` | Heading hold: gyro and AHRS heading, target, error (deg), integral and controller correction (rad/s, 0 when released), engaged, and the yaw-rate command the policy saw; blank without `--heading-hold` |
| `imu_age_ms` | Time since the loop last saw a new AHRS seq or raw-IMU frame count (the stale-stop input); about 0 on a healthy stream |
| `imu_host_age_ms` | Monotonic time right after `imu.read` minus `sample.host_timestamp_ns`: AHRS publish to policy read age, not raw-gyro age; excludes the device/wire delay d0 and the raw-gyro hold |
| `imu_device_dt_ms` | Difference of consecutive ticks' `device_timestamp_us` (AHRS device clock) |
| `imu_host_dt_ms` | Difference of consecutive ticks' `host_timestamp_ns` |
| `imu_seq_gap` | Seq difference minus 1 between consecutive ticks; seq counts host-accepted AHRS samples, not wire packets, so it is no packet-loss count; `1` is normal (100 Hz publish, 50 Hz read), `-1` = the same sample read twice |
| `tick_period_ms` | Unclamped period between loop ticks (`dt_ms` is the clamped slew dt) |
| `<joint>.age_ms` | Live only: time since `FeedbackHub.pump` drained the joint's last type 0x02 frame, taken when the row is written (after send); drain-time age, one stamp per pump batch, not acquisition age |
| `<joint>.rx_age_ms` | Live only: wall time when the row is written minus the socketcan kernel receive timestamp of that frame |
| `late_ms`, `poll_ms`, `prev_work_ms` | Live only: tick start minus its schedule, feedback drain time, previous tick's work time before sleep |
| `rx_frames.<channel>` | Live only: CAN frames of any type drained from that bus this tick |
| `acc_*`, `quat_*`, `imu_temp_c` | AHRS linear acceleration, orientation and IMU temperature of the sample read this tick |
| `<joint>.mode`, `.action`, `.slew_vel` | Live only: motor mode status, runner-clipped action fed back to the policy, slew-limiter velocity |
| `*_arrays.npz` | Per-step `obs` (235), `raw_action`, `policy_action`, `targets`, `pos`, `vel`, `torque`, `gyro`, `gravity`, `velocity_command` (float32), `commanded`, `t_s`, `wall_time` (float64), `step`; first 60000 steps; written after the brake. Read mode: `obs`, `pos`, `vel`, `raw_action`, `targets`, `gyro`, `gravity` |
| `*_phases.csv` | Live only: per-joint feedback at enable, every approach tick and every brake cycle (100 Hz) |
| `*_meta.json` | Policy identity, settings, saved motor parameters (`limit_torque`, `limit_cur`, `vbus`, `run_mode`, `zero_sta`), git state, CAN and IMU counters; `end` holds the exit reason, approach errors, shutdown report and run statistics |

<br>

```bash
# Example
cd ~/humanoid_project/robonex-deploy/scripts/policy_test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### `print_policy_values.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--manifest` | `Required` | Policy manifest path |

```bash
# Example
python3 scripts/policy_test/print_policy_values.py \
  --manifest policies/<run>/policy_manifest.json
```

<br>

### `print_policy_action.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--policy` | `Required` | ONNX policy path |

```bash
# Example
python3 scripts/policy_test/print_policy_action.py \
  --policy policies/<run>/policy.onnx
```
