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
│   │   └── scenario_metrics.py
│   ├── sim_to_sim/
│   │   └── isaac_to_mujoco.py
│   ├── sim_to_real/
│   │   ├── safety.py
│   │   ├── mujoco_to_real.py
│   │   ├── process_mujoco_to_real.py
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
| - | `--match-isaac` | Off | Robot-robot collisions off and passive-joint damping 0, as in the Isaac model |
| - | `--hip-yaw-kp` | Model value | Hip-yaw position gain override (diagnostic) |
| - | `--hip-yaw-backlash` | - | Hip-yaw free play in rad, no torque inside it (diagnostic) |

```bash
# Example
cd ~/humanoid_project/robonex-deploy
source .venv/bin/activate

python3 scripts/sim_to_sim/isaac_to_mujoco.py \
  --policy policies/<run>/policy.onnx

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

## sim-to-real

### `mujoco_to_real.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--motor-id` | `1`–`12` | One or more motor IDs to control |

```bash
# Example
python3 scripts/sim_to_real/mujoco_to_real.py \
  --motor-id 4
```

<br>

### `process_mujoco_to_real.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--motor-id` | `1`–`12` | One or more motor IDs to control |
| - | `--time-scale` | `1.0` | Sequence duration multiplier |
| - | `--dry-run` | Off | Validate without CAN |

```bash
# Example
python3 scripts/sim_to_real/process_mujoco_to_real.py --dry-run

python3 scripts/sim_to_real/process_mujoco_to_real.py \
  --time-scale 2
```

<br>

### `real_to_mujoco.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--motor-id` | `1`–`12` | One or more motor IDs to read |

```bash
# Example
python3 scripts/sim_to_real/real_to_mujoco.py --motor-id 4
```

<br>

## policy-test

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
