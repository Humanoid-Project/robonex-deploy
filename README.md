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
```

The required schema-2 manifest verifies the policy file, MuJoCo XML/mesh bundle, action
contract, and `robonex-common` runtime source before simulation starts.

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
