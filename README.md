# robonex-deploy

## Structure

```text
robonex-deploy/
├── README.md
├── SNAPSHOT.md
├── check_sync.py
├── requirements.txt
├── assets/
│   ├── meshes/                  37 STL files
│   └── mujoco/
│       ├── robonex.xml          free-base closed-loop model
│       ├── scene.xml
│       ├── robonex_fixed.xml    fixed-base
│       ├── scene_fixed.xml
│       └── full_limit/          kinematic-maximum joint ranges
├── policies/                    empty; drop an exported run here
└── scripts/
    ├── robonex_can.py           CAN primitives copied from Robstride-Motor-Test
    ├── sim_to_sim/
    │   └── play_policy.py       Isaac policy -> MuJoCo
    ├── sim_to_real/
    │   ├── mujoco_to_real.py    MuJoCo -> motors
    │   └── real_to_mujoco.py    motors -> MuJoCo
    └── policy_test/
        ├── print_policy_values.py
        ├── print_policy_action.py
        ├── CMakeLists.txt
        ├── n100_binding.cpp
        └── n100_cpp/
```

No policy ships with this repo. Policies exported before 2026-08-29 are **incompatible** — the action mapping changed from a plain `scale` to `offset + scale` normalization, so an old policy silently produces joint targets offset by up to 27.5 deg.

<br>

## Setup

```bash
# Example
cd ~/humanoid_project/robonex-deploy
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install python-can
```

### `check_sync.py`

`scripts/robonex_can.py` and `assets/mujoco/` are copies. This compares them against the originals in `Robstride-Motor-Test` and `robonex_description`.

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--quiet` | No | Off | Print only drift, not matches |

```bash
# Example
python3 check_sync.py
python3 check_sync.py --quiet
```

<br>

## sim_to_sim

### `scripts/sim_to_sim/play_policy.py`

Evaluates an Isaac-trained ONNX policy in MuJoCo.

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--policy` | **Yes** | - | ONNX policy path |
| `--model` | No | `assets/mujoco/scene.xml` | MuJoCo scene path |
| `--duration` | No | `15.0` headless, unlimited in viewer | Evaluation seconds; `0` is unlimited |
| `--policy-hz` | No | `50.0` | Policy update rate |
| `--spawn` | No | `mujoco` | `mujoco` 1.085 m or `isaac` 1.0789 m |
| `--viewer` | No | Off | Passive MuJoCo viewer |
| `--stop-on-fall` | No | Off | Close viewer at the fall threshold |
| `--real-time` | No | Off | Pace headless mode in real time |
| `--check-only` | No | Off | Load and mapping check without `mj_step` |
| `--output` | No | `-` | JSON output path |
| `--minimum-height` | No | `0.6` | Fall threshold in meters |
| `--max-raw-action` | No | `20.0` | Pre-clip raw action abort threshold |
| `--max-constraint-error` | No | `0.05` | Equality error abort threshold |
| `--max-joint-overrun` | No | `0.05` | Actuated joint overrun abort threshold in rad |
| `--max-body-position` | No | `100.0` | Body-coordinate divergence threshold in meters |

```bash
# Example
.venv/bin/python scripts/sim_to_sim/play_policy.py \
  --policy policies/<run>/policy.onnx \
  --check-only

.venv/bin/python scripts/sim_to_sim/play_policy.py \
  --policy policies/<run>/policy.onnx \
  --spawn mujoco \
  --output results/<run>_mujoco.json

.venv/bin/python scripts/sim_to_sim/play_policy.py \
  --policy policies/<run>/policy.onnx \
  --viewer --spawn mujoco --stop-on-fall --duration 30
```

<br>

## sim_to_real

### `scripts/sim_to_real/mujoco_to_real.py`

Drives the real motors to follow a MuJoCo simulation.

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--hardware` | No | Off | Enable real CAN driving |
| `--motor-id` | No | `1~12` | Motor IDs to drive |
| `--model` | No | `assets/mujoco/scene_fixed.xml` | Fixed-base MJCF scene |
| `--interface` | No | `socketcan` | python-can interface |
| `--host-id` | No | `0xFD` | Host CAN ID |
| `--rate` | No | `100.0` | Command rate (Hz) |
| `--max-speed` | No | `0.10` | Max target speed (rad/s) |
| `--max-accel` | No | `0.25` | Max target acceleration (rad/s²) |
| `--kp` | No | `40.0` | Position gain |
| `--kd` | No | `2.0` | Damping gain |
| `--zero-tolerance-deg` | No | `3.0` | Allowed error to declare zero reached (deg) |
| `--limit-margin-deg` | No | `3.0` | Margin inside the URDF hardstop (deg) |
| `--feedback-timeout` | No | `0.30` | type-`0x02` freshness watchdog (s) |
| `--overspeed` | No | `2.0` | Overspeed stop threshold (rad/s) |
| `--max-error-deg` | No | `25.0` | Tracking-error stop threshold (deg) |
| `--max-temp` | No | `70.0` | Overtemperature stop threshold (°C) |
| `--brake-time` | No | `0.20` | Active braking time before shutdown (s) |
| `--yes` | No | Off | Skip the hardware-start confirmation prompt |
| `--headless` | No | Off | Run without the viewer |
| `--duration` | No | - | Auto-stop after this many seconds |

```bash
# Example
python3 scripts/sim_to_real/mujoco_to_real.py \
  --headless \
  --duration 5 \
  --motor-id 4

python3 scripts/sim_to_real/mujoco_to_real.py \
  --hardware \
  --motor-id 4
```

### `scripts/sim_to_real/real_to_mujoco.py`

Mirrors the real motor positions into a MuJoCo viewer.

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--hardware` | No | Off | Enable real CAN reading |
| `--motor-id` | No | `1~12` | Motor IDs to read |
| `--model` | No | `assets/mujoco/full_limit/scene_fixed_full_limit.xml` | Fixed-base MJCF scene |
| `--interface` | No | `socketcan` | python-can interface |
| `--host-id` | No | `0xFD` | Host CAN ID |
| `--rate` | No | `30.0` | Per-motor `mechPos` refresh rate (Hz) |
| `--read-timeout` | No | `0.03` | Per-`mechPos` response wait (s) |
| `--startup-timeout` | No | `2.0` | First-position collection limit (s) |
| `--stale-timeout` | No | `0.5` | Abort after this long without a fresh position (s) |
| `--limit-tolerance-deg` | No | `1.0` | Encoder error allowed outside the model range (deg) |
| `--yes` | No | Off | Skip the start confirmation prompt |
| `--headless` | No | Off | Run without the viewer; requires `--duration` |
| `--duration` | No | - | Auto-stop after this many seconds |

```bash
# Example
python3 scripts/sim_to_real/real_to_mujoco.py \
  --hardware \
  --motor-id 4

python3 scripts/sim_to_real/real_to_mujoco.py \
  --hardware \
  --headless \
  --duration 10
```

<br>

## policy_test

### Python Binding

```bash
# Example
cd ~/humanoid_project/robonex-deploy/scripts/policy_test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### `scripts/policy_test/print_policy_values.py`

Prints the 42-D observation vector built from live CAN and IMU data.

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--imu-port` | No | `/dev/ttyUSB0` | N100 serial port |
| `--channels` | No | `can0 can1` | CAN channels to read |
| `--interface` | No | `socketcan` | python-can interface |
| `--timeout` | No | `0.02` | Per-motor response timeout (s) |
| `--rate` | No | `10.0` | Screen refresh rate (Hz) |

```bash
# Example
python3 scripts/policy_test/print_policy_values.py --imu-port /dev/ttyUSB0
```

### `scripts/policy_test/print_policy_action.py`

Runs an ONNX policy on the live observation and prints the resulting joint targets. Read-only: it never sends the action over CAN.

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--policy` | **Yes** | - | ONNX policy path |
| `--imu-port` | No | `/dev/ttyUSB0` | N100 serial port |
| `--channels` | No | `can0 can1` | CAN channels to read |
| `--interface` | No | `socketcan` | python-can interface |
| `--timeout` | No | `0.02` | Per-motor response timeout (s) |
| `--rate` | No | `10.0` | Screen refresh rate (Hz) |

```bash
# Example
python3 scripts/policy_test/print_policy_action.py \
  --policy policies/<run>/policy.onnx \
  --imu-port /dev/ttyUSB0
```

<br>

## Policy contract

Observation (42), raw and unnormalized — the ONNX embeds its own normalization:

```text
joint_pos_rel(12) + joint_vel_rel(12) + imu_ang_vel(3) + projected_gravity(3) + last_action(12)
```

Action pipeline, which must mirror Isaac Lab term for term:

```text
clip(raw, ±3.0) -> × scale + offset -> clip(per-joint target limits)
```

`last_action` in the observation is the **runner-clipped** value, not the raw network output.

Motor order (12), Isaac articulation DOF order — not CAN ID order:

```text
0:l_hip_yaw     1:r_hip_yaw     2:l_hip_pitch    3:r_hip_pitch
4:l_hip_roll    5:r_hip_roll    6:l_knee_pitch   7:r_knee_pitch
8:l_ankle_lower 9:l_ankle_upper 10:r_ankle_lower 11:r_ankle_upper
```
