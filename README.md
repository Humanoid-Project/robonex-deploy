# robonex-deploy

## Setup
```bash
# Example
cd ~/humanoid_project
git clone https://github.com/Humanoid-Project/robonex-deploy.git
git clone https://github.com/Humanoid-Project/robonex-description.git
git clone https://github.com/Humanoid-Project/imu-n100-test.git IMU_N100_Test
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
│   │   └── play_policy.py
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

### `play_policy.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--manifest` | Yes | - | `policy_manifest.json` |
| `--description-root` | No | Sibling checkout | `robonex-description` path |
| `--model` | No | Manifest model | MJCF override |
| `--spawn` | No | `mujoco` | `mujoco` or `isaac` |
| `--duration` | No | Viewer: unlimited; headless: `15` | Stop after this many simulation seconds |
| `--viewer` | No | Off | Open the MuJoCo viewer |
| `--stop-on-fall` | No | Off | Stop when height drops below `--minimum-height` |
| `--real-time` | No | Off | Pace the sim to wall clock |
| `--check-only` | No | Off | Validate the manifest and exit |
| `--output` | No | - | Write a log file |
| `--minimum-height` | No | `0.6` | Fall height (m) |
| `--max-raw-action` | No | `20.0` | Raw-action abort threshold |
| `--max-constraint-error` | No | `0.05` | Closure-error abort threshold (m) |
| `--max-joint-overrun` | No | `0.05` | Joint-limit overrun abort (rad) |
| `--max-body-position` | No | `100.0` | Body-position abort (m) |

```bash
# Example
cd ~/humanoid_project/robonex-deploy
source .venv/bin/activate

python3 scripts/sim_to_sim/play_policy.py \
  --manifest policies/<run>/policy_manifest.json \
  --check-only

python3 scripts/sim_to_sim/play_policy.py \
  --manifest policies/<run>/policy_manifest.json \
  --viewer \
  --spawn mujoco
```

<br>

## sim-to-real

### `mujoco_to_real.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--hardware` | No | Off | Enable real CAN motor control |
| `--motor-id`, `--motor-ids` | No | `1`–`12` | Motor IDs to control |
| `--model` | No | `robonex-description/mujoco/basic/scene_fixed.xml` | Fixed-base MJCF |
| `--interface` | No | `socketcan` | python-can interface |
| `--host-id` | No | `0xFD` | Host CAN ID |
| `--rate` | No | `100.0` | Command rate (Hz) |
| `--max-speed` | No | `0.10` | Max target speed (rad/s) |
| `--max-accel` | No | `0.25` | Max target acceleration (rad/s²) |
| `--kp` | No | `40.0` | Position gain |
| `--kd` | No | `2.0` | Velocity gain |
| `--zero-tolerance-deg` | No | `3.0` | Zero-reach band (deg) |
| `--limit-margin-deg` | No | `3.0` | Inner joint-limit margin (deg) |
| `--feedback-timeout` | No | `0.30` | Type `0x02` freshness timeout (s) |
| `--overspeed` | No | `2.0` | Measured-speed stop (rad/s) |
| `--max-error-deg` | No | `25.0` | Tracking-error stop (deg) |
| `--max-temp` | No | `70.0` | Temperature stop (°C) |
| `--brake-time` | No | `0.20` | Damping time before shutdown (s) |
| `--yes` | No | Off | Skip the hardware prompt |
| `--headless` | No | Off | No viewer; requires `--duration` |
| `--duration` | No | - | Stop after this many seconds |

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

<br>

### `process_mujoco_to_real.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--hardware` | No | Off | Enable real CAN motor control |
| `--motor-id`, `--motor-ids` | No | `1`–`12` | Motor IDs to control |
| `--interface` | No | `socketcan` | python-can interface |
| `--host-id` | No | `0xFD` | Host CAN ID |
| `--model` | No | `robonex-description/mujoco/basic/scene_fixed.xml` | Fixed-base MJCF |
| `--rate` | No | `100.0` | Command rate (Hz) |
| `--max-speed` | No | `1.0` | Maximum timed-segment speed (rad/s) |
| `--time-scale` | No | `1.0` | Sequence duration multiplier |
| `--approach-speed` | No | `0.10` | Speed for zero and first-keyframe moves (rad/s) |
| `--approach-accel` | No | `0.25` | Acceleration for approach moves (rad/s²) |
| `--kp` | No | `40.0` | Position gain |
| `--kd` | No | `2.0` | Velocity gain |
| `--zero-tolerance-deg` | No | `3.0` | Zero-reach band (deg) |
| `--limit-margin-deg` | No | `2.0` | Inner joint-limit margin (deg) |
| `--feedback-timeout` | No | `0.30` | Type `0x02` freshness timeout (s) |
| `--overspeed` | No | `2.0` | Measured-speed stop (rad/s) |
| `--max-error-deg` | No | `25.0` | Tracking-error stop (deg) |
| `--max-temp` | No | `70.0` | Temperature stop (°C) |
| `--brake-time` | No | `0.20` | Damping time before shutdown (s) |
| `--yes` | No | Off | Skip the hardware prompt |
| `--headless` | No | Off | Run without a viewer |
| `--dry-run` | No | Off | Validate without opening CAN |

```bash
# Example
python3 scripts/sim_to_real/process_mujoco_to_real.py --dry-run

python3 scripts/sim_to_real/process_mujoco_to_real.py \
  --headless \
  --time-scale 2

python3 scripts/sim_to_real/process_mujoco_to_real.py \
  --hardware \
  --time-scale 2
```

<br>

### `real_to_mujoco.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--hardware` | No | Off | Enable real CAN position reads |
| `--motor-id`, `--motor-ids` | No | `1`–`12` | Motor IDs to read |
| `--model` | No | `robonex-description/mujoco/full_limit/scene_fixed.xml` | Fixed-base MJCF |
| `--interface` | No | `socketcan` | python-can interface |
| `--host-id` | No | `0xFD` | Host CAN ID |
| `--rate` | No | `30.0` | mechPos rate per motor (Hz) |
| `--read-timeout` | No | `0.03` | One mechPos request timeout (s) |
| `--startup-timeout` | No | `2.0` | Initial collection timeout (s) |
| `--stale-timeout` | No | `0.5` | Stale-sample timeout (s) |
| `--limit-tolerance-deg` | No | `1.0` | Encoder band outside model limits (deg) |
| `--yes` | No | Off | Skip the start prompt |
| `--headless` | No | Off | No viewer; requires `--duration` |
| `--duration` | No | - | Stop after this many seconds |

```bash
# Example
python3 scripts/sim_to_real/real_to_mujoco.py --hardware --motor-id 4
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

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--manifest` | Yes | - | `policy_manifest.json` |
| `--imu-port` | No | `/dev/ttyUSB0` | N100 serial port |
| `--channels` | No | `can0 can1` | CAN channels |
| `--interface` | No | `socketcan` | python-can interface |
| `--timeout` | No | `0.02` | One parameter-request timeout (s) |
| `--rate` | No | `10.0` | Display rate (Hz) |

```bash
# Example
python3 scripts/policy_test/print_policy_values.py \
  --manifest policies/<run>/policy_manifest.json \
  --imu-port /dev/ttyUSB0
```

<br>

### `print_policy_action.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--manifest` | Yes | - | `policy_manifest.json` |
| `--imu-port` | No | `/dev/ttyUSB0` | N100 serial port |
| `--channels` | No | `can0 can1` | CAN channels |
| `--interface` | No | `socketcan` | python-can interface |
| `--timeout` | No | `0.02` | One parameter-request timeout (s) |
| `--rate` | No | `10.0` | Display rate (Hz) |

```bash
# Example
python3 scripts/policy_test/print_policy_action.py \
  --manifest policies/<run>/policy_manifest.json \
  --imu-port /dev/ttyUSB0
```

## Parameter inventory

All CLI tables above also support `-h` / `--help`. Values below are source settings or manifest/environment inputs, not additional CLI flags.

### Source limits and timing

| Script | Setting | Current value / meaning |
| --- | --- | --- |
| `mujoco_to_real.py` | `MAX_CONFIG_SPEED` | `0.5` rad/s cap for `--max-speed` |
| `mujoco_to_real.py` | `MAX_CONFIG_ACCEL`, `MAX_CONFIG_RATE` | `2.0` rad/s², `200.0` Hz |
| `mujoco_to_real.py` | `ZERO_SETTLE_TIMEOUT` | `5.0` s zero-settling allowance |
| `process_mujoco_to_real.py` | `MAX_CONFIG_SPEED` | `2.0` rad/s cap for `--max-speed` |
| `process_mujoco_to_real.py` | `MAX_CONFIG_ACCEL`, `MAX_CONFIG_RATE` | `2.0` rad/s², `200.0` Hz |
| `process_mujoco_to_real.py` | `ZERO_SETTLE_TIMEOUT` | `5.0` s zero-settling allowance |
| `process_mujoco_to_real.py` | `MIN_SEGMENT_TIME` | `0.05` s minimum scaled keyframe duration |
| `real_to_mujoco.py` | `MAX_RATE` | `100.0` Hz maximum accepted read rate |
| Both policy-test scripts | `MOUNT_ROLL_DEG` | `180.0` deg IMU correction imported from common |

The two motor-driving scripts require `--overspeed > --max-speed`; the timed player also requires `--approach-speed <= --max-speed`. Gains must fit every selected motor's common encoding range. A margin that empties any joint's allowed interval is rejected. `--duration 0` means unlimited only in `play_policy.py`; the sim-to-real duration options require a finite positive value.

### Policy and path inputs

| Input | Source / current behavior |
| --- | --- |
| Action offsets, scales, target clips | `policy_manifest.json`; 12 values/pairs in `joint_order` |
| `runner_action_clip`, `policy_hz` | Manifest; current exporters write `3.0`, `50.0` Hz |
| `observation_size`, `action_size` | Manifest; current exporters write `42`, `12` |
| `ROBONEX_DESCRIPTION_ROOT` | Description checkout override; `--description-root` takes precedence in `play_policy.py` |
| `ROBONEX_COMMON_ROOT` | Common checkout override still required by `play_policy.py`'s Git-SHA validation, despite tag-based package installation |
| `IMU_N100_TEST_ROOT` | IMU SDK root used by the policy-test CMake build |

### `process_mujoco_to_real.py` source sequence

`SEQUENCE` is a list of `(duration_s, {motor_id: target_rad})` keyframes. Targets below are motor-space radians. `--time-scale` multiplies durations; movement from zero to keyframe 0 uses the approach speed/acceleration instead of its stored duration. Keyframes 1–10 are timed linear interpolation segments. Editing the sequence currently requires editing source.

| Keyframe | Stored seconds | ID1 | ID2 | ID3 | ID4 | ID5 | ID6 | ID7 | ID8 | ID9 | ID10 | ID11 | ID12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.35 | 0 | 0 | 0 | -0.12 | 0.05 | -0.05 | 0 | 0 | 0 | 0.12 | -0.05 | 0.05 |
| 1 | 0.35 | 0 | 0.1646 | 0 | -0.4433 | 0.1088 | -0.1088 | 0 | 0 | 0 | 0.12 | -0.05 | 0.05 |
| 2 | 0.35 | 0 | 0.2663 | 0 | -0.6431 | 0.1451 | -0.1451 | 0 | 0 | 0 | 0.12 | -0.05 | 0.05 |
| 3 | 0.35 | 0 | 0.2663 | 0 | -0.6431 | 0.1451 | -0.1451 | 0 | 0 | 0 | 0.12 | -0.05 | 0.05 |
| 4 | 0.35 | 0 | 0.1646 | 0 | -0.4433 | 0.1088 | -0.1088 | 0 | 0 | 0 | 0.12 | -0.05 | 0.05 |
| 5 | 0.35 | 0 | 0 | 0 | -0.12 | 0.05 | -0.05 | 0 | 0 | 0 | 0.12 | -0.05 | 0.05 |
| 6 | 0.35 | 0 | 0 | 0 | -0.12 | 0.05 | -0.05 | 0 | -0.1646 | 0 | 0.4433 | -0.1088 | 0.1088 |
| 7 | 0.35 | 0 | 0 | 0 | -0.12 | 0.05 | -0.05 | 0 | -0.2663 | 0 | 0.6431 | -0.1451 | 0.1451 |
| 8 | 0.35 | 0 | 0 | 0 | -0.12 | 0.05 | -0.05 | 0 | -0.2663 | 0 | 0.6431 | -0.1451 | 0.1451 |
| 9 | 0.35 | 0 | 0 | 0 | -0.12 | 0.05 | -0.05 | 0 | -0.1646 | 0 | 0.4433 | -0.1088 | 0.1088 |
| 10 | 0.35 | 0 | 0 | 0 | -0.12 | 0.05 | -0.05 | 0 | 0 | 0 | 0.12 | -0.05 | 0.05 |
