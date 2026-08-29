# robonex-deploy

RoboNex 정책과 자세를 Isaac, MuJoCo, 실물 사이에서 검증·배포하는 저장소입니다. 로봇 형상은 `robonex_description`, 공통 관절·모터·CAN 계약은 `robonex-common`, N100 SDK는 확정된 `IMU_N100_Test` checkout을 직접 사용합니다.

## Setup

```bash
cd ~/humanoid_project
git clone https://github.com/Humanoid-Project/robonex-common.git
git clone https://github.com/Humanoid-Project/robonex_description.git
git clone https://github.com/Humanoid-Project/IMU_N100_Test.git
git clone https://github.com/Humanoid-Project/robonex-deploy.git
cd robonex-deploy
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

네 저장소를 같은 상위 폴더에 두는 구성을 권장합니다. 다른 배치에서는 다음 환경변수를 사용합니다.

```bash
export ROBONEX_DESCRIPTION_ROOT=/absolute/path/to/robonex_description
export ROBONEX_COMMON_ROOT=/absolute/path/to/robonex-common
export IMU_N100_TEST_ROOT=/absolute/path/to/IMU_N100_Test
```

## Policies

`policies/<run>/`에는 최소한 ONNX 정책과 `policy_manifest.json`을 함께 둡니다. manifest는 `robonex_balancing/scripts/export_policy_manifest.py`로 생성하며 정책 SHA-256, action 순서·offset·scale·clip, 주기, 학습·모델·공통 저장소 커밋을 포함합니다.

## sim-to-sim

```bash
.venv/bin/python scripts/sim_to_sim/play_policy.py \
  --manifest policies/<run>/policy_manifest.json \
  --check-only

.venv/bin/python scripts/sim_to_sim/play_policy.py \
  --manifest policies/<run>/policy_manifest.json \
  --viewer --spawn mujoco --stop-on-fall --duration 30
```

기본 MuJoCo 모델과 정책 주기는 manifest에서 결정됩니다. `robonex_description` checkout의 현재 commit이 manifest와 다르면 실행을 거부합니다. 진단 목적으로 다른 모델을 확인할 때만 `--model`을 명시합니다.

## sim-to-real

기본 모델은 `robonex_description/mujoco`에서 직접 읽습니다.

```bash
python3 scripts/sim_to_real/mujoco_to_real.py --headless --duration 5 --motor-id 4
python3 scripts/sim_to_real/real_to_mujoco.py --hardware --motor-id 4
```

`mujoco_to_real.py`는 `--hardware` 없이는 CAN 구동을 하지 않습니다. 실물 실행은 장착 상태와 비상정지 수단을 확인한 뒤 사용자가 직접 수행합니다.

## policy-test

Python N100 binding은 복사된 SDK가 아니라 `IMU_N100_Test/src/cpp_n100`을 빌드합니다.

```bash
cd scripts/policy_test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

```bash
python3 scripts/policy_test/print_policy_values.py \
  --manifest policies/<run>/policy_manifest.json \
  --imu-port /dev/ttyUSB0

python3 scripts/policy_test/print_policy_action.py \
  --manifest policies/<run>/policy_manifest.json \
  --imu-port /dev/ttyUSB0
```

두 도구는 CAN 파라미터 읽기만 수행하며 정책 action을 모터로 전송하지 않습니다.
