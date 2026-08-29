# Snapshot

`scripts/robonex_can.py` and everything under `assets/` is a **copy**. `check_sync.py` compares
them against the originals; run it after any change to `Robstride-Motor-Test` or
`robonex_description`.

## Copied code

| Local file | Original |
|---|---|
| `scripts/robonex_can.py` | `Robstride-Motor-Test/scripts/motor_control/motor_test/set_motor_pose.py (lines 10-213) + scripts/measurements/common.py (JOINT_LIMITS_RAD, joint_limit_for, exceeds_joint_limit)` |

## Moved scripts

These were moved here on 2026-08-29 and deleted from their old homes.

| Local file | Former location |
|---|---|
| `scripts/sim_to_sim/play_policy.py` | `sim-to-sim/play_policy.py` |
| `scripts/sim_to_real/mujoco_to_real.py` | `Robstride-Motor-Test/scripts/motor_control/mujoco_to_real/mujoco_to_real.py` |
| `scripts/sim_to_real/real_to_mujoco.py` | `Robstride-Motor-Test/scripts/motor_control/mujoco_to_real/real_to_mujoco.py` |
| `scripts/policy_test/print_policy_values.py` | `Robstride-Motor-Test/scripts/motor_control/policy_test/print_policy_values.py` |
| `scripts/policy_test/print_policy_action.py` | `Robstride-Motor-Test/scripts/motor_control/policy_test/print_policy_action.py` |

## Copied assets

| Local file | Original |
|---|---|
| `assets/mujoco/robonex.xml` | `robonex_description/mujoco/robonex.xml` |
| `assets/mujoco/scene.xml` | `robonex_description/mujoco/scene.xml` |
| `assets/mujoco/robonex_fixed.xml` | `robonex_description/mujoco/robonex_fixed.xml` |
| `assets/mujoco/scene_fixed.xml` | `robonex_description/mujoco/scene_fixed.xml` |
| `assets/mujoco/full_limit/robonex_full_limit.xml` | `robonex_description/mujoco/full_limit/robonex_full_limit.xml` |
| `assets/mujoco/full_limit/robonex_fixed_full_limit.xml` | `robonex_description/mujoco/full_limit/robonex_fixed_full_limit.xml` |
| `assets/mujoco/full_limit/scene_full_limit.xml` | `robonex_description/mujoco/full_limit/scene_full_limit.xml` |
| `assets/mujoco/full_limit/scene_fixed_full_limit.xml` | `robonex_description/mujoco/full_limit/scene_fixed_full_limit.xml` |
| `assets/meshes/*.stl` | `robonex_description/meshes/*.stl` referenced by `robonex.xml` |

## SHA-256

```text
4c898e1fd82093c4869827f9156c78b71ca9c15f96a8abc318f8f263c786cebe  assets/mujoco/robonex.xml
8e672ead14ffdc89539c973e66d5bf385555ce67c23faf84ec3c8f7e18855b83  assets/mujoco/scene.xml
ef88fed7f7410e0a2731c9da0df5ad97c9061dc7b2ca88e753ef598cd4da5f7e  assets/mujoco/robonex_fixed.xml
4671caed23a2d731619c1a0f099595879c199c6cf96f01116ec4f8bcfbdcf8ee  assets/mujoco/scene_fixed.xml
ce8339e90dfe4df119bbd38f1ccf9a49121642fd4543f22c47bae04c00fca376  assets/mujoco/full_limit/robonex_full_limit.xml
259b4843a2bc2b5271b9537c8f3cb475843ae897c18d0fc761fffd4264889e2e  assets/mujoco/full_limit/robonex_fixed_full_limit.xml
8a318e423220d1cb9fcefd291262cd48a1ccc8f59ac3aa1c9fc5ad6ee6876e8e  assets/mujoco/full_limit/scene_full_limit.xml
52f5bcdcc4466bb82f0e553c3e366d6ae9ad2d3aa1e21d94a383ee3e790c6a29  assets/mujoco/full_limit/scene_fixed_full_limit.xml
badef92837391ea35e74cf9c9827b3cd4233eeba835c1f2b274d8fb87d18f7f0  scripts/robonex_can.py
```

Mesh count: 37.
