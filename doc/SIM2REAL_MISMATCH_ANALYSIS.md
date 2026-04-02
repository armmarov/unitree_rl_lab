# PM01 Sim-to-Real Mismatch Analysis

**Date:** 2026-04-02
**Analyzed by:** Claude Opus 4.6
**Training source:** `unitree_rl_lab` Run 8 (50000 iters, reward 56.48)
**Deployment target:** `engineai_ros2_workspace/src/interface_example/`

---

## Problem

Robot falls immediately when running the trained policy on the ROS2 deployment (both in MuJoCo sim and hardware).

---

## Root Cause: Multiple Mismatches Between Training and Deployment

The ROS2 deployment code was written for a **different policy** with different observations, scales, gains, and frequency. It does not match our trained Run 8 policy.

---

## Mismatch Summary

| Parameter | Training (IsaacLab) | ROS2 Deployment | Impact |
|-----------|--------------------|-----------------| -------|
| **Policy model** | `policy.onnx` (ONNX, 225 inputs) | `pm01_v2_rough_ppo_42obs.mnn` (MNN, 635 inputs) | **CRITICAL — different model entirely** |
| **Observation size** | 225 (45 x 5 history) | 635 (42 x 15 history + 2 clock + 3 cmd) | **CRITICAL — wrong data fed to policy** |
| **Orientation repr.** | `projected_gravity` (3D gravity vector in body frame) | Euler angles (roll, pitch, yaw) | **CRITICAL — policy interprets orientation incorrectly** |
| **History length** | 5 timesteps | 15 timesteps | **CRITICAL — observation buffer wrong size** |
| **Clock signal** | None | sin/cos phase appended | **CRITICAL — extra inputs policy doesn't expect** |
| **Ang vel scale** | 0.2 | 1.0 | **HIGH — angular velocity 5x too large** |
| **Action scale** | 0.25 | 0.5 | **HIGH — joint movements 2x too large** |
| **Ankle damping (Kd)** | 2.0 | 0.2 | **HIGH — ankle damping 10x too low, oscillation** |
| **Control frequency** | 50 Hz (0.02s) | 100 Hz (0.01s) | **MEDIUM — actions applied 2x too fast** |
| **Transition warmup** | N/A (sim starts standing) | 0.5s ramp-in | **LOW — may be too aggressive** |
| **Gamepad deadband** | N/A | None | **LOW — noisy velocity commands** |

---

## Detailed Analysis

### 1. CRITICAL: Wrong Policy Model

**Training:** Exported `policy.onnx` — ONNX format, expects 225-dim input
**ROS2:** Loads `policies/pm01_v2_rough_ppo_42obs.mnn` — MNN format, expects 635-dim input

These are completely different models from different training runs. The deployed policy was not trained with our Run 8 config.

**Fix:** Convert our `policy.onnx` to MNN format, or update the ROS2 code to use ONNX Runtime instead of MNN.

```bash
# Convert ONNX to MNN (requires MNNConvert tool):
MNNConvert -f ONNX --modelFile policy.onnx --MNNModel policy.mnn
```

### 2. CRITICAL: Wrong Observation Format

**Training observation (45 dims per timestep, 5 history = 225 total):**

| Index | Name | Dims | Scale | Noise |
|-------|------|------|-------|-------|
| 0-2 | `base_ang_vel` | 3 | 0.2 | ±0.2 |
| 3-5 | `projected_gravity` | 3 | 1.0 | ±0.05 |
| 6-8 | `velocity_commands` | 3 | 1.0 | — |
| 9-20 | `joint_pos_rel` | 12 | 1.0 | ±0.01 |
| 21-32 | `joint_vel_rel` | 12 | 0.05 | ±1.5 |
| 33-44 | `last_action` | 12 | 1.0 | — |

**Total per timestep:** 45 dims
**History:** 5 timesteps concatenated = **225 dims**
**Format:** `[obs_t, obs_t-1, obs_t-2, obs_t-3, obs_t-4]`

**ROS2 observation (42 dims per timestep, 15 history + extras = 635 total):**

| Index | Name | Dims | Scale |
|-------|------|------|-------|
| 0-11 | `joint_pos_rel` | 12 | 1.0 |
| 12-23 | `joint_vel` | 12 | 0.05 |
| 24-35 | `last_action` | 12 | 1.0 |
| 36-38 | `base_ang_vel` | 3 | **1.0** (should be 0.2) |
| 39-41 | `euler_angles` | 3 | 1.0 |

**Total per timestep:** 42 dims
**History:** 15 timesteps = 630 dims
**Extras appended:** clock sin/cos (2) + velocity cmd (3) = 5 dims
**Grand total:** **635 dims**

**Mismatches:**
- Order is different (joint data first vs angular vel first)
- Orientation: `projected_gravity` (training) vs `euler_angles` (ROS2) — completely different representation
- Angular velocity scale: 0.2 (training) vs 1.0 (ROS2)
- History: 5 (training) vs 15 (ROS2)
- Clock signal: absent in training, present in ROS2
- Velocity commands: embedded in observation (training) vs appended separately (ROS2)

**Fix:** Rewrite `rl_basic_example.cc` observation construction to match training format exactly. See "Required ROS2 Changes" section below.

### 3. HIGH: Wrong Action Scale

**Training:** `action_scale = 0.25` (all joints)
**ROS2:** `action_scale = 0.5` (all joints)

Policy outputs are scaled 2x too large. A policy output of 1.0 should produce 0.25 rad offset, but ROS2 produces 0.5 rad — double the intended movement.

**Fix:** In `rl_basic_param.yaml`:
```yaml
action_scale:
  - [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
  - [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
```

### 4. HIGH: Wrong Ankle Damping

**Training:** `ankle_kd = 2.0` (both pitch and roll)
**ROS2:** `ankle_kd = 0.2` (10x too low)

Low ankle damping causes uncontrolled oscillation at the feet — robot loses ground contact stability.

**Fix:** In `rl_basic_param.yaml`:
```yaml
joint_kd: [7.0, 5.0, 5.0, 7.0, 2.0, 2.0, 7.0, 5.0, 5.0, 7.0, 2.0, 2.0]
#                                ^^^  ^^^                       ^^^  ^^^
#                          ankle_pitch ankle_roll (was 0.2, should be 2.0)
```

### 5. HIGH: Wrong Angular Velocity Scale

**Training:** `base_ang_vel` scaled by **0.2**
**ROS2:** `observation_scale_angular_vel = 1.0`

Policy receives angular velocity 5x larger than it was trained on. It interprets slow rotations as fast rotations and over-corrects.

**Fix:** In `rl_basic_param.yaml`:
```yaml
observation_scale_angular_vel: 0.2
```

### 6. MEDIUM: Wrong Control Frequency

**Training:** 50 Hz (decimation=4, dt=0.005 → step_dt=0.02)
**ROS2:** 100 Hz (control_dt=0.01)

Policy actions are applied twice per training timestep. Movements accumulate faster than trained.

**Fix:** In `rl_basic_param.yaml`:
```yaml
control_dt: 0.02  # was 0.01
```

### 7. LOW: Transition Time

**Training:** Robot starts standing in simulation (no transition)
**ROS2:** 0.5s linear ramp from initial position to policy control
**MuJoCo sim:** 1.0s warmup with base pinned

The 0.5s transition may be too aggressive. Consider increasing to 1.0-2.0s.

### 8. LOW: Gamepad Deadband

**ROS2:** No deadband on analog sticks — noisy commands at zero velocity
**Fix:** Add deadband of ±0.1 to gamepad inputs

---

## Required ROS2 Changes (Priority Order)

### Priority 1: Use correct policy model

Either:
- Convert `policy.onnx` to MNN: `MNNConvert -f ONNX --modelFile policy.onnx --MNNModel policy.mnn`
- Or switch ROS2 code from MNN to ONNX Runtime

### Priority 2: Fix observation construction

Rewrite observation pipeline in `rl_basic_example.cc` to match training:

```
Per-timestep observation (45 dims):
[0:3]   = imu_angular_velocity * 0.2
[3:6]   = projected_gravity (rotate [0,0,-1] by inverse of IMU quaternion)
[6:9]   = velocity_command (vx, vy, wz)
[9:21]  = joint_pos - default_joint_pos
[21:33] = joint_vel * 0.05
[33:45] = last_action (raw policy output, not scaled)

Full observation (225 dims):
[obs_t | obs_t-1 | obs_t-2 | obs_t-3 | obs_t-4]
```

**Critical: `projected_gravity` calculation:**
```cpp
// NOT euler angles! Rotate gravity vector into body frame:
Eigen::Quaterniond q(imu_w, imu_x, imu_y, imu_z);
Eigen::Vector3d gravity_world(0.0, 0.0, -1.0);
Eigen::Vector3d projected_gravity = q.inverse() * gravity_world;
// Result: [gx, gy, gz] in body frame
```

**Remove:** Clock signal (sin/cos phase) — not used in training
**Remove:** Appended velocity commands — already in observation at index 6-8

### Priority 3: Fix action scale

```yaml
action_scale: 0.25  # was 0.5
```

### Priority 4: Fix ankle damping

```yaml
ankle_kd: 2.0  # was 0.2
```

### Priority 5: Fix angular velocity scale

```yaml
observation_scale_angular_vel: 0.2  # was 1.0
```

### Priority 6: Fix control frequency

```yaml
control_dt: 0.02  # was 0.01
```

---

## Verification Checklist

After applying fixes, verify:

- [ ] Policy model loaded is our Run 8 export (225 input dims, 12 output dims)
- [ ] Observation vector is exactly 225 elements
- [ ] Observation order: ang_vel, projected_gravity, vel_cmd, joint_pos_rel, joint_vel_rel, last_action
- [ ] `projected_gravity` computed as `q.inverse() * [0,0,-1]`, NOT euler angles
- [ ] Angular velocity scaled by 0.2
- [ ] Joint velocity scaled by 0.05
- [ ] History buffer is 5 timesteps (not 15)
- [ ] No clock signal appended
- [ ] Action scale is 0.25 (not 0.5)
- [ ] Ankle Kd is 2.0 (not 0.2)
- [ ] Control loop runs at 50 Hz (not 100 Hz)
- [ ] Default joint positions match: [-0.24, 0, 0, 0.48, -0.24, 0, -0.24, 0, 0, 0.48, -0.24, 0]

---

## Reference: Training Config Sources

| Parameter | File | Line |
|-----------|------|------|
| Observation terms | `velocity_env_cfg.py` | 206-216 |
| Action scale | `velocity_env_cfg.py` | 213-218 |
| PD gains | `pm01.py` | 59-101 |
| Default joint pos | `pm01.py` | 42-54 |
| Control frequency | `velocity_env_cfg.py` | 364-365 (decimation=4, dt=0.005) |
| History length | `velocity_env_cfg.py` | 214-215 |
| ONNX model | `deploy/robots/pm01_12dof/config/policy/velocity/v0/exported/policy.onnx` | — |
