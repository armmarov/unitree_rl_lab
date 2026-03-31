# Training History v1 — Run 1: Initial PM01 Training

**Robot:** EngineAI PM01 (12-DOF, legs only)
**Task:** `Unitree-PM01-12dof-Velocity`
**Date:** 2026-03-30
**Branch:** `pm01-training`

---

## Run Summary

| Field | Value |
|-------|-------|
| Run ID | `2026-03-30_20-47-12` |
| Iterations | ~1510 / 50000 |
| Duration | ~20 min (stopped manually) |
| Final mean reward | 12.76 |
| Final episode length | ~460 |
| Outcome | **Stopped manually** |

## Config

- PPO: `BasePPORunnerCfg` (shared, lr=1e-3, gamma=0.99)
- Terrain: 100% flat (`MeshPlaneTerrainCfg` proportion=0.5, nothing else)
- Commands: `lin_vel_x=(-0.1, 0.1)`, `lin_vel_y=(0.0, 0.0)`, `ang_vel_z=(0.0, 0.0)`
- Gait weight: 0.5

## Key Metrics at Stop

| Metric | Value |
|--------|-------|
| Train/mean_reward | 12.76 |
| Train/mean_episode_length | ~460 |
| Episode_Reward/track_lin_vel_xy | 0.41 |
| Episode_Reward/gait | **0.00** |
| Episode_Reward/feet_clearance | 0.41 |
| Loss/value_function | 0.068 |

## Issue Found

**Gait reward stuck at 0.0.** Root cause: the `feet_gait` reward function gates output to zero when `cmd_norm <= 0.1`. With only `lin_vel_x` varying in `(-0.1, 0.1)` and y/yaw at zero, the command norm was almost always below the 0.1 threshold.

## Resolution

Added `lin_vel_y=(-0.1, 0.1)` and `ang_vel_z=(-0.1, 0.1)` to initial command ranges so the 3-axis norm can exceed the 0.1 gating threshold. Applied in Run 2.

## Lessons Learned

1. **Always check reward gating thresholds.** The `feet_gait` function gates reward to zero when command norm is below 0.1. Initial command ranges must produce norms that exceed this threshold.
