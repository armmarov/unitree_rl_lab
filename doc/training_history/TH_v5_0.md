# Training History v5 — Run 5: Returns Clamp Inside Mini-Batch Loop (Failed)

**Robot:** EngineAI PM01 (12-DOF, legs only)
**Task:** `Unitree-PM01-12dof-Velocity`
**Date:** 2026-03-31
**Branch:** `pm01-training`
**Training Plan:** `TP_v5_0.md`

---

## Run Summary

| Field | Value |
|-------|-------|
| Run ID | `2026-03-31_15-49-42` |
| Iterations | 3419 / 50000 |
| Duration | ~1.3 hours |
| Final mean reward | -7.32 |
| Final episode length | 186 |
| Outcome | **Crashed** — `RuntimeError: normal expects all elements of std >= 0.0` |

## Config

- PPO: `PM01PPORunnerCfg` (lr=3e-4, gamma=0.98, vlc=0.5, empirical_normalization=True)
- Terrain: diverse (flat/rough/slopes/stairs)
- Gait weight: 1.0
- Reward rebalancing: joint_vel=-0.005, energy=-5e-5, ankle deviation=-0.5
- Venv patches: `returns_batch.clamp(-100, 100)` in ppo.py

## Key Metrics at Crash

| Metric | Value |
|--------|-------|
| Train/mean_reward | -7.32 |
| Episode_Reward/action_rate | -69,854,312 |
| Episode_Termination/bad_orientation | 98.6% |
| Loss/value_function | — (spikes present) |

## Why It Failed

Returns clamp at ±100 was in the correct location (inside mini-batch loop) but the policy itself diverged. The actor network output extreme actions (action_rate = -69M), causing returns to go extreme before being clamped. The clamp prevented loss overflow but couldn't prevent the actor weights from being corrupted by prior bad gradient updates.

The diverse terrain (99% bad_orientation) created extreme reward variance that overwhelmed all fixes.

## Resolution

Identified that too many changes were applied simultaneously. Needed to step back to Run 2's proven config.
