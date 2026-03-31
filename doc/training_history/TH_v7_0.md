# Training History v7 — Run 7: Clean Log Std, No Venv Patches (Longest Run)

**Robot:** EngineAI PM01 (12-DOF, legs only)
**Task:** `Unitree-PM01-12dof-Velocity`
**Date:** 2026-03-31
**Branch:** `pm01-training`
**Training Plan:** `TP_v7_0.md`

---

## Run Summary

| Field | Value |
|-------|-------|
| Run ID | `2026-03-31_20-15-30` |
| Iterations | 17634 / 50000 |
| Duration | ~6.3 hours |
| Final mean reward | -6.1e15 (diverged at crash) |
| Final episode length | 986 |
| Outcome | **Crashed** — `RuntimeError: normal expects all elements of std >= 0.0` |

## Config

- PPO: `PM01PPORunnerCfg` — only override: `noise_std_type="log"`
- All other PPO params: BasePPORunnerCfg defaults (lr=1e-3, gamma=0.99, vlc=1.0)
- Terrain: 100% flat (Run 2 config)
- Rewards: original weights (gait=0.5, joint_vel=-0.001, energy=-2e-5)
- No venv patches — clean rsl_rl install

## Training Progression (Before Crash)

Training was healthy and matched Run 2's behavior until the crash:

| Metric | Run 7 (pre-crash) | Run 2 (pre-crash) |
|--------|-------------------|-------------------|
| Mean reward | ~51 | ~51 |
| Episode length | ~986-1000 | ~1000 |
| Gait | 0.62 | 0.69 |
| Terrain level | 5.31 | 5.36 |
| Bad orientation | 7.5% | 0.3% |

## Key Metrics at Crash (iter 17634)

| Metric | Value |
|--------|-------|
| Train/mean_reward | -6.1e15 |
| Episode_Reward/action_rate | -2.05e14 |
| Episode_Termination/bad_orientation | 7.5% |
| Loss/value_function | 2.04e31 |

## What Worked

- **`noise_std_type="log"` extended survival by 4000+ iterations** (17634 vs Run 2's 13258)
- Training was completely healthy for 17000+ iterations
- Matched Run 2's reward/gait/terrain metrics
- No venv patches needed for the log std fix

## What Still Failed

- **Value loss still spiked** at ~iter 9500 (first spike) and iter 17634 (fatal)
- The log std prevents std from going negative, but cannot prevent the actor network weights from diverging when value loss explodes
- Actor output extreme actions → `action_rate: -2.05e14` → crash

## Root Cause Confirmed

PM01's returns are ~2x G1's (per-step reward 2.24 vs 1.08). With `gamma=0.99`, outlier returns reach magnitudes where the squared value loss `(return - prediction)^2` cascades into optimizer corruption.

**G1 value loss:** max 0.43 across 5916 iters (completely stable)
**PM01 value loss:** spikes to 1e13+ then 1e31 (catastrophic)

## Resolution

Need `gamma=0.95` to cap return magnitude to ~20 (matching G1's range). This addresses the root cause — returns too large for fp32 squared loss — rather than treating symptoms.

See `TP_v8_0.md`.
