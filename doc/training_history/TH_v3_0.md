# Training History v3 — Run 3: PPO Stability + Terrain Diversity + Gait Weight

**Robot:** EngineAI PM01 (12-DOF, legs only)
**Task:** `Unitree-PM01-12dof-Velocity`
**Date:** 2026-03-31
**Branch:** `pm01-training`
**Training Plan:** `TP_v3_0.md`

---

## Run Summary

| Field | Value |
|-------|-------|
| Run ID | `2026-03-31_08-36-01` |
| Iterations | 7051 / 50000 |
| Duration | ~2.8 hours |
| Final mean reward | 33.28 |
| Final episode length | 787 |
| Outcome | **Crashed** — `RuntimeError: normal expects all elements of std >= 0.0` |

## Config Changes from Run 2

| Parameter | Run 2 | Run 3 |
|-----------|-------|-------|
| PPO config | `BasePPORunnerCfg` | `PM01PPORunnerCfg` |
| learning_rate | 1e-3 | 5e-4 |
| gamma | 0.99 | 0.98 |
| value_loss_coef | 1.0 | 0.5 |
| empirical_normalization | False | True |
| Terrain | 100% flat | 20% flat, 20% rough, 20% slope up, 20% slope down, 20% stairs |
| Gait weight | 0.5 | 1.0 |

## Training Progression

| Iter | Mean Reward | Episode Length | Gait | Terrain Level | Value Loss |
|------|-------------|---------------|------|---------------|------------|
| 0 | — | — | 0.004 | 4.17 | — |
| ~3500 | ~33 | ~760 | ~0.70 | ~0.5 | ~0.03 |
| 7042 | 34.27 | 762 | ~0.80 | ~0.02 | 0.032 |
| 7044 | -6.35e14 | 761 | — | — | 2.85e29 (spike) |
| 7045 | 35.05 | 782 | — | — | 0.309 (recovered) |
| 7048 | 34.17 | 779 | — | — | 1.843 (unstable) |
| 7049 | 34.45 | 781 | — | — | 39.53 (escalating) |
| 7051 | 33.28 | 788 | 0.80 | 0.017 | 162.56 (fatal) |

## Key Metrics at Crash

| Metric | Value |
|--------|-------|
| Train/mean_reward | 33.28 |
| Train/mean_episode_length | 787 |
| Episode_Reward/gait | 0.80 |
| Curriculum/terrain_levels | 0.017 (regressed from 4.17) |
| Loss/value_function | 162.56 (cascading from spike at 7044) |

## Issues Found

### Issue 1: Value Loss Explosion — Still Occurs Despite PPO Fixes

The PPO stability changes (lr=5e-4, gamma=0.98, value_loss_coef=0.5, empirical_normalization=True) were **not sufficient** to prevent the crash. Value loss spiked at iter 7044 to 2.85e29, partially recovered, then cascaded through iters 7048-7051 before crashing.

**Conclusion:** Hyperparameter tuning alone cannot prevent this. The root cause is unbounded GAE returns entering the value loss computation. Returns must be clamped at the source.

### Issue 2: Terrain Curriculum Regressed

Terrain level dropped from 4.17 to 0.017 over 7051 iterations. The diverse terrain (slopes, stairs, rough) was too challenging — the robot could not maintain performance on harder terrain and the curriculum demoted it back to the easiest levels.

Episode length never reached 1000 (peaked at ~820), indicating the robot was regularly falling on non-flat terrain.

### What Improved

- **Gait reward reached 0.80** (vs Run 2's 0.69) — increasing gait weight from 0.5 to 1.0 worked as intended
- **Gait learning was faster** — reached 0.70 by iter ~3500 (vs Run 2's ~10000 for 0.70)

## Comparison with Run 2

| Metric | Run 2 (iter 7051) | Run 3 (iter 7051) |
|--------|-------------------|-------------------|
| Mean reward | ~50 | ~34 |
| Episode length | ~1000 | ~788 |
| Gait | ~0.67 | **0.80** |
| Terrain level | ~5.3 (all flat) | 0.017 (regressed) |
| Value loss | ~0.008 | 162 (cascading) |
| Stability | Stable at this point | Crashing |

Run 3 has better gait but worse overall performance due to harder terrain and earlier crash.

## Lessons Learned

1. **Value loss clamping is required.** Hyperparameter tuning (lr, gamma, value_loss_coef, normalization) reduces frequency of spikes but does not prevent them. The loss computation itself must be bounded.

2. **Terrain diversity works for gait quality** — gait reward improved significantly with diverse terrain, even though overall reward was lower.

3. **Terrain curriculum can regress.** Starting with 5 terrain types at equal proportion may be too aggressive. Consider starting with higher flat proportion (e.g., 40%) and lower stairs proportion.

4. **Value loss spikes can cascade.** Unlike Run 2 where the spike at iter 6886 was a single-step recovery, Run 3's spike at iter 7044 led to a multi-step cascade (7044 → 7048 → 7049 → 7051) before crashing. The Adam optimizer was progressively corrupted.

## Resolution

Required fix added to `TP_v3_0.md`:
- Monkey-patch PPO update to clamp returns to `[-1000, 1000]` before value loss computation
- Further reduce learning rate to 3e-4
