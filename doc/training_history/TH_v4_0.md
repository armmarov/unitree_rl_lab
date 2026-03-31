# Training History v4 — Run 4: Returns Clamping (Monkey-Patch, Failed)

**Robot:** EngineAI PM01 (12-DOF, legs only)
**Task:** `Unitree-PM01-12dof-Velocity`
**Date:** 2026-03-31
**Branch:** `pm01-training`
**Training Plan:** `TP_v4_0.md`

---

## Run Summary

| Field | Value |
|-------|-------|
| Run ID | `2026-03-31_13-46-27` |
| Iterations | 3243 / 50000 |
| Duration | ~1.1 hours |
| Final mean reward | -39,963,252 (diverged) |
| Final episode length | 485 |
| Outcome | **Crashed** — `RuntimeError: normal expects all elements of std >= 0.0` |

## Config Changes from Run 3

| Parameter | Run 3 | Run 4 |
|-----------|-------|-------|
| learning_rate | 5e-4 | 3e-4 |
| Returns clamping | None | Monkey-patch in train.py (clamp ±1000 on `self.storage.returns`) |
| joint_vel weight | -0.001 | -0.005 |
| energy weight | -2e-5 | -5e-5 |
| joint_deviation_ankles | None | New term, weight -0.5 |

All other settings carried from Run 3 (terrain, gait weight=1.0, gamma=0.98, etc.)

## Training Progression

| Iter | Mean Reward | Episode Length | Gait | Terrain Level | Value Loss |
|------|-------------|---------------|------|---------------|------------|
| 0 | -1.26 | 13 | 0.004 | 4.17 | — |
| ~1600 | ~12 | ~420 | ~0.10 | ~0.5 | 107 (first spike) |
| ~3234 | 13.96 | 433 | ~0.19 | ~0.0 | 764,232 |
| ~3238 | -1.4e9 | 439 | — | — | 15,445,799 |
| 3243 | -39,963,252 | 485 | 0.20 | 0.0 | 1,600,593 |

## Why the Monkey-Patch Failed

The `_safe_update` function in `train.py` clamped `self.storage.returns` **before** calling `update()`:

```python
def _safe_update(self):
    max_return = 1000.0
    if self.storage.returns.abs().max() > max_return:
        self.storage.returns = self.storage.returns.clamp(-max_return, max_return)
    return _original_update(self)
```

**Problem 1:** Inside `update()`, returns are sliced into mini-batches. The clamp on the full buffer may not propagate correctly to all mini-batch slices depending on tensor memory layout.

**Problem 2:** Even with clamped returns, the critic network's `value_batch` output can diverge independently if the network is already partially corrupted from a prior spike. The value loss `(value_batch - returns_batch).pow(2)` can still be huge if `value_batch` is large.

**Problem 3:** The clamp threshold (±1000) was too loose. Normal returns are ~20-50, so a return of 999 would produce value_loss of ~(999-50)^2 = 900,601 — still large enough to corrupt Adam.

## Key Observations

- First value loss spike at iter 1604 (only 107) — spikes started even earlier than Runs 2-3
- Reward never exceeded ~14 — Run 4 crashed before reaching Run 3's peak of ~34
- Terrain curriculum regressed to 0.0 (from 4.17) — robot couldn't handle any terrain
- Gait only reached 0.20 — much lower than Runs 2-3 (0.69-0.80)
- The monkey-patch is visible in the traceback: `train.py line 214, _safe_update`

## Comparison with Previous Runs

| Metric | Run 2 | Run 3 | Run 4 |
|--------|-------|-------|-------|
| Crash iteration | 13258 | 7051 | **3243** |
| Peak reward | ~51 | ~34 | **~14** |
| Peak gait | 0.72 | 0.80 | **0.20** |
| First spike >1.0 | iter 1439 | iter 1398 | **iter 1604** |
| Spikes >1.0 (total) | 114 | 120 | **38** (fewer iters) |
| Spike rate | 1.1% | 1.7% | **1.2%** |

Crash came progressively earlier across runs. Run 4 crashed before the robot could learn meaningful locomotion.

## Lessons Learned

1. **Clamping returns before update() is insufficient.** The fix must be inside the mini-batch loop, directly before the value loss computation where `returns_batch` is used.

2. **Clamp threshold matters.** ±1000 is too loose — normal returns of ~50 plus a diverging critic can still produce explosive losses. ±100 provides sufficient headroom (2x normal) while preventing dangerous magnitudes.

3. **The crash gets worse with more changes.** Runs 2→3→4 crashed at 13258→7051→3243. Additional reward terms and terrain complexity may increase return variance, making spikes more frequent/severe.

## Resolution

Fix applied in Run 5 (`TP_v5_0.md`):
- Clamp `returns_batch` directly inside the PPO mini-batch loop (line 308 of `rsl_rl/algorithms/ppo.py`)
- Tighter clamp at ±100 (was ±1000)
- Removed monkey-patch from `train.py`
