# PM01 Training Plan — Run 5

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| v5.0 | 2026-03-31 | armmarov | Returns clamping inside mini-batch loop (fixes v4's insufficient monkey-patch) |

**Previous runs:** `TH_v1_0.md` (Run 1), `TH_v2_0.md` (Run 2), `TH_v3_0.md` (Run 3), Run 4 (crashed, see below)
**Root cause analysis:** `ANALYSIS_value_loss_crash.md`

---

## Run 4 Result: CRASHED

Run 4's monkey-patch (`train.py _safe_update`) clamped `self.storage.returns` before `update()`, but the crash still occurred at iter 3243. The clamping was **too shallow** — it only affected the full return buffer, not the mini-batch slices inside the update loop. Additionally, even with clamped returns, the critic's `value_batch` output can diverge if the network is already corrupted, producing huge `(value_batch - returns_batch).pow(2)` regardless.

Value loss was in the millions/trillions throughout iters 3234-3243 despite the clamp, confirming the patch location was wrong.

---

## What Changed from v4

### v4 approach (failed): Monkey-patch in `train.py`

```python
# Clamped self.storage.returns BEFORE update() — too late, mini-batches bypass this
def _safe_update(self):
    self.storage.returns = self.storage.returns.clamp(-1000, 1000)
    return _original_update(self)
```

**Why it failed:** Inside `update()`, returns are sliced into mini-batches and the value network produces `value_batch` independently. Even with clamped returns, `value_batch` can be arbitrarily large if the critic is already corrupted from a prior spike.

### v5 approach (fix): Clamp `returns_batch` inside the mini-batch loop

```python
# Line 304-308 in rsl_rl/algorithms/ppo.py, BEFORE value loss computation
returns_batch = returns_batch.clamp(-100.0, 100.0)
```

**Why this works:** Clamping happens **inside** the mini-batch loop, directly before `returns_batch` enters the squared loss computation on lines 315-317. This bounds the maximum possible value loss per mini-batch to `(value_batch - 100)^2`, which prevents fp32 overflow regardless of what the critic outputs.

---

## Changes Required for Run 5

### Change 1: Clamp returns_batch inside PPO update loop (DONE)

**File:** `venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py`

**Location:** After line 302 (surrogate_loss), before line 310 (value function loss)

**What was added:**
```python
# PM01 stability: clamp returns_batch to prevent fp32 overflow in value loss.
# PM01's per-step reward is ~2x G1's (fewer joints = fewer penalties), producing
# larger GAE returns. Outlier batches cause value_loss spikes → Adam corruption → crash.
# Normal returns are ~20-125; clamping at ±100 only affects extreme outliers.
returns_batch = returns_batch.clamp(-100.0, 100.0)
```

**Why ±100 instead of ±1000:** Normal returns are ~20-50 at plateau. A clamp at 100 gives 2x headroom while being tight enough to prevent the squared error from reaching dangerous magnitudes. `(value - 100)^2 = 10,000` at worst — safe for fp32 and Adam.

**WARNING:** This edits the venv package directly. If `rsl-rl-lib` is reinstalled, the patch will be lost. Consider:
- Adding a post-install patch script
- Or forking rsl-rl-lib with this fix

### Change 2: Remove monkey-patch from train.py (DONE)

**File:** `scripts/rsl_rl/train.py`

The `_safe_update` monkey-patch from v4 has been removed since the fix is now directly in rsl_rl.

### Change 3: All other changes from v4 carried forward

Already in place from v4:
- `PM01PPORunnerCfg`: lr=3e-4, gamma=0.98, value_loss_coef=0.5, empirical_normalization=True
- Gym registration points to `PM01PPORunnerCfg`
- Terrain diversity: flat 20%, rough 20%, slopes 40%, stairs 20%
- Gait weight = 1.0
- Reward rebalancing: joint_vel=-0.005, energy=-5e-5, joint_deviation_ankles=-0.5
- Command ranges: x/y/z all (-0.1, 0.1)

---

## Expected Results for Run 5

| Metric | Run 4 (crashed) | Run 5 expected |
|--------|-----------------|----------------|
| Training stability | Crashed at iter 3243 | Should complete 50000 — returns clamped inside update loop |
| Value loss | Millions/trillions | Bounded by clamp — max ~10,000 per mini-batch |
| Mean reward | ~14 at crash | Should grow past Run 3's ~34 with stable training |
| Terrain level | — | Should show real curriculum progression |
| Gait reward | — | Should reach 0.7+ as in Runs 2-3 |

---

## How to Run

**Fresh start:**
```bash
./unitree_rl_lab.sh -t --task Unitree-PM01-12dof-Velocity
```

**Monitor:**
```bash
tensorboard --logdir logs/rsl_rl/unitree_pm01_12dof_velocity/
```

**Play:**
```bash
./unitree_rl_lab.sh -p --task Unitree-PM01-12dof-Velocity --load_run <run_name>
```

---

## Monitoring

- `Loss/value_function > 100` — warning, clamp is activating frequently
- `Loss/value_function > 10,000` — should not happen with clamp at ±100
- `Loss/entropy` dropping toward 0 (kill threshold: < 1.5) — policy collapsing
- `Curriculum/terrain_levels` — should show upward progression

---

## Pre-Training Sign-Off Checklist

Verified by: _______________
Date: _______________

### Change 1: Returns clamping in rsl_rl source
- [ ] `returns_batch = returns_batch.clamp(-100.0, 100.0)` added at line 308 in `venv/.../rsl_rl/algorithms/ppo.py`
- [ ] Located AFTER surrogate_loss computation, BEFORE value function loss
- [ ] Comment block explains rationale

### Change 2: Monkey-patch removed
- [ ] `_safe_update` function removed from `scripts/rsl_rl/train.py`
- [ ] No `import types` or `MethodType` remaining in the patch area

### Change 3: Carried forward from v4
- [ ] `PM01PPORunnerCfg` with lr=3e-4, gamma=0.98, value_loss_coef=0.5
- [ ] `empirical_normalization = True`
- [ ] Gym registration points to `PM01PPORunnerCfg`
- [ ] Terrain: 5 sub-terrains with correct IsaacLab class names
- [ ] Gait weight = 1.0
- [ ] `joint_vel` weight = -0.005, `energy` weight = -5e-5
- [ ] `joint_deviation_ankles` present with weight -0.5
- [ ] Command ranges: all three axes at (-0.1, 0.1)

### Status: PENDING REVIEW
