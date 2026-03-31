# PM01 Training Plan — Run 4

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| v4.0 | 2026-03-31 | armmarov | Returns clamping + reward rebalancing to fix value loss crash |

**Previous runs:** `TH_v1_0.md` (Run 1), `TH_v2_0.md` (Run 2), `TH_v3_0.md` (Run 3)
**Root cause analysis:** `ANALYSIS_value_loss_crash.md`

---

## Problem Summary

Runs 2 and 3 both crashed with `RuntimeError: normal expects all elements of std >= 0.0` caused by value loss overflow (fp32). PPO hyperparameter tuning in Run 3 (lr=5e-4, gamma=0.98, value_loss_coef=0.5) was **not sufficient** — Run 3 crashed earlier (iter 7051 vs 13258).

**Root cause:** PM01's per-step reward (+2.24) is 2x G1's (+1.08) due to missing penalty terms (no arms/waist) and fewer joints. This produces higher discounted returns, larger squared errors in value loss, and fp32 overflow. G1 had zero spikes > 1.0 across 5916 iterations; PM01 had 100+ spikes per run.

See `ANALYSIS_value_loss_crash.md` for full details.

---

## Goals

- **Goal A: Eliminate value loss crashes** — clamp returns to prevent fp32 overflow
- **Goal B: Rebalance rewards** — bring PM01's positive/negative ratio closer to G1's for long-term stability
- **Goal C: Continue terrain + gait improvements** — carry forward Run 3's terrain diversity and gait weight

---

## Changes Required for Run 4

### Change 1: Add returns clamping to training script

**File:** `scripts/rsl_rl/train.py`

**Why:** The value loss explosion originates from unbounded GAE returns entering the squared loss computation. Normal returns are ~20-50; the overflow happens when outlier returns reach fp32 limits. Clamping to `[-1000, 1000]` prevents overflow without affecting normal training.

**Add after line 184** (after `runner = OnPolicyRunner(...)`) **and before line 204** (`runner.learn(...)`):

```python
# --- Monkey-patch PPO to clamp returns and prevent fp32 overflow ---
import types
import torch

_original_update = runner.alg.update.__func__

def _safe_update(self):
    """Wrapper that clamps returns to prevent fp32 overflow in value loss."""
    returns = self.storage.returns
    max_return = 1000.0
    if returns.abs().max() > max_return:
        self.storage.returns = returns.clamp(-max_return, max_return)
    return _original_update(self)

runner.alg.update = types.MethodType(_safe_update, runner.alg)
# --- End monkey-patch ---
```

**Alternative (if monkey-patch doesn't work):** Directly edit `venv/.../rsl_rl/algorithms/ppo.py`, add before line 309:
```python
returns_batch = returns_batch.clamp(-1000.0, 1000.0)
```

### Change 2: Rebalance PM01 reward terms

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/pm01/12dof/velocity_env_cfg.py`

**Why:** PM01's positive/negative reward ratio is 1.90-2.46 vs G1's 1.47. Adding penalties and adjusting weights brings PM01 closer to G1's balance, reducing return variance and making training fundamentally more stable.

**Changes in `RewardsCfg`:**

```python
# -- increase existing penalties
joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)        # was -0.001
energy = RewTerm(func=mdp.energy, weight=-5e-5)                  # was -2e-5

# -- add ankle deviation penalty (new term)
joint_deviation_ankles = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=[
            "j04_ankle_pitch_l", "j10_ankle_pitch_r",
            "j05_ankle_roll_l", "j11_ankle_roll_r",
        ],
    )},
)
```

**Expected effect on reward balance:**

| Metric | Run 3 | Run 4 target |
|--------|-------|--------------|
| Positive/negative ratio | ~2.46 | ~1.6-1.8 |
| `joint_vel` penalty | -0.03 | ~-0.15 |
| `energy` penalty | -0.001 | ~-0.005 |
| `joint_deviation_ankles` | (none) | ~-0.10 |

### Change 3: Lower learning rate further

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

**Change in `PM01PPORunnerCfg`:**
```python
learning_rate=3.0e-4,    # was 5e-4
```

### Change 4: Carry forward from Run 3 (already in place)

Verify these are still in place from Run 3:
- Terrain diversity (flat 20%, rough 20%, slopes 40%, stairs 20%)
- Gait weight = 1.0
- Command ranges with y/yaw
- `empirical_normalization = True`
- `gamma = 0.98`
- `value_loss_coef = 0.5`

---

## Expected Results for Run 4

| Metric | Run 3 (crashed) | Run 4 expected |
|--------|-----------------|----------------|
| Training stability | Crashed at iter 7051 | Should complete 50000 — returns clamping prevents overflow |
| Value loss spikes | 120 spikes > 1.0 | Spikes should still occur but be bounded (clamped returns → max loss ~1M not ~1e21) |
| Mean reward | ~34 (harder terrain) | May be lower initially due to increased penalties, should recover |
| Gait reward | 0.80 | Similar — gait weight still 1.0 |
| Terrain level | Regressed to 0.017 | Should show real progression with stable training |
| Episode length | ~788 | Should improve as terrain learning stabilizes |

---

## How to Run

**Fresh start (required — reward weights changed):**
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

Watch for these during training:
- `Loss/value_function > 1000` — warning, returns clamping should cap this but monitor
- `Loss/value_function > 1e6` — critical, clamping may not be working correctly
- `Loss/entropy` dropping sharply toward 0 (kill threshold: < 1.5) — policy collapsing
- `Curriculum/terrain_levels` — should show upward progression (was regressing in Run 3)
- `Train/mean_reward` — may start lower than Run 3 due to extra penalties, but should grow steadily

---

## Checkpoints to Evaluate

- `model_1000.pt` — early learning
- `model_5000.pt` — mid training (compare with Run 3's crash point at 7051)
- `model_10000.pt` — past Run 2's crash point (13258), should be stable
- `model_25000.pt` — half way
- `model_50000.pt` — final policy

---

## If Training Still Crashes

- Lower `max_return` clamp from 1000 to 100
- Reduce `num_learning_epochs` from 5 to 3
- Try `gamma=0.97`
- Reduce positive reward weights: `track_lin_vel_xy` from 1.0 to 0.8

## If Terrain Curriculum Still Regresses

- Increase flat proportion from 20% to 40%, reduce stairs to 10%
- Lower stair `step_height_range` from (0.03, 0.10) to (0.02, 0.06)
- Lower slope `slope_range` from (0.0, 0.4) to (0.0, 0.25)

## Deployment Preparation

- Export best checkpoint to ONNX for real robot inference
- Verify `joint_sdk_names` order matches PM01 hardware SDK
- Test with `sim2sim` before deploying to hardware

---

## Pre-Training Sign-Off Checklist

Verified by: _______________
Date: _______________

### Change 1: Returns clamping
- [ ] Monkey-patch added to `scripts/rsl_rl/train.py` after runner creation
- [ ] `max_return = 1000.0`
- [ ] Verified patch runs without import errors

### Change 2: Reward rebalancing
- [ ] `joint_vel` weight changed to `-0.005` (was -0.001)
- [ ] `energy` weight changed to `-5e-5` (was -2e-5)
- [ ] `joint_deviation_ankles` new term added with weight `-0.5`
- [ ] Ankle joint names match URDF: `j04_ankle_pitch_l`, `j10_ankle_pitch_r`, `j05_ankle_roll_l`, `j11_ankle_roll_r`

### Change 3: Learning rate
- [ ] `learning_rate = 3.0e-4` in `PM01PPORunnerCfg` (was 5e-4)

### Change 4: Carried forward from Run 3
- [ ] Terrain: 5 sub-terrains with correct class names
- [ ] Gait weight = 1.0
- [ ] Command ranges: x=(-0.1,0.1), y=(-0.1,0.1), z=(-0.1,0.1)
- [ ] `empirical_normalization = True`
- [ ] `gamma = 0.98`
- [ ] `value_loss_coef = 0.5`

### Status: PENDING REVIEW
