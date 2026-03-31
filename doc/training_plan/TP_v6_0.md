# PM01 Training Plan — Run 6

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| v6.0 | 2026-03-31 | armmarov | Step back to Run 2 config + log std fix. One change at a time. |

**Previous runs:** `TH_v1_0` through `TH_v4_0`, Run 5 (crashed iter 3419)
**Root cause analysis:** `ANALYSIS_value_loss_crash.md`

---

## Lessons from Runs 2-5: We Changed Too Much

| Run | Terrain | Bad Orientation | Crash Iter | Changes from Run 2 |
|-----|---------|-----------------|------------|---------------------|
| Run 2 | 100% flat | 0.3% | 13258 | — (baseline) |
| Run 3 | Diverse | 47% | 7051 | +terrain +PPO +gait weight |
| Run 4 | Diverse | 99% | 3243 | +rewards +clamp (monkey-patch) |
| Run 5 | Diverse | 99% | 3419 | +returns clamp (ppo.py) |

**Crashes got earlier with each run** — not because the fixes were wrong, but because we kept adding difficulty (terrain, penalties) at the same time as trying to fix stability. With 99% bad_orientation in Runs 4-5, the robot was falling almost every episode, creating extreme reward variance that overwhelmed every fix we applied.

**Run 2 was the most stable** because the task was simple (flat terrain, fewer penalties). The crash at iter 13258 was a rare outlier after 13000 iterations of healthy training.

---

## Root Cause (Definitive, Two Problems)

### Problem 1: Crash Mechanism — `noise_std_type="scalar"`

Every crash is `RuntimeError: normal expects all elements of std >= 0.0`. The actor's noise std is stored as a raw `nn.Parameter` that gradient descent can push below zero:

```python
# actor_critic.py line 73
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))  # no lower bound!
```

rsl_rl already has the fix: `noise_std_type="log"` uses `std = exp(log_std)` which is mathematically always positive.

**This is the only essential fix.** Everything else is secondary.

### Problem 2: Training Difficulty — Too Many Changes at Once

Runs 3-5 simultaneously changed terrain, rewards, and PPO params. The diverse terrain was too hard (99% bad_orientation), creating extreme reward variance that made Problem 1 trigger much earlier.

---

## Strategy: One Change at a Time

1. **Run 6 (this plan):** Fix the crash mechanism ONLY. Use Run 2's proven config + log std.
2. **Run 7 (future):** If Run 6 completes 50000 iters, add terrain diversity.
3. **Run 8 (future):** Tune rewards and gait weight.

---

## Changes Required for Run 6

### Change 1: Switch to log std parameterization (THE FIX)

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

Add `policy` override in `PM01PPORunnerCfg` with `noise_std_type="log"`:

```python
@configclass
class PM01PPORunnerCfg(BasePPORunnerCfg):
    """PPO config for PM01.

    Only change from BasePPORunnerCfg: noise_std_type='log' to prevent
    std from going negative (exp(x) > 0 for all x).
    """
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
```

**Revert all other PM01PPORunnerCfg overrides back to BasePPORunnerCfg defaults:**
- `empirical_normalization = False` (remove override)
- `algorithm` — remove override, inherit BasePPORunnerCfg (lr=1e-3, gamma=0.99, value_loss_coef=1.0)

### Change 2: Add std clamp as safety net

**File:** `venv/lib/python3.11/site-packages/rsl_rl/modules/actor_critic.py`

Change `update_distribution` (lines 106-109):
```python
if self.noise_std_type == "scalar":
    std = self.std.expand_as(mean).clamp(min=0.01)
elif self.noise_std_type == "log":
    std = torch.exp(self.log_std).expand_as(mean).clamp(min=0.01)
```

**WARNING:** Venv patch, lost on `rsl-rl-lib` reinstall.

### Change 3: Keep returns clamp as safety net

**File:** `venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py` (line 308)

Keep `returns_batch = returns_batch.clamp(-100.0, 100.0)` — doesn't hurt, provides secondary protection.

### Change 4: Revert velocity_env_cfg.py to Run 2 config

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/pm01/12dof/velocity_env_cfg.py`

Revert these changes back to Run 2 values:

```python
# Terrain — back to flat only
sub_terrains={
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
}

# Rewards — back to original
joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)       # was -0.005
energy = RewTerm(func=mdp.energy, weight=-2e-5)                  # was -5e-5
gait = RewTerm(func=mdp.feet_gait, weight=0.5, ...)             # was 1.0

# Remove joint_deviation_ankles entirely (was new in Run 4)
```

**Keep these from Run 2 (already correct):**
- Command ranges: x/y/z all (-0.1, 0.1) — needed for gait reward gating
- `RobotPlayEnvCfg` using `limit_ranges`
- Termination `minimum_height=0.3`
- Undesired contacts `(?!.*ankle.*)`

---

## Defense in Depth

| Layer | What | Where | Prevents |
|-------|------|-------|----------|
| 1. Log std | `std = exp(log_std)` always > 0 | Config (no venv patch) | std going negative |
| 2. Std clamp | `.clamp(min=0.01)` | actor_critic.py (venv) | Edge case underflow |
| 3. Returns clamp | `returns_batch.clamp(-100, 100)` | ppo.py (venv) | Value loss overflow |

---

## Expected Results for Run 6

| Metric | Run 2 (baseline) | Run 6 expected |
|--------|-------------------|----------------|
| Training stability | Crashed at 13258 | Should complete 50000 — log std prevents crash |
| Mean reward | Plateaued at ~51 | Similar plateau at ~51 (same config) |
| Gait | 0.69-0.72 | Similar |
| Episode length | 1000 | 1000 |
| Terrain level | 5.3 (meaningless, flat) | Same (flat terrain, will address in Run 7) |
| Bad orientation | 0.3% | Similar |

**Run 6's success criteria:** Complete 50000 iterations without crash. Reward and gait quality are expected to match Run 2. The plateau problem is deliberately deferred to Run 7.

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

- `Loss/value_function` — may still spike but should not cause crash (std stays positive)
- `Loss/entropy` — should stay above 1.5
- `Train/mean_reward` — should reach ~51 (matching Run 2)
- `Episode_Termination/bad_orientation` — should stay below 5% on flat terrain

---

## Future Runs (After Run 6 Succeeds)

### Run 7: Add terrain diversity
- Keep log std + all safety nets from Run 6
- Change terrain to diverse mix (flat/rough/slopes/stairs)
- Keep rewards unchanged
- **If bad_orientation > 50%:** terrain is too hard, increase flat proportion or reduce slope/stair difficulty

### Run 8: Tune rewards for walking quality
- Increase gait weight (0.5 → 1.0)
- Add ankle deviation penalty
- Increase joint_vel/energy penalties
- Only after terrain training is stable

---

## Pre-Training Sign-Off Checklist

Verified by: _______________
Date: _______________

### Change 1: Log std parameterization
- [ ] `PM01PPORunnerCfg` has `policy` with `noise_std_type="log"`
- [ ] All other PPO overrides removed (inherits BasePPORunnerCfg defaults: lr=1e-3, gamma=0.99, vlc=1.0)
- [ ] `empirical_normalization` override removed (defaults to False)

### Change 2: Std clamp (venv)
- [ ] `.clamp(min=0.01)` on line 107 in `actor_critic.py` (scalar path)
- [ ] `.clamp(min=0.01)` on line 109 in `actor_critic.py` (log path)

### Change 3: Returns clamp (venv, from v5)
- [ ] `returns_batch.clamp(-100, 100)` at line 308 in `ppo.py`

### Change 4: Reverted to Run 2 config
- [ ] Terrain: only `"flat": MeshPlaneTerrainCfg(proportion=0.5)`
- [ ] `joint_vel` weight = -0.001
- [ ] `energy` weight = -2e-5
- [ ] `gait` weight = 0.5
- [ ] `joint_deviation_ankles` removed
- [ ] Command ranges: all three axes (-0.1, 0.1) (keep, needed for gait)

### Status: PENDING REVIEW
