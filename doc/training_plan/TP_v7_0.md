# PM01 Training Plan — Run 7

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| v7.0 | 2026-03-31 | armmarov | Clean slate — remove ALL venv patches, use only config-level fix |

**Previous runs:** Runs 1-6 (all crashed)

---

## What Went Wrong in Runs 3-6

We created a cascading mess of venv patches that interfered with each other:

1. **ppo.py line 308:** `returns_batch.clamp(-100, 100)` — clamps returns
2. **ppo.py line 372:** `if value_loss.item() > 100: continue` — skips gradient steps
3. **actor_critic.py line 107/109:** `.clamp(min=0.01)` — clamps std

**The death spiral in Run 6:**
- Returns clamp + value loss skip caused ALL gradient updates to be skipped (losses showed 0.0000)
- With no gradient updates, the policy drifted randomly
- Actor weights diverged → outputs inf → `action_rate: -inf` → crash

**We need to start clean.** Remove ALL venv patches. Use ONLY the config-level fix (`noise_std_type="log"`).

---

## Root Cause (Final)

The ONLY thing that needs to change from the original working Run 2 config:

**`noise_std_type="log"`** — prevents std from going negative via `exp()` parameterization.

Everything else (returns clamp, value loss skip, std clamp) was unnecessary and created interference. Run 2 trained stably for 13258 iterations with the default config. The only reason it crashed was the scalar std going negative from a rare gradient spike.

---

## Changes Required for Run 7

### Change 1: Remove ALL venv patches

**File:** `venv/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py`

Remove both patches:
- Line 304-308: Remove `returns_batch = returns_batch.clamp(-100.0, 100.0)`
- Line 371-373: Remove `if value_loss.item() > 100.0: continue`

Restore to original rsl_rl code.

**File:** `venv/lib/python3.11/site-packages/rsl_rl/modules/actor_critic.py`

Remove clamp patches:
- Line 107: Change back to `std = self.std.expand_as(mean)` (remove `.clamp(min=0.01)`)
- Line 109: Change back to `std = torch.exp(self.log_std).expand_as(mean)` (remove `.clamp(min=0.01)`)

**Easiest way to restore both files:**
```bash
pip install --force-reinstall rsl-rl-lib==3.1.3
```

### Change 2: Keep PM01PPORunnerCfg with log std (already in place)

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

Already correct from Run 6:
```python
@configclass
class PM01PPORunnerCfg(BasePPORunnerCfg):
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
```

### Change 3: Keep velocity_env_cfg.py at Run 2 config (already in place)

Already correct from Run 6:
- Terrain: 100% flat
- Gait weight: 0.5
- Original reward weights
- Command ranges with y/z

---

## Summary: What's Different from Run 2

| Parameter | Run 2 | Run 7 |
|-----------|-------|-------|
| `noise_std_type` | `"scalar"` (default) | **`"log"`** |
| Everything else | Same | Same |

That's it. One config change. No venv patches.

---

## Expected Results

| Metric | Run 2 (crashed iter 13258) | Run 7 expected |
|--------|---------------------------|----------------|
| Training stability | Crashed at 13258 | Should complete 50000 |
| Mean reward | Plateaued at ~51 | Same plateau at ~51 |
| Gait | 0.69-0.72 | Same |
| Episode length | 1000 | 1000 |

---

## How to Run

**Step 1: Restore clean rsl_rl (remove all venv patches):**
```bash
pip install --force-reinstall rsl-rl-lib==3.1.3
```

**Step 2: Train:**
```bash
./unitree_rl_lab.sh -t --task Unitree-PM01-12dof-Velocity
```

**Monitor:**
```bash
tensorboard --logdir logs/rsl_rl/unitree_pm01_12dof_velocity/
```

---

## Monitoring

- `Loss/value_function` — may spike occasionally (as in Run 2) but should NOT cause crash
- `Train/mean_reward` — should reach ~51
- `Episode_Termination/bad_orientation` — should stay below 5%
- `Episode_Reward/action_rate` — should NEVER show -inf (if it does, something is wrong)

---

## After Run 7 Succeeds

### Run 8: Add terrain diversity
- Keep `noise_std_type="log"`
- Add diverse terrain (start gentle: 40% flat, 20% rough, 20% slope, 20% stairs with low difficulty)
- Keep all rewards at Run 2 values

### Run 9: Tune rewards
- Increase gait weight
- Add ankle deviation
- Only after stable terrain training

---

## Pre-Training Sign-Off Checklist

Verified by: _______________
Date: _______________

### Change 1: Clean rsl_rl
- [ ] `pip install --force-reinstall rsl-rl-lib==3.1.3` executed
- [ ] Verify NO clamp in `actor_critic.py` lines 107/109
- [ ] Verify NO clamp or skip in `ppo.py` lines 304-308, 371-373
- [ ] Verify files are clean: `grep -n "clamp\|continue" venv/.../rsl_rl/algorithms/ppo.py` shows no custom patches
- [ ] Verify files are clean: `grep -n "clamp" venv/.../rsl_rl/modules/actor_critic.py` shows no custom patches

### Change 2: PM01PPORunnerCfg
- [ ] `noise_std_type="log"` in policy config
- [ ] No algorithm override (inherits BasePPORunnerCfg defaults)

### Change 3: velocity_env_cfg.py
- [ ] Terrain: `"flat": MeshPlaneTerrainCfg(proportion=0.5)` only
- [ ] Gait weight = 0.5
- [ ] `joint_vel` = -0.001, `energy` = -2e-5
- [ ] No `joint_deviation_ankles`
- [ ] Command ranges: all three axes (-0.1, 0.1)

### Status: PENDING REVIEW
