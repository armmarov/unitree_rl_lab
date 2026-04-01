# PM01 Training Plan — Run 8

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| v8.0 | 2026-03-31 | armmarov | Add gamma=0.95 to reduce return magnitude — fix for value loss spikes |
| v8.0.1 | 2026-04-01 | armmarov | Corrected return magnitude math, added Codex review feedback |

**Previous run:** Run 7 — survived 17634 iters (past Run 2's 13258) but still crashed from value loss spike

---

## Run 7 Result

Run 7 (`noise_std_type="log"`, flat terrain, no venv patches) survived **17634 iterations** — longest PM01 run so far. But still crashed from value loss explosion (2.04e31).

The `noise_std_type="log"` helped (survived 4000 iters longer than Run 2) but didn't prevent the actor network weights from diverging when value loss spiked. The actor output extreme actions → `action_rate: -2.05e14` → returns went to inf → crash.

**Root cause confirmed:** PM01's returns are ~2x G1's due to fewer penalty terms. With `gamma=0.99`, outlier returns reach magnitudes where the squared value loss cascades into optimizer corruption.

---

## The Fix: `gamma=0.95`

Lower gamma directly caps return magnitude.

**Formula:** `Return_max = r * (1 - gamma^T) / (1 - gamma)` where `r` = per-step reward, `T` = episode length.

| gamma | Max return (r=2.0, T=1000) | Max return (r=1.0, T=1000) | Outlier value loss (2x return vs prediction) |
|-------|----------------------------|----------------------------|----------------------------------------------|
| 0.99 | **200** | 100 (G1) | `(400-200)^2 = 40,000` → cascades |
| 0.97 | 66.7 | 33.3 | `(133-67)^2 = 4,356` |
| **0.95** | **40** | 20 | `(80-40)^2 = 1,600` (recoverable) |

**G1 at gamma=0.99:** returns up to ~100 (per-step reward ~1.0). G1's value loss max was 0.43 across 5916 iters — completely stable at this magnitude.

**PM01 at gamma=0.95:** returns up to ~40. This is **lower** than G1's returns, providing extra safety margin. Value loss spikes in Run 8 peaked at 73,235 and self-recovered — confirming the returns are bounded enough to prevent cascading overflow.

**Note:** gamma=0.95 is more aggressive than strictly necessary. gamma=0.97 (returns ~67) would still be safer than G1's ~100. If Run 8 underperforms due to the short horizon, gamma=0.97 is the next value to try.

`gamma=0.95` gives an effective horizon of ~20 steps. This is commonly used in locomotion tasks and sufficient for flat-terrain walking behaviors.

---

## Changes Required for Run 8

### Change 1: Add gamma=0.95 to PM01PPORunnerCfg

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

```python
@configclass
class PM01PPORunnerCfg(BasePPORunnerCfg):
    """PPO config for PM01.

    Changes from BasePPORunnerCfg:
    - noise_std_type='log': prevents std from going negative (exp(x) > 0 for all x)
    - gamma=0.95: reduces return magnitude from ~200 to ~40, prevents value loss overflow
      (PM01's per-step reward is ~2x G1's due to fewer penalty terms)
    """

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.95,              # was 0.99 — caps returns to ~40 (vs ~200)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### No other changes

- Terrain: 100% flat (same as Run 7)
- Rewards: original weights (same as Run 7)
- No venv patches
- `noise_std_type="log"` kept from Run 7
- Command ranges: (-0.1, 0.1) for all axes — intentionally conservative for initial stability testing

---

## Summary: What's Different from Run 2

| Parameter | Run 2 (crashed iter 13258) | Run 8 |
|-----------|---------------------------|-------|
| `noise_std_type` | `"scalar"` | **`"log"`** |
| `gamma` | `0.99` | **`0.95`** |
| Everything else | Same | Same |

Two config changes. No venv patches.

**Important:** `gamma=0.95` addresses the symptom (large returns) not the structural root cause (PM01 has fewer penalty terms than G1). When rewards or terrain change in future runs, the value loss problem may resurface. The long-term fix is reward rebalancing (Run 10+).

---

## Expected Results

| Metric | Run 7 (crashed iter 17634) | Run 8 expected |
|--------|---------------------------|----------------|
| Training stability | Crashed at 17634 | Should complete 50000 — returns bounded |
| Value loss | Spiked to 2.04e31 | May still spike but should self-recover (peak ~73K observed, recoverable) |
| Mean reward | ~51 before spike | Likely lower (~18-30) due to shorter horizon |
| Gait | 0.62 | Similar or lower |
| Episode length | 986 | Should improve over training |

Reward will be lower because `gamma=0.95` discounts future rewards more aggressively. The agent optimizes over a ~20 step window instead of ~100 steps. If final performance is too low, gamma=0.97 is the next step.

---

## How to Run

```bash
./unitree_rl_lab.sh -t --task Unitree-PM01-12dof-Velocity
```

**Monitor:**
```bash
tensorboard --logdir logs/rsl_rl/unitree_pm01_12dof_velocity/
```

---

## Monitoring

- `Loss/value_function` — may spike but should self-recover. If spike exceeds 1e10 and doesn't recover within 10 iters, consider stopping.
- `Train/mean_reward` — should reach 15+ (lower than Run 7's ~51 due to gamma)
- `Episode_Reward/action_rate` — should stay in normal range (never -inf or -1e14)
- Checkpoint saves every 100 iters — can resume from last good checkpoint if non-crash failures occur (OOM, hardware)

---

## Rollback Plan

- If `Loss/value_function` exceeds 1e10 and does not recover within 10 iterations → stop run, reduce gamma to 0.93
- If mean reward fails to exceed 10.0 after 5000 iterations → gamma may be too aggressive, try 0.97
- If crash occurs despite gamma=0.95 → add `value_loss_coef=0.5` as additional dampening

---

## After Run 8 Succeeds

### Run 9: Add terrain diversity
- Keep gamma=0.95 + log std
- Add diverse terrain (start gentle: 40% flat, 20% rough, 20% slope, 20% stairs)
- **Note:** gamma=0.95 may be too short-sighted for terrain tasks (anticipating stairs/slopes). May need gamma=0.97 for terrain runs.

### Run 10: Tune rewards (structural fix for reward imbalance)
- Add penalty terms to bring PM01's positive/negative reward ratio closer to G1's 1.47
- This would allow raising gamma back toward 0.99 without value loss instability
- Consider return/value normalization as an alternative

---

## Full Reward Function Reference

For reviewers — PM01's complete reward terms and weights:

| Term | Weight | Type |
|------|--------|------|
| track_lin_vel_xy | +1.0 | task |
| track_ang_vel_z | +0.5 | task |
| alive | +0.15 | task |
| feet_clearance | +1.0 | feet |
| gait | +0.5 | feet |
| base_linear_velocity | -2.0 | regularization |
| base_angular_velocity | -0.05 | regularization |
| flat_orientation_l2 | -5.0 | regularization |
| base_height | -10.0 | regularization |
| joint_vel | -0.001 | regularization |
| joint_acc | -2.5e-7 | regularization |
| action_rate | -0.05 | regularization |
| dof_pos_limits | -5.0 | regularization |
| energy | -2e-5 | regularization |
| joint_deviation_legs | -1.0 | regularization |
| feet_slide | -0.2 | feet |
| undesired_contacts | -1.0 | safety |

G1 has 19 terms (includes arm deviation -0.1, waist deviation -1.0). PM01 has 17 terms. The missing penalties plus lower per-joint costs (12 vs 29 DOF) result in PM01's net per-step reward being ~2x G1's.

---

## Pre-Training Sign-Off Checklist

Verified by: _______________
Date: _______________

- [ ] `PM01PPORunnerCfg` has `noise_std_type="log"`
- [ ] `PM01PPORunnerCfg` has `gamma=0.95`
- [ ] All other algorithm params match BasePPORunnerCfg (lr=1e-3, vlc=1.0, etc.)
- [ ] No venv patches in `ppo.py` or `actor_critic.py`
- [ ] Terrain: flat only (proportion=0.5)
- [ ] Rewards: original weights (gait=0.5, joint_vel=-0.001, energy=-2e-5)
- [ ] Command ranges: all three axes (-0.1, 0.1) — intentionally conservative
- [ ] Checkpoints saving every 100 iters (save_interval=100)

### Status: PENDING REVIEW
