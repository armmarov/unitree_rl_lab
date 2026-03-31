# PM01 Training Plan — Run 8

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| v8.0 | 2026-04-01 | armmarov | Add gamma=0.95 to reduce return magnitude — solid fix for value loss spikes |

**Previous run:** Run 7 — survived 17634 iters (past Run 2's 13258) but still crashed from value loss spike

---

## Run 7 Result

Run 7 (`noise_std_type="log"`, flat terrain, no venv patches) survived **17634 iterations** — longest PM01 run so far. But still crashed from value loss explosion (2.04e31).

The `noise_std_type="log"` helped (survived 4000 iters longer than Run 2) but didn't prevent the actor network weights from diverging when value loss spiked. The actor output extreme actions → `action_rate: -2.05e14` → returns went to inf → crash.

**Root cause confirmed:** PM01's returns are ~2x G1's due to fewer penalty terms. With `gamma=0.99`, outlier returns reach magnitudes where the squared value loss cascades into optimizer corruption.

---

## The Fix: `gamma=0.95`

Lower gamma directly caps return magnitude:

| gamma | Max return (per-step reward ~2.0, ep_len 1000) | Outlier value loss (2x return) |
|-------|------------------------------------------------|-------------------------------|
| 0.99 | ~100 | 10,000 → cascades to inf |
| 0.97 | ~33 | 1,089 |
| **0.95** | **~20** (matches G1's range) | **400** (safe) |

G1 is stable with `gamma=0.99` because its returns max at ~20 (per-step reward ~1.0). PM01 with `gamma=0.95` will have similar return magnitude — same stability.

`gamma=0.95` is commonly used in locomotion tasks. The agent considers ~20 steps into the future, sufficient for walking behaviors.

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
    - gamma=0.95: reduces return magnitude to match G1's range, prevents value loss overflow
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
        gamma=0.95,              # was 0.99 — caps returns to ~20 (same as G1)
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

---

## Summary: What's Different from Run 2

| Parameter | Run 2 (crashed iter 13258) | Run 8 |
|-----------|---------------------------|-------|
| `noise_std_type` | `"scalar"` | **`"log"`** |
| `gamma` | `0.99` | **`0.95`** |
| Everything else | Same | Same |

Two config changes. No venv patches. No workarounds.

---

## Expected Results

| Metric | Run 7 (crashed iter 17634) | Run 8 expected |
|--------|---------------------------|----------------|
| Training stability | Crashed at 17634 | Should complete 50000 |
| Value loss | Spiked to 2.04e31 | Should stay < 1.0 (returns bounded by gamma) |
| Mean reward | ~51 before spike | May be slightly lower (~40-50) due to shorter horizon |
| Gait | 0.62 | Similar |
| Episode length | 986 | 1000 |

Reward may be slightly lower because `gamma=0.95` discounts future rewards more aggressively. But training should be completely stable.

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

- `Loss/value_function` — should stay < 1.0 throughout, no spikes. If spikes appear, gamma needs to be even lower.
- `Train/mean_reward` — should reach 40+
- `Episode_Reward/action_rate` — should stay in normal range (never -inf or -1e14)

---

## After Run 8 Succeeds

### Run 9: Add terrain diversity
- Keep gamma=0.95 + log std
- Add diverse terrain (start gentle: 40% flat, 20% rough, 20% slope, 20% stairs)

### Run 10: Tune rewards
- Increase gait weight
- Add more penalties to improve walking quality

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
- [ ] Command ranges: all three axes (-0.1, 0.1)

### Status: PENDING REVIEW
