# Training History v0 — PM01 12-DOF Initial Runs

**Robot:** EngineAI PM01 (12-DOF, legs only)
**Task:** `Unitree-PM01-12dof-Velocity`
**Date:** 2026-03-30
**Branch:** `pm01-training`

---

## Run 1

| Field | Value |
|-------|-------|
| Run ID | `2026-03-30_20-47-12` |
| Iterations | ~1510 / 50000 |
| Duration | ~20 min (stopped manually) |
| Final mean reward | 12.76 |
| Final episode length | ~460 |
| Outcome | **Stopped manually** |

### Config

- PPO: `BasePPORunnerCfg` (shared, lr=1e-3, gamma=0.99)
- Terrain: 100% flat (`MeshPlaneTerrainCfg` proportion=0.5, nothing else)
- Commands: `lin_vel_x=(-0.1, 0.1)`, `lin_vel_y=(0.0, 0.0)`, `ang_vel_z=(0.0, 0.0)`
- Gait weight: 0.5

### Key Metrics at Stop

| Metric | Value |
|--------|-------|
| Train/mean_reward | 12.76 |
| Train/mean_episode_length | ~460 |
| Episode_Reward/track_lin_vel_xy | 0.41 |
| Episode_Reward/gait | **0.00** |
| Episode_Reward/feet_clearance | 0.41 |
| Loss/value_function | 0.068 |

### Issue Found

**Gait reward stuck at 0.0.** Root cause: the `feet_gait` reward function gates output to zero when `cmd_norm <= 0.1`. With only `lin_vel_x` varying in `(-0.1, 0.1)` and y/yaw at zero, the command norm was almost always below the 0.1 threshold.

### Resolution

Added `lin_vel_y=(-0.1, 0.1)` and `ang_vel_z=(-0.1, 0.1)` to initial command ranges so the 3-axis norm can exceed the 0.1 gating threshold.

---

## Run 2

| Field | Value |
|-------|-------|
| Run ID | `2026-03-30_21-20-41` |
| Iterations | 13258 / 50000 |
| Duration | ~4.5 hours |
| Final mean reward | 51.71 |
| Final episode length | 1000 (max) |
| Outcome | **Crashed** — `RuntimeError: normal expects all elements of std >= 0.0` |

### Config Changes from Run 1

- Commands: `lin_vel_y=(-0.1, 0.1)`, `ang_vel_z=(-0.1, 0.1)` added

### Training Progression

| Iter | Mean Reward | Episode Length | Gait | Terrain Level | Track Lin Vel |
|------|-------------|---------------|------|---------------|---------------|
| 0 | -0.04 | 12 | 0.00 | 0.0 | 0.00 |
| ~900 | 11.19 | 460 | 0.00 | — | 0.41 |
| ~2500 | ~40 | ~950 | ~0.50 | ~5.0 | ~0.82 |
| ~3600 | ~46 | 1000 | ~0.59 | ~5.3 | ~0.86 |
| ~5000 | ~49 | 1000 | ~0.65 | ~5.3 | ~0.87 |
| ~10000 | ~51 | 1000 | ~0.70 | ~5.4 | ~0.87 |
| 13258 | 51.71 | 1000 | 0.69 | 5.36 | 0.87 |

### Key Metrics at Crash

| Metric | Value |
|--------|-------|
| Train/mean_reward | 51.71 |
| Train/mean_episode_length | 1000 |
| Episode_Reward/track_lin_vel_xy | 0.87 |
| Episode_Reward/track_ang_vel_z | 0.45 |
| Episode_Reward/gait | 0.69 |
| Episode_Reward/feet_clearance | 0.95 |
| Episode_Reward/action_rate | -0.29 |
| Episode_Reward/joint_deviation_legs | -0.11 |
| Curriculum/terrain_levels | 5.36 |
| Curriculum/lin_vel_cmd_levels | 1.00 |
| Episode_Termination/time_out | 0.997 |
| Episode_Termination/bad_orientation | 0.003 |
| Loss/value_function | 1.8e+21 (exploded) |

### Best Checkpoint

`model_12600.pt` — reward=51.78, gait=0.724

### Issues Found

#### Issue 1: Value Function Loss Explosion (Fatal)

The value function loss jumped from 0.007 to 1.8e+21 in a single iteration at step 13258.

**Crash chain:**
1. Rollout batch produced unusually large GAE returns
2. `value_loss` overflowed fp32 (0.007 → 1.8e+21)
3. Adam optimizer's second moment estimate corrupted
4. Actor noise std pushed negative
5. `Normal(mean, std<0)` → crash

**Not a one-off:** A prior recoverable spike to 15,000,000 occurred at iter 6886 and self-recovered. The instability was recurring.

#### Issue 2: Reward Plateau

Reward plateaued at ~51 since iter ~3600 (9,700 iterations before crash). Terrain level stuck at 5.3. Velocity command curriculum maxed out at iter ~2500.

**Root cause:** Terrain is 100% flat — the terrain curriculum levels are meaningless since every level is flat. The robot mastered flat walking by iter ~3600 and had nothing harder to learn.

#### Issue 3: Gait Weight Imbalance

Gait reward weight (0.5) is half of velocity tracking weight (1.0). The robot prioritizes speed over walking quality.

---

## Comparison: PM01 vs G1 at Same Training Step (~900)

| Metric (step 900) | G1 29-DOF | PM01 12-DOF |
|---|---|---|
| Mean reward | 6.17 | **11.19** |
| Episode length | 419 | **460** |
| Track lin vel xy | 0.36 | **0.41** |
| Track ang vel z | 0.06 | **0.17** |
| Feet clearance | 0.41 | **0.44** |
| Gait | 0.11 | 0.00 |
| Total per-step reward | 0.32 | **0.56** |

PM01 learned faster than G1 at the same step count, likely due to fewer joints (12 vs 29) and fewer penalty terms. The only area G1 led was gait (0.11 vs 0.0), which was caused by the command gating issue in PM01 (fixed in Run 2).

---

## Files Changed in v0

| File | Change |
|------|--------|
| `unitree_rl_lab.sh` | Switched from conda to python venv |
| `assets/robots/unitree.py` | Set UNITREE_MODEL_DIR and UNITREE_ROS_DIR paths |
| `assets/robots/pm01.py` | **New** — PM01 robot asset config |
| `tasks/locomotion/robots/pm01/__init__.py` | **New** — empty package init |
| `tasks/locomotion/robots/pm01/12dof/__init__.py` | **New** — gym registration |
| `tasks/locomotion/robots/pm01/12dof/velocity_env_cfg.py` | **New** — task environment config |
| `unitree_model/PM01/12dof/` | **New** — URDF and mesh files |

---

## Lessons Learned

1. **Always check reward gating thresholds.** The `feet_gait` function gates reward to zero when command norm is below 0.1. Initial command ranges must produce norms that exceed this threshold.

2. **100% flat terrain makes terrain curriculum meaningless.** Always include diverse sub-terrains if using terrain curriculum.

3. **fp32 value loss overflow is a real risk** for robots with fewer joints/penalty terms (higher per-step reward). Monitor `Loss/value_function` for spikes > 1000.

4. **Reward plateau diagnosis:** If reward stops improving, check whether curriculum has maxed out and whether terrain actually varies across difficulty levels.

5. **PM01 trains faster than G1 per iteration** due to simpler action space (12 vs 29 DOF), but needs different PPO hyperparameters (lower LR, lower gamma) for stability.
