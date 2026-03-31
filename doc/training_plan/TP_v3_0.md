# PM01 Training Plan

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| v3.0 | 2026-03-31 | armmarov | Training plan for Run 3: PPO stability + terrain diversity + gait weight |

---

## Current Status

### Training Runs Completed

| Run | Config | Iterations | Mean Reward | Outcome |
|-----|--------|-----------|-------------|---------|
| Run 1 (`2026-03-30_20-47-12`) | Initial config, `lin_vel_x=(-0.1,0.1)` only | ~1510 | 12.76 | Stopped manually. Gait reward stuck at 0.0 due to command gating. |
| Run 2 (`2026-03-30_21-20-41`) | Fixed commands with y/yaw ranges | 13258/50000 | 51.71 | **Crashed.** Value function loss exploded from 0.007 to 1.8e21. |

### Issue 1: Gait Reward Gating (RESOLVED in Run 2)

The `feet_gait` reward function gates output to zero when `cmd_norm <= 0.1`. Run 1 had `lin_vel_y=(0,0)` and `ang_vel_z=(0,0)`, so command norm was almost always below threshold.

**Fix applied in Run 2:** Added `lin_vel_y=(-0.1, 0.1)` and `ang_vel_z=(-0.1, 0.1)` to initial ranges. Gait reward rose to 0.69 by iteration 13258.

### Issue 2: Training Crash at Iteration 13258 (OPEN)

Run 2 crashed with `RuntimeError: normal expects all elements of std >= 0.0`.

**Crash chain (root cause analysis):**
1. A rollout batch produced unusually large GAE returns (likely from a terrain reward spike)
2. `value_loss` jumped from `0.007` → `1.8e+21` (fp32 near-overflow) — this happened in the loss computation itself, before gradients
3. The overflow corrupted Adam optimizer's second moment estimate (which accumulates squared gradients)
4. The corrupted optimizer pushed the actor's noise std negative on the next parameter update
5. `Normal(mean, std<0)` → crash

**This was not a one-off.** TensorBoard shows a prior recoverable spike at **iter 6886** (value_loss = 15,000,000) that self-recovered within 1-2 iterations. The instability was recurring for thousands of iterations before the fatal overflow at iter 13258.

**Why `max_grad_norm=1.0` didn't help:** Gradient clipping operates on the backward pass, but the overflow happened in the loss value computation itself (forward pass). By the time gradients are computed, the damage is already done.

### Issue 3: Reward Plateau — Root Cause Identified

Reward plateaued at ~51 since **iter ~3600** — approximately 9,700 iterations before the crash, with no upward trend:

| Iter range | Mean reward | Terrain level | Track_lin_vel |
|---|---|---|---|
| 3600–13258 | 46–52 (flat) | 5.27–5.44 (locked) | 0.862–0.877 (flat) |

**Root cause: Terrain is 100% flat.** The terrain config only defines one sub-terrain:
```python
sub_terrains={
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
}
```
The remaining 50% has no definition. The terrain curriculum levels are meaningless — every level is flat. The robot mastered flat walking by iter ~3600 and had nothing harder to learn.

Additionally, gait reward weight (0.5) is half of velocity tracking weight (1.0), so the robot prioritizes speed over walking quality.

---

## Training Plan v3

### Goals

- **Goal A: Stable training** — complete 50000 iterations without crashes
- **Goal B: Break plateau** — improve beyond reward ~51 with terrain diversity and reward tuning

### Phased Approach

To avoid changing too many variables at once, changes are split into two phases:

- **Run 3:** PPO stability fixes + terrain diversity + gait weight increase (highest-impact changes)
- **Run 4 (if needed):** Add feet_air_time reward, adjust feet_clearance target, tune gait period

### Changes Required for Run 3

#### Change 1: Create PM01-specific PPO config

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

**Why:** The shared `BasePPORunnerCfg` uses `learning_rate=1e-3`, `value_loss_coef=1.0`, and `gamma=0.99`. PM01 has fewer joints (12 vs 29) and fewer penalty terms, resulting in higher per-step rewards. The long return horizon (`gamma=0.99`) amplifies GAE return variance, which caused the fp32 overflow. Observation normalization further reduces return variance.

**What to add:**
```python
@configclass
class PM01PPORunnerCfg(BasePPORunnerCfg):
    """PPO config for PM01 with stability improvements."""
    empirical_normalization = True        # was False — normalize observations to reduce return variance
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,              # was 1.0 — dampen critic updates
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,             # was 1e-3 — halve learning rate
        schedule="adaptive",
        gamma=0.98,                       # was 0.99 — shorter return horizon reduces GAE magnitude
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

**Rationale for each change:**
- `learning_rate=5e-4`: Reduces update magnitude, less likely to amplify spikes
- `value_loss_coef=0.5`: Dampens critic gradient contribution
- `gamma=0.98`: Shorter return horizon directly reduces the magnitude of GAE returns that caused the overflow
- `empirical_normalization=True`: Normalizes observations, reducing return variance

#### Change 2: Register PM01 gym env with new config

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/pm01/12dof/__init__.py`

**What to change:**
```python
gym.register(
    id="Unitree-PM01-12dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PM01PPORunnerCfg",  # was BasePPORunnerCfg
    },
)
```

#### Change 3: Add terrain diversity

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/pm01/12dof/velocity_env_cfg.py`

**Why:** The current terrain is 100% flat. The terrain curriculum has nothing to progress through. This is the primary cause of the reward plateau.

**Replace the current `COBBLESTONE_ROAD_CFG` sub_terrains:**

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=1.0,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=1.0,
        ),
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.03, 0.10),
            step_width=0.35,
            platform_width=1.0,
        ),
    },
)
```

**Notes:**
- Flat reduced from 100% to 20% — still present for easy episodes
- Rough terrain adds small random bumps — teaches foot placement
- Slopes teach balance under gravity bias (uses `HfPyramidSlopedTerrainCfg` / `HfInvertedPyramidSlopedTerrainCfg`)
- Stairs are conservative (`0.03–0.10m` step height) — PM01 is a 12-DOF legs-only biped with no arms for balance, so keep stair difficulty low initially
- Stair params use `step_height_range` and `step_width` (float, not range) per IsaacLab API

#### Change 4: Increase gait reward weight

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/pm01/12dof/velocity_env_cfg.py`

**Why:** Gait weight (0.5) is half of velocity tracking weight (1.0). The robot prioritizes speed over walking quality. Equalizing them pushes for proper stepping.

**Change in `RewardsCfg`:**
```python
gait = RewTerm(
    func=mdp.feet_gait,
    weight=1.0,       # was 0.5
    ...
)
```

#### Change 5: Confirm velocity command ranges (already fixed)

**File:** `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/pm01/12dof/velocity_env_cfg.py`

**Verify this is in place:**
```python
ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(-0.1, 0.1),
    lin_vel_y=(-0.1, 0.1),   # must not be (0.0, 0.0)
    ang_vel_z=(-0.1, 0.1),   # must not be (0.0, 0.0)
)
```

---

### Expected Results for Run 3

| Metric | Run 2 (before crash) | Run 3 expected |
|--------|---------------------|----------------|
| Training stability | Crashed at iter 13258 | Should complete 50000 iterations |
| Value function loss | Exploded to 1.8e21 | Should stay below 1.0. Monitor for spikes >1000. |
| Mean reward | Plateaued at ~51 on flat terrain | Likely lower initially (harder terrain), should grow as curriculum progresses |
| Terrain level | Stuck at 5.3 (meaningless — all flat) | Should show real progression as robot learns slopes/stairs |
| Gait reward | 0.69–0.72 | Should be higher with weight=1.0, providing stronger learning signal |
| Episode length | 1000 (max) | May be shorter initially on harder terrain, should recover |

### How to Run

**Fresh start (required — terrain and PPO config both changed):**
```bash
./unitree_rl_lab.sh -t --task Unitree-PM01-12dof-Velocity
```

**Why fresh start:**
- Terrain changed from 100% flat to diverse — old policy is optimized for flat only
- `gamma` changed from 0.99 to 0.98 — old critic value estimates are calibrated for wrong discount factor
- `empirical_normalization=True` builds running stats from scratch — old policy was trained without normalization
- PM01 reaches plateau in ~1.5 hours — retraining cost is low

**Monitor:**
```bash
tensorboard --logdir logs/rsl_rl/unitree_pm01_12dof_velocity/
```

**Play (replace `<run_name>` with the timestamped folder):**
```bash
./unitree_rl_lab.sh -p --task Unitree-PM01-12dof-Velocity --load_run <run_name>
```

### Monitoring

Watch for these during training:
- `Loss/value_function > 1000` — warning, potential instability returning
- `Loss/value_function > 1e10` — critical, likely to crash soon
- `Loss/entropy` dropping sharply toward 0 (kill threshold: < 1.5) — policy collapsing. This is more reliable than `Policy/mean_noise_std` which may not be logged by RSL-RL.
- `Curriculum/terrain_levels` — should now show real progression (was stuck at 5.3 on flat)

### Checkpoints to Evaluate

After training, play these checkpoints to assess walking quality:
- `model_1000.pt` — early learning (should be standing/shuffling)
- `model_5000.pt` — mid training (should be attempting to walk)
- `model_10000.pt` — should have stable gait
- `model_50000.pt` — final policy

---

## Run 4 (If Needed): Fine-Tuning Walking Quality

If Run 3 completes successfully but walking quality still needs improvement, apply these changes and start fresh again:

### Add foot air time reward

**Note:** Verify `mdp.feet_air_time_positive_biped` exists in your IsaacLab version — the function name varies across versions. If not found, search for `feet_air_time` variants: `grep -r "def feet_air_time" /path/to/IsaacLab/source/`.

```python
feet_air_time = RewTerm(
    func=mdp.feet_air_time_positive_biped,
    weight=0.25,
    params={
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=PM01_FOOT_REGEX),
        "threshold": 0.3,    # minimum 0.3s airborne per step
    },
)
```

### Increase feet clearance target

```python
feet_clearance = RewTerm(
    func=mdp.foot_clearance_reward,
    weight=1.0,
    params={
        "std": 0.05,
        "tanh_mult": 2.0,
        "target_height": 0.20,   # was 0.15 — push for higher steps
        "asset_cfg": SceneEntityCfg("robot", body_names=PM01_FOOT_REGEX),
    },
)
```

### Experiment with gait period

Try `period=1.0` instead of `0.8`. If PM01's legs have different proportions than G1, the natural walking cadence may be different. Compare gait reward curves between `0.8` and `1.0` in separate runs.

---

## If Training Still Crashes

- Lower learning rate further to `3e-4` or `1e-4`
- Reduce `num_learning_epochs` from 5 to 3
- Patch RSL-RL to clamp value loss before backprop (skip update if `value_loss > threshold`)

## Deployment Preparation

- Export best checkpoint to ONNX for real robot inference
- Verify `joint_sdk_names` order matches PM01 hardware SDK
- Test with `sim2sim` before deploying to hardware

---

## Pre-Training Sign-Off Checklist

Verified by: Claude Opus 4.6
Date: 2026-03-31

### Change 1: PM01-specific PPO config
- [x] `PM01PPORunnerCfg` class added to `rsl_rl_ppo_cfg.py`
- [x] `empirical_normalization = True`
- [x] `learning_rate = 5.0e-4` (was 1e-3)
- [x] `value_loss_coef = 0.5` (was 1.0)
- [x] `gamma = 0.98` (was 0.99)
- [x] All other PPO params inherited correctly from `BasePPORunnerCfg`

### Change 2: Gym registration updated
- [x] `rsl_rl_cfg_entry_point` points to `PM01PPORunnerCfg` (was `BasePPORunnerCfg`)

### Change 3: Terrain diversity
- [x] `flat` proportion reduced to 0.2 (was 0.5, sole terrain)
- [x] `rough` (`HfRandomUniformTerrainCfg`) added — proportion 0.2
- [x] `slope_up` (`HfPyramidSlopedTerrainCfg`) added — proportion 0.2
- [x] `slope_down` (`HfInvertedPyramidSlopedTerrainCfg`) added — proportion 0.2
- [x] `stairs_up` (`HfPyramidStairsTerrainCfg`) added — proportion 0.2
- [x] Terrain class names verified against IsaacLab source (`hf_terrains_cfg.py`)
- [x] Stair params verified: `step_height_range` (tuple), `step_width` (float), `platform_width` (float)

### Change 4: Gait reward weight
- [x] `gait` weight changed to `1.0` (was 0.5)

### Change 5: Velocity command ranges
- [x] `lin_vel_x=(-0.1, 0.1)` — present
- [x] `lin_vel_y=(-0.1, 0.1)` — present (was 0.0, 0.0 in Run 1)
- [x] `ang_vel_z=(-0.1, 0.1)` — present (was 0.0, 0.0 in Run 1)

### Other verifications
- [x] `pm01.py` uses `UnitreeArticulationCfg` with `joint_sdk_names`
- [x] `pm01.py` uses `UnitreeUrdfFileCfg` (not raw `UrdfFileCfg`)
- [x] `pm01.py` uses `UNITREE_MODEL_DIR` for path resolution
- [x] Termination `minimum_height=0.3` (lowered from 0.5)
- [x] Undesired contacts regex `(?!.*ankle.*)` matches G1 pattern
- [x] `RobotPlayEnvCfg` uses `limit_ranges` (not hardcoded velocity)

### Status: APPROVED for Run 3
