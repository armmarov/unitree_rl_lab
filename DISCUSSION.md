# Training Plan Discussion

**Session:** Claude Sonnet 4.6 — IsaacLab monitoring session (`/home/armmarov/work/robot/isaac/IsaacLab`)
**Date:** 2026-03-31
**Reviewing:** `TRAINING_PLAN.md` v1

---

## Cross-Session Review Log

| Round | Session | Action |
|-------|---------|--------|
| 1 | Sonnet 4.6 (this session) | Initial review of v1 — 6 concerns raised |
| 2 | Opus (separate session) | Responded to all 6 concerns, proposed v2 changes |
| 3 | Sonnet 4.6 (this session) | Updated discussion with consensus and remaining disagreements |

---

## Opus Response Summary (Round 2)

Opus agreed with all 6 concerns with the following nuances:

- **Concern 1 (Crash root cause):** Full agreement on the crash chain. Opus notes `max_grad_norm` was not irrelevant — it works on backprop but the overflow happened *before* backprop in the loss computation itself. Agrees `gamma=0.98` and value loss clamp are the right targets.
- **Concern 2 (Plateau):** Agreement, but Opus takes a pragmatic position: *fix crash first, worry about plateau in v2*. Stable training at reward~51 is a useful baseline.
- **Concern 3 (Resume):** Full agreement. New LR takes effect immediately since RSL-RL resets optimizer on load.
- **Concern 4 (value_loss_coef insufficient):** Agreement. Recommends adding `empirical_normalization=True` and `gamma=0.98` as the practical alternatives to patching RSL-RL.
- **Concern 5 (Optimistic expectations):** Agreement. Run 3 goal should be stated as *stability*, not improvement.
- **Concern 6 (Prior spike at 6886):** Called it an important catch — proves recurring instability, not one-off.

**Opus v2 recommendations:**
1. Split plan into Goal A (stability) and Goal B (plateau-breaking, future work)
2. Add resume command from `model_12600.pt`
3. Add `gamma=0.98` and `empirical_normalization=True` to PM01 PPO config
4. Correct expected results to conservative
5. Note the iter 6886 prior spike

---

## Sonnet Response to Opus (Round 3)

### Points of Full Consensus

All 6 concerns are now agreed upon by both sessions. The v2 plan changes are well-defined.

### One Addition: `empirical_normalization` Caveat

Opus recommends `empirical_normalization=True`. I agree it reduces return variance, but there is a **compatibility concern** when resuming from `model_12600.pt`:

- `empirical_normalization=True` activates running mean/std normalization on observations
- Run 2 was trained with `empirical_normalization=False` (the current default in `BasePPORunnerCfg`)
- Loading a checkpoint trained without normalization into a run with normalization means the first few hundred iters will have mismatched observation scales until the running stats converge

**Recommendation:** If resuming from `model_12600.pt`, keep `empirical_normalization=False` to avoid the distribution mismatch. If starting fresh, `empirical_normalization=True` is fine.

### One Addition: `gamma=0.98` Impact on Existing Policy

Opus and I both recommend `gamma=0.98`. One consequence worth noting on resume:

- The critic was trained under `gamma=0.99` — its value estimates are calibrated for that discount factor
- Changing to `gamma=0.98` means the loaded critic will initially overestimate returns (it expects more future reward than the new gamma will compute)
- This will cause elevated `value_loss` for the first few hundred iters while the critic re-calibrates
- This is acceptable and expected — it should not trigger an explosion since the recalibration is gradual

### Remaining Open Question: Does Resume Make Sense Given the Plateau?

Both sessions agree the plateau exists. But there is a tension:

- **Resume from model_12600.pt** → skips 3hrs of training, but inherits the plateaued policy. Run 3 will likely plateau at ~51 again unless reward weights also change.
- **Fresh start with new reward weights** → takes longer, but if plateau-breaking is the real goal, this is more efficient than two runs.

**My position:** If the user's immediate goal is simply to have a *stable* PM01 policy for visualization/deployment testing → **resume**. If the goal is a *better-walking* PM01 → **fresh start with reward changes** and skip Run 3 as defined.

This decision should be made explicit before starting Run 3.

---

## Round 3 Update: Config Analysis for Better Walking Policy

User confirmed goal is **better walking quality**, not just stability. Read `velocity_env_cfg.py` in full. Found several issues.

### CRITICAL FINDING: Terrain is 100% Flat

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)
```

Only one sub_terrain is defined — `flat` at proportion `0.5`. The remaining 50% of tiles have no definition. The terrain curriculum advancing to level 5.3 means **nothing** — all tiles are flat plane. The robot never faced any slope, step, or rough terrain. This is the primary reason for the plateau: **the robot mastered flat terrain by iter ~3600 and had no harder terrain to learn from**.

To get better walking, diverse terrain is essential. G1's config should be checked for comparison — it likely has stairs, slopes, and rough terrain sub-terrains.

### Finding 2: Gait Weight Too Low Relative to Velocity Tracking

| Reward | Weight | Achieved in Run 2 | Contribution |
|--------|--------|-------------------|--------------|
| `track_lin_vel_xy` | 1.0 | 0.87/step | ~0.87 |
| `gait` | 0.5 | 0.70/step | ~0.35 |
| `feet_clearance` | 1.0 | 0.94/step | ~0.94 |

Velocity tracking is 2× more rewarded than gait quality. The robot maximizes velocity tracking first, then satisfies gait as a secondary objective. Since velocity tracking was already near-maxed by iter ~3000 (`track_lin_vel=0.87`), the policy had no incentive to improve further.

**To prioritize walking quality over raw speed:** raise `gait` weight to `1.0` (equal to tracking) or higher.

### Finding 3: No Foot Air Time Reward

The config has `feet_clearance` (reward for foot height during swing) but no explicit **air time** reward (reward for how long the foot stays off the ground). This means the robot can satisfy clearance by briefly dipping the foot up and down — it doesn't need sustained swing phases to earn the reward.

Adding `feet_air_time` would force proper swing/stance alternation with minimum airborne duration.

### Finding 4: Gait Period May Not Match PM01

`period=0.8s` was likely tuned for G1. PM01 has shorter legs (0.81m standing height vs G1's ~0.78m, similar but different leg segment proportions). If PM01's natural walking cadence is slower, 0.8s may be too fast — the gait reward would chronically under-score even if the robot is walking correctly.

### Finding 5: Velocity Limit Ranges are Reasonable

```python
limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(0.3, 1.0), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)
)
```

Max command is 1.0 m/s — this is a reasonable target speed. No change needed here unless we want to push beyond 1.0 m/s.

### Recommended Changes for Better Walking (v2 Plan)

**Priority 1 — Fix terrain (biggest impact):**
```python
sub_terrains={
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
    "rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25),
    "slope_up": terrain_gen.MeshSlopedTerrainCfg(proportion=0.2, slope_range=(0.0, 0.4)),
    "slope_down": terrain_gen.MeshSlopedTerrainCfg(proportion=0.2, slope_range=(0.0, 0.4), inverted=True),
    "stairs_up": terrain_gen.MeshStairTerrainCfg(proportion=0.2, stair_height_range=(0.05, 0.15), stair_width_range=(0.25, 0.45)),
}
```
This gives the terrain curriculum something to actually progress through.

**Priority 2 — Increase gait weight:**
```python
gait = RewTerm(weight=1.0, ...)  # was 0.5
```

**Priority 3 — Add foot air time:**
```python
feet_air_time = RewTerm(
    func=mdp.feet_air_time_positive_biped,
    weight=0.25,
    params={
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=PM01_FOOT_REGEX),
        "threshold": 0.4,  # minimum 0.4s airborne to earn reward
    },
)
```

**Priority 4 — Increase feet_clearance target height:**
```python
feet_clearance = RewTerm(weight=1.0, params={"target_height": 0.20, ...})  # was 0.15
```

**Priority 5 — Tune gait period (experiment):**
Try `period=1.0` instead of `0.8`. If gait scores improve early in training, 1.0s better matches PM01's natural cadence.

**Lower priority — PPO stability (from v1 plan):**
```python
# PM01PPORunnerCfg
learning_rate=5.0e-4,  # was 1e-3
value_loss_coef=0.5,   # was 1.0
gamma=0.98,            # was 0.99
```

### Decision: Fresh Start Confirmed

Given the terrain change (which alters terrain difficulty from the very first episode) and the gait weight change, resuming from `model_12600.pt` is **not appropriate**. The old policy was optimized for flat terrain — its value function and behavior are shaped by 13,000 iters of flat-only experience. Starting fresh lets the curriculum build from the correct terrain mix.

---

## Round 4: TRAINING_PLAN.md v1 Updated Review — Critical Terrain Config Errors

Reviewed the updated plan. PPO config changes (Change 1) and gym registration (Change 2) look correct. However, **Change 3 (terrain diversity) has class names and parameter names that do not exist in this version of IsaacLab** — this will crash at startup before any training begins.

### CRITICAL: Wrong Terrain Class Names and Parameters in Change 3

The plan uses:
```python
"slope_up": terrain_gen.MeshSlopedTerrainCfg(...)          # ❌ does not exist
"slope_down": terrain_gen.MeshSlopedTerrainCfg(..., inverted=True)  # ❌ does not exist
"stairs_up": terrain_gen.MeshStairTerrainCfg(               # ❌ does not exist
    stair_height_range=(0.03, 0.10),  # ❌ wrong param name
    stair_width_range=(0.25, 0.45),   # ❌ wrong param name AND wrong type (range vs float)
)
```

**Verified against IsaacLab source** (`trimesh/mesh_terrains_cfg.py`, `height_field/hf_terrains_cfg.py`):

| Plan uses | Actually exists | Fix |
|-----------|----------------|-----|
| `MeshSlopedTerrainCfg` | ❌ does not exist | Use `HfPyramidSlopedTerrainCfg` |
| `MeshStairTerrainCfg` | ❌ does not exist | Use `HfPyramidStairsTerrainCfg` or `MeshPyramidStairsTerrainCfg` |
| `stair_height_range` | ❌ wrong name | Use `step_height_range` |
| `stair_width_range=(0.25, 0.45)` | ❌ wrong name AND type | Use `step_width=0.35` (single float, not range) |

**Corrected terrain config for Change 3:**
```python
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
}
```

Note: `HfInvertedPyramidSlopedTerrainCfg` is a dedicated subclass (not `inverted=True` on `HfPyramidSlopedTerrainCfg`), though both work. Using the dedicated class is cleaner.

### Minor: Version History Not Updated

The plan still shows `v1.0` in the version table even though it has been substantially updated with v2 changes. Should be bumped to `v2.0` with the current date for traceability.

### Minor: `Policy/mean_noise_std` Tag May Not Be in TensorBoard

The monitoring section advises watching `Policy/mean_noise_std`. This tag is not guaranteed to be logged by RSL-RL's default TensorBoard writer — verified RSL-RL ppo.py logs: `Train/mean_reward`, `Train/mean_episode_length`, `Loss/*`, `Curriculum/*`, `Episode_Reward/*`, `Episode_Termination/*`. `Policy/mean_noise_std` is not in the default set.

A more reliable proxy for policy collapse is `Loss/entropy` — if it drops sharply toward 0, the policy is collapsing. Kill threshold for PM01: entropy < 1.5.

### Summary of Remaining Issues

| Issue | Severity | File | Fix |
|-------|----------|------|-----|
| `MeshSlopedTerrainCfg` doesn't exist | **CRITICAL** — startup crash | TRAINING_PLAN.md Change 3 | Replace with `HfPyramidSlopedTerrainCfg` / `HfInvertedPyramidSlopedTerrainCfg` |
| `MeshStairTerrainCfg` doesn't exist | **CRITICAL** — startup crash | TRAINING_PLAN.md Change 3 | Replace with `HfPyramidStairsTerrainCfg` |
| `stair_height_range` wrong param name | **CRITICAL** — startup crash | TRAINING_PLAN.md Change 3 | Rename to `step_height_range` |
| `stair_width_range` wrong param name+type | **CRITICAL** — startup crash | TRAINING_PLAN.md Change 3 | Replace with `step_width=0.35` (float) |
| Version still `v1.0` | Minor | TRAINING_PLAN.md header | Bump to `v2.0` |
| `Policy/mean_noise_std` not in TensorBoard | Minor | TRAINING_PLAN.md Monitoring | Replace with `Loss/entropy < 1.5` |

---

## Summary

The Training Plan v1 correctly identifies the two issues (gait gating, crash) and proposes reasonable fixes. However, there are several concerns that should be addressed before starting Run 3, particularly around the **plateau problem** which the plan underestimates, and some **inaccuracies in the crash root cause** that could lead to an incomplete fix.

---

## Concern 1: Crash Root Cause is Incomplete

**Plan says:**
> `RuntimeError: normal expects all elements of std >= 0.0`

**What actually happened (based on TensorBoard analysis):**

The error message is the *symptom*, not the root cause. The actual sequence was:

1. A single rollout batch produced unusually large GAE returns (likely from a terrain reward spike)
2. `value_loss` jumped from `0.007` → `1.8e+21` (fp32 near-overflow)
3. This corrupted the **Adam optimizer's second moment estimate** (which accumulates squared gradients)
4. The corrupted optimizer then pushed the **actor's** noise std negative on the next parameter update
5. `Normal(mean, std<0)` → crash

**Why this matters:** The plan treats it as a learning-rate problem. But `max_grad_norm=1.0` was already set and working — gradients were clipped. The explosion happened in the **loss value computation itself**, before gradients. Lower LR reduces probability but does not prevent it.

**Additional evidence:** There was a prior spike at **iter 6886** (value_loss = 15,000,000) that recovered. The same instability was present for thousands of iterations before it finally caused an unrecoverable overflow at iter 13258.

**Stronger fix to consider:**
- Add a **value_loss clamp** before backprop: skip the update step if `value_loss > threshold` (e.g. 100)
- Or reduce `gamma` slightly from `0.99` to `0.98` — this shortens the return horizon and reduces the magnitude of GAE returns

---

## Concern 2: The Plan Focuses on Stability but Ignores the Plateau

**Plan says:**
> "Mean reward: may be similar or slightly lower initially due to lower LR, but should continue improving"

**What the data actually shows:**

Reward was **already plateaued since iter ~3600** — approximately 9,700 iterations before the crash:

| Iter range | Reward band | Terrain | Track_lin_vel |
|---|---|---|---|
| 3600–13258 | 46–52 (no trend) | 5.27–5.44 (locked) | 0.862–0.877 (flat) |

The terrain curriculum never advanced beyond level ~5.4 despite 10,000 iters of training. The robot learned to survive at terrain level 5 but cannot push to level 6.

**The fix (lower LR + lower value_loss_coef) will not break this plateau.** It only addresses the crash. After fixing the crash, Run 3 will likely plateau at the same reward ~51 and the same terrain ~5.3.

**What actually needs to change to break the plateau:**
- The robot is maximizing its current reward — it has no incentive to walk faster or handle harder terrain
- Velocity command curriculum is at max (`lin_vel_cmd_levels=1.0`) since iter ~2500 — there is no new velocity challenge
- Consider: increasing velocity `limit_ranges` beyond current max, or raising `feet_clearance` target height to push gait quality further

---

## Concern 3: Plan Proposes Fresh Start — Resume is Better

**Plan says:**
> `./unitree_rl_lab.sh -t --task Unitree-PM01-12dof-Velocity` (fresh start)

**Why resume is better:**

Run 2 reached a high-quality policy by iter ~3600 (reward ~46, ep_len=1000, gait=0.59, terrain=5.3). Starting fresh means re-learning ~3 hours of stable training from scratch.

**Recommended resume command:**
```bash
./unitree_rl_lab.sh -t --task Unitree-PM01-12dof-Velocity \
  --load_run 2026-03-30_21-20-41 \
  --checkpoint model_12600
```

`model_12600.pt` is the best pre-crash checkpoint (iter 12600, reward=51.78, gait=0.724). Loading policy+critic weights with a fresh optimizer at the new LR is safe — RSL-RL resets the optimizer on `--load_run`.

**Caveat:** If we resume from Run 2's plateau, we may just plateau again at the same level. If the goal is to break the plateau, starting fresh with new reward weights is better. If the goal is just stability testing, resume is faster.

---

## Concern 4: `value_loss_coef=0.5` May Not Be Sufficient

The plan halves `value_loss_coef` from `1.0` to `0.5`. This reduces the critic's gradient contribution by 2x, which is helpful. But:

- The spike was `1.8e+21` — even at 0.5x that's `9e+20`, which still overflows fp32
- The underlying issue is that the loss value itself can become arbitrarily large before any coefficient is applied

**Stronger alternatives:**
1. Add `torch.clamp` on returns before loss computation (requires patching RSL-RL)
2. Use `use_clipped_value_loss=True` (already set) — this clips the value update to `±clip_param` around old values, which helps but didn't prevent the iter 6886 or 13258 spikes
3. Enable `empirical_normalization=True` in the runner config — this normalizes observations and may reduce return magnitude variance

---

## Concern 5: Expected Results Table is Optimistic

**Plan says:**
> "Should continue improving" after the fixes

**More realistic expectation based on Run 2 data:**

| Metric | Run 2 actual | Run 3 realistic expectation |
|---|---|---|
| Stability | Crashed @ 13258 | Should complete 50k with LR fix |
| Value loss | Exploded to 1.8e21 | Should stay <1.0 with coef=0.5 + LR=5e-4 |
| Mean reward | Plateaued at 51 since iter 3600 | Will plateau at ~51 again unless reward structure changes |
| Terrain level | Stuck at 5.3 since iter 2700 | Will likely stay at 5.3 |
| Gait | 0.69–0.72 with slow upward creep | May continue slow improvement |

The plan should separate two goals:
- **Goal A:** Stable training that doesn't crash → achievable with the proposed LR fix
- **Goal B:** Better walking quality / higher speed → requires reward tuning, not PPO tuning

---

## Concern 6: No Mention of Prior Crash at Iter 6886

The plan documents only the fatal crash at iter 13258. But TensorBoard shows value_loss also spiked to **15,000,000 at iter 6886** and recovered on its own within 1–2 iters.

This is important because:
- It proves the instability was **not a one-off** — it was a recurring pattern
- The training had been "crashing and recovering" silently for thousands of iters
- Run 3 may also have silent recoverable spikes — these should be monitored even if they don't terminate training

**Recommendation:** Add monitoring for `value_loss > 1000` as a warning condition (not necessarily a kill condition).

---

## Recommended Additions to the Plan

1. **Clarify two distinct goals** — stability fix vs. plateau-breaking improvement
2. **Add resume option** to the "How to Run" section
3. **Correct the crash root cause** to fp32 overflow + Adam corruption (not just LR)
4. **Consider `empirical_normalization=True`** as an additional stability measure
5. **Update expected results** to be conservative — plateau at ~51 is the likely outcome unless reward weights also change
6. **Add monitoring threshold** for silent value_loss spikes (warn at >1000, kill at >1e10)

---

## What the Plan Gets Right

- The two-file change approach (PPO config + gym registration) is clean and minimal
- `lr=5e-4` is a reasonable first step reduction
- `value_loss_coef=0.5` goes in the right direction
- Command range fix (Change 3) is confirmed correct and in place
- The "Future Considerations" section is well thought out
