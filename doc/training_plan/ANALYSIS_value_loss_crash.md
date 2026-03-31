# Analysis: Why PM01 Crashes But G1 Doesn't

**Date:** 2026-03-31
**Analyzed by:** Claude Opus 4.6
**Runs analyzed:** G1 (5916 iters), PM01 Run 2 (13258 iters), PM01 Run 3 (7051 iters)

---

## Summary

PM01 has a **structural reward imbalance** that makes it fundamentally more prone to value loss overflow than G1. This is not a hyperparameter problem — it's a reward design problem. The fix requires either clamping returns at the computation level or rebalancing PM01's reward terms.

---

## Evidence: Spike Frequency

| Metric | G1 (5916 iters) | PM01 Run 2 (13258 iters) | PM01 Run 3 (7051 iters) |
|---|---|---|---|
| Value loss spikes > 1.0 | **0** (0.0%) | 114 (1.1%) | 120 (1.7%) |
| Spikes > 10 | **0** | 93 | 105 |
| Spikes > 100 | **0** | 77 | 92 |
| Spikes > 1K | **0** | 65 | 82 |
| Spikes > 1M | **0** | 39 | 49 |
| Max value loss | **0.43** | 1.8e+21 | inf |
| Median value loss | 0.015 | 0.009 | 0.035 |

G1 has **zero** spikes above 1.0 across its entire training. PM01 has over 100 in every run. This is a structural difference, not random bad luck.

---

## Root Cause: PM01's Per-Step Reward is 2x Higher Than G1

### Reward Breakdown at Step ~3000

#### G1 (19 terms, net per-step: +1.08)

| Term | Value | Type |
|------|-------|------|
| track_lin_vel_xy | +0.61 | positive |
| track_ang_vel_z | +0.15 | positive |
| alive | +0.12 | positive |
| feet_clearance | +0.78 | positive |
| gait | +0.41 | positive |
| action_rate | -0.44 | penalty |
| joint_vel | **-0.13** | penalty (29 joints) |
| joint_deviation_arms | **-0.11** | penalty (PM01 doesn't have) |
| joint_deviation_legs | -0.08 | penalty |
| joint_deviation_waists | **-0.06** | penalty (PM01 doesn't have) |
| joint_acc | -0.05 | penalty |
| base_angular_velocity | -0.04 | penalty |
| feet_slide | -0.03 | penalty |
| flat_orientation_l2 | -0.01 | penalty |
| Others | -0.02 | penalty |
| **TOTAL** | **+1.08** | |

#### PM01 Run 2 (17 terms, net per-step: +2.24)

| Term | Value | Type |
|------|-------|------|
| track_lin_vel_xy | **+0.84** | positive (higher than G1) |
| track_ang_vel_z | **+0.42** | positive (higher than G1) |
| alive | +0.15 | positive |
| feet_clearance | **+0.93** | positive (higher than G1) |
| gait | **+0.55** | positive (higher than G1) |
| action_rate | -0.35 | penalty |
| joint_deviation_legs | -0.14 | penalty |
| joint_vel | **-0.03** | penalty (12 joints vs G1's 29) |
| feet_slide | -0.04 | penalty |
| base_linear_velocity | -0.03 | penalty |
| joint_acc | -0.03 | penalty |
| Others | -0.04 | penalty |
| **TOTAL** | **+2.24** | |

### Why PM01's Reward is Higher

1. **Missing 2 penalty terms** that G1 has:
   - `joint_deviation_arms` (-0.11 for G1) — PM01 has no arms
   - `joint_deviation_waists` (-0.06 for G1) — PM01 has no waist
   - Combined: G1 pays **-0.17 per step** that PM01 doesn't

2. **Lower joint penalties** due to fewer joints (12 vs 29):
   - `joint_vel`: G1 = -0.13, PM01 = -0.03 (4x less)
   - `joint_acc`: G1 = -0.05, PM01 = -0.03

3. **Higher positive rewards** — PM01's simpler body makes it easier to:
   - Track velocity: 0.84 vs 0.61 (+0.23)
   - Track angular velocity: 0.42 vs 0.15 (+0.27)
   - Feet clearance: 0.93 vs 0.78 (+0.15)
   - Gait: 0.55 vs 0.41 (+0.14)

4. **Positive/negative ratio:**
   - G1: 1.47 (balanced)
   - PM01: 1.90 (unbalanced toward positive)
   - PM01 Run 3: 2.46 (even more unbalanced with gait weight=1.0)

---

## How Higher Reward Causes Instability

### Discounted Return Magnitude

With `gamma=0.99` and episode length 1000:
- **G1:** per-step reward ~1.08 → discounted return ≈ 21
- **PM01:** per-step reward ~2.24 → discounted return ≈ 45

### The Squared Error Amplification

Value loss = `(return - predicted_value)^2`

When a terrain spike or lucky episode produces a return outlier:

| Scenario | Return | Prediction | Loss |
|----------|--------|------------|------|
| G1 normal | 21 | 20 | 1 |
| G1 outlier | 50 | 20 | 900 |
| PM01 normal | 45 | 44 | 1 |
| PM01 outlier | 100 | 44 | **3,136** |
| PM01 extreme outlier | 200 | 44 | **24,336** |

PM01's higher baseline means outliers produce much larger squared errors. Once a large loss enters the Adam optimizer's second moment estimate, it can corrupt the optimizer state and cascade into further instability.

### Why G1 Never Spikes

G1's reward is better balanced:
- More penalty terms act as "dampers" on reward variance
- Lower positive/negative ratio means fewer extreme positive outliers
- Joint penalties scale with DOF count (29 joints), providing natural regularization

---

## Would G1 Eventually Crash?

**Unlikely with current config.** G1's maximum value loss after 5916 iterations was 0.43 — three orders of magnitude below PM01's first spike. The reward structure is fundamentally more stable due to:
- More penalty terms providing natural regularization
- Lower per-step reward magnitude
- Better positive/negative balance

If G1 were trained for 50000+ iterations at higher reward levels (e.g., after full terrain mastery), it's theoretically possible but significantly less likely.

---

## Fix Options

### Option A: Clamp Returns (Recommended — fastest, most robust)

Clamp GAE returns to `[-1000, 1000]` before they enter the value loss computation. This is a safety net that doesn't affect normal training (normal returns are ~20-50) but prevents the fp32 overflow.

See `TP_v3_0.md` for implementation details.

**Pros:** Simple, doesn't change reward structure, prevents crash regardless of cause
**Cons:** May mask underlying reward issues, could clip legitimate high-value trajectories (unlikely in practice)

### Option B: Rebalance PM01 Rewards (Recommended for long-term)

Bring PM01's positive/negative ratio closer to G1's 1.47.

**Approach 1 — Add penalty terms:**
```python
# Ankle joint deviation — keep ankles near default
joint_deviation_ankles = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot",
        joint_names=["j04_ankle_pitch_l", "j10_ankle_pitch_r",
                     "j05_ankle_roll_l", "j11_ankle_roll_r"],
    )},
)
```

**Approach 2 — Reduce positive reward weights:**
```python
track_lin_vel_xy = RewTerm(weight=0.8, ...)   # was 1.0
track_ang_vel_z = RewTerm(weight=0.4, ...)    # was 0.5
```

**Approach 3 — Increase existing penalties:**
```python
joint_vel = RewTerm(weight=-0.005, ...)       # was -0.001
energy = RewTerm(weight=-5e-5, ...)           # was -2e-5
```

**Pros:** Addresses root cause, makes training fundamentally more stable
**Cons:** Requires experimentation to find right balance, may slow learning

### Option C: Both A and B

Apply returns clamping as a safety net (Option A) while also rebalancing rewards (Option B). This is the most robust long-term approach.

---

## Recommendation

1. **Immediate (Run 4):** Apply returns clamping (Option A) to unblock training
2. **Next iteration (Run 5+):** Experiment with reward rebalancing (Option B) to reduce reliance on clamping

The returns clamp is a safety net, not a permanent fix. The reward imbalance should be addressed for long-term training quality.
