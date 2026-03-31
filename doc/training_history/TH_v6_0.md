# Training History v6 — Run 6: Log Std + Venv Patches Death Spiral (Failed)

**Robot:** EngineAI PM01 (12-DOF, legs only)
**Task:** `Unitree-PM01-12dof-Velocity`
**Date:** 2026-03-31
**Branch:** `pm01-training`
**Training Plan:** `TP_v6_0.md` (early version with venv patches)

---

## Run Summary

| Field | Value |
|-------|-------|
| Run ID | (during TP_v6 development, before clean approach) |
| Iterations | ~3356 / 50000 |
| Duration | ~1 hour |
| Final mean reward | -3.7e28 (diverged) |
| Final episode length | 228 |
| Outcome | **Crashed** — `RuntimeError: normal expects all elements of std >= 0.0` |

## Config

- PPO: `PM01PPORunnerCfg` (lr=3e-4, gamma=0.98, vlc=0.5, empirical_normalization=True)
- `noise_std_type="log"`
- Terrain: diverse (flat/rough/slopes/stairs)
- Venv patches: returns clamp ±100, value_loss skip >100, std clamp min=0.01

## Why It Failed

The three venv patches interfered with each other:
1. Returns clamp + value loss skip caused ALL gradient updates to be skipped (losses showed 0.0000)
2. With no gradient updates, the policy drifted randomly
3. Actor weights diverged → actions = inf → `action_rate: -inf` → crash

**Key evidence:** `Mean value_function loss: 0.0000`, `Mean surrogate loss: 0.0000` — zero learning was happening.

## Resolution

Removed ALL venv patches. Adopted clean approach: config-only changes (`noise_std_type="log"`).
