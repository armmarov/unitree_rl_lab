# PM01 12-DOF Integration - Code Review

Reviewed: 2026-03-30
Status: Needs changes before training

---

## Files Reviewed

| File | Path |
|------|------|
| Robot asset config | `source/unitree_rl_lab/unitree_rl_lab/assets/robots/pm01.py` |
| Task env config | `source/.../tasks/locomotion/robots/pm01/12dof/velocity_env_cfg.py` |
| Gym registration | `source/.../tasks/locomotion/robots/pm01/12dof/__init__.py` |
| Parent init | `source/.../tasks/locomotion/robots/pm01/__init__.py` |
| URDF model | `unitree_model/PM01/12dof/urdf/pm01_only_legs_simple_collision.urdf` |

---

## 1. `pm01.py` - Robot Asset Config

### 1.1 [HIGH] Use `UnitreeArticulationCfg` instead of `ArticulationCfg`

All other robots in the project (G1, Go2, H1) use `UnitreeArticulationCfg` from `unitree.py`, which adds:
- `joint_sdk_names: list[str]` - required for real robot deployment
- `soft_joint_pos_limit_factor` default of `0.9`

**Current:**
```python
from isaaclab.assets.articulation import ArticulationCfg

PM01_CFG = ArticulationCfg(...)
```

**Should be:**
```python
from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg

PM01_CFG = UnitreeArticulationCfg(
    ...,
    joint_sdk_names=[
        "j00_hip_pitch_l", "j01_hip_roll_l", "j02_hip_yaw_l",
        "j03_knee_pitch_l", "j04_ankle_pitch_l", "j05_ankle_roll_l",
        "j06_hip_pitch_r", "j07_hip_roll_r", "j08_hip_yaw_r",
        "j09_knee_pitch_r", "j10_ankle_pitch_r", "j11_ankle_roll_r",
    ],
)
```

### 1.2 [HIGH] Use `UnitreeUrdfFileCfg` instead of raw `sim_utils.UrdfFileCfg`

The project provides `UnitreeUrdfFileCfg` (in `unitree.py`) with all the rigid body, articulation, and joint drive properties pre-configured. The current code duplicates all of these manually.

Additionally, `UnitreeUrdfFileCfg` includes `replace_cylinders_with_capsules=True` which the current config is **missing**. This can cause collision geometry issues.

**Current:**
```python
spawn=sim_utils.UrdfFileCfg(
    asset_path=PM01_URDF_PATH,
    fix_base=False,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(...),   # duplicated
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(...),  # duplicated
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(...),  # duplicated
)
```

**Should be:**
```python
from unitree_rl_lab.assets.robots.unitree import UNITREE_MODEL_DIR, UnitreeUrdfFileCfg

spawn=UnitreeUrdfFileCfg(
    asset_path=f"{UNITREE_MODEL_DIR}/PM01/12dof/urdf/pm01_only_legs_simple_collision.urdf",
)
```

### 1.3 [MEDIUM] Fragile URDF path resolution

The current path uses `os.path.dirname(__file__)` with five levels of `..` to reach the repo root. This breaks if the file is moved. All other robots use `UNITREE_MODEL_DIR` from `unitree.py`.

**Current:**
```python
PM01_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../..",
    "unitree_model/PM01/12dof/urdf/pm01_only_legs_simple_collision.urdf",
)
```

**Should be:**
```python
from unitree_rl_lab.assets.robots.unitree import UNITREE_MODEL_DIR
# then use f"{UNITREE_MODEL_DIR}/PM01/12dof/urdf/..." in spawn config
```

Note: `UNITREE_MODEL_DIR` must be set to the correct path in `unitree.py` (currently it's set to a user-specific absolute path).

### 1.4 [LOW] Initial spawn height

`pos=(0.0, 0.0, 0.92)` - verify this is correct for PM01. The reward target height is `0.8132` and termination minimum is `0.5`. The spawn height should be slightly above the nominal standing height to allow the robot to settle. 0.92 seems reasonable if PM01's standing height is ~0.81m.

---

## 2. `velocity_env_cfg.py` - Task Environment Config

### 2.1 [MEDIUM] Command velocity ranges may hinder early training

**Current:**
```python
ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(0.4, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
)
```

This starts with a **fixed 0.4 m/s forward velocity**. G1 starts with `(-0.1, 0.1)` — a small range including zero. Starting with a fixed non-zero velocity makes early training harder because the robot must learn to walk before it can learn to stand. The curriculum `limit_ranges` will expand later.

**Recommendation:** Start with a small range around zero:
```python
ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(-0.1, 0.1), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
)
```

### 2.2 [MEDIUM] Termination minimum height may be too aggressive

**Current:** `minimum_height=0.5` (terminates if base drops below 0.5m)
**G1 reference:** `minimum_height=0.2`

With spawn at 0.92m and target at 0.81m, a 0.5m threshold means the robot is terminated when it drops ~38% from target. Early in training the robot will fall often — a strict threshold means fewer learning steps per episode.

**Recommendation:** Consider lowering to `0.3` or `0.4` to give the robot more time to recover during early training. Can be tightened later via curriculum.

### 2.3 [LOW] `undesired_contacts` regex difference

**Current PM01:** `["(?!.*ankle_roll.*).*"]` - only excludes `ankle_roll` links
**G1 reference:** `["(?!.*ankle.*).*"]` - excludes all `ankle` links

This means PM01's `link_ankle_pitch_l` and `link_ankle_pitch_r` will be penalized as undesired contacts. If ankle pitch links can legitimately touch the ground during walking, change to:
```python
"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
```

### 2.4 [LOW] `RobotPlayEnvCfg` hardcodes velocity instead of using `limit_ranges`

**Current PM01:**
```python
self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
)
```

**G1 reference:**
```python
self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
```

G1's approach is cleaner — it reuses the `limit_ranges` so play mode tests the full trained command range. Hardcoding `(0.5, 0.5)` only tests one fixed speed.

### 2.5 [LOW] Push robot interval is gentler than G1

**PM01:** `interval_range_s=(8.0, 8.0)`, velocity `(-1.0, 1.0)`
**G1:** `interval_range_s=(5.0, 5.0)`, velocity `(-0.5, 0.5)`

PM01 pushes less frequently but harder. This is a design choice — just note that stronger pushes with a higher termination threshold (0.5) may cause frequent early terminations.

---

## 3. `12dof/__init__.py` - Gym Registration

**No issues.** Registration looks correct:
- ID: `Unitree-PM01-12dof-Velocity`
- Entry points correctly reference local module configs
- RSL-RL config correctly points to shared `BasePPORunnerCfg`

---

## 4. `pm01/__init__.py` - Parent Package Init

**No issues.** Empty file is correct — the sub-package `12dof/__init__.py` handles registration. The auto-discovery in `tasks/__init__.py` will recursively import it.

---

## 5. URDF Verification

**Joint/link names match across all files.** Verified:
- 12 revolute joints: `j00` through `j11` (6 left leg, 6 right leg)
- Fixed joints: `j12` through `j23` (torso, arms, head — not actuated)
- Base link: `link_base` (correctly referenced in height scanner and events)
- Foot links: `link_ankle_roll_l`, `link_ankle_roll_r` (correctly used in contact/reward configs)

---

## Summary of Required Changes

| Priority | File | Issue |
|----------|------|-------|
| HIGH | `pm01.py` | Use `UnitreeArticulationCfg` with `joint_sdk_names` |
| HIGH | `pm01.py` | Use `UnitreeUrdfFileCfg` instead of raw `UrdfFileCfg` (missing `replace_cylinders_with_capsules`) |
| MEDIUM | `pm01.py` | Use `UNITREE_MODEL_DIR` for path resolution |
| MEDIUM | `velocity_env_cfg.py` | Start command ranges from near-zero, not fixed 0.4 m/s |
| MEDIUM | `velocity_env_cfg.py` | Lower termination min height from 0.5 to ~0.3 |
| LOW | `velocity_env_cfg.py` | Consider `(?!.*ankle.*)` regex for undesired contacts |
| LOW | `velocity_env_cfg.py` | Use `limit_ranges` in `RobotPlayEnvCfg` |
