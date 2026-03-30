"""PM01 12-DOF biped robot configuration for unitree_rl_lab."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_MODEL_DIR, UnitreeArticulationCfg, UnitreeUrdfFileCfg

PM01_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{UNITREE_MODEL_DIR}/PM01/12dof/urdf/pm01_only_legs_simple_collision.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            "j00_hip_pitch_l":  -0.24,
            "j01_hip_roll_l":    0.0,
            "j02_hip_yaw_l":     0.0,
            "j03_knee_pitch_l":  0.48,
            "j04_ankle_pitch_l": -0.24,
            "j05_ankle_roll_l":  0.0,
            "j06_hip_pitch_r":  -0.24,
            "j07_hip_roll_r":    0.0,
            "j08_hip_yaw_r":     0.0,
            "j09_knee_pitch_r":  0.48,
            "j10_ankle_pitch_r": -0.24,
            "j11_ankle_roll_r":  0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j00_hip_pitch_l", "j06_hip_pitch_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=70.0,
            damping=7.0,
            armature=0.0453,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["j01_hip_roll_l", "j07_hip_roll_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=50.0,
            damping=5.0,
            armature=0.0453,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["j02_hip_yaw_l", "j08_hip_yaw_r"],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=50.0,
            damping=5.0,
            armature=0.0067,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["j03_knee_pitch_l", "j09_knee_pitch_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=70.0,
            damping=7.0,
            armature=0.0453,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[
                "j04_ankle_pitch_l", "j05_ankle_roll_l",
                "j10_ankle_pitch_r", "j11_ankle_roll_r",
            ],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=20.0,
            damping=2.0,
            armature=0.0067,
        ),
    },
    joint_sdk_names=[
        "j00_hip_pitch_l", "j01_hip_roll_l", "j02_hip_yaw_l",
        "j03_knee_pitch_l", "j04_ankle_pitch_l", "j05_ankle_roll_l",
        "j06_hip_pitch_r", "j07_hip_roll_r", "j08_hip_yaw_r",
        "j09_knee_pitch_r", "j10_ankle_pitch_r", "j11_ankle_roll_r",
    ],
)
