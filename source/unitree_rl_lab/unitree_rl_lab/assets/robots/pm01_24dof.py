"""PM01 24-DOF biped robot configuration (legs + waist + arms + head)."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_MODEL_DIR, UnitreeArticulationCfg, UnitreeUrdfFileCfg

PM01_24DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{UNITREE_MODEL_DIR}/PM01/24dof/urdf/pm01_24dof.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            # Left leg
            "j00_hip_pitch_l": -0.24,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.48,
            "j04_ankle_pitch_l": -0.24,
            "j05_ankle_roll_l": 0.0,
            # Right leg
            "j06_hip_pitch_r": -0.24,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.48,
            "j10_ankle_pitch_r": -0.24,
            "j11_ankle_roll_r": 0.0,
            # Waist
            "j12_waist_yaw": 0.0,
            # Left arm
            "j13_shoulder_pitch_l": 0.0,
            "j14_shoulder_roll_l": 0.25,
            "j15_shoulder_yaw_l": 0.0,
            "j16_elbow_pitch_l": 0.7,
            "j17_elbow_yaw_l": 0.0,
            # Right arm
            "j18_shoulder_pitch_r": 0.0,
            "j19_shoulder_roll_r": -0.25,
            "j20_shoulder_yaw_r": 0.0,
            "j21_elbow_pitch_r": 0.7,
            "j22_elbow_yaw_r": 0.0,
            # Head
            "j23_head_yaw": 0.0,
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
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["j12_waist_yaw"],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=50.0,
            damping=5.0,
            armature=0.0067,
        ),
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=[
                "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l",
                "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r",
            ],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=40.0,
            damping=4.0,
            armature=0.0067,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=[
                "j16_elbow_pitch_l", "j17_elbow_yaw_l",
                "j21_elbow_pitch_r", "j22_elbow_yaw_r",
            ],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=40.0,
            damping=4.0,
            armature=0.0067,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["j23_head_yaw"],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=40.0,
            damping=4.0,
            armature=0.0067,
        ),
    },
    joint_sdk_names=[
        "j00_hip_pitch_l", "j01_hip_roll_l", "j02_hip_yaw_l",
        "j03_knee_pitch_l", "j04_ankle_pitch_l", "j05_ankle_roll_l",
        "j06_hip_pitch_r", "j07_hip_roll_r", "j08_hip_yaw_r",
        "j09_knee_pitch_r", "j10_ankle_pitch_r", "j11_ankle_roll_r",
        "j12_waist_yaw",
        "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l",
        "j16_elbow_pitch_l", "j17_elbow_yaw_l",
        "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r",
        "j21_elbow_pitch_r", "j22_elbow_yaw_r",
        "j23_head_yaw",
    ],
)
