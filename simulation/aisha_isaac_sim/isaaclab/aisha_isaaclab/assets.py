"""AI-SHA asset configuration for Isaac Lab."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


SIM_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
AISHA_LOADED_USD = SIM_PACKAGE_ROOT / "usd" / "aisha_loaded.usd"
AISHA_PRESENTATION_USD = SIM_PACKAGE_ROOT / "usd" / "aisha_loaded_presentation.usda"


AISHA_LOADED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(AISHA_LOADED_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.80, 0.0, 0.03),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "drive_wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=6.0,
            velocity_limit_sim=16.755,
            stiffness=0.0,
            damping=120.0,
        )
    },
)


AISHA_PRESENTATION_CFG = AISHA_LOADED_CFG.replace(
    spawn=AISHA_LOADED_CFG.spawn.replace(usd_path=str(AISHA_PRESENTATION_USD))
)
