"""Compact Isaac loop for native Nav2 dynamic-costmap detour validation."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from aisha_isaaclab.assets import AISHA_PRESENTATION_CFG
from aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env import (
    AishaPhase3DynamicSafetyEnv,
    AishaPhase6HighSpeed80SafetyEnvCfg,
)


START_XY_M = (1.30, 0.45)
GOAL_XY_M = (10.70, 0.45)
BLOCKER_CENTRE_XY_M = (6.00, 2.10)
BLOCKER_SIZE_XYZ_M = (0.36, 1.65, 1.20)


def _static_box(
    size: tuple[float, float, float],
    colour: tuple[float, float, float],
) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=colour,
            roughness=0.72,
        ),
    )


def _blocker() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DynamicObstacle_0",
        spawn=sim_utils.CuboidCfg(
            size=BLOCKER_SIZE_XYZ_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.70,
                dynamic_friction=0.60,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.22, 0.03),
                emissive_color=(0.10, 0.01, 0.0),
                roughness=0.42,
                metallic=0.06,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


def _ray_targets() -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Phase7CStatic_.*",
            is_shared=False,
            merge_prim_meshes=True,
            track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/DynamicObstacle_.*",
            is_shared=False,
            merge_prim_meshes=True,
            track_mesh_transforms=True,
        ),
    ]


@configclass
class AishaPhase7CNativeDetourSceneCfg(InteractiveSceneCfg):
    """Two branches around one central island, plus a removable blocker."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65,
                dynamic_friction=0.55,
                restitution=0.0,
            ),
        ),
    )
    robot = AISHA_PRESENTATION_CFG
    phase7c_static_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Phase7CStatic_North",
        spawn=_static_box((11.00, 0.15, 1.50), (0.68, 0.72, 0.78)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.00, 3.075, 0.75)),
    )
    phase7c_static_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Phase7CStatic_South",
        spawn=_static_box((11.00, 0.15, 1.50), (0.68, 0.72, 0.78)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.00, -3.075, 0.75)),
    )
    phase7c_static_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Phase7CStatic_West",
        spawn=_static_box((0.15, 6.30, 1.50), (0.68, 0.72, 0.78)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.425, 0.0, 0.75)),
    )
    phase7c_static_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Phase7CStatic_East",
        spawn=_static_box((0.15, 6.30, 1.50), (0.68, 0.72, 0.78)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(11.575, 0.0, 0.75)),
    )
    phase7c_static_island = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Phase7CStatic_Island",
        spawn=_static_box((3.00, 2.40, 1.20), (0.18, 0.28, 0.42)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.00, 0.0, 0.60)),
    )
    dynamic_obstacle_0 = _blocker()
    crown_lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/lidar_link",
        update_period=0.10,
        offset=MultiMeshRayCasterCfg.OffsetCfg(),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=10.0,
        ),
        max_distance=10.0,
        mesh_prim_paths=_ray_targets(),
        reference_meshes=True,
        update_mesh_ids=False,
        debug_vis=False,
    )
    front_lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_lidar_link",
        update_period=0.05,
        offset=MultiMeshRayCasterCfg.OffsetCfg(),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-60.0, 60.0),
            horizontal_res=1.0,
        ),
        max_distance=10.0,
        mesh_prim_paths=_ray_targets(),
        reference_meshes=True,
        update_mesh_ids=False,
        debug_vis=False,
    )
    start_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StartMarker",
        spawn=sim_utils.CylinderCfg(
            radius=0.22,
            height=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.42, 0.95),
                emissive_color=(0.01, 0.04, 0.18),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*START_XY_M, 0.005)),
    )
    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.78, 0.34),
                emissive_color=(0.01, 0.16, 0.05),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*GOAL_XY_M, 0.005)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1900.0, color=(0.90, 0.93, 1.0)),
    )


@configclass
class AishaPhase7CNativeDetourEnvCfg(AishaPhase6HighSpeed80SafetyEnvCfg):
    """Frozen Phase 6/3N safety stack in the isolated detour topology."""

    scene: AishaPhase7CNativeDetourSceneCfg = AishaPhase7CNativeDetourSceneCfg(
        num_envs=1,
        env_spacing=20.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    dynamic_obstacle_count = 1
    maximum_active_obstacles = 1
    dynamic_obstacle_route_fractions = (0.50,)
    dynamic_obstacle_activation_probability = 0.0
    high_speed_segment_ids = (0,)
    non_high_speed_maximum_mps = 0.50
    high_speed_maximum_mps = 0.80
    start_heading_mode = "outgoing"
    start_lateral_jitter_m = 0.0
    start_yaw_jitter_rad = 0.0
    goal_jitter_m = 0.0
    goal_tolerance_m_by_segment = (0.18,) * 12
    temporary_blockage_segment_id = 0
    temporary_blockage_centre_xy_m = BLOCKER_CENTRE_XY_M
    temporary_blockage_size_xyz_m = BLOCKER_SIZE_XYZ_M

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (3.2, 9.5, 10.5)
        self.viewer.lookat = (6.0, 0.0, 0.4)


class AishaPhase7CNativeDetourEnv(AishaPhase3DynamicSafetyEnv):
    """Expose an operator-triggered barrier while keeping policy input sensor-only."""

    cfg: AishaPhase7CNativeDetourEnvCfg

    def __init__(
        self,
        cfg: AishaPhase7CNativeDetourEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        starts = torch.tensor(
            [START_XY_M] * 12, dtype=torch.float32, device=self.device
        )
        goals = torch.tensor(
            [GOAL_XY_M] * 12, dtype=torch.float32, device=self.device
        )
        self._segment_starts.copy_(starts)
        self._segment_goals.copy_(goals)
        self._segment_incoming_headings.zero_()
        self._phase7c_blockage_triggered = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._phase7c_blockage_released = torch.zeros_like(
            self._phase7c_blockage_triggered
        )

    def _sample_dynamic_obstacles(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "_obstacle_active"):
            return
        self._obstacle_active[:, env_ids] = False
        self._obstacle_pause_phase[:, env_ids] = 0.0
        if hasattr(self, "_phase7c_blockage_triggered"):
            self._phase7c_blockage_triggered[env_ids] = False
            self._phase7c_blockage_released[env_ids] = False

    def activate_temporary_blockage(self) -> None:
        self._phase7c_blockage_triggered[:] = True
        self._phase7c_blockage_released[:] = False

    def release_temporary_blockage(self) -> None:
        self._phase7c_blockage_released[:] = True

    def _update_dynamic_obstacles(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "_phase7c_blockage_triggered"):
            triggered = self._phase7c_blockage_triggered[env_ids]
            released = self._phase7c_blockage_released[env_ids]
            active = triggered & ~released
        else:
            triggered = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            active = triggered
        self._obstacle_active[0, env_ids] = active
        self._obstacle_pause_phase[0, env_ids] = triggered.float()

        origins = self.scene.env_origins[env_ids, :2]
        blocker = self._dynamic_obstacles[0]
        root_pose = blocker.data.default_root_state[env_ids, :7].clone()
        root_pose[:, 0] = origins[:, 0] + BLOCKER_CENTRE_XY_M[0]
        root_pose[:, 1] = origins[:, 1] + BLOCKER_CENTRE_XY_M[1]
        root_pose[:, 2] = torch.where(
            active,
            torch.full_like(root_pose[:, 2], BLOCKER_SIZE_XYZ_M[2] / 2.0),
            torch.full_like(root_pose[:, 2], -5.0),
        )
        root_pose[:, 3] = 1.0
        root_pose[:, 4:] = 0.0
        blocker.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        blocker.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )

    def blockage_state(self) -> dict[str, torch.Tensor]:
        triggered = self._phase7c_blockage_triggered.clone()
        released = self._phase7c_blockage_released.clone()
        return {
            "triggered": triggered,
            "active": triggered & ~released,
            "cleared": triggered & released,
            "blocker_position_xy_m": (
                self._dynamic_obstacles[0].data.root_pos_w[:, :2]
                - self.scene.env_origins[:, :2]
            ).clone(),
        }
