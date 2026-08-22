"""Run the frozen Phase 3M stack and Phase 3N safety actor in administration.usd."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from aisha_isaaclab.assets import SIM_PACKAGE_ROOT
from aisha_isaaclab.tasks.office_nav.administration_live_env import (
    ADMINISTRATION_LIVE_USD,
    PRESENTATION_ROBOT_USD,
    AishaAdministrationLiveSceneCfg,
    administration_collision_raycast_targets,
)
from aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env import (
    AishaPhase3DynamicSafetyEnv,
    AishaPhase3DynamicSafetyEnvCfg,
    _person_proxy,
)


PHASE3N_LIVE_GOAL_TOLERANCES_M = (
    0.45,
    0.45,
    0.45,
    0.22,  # Vice-Principal visit: presentation stop tolerance, not geometry.
    0.45,
    0.45,
    0.45,
    0.45,
    0.22,  # Principal visit: presentation stop tolerance, not geometry.
    0.45,
    0.45,
    0.45,
)

PHASE3N_PRESENTATION_GOAL_TOLERANCES_M = (
    *PHASE3N_LIVE_GOAL_TOLERANCES_M[:9],
    0.20,  # Reach the principal-departure centreline before return guarding.
    *PHASE3N_LIVE_GOAL_TOLERANCES_M[10:],
)

PHASE4A_PEDESTRIAN_USD = SIM_PACKAGE_ROOT / "usd" / "aisha_pedestrian_showcase.usda"


def _showcase_person_proxy() -> RigidObjectCfg:
    """Create the presentation-only humanoid with a 0.48 m torso envelope."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DynamicObstacle_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PHASE4A_PEDESTRIAN_USD),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


@configclass
class AishaAdministrationDynamicSafetySceneCfg(AishaAdministrationLiveSceneCfg):
    """Walkthrough-matched administration scene with ray-visible people."""

    dynamic_obstacle_0 = _person_proxy(0)
    dynamic_obstacle_1 = _person_proxy(1)
    dynamic_obstacle_2 = _person_proxy(2)
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
        mesh_prim_paths=administration_collision_raycast_targets()
        + [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/DynamicObstacle_.*",
                is_shared=False,
                merge_prim_meshes=True,
                track_mesh_transforms=True,
            )
        ],
        reference_meshes=True,
        update_mesh_ids=False,
        debug_vis=False,
    )


@configclass
class AishaAdministrationSafetyPresentationSceneCfg(
    AishaAdministrationLiveSceneCfg
):
    """Static presentation scene; dynamic safety is proven by a separate gate."""

    dynamic_obstacle_0 = _person_proxy(0)
    dynamic_obstacle_1 = _person_proxy(1)
    dynamic_obstacle_2 = _person_proxy(2)


@configclass
class AishaAdministrationDynamicSafetyShowcaseSceneCfg(
    AishaAdministrationDynamicSafetySceneCfg
):
    """Live scene with a stylized human on the existing conservative collider."""

    dynamic_obstacle_0 = _showcase_person_proxy()


@configclass
class AishaAdministrationDynamicSafetyEnvCfg(AishaPhase3DynamicSafetyEnvCfg):
    """One-action live-scene gate for the packaged Phase 3N checkpoint."""

    scene: AishaAdministrationDynamicSafetySceneCfg = (
        AishaAdministrationDynamicSafetySceneCfg(
            num_envs=1,
            env_spacing=55.0,
            replicate_physics=False,
            clone_in_fabric=False,
        )
    )
    goal_tolerance_m_by_segment = PHASE3N_LIVE_GOAL_TOLERANCES_M

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (-4.5, 4.5, 5.5)
        self.viewer.lookat = (0.0, 0.0, 0.55)


@configclass
class AishaAdministrationSafetyPresentationEnvCfg(
    AishaAdministrationDynamicSafetyEnvCfg
):
    """Reproducible final-film scenario with no stochastic pedestrian proxy."""

    scene: AishaAdministrationSafetyPresentationSceneCfg = (
        AishaAdministrationSafetyPresentationSceneCfg(
            num_envs=1,
            env_spacing=55.0,
            replicate_physics=False,
            clone_in_fabric=False,
        )
    )
    dynamic_obstacle_activation_probability = 0.0
    dynamic_crossing_creep_segment_ids = ()
    predictive_stop_segment_ids = (6, 11)
    goal_tolerance_m_by_segment = PHASE3N_PRESENTATION_GOAL_TOLERANCES_M


@configclass
class AishaAdministrationDynamicSafetyShowcaseEnvCfg(
    AishaAdministrationDynamicSafetyEnvCfg
):
    """Deterministic, presentation-only crossing on principal-approach segment 7."""

    scene: AishaAdministrationDynamicSafetyShowcaseSceneCfg = (
        AishaAdministrationDynamicSafetyShowcaseSceneCfg(
            num_envs=1,
            env_spacing=55.0,
            replicate_physics=False,
            clone_in_fabric=False,
        )
    )
    fixed_segment_id = 7
    route_chain_mode = False
    episode_length_s = 60.0
    start_lateral_jitter_m = 0.0
    start_yaw_jitter_rad = 0.0
    start_linear_velocity_range_mps = (0.0, 0.0)
    goal_jitter_m = 0.0

    # Remove stochastic sim-to-real perturbations only in this repeatable film
    # scenario. The accepted checkpoint and all Phase 3N evaluation tasks keep
    # their full domain-randomized contracts.
    curriculum_minimum_strength = 1.0
    action_latency_steps_range = (0, 0)
    motor_strength_scale_range = (1.0, 1.0)
    wheel_radius_scale_range = (1.0, 1.0)
    wheel_track_scale_range = (1.0, 1.0)
    drive_joint_damping_range = (120.0, 120.0)
    base_mass_scale_range = (1.0, 1.0)
    robot_static_friction_range = (0.60, 0.60)
    robot_dynamic_friction_range = (0.50, 0.50)
    observation_lidar_noise_std_m = 0.0
    observation_lidar_dropout_probability = 0.0
    lidar_episode_bias_range_m = (0.0, 0.0)
    lidar_episode_scale_range = (1.0, 1.0)

    maximum_active_obstacles = 1
    dynamic_obstacle_activation_probability = 1.0
    dynamic_obstacle_social_retreat_speed_mps = 0.0
    showcase_route_fraction = 0.52
    showcase_crossing_half_span_m = 1.15
    showcase_crossing_speed_mps = 0.48
    showcase_trigger_distance_m = 2.15


class AishaAdministrationDynamicSafetyEnv(AishaPhase3DynamicSafetyEnv):
    """Live wheel physics, full-ring sensing, and the frozen Phase 3M stack."""

    cfg: AishaAdministrationDynamicSafetyEnvCfg

    def __init__(
        self,
        cfg: AishaAdministrationDynamicSafetyEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        missing = [
            path
            for path in (ADMINISTRATION_LIVE_USD, PRESENTATION_ROBOT_USD)
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "missing live administration assets; run "
                f"isaaclab/tools/build_administration_live_assets.py first: {missing}"
            )
        super().__init__(cfg, render_mode, **kwargs)


class AishaAdministrationDynamicSafetyShowcaseEnv(
    AishaAdministrationDynamicSafetyEnv
):
    """Repeatable crossing used to film the frozen Phase 3N safety actor."""

    cfg: AishaAdministrationDynamicSafetyShowcaseEnvCfg

    def __init__(
        self,
        cfg: AishaAdministrationDynamicSafetyShowcaseEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        if not PHASE4A_PEDESTRIAN_USD.is_file():
            raise FileNotFoundError(PHASE4A_PEDESTRIAN_USD)
        super().__init__(cfg, render_mode, **kwargs)

    def _sample_dynamic_obstacles(self, env_ids: torch.Tensor) -> None:
        """Place one person off-route and arm a robot-proximity crossing trigger."""
        super()._sample_dynamic_obstacles(env_ids)
        segment_ids = self._segment_ids[env_ids]
        starts = self._segment_starts[segment_ids]
        goals = self._segment_goals[segment_ids]
        route_direction = goals - starts
        route_unit = route_direction / torch.linalg.norm(
            route_direction, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        crossing_axis = torch.stack((-route_unit[:, 1], route_unit[:, 0]), dim=-1)
        centre = starts + route_direction * self.cfg.showcase_route_fraction

        self._obstacle_active[:, env_ids] = False
        self._obstacle_active[0, env_ids] = True
        self._obstacle_centres[0, env_ids] = centre
        self._obstacle_axes[0, env_ids] = crossing_axis
        self._obstacle_half_spans[0, env_ids] = self.cfg.showcase_crossing_half_span_m
        # In this presentation-only override, the inherited buffers store
        # linear crossing position and the latched proximity trigger.
        self._obstacle_yield_offsets[0, env_ids] = -self.cfg.showcase_crossing_half_span_m
        self._obstacle_pause_phase[0, env_ids] = 0.0
        self._obstacle_angular_speeds[0, env_ids] = self.cfg.showcase_crossing_speed_mps
        self._obstacle_phases[0, env_ids] = 0.0
        for obstacle_index in range(1, self.cfg.dynamic_obstacle_count):
            self._obstacle_yield_offsets[obstacle_index, env_ids] = 0.0
            self._obstacle_pause_phase[obstacle_index, env_ids] = 0.0

    def _update_dynamic_obstacles(self, env_ids: torch.Tensor | None = None) -> None:
        """Walk once across the route after the robot enters the filmed encounter."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        origins = self.scene.env_origins[env_ids, :2]
        robot_local_xy = self._robot.data.root_pos_w[env_ids, :2] - origins
        centre = self._obstacle_centres[0, env_ids]
        trigger_now = (
            torch.linalg.norm(robot_local_xy - centre, dim=1)
            <= self.cfg.showcase_trigger_distance_m
        )
        self._obstacle_pause_phase[0, env_ids] = torch.maximum(
            self._obstacle_pause_phase[0, env_ids], trigger_now.float()
        )
        triggered = self._obstacle_pause_phase[0, env_ids] > 0.5
        progress = self._obstacle_yield_offsets[0, env_ids]
        progress += (
            triggered.float()
            * self.cfg.showcase_crossing_speed_mps
            * self.step_dt
        )
        progress.clamp_(
            -self.cfg.showcase_crossing_half_span_m,
            self.cfg.showcase_crossing_half_span_m,
        )
        self._obstacle_yield_offsets[0, env_ids] = progress

        axis = self._obstacle_axes[0, env_ids]
        local_xy = centre + axis * progress.unsqueeze(-1)
        person = self._dynamic_obstacles[0]
        root_pose = person.data.default_root_state[env_ids, :7].clone()
        root_pose[:, :2] = origins + local_xy
        root_pose[:, 2] = 0.85
        yaw = torch.atan2(axis[:, 1], axis[:, 0])
        root_pose[:, 3] = torch.cos(0.5 * yaw)
        root_pose[:, 4] = 0.0
        root_pose[:, 5] = 0.0
        root_pose[:, 6] = torch.sin(0.5 * yaw)
        velocity = torch.zeros((len(env_ids), 6), device=self.device)
        walking = triggered & (
            progress < self.cfg.showcase_crossing_half_span_m - 1.0e-5
        )
        velocity[:, :2] = (
            axis
            * walking.float().unsqueeze(-1)
            * self.cfg.showcase_crossing_speed_mps
        )
        person.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        person.write_root_velocity_to_sim(velocity, env_ids=env_ids)

        for obstacle_index in range(1, self.cfg.dynamic_obstacle_count):
            obstacle = self._dynamic_obstacles[obstacle_index]
            hidden_pose = obstacle.data.default_root_state[env_ids, :7].clone()
            hidden_pose[:, :2] = origins
            hidden_pose[:, 2] = -5.0
            hidden_pose[:, 3] = 1.0
            hidden_pose[:, 4:] = 0.0
            obstacle.write_root_pose_to_sim(hidden_pose, env_ids=env_ids)
            obstacle.write_root_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
            )

    def showcase_state(self) -> dict[str, torch.Tensor]:
        """Expose presentation telemetry without adding privileged policy input."""
        progress = self._obstacle_yield_offsets[0].clone()
        normalized = (
            progress + self.cfg.showcase_crossing_half_span_m
        ) / (2.0 * self.cfg.showcase_crossing_half_span_m)
        return {
            "triggered": self._obstacle_pause_phase[0] > 0.5,
            "crossing_progress": normalized.clamp(0.0, 1.0),
            "person_position_xy_m": (
                self._dynamic_obstacles[0].data.root_pos_w[:, :2]
                - self.scene.env_origins[:, :2]
            ).clone(),
        }
