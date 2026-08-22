"""Phase 3 dynamic-obstacle and sim-to-real domain-randomization curriculum."""

from __future__ import annotations

import math
import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from aisha_isaaclab.tasks.office_nav.block_a_sensor_env import (
    AishaBlockASensorEnv,
    AishaBlockASensorEnvCfg,
    AishaBlockASensorSceneCfg,
    COURSE_USD,
    ROUTE_SEGMENTS,
)
from aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env import (
    PHASE2_GOAL_TOLERANCES,
    TURN_DIRECTION_HINTS,
)


PHASE3_FROZEN_ROUTE_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "aisha_phase3_frozen_route_model_2225.pt"
)
PHASE3_FROZEN_ROUTE_CHECKPOINT_SHA256 = (
    "52f0094674dea901b4b7f3d7717bc9c2b014a6dc2d8e22cca768f783f4a9c0c8"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _person_proxy(index: int) -> RigidObjectCfg:
    """Create a conservative, kinematic person proxy visible to physics and rays."""
    palette = (
        (0.10, 0.34, 0.62),
        (0.62, 0.20, 0.16),
        (0.17, 0.48, 0.28),
    )
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/DynamicObstacle_{index}",
        spawn=sim_utils.CapsuleCfg(
            radius=0.24,
            height=1.70,
            axis="Z",
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
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.60,
                dynamic_friction=0.50,
                restitution=0.0,
                friction_combine_mode="min",
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=palette[index],
                roughness=0.62,
                metallic=0.0,
            ),
        ),
        # Inactive proxies stay below every navigable floor and outside the
        # horizontal LD19 scan plane.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )


@configclass
class AishaPhase3DynamicSceneCfg(AishaBlockASensorSceneCfg):
    """Replicated route course with three independently moving person proxies."""

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
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Course",
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
        ],
        reference_meshes=True,
        # Isaac Lab 2.3.2 allocates mesh-id results as (N, B, 1) but the
        # multi-mesh Warp query currently returns (N, B).  The policy and
        # collision truth need hit ranges, not mesh labels, so leave labels
        # disabled while retaining transform tracking for moving obstacles.
        update_mesh_ids=False,
        debug_vis=False,
    )


@configclass
class AishaPhase3DynamicDREnvCfg(AishaBlockASensorEnvCfg):
    """Checkpoint-compatible PPO curriculum for people and dynamics variation."""

    scene: AishaPhase3DynamicSceneCfg = AishaPhase3DynamicSceneCfg(
        num_envs=32,
        env_spacing=50.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    episode_length_s = 70.0
    linear_velocity_range_mps = (0.0, 0.50)
    start_lateral_jitter_m = 0.08
    start_yaw_jitter_rad = math.radians(18.0)
    start_heading_mode = "incoming"
    start_transition_backoff_m_by_segment = (
        0.00,
        0.45,
        0.45,
        0.45,
        0.20,
        0.45,
        0.45,
        0.45,
        0.45,
        0.20,
        0.45,
        0.45,
    )
    start_linear_velocity_range_mps = (0.0, 0.35)
    goal_jitter_m = 0.06
    goal_tolerance_m_by_segment = PHASE2_GOAL_TOLERANCES
    turn_direction_hint_rad_by_segment = TURN_DIRECTION_HINTS
    segment_sampling_weights = (
        10.0,
        16.0,
        12.0,
        3.0,
        3.0,
        16.0,
        12.0,
        12.0,
        3.0,
        3.0,
        12.0,
        10.0,
    )

    # LD19 observation randomization. These perturb only policy observations;
    # termination always uses the uncorrupted geometric ray ranges.
    observation_lidar_noise_std_m = 0.03
    observation_lidar_dropout_probability = 0.01
    lidar_episode_bias_range_m = (-0.025, 0.025)
    lidar_episode_scale_range = (0.985, 1.015)

    # Actuation and rigid-body randomization. Ranges are deliberately modest
    # because hardware-specific calibration has not yet been measured.
    action_latency_steps_range = (0, 2)
    motor_strength_scale_range = (0.90, 1.10)
    wheel_radius_scale_range = (0.97, 1.03)
    wheel_track_scale_range = (0.98, 1.02)
    drive_joint_damping_range = (96.0, 144.0)
    base_mass_scale_range = (0.88, 1.12)
    robot_static_friction_range = (0.45, 0.75)
    robot_dynamic_friction_range = (0.35, 0.65)

    # Preserve the accepted static-route skill before progressively exposing
    # the policy to the full perturbation distribution. At 32 steps/iteration,
    # this gives 100 PPO iterations of rehearsal and a 350-iteration ramp.
    curriculum_warmup_policy_steps = 3_200
    curriculum_ramp_policy_steps = 11_200
    curriculum_minimum_strength = 0.0

    # Dynamic-person curriculum. People cross only open hall/atrium route legs
    # with enough lateral space. Door, in-office, and the exact segment-6
    # principal U-turn retain furniture/static obstacles but no non-yielding
    # kinematic pedestrian crossing.
    dynamic_obstacle_count = 3
    maximum_active_obstacles = 2
    dynamic_obstacle_segment_ids = (0, 1, 2, 5, 7, 10, 11)
    dynamic_obstacle_activation_probability = 0.60
    dynamic_obstacle_crossing_speed_range_mps = (0.25, 0.65)
    dynamic_obstacle_path_half_span_range_m = (0.85, 1.25)
    dynamic_obstacle_route_fractions = (0.32, 0.56, 0.76)
    dynamic_obstacle_yield_radius_m = 1.10

    reward_progress = 14.0
    reward_heading_alignment = 0.02
    reward_heading_progress = 8.0
    penalty_wrong_uturn_direction = -0.05
    penalty_misaligned_forward = -0.05
    penalty_near_obstacle = -0.01
    penalty_forward_near_obstacle = -0.12
    forward_near_obstacle_distance_m = 1.20
    penalty_collision = -100.0


class AishaPhase3DynamicDREnv(AishaBlockASensorEnv):
    """Learn stopping/avoidance under moving people and plausible sim variation."""

    cfg: AishaPhase3DynamicDREnvCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._dynamic_obstacles = [
            self.scene.rigid_objects[f"dynamic_obstacle_{index}"]
            for index in range(self.cfg.dynamic_obstacle_count)
        ]

    def __init__(self, cfg: AishaPhase3DynamicDREnvCfg, render_mode: str | None = None, **kwargs):
        if not COURSE_USD.is_file():
            raise FileNotFoundError(
                f"missing {COURSE_USD}; run isaaclab/tools/build_block_a_training_course.py"
            )
        super().__init__(cfg, render_mode, **kwargs)
        if len(self._dynamic_obstacles) != self.cfg.dynamic_obstacle_count:
            raise RuntimeError("dynamic obstacle scene/config count mismatch")
        if len(self.cfg.dynamic_obstacle_route_fractions) != self.cfg.dynamic_obstacle_count:
            raise ValueError("dynamic_obstacle_route_fractions must match dynamic_obstacle_count")

        self._action_history = torch.zeros((self.num_envs, 3, 2), device=self.device)
        self._action_latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motor_strength = torch.ones((self.num_envs, 2), device=self.device)
        self._wheel_radius_scale = torch.ones(self.num_envs, device=self.device)
        self._wheel_track_scale = torch.ones(self.num_envs, device=self.device)
        self._lidar_episode_bias = torch.zeros(self.num_envs, device=self.device)
        self._lidar_episode_scale = torch.ones(self.num_envs, device=self.device)
        self._mass_scale = torch.ones(self.num_envs, device=self.device)
        self._static_friction = torch.ones(self.num_envs, device=self.device)
        self._dynamic_friction = torch.ones(self.num_envs, device=self.device)
        self._drive_damping = torch.full((self.num_envs, 2), 120.0, device=self.device)

        obstacle_shape = (self.cfg.dynamic_obstacle_count, self.num_envs)
        self._obstacle_active = torch.zeros(obstacle_shape, dtype=torch.bool, device=self.device)
        self._obstacle_centres = torch.zeros((*obstacle_shape, 2), device=self.device)
        self._obstacle_axes = torch.zeros((*obstacle_shape, 2), device=self.device)
        self._obstacle_half_spans = torch.ones(obstacle_shape, device=self.device)
        self._obstacle_angular_speeds = torch.zeros(obstacle_shape, device=self.device)
        self._obstacle_phases = torch.zeros(obstacle_shape, device=self.device)
        self._obstacle_pause_phase = torch.zeros(obstacle_shape, device=self.device)

        base_ids, _ = self._robot.find_bodies("base_link")
        if len(base_ids) != 1:
            raise RuntimeError(f"expected one base_link for mass randomization, found {base_ids}")
        self._base_body_id = int(base_ids[0])
        self._episode_sums["forward_near_obstacle"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    @staticmethod
    def _uniform(
        count: int, value_range: tuple[float, float], device: str | torch.device
    ) -> torch.Tensor:
        return torch.empty(count, device=device).uniform_(*value_range)

    def _curriculum_strength(self) -> float:
        elapsed = max(0, int(self.common_step_counter) - self.cfg.curriculum_warmup_policy_steps)
        ramp = max(1, self.cfg.curriculum_ramp_policy_steps)
        return max(
            float(self.cfg.curriculum_minimum_strength),
            min(1.0, elapsed / ramp),
        )

    def _blended_uniform(
        self,
        count: int,
        value_range: tuple[float, float],
        nominal: float,
        strength: float,
    ) -> torch.Tensor:
        sampled = self._uniform(count, value_range, self.device)
        return nominal + strength * (sampled - nominal)

    def _randomize_physics(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        strength = self._curriculum_strength()
        latency_min, latency_max = self.cfg.action_latency_steps_range
        latency_max = max(latency_min, int(round(latency_max * strength)))
        self._action_latency_steps[env_ids] = torch.randint(
            latency_min,
            latency_max + 1,
            (count,),
            device=self.device,
        )
        sampled_motor_strength = torch.empty((count, 2), device=self.device).uniform_(
            *self.cfg.motor_strength_scale_range
        )
        self._motor_strength[env_ids] = 1.0 + strength * (sampled_motor_strength - 1.0)
        self._wheel_radius_scale[env_ids] = self._blended_uniform(
            count, self.cfg.wheel_radius_scale_range, 1.0, strength
        )
        self._wheel_track_scale[env_ids] = self._blended_uniform(
            count, self.cfg.wheel_track_scale_range, 1.0, strength
        )
        self._lidar_episode_bias[env_ids] = strength * self._uniform(
            count, self.cfg.lidar_episode_bias_range_m, self.device
        )
        self._lidar_episode_scale[env_ids] = self._blended_uniform(
            count, self.cfg.lidar_episode_scale_range, 1.0, strength
        )
        self._mass_scale[env_ids] = self._blended_uniform(
            count, self.cfg.base_mass_scale_range, 1.0, strength
        )
        self._static_friction[env_ids] = self._blended_uniform(
            count, self.cfg.robot_static_friction_range, 0.60, strength
        )
        self._dynamic_friction[env_ids] = torch.minimum(
            self._blended_uniform(
                count, self.cfg.robot_dynamic_friction_range, 0.50, strength
            ),
            self._static_friction[env_ids],
        )
        sampled_damping = torch.empty((count, 2), device=self.device).uniform_(
            *self.cfg.drive_joint_damping_range
        )
        damping = 120.0 + strength * (sampled_damping - 120.0)
        self._drive_damping[env_ids] = damping
        self._robot.write_joint_damping_to_sim(
            damping,
            joint_ids=self._wheel_ids,
            env_ids=env_ids,
        )

        cpu_ids = env_ids.cpu()
        masses = self._robot.root_physx_view.get_masses()
        default_mass = self._robot.data.default_mass.cpu()
        masses[cpu_ids, self._base_body_id] = (
            default_mass[cpu_ids, self._base_body_id] * self._mass_scale[env_ids].cpu()
        )
        self._robot.root_physx_view.set_masses(masses, cpu_ids)
        inertias = self._robot.root_physx_view.get_inertias()
        default_inertia = self._robot.data.default_inertia.cpu()
        inertias[cpu_ids, self._base_body_id] = (
            default_inertia[cpu_ids, self._base_body_id]
            * self._mass_scale[env_ids].cpu().unsqueeze(-1)
        )
        self._robot.root_physx_view.set_inertias(inertias, cpu_ids)

        materials = self._robot.root_physx_view.get_material_properties()
        materials[cpu_ids, :, 0] = self._static_friction[env_ids].cpu().unsqueeze(-1)
        materials[cpu_ids, :, 1] = self._dynamic_friction[env_ids].cpu().unsqueeze(-1)
        materials[cpu_ids, :, 2] = 0.0
        self._robot.root_physx_view.set_material_properties(materials, cpu_ids)

    def _sample_dynamic_obstacles(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        strength = self._curriculum_strength()
        segment_ids = self._segment_ids[env_ids]
        allowed = torch.zeros(count, dtype=torch.bool, device=self.device)
        for segment_id in self.cfg.dynamic_obstacle_segment_ids:
            allowed |= segment_ids == segment_id
        maximum_active = 1 if strength < 0.75 else self.cfg.maximum_active_obstacles
        active_count = torch.randint(
            1,
            maximum_active + 1,
            (count,),
            device=self.device,
        )
        active_count = torch.where(allowed, active_count, torch.zeros_like(active_count))

        starts = self._segment_starts[segment_ids]
        goals = self._segment_goals[segment_ids]
        route_direction = goals - starts
        route_unit = route_direction / torch.linalg.norm(route_direction, dim=1, keepdim=True).clamp_min(1.0e-6)
        crossing_axis = torch.stack((-route_unit[:, 1], route_unit[:, 0]), dim=-1)

        for obstacle_index in range(self.cfg.dynamic_obstacle_count):
            probability_gate = (
                torch.rand(count, device=self.device)
                < self.cfg.dynamic_obstacle_activation_probability * strength
            )
            active = allowed & (active_count > obstacle_index) & probability_gate
            fraction = self.cfg.dynamic_obstacle_route_fractions[obstacle_index]
            fraction_jitter = torch.empty(count, device=self.device).uniform_(-0.05, 0.05)
            centre = starts + route_direction * (fraction + fraction_jitter).unsqueeze(-1)
            self._obstacle_active[obstacle_index, env_ids] = active
            self._obstacle_centres[obstacle_index, env_ids] = centre
            self._obstacle_axes[obstacle_index, env_ids] = crossing_axis
            half_span = self._uniform(
                count, self.cfg.dynamic_obstacle_path_half_span_range_m, self.device
            )
            speed = self._uniform(
                count, self.cfg.dynamic_obstacle_crossing_speed_range_mps, self.device
            )
            direction_sign = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                -torch.ones(count, device=self.device),
                torch.ones(count, device=self.device),
            )
            self._obstacle_half_spans[obstacle_index, env_ids] = half_span
            self._obstacle_angular_speeds[obstacle_index, env_ids] = (
                direction_sign * speed / half_span
            )
            self._obstacle_phases[obstacle_index, env_ids] = (
                -0.5 * math.pi
                + torch.empty(count, device=self.device).uniform_(-0.12, 0.12)
            )
            self._obstacle_pause_phase[obstacle_index, env_ids] = 0.0

    def _update_dynamic_obstacles(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elapsed = self.episode_length_buf[env_ids].float() * self.step_dt
        origins = self.scene.env_origins[env_ids, :2]
        for obstacle_index, obstacle in enumerate(self._dynamic_obstacles):
            phase = (
                self._obstacle_phases[obstacle_index, env_ids]
                + elapsed * self._obstacle_angular_speeds[obstacle_index, env_ids]
                - self._obstacle_pause_phase[obstacle_index, env_ids]
            )
            lateral = self._obstacle_half_spans[obstacle_index, env_ids] * torch.sin(phase)
            local_xy = (
                self._obstacle_centres[obstacle_index, env_ids]
                + self._obstacle_axes[obstacle_index, env_ids] * lateral.unsqueeze(-1)
            )
            active = self._obstacle_active[obstacle_index, env_ids]
            robot_local_xy = self._robot.data.root_pos_w[env_ids, :2] - origins
            yielding = active & (
                torch.linalg.norm(local_xy - robot_local_xy, dim=1)
                < self.cfg.dynamic_obstacle_yield_radius_m
            )
            # Kinematic people pause instead of walking into a stopped robot.
            # This is environment behaviour only; the policy observes ordinary
            # LiDAR ranges and receives no pedestrian position/velocity state.
            self._obstacle_pause_phase[obstacle_index, env_ids] += (
                yielding.float()
                * self.step_dt
                * self._obstacle_angular_speeds[obstacle_index, env_ids]
            )
            root_pose = obstacle.data.default_root_state[env_ids, :7].clone()
            root_pose[:, :2] = origins + local_xy
            root_pose[:, 2] = torch.where(
                active,
                torch.full_like(lateral, 0.85),
                torch.full_like(lateral, -5.0),
            )
            root_pose[:, 3] = 1.0
            root_pose[:, 4:] = 0.0
            velocity = torch.zeros((len(env_ids), 6), device=self.device)
            lateral_velocity = (
                self._obstacle_half_spans[obstacle_index, env_ids]
                * torch.cos(phase)
                * self._obstacle_angular_speeds[obstacle_index, env_ids]
            )
            velocity[:, :2] = (
                self._obstacle_axes[obstacle_index, env_ids]
                * lateral_velocity.unsqueeze(-1)
            )
            velocity[~active | yielding] = 0.0
            obstacle.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            obstacle.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    def _lidar_observation_ranges(self) -> torch.Tensor:
        ranges = self._lidar_ranges()
        if not hasattr(self, "_lidar_episode_scale"):
            return ranges
        strength = self._curriculum_strength()
        if self.cfg.observation_lidar_noise_std_m > 0.0:
            ranges = ranges + torch.randn_like(ranges) * (
                self.cfg.observation_lidar_noise_std_m * strength
            )
        if self.cfg.observation_lidar_dropout_probability > 0.0:
            drop = torch.rand_like(ranges) < (
                self.cfg.observation_lidar_dropout_probability * strength
            )
            ranges = torch.where(drop, self.cfg.lidar_max_range_m, ranges)
        ranges = (
            ranges * self._lidar_episode_scale.unsqueeze(-1)
            + self._lidar_episode_bias.unsqueeze(-1)
        )
        return ranges.clamp(self.cfg.lidar_min_range_m, self.cfg.lidar_max_range_m)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._update_dynamic_obstacles()
        self._action_history[:, 2] = self._action_history[:, 1]
        self._action_history[:, 1] = self._action_history[:, 0]
        self._action_history[:, 0] = actions
        delayed = torch.gather(
            self._action_history,
            1,
            self._action_latency_steps.view(-1, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)

        self._previous_actions.copy_(self._actions)
        self._actions = delayed.clone().clamp(-1.0, 1.0)
        minimum, maximum = self.cfg.linear_velocity_range_mps
        linear = minimum + (self._actions[:, 0] + 1.0) * 0.5 * (maximum - minimum)
        angular = self._actions[:, 1] * self.cfg.angular_velocity_max_rad_s
        half_track = self.cfg.wheel_track_m * self._wheel_track_scale / 2.0
        wheel_radius = self.cfg.wheel_radius_m * self._wheel_radius_scale
        self._wheel_targets[:, 0] = (linear - angular * half_track) / wheel_radius
        self._wheel_targets[:, 1] = (linear + angular * half_track) / wheel_radius
        self._wheel_targets *= self._motor_strength
        self._wheel_targets.clamp_(
            -self.cfg.wheel_speed_limit_rad_s,
            self.cfg.wheel_speed_limit_rad_s,
        )

    def _get_rewards(self) -> torch.Tensor:
        rewards = super()._get_rewards()
        lidar = self._lidar_ranges()
        # The five front-facing rays cover +/-20 degrees. Penalizing forward
        # intent here teaches stopping/avoidance without exposing privileged
        # obstacle position or velocity to the policy.
        front_minimum = torch.amin(lidar[:, 16:21], dim=1)
        normalized_forward = ((self._actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        forward_near = (
            (front_minimum < self.cfg.forward_near_obstacle_distance_m).float()
            * normalized_forward
            * self.cfg.penalty_forward_near_obstacle
        )
        self._episode_sums["forward_near_obstacle"] += forward_near
        return rewards + forward_near

    def _dynamic_obstacle_overlap(self) -> torch.Tensor:
        """Classify footprint contacts with active person proxies for evaluation."""
        base_xy = self._robot.data.root_pos_w[:, :2]
        quaternion = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quaternion[:, 0] * quaternion[:, 3] + quaternion[:, 1] * quaternion[:, 2]),
            1.0 - 2.0 * (quaternion[:, 2].square() + quaternion[:, 3].square()),
        )
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        overlap = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        person_radius = 0.24
        for obstacle_index, obstacle in enumerate(self._dynamic_obstacles):
            delta = obstacle.data.root_pos_w[:, :2] - base_xy
            local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
            local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
            overlap |= (
                self._obstacle_active[obstacle_index]
                & (local_x >= self.cfg.robot_rear_x_m - person_radius)
                & (local_x <= self.cfg.robot_front_x_m + person_radius)
                & (torch.abs(local_y) <= self.cfg.robot_half_width_m + person_radius)
            )
        return overlap

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        outcomes = self.extras["episode_outcomes"]
        collision = outcomes["collision"]
        dynamic_collision = collision & self._dynamic_obstacle_overlap()
        outcomes["dynamic_obstacle_collision"] = dynamic_collision
        outcomes["static_collision"] = collision & ~dynamic_collision
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        # DirectRLEnv may call the reset hook during base construction. The
        # Phase 3 buffers are initialized immediately afterwards and are fully
        # active for the first user-visible reset.
        if not hasattr(self, "_action_history"):
            return
        self._randomize_physics(env_ids)
        self._sample_dynamic_obstacles(env_ids)
        self._action_history[env_ids] = self._actions[env_ids].unsqueeze(1).expand(-1, 3, -1)
        self._update_dynamic_obstacles(env_ids)
        self.extras["domain_randomization"] = {
            "curriculum_strength": self._curriculum_strength(),
            "action_latency_steps": self._action_latency_steps.clone(),
            "motor_strength": self._motor_strength.clone(),
            "wheel_radius_scale": self._wheel_radius_scale.clone(),
            "wheel_track_scale": self._wheel_track_scale.clone(),
            "base_mass_scale": self._mass_scale.clone(),
            "static_friction": self._static_friction.clone(),
            "dynamic_friction": self._dynamic_friction.clone(),
            "active_obstacle_count": self._obstacle_active.sum(dim=0).clone(),
        }


@configclass
class AishaPhase3SafetyResidualEnvCfg(AishaPhase3DynamicDREnvCfg):
    """Full-strength safety adaptation over a hash-locked route actor."""

    frozen_route_checkpoint = str(PHASE3_FROZEN_ROUTE_CHECKPOINT)
    frozen_route_checkpoint_sha256 = PHASE3_FROZEN_ROUTE_CHECKPOINT_SHA256

    # The learned residual may remove all forward speed, but it cannot reverse
    # the robot, increase speed, flip steering sign, or add steering magnitude.
    maximum_angular_attenuation = 0.25

    # There is no route-learning warm-up: the route actor is frozen and the
    # safety controller trains against the complete declared perturbation set.
    curriculum_warmup_policy_steps = 0
    curriculum_ramp_policy_steps = 1
    curriculum_minimum_strength = 1.0
    segment_sampling_weights = (
        18.0,
        18.0,
        18.0,
        4.0,
        4.0,
        18.0,
        6.0,
        18.0,
        4.0,
        4.0,
        18.0,
        18.0,
    )

    # These shaping terms use only the same front LiDAR ranges available in
    # the policy observation. Simulator obstacle identity remains evaluation
    # truth, not a policy input.
    safety_closing_distance_m = 1.80
    safety_clear_distance_m = 2.00
    safety_closing_delta_m = 0.01
    reward_brake_while_closing = 0.15
    penalty_unmitigated_closing = -0.25
    penalty_unnecessary_brake = -0.02
    penalty_clear_path_angular_attenuation = -0.01
    penalty_forward_near_obstacle = -0.40
    forward_near_obstacle_distance_m = 1.60


class AishaPhase3SafetyResidualEnv(AishaPhase3DynamicDREnv):
    """Train a recurrent slow/stop layer without modifying the route policy."""

    cfg: AishaPhase3SafetyResidualEnvCfg

    def __init__(self, cfg: AishaPhase3SafetyResidualEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        checkpoint_path = Path(self.cfg.frozen_route_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing frozen route checkpoint: {checkpoint_path}")
        checkpoint_sha256 = _sha256(checkpoint_path)
        if checkpoint_sha256 != self.cfg.frozen_route_checkpoint_sha256:
            raise RuntimeError(
                "frozen route checkpoint hash mismatch: "
                f"{checkpoint_sha256} != {self.cfg.frozen_route_checkpoint_sha256}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint["model_state_dict"]
        self._frozen_route_actor = nn.Sequential(
            nn.Linear(46, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 2),
        ).to(self.device)
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        }
        self._frozen_route_actor.load_state_dict(actor_state, strict=True)
        self._frozen_route_actor.eval()
        self._frozen_route_actor.requires_grad_(False)
        self._frozen_route_obs_mean = state["actor_obs_normalizer._mean"].to(self.device)
        self._frozen_route_obs_std = state["actor_obs_normalizer._std"].to(self.device)
        self._frozen_route_checkpoint_path = checkpoint_path
        self._frozen_route_checkpoint_actual_sha256 = checkpoint_sha256

        self._residual_action_history = torch.zeros((self.num_envs, 3, 2), device=self.device)
        self._residual_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._applied_residual_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._base_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._requested_combined_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._applied_brake_fraction = torch.zeros(self.num_envs, device=self.device)
        self._applied_angular_attenuation = torch.zeros(self.num_envs, device=self.device)
        self._previous_front_minimum = torch.full(
            (self.num_envs,), self.cfg.lidar_max_range_m, device=self.device
        )
        for name in (
            "brake_while_closing",
            "unmitigated_closing",
            "unnecessary_brake",
            "clear_path_angular_attenuation",
        ):
            self._episode_sums[name] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        # The route actor and recurrent residual receive the exact same sampled
        # observation, including the same LiDAR noise/dropout realization.
        self._last_policy_observation = observations["policy"].detach()
        return observations

    def _route_actions(self) -> torch.Tensor:
        if not hasattr(self, "_last_policy_observation"):
            self._last_policy_observation = super()._get_observations()["policy"].detach()
        normalized = (
            self._last_policy_observation - self._frozen_route_obs_mean
        ) / (self._frozen_route_obs_std + 1.0e-2)
        with torch.inference_mode():
            return self._frozen_route_actor(normalized).clamp(-1.0, 1.0)

    def _compose_residual_actions(
        self, base_actions: torch.Tensor, residual_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_actions = residual_actions.clamp(-1.0, 1.0)
        brake_fraction = torch.relu(-residual_actions[:, 0])
        angular_attenuation = (
            self.cfg.maximum_angular_attenuation * torch.relu(-residual_actions[:, 1])
        )
        base_forward_fraction = ((base_actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        combined_forward_fraction = base_forward_fraction * (1.0 - brake_fraction)
        combined = torch.stack(
            (
                combined_forward_fraction * 2.0 - 1.0,
                base_actions[:, 1] * (1.0 - angular_attenuation),
            ),
            dim=1,
        )
        return combined, brake_fraction, angular_attenuation

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._residual_actions = actions.clone().clamp(-1.0, 1.0)
        self._base_actions = self._route_actions()
        combined, _, _ = self._compose_residual_actions(
            self._base_actions, self._residual_actions
        )
        self._requested_combined_actions = combined

        self._residual_action_history[:, 2] = self._residual_action_history[:, 1]
        self._residual_action_history[:, 1] = self._residual_action_history[:, 0]
        self._residual_action_history[:, 0] = self._residual_actions
        self._applied_residual_actions = torch.gather(
            self._residual_action_history,
            1,
            self._action_latency_steps.view(-1, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)
        self._applied_brake_fraction = torch.relu(-self._applied_residual_actions[:, 0])
        self._applied_angular_attenuation = (
            self.cfg.maximum_angular_attenuation
            * torch.relu(-self._applied_residual_actions[:, 1])
        )
        super()._pre_physics_step(combined)

    def _get_rewards(self) -> torch.Tensor:
        rewards = super()._get_rewards()
        front_minimum = torch.amin(self._lidar_ranges()[:, 16:21], dim=1)
        closing_delta = (self._previous_front_minimum - front_minimum).clamp_min(0.0)
        closing = (
            (front_minimum < self.cfg.safety_closing_distance_m)
            & (closing_delta > self.cfg.safety_closing_delta_m)
        ).float()
        clear = (front_minimum > self.cfg.safety_clear_distance_m).float()
        normalized_forward = ((self._actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        brake_while_closing = (
            closing
            * self._applied_brake_fraction
            * self.cfg.reward_brake_while_closing
        )
        unmitigated_closing = (
            closing
            * (1.0 - self._applied_brake_fraction)
            * normalized_forward
            * self.cfg.penalty_unmitigated_closing
        )
        unnecessary_brake = (
            clear * self._applied_brake_fraction * self.cfg.penalty_unnecessary_brake
        )
        clear_path_angular_attenuation = (
            clear
            * self._applied_angular_attenuation
            * self.cfg.penalty_clear_path_angular_attenuation
        )
        self._episode_sums["brake_while_closing"] += brake_while_closing
        self._episode_sums["unmitigated_closing"] += unmitigated_closing
        self._episode_sums["unnecessary_brake"] += unnecessary_brake
        self._episode_sums["clear_path_angular_attenuation"] += clear_path_angular_attenuation
        self._previous_front_minimum.copy_(front_minimum)
        return (
            rewards
            + brake_while_closing
            + unmitigated_closing
            + unnecessary_brake
            + clear_path_angular_attenuation
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if not hasattr(self, "_residual_action_history"):
            return
        self._residual_action_history[env_ids] = 0.0
        self._residual_actions[env_ids] = 0.0
        self._applied_residual_actions[env_ids] = 0.0
        self._applied_brake_fraction[env_ids] = 0.0
        self._applied_angular_attenuation[env_ids] = 0.0
        self._previous_front_minimum[env_ids] = torch.amin(
            self._lidar_ranges()[env_ids, 16:21], dim=1
        )
        self.extras["safety_residual"] = {
            "frozen_route_checkpoint": str(self._frozen_route_checkpoint_path),
            "frozen_route_checkpoint_sha256": self._frozen_route_checkpoint_actual_sha256,
            "maximum_angular_attenuation": self.cfg.maximum_angular_attenuation,
        }


@configclass
class AishaPhase3ClearancePlannerEnvCfg(AishaPhase3SafetyResidualEnvCfg):
    """Clearance-projected local steering with an independent protective stop."""

    # Action 0 retains the proven residual brake boundary. Action 1 requests a
    # small signed correction around the frozen route actor; it is never sent
    # directly to the wheels and may be rejected by the local planner.
    maximum_lateral_correction_rad_s = 0.35

    # The projector tests the measured rectangular footprint against the
    # uncorrupted 10 Hz LiDAR hit cloud. A rectangle, rather than the robot's
    # much larger pivot-sweep circle, preserves valid transit through the
    # plan-assumed 1.40 m presentation doors.
    planner_activation_range_m = 2.20
    planner_prediction_horizon_s = 1.00
    planner_prediction_samples = 5
    planner_footprint_margin_m = 0.08
    planner_minimum_predicted_clearance_m = 0.04
    planner_minimum_clearance_improvement_m = 0.03
    planner_allowed_safe_clearance_degradation_m = 0.02
    planner_goal_alignment_tolerance_rad = math.radians(20.0)

    # These are clearances beyond the exact per-ray rectangular envelope. The
    # release threshold is larger to prevent one-scan stop/start chatter.
    protective_stop_front_ray_start = 15
    protective_stop_front_ray_end = 22
    protective_stop_trigger_clearance_m = 0.60
    protective_stop_release_clearance_m = 0.75

    reward_clearance_improvement = 0.25
    penalty_rejected_steering_request = -0.01
    penalty_clear_path_steering_request = -0.01
    penalty_protective_stop_intervention = -0.03


class AishaPhase3ClearancePlannerEnv(AishaPhase3SafetyResidualEnv):
    """Train a bounded local avoidance request behind hard runtime gates.

    The frozen network remains the map/route authority. The recurrent policy
    may brake and request a small signed angular correction. The request is
    accepted only if a short-horizon rectangular-footprint projection remains
    clear and route aligned. An independent LiDAR latch can always remove
    forward motion after domain-randomization latency has been applied.
    """

    cfg: AishaPhase3ClearancePlannerEnvCfg

    def __init__(
        self,
        cfg: AishaPhase3ClearancePlannerEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.planner_prediction_samples < 1:
            raise ValueError("planner_prediction_samples must be positive")
        if not 0.0 < self.cfg.planner_prediction_horizon_s <= 2.0:
            raise ValueError("planner_prediction_horizon_s must be in (0, 2]")
        if (
            self.cfg.protective_stop_release_clearance_m
            <= self.cfg.protective_stop_trigger_clearance_m
        ):
            raise ValueError("protective stop release clearance must exceed trigger clearance")
        if (
            self.cfg.maximum_lateral_correction_rad_s
            > self.cfg.angular_velocity_max_rad_s
        ):
            raise ValueError("lateral correction cannot exceed the task angular limit")

        self._planner_ray_angles = torch.deg2rad(
            torch.arange(-180.0, 180.0, 10.0, device=self.device)
        )
        self._planner_prediction_times = torch.linspace(
            self.cfg.planner_prediction_horizon_s / self.cfg.planner_prediction_samples,
            self.cfg.planner_prediction_horizon_s,
            self.cfg.planner_prediction_samples,
            device=self.device,
        )
        self._base_action_history = torch.zeros((self.num_envs, 3, 2), device=self.device)
        self._protective_stop_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._protective_stop_intervened = torch.zeros_like(self._protective_stop_latched)
        self._planner_request_accepted = torch.zeros_like(self._protective_stop_latched)
        self._planner_request_active = torch.zeros_like(self._protective_stop_latched)
        self._planner_baseline_clearance = torch.zeros(self.num_envs, device=self.device)
        self._planner_candidate_clearance = torch.zeros(self.num_envs, device=self.device)
        self._planner_applied_clearance = torch.zeros(self.num_envs, device=self.device)
        self._applied_steering_request = torch.zeros(self.num_envs, device=self.device)
        # Episode-level counters are evaluation telemetry only. They expose
        # whether a failure is a stop-latch stall, a rejected steering request,
        # or a clearance miss without changing the controller boundary.
        self._episode_planner_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_protective_stop_steps = torch.zeros_like(
            self._episode_planner_steps
        )
        self._episode_stop_intervention_steps = torch.zeros_like(
            self._episode_planner_steps
        )
        self._episode_planner_request_steps = torch.zeros_like(
            self._episode_planner_steps
        )
        self._episode_planner_accept_steps = torch.zeros_like(
            self._episode_planner_steps
        )
        self._episode_abs_steering_request_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_brake_fraction_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_base_angular_command_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_applied_angular_command_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_minimum_applied_clearance = torch.full(
            (self.num_envs,), self.cfg.lidar_max_range_m, device=self.device
        )
        for name in (
            "clearance_improvement",
            "rejected_steering_request",
            "clear_path_steering_request",
            "protective_stop_intervention",
        ):
            self._episode_sums[name] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )

    def _compose_planner_request(
        self, base_actions: torch.Tensor, residual_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map policy output to brake plus a bounded steering proposal."""
        residual_actions = residual_actions.clamp(-1.0, 1.0)
        brake_fraction = torch.relu(-residual_actions[:, 0])
        base_forward_fraction = ((base_actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        combined_forward_fraction = base_forward_fraction * (1.0 - brake_fraction)
        maximum_normalized_correction = (
            self.cfg.maximum_lateral_correction_rad_s
            / self.cfg.angular_velocity_max_rad_s
        )
        requested_angular = (
            base_actions[:, 1] + maximum_normalized_correction * residual_actions[:, 1]
        ).clamp(-1.0, 1.0)
        applied_request = requested_angular - base_actions[:, 1]
        combined = torch.stack(
            (combined_forward_fraction * 2.0 - 1.0, requested_angular), dim=1
        )
        return combined, brake_fraction, applied_request

    def _predict_candidate_geometry(
        self,
        candidate_actions: torch.Tensor,
        exact_lidar_ranges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return minimum swept clearance and terminal goal-heading error.

        ``candidate_actions`` has shape ``(num_envs, candidates, 2)``. LiDAR
        hits are converted from the sensor origin to the current base frame,
        then tested against the oriented rectangular footprint at every
        predicted unicycle pose.
        """
        minimum, maximum = self.cfg.linear_velocity_range_mps
        linear = minimum + (candidate_actions[..., 0] + 1.0) * 0.5 * (maximum - minimum)
        angular = candidate_actions[..., 1] * self.cfg.angular_velocity_max_rad_s
        times = self._planner_prediction_times.view(1, 1, -1)
        yaw = angular.unsqueeze(-1) * times
        near_straight = torch.abs(angular) < 1.0e-4
        safe_angular = torch.where(near_straight, torch.ones_like(angular), angular)
        pose_x = torch.where(
            near_straight.unsqueeze(-1),
            linear.unsqueeze(-1) * times,
            linear.unsqueeze(-1) / safe_angular.unsqueeze(-1) * torch.sin(yaw),
        )
        pose_y = torch.where(
            near_straight.unsqueeze(-1),
            torch.zeros_like(yaw),
            linear.unsqueeze(-1) / safe_angular.unsqueeze(-1) * (1.0 - torch.cos(yaw)),
        )

        point_x = (
            self.cfg.lidar_x_m
            + exact_lidar_ranges * torch.cos(self._planner_ray_angles).unsqueeze(0)
        )
        point_y = exact_lidar_ranges * torch.sin(self._planner_ray_angles).unsqueeze(0)
        delta_x = point_x[:, None, None, :] - pose_x[..., None]
        delta_y = point_y[:, None, None, :] - pose_y[..., None]
        cos_yaw = torch.cos(yaw)[..., None]
        sin_yaw = torch.sin(yaw)[..., None]
        local_x = cos_yaw * delta_x + sin_yaw * delta_y
        local_y = -sin_yaw * delta_x + cos_yaw * delta_y

        rear = self.cfg.robot_rear_x_m - self.cfg.planner_footprint_margin_m
        front = self.cfg.robot_front_x_m + self.cfg.planner_footprint_margin_m
        half_width = self.cfg.robot_half_width_m + self.cfg.planner_footprint_margin_m
        outside_x = torch.maximum(
            torch.maximum(rear - local_x, local_x - front),
            torch.zeros_like(local_x),
        )
        outside_y = torch.relu(torch.abs(local_y) - half_width)
        outside_distance = torch.sqrt(outside_x.square() + outside_y.square())
        inside = (
            (local_x >= rear)
            & (local_x <= front)
            & (torch.abs(local_y) <= half_width)
        )
        penetration = torch.minimum(
            torch.minimum(local_x - rear, front - local_x),
            half_width - torch.abs(local_y),
        )
        signed_clearance = torch.where(inside, -penetration, outside_distance)
        minimum_clearance = torch.amin(signed_clearance, dim=(2, 3))

        goal_x, goal_y, _, _ = self._goal_geometry()
        final_x = pose_x[..., -1]
        final_y = pose_y[..., -1]
        final_yaw = yaw[..., -1]
        goal_delta_x = goal_x.unsqueeze(1) - final_x
        goal_delta_y = goal_y.unsqueeze(1) - final_y
        goal_x_at_horizon = (
            torch.cos(final_yaw) * goal_delta_x + torch.sin(final_yaw) * goal_delta_y
        )
        goal_y_at_horizon = (
            -torch.sin(final_yaw) * goal_delta_x + torch.cos(final_yaw) * goal_delta_y
        )
        goal_heading_error = torch.abs(torch.atan2(goal_y_at_horizon, goal_x_at_horizon))
        return minimum_clearance, goal_heading_error

    def _clearance_project_actions(
        self,
        delayed_base_actions: torch.Tensor,
        delayed_requested_actions: torch.Tensor,
        exact_lidar_ranges: torch.Tensor,
    ) -> torch.Tensor:
        """Accept only safe, useful, route-consistent lateral corrections."""
        route_aligned = torch.stack(
            (delayed_requested_actions[:, 0], delayed_base_actions[:, 1]), dim=1
        )
        candidates = torch.stack((route_aligned, delayed_requested_actions), dim=1)
        clearances, heading_errors = self._predict_candidate_geometry(
            candidates, exact_lidar_ranges
        )
        baseline_clearance = clearances[:, 0]
        candidate_clearance = clearances[:, 1]
        baseline_safe = (
            baseline_clearance >= self.cfg.planner_minimum_predicted_clearance_m
        )
        candidate_safe = (
            candidate_clearance >= self.cfg.planner_minimum_predicted_clearance_m
        )
        clearance_preserved = torch.where(
            baseline_safe,
            candidate_clearance
            >= baseline_clearance - self.cfg.planner_allowed_safe_clearance_degradation_m,
            candidate_clearance
            >= baseline_clearance + self.cfg.planner_minimum_clearance_improvement_m,
        )
        goal_alignment_preserved = (
            heading_errors[:, 1]
            <= heading_errors[:, 0] + self.cfg.planner_goal_alignment_tolerance_rad
        )
        correction_requested = (
            torch.abs(delayed_requested_actions[:, 1] - delayed_base_actions[:, 1])
            > 1.0e-5
        )
        near_obstacle = (
            torch.amin(exact_lidar_ranges, dim=1) < self.cfg.planner_activation_range_m
        )
        accepted = (
            correction_requested
            & near_obstacle
            & candidate_safe
            & clearance_preserved
            & goal_alignment_preserved
        )
        projected = torch.where(accepted.unsqueeze(1), delayed_requested_actions, route_aligned)
        self._planner_request_active = correction_requested & near_obstacle
        self._planner_request_accepted = accepted
        self._planner_baseline_clearance.copy_(baseline_clearance)
        self._planner_candidate_clearance.copy_(candidate_clearance)
        self._planner_applied_clearance.copy_(
            torch.where(accepted, candidate_clearance, baseline_clearance)
        )
        return projected

    def _apply_protective_stop(
        self, actions: torch.Tensor, exact_lidar_ranges: torch.Tensor
    ) -> torch.Tensor:
        """Remove forward motion using an independent hysteretic LiDAR gate."""
        ray_slice = slice(
            self.cfg.protective_stop_front_ray_start,
            self.cfg.protective_stop_front_ray_end,
        )
        envelope = self._lidar_envelope_ranges[ray_slice].unsqueeze(0)
        front_ranges = exact_lidar_ranges[:, ray_slice]
        trigger = torch.any(
            front_ranges
            <= envelope + self.cfg.protective_stop_trigger_clearance_m,
            dim=1,
        )
        release = torch.all(
            front_ranges
            >= envelope + self.cfg.protective_stop_release_clearance_m,
            dim=1,
        )
        self._protective_stop_latched |= trigger
        self._protective_stop_latched &= ~release
        moving_forward = actions[:, 0] > -1.0 + 1.0e-6
        self._protective_stop_intervened = self._protective_stop_latched & moving_forward
        protected = actions.clone()
        protected[:, 0] = torch.where(
            self._protective_stop_latched,
            torch.full_like(protected[:, 0], -1.0),
            protected[:, 0],
        )
        return protected

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Delay models the nominal command path. Both safety gates run after
        # that delay, so neither can be bypassed by the randomized latency.
        self._update_dynamic_obstacles()
        self._residual_actions = actions.clone().clamp(-1.0, 1.0)
        self._base_actions = self._route_actions()
        requested, _, _ = self._compose_planner_request(
            self._base_actions, self._residual_actions
        )
        self._requested_combined_actions = requested

        for history, current in (
            (self._action_history, requested),
            (self._base_action_history, self._base_actions),
            (self._residual_action_history, self._residual_actions),
        ):
            history[:, 2] = history[:, 1]
            history[:, 1] = history[:, 0]
            history[:, 0] = current
        gather_index = self._action_latency_steps.view(-1, 1, 1).expand(-1, 1, 2)
        delayed_request = torch.gather(self._action_history, 1, gather_index).squeeze(1)
        delayed_base = torch.gather(self._base_action_history, 1, gather_index).squeeze(1)
        self._applied_residual_actions = torch.gather(
            self._residual_action_history, 1, gather_index
        ).squeeze(1)
        self._applied_brake_fraction = torch.relu(-self._applied_residual_actions[:, 0])
        _, _, self._applied_steering_request = self._compose_planner_request(
            delayed_base, self._applied_residual_actions
        )

        exact_lidar_ranges = self._lidar_ranges()
        projected = self._clearance_project_actions(
            delayed_base, delayed_request, exact_lidar_ranges
        )
        protected = self._apply_protective_stop(projected, exact_lidar_ranges)
        self._episode_planner_steps += 1
        self._episode_protective_stop_steps += self._protective_stop_latched.long()
        self._episode_stop_intervention_steps += self._protective_stop_intervened.long()
        self._episode_planner_request_steps += self._planner_request_active.long()
        self._episode_planner_accept_steps += self._planner_request_accepted.long()
        self._episode_abs_steering_request_sum += torch.abs(
            self._applied_steering_request
        )
        self._episode_brake_fraction_sum += self._applied_brake_fraction
        self._episode_base_angular_command_sum += delayed_base[:, 1]
        self._episode_applied_angular_command_sum += protected[:, 1]
        self._episode_minimum_applied_clearance.copy_(
            torch.minimum(
                self._episode_minimum_applied_clearance,
                self._planner_applied_clearance,
            )
        )
        self._previous_actions.copy_(self._actions)
        self._actions = protected.clamp(-1.0, 1.0)

        minimum, maximum = self.cfg.linear_velocity_range_mps
        linear = minimum + (self._actions[:, 0] + 1.0) * 0.5 * (maximum - minimum)
        angular = self._actions[:, 1] * self.cfg.angular_velocity_max_rad_s
        half_track = self.cfg.wheel_track_m * self._wheel_track_scale / 2.0
        wheel_radius = self.cfg.wheel_radius_m * self._wheel_radius_scale
        self._wheel_targets[:, 0] = (linear - angular * half_track) / wheel_radius
        self._wheel_targets[:, 1] = (linear + angular * half_track) / wheel_radius
        self._wheel_targets *= self._motor_strength
        self._wheel_targets.clamp_(
            -self.cfg.wheel_speed_limit_rad_s,
            self.cfg.wheel_speed_limit_rad_s,
        )

    def _get_rewards(self) -> torch.Tensor:
        rewards = AishaPhase3DynamicDREnv._get_rewards(self)
        front_minimum = torch.amin(self._lidar_ranges()[:, 16:21], dim=1)
        closing_delta = (self._previous_front_minimum - front_minimum).clamp_min(0.0)
        closing = (
            (front_minimum < self.cfg.safety_closing_distance_m)
            & (closing_delta > self.cfg.safety_closing_delta_m)
        ).float()
        clear = (front_minimum > self.cfg.safety_clear_distance_m).float()
        normalized_forward = ((self._actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        brake_while_closing = (
            closing * self._applied_brake_fraction * self.cfg.reward_brake_while_closing
        )
        unmitigated_closing = (
            closing
            * (1.0 - self._applied_brake_fraction)
            * normalized_forward
            * self.cfg.penalty_unmitigated_closing
        )
        unnecessary_brake = (
            clear * self._applied_brake_fraction * self.cfg.penalty_unnecessary_brake
        )
        clearance_improvement = (
            self._planner_request_accepted.float()
            * torch.relu(
                self._planner_candidate_clearance - self._planner_baseline_clearance
            ).clamp_max(0.50)
            * self.cfg.reward_clearance_improvement
        )
        rejected_request = (
            (self._planner_request_active & ~self._planner_request_accepted).float()
            * torch.abs(self._applied_steering_request)
            * self.cfg.penalty_rejected_steering_request
        )
        clear_path_request = (
            clear
            * torch.abs(self._applied_steering_request)
            * self.cfg.penalty_clear_path_steering_request
        )
        stop_intervention = (
            self._protective_stop_intervened.float()
            * self.cfg.penalty_protective_stop_intervention
        )
        for name, value in (
            ("brake_while_closing", brake_while_closing),
            ("unmitigated_closing", unmitigated_closing),
            ("unnecessary_brake", unnecessary_brake),
            ("clearance_improvement", clearance_improvement),
            ("rejected_steering_request", rejected_request),
            ("clear_path_steering_request", clear_path_request),
            ("protective_stop_intervention", stop_intervention),
        ):
            self._episode_sums[name] += value
        self._previous_front_minimum.copy_(front_minimum)
        return (
            rewards
            + brake_while_closing
            + unmitigated_closing
            + unnecessary_brake
            + clearance_improvement
            + rejected_request
            + clear_path_request
            + stop_intervention
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if hasattr(self, "_episode_planner_steps"):
            outcomes = self.extras["episode_outcomes"]
            outcomes.update(
                {
                    "planner_steps": self._episode_planner_steps.clone(),
                    "protective_stop_steps": (
                        self._episode_protective_stop_steps.clone()
                    ),
                    "protective_stop_intervention_steps": (
                        self._episode_stop_intervention_steps.clone()
                    ),
                    "planner_request_steps": (
                        self._episode_planner_request_steps.clone()
                    ),
                    "planner_accept_steps": self._episode_planner_accept_steps.clone(),
                    "abs_steering_request_sum": (
                        self._episode_abs_steering_request_sum.clone()
                    ),
                    "brake_fraction_sum": self._episode_brake_fraction_sum.clone(),
                    "base_angular_command_sum": (
                        self._episode_base_angular_command_sum.clone()
                    ),
                    "applied_angular_command_sum": (
                        self._episode_applied_angular_command_sum.clone()
                    ),
                    "minimum_applied_clearance_m": (
                        self._episode_minimum_applied_clearance.clone()
                    ),
                    "final_protective_stop_latched": (
                        self._protective_stop_latched.clone()
                    ),
                }
            )
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if not hasattr(self, "_base_action_history"):
            return
        self._base_action_history[env_ids] = 0.0
        self._protective_stop_latched[env_ids] = False
        self._protective_stop_intervened[env_ids] = False
        self._planner_request_accepted[env_ids] = False
        self._planner_request_active[env_ids] = False
        self._planner_baseline_clearance[env_ids] = 0.0
        self._planner_candidate_clearance[env_ids] = 0.0
        self._planner_applied_clearance[env_ids] = 0.0
        self._applied_steering_request[env_ids] = 0.0
        self._episode_planner_steps[env_ids] = 0
        self._episode_protective_stop_steps[env_ids] = 0
        self._episode_stop_intervention_steps[env_ids] = 0
        self._episode_planner_request_steps[env_ids] = 0
        self._episode_planner_accept_steps[env_ids] = 0
        self._episode_abs_steering_request_sum[env_ids] = 0.0
        self._episode_brake_fraction_sum[env_ids] = 0.0
        self._episode_base_angular_command_sum[env_ids] = 0.0
        self._episode_applied_angular_command_sum[env_ids] = 0.0
        self._episode_minimum_applied_clearance[env_ids] = self.cfg.lidar_max_range_m
        self.extras["clearance_planner"] = {
            "prediction_horizon_s": self.cfg.planner_prediction_horizon_s,
            "prediction_samples": self.cfg.planner_prediction_samples,
            "rectangular_footprint_margin_m": self.cfg.planner_footprint_margin_m,
            "maximum_lateral_correction_rad_s": self.cfg.maximum_lateral_correction_rad_s,
            "protective_stop_trigger_clearance_m": (
                self.cfg.protective_stop_trigger_clearance_m
            ),
            "protective_stop_release_clearance_m": (
                self.cfg.protective_stop_release_clearance_m
            ),
        }


@configclass
class AishaPhase3TargetedRecoveryEnvCfg(AishaPhase3ClearancePlannerEnvCfg):
    """Recover hard pivots while retaining the complete Phase 3L route skill."""

    # Segments 4 and 9 are the two in-office 180-degree departures. Segment 6
    # is the tight atrium-to-principal turn. Together they receive 60/96
    # (62.5%) of resets while every other route leg retains 4/96 rehearsal.
    targeted_recovery_segment_ids = (4, 6, 9)
    office_departure_segment_ids = (4, 9)
    # Phase 3M may overcome a wrong-way frozen U-turn request, but the resulting
    # command remains bounded by the 1.0 rad/s task limit and must pass the
    # exact same rectangular-footprint projection before reaching the wheels.
    maximum_lateral_correction_rad_s = 0.70
    segment_sampling_weights = (
        4.0,
        4.0,
        4.0,
        4.0,
        18.0,
        4.0,
        24.0,
        4.0,
        4.0,
        18.0,
        4.0,
        4.0,
    )

    # Recovery probes may intentionally crawl through a tight projected path;
    # allow enough wall-clock policy steps for the 5 m principal-return leg.
    episode_length_s = 100.0
    recovery_supervisor_enabled = True

    # Phase 3L model 200 requested only 0.004 rad/s of mean added steering on
    # the failed office departures. These terms reward realized heading
    # reduction and safe, correctly signed planner requests until the pivot is
    # aligned. They do not bypass projection or the independent stop latch.
    targeted_turn_alignment_rad = math.radians(25.0)
    targeted_pivot_brake_threshold_rad = math.radians(60.0)
    reward_targeted_heading_progress = 18.0
    reward_targeted_aligned_steering_request = 0.006
    penalty_targeted_wrong_steering_request = -0.08
    penalty_targeted_turn_inactivity = -0.012
    penalty_targeted_pivot_forward = -0.12
    penalty_targeted_aligned_nonforward = -0.20

    # The selected ZLTECH drive is rated for 6 Nm continuous and 18 Nm peak
    # for at most 3 seconds. Keep the ordinary actuator contract at 6 Nm and
    # expose the controller-timed peak only for a commanded stationary pivot
    # on the targeted recovery legs. The gate is intentionally one-shot per
    # episode; no unverified thermal cooldown/retrigger model is assumed.
    rated_motor_effort_limit_nm = 6.0
    peak_motor_effort_limit_nm = 18.0
    peak_motor_time_limit_s = 3.0
    peak_pivot_minimum_heading_error_rad = math.radians(60.0)
    peak_pivot_minimum_angular_command_rad_s = 0.35

    # A deterministic recovery supervisor closes the gap that PPO exploration
    # exposed but did not retain in its mean action. It may command a stopped,
    # goal-signed office pivot only after the same rectangular-footprint
    # clearance projection used by the residual planner. Hysteresis prevents
    # brake and steering chatter.
    pivot_supervisor_engage_heading_error_rad = math.radians(60.0)
    pivot_supervisor_release_heading_error_rad = math.radians(25.0)
    pivot_supervisor_angular_command_rad_s = 0.55
    office_departure_protective_release_clearance_m = 0.04
    predictive_stop_segment_ids = (6, 10, 11)
    predictive_stop_trigger_clearance_m = 0.10
    predictive_stop_release_clearance_m = 0.22
    predictive_creep_linear_velocity_mps = 0.10
    dynamic_crossing_creep_segment_ids = (10,)
    dynamic_crossing_predictive_creep_linear_velocity_mps = 0.08

    # The imported robot deliberately uses four fixed-sphere proxies for its
    # unmeasured swivel castors. Preserve their declared low-friction contact
    # class instead of overwriting every robot collider with drive-wheel
    # friction during domain randomization.
    castor_static_friction_range = (0.15, 0.25)
    castor_dynamic_friction_range = (0.10, 0.20)

    # Segment 6 reached a negative projected baseline clearance in the
    # diagnostic rollout. Give accepted clearance improvements a stronger
    # signal and penalize low applied clearance before contact.
    targeted_clearance_segment_id = 6
    targeted_low_clearance_m = 0.12
    reward_targeted_clearance_improvement = 0.75
    penalty_targeted_low_clearance = -0.06


@configclass
class AishaPhase3TargetedRecoveryTrainingEnvCfg(AishaPhase3TargetedRecoveryEnvCfg):
    """Corrected-physics PPO curriculum without deterministic recovery actions."""

    recovery_supervisor_enabled = False
    episode_length_s = 70.0


class AishaPhase3TargetedRecoveryEnv(AishaPhase3ClearancePlannerEnv):
    """Retention-safe Phase 3M curriculum for office departures and segment 6."""

    cfg: AishaPhase3TargetedRecoveryEnvCfg

    def __init__(
        self,
        cfg: AishaPhase3TargetedRecoveryEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        recovery_ids = tuple(int(value) for value in self.cfg.targeted_recovery_segment_ids)
        if len(set(recovery_ids)) != len(recovery_ids):
            raise ValueError("targeted recovery segment ids must be unique")
        if any(value < 0 or value >= len(ROUTE_SEGMENTS) for value in recovery_ids):
            raise ValueError("targeted recovery segment id is outside ROUTE_SEGMENTS")
        if self.cfg.targeted_clearance_segment_id not in recovery_ids:
            raise ValueError("targeted clearance segment must be in the recovery set")
        self._targeted_recovery_ids = torch.tensor(
            recovery_ids, dtype=torch.long, device=self.device
        )
        if self.cfg.rated_motor_effort_limit_nm != 6.0:
            raise ValueError("rated motor effort must retain the 6 Nm robot contract")
        if self.cfg.peak_motor_effort_limit_nm != 18.0:
            raise ValueError("peak motor effort must match the declared 18 Nm motor peak")
        if not 0.0 < self.cfg.peak_motor_time_limit_s <= 3.0:
            raise ValueError("peak motor time limit must be in (0, 3] seconds")
        self._peak_torque_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._peak_torque_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pivot_supervisor_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._predictive_stop_latched = torch.zeros_like(
            self._pivot_supervisor_latched
        )
        self._recovery_supervisor_brake_active = torch.zeros_like(
            self._pivot_supervisor_latched
        )
        self._pivot_supervisor_steering_active = torch.zeros_like(
            self._pivot_supervisor_latched
        )
        self._office_departure_protective_release_active = torch.zeros_like(
            self._pivot_supervisor_latched
        )
        self._episode_peak_torque_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_pivot_supervisor_steps = torch.zeros_like(
            self._episode_peak_torque_steps
        )
        self._episode_predictive_stop_steps = torch.zeros_like(
            self._episode_peak_torque_steps
        )
        self._episode_pivot_supervisor_steering_steps = torch.zeros_like(
            self._episode_peak_torque_steps
        )
        self._episode_office_departure_protective_release_steps = torch.zeros_like(
            self._episode_peak_torque_steps
        )
        self._castor_material_shape_ids = self._resolve_material_shape_ids(
            (
                "castor_fl_link",
                "castor_fr_link",
                "castor_rl_link",
                "castor_rr_link",
            )
        )
        self._castor_static_friction = torch.full(
            (self.num_envs,), 0.20, device=self.device
        )
        self._castor_dynamic_friction = torch.full(
            (self.num_envs,), 0.15, device=self.device
        )
        for name in (
            "targeted_heading_progress",
            "targeted_aligned_steering_request",
            "targeted_wrong_steering_request",
            "targeted_turn_inactivity",
            "targeted_pivot_forward",
            "targeted_aligned_nonforward",
            "targeted_clearance_improvement",
            "targeted_low_clearance",
        ):
            self._episode_sums[name] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )

        if (
            self.cfg.pivot_supervisor_release_heading_error_rad
            >= self.cfg.pivot_supervisor_engage_heading_error_rad
        ):
            raise ValueError("pivot supervisor release must be below its engage angle")
        if not (
            self.cfg.peak_pivot_minimum_angular_command_rad_s
            <= self.cfg.pivot_supervisor_angular_command_rad_s
            <= self.cfg.angular_velocity_max_rad_s
        ):
            raise ValueError("pivot supervisor angular command is outside its motor/task limits")
        if (
            self.cfg.predictive_stop_release_clearance_m
            <= self.cfg.predictive_stop_trigger_clearance_m
        ):
            raise ValueError("predictive stop release must exceed its trigger clearance")
        if not (
            self.cfg.linear_velocity_range_mps[0]
            <= self.cfg.predictive_creep_linear_velocity_mps
            < self.cfg.linear_velocity_range_mps[1]
        ):
            raise ValueError("predictive creep speed is outside the task linear range")
        if not (
            self.cfg.linear_velocity_range_mps[0]
            <= self.cfg.dynamic_crossing_predictive_creep_linear_velocity_mps
            <= self.cfg.predictive_creep_linear_velocity_mps
        ):
            raise ValueError("dynamic crossing creep must not exceed predictive creep")
        if any(
            value < 0 or value >= len(ROUTE_SEGMENTS)
            for value in self.cfg.predictive_stop_segment_ids
        ):
            raise ValueError("predictive stop segment id is outside ROUTE_SEGMENTS")
        self._office_departure_ids = torch.tensor(
            self.cfg.office_departure_segment_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._predictive_stop_segment_ids = torch.tensor(
            self.cfg.predictive_stop_segment_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._dynamic_crossing_creep_segment_ids = torch.tensor(
            self.cfg.dynamic_crossing_creep_segment_ids,
            dtype=torch.long,
            device=self.device,
        )

    def _apply_protective_stop(
        self, actions: torch.Tensor, exact_lidar_ranges: torch.Tensor
    ) -> torch.Tensor:
        """Release the front-ray latch after a clear, aligned office pivot.

        The ordinary gate uses a generous 0.60 m forward buffer. That buffer
        correctly stops transit but cannot release inside the plan-assumed
        1.40 m presentation rooms. The exception requires both route alignment
        and the full rectangular-footprint prediction to be clear.
        """
        protected = super()._apply_protective_stop(actions, exact_lidar_ranges)
        if not self.cfg.recovery_supervisor_enabled:
            return protected
        _, _, _, heading_error = self._goal_geometry()
        office_departure = torch.any(
            self._segment_ids.unsqueeze(1) == self._office_departure_ids.unsqueeze(0),
            dim=1,
        )
        self._office_departure_protective_release_active = (
            office_departure
            & (
                torch.abs(heading_error)
                <= self.cfg.pivot_supervisor_release_heading_error_rad
            )
            & (
                self._planner_applied_clearance
                >= self.cfg.office_departure_protective_release_clearance_m
            )
        )
        self._protective_stop_latched &= ~(
            self._office_departure_protective_release_active
        )
        self._protective_stop_intervened &= ~(
            self._office_departure_protective_release_active
        )
        protected = torch.where(
            self._office_departure_protective_release_active.unsqueeze(1),
            actions,
            protected,
        )
        self._episode_office_departure_protective_release_steps += (
            self._office_departure_protective_release_active.long()
        )
        return protected

    def _resolve_material_shape_ids(self, body_names: tuple[str, ...]) -> list[int]:
        """Map articulation body names to the flattened PhysX shape buffer."""
        shape_counts: list[int] = []
        for link_path in self._robot.root_physx_view.link_paths[0]:
            rigid_view = self._robot._physics_sim_view.create_rigid_body_view(link_path)
            shape_counts.append(int(rigid_view.max_shapes))
        if len(shape_counts) != len(self._robot.body_names):
            raise RuntimeError("articulation body and material-shape tables disagree")
        if sum(shape_counts) != self._robot.root_physx_view.max_shapes:
            raise RuntimeError("flattened articulation material-shape count is inconsistent")

        shape_ids: list[int] = []
        for body_name in body_names:
            body_ids, _ = self._robot.find_bodies(body_name)
            if len(body_ids) != 1:
                raise RuntimeError(
                    f"expected one {body_name} for castor material routing, found {body_ids}"
                )
            body_id = int(body_ids[0])
            start = sum(shape_counts[:body_id])
            shape_ids.extend(range(start, start + shape_counts[body_id]))
        if not shape_ids:
            raise RuntimeError("castor material routing resolved no collision shapes")
        return shape_ids

    def _randomize_physics(self, env_ids: torch.Tensor) -> None:
        super()._randomize_physics(env_ids)
        # The parent samples the drive contact range across all shapes. Restore
        # the four sphere-proxy castors to their own low-friction uncertainty
        # band so a simulated pivot does not scrub four artificial fixed feet.
        count = len(env_ids)
        strength = self._curriculum_strength()
        self._castor_static_friction[env_ids] = self._blended_uniform(
            count, self.cfg.castor_static_friction_range, 0.20, strength
        )
        self._castor_dynamic_friction[env_ids] = torch.minimum(
            self._blended_uniform(
                count, self.cfg.castor_dynamic_friction_range, 0.15, strength
            ),
            self._castor_static_friction[env_ids],
        )
        cpu_ids = env_ids.cpu()
        shape_ids = torch.as_tensor(self._castor_material_shape_ids, dtype=torch.long)
        materials = self._robot.root_physx_view.get_material_properties()
        materials[cpu_ids[:, None], shape_ids[None, :], 0] = (
            self._castor_static_friction[env_ids].cpu().unsqueeze(1)
        )
        materials[cpu_ids[:, None], shape_ids[None, :], 1] = (
            self._castor_dynamic_friction[env_ids].cpu().unsqueeze(1)
        )
        materials[cpu_ids[:, None], shape_ids[None, :], 2] = 0.0
        self._robot.root_physx_view.set_material_properties(materials, cpu_ids)

    def _update_pivot_torque_limits(
        self,
        applied_actions: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> None:
        """Schedule the specified motor peak for stopped, large-angle pivots."""
        targeted = torch.any(
            self._segment_ids.unsqueeze(1) == self._targeted_recovery_ids.unsqueeze(0),
            dim=1,
        )
        zero_translation_command = applied_actions[:, 0] <= -1.0 + 1.0e-6
        active_turn_command = (
            torch.abs(applied_actions[:, 1]) * self.cfg.angular_velocity_max_rad_s
            >= self.cfg.peak_pivot_minimum_angular_command_rad_s
        )
        large_heading_error = (
            torch.abs(heading_error)
            >= self.cfg.peak_pivot_minimum_heading_error_rad
        )
        time_available = (
            self._peak_torque_elapsed_s + 0.5 * self.step_dt
            <= self.cfg.peak_motor_time_limit_s
        )
        self._peak_torque_active = (
            targeted
            & zero_translation_command
            & active_turn_command
            & large_heading_error
            & time_available
        )
        self._peak_torque_elapsed_s.add_(
            self._peak_torque_active.float() * self.step_dt
        ).clamp_max_(self.cfg.peak_motor_time_limit_s)
        self._episode_peak_torque_steps += self._peak_torque_active.long()

        effort_limits = torch.full(
            (self.num_envs, len(self._wheel_ids)),
            self.cfg.rated_motor_effort_limit_nm,
            device=self.device,
        )
        effort_limits[self._peak_torque_active] = self.cfg.peak_motor_effort_limit_nm
        self._robot.write_joint_effort_limit_to_sim(
            effort_limits,
            joint_ids=self._wheel_ids,
        )

    def _apply_recovery_supervisor(self, heading_error: torch.Tensor) -> None:
        """Stop for bounded pivots and cap translation near predicted contact.

        Normal transit remains under the route actor and learned residual. For
        the two declared 180-degree departures, this layer supplies the
        route-planner turn sign and a minimum pivot rate only when a projected
        one-second footprint sweep is clear.
        """
        office_departure = torch.any(
            self._segment_ids.unsqueeze(1) == self._office_departure_ids.unsqueeze(0),
            dim=1,
        )
        absolute_heading_error = torch.abs(heading_error)
        self._pivot_supervisor_latched |= office_departure & (
            absolute_heading_error
            >= self.cfg.pivot_supervisor_engage_heading_error_rad
        )
        self._pivot_supervisor_latched &= ~(
            (~office_departure)
            | (
                absolute_heading_error
                <= self.cfg.pivot_supervisor_release_heading_error_rad
            )
        )

        predictive_scope = torch.any(
            self._segment_ids.unsqueeze(1)
            == self._predictive_stop_segment_ids.unsqueeze(0),
            dim=1,
        )
        self._predictive_stop_latched |= predictive_scope & (
            self._planner_applied_clearance
            <= self.cfg.predictive_stop_trigger_clearance_m
        )
        self._predictive_stop_latched &= ~(
            (~predictive_scope)
            | (
                self._planner_applied_clearance
                >= self.cfg.predictive_stop_release_clearance_m
            )
        )

        self._recovery_supervisor_brake_active = (
            self._pivot_supervisor_latched | self._predictive_stop_latched
        )

        normalized_pivot_rate = (
            self.cfg.pivot_supervisor_angular_command_rad_s
            / self.cfg.angular_velocity_max_rad_s
        )
        pivot_candidate = self._actions.clone()
        pivot_candidate[:, 0] = -1.0
        pivot_candidate[:, 1] = torch.sign(heading_error) * torch.maximum(
            torch.abs(self._actions[:, 1]),
            torch.full_like(self._actions[:, 1], normalized_pivot_rate),
        )
        pivot_clearance, _ = self._predict_candidate_geometry(
            pivot_candidate.unsqueeze(1),
            self._lidar_ranges(),
        )
        self._pivot_supervisor_steering_active = (
            self._pivot_supervisor_latched
            & (
                pivot_clearance[:, 0]
                >= self.cfg.planner_minimum_predicted_clearance_m
            )
        )
        minimum, maximum = self.cfg.linear_velocity_range_mps
        dynamic_crossing_creep = torch.any(
            self._segment_ids.unsqueeze(1)
            == self._dynamic_crossing_creep_segment_ids.unsqueeze(0),
            dim=1,
        )
        creep_velocity = torch.where(
            dynamic_crossing_creep,
            torch.full_like(
                self._actions[:, 0],
                self.cfg.dynamic_crossing_predictive_creep_linear_velocity_mps,
            ),
            torch.full_like(
                self._actions[:, 0],
                self.cfg.predictive_creep_linear_velocity_mps,
            ),
        )
        normalized_creep = (
            2.0
            * (creep_velocity - minimum)
            / (maximum - minimum)
            - 1.0
        )
        self._actions[:, 0] = torch.where(
            self._pivot_supervisor_latched,
            torch.full_like(self._actions[:, 0], -1.0),
            self._actions[:, 0],
        )
        self._actions[:, 0] = torch.where(
            self._predictive_stop_latched,
            torch.minimum(
                self._actions[:, 0],
                normalized_creep,
            ),
            self._actions[:, 0],
        )
        self._actions[:, 1] = torch.where(
            self._pivot_supervisor_steering_active,
            pivot_candidate[:, 1],
            self._actions[:, 1],
        )
        predictive_brake_fraction = 1.0 - (
            creep_velocity - minimum
        ) / (maximum - minimum)
        supervisor_brake_fraction = torch.where(
            self._pivot_supervisor_latched,
            torch.ones_like(self._applied_brake_fraction),
            torch.where(
                self._predictive_stop_latched,
                predictive_brake_fraction,
                torch.zeros_like(self._applied_brake_fraction),
            ),
        )
        self._applied_brake_fraction = torch.maximum(
            self._applied_brake_fraction,
            supervisor_brake_fraction,
        )
        self._episode_pivot_supervisor_steps += self._pivot_supervisor_latched.long()
        self._episode_predictive_stop_steps += self._predictive_stop_latched.long()
        self._episode_pivot_supervisor_steering_steps += (
            self._pivot_supervisor_steering_active.long()
        )

        linear = minimum + (self._actions[:, 0] + 1.0) * 0.5 * (maximum - minimum)
        angular = self._actions[:, 1] * self.cfg.angular_velocity_max_rad_s
        half_track = self.cfg.wheel_track_m * self._wheel_track_scale / 2.0
        wheel_radius = self.cfg.wheel_radius_m * self._wheel_radius_scale
        self._wheel_targets[:, 0] = (linear - angular * half_track) / wheel_radius
        self._wheel_targets[:, 1] = (linear + angular * half_track) / wheel_radius
        self._wheel_targets *= self._motor_strength
        self._wheel_targets.clamp_(
            -self.cfg.wheel_speed_limit_rad_s,
            self.cfg.wheel_speed_limit_rad_s,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        _, _, _, heading_error = self._goal_geometry()
        if self.cfg.recovery_supervisor_enabled:
            self._apply_recovery_supervisor(heading_error)
        self._update_pivot_torque_limits(self._actions, heading_error)

    def _get_rewards(self) -> torch.Tensor:
        previous_abs_heading_error = self._previous_abs_heading_error.clone()
        rewards = super()._get_rewards()
        _, _, _, heading_error = self._goal_geometry()
        abs_heading_error = torch.abs(heading_error)
        targeted = torch.any(
            self._segment_ids.unsqueeze(1) == self._targeted_recovery_ids.unsqueeze(0),
            dim=1,
        ).float()
        turning = targeted * (
            abs_heading_error > self.cfg.targeted_turn_alignment_rad
        ).float()
        pivoting = targeted * (
            abs_heading_error > self.cfg.targeted_pivot_brake_threshold_rad
        ).float()
        aligned = targeted * (
            abs_heading_error <= self.cfg.targeted_turn_alignment_rad
        ).float()
        heading_progress = (
            previous_abs_heading_error - abs_heading_error
        ).clamp(-0.35, 0.35)
        maximum_normalized_correction = (
            self.cfg.maximum_lateral_correction_rad_s
            / self.cfg.angular_velocity_max_rad_s
        )
        signed_alignment = (
            self._applied_steering_request * torch.sign(heading_error)
        )
        normalized_aligned_request = (
            torch.relu(signed_alignment) / maximum_normalized_correction
        ).clamp(0.0, 1.0)
        normalized_wrong_request = (
            torch.relu(-signed_alignment) / maximum_normalized_correction
        ).clamp(0.0, 1.0)
        accepted = self._planner_request_accepted.float()

        targeted_heading_progress = (
            targeted
            * heading_progress
            * self.cfg.reward_targeted_heading_progress
        )
        targeted_aligned_steering = (
            turning
            * accepted
            * normalized_aligned_request
            * self.cfg.reward_targeted_aligned_steering_request
        )
        targeted_wrong_steering = (
            turning
            * normalized_wrong_request
            * self.cfg.penalty_targeted_wrong_steering_request
        )
        targeted_turn_inactivity = (
            turning
            * (1.0 - normalized_aligned_request)
            * self.cfg.penalty_targeted_turn_inactivity
        )
        normalized_forward_command = (
            (self._actions[:, 0] + 1.0) * 0.5
        ).clamp(0.0, 1.0)
        targeted_pivot_forward = (
            pivoting
            * normalized_forward_command
            * self.cfg.penalty_targeted_pivot_forward
        )
        targeted_aligned_nonforward = (
            aligned
            * (1.0 - normalized_forward_command)
            * self.cfg.penalty_targeted_aligned_nonforward
        )

        clearance_target = (
            self._segment_ids == self.cfg.targeted_clearance_segment_id
        ).float()
        clearance_improvement = torch.relu(
            self._planner_candidate_clearance - self._planner_baseline_clearance
        ).clamp_max(0.50)
        targeted_clearance_improvement = (
            clearance_target
            * accepted
            * clearance_improvement
            * self.cfg.reward_targeted_clearance_improvement
        )
        clearance_deficit = (
            torch.relu(
                self.cfg.targeted_low_clearance_m
                - self._planner_applied_clearance
            )
            / self.cfg.targeted_low_clearance_m
        ).clamp(0.0, 2.0)
        targeted_low_clearance = (
            clearance_target
            * clearance_deficit
            * self.cfg.penalty_targeted_low_clearance
        )

        shaped_rewards = (
            ("targeted_heading_progress", targeted_heading_progress),
            ("targeted_aligned_steering_request", targeted_aligned_steering),
            ("targeted_wrong_steering_request", targeted_wrong_steering),
            ("targeted_turn_inactivity", targeted_turn_inactivity),
            ("targeted_pivot_forward", targeted_pivot_forward),
            ("targeted_aligned_nonforward", targeted_aligned_nonforward),
            ("targeted_clearance_improvement", targeted_clearance_improvement),
            ("targeted_low_clearance", targeted_low_clearance),
        )
        for name, value in shaped_rewards:
            self._episode_sums[name] += value
            rewards += value
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if hasattr(self, "_episode_peak_torque_steps"):
            self.extras["episode_outcomes"].update(
                {
                    "peak_torque_steps": self._episode_peak_torque_steps.clone(),
                    "peak_torque_elapsed_s": self._peak_torque_elapsed_s.clone(),
                    "final_peak_torque_active": self._peak_torque_active.clone(),
                    "pivot_supervisor_steps": (
                        self._episode_pivot_supervisor_steps.clone()
                    ),
                    "predictive_stop_steps": (
                        self._episode_predictive_stop_steps.clone()
                    ),
                    "pivot_supervisor_steering_steps": (
                        self._episode_pivot_supervisor_steering_steps.clone()
                    ),
                    "office_departure_protective_release_steps": (
                        self._episode_office_departure_protective_release_steps.clone()
                    ),
                    "final_recovery_supervisor_brake_active": (
                        self._recovery_supervisor_brake_active.clone()
                    ),
                }
            )
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if hasattr(self, "_targeted_recovery_ids"):
            self._peak_torque_elapsed_s[env_ids] = 0.0
            self._peak_torque_active[env_ids] = False
            self._pivot_supervisor_latched[env_ids] = False
            self._predictive_stop_latched[env_ids] = False
            self._recovery_supervisor_brake_active[env_ids] = False
            self._pivot_supervisor_steering_active[env_ids] = False
            self._office_departure_protective_release_active[env_ids] = False
            self._episode_peak_torque_steps[env_ids] = 0
            self._episode_pivot_supervisor_steps[env_ids] = 0
            self._episode_predictive_stop_steps[env_ids] = 0
            self._episode_pivot_supervisor_steering_steps[env_ids] = 0
            self._episode_office_departure_protective_release_steps[env_ids] = 0
            rated_limits = torch.full(
                (len(env_ids), len(self._wheel_ids)),
                self.cfg.rated_motor_effort_limit_nm,
                device=self.device,
            )
            self._robot.write_joint_effort_limit_to_sim(
                rated_limits,
                joint_ids=self._wheel_ids,
                env_ids=env_ids,
            )
            self.extras["targeted_recovery"] = {
                "segment_ids": tuple(self.cfg.targeted_recovery_segment_ids),
                "sampling_weights": tuple(self.cfg.segment_sampling_weights),
                "recovery_supervisor_enabled": self.cfg.recovery_supervisor_enabled,
                "retention_segments_all_nonzero": all(
                    weight > 0.0 for weight in self.cfg.segment_sampling_weights
                ),
                "motor_effort_contract": {
                    "rated_nm": self.cfg.rated_motor_effort_limit_nm,
                    "peak_nm": self.cfg.peak_motor_effort_limit_nm,
                    "peak_time_limit_s": self.cfg.peak_motor_time_limit_s,
                    "peak_scope": "targeted stationary large-heading pivots only",
                },
                "recovery_supervisor_contract": {
                    "authority": (
                        "remove translation; goal-signed steering only during "
                        "clearance-projected office pivots"
                    ),
                    "pivot_segment_ids": tuple(self.cfg.office_departure_segment_ids),
                    "pivot_engage_heading_error_rad": (
                        self.cfg.pivot_supervisor_engage_heading_error_rad
                    ),
                    "pivot_release_heading_error_rad": (
                        self.cfg.pivot_supervisor_release_heading_error_rad
                    ),
                    "pivot_angular_command_rad_s": (
                        self.cfg.pivot_supervisor_angular_command_rad_s
                    ),
                    "office_departure_front_latch_release_clearance_m": (
                        self.cfg.office_departure_protective_release_clearance_m
                    ),
                    "predictive_stop_segment_ids": tuple(
                        self.cfg.predictive_stop_segment_ids
                    ),
                    "predictive_stop_trigger_clearance_m": (
                        self.cfg.predictive_stop_trigger_clearance_m
                    ),
                    "predictive_stop_release_clearance_m": (
                        self.cfg.predictive_stop_release_clearance_m
                    ),
                    "predictive_creep_linear_velocity_mps": (
                        self.cfg.predictive_creep_linear_velocity_mps
                    ),
                    "dynamic_crossing_creep_segment_ids": tuple(
                        self.cfg.dynamic_crossing_creep_segment_ids
                    ),
                    "dynamic_crossing_predictive_creep_linear_velocity_mps": (
                        self.cfg.dynamic_crossing_predictive_creep_linear_velocity_mps
                    ),
                },
                "castor_contact_contract": {
                    "model": "fixed_sphere_low_friction_proxy",
                    "static_friction_range": self.cfg.castor_static_friction_range,
                    "dynamic_friction_range": self.cfg.castor_dynamic_friction_range,
                    "shape_count": len(self._castor_material_shape_ids),
                },
            }
            if "domain_randomization" in self.extras:
                self.extras["domain_randomization"].update(
                    {
                        "castor_static_friction": self._castor_static_friction.clone(),
                        "castor_dynamic_friction": self._castor_dynamic_friction.clone(),
                    }
                )


@configclass
class AishaPhase3Segment6RehearsalEnvCfg(AishaPhase3DynamicDREnvCfg):
    """Target the principal-office turn without forgetting the other route legs."""

    # Segment 6 receives 40/62 (64.5%) of resets. Every other route leg keeps
    # 2/62 so the recovery run still rehearses the complete mission.
    segment_sampling_weights = (
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        40.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    )

    # Resume from a zero-collision Phase 3 candidate at 75% declared
    # perturbation strength, then reach full strength during this 300-iteration
    # run. This avoids returning to a static-only warm-up.
    curriculum_warmup_policy_steps = 0
    curriculum_ramp_policy_steps = 9_600
    curriculum_minimum_strength = 0.75


class AishaPhase3Segment6RehearsalEnv(AishaPhase3DynamicDREnv):
    """Phase 3 segment-6 recovery environment with whole-route retention."""

    cfg: AishaPhase3Segment6RehearsalEnvCfg


@configclass
class AishaPhase3Segment6SpecialistEnvCfg(AishaPhase3DynamicDREnvCfg):
    """Robustify the proven learned principal-turn skill as an ensemble specialist."""

    fixed_segment_id = 6
    curriculum_warmup_policy_steps = 3_200
    curriculum_ramp_policy_steps = 11_200
    curriculum_minimum_strength = 0.0
    penalty_near_obstacle = -0.05
    penalty_forward_near_obstacle = -0.50
    forward_near_obstacle_distance_m = 1.50


class AishaPhase3Segment6SpecialistEnv(AishaPhase3DynamicDREnv):
    """Fixed-skill Phase 3 curriculum used only by the learned route ensemble."""

    cfg: AishaPhase3Segment6SpecialistEnvCfg
