"""Sensor-grounded waypoint curriculum over the plan-derived Block A route."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from aisha_isaaclab.assets import AISHA_LOADED_CFG, SIM_PACKAGE_ROOT


COURSE_USD = SIM_PACKAGE_ROOT / "usd" / "block_a_training_course.usda"
ADMIN_CONFIG = SIM_PACKAGE_ROOT / "config" / "administration_assumptions.yaml"
_ADMIN = yaml.safe_load(ADMIN_CONFIG.read_text(encoding="utf-8"))
_WAYPOINTS = {item["id"]: (float(item["x_m"]), float(item["y_m"])) for item in _ADMIN["route"]["waypoints"]}

# Directed segments cover the complete demonstration loop. Training samples one
# segment per episode; presentation playback chains the same waypoints.
ROUTE_SEGMENTS = (
    ("home", "east_atrium_exit"),
    ("east_atrium_exit", "vice_principal_turn"),
    ("vice_principal_turn", "vice_principal_approach"),
    ("vice_principal_approach", "vice_principal"),
    ("vice_principal", "vice_principal_depart"),
    ("vice_principal_depart", "hallway_return"),
    ("hallway_return", "principal_turn"),
    ("principal_turn", "principal_approach"),
    ("principal_approach", "principal"),
    ("principal", "principal_depart"),
    ("principal_depart", "atrium_return"),
    ("atrium_return", "home_return"),
)


@configclass
class AishaBlockASensorSceneCfg(InteractiveSceneCfg):
    """Replicated Block A course, loaded AI-SHA and production-contract sensors."""

    course = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Course",
        spawn=sim_utils.UsdFileCfg(usd_path=str(COURSE_USD)),
    )
    robot = AISHA_LOADED_CFG
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
            )
        ],
        reference_meshes=False,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.88, 0.91, 0.96)),
    )


@configclass
class AishaBlockASensorEnvCfg(DirectRLEnvCfg):
    """Configuration for the second AI-SHA learning curriculum."""

    decimation = 4
    episode_length_s = 55.0
    action_space = 2
    observation_space = 46
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: AishaBlockASensorSceneCfg = AishaBlockASensorSceneCfg(
        num_envs=64,
        env_spacing=50.0,
        replicate_physics=True,
        # MultiMeshRayCaster needs concrete USD target prims for every cloned
        # course in Isaac Lab commit 80094be; Fabric-only clones are not visible
        # to its target discovery step.
        clone_in_fabric=False,
    )

    wheel_radius_m = 0.10
    wheel_track_m = 0.72
    wheel_speed_limit_rad_s = 16.755
    linear_velocity_range_mps = (0.0, 0.50)
    angular_velocity_max_rad_s = 1.0
    lidar_min_range_m = 0.12
    lidar_max_range_m = 10.0
    lidar_training_bins = 36
    lidar_source_scan_rate_hz = 10.0
    lidar_source_accuracy_m = 0.03
    observation_lidar_noise_std_m = 0.0
    observation_lidar_dropout_probability = 0.0
    start_lateral_jitter_m = 0.05
    start_yaw_jitter_rad = math.radians(8.0)
    start_heading_mode = "outgoing"
    start_transition_backoff_m = 0.0
    # Optional per-segment reset backoff. This recreates the physical route-chain
    # handoff, which occurs at the preceding segment's goal tolerance rather
    # than exactly at the nominal waypoint centre.
    start_transition_backoff_m_by_segment: tuple[float, ...] | None = None
    # Optional curriculum reset velocity. A non-zero range recreates a real
    # waypoint handoff: the robot arrives with wheel momentum and its previous
    # forward action still present in the policy observation.
    start_linear_velocity_range_mps = (0.0, 0.0)
    goal_jitter_m = 0.03
    goal_tolerance_m = 0.45
    # Defaults preserve the original uniform tolerance. Route-specific tasks
    # may use a larger presentation stand-off at selected visit waypoints.
    goal_tolerance_m_by_segment = (0.45,) * 12
    fixed_segment_id: int | None = None
    route_chain_mode: bool = False
    # A high-level route planner may disambiguate the direction of an exact
    # 180-degree goal reversal. Values bias the goal-heading observation and
    # its heading-progress reward; they never replace the policy's wheel action.
    turn_direction_hint_rad_by_segment = (0.0,) * 12
    # Evaluation-only mode: keep each parallel environment assigned to one
    # route segment so the evaluator can enforce an equal quota per segment.
    balanced_segment_assignment: bool = False
    balanced_segment_ids: tuple[int, ...] | None = None
    # Door entry/exit and the atrium-to-principal turn are deliberately
    # oversampled after held-out evaluation identified them as the hard cases.
    segment_sampling_weights = (1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0)
    robot_rear_x_m = -0.455
    robot_front_x_m = 0.725
    robot_half_width_m = 0.384
    lidar_x_m = 0.500
    lidar_collision_margin_m = 0.025

    reward_progress = 12.0
    reward_goal_proximity = 0.002
    reward_heading_alignment = 0.002
    reward_heading_progress = 0.0
    penalty_wrong_uturn_direction = 0.0
    reward_obstacle_clearance = 0.001
    penalty_near_obstacle = -0.002
    penalty_action_rate = -0.005
    penalty_yaw_rate = -0.002
    penalty_misaligned_forward = 0.0
    misaligned_heading_threshold_rad = math.radians(25.0)
    # Optional dense training signal for policies that learn to pivot safely
    # but then command zero forward speed. Kept disabled for all established
    # tasks; focused curricula may penalize non-forward actions after alignment.
    penalty_aligned_nonforward = 0.0
    penalty_time = -0.01
    penalty_stall = -0.02
    reward_success = 60.0
    penalty_collision = -25.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (-8.0, 12.0, 16.0)
        self.viewer.lookat = (7.5, -3.5, 0.0)


class AishaBlockASensorEnv(DirectRLEnv):
    """Learn goal-conditioned navigation from a downsampled LD19 ray scan."""

    cfg: AishaBlockASensorEnvCfg

    def __init__(self, cfg: AishaBlockASensorEnvCfg, render_mode: str | None = None, **kwargs):
        if not COURSE_USD.is_file():
            raise FileNotFoundError(
                f"missing {COURSE_USD}; run isaaclab/tools/build_block_a_training_course.py"
            )
        super().__init__(cfg, render_mode, **kwargs)

        left_ids, _ = self._robot.find_joints("left_wheel_joint")
        right_ids, _ = self._robot.find_joints("right_wheel_joint")
        if len(left_ids) != 1 or len(right_ids) != 1:
            raise RuntimeError(f"expected one driven wheel per side, found left={left_ids}, right={right_ids}")
        self._wheel_ids = [left_ids[0], right_ids[0]]

        starts = [_WAYPOINTS[start] for start, _ in ROUTE_SEGMENTS]
        goals = [_WAYPOINTS[goal] for _, goal in ROUTE_SEGMENTS]
        self._segment_starts = torch.tensor(starts, dtype=torch.float32, device=self.device)
        self._segment_goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
        outgoing_directions = self._segment_goals - self._segment_starts
        incoming_directions = outgoing_directions.clone()
        incoming_directions[1:] = self._segment_starts[1:] - self._segment_starts[:-1]
        self._segment_incoming_headings = torch.atan2(
            incoming_directions[:, 1], incoming_directions[:, 0]
        )
        if self.cfg.start_heading_mode not in ("outgoing", "incoming"):
            raise ValueError("start_heading_mode must be 'outgoing' or 'incoming'")
        start_velocity_min, start_velocity_max = self.cfg.start_linear_velocity_range_mps
        if not 0.0 <= start_velocity_min <= start_velocity_max <= self.cfg.linear_velocity_range_mps[1]:
            raise ValueError(
                "start_linear_velocity_range_mps must be non-negative, ordered, and within the commanded range"
            )
        self._segment_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if len(self.cfg.turn_direction_hint_rad_by_segment) != len(ROUTE_SEGMENTS):
            raise ValueError("turn_direction_hint_rad_by_segment must match ROUTE_SEGMENTS")
        self._turn_direction_hints = torch.tensor(
            self.cfg.turn_direction_hint_rad_by_segment,
            dtype=torch.float32,
            device=self.device,
        )
        if self.cfg.start_transition_backoff_m_by_segment is not None:
            if len(self.cfg.start_transition_backoff_m_by_segment) != len(ROUTE_SEGMENTS):
                raise ValueError("start_transition_backoff_m_by_segment must match ROUTE_SEGMENTS")
            self._start_transition_backoffs = torch.tensor(
                self.cfg.start_transition_backoff_m_by_segment,
                dtype=torch.float32,
                device=self.device,
            )
            if torch.any(self._start_transition_backoffs < 0.0):
                raise ValueError("start_transition_backoff_m_by_segment values must be non-negative")
        else:
            self._start_transition_backoffs = torch.full(
                (len(ROUTE_SEGMENTS),),
                float(self.cfg.start_transition_backoff_m),
                dtype=torch.float32,
                device=self.device,
            )
        if len(self.cfg.goal_tolerance_m_by_segment) != len(ROUTE_SEGMENTS):
            raise ValueError("goal_tolerance_m_by_segment must match ROUTE_SEGMENTS")
        self._goal_tolerances = torch.tensor(
            self.cfg.goal_tolerance_m_by_segment,
            dtype=torch.float32,
            device=self.device,
        )
        if torch.any(self._goal_tolerances <= 0.0):
            raise ValueError("goal_tolerance_m_by_segment values must be positive")
        if len(self.cfg.segment_sampling_weights) != len(ROUTE_SEGMENTS):
            raise ValueError("segment_sampling_weights must have one entry per route segment")
        self._segment_sampling_weights = torch.tensor(
            self.cfg.segment_sampling_weights, dtype=torch.float32, device=self.device
        )
        if torch.any(self._segment_sampling_weights <= 0.0):
            raise ValueError("segment_sampling_weights must be positive")
        self._goal_w = torch.zeros((self.num_envs, 2), device=self.device)
        self._actions = torch.zeros((self.num_envs, gym.spaces.flatdim(self.single_action_space)), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._wheel_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_abs_heading_error = torch.zeros(self.num_envs, device=self.device)
        ray_angles = torch.deg2rad(torch.arange(-180.0, 180.0, 10.0, device=self.device))
        ray_cos, ray_sin = torch.cos(ray_angles), torch.sin(ray_angles)
        x_distance = torch.where(
            ray_cos >= 0.0,
            (self.cfg.robot_front_x_m - self.cfg.lidar_x_m) / ray_cos.clamp_min(1.0e-6),
            (self.cfg.robot_rear_x_m - self.cfg.lidar_x_m) / ray_cos.clamp_max(-1.0e-6),
        )
        y_distance = self.cfg.robot_half_width_m / torch.abs(ray_sin).clamp_min(1.0e-6)
        self._lidar_envelope_ranges = torch.minimum(x_distance, y_distance) + self.cfg.lidar_collision_margin_m

        reward_names = (
            "progress",
            "goal_proximity",
            "heading_alignment",
            "heading_progress",
            "wrong_uturn_direction",
            "obstacle_clearance",
            "near_obstacle",
            "action_rate",
            "yaw_rate",
            "misaligned_forward",
            "aligned_nonforward",
            "time",
            "stall",
            "success",
            "collision",
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) for name in reward_names
        }

    def _setup_scene(self) -> None:
        self._robot = self.scene.articulations["robot"]
        self._crown_lidar = self.scene.sensors["crown_lidar"]

    def _local_xy(self) -> torch.Tensor:
        return self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]

    def _apply_turn_direction_hint(
        self, heading_error: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        """Choose a route-consistent side only for near-180-degree reversals."""
        hints = self._turn_direction_hints[segment_ids]
        absolute_error = torch.abs(heading_error)
        hint_magnitude = torch.abs(hints)
        fade_start = math.pi - 2.0 * hint_magnitude
        near_uturn = absolute_error > fade_start
        # Fade the planner's sign cue continuously from its full value at 180
        # degrees to zero at 120 degrees (for a 30-degree hint). This supplies
        # immediate progress without a plateau or a discontinuity.
        fade_weight = ((absolute_error - fade_start) / (2.0 * hint_magnitude).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        hinted_magnitude = absolute_error - hint_magnitude * fade_weight
        hinted_error = torch.sign(hints) * hinted_magnitude
        return torch.where(near_uturn & (hints != 0.0), hinted_error, heading_error)

    def _goal_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_w = self._goal_w - self._robot.data.root_pos_w[:, :2]
        quat = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
        )
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        goal_x_b = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        goal_y_b = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        distance = torch.linalg.norm(delta_w, dim=1)
        raw_heading_error = torch.atan2(goal_y_b, goal_x_b)
        heading_error = self._apply_turn_direction_hint(raw_heading_error, self._segment_ids)
        hints = self._turn_direction_hints[self._segment_ids]
        virtual_turn_subgoal = (
            torch.abs(raw_heading_error) > math.pi - 2.0 * torch.abs(hints)
        ) & (hints != 0.0)
        # Keep all goal features coherent while the route planner resolves the
        # U-turn: the actor sees a signed virtual local direction, while the
        # physical goal position and distance remain unchanged.
        goal_x_b = torch.where(virtual_turn_subgoal, distance * torch.cos(heading_error), goal_x_b)
        goal_y_b = torch.where(virtual_turn_subgoal, distance * torch.sin(heading_error), goal_y_b)
        return goal_x_b, goal_y_b, distance, heading_error

    def _lidar_ranges(self) -> torch.Tensor:
        hit_vectors = self._crown_lidar.data.ray_hits_w - self._crown_lidar.data.pos_w.unsqueeze(1)
        ranges = torch.linalg.norm(hit_vectors, dim=-1)
        ranges = torch.nan_to_num(
            ranges,
            nan=self.cfg.lidar_max_range_m,
            posinf=self.cfg.lidar_max_range_m,
            neginf=self.cfg.lidar_min_range_m,
        )
        return ranges.clamp(self.cfg.lidar_min_range_m, self.cfg.lidar_max_range_m)

    def _lidar_observation_ranges(self) -> torch.Tensor:
        """Return randomized policy observations without corrupting collision truth."""
        ranges = self._lidar_ranges()
        if self.cfg.observation_lidar_noise_std_m > 0.0:
            ranges = ranges + torch.randn_like(ranges) * self.cfg.observation_lidar_noise_std_m
        if self.cfg.observation_lidar_dropout_probability > 0.0:
            drop = torch.rand_like(ranges) < self.cfg.observation_lidar_dropout_probability
            ranges = torch.where(drop, self.cfg.lidar_max_range_m, ranges)
        return ranges.clamp(self.cfg.lidar_min_range_m, self.cfg.lidar_max_range_m)

    def _lidar_envelope_collision(self) -> torch.Tensor:
        """Conservatively flag obstacles inside the robot's rectangular envelope."""
        return torch.any(self._lidar_ranges() <= self._lidar_envelope_ranges.unsqueeze(0), dim=1)

    def _termination_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_xy = self._local_xy()
        _, _, distance, _ = self._goal_geometry()
        projected_gravity_xy = torch.linalg.norm(self._robot.data.projected_gravity_b[:, :2], dim=1)
        collision = self._lidar_envelope_collision() | (projected_gravity_xy > 0.55)
        success = distance < self._goal_tolerances[self._segment_ids]
        out_of_bounds = (
            (local_xy[:, 0] < -7.5)
            | (local_xy[:, 0] > 24.0)
            | (local_xy[:, 1] < -13.5)
            | (local_xy[:, 1] > 7.5)
        )
        invalid = out_of_bounds | ~torch.isfinite(local_xy).all(dim=1)
        return collision, success, invalid

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone().clamp(-1.0, 1.0)
        minimum, maximum = self.cfg.linear_velocity_range_mps
        linear = minimum + (self._actions[:, 0] + 1.0) * 0.5 * (maximum - minimum)
        angular = self._actions[:, 1] * self.cfg.angular_velocity_max_rad_s
        half_track = self.cfg.wheel_track_m / 2.0
        self._wheel_targets[:, 0] = (linear - angular * half_track) / self.cfg.wheel_radius_m
        self._wheel_targets[:, 1] = (linear + angular * half_track) / self.cfg.wheel_radius_m
        self._wheel_targets.clamp_(-self.cfg.wheel_speed_limit_rad_s, self.cfg.wheel_speed_limit_rad_s)

    def _apply_action(self) -> None:
        self._robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        goal_x_b, goal_y_b, distance, heading_error = self._goal_geometry()
        lidar = self._lidar_observation_ranges() / self.cfg.lidar_max_range_m
        state = torch.stack(
            (
                (goal_x_b / 12.0).clamp(-1.0, 1.0),
                (goal_y_b / 12.0).clamp(-1.0, 1.0),
                (distance / 12.0).clamp(0.0, 1.5),
                torch.sin(heading_error),
                torch.cos(heading_error),
                (self._robot.data.root_lin_vel_b[:, 0] / 0.5).clamp(-2.0, 2.0),
                self._robot.data.root_ang_vel_b[:, 2].clamp(-2.0, 2.0),
                self._actions[:, 0],
                self._actions[:, 1],
                (torch.amin(self._lidar_ranges(), dim=1) / self.cfg.lidar_max_range_m).clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        return {"policy": torch.cat((state, lidar), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        _, _, distance, heading_error = self._goal_geometry()
        lidar = self._lidar_ranges()
        min_range = torch.amin(lidar, dim=1)
        collision, success, invalid = self._termination_masks()
        progress = (self._previous_distance - distance).clamp(-0.20, 0.20)
        abs_heading_error = torch.abs(heading_error)
        heading_progress = (self._previous_abs_heading_error - abs_heading_error).clamp(-0.35, 0.35)
        turn_hints = self._turn_direction_hints[self._segment_ids]
        near_hinted_uturn = (abs_heading_error > math.radians(90.0)) & (turn_hints != 0.0)
        normalized_forward_command = ((self._actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        rewards = {
            "progress": progress * self.cfg.reward_progress,
            "goal_proximity": torch.exp(-distance / 2.0) * self.cfg.reward_goal_proximity,
            "heading_alignment": torch.exp(-torch.square(heading_error) / 0.40)
            * self.cfg.reward_heading_alignment,
            "heading_progress": heading_progress * self.cfg.reward_heading_progress,
            "wrong_uturn_direction": (
                near_hinted_uturn.float()
                * torch.relu(-self._actions[:, 1] * torch.sign(turn_hints))
                * self.cfg.penalty_wrong_uturn_direction
            ),
            "obstacle_clearance": (min_range / self.cfg.lidar_max_range_m).clamp(0.0, 1.0)
            * self.cfg.reward_obstacle_clearance,
            "near_obstacle": (min_range < 0.55).float() * self.cfg.penalty_near_obstacle,
            "action_rate": torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
            * self.cfg.penalty_action_rate,
            "yaw_rate": torch.square(self._robot.data.root_ang_vel_b[:, 2]) * self.cfg.penalty_yaw_rate,
            "misaligned_forward": (
                (torch.abs(heading_error) > self.cfg.misaligned_heading_threshold_rad).float()
                * normalized_forward_command
                * self.cfg.penalty_misaligned_forward
            ),
            "aligned_nonforward": (
                (abs_heading_error < self.cfg.misaligned_heading_threshold_rad).float()
                * (distance > self._goal_tolerances[self._segment_ids]).float()
                * (1.0 - normalized_forward_command)
                * self.cfg.penalty_aligned_nonforward
            ),
            "time": torch.ones_like(distance) * self.cfg.penalty_time,
            "stall": (
                (torch.abs(self._robot.data.root_lin_vel_b[:, 0]) < 0.03)
                & (distance > self._goal_tolerances[self._segment_ids])
                & (abs_heading_error < self.cfg.misaligned_heading_threshold_rad)
            ).float()
            * self.cfg.penalty_stall,
            "success": success.float() * self.cfg.reward_success,
            "collision": (collision | invalid).float() * self.cfg.penalty_collision,
        }
        self._previous_distance.copy_(distance)
        self._previous_abs_heading_error.copy_(abs_heading_error)
        for name, value in rewards.items():
            self._episode_sums[name] += value

        if self.cfg.route_chain_mode:
            reached_segment_ids = self._segment_ids.clone()
            advance = success & (reached_segment_ids < len(ROUTE_SEGMENTS) - 1)
            self.extras["route_chain"] = {
                "waypoint_reached": success.clone(),
                "reached_segment_id": reached_segment_ids,
            }
            if torch.any(advance):
                advance_ids = torch.nonzero(advance, as_tuple=False).flatten()
                next_segments = reached_segment_ids[advance_ids] + 1
                self._segment_ids[advance_ids] = next_segments
                self._goal_w[advance_ids] = (
                    self.scene.env_origins[advance_ids, :2] + self._segment_goals[next_segments]
                )
                self._previous_distance[advance_ids] = torch.linalg.norm(
                    self._goal_w[advance_ids] - self._robot.data.root_pos_w[advance_ids, :2], dim=1
                )
                next_heading_error = self._goal_geometry()[3]
                self._previous_abs_heading_error[advance_ids] = torch.abs(next_heading_error[advance_ids])
        return torch.stack(tuple(rewards.values()), dim=0).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        collision, success, invalid = self._termination_masks()
        final_success = success
        if self.cfg.route_chain_mode:
            final_success = success & (self._segment_ids == len(ROUTE_SEGMENTS) - 1)
        terminated = collision | final_success | invalid
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        _, _, distance, heading_error = self._goal_geometry()
        lidar_ranges = self._lidar_ranges()
        minimum_lidar_range, minimum_lidar_ray_index = torch.min(lidar_ranges, dim=1)
        self.extras["episode_outcomes"] = {
            "success": final_success.clone(),
            "waypoint_reached": success.clone(),
            "collision": (collision | invalid).clone(),
            "time_out": time_out.clone(),
            "final_distance_m": distance.clone(),
            "final_heading_error_rad": heading_error.clone(),
            "segment_id": self._segment_ids.clone(),
            "position_xy_m": self._local_xy().clone(),
            "minimum_lidar_range_m": minimum_lidar_range.clone(),
            "minimum_lidar_ray_index": minimum_lidar_ray_index.clone(),
        }
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        collision, success, invalid = self._termination_masks()
        self.extras["log"] = {}
        for name, episode_sum in self._episode_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = episode_sum[env_ids].mean().item()
            episode_sum[env_ids] = 0.0
        self.extras["log"]["Metrics/success_rate"] = success[env_ids].float().mean().item()
        self.extras["log"]["Metrics/collision_rate"] = (collision | invalid)[env_ids].float().mean().item()
        self.extras["log"]["Episode_Termination/time_out"] = self.reset_time_outs[env_ids].float().sum().item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._wheel_targets[env_ids] = 0.0

        count = len(env_ids)
        if self.cfg.route_chain_mode:
            segment_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        elif self.cfg.fixed_segment_id is not None:
            if not 0 <= self.cfg.fixed_segment_id < len(ROUTE_SEGMENTS):
                raise ValueError(f"fixed_segment_id must be in [0, {len(ROUTE_SEGMENTS) - 1}]")
            segment_ids = torch.full(
                (count,), self.cfg.fixed_segment_id, dtype=torch.long, device=self.device
            )
        elif self.cfg.balanced_segment_assignment:
            if self.cfg.balanced_segment_ids is None:
                segment_ids = env_ids.remainder(len(ROUTE_SEGMENTS))
            else:
                if not self.cfg.balanced_segment_ids:
                    raise ValueError("balanced_segment_ids must not be empty")
                if any(
                    value < 0 or value >= len(ROUTE_SEGMENTS)
                    for value in self.cfg.balanced_segment_ids
                ):
                    raise ValueError("balanced_segment_ids contains an invalid route segment")
                balanced_ids = torch.tensor(
                    self.cfg.balanced_segment_ids, dtype=torch.long, device=self.device
                )
                segment_ids = balanced_ids[env_ids.remainder(len(balanced_ids))]
        else:
            segment_ids = torch.multinomial(self._segment_sampling_weights, count, replacement=True)
        self._segment_ids[env_ids] = segment_ids
        starts = self._segment_starts[segment_ids].clone()
        goals = self._segment_goals[segment_ids].clone()
        direction = goals - starts
        heading = torch.atan2(direction[:, 1], direction[:, 0])
        if self.cfg.start_heading_mode == "incoming":
            heading = self._segment_incoming_headings[segment_ids].clone()
            transition_backoff = self._start_transition_backoffs[segment_ids]
            starts[:, 0] -= torch.cos(heading) * transition_backoff
            starts[:, 1] -= torch.sin(heading) * transition_backoff
        lateral = torch.empty(count, device=self.device).uniform_(
            -self.cfg.start_lateral_jitter_m, self.cfg.start_lateral_jitter_m
        )
        direction_norm = torch.linalg.norm(direction, dim=1).clamp_min(1.0e-6)
        starts[:, 0] += -direction[:, 1] / direction_norm * lateral
        starts[:, 1] += direction[:, 0] / direction_norm * lateral
        yaw = heading + torch.empty(count, device=self.device).uniform_(
            -self.cfg.start_yaw_jitter_rad, self.cfg.start_yaw_jitter_rad
        )
        start_linear_velocity = torch.empty(count, device=self.device).uniform_(
            *self.cfg.start_linear_velocity_range_mps
        )
        goals += torch.empty_like(goals).uniform_(-self.cfg.goal_jitter_m, self.cfg.goal_jitter_m)
        self._goal_w[env_ids] = self.scene.env_origins[env_ids, :2] + goals

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, 0] = self.scene.env_origins[env_ids, 0] + starts[:, 0]
        root_state[:, 1] = self.scene.env_origins[env_ids, 1] + starts[:, 1]
        root_state[:, 2] = self.scene.env_origins[env_ids, 2] + 0.03
        zeros = torch.zeros_like(yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_state[:, 7:] = 0.0
        root_state[:, 7] = start_linear_velocity * torch.cos(yaw)
        root_state[:, 8] = start_linear_velocity * torch.sin(yaw)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_vel[:, self._wheel_ids] = start_linear_velocity.unsqueeze(1) / self.cfg.wheel_radius_m
        command_minimum, command_maximum = self.cfg.linear_velocity_range_mps
        if command_maximum > command_minimum:
            normalized_forward = (
                2.0 * (start_linear_velocity - command_minimum) / (command_maximum - command_minimum) - 1.0
            ).clamp(-1.0, 1.0)
            self._actions[env_ids, 0] = normalized_forward
            self._previous_actions[env_ids, 0] = normalized_forward
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._previous_distance[env_ids] = torch.linalg.norm(self._goal_w[env_ids] - root_state[:, :2], dim=1)
        initial_heading_error = torch.atan2(
            goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0]
        ) - yaw
        initial_heading_error = torch.atan2(torch.sin(initial_heading_error), torch.cos(initial_heading_error))
        initial_heading_error = self._apply_turn_direction_hint(initial_heading_error, segment_ids)
        self._previous_abs_heading_error[env_ids] = torch.abs(initial_heading_error)
