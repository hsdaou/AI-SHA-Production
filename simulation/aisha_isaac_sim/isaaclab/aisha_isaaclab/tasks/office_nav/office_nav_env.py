"""Physics-driven AI-SHA doorway navigation task for Isaac Lab."""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from aisha_isaaclab.assets import AISHA_LOADED_CFG


def _wall_cfg(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.80, 0.82), roughness=0.72),
    )


@configclass
class AishaDoorwaySceneCfg(InteractiveSceneCfg):
    """A replicated 2.80 m corridor with a 1.05 m doorway."""

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
    robot = AISHA_LOADED_CFG

    left_corridor_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftCorridorWall",
        spawn=_wall_cfg((8.0, 0.15, 1.50)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.50, 1.475, 0.75)),
    )
    right_corridor_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RightCorridorWall",
        spawn=_wall_cfg((8.0, 0.15, 1.50)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.50, -1.475, 0.75)),
    )
    door_wall_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DoorWallLeft",
        spawn=_wall_cfg((0.15, 0.875, 1.50)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.50, 0.9625, 0.75)),
    )
    door_wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DoorWallRight",
        spawn=_wall_cfg((0.15, 0.875, 1.50)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.50, -0.9625, 0.75)),
    )
    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.CylinderCfg(
            radius=0.24,
            height=0.012,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.76, 0.40),
                emissive_color=(0.02, 0.18, 0.08),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.30, 0.0, 0.006)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.88, 0.90, 0.94)),
    )


@configclass
class AishaOfficeNavEnvCfg(DirectRLEnvCfg):
    """Configuration for the first AI-SHA Isaac Lab curriculum stage."""

    decimation = 4
    episode_length_s = 18.0
    action_space = 2
    observation_space = 10
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
    scene: AishaDoorwaySceneCfg = AishaDoorwaySceneCfg(
        num_envs=64,
        env_spacing=9.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    wheel_radius_m = 0.10
    wheel_track_m = 0.72
    wheel_speed_limit_rad_s = 16.755
    linear_velocity_min_mps = -0.10
    linear_velocity_max_mps = 0.50
    angular_velocity_max_rad_s = 1.0

    start_x_m = -1.80
    start_y_range_m = (-0.30, 0.30)
    start_yaw_range_rad = (-math.radians(14.0), math.radians(14.0))
    goal_xy_m = (4.30, 0.0)
    goal_tolerance_m = 0.35
    corridor_half_width_m = 1.40
    doorway_x_m = 2.50
    doorway_clear_width_m = 1.05
    robot_half_width_m = 0.384
    doorway_collision_x_half_extent_m = 0.67

    reward_progress = 8.0
    reward_goal_proximity = 0.04
    reward_heading_alignment = 0.02
    reward_doorway_alignment = 0.04
    penalty_action_rate = -0.01
    penalty_yaw_rate = -0.002
    penalty_time = -0.005
    reward_success = 25.0
    penalty_collision = -20.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (-6.5, 6.5, 5.0)
        self.viewer.lookat = (1.4, 0.0, 0.45)


class AishaOfficeNavEnv(DirectRLEnv):
    """Train AI-SHA to align and drive through a constrained doorway."""

    cfg: AishaOfficeNavEnvCfg

    def __init__(self, cfg: AishaOfficeNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        left_ids, _ = self._robot.find_joints("left_wheel_joint")
        right_ids, _ = self._robot.find_joints("right_wheel_joint")
        if len(left_ids) != 1 or len(right_ids) != 1:
            raise RuntimeError(f"expected one driven wheel per side, found left={left_ids}, right={right_ids}")
        self._wheel_ids = [left_ids[0], right_ids[0]]

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros((self.num_envs, action_dim), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._wheel_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._goal_w = torch.zeros((self.num_envs, 2), device=self.device)
        self._goal_w[:] = self.scene.env_origins[:, :2]
        self._goal_w[:, 0] += self.cfg.goal_xy_m[0]
        self._goal_w[:, 1] += self.cfg.goal_xy_m[1]
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)

        reward_names = (
            "progress",
            "goal_proximity",
            "heading_alignment",
            "doorway_alignment",
            "action_rate",
            "yaw_rate",
            "time",
            "success",
            "collision",
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) for name in reward_names
        }

    def _setup_scene(self) -> None:
        self._robot = self.scene.articulations["robot"]

    def _local_xy(self) -> torch.Tensor:
        return self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]

    def _goal_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_w = self._goal_w - self._robot.data.root_pos_w[:, :2]
        quat = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
        )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        goal_x_b = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        goal_y_b = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        distance = torch.linalg.norm(delta_w, dim=1)
        heading_error = torch.atan2(goal_y_b, goal_x_b)
        return goal_x_b, goal_y_b, distance, heading_error

    def _termination_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_xy = self._local_xy()
        _, _, distance, _ = self._goal_geometry()
        door_centre_allowance = self.cfg.doorway_clear_width_m / 2.0 - self.cfg.robot_half_width_m
        in_door_plane = torch.abs(local_xy[:, 0] - self.cfg.doorway_x_m) < self.cfg.doorway_collision_x_half_extent_m
        door_collision = in_door_plane & (torch.abs(local_xy[:, 1]) > door_centre_allowance)
        side_collision = torch.abs(local_xy[:, 1]) > (
            self.cfg.corridor_half_width_m - self.cfg.robot_half_width_m
        )
        projected_gravity_xy = torch.linalg.norm(self._robot.data.projected_gravity_b[:, :2], dim=1)
        tipped = projected_gravity_xy > 0.55
        out_of_bounds = (local_xy[:, 0] < -2.65) | (local_xy[:, 0] > 5.65)
        collision = door_collision | side_collision | tipped
        success = (distance < self.cfg.goal_tolerance_m) & (local_xy[:, 0] > self.cfg.doorway_x_m)
        invalid = out_of_bounds | ~torch.isfinite(local_xy).all(dim=1)
        return collision, success, invalid

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone().clamp(-1.0, 1.0)
        linear = self.cfg.linear_velocity_min_mps + (self._actions[:, 0] + 1.0) * 0.5 * (
            self.cfg.linear_velocity_max_mps - self.cfg.linear_velocity_min_mps
        )
        angular = self._actions[:, 1] * self.cfg.angular_velocity_max_rad_s
        half_track = self.cfg.wheel_track_m / 2.0
        self._wheel_targets[:, 0] = (linear - angular * half_track) / self.cfg.wheel_radius_m
        self._wheel_targets[:, 1] = (linear + angular * half_track) / self.cfg.wheel_radius_m
        self._wheel_targets.clamp_(
            -self.cfg.wheel_speed_limit_rad_s,
            self.cfg.wheel_speed_limit_rad_s,
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        goal_x_b, goal_y_b, distance, heading_error = self._goal_geometry()
        obs = torch.stack(
            (
                (goal_x_b / 6.5).clamp(-1.0, 1.0),
                (goal_y_b / 6.5).clamp(-1.0, 1.0),
                (distance / 6.5).clamp(0.0, 1.5),
                torch.sin(heading_error),
                torch.cos(heading_error),
                (self._robot.data.root_lin_vel_b[:, 0] / 0.5).clamp(-2.0, 2.0),
                (self._robot.data.root_lin_vel_b[:, 1] / 0.5).clamp(-2.0, 2.0),
                self._robot.data.root_ang_vel_b[:, 2].clamp(-2.0, 2.0),
                self._actions[:, 0],
                self._actions[:, 1],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        local_xy = self._local_xy()
        _, _, distance, heading_error = self._goal_geometry()
        collision, success, invalid = self._termination_masks()
        progress = (self._previous_distance - distance).clamp(-0.20, 0.20)
        door_zone = torch.exp(-torch.square((local_xy[:, 0] - self.cfg.doorway_x_m) / 1.15))
        door_alignment = door_zone * torch.exp(-torch.square(local_xy[:, 1] / 0.20))
        rewards = {
            "progress": progress * self.cfg.reward_progress,
            "goal_proximity": torch.exp(-distance / 1.3) * self.cfg.reward_goal_proximity,
            "heading_alignment": torch.exp(-torch.square(heading_error) / 0.35)
            * self.cfg.reward_heading_alignment,
            "doorway_alignment": door_alignment * self.cfg.reward_doorway_alignment,
            "action_rate": torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
            * self.cfg.penalty_action_rate,
            "yaw_rate": torch.square(self._robot.data.root_ang_vel_b[:, 2]) * self.cfg.penalty_yaw_rate,
            "time": torch.ones_like(distance) * self.cfg.penalty_time,
            "success": success.float() * self.cfg.reward_success,
            "collision": (collision | invalid).float() * self.cfg.penalty_collision,
        }
        self._previous_distance.copy_(distance)
        for name, value in rewards.items():
            self._episode_sums[name] += value
        return torch.stack(tuple(rewards.values()), dim=0).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        collision, success, invalid = self._termination_masks()
        terminated = collision | success | invalid
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        _, _, distance, _ = self._goal_geometry()
        # Clone outcome tensors before DirectRLEnv resets completed environments.
        # The held-out evaluator consumes these values from the returned extras
        # instead of inferring results from the post-reset state.
        self.extras["episode_outcomes"] = {
            "success": success.clone(),
            "collision": (collision | invalid).clone(),
            "time_out": time_out.clone(),
            "final_distance_m": distance.clone(),
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
        start_y = torch.empty(count, device=self.device).uniform_(*self.cfg.start_y_range_m)
        start_yaw = torch.empty(count, device=self.device).uniform_(*self.cfg.start_yaw_range_rad)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, 0] = self.scene.env_origins[env_ids, 0] + self.cfg.start_x_m
        root_state[:, 1] = self.scene.env_origins[env_ids, 1] + start_y
        root_state[:, 2] = self.scene.env_origins[env_ids, 2] + 0.03
        zeros = torch.zeros_like(start_yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, start_yaw)
        root_state[:, 7:] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        start_delta = self._goal_w[env_ids] - root_state[:, :2]
        self._previous_distance[env_ids] = torch.linalg.norm(start_delta, dim=1)
