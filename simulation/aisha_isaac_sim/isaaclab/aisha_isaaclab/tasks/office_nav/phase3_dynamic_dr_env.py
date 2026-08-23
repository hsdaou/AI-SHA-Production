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
    _ADMIN,
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
PHASE3M_FROZEN_RECOVERY_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "aisha_phase3m_hybrid_recovery_model_125.pt"
)
PHASE3M_FROZEN_RECOVERY_CHECKPOINT_SHA256 = (
    "bc8727e3ea42c8b29ca74fa5a535fd37b1600633ffd8bf606b02220a557c1a0d"
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
    randomize_contact_materials = True

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
    # Optional social-navigation assumption. Zero preserves the original
    # pause-in-place proxy; Phase 3N enables a slow lateral step-away response.
    dynamic_obstacle_social_retreat_speed_mps = 0.0
    dynamic_obstacle_social_retreat_maximum_m = 0.60

    reward_progress = 14.0
    reward_heading_alignment = 0.02
    reward_heading_progress = 8.0
    penalty_wrong_uturn_direction = -0.05
    penalty_misaligned_forward = -0.05
    penalty_near_obstacle = -0.01
    penalty_forward_near_obstacle = -0.12
    forward_near_obstacle_distance_m = 1.20
    penalty_collision = -100.0

    # Site-measured presentation constraint. These deterministic bounds are a
    # simulation-only motion envelope around the two 0.85 m apertures; PPO must
    # align before entering because rotation is suppressed only at the frame.
    tight_door_segment_ids = (3, 4, 8, 9)
    tight_door_maximum_speed_mps = 0.10
    tight_door_slow_zone_normal_half_extent_m = 1.00
    tight_door_zone_tangent_half_extent_m = 1.10
    # Begin the straight-through constraint before the 0.725 m forward body
    # extent reaches the frame, including a conservative contact allowance.
    tight_door_no_rotation_normal_half_extent_m = 0.80
    tight_door_no_rotation_tangent_half_extent_m = 1.10
    tight_door_alignment_hold_normal_half_extent_m = 1.60
    tight_door_alignment_hold_heading_error_rad = math.radians(3.0)
    tight_door_alignment_hold_maximum_yaw_rate_rad_s = 0.05
    tight_door_alignment_breakaway_angular_action = 0.0
    tight_door_alignment_breakaway_maximum_yaw_rate_rad_s = 0.01
    tight_door_alignment_release_depth_m = 0.75
    tight_door_alignment_minimum_speed_mps = 0.0
    tight_door_traction_compensation_enabled = False
    tight_door_traction_target_speed_mps = 0.08
    tight_door_traction_command_ceiling_mps = 0.24
    tight_door_traction_gain = 2.0
    tight_door_traction_overspeed_stop_mps = 0.11
    tight_door_straight_heading_gain = 4.0
    tight_door_straight_yaw_damping_gain = 2.0
    tight_door_straight_maximum_angular_action = 0.35


@configclass
class AishaMeasuredTightDoorEnvCfg(AishaPhase3DynamicDREnvCfg):
    """Focused first-stage adaptation to padded 0.85 m office apertures."""

    # Allow the heavy platform time to align, settle its yaw rate, and traverse
    # the doorway at the conservative 0.10 m/s safety cap.  This changes only
    # the episode time allowance; it does not relax geometry or collision gates.
    episode_length_s = 100.0
    dynamic_obstacle_activation_probability = 0.0
    dynamic_obstacle_social_retreat_speed_mps = 0.0
    # At 0.90 m normal distance the 0.725 m chassis end has cleared the wall
    # plane by 175 mm. Release the 0.10 m/s cap there so the 171 kg platform
    # does not static-lock immediately after a safe crossing.
    tight_door_slow_zone_normal_half_extent_m = 0.90
    # Begin mapped centreline convergence upstream of either office frame,
    # then enforce the final straight-through heading over the last 0.80 m.
    tight_door_alignment_hold_normal_half_extent_m = 2.00
    # The 0.85 m VP aperture leaves only millimetres of corner clearance for
    # the long rectangular chassis.  Require near-square entry and provide a
    # finite in-place turn request to overcome fixed-castor static friction.
    tight_door_alignment_hold_heading_error_rad = math.radians(0.5)
    tight_door_alignment_breakaway_angular_action = 0.55
    # Keep the mapped centreline handoff active until the full body is clear
    # and bridge the low-speed dead zone that previously caused safe timeouts.
    tight_door_alignment_release_depth_m = 2.00
    tight_door_alignment_minimum_speed_mps = 0.10
    segment_sampling_weights = (
        1.0,
        1.0,
        2.0,
        30.0,
        30.0,
        2.0,
        2.0,
        2.0,
        30.0,
        30.0,
        2.0,
        1.0,
    )
    start_lateral_jitter_m = 0.025
    start_yaw_jitter_rad = math.radians(5.0)
    start_linear_velocity_range_mps = (0.0, 0.10)
    # Segment 10's historical 0.45 m incoming-handoff backoff lands the
    # corrected Principal departure pose inside the diagonal jamb.  Start at
    # the audited clear waypoint instead; all other handoffs retain Phase 3.
    start_transition_backoff_m_by_segment = (
        0.00,
        0.45,
        0.45,
        0.45,
        0.00,
        0.45,
        0.45,
        0.45,
        0.45,
        0.00,
        0.00,
        0.45,
    )
    goal_jitter_m = 0.015
    observation_lidar_noise_std_m = 0.005
    observation_lidar_dropout_probability = 0.0
    lidar_episode_bias_range_m = (-0.005, 0.005)
    lidar_episode_scale_range = (0.998, 1.002)
    curriculum_warmup_policy_steps = 0
    curriculum_ramp_policy_steps = 6_400
    curriculum_minimum_strength = 0.25
    # The imported robot intentionally gives the fixed-sphere castors much
    # lower friction than the drive wheels.  The original Phase 3 scalar
    # randomizer writes one coefficient to every shape, erases that split and
    # static-locks the 171 kg proxy at the 0.10 m/s doorway command.  Preserve
    # the imported heterogeneous materials until per-shape ranges are added.
    randomize_contact_materials = False


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
        self._obstacle_yield_offsets = torch.zeros(obstacle_shape, device=self.device)

        base_ids, _ = self._robot.find_bodies("base_link")
        if len(base_ids) != 1:
            raise RuntimeError(f"expected one base_link for mass randomization, found {base_ids}")
        self._base_body_id = int(base_ids[0])
        self._episode_sums["forward_near_obstacle"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        door_order = ("vice_principal", "principal")
        self._tight_door_centres = torch.tensor(
            [_ADMIN["doors"][name]["centre_xy_m"] for name in door_order],
            dtype=torch.float32,
            device=self.device,
        )
        angles = torch.deg2rad(
            torch.tensor(
                [_ADMIN["doors"][name]["wall_rotation_deg"] for name in door_order],
                dtype=torch.float32,
                device=self.device,
            )
        )
        self._tight_door_tangents = torch.stack(
            (torch.cos(angles), torch.sin(angles)), dim=1
        )
        self._tight_door_normals = torch.stack(
            (-torch.sin(angles), torch.cos(angles)), dim=1
        )
        self._tight_door_slow_zone_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._tight_door_no_rotation_active = torch.zeros_like(
            self._tight_door_slow_zone_active
        )
        self._tight_door_alignment_hold_active = torch.zeros_like(
            self._tight_door_slow_zone_active
        )

    def _apply_tight_door_motion_envelope(self) -> None:
        local_xy = self._local_xy()
        slow = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        no_rotation = torch.zeros_like(slow)
        alignment_scope = torch.zeros_like(slow)
        straight_heading_action = torch.zeros(self.num_envs, device=self.device)
        straight_heading_error = torch.zeros(self.num_envs, device=self.device)
        quaternion = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0
            * (
                quaternion[:, 0] * quaternion[:, 3]
                + quaternion[:, 1] * quaternion[:, 2]
            ),
            1.0
            - 2.0
            * (quaternion[:, 2].square() + quaternion[:, 3].square()),
        )
        yaw_rate = self._robot.data.root_ang_vel_b[:, 2]
        for door_index, segment_ids in enumerate(((3, 4), (8, 9))):
            segment_scope = torch.zeros_like(slow)
            for segment_id in segment_ids:
                segment_scope |= self._segment_ids == segment_id
            relative = local_xy - self._tight_door_centres[door_index]
            tangent_distance = torch.abs(
                torch.sum(relative * self._tight_door_tangents[door_index], dim=1)
            )
            normal_distance = torch.abs(
                torch.sum(relative * self._tight_door_normals[door_index], dim=1)
            )
            in_tangent_zone = (
                tangent_distance
                <= self.cfg.tight_door_zone_tangent_half_extent_m
            )
            slow |= (
                segment_scope
                & in_tangent_zone
                & (
                    normal_distance
                    <= self.cfg.tight_door_slow_zone_normal_half_extent_m
                )
            )
            no_rotation |= (
                segment_scope
                & in_tangent_zone
                & (
                    normal_distance
                    <= self.cfg.tight_door_no_rotation_normal_half_extent_m
                )
                & (
                    tangent_distance
                    <= self.cfg.tight_door_no_rotation_tangent_half_extent_m
                )
            )
            for segment_id, direction_sign in zip(
                segment_ids, (-1.0, 1.0), strict=True
            ):
                exact_segment = self._segment_ids == segment_id
                signed_normal = torch.sum(
                    relative * self._tight_door_normals[door_index], dim=1
                )
                goal_side_depth = direction_sign * signed_normal
                exact_alignment_scope = (
                    exact_segment
                    & in_tangent_zone
                    & (
                        normal_distance
                        <= self.cfg.tight_door_alignment_hold_normal_half_extent_m
                    )
                    & (
                        goal_side_depth
                        < self.cfg.tight_door_alignment_release_depth_m
                    )
                )
                alignment_scope |= exact_alignment_scope
                desired_direction = direction_sign * self._tight_door_normals[door_index]
                normal_yaw = torch.atan2(desired_direction[1], desired_direction[0])
                approach_stage = (
                    self._tight_door_centres[door_index]
                    - direction_sign * self._tight_door_normals[door_index]
                )
                crossing_stage = (
                    self._tight_door_centres[door_index]
                    + direction_sign * self._tight_door_normals[door_index]
                )
                approach_depth = -direction_sign * signed_normal
                selected_stage = torch.where(
                    (approach_depth > 1.05).unsqueeze(1),
                    approach_stage.unsqueeze(0),
                    crossing_stage.unsqueeze(0),
                )
                stage_delta = selected_stage - local_xy
                stage_yaw = torch.atan2(stage_delta[:, 1], stage_delta[:, 0])
                use_stage_yaw = (
                    goal_side_depth < 0.75
                ) & (
                    normal_distance
                    > self.cfg.tight_door_no_rotation_normal_half_extent_m
                )
                desired_yaw = torch.where(
                    use_stage_yaw,
                    stage_yaw,
                    normal_yaw,
                )
                yaw_error = torch.atan2(
                    torch.sin(desired_yaw - yaw), torch.cos(desired_yaw - yaw)
                )
                correction = (
                    self.cfg.tight_door_straight_heading_gain * yaw_error
                    - self.cfg.tight_door_straight_yaw_damping_gain * yaw_rate
                ).clamp(
                    -self.cfg.tight_door_straight_maximum_angular_action,
                    self.cfg.tight_door_straight_maximum_angular_action,
                )
                needs_breakaway = (
                    torch.abs(yaw_error)
                    > self.cfg.tight_door_alignment_hold_heading_error_rad
                ) & (
                    torch.abs(correction)
                    < self.cfg.tight_door_alignment_breakaway_angular_action
                ) & (
                    torch.abs(yaw_rate)
                    < self.cfg.tight_door_alignment_breakaway_maximum_yaw_rate_rad_s
                )
                correction = torch.where(
                    needs_breakaway,
                    torch.sign(yaw_error)
                    * self.cfg.tight_door_alignment_breakaway_angular_action,
                    correction,
                )
                straight_heading_action = torch.where(
                    exact_segment, correction, straight_heading_action
                )
                straight_heading_error = torch.where(
                    exact_segment, yaw_error, straight_heading_error
                )

        minimum, maximum = self.cfg.linear_velocity_range_mps
        maximum_action = (
            2.0
            * (self.cfg.tight_door_maximum_speed_mps - minimum)
            / (maximum - minimum)
            - 1.0
        )
        self._actions[slow, 0] = torch.minimum(
            self._actions[slow, 0],
            torch.full_like(self._actions[slow, 0], maximum_action),
        )
        heading_alignment_hold = alignment_scope & (
            torch.abs(straight_heading_error)
            > self.cfg.tight_door_alignment_hold_heading_error_rad
        )
        yaw_settling_hold = (
            alignment_scope
            & ~heading_alignment_hold
            & (
                torch.abs(self._robot.data.root_ang_vel_b[:, 2])
                > self.cfg.tight_door_alignment_hold_maximum_yaw_rate_rad_s
            )
        )
        alignment_hold = heading_alignment_hold | yaw_settling_hold
        # Door-normal heading is a low-level safety invariant throughout the
        # alignment envelope, including the short pre-frame stopping region.
        # Leaving angular control to the policy there can rotate the long body
        # toward a jamb while its centre is still outside the aperture.
        controller_scope = alignment_scope | no_rotation
        self._actions[controller_scope, 1] = straight_heading_action[controller_scope]
        minimum_alignment_action = (
            2.0
            * (self.cfg.tight_door_alignment_minimum_speed_mps - minimum)
            / (maximum - minimum)
            - 1.0
        )
        alignment_drive = alignment_scope & ~alignment_hold
        self._actions[alignment_drive, 0] = torch.maximum(
            self._actions[alignment_drive, 0],
            torch.full_like(
                self._actions[alignment_drive, 0], minimum_alignment_action
            ),
        )
        # The full presentation USD has materially more floor/contact load than
        # the lightweight training course. A live-only closed-loop feedforward
        # option can request extra wheel speed while regulating measured chassis
        # speed below the unchanged doorway limit. It is disabled for training
        # and evaluation and never changes collision or acceptance geometry.
        if self.cfg.tight_door_traction_compensation_enabled:
            forward_speed = self._robot.data.root_lin_vel_b[:, 0].clamp_min(0.0)
            speed_error = (
                self.cfg.tight_door_traction_target_speed_mps - forward_speed
            ).clamp_min(0.0)
            compensated_speed = (
                self.cfg.tight_door_alignment_minimum_speed_mps
                + self.cfg.tight_door_traction_gain * speed_error
            ).clamp(max=self.cfg.tight_door_traction_command_ceiling_mps)
            compensated_action = (
                2.0 * (compensated_speed - minimum) / (maximum - minimum) - 1.0
            )
            self._actions[alignment_drive, 0] = torch.maximum(
                self._actions[alignment_drive, 0],
                compensated_action[alignment_drive],
            )
            overspeed = alignment_drive & (
                forward_speed > self.cfg.tight_door_traction_overspeed_stop_mps
            )
            self._actions[overspeed, 0] = -1.0
        self._actions[alignment_hold, 0] = -1.0
        # Apply this after the straight-heading controller: once aligned, the
        # platform must coast its residual yaw rate below the release limit
        # instead of immediately receiving another proportional correction.
        self._actions[yaw_settling_hold, 1] = 0.0
        self._tight_door_slow_zone_active.copy_(slow)
        self._tight_door_no_rotation_active.copy_(no_rotation)
        self._tight_door_alignment_hold_active.copy_(alignment_hold)
        self.extras["tight_door_motion_envelope"] = {
            "simulation_only": True,
            "maximum_doorway_speed_mps": self.cfg.tight_door_maximum_speed_mps,
            "minimum_alignment_speed_mps": (
                self.cfg.tight_door_alignment_minimum_speed_mps
            ),
            "slow_zone_active": slow.clone(),
            "no_rotation_zone_active": no_rotation.clone(),
            "alignment_hold_active": alignment_hold.clone(),
            "yaw_settling_hold_active": yaw_settling_hold.clone(),
            "straight_heading_stabilization_active": controller_scope.clone(),
            "live_traction_compensation_enabled": bool(
                self.cfg.tight_door_traction_compensation_enabled
            ),
            "live_traction_target_speed_mps": (
                self.cfg.tight_door_traction_target_speed_mps
                if self.cfg.tight_door_traction_compensation_enabled
                else None
            ),
            "live_traction_overspeed_stop_mps": (
                self.cfg.tight_door_traction_overspeed_stop_mps
                if self.cfg.tight_door_traction_compensation_enabled
                else None
            ),
        }

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

        if self.cfg.randomize_contact_materials:
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
            self._obstacle_yield_offsets[obstacle_index, env_ids] = 0.0

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
            lateral = (
                self._obstacle_half_spans[obstacle_index, env_ids] * torch.sin(phase)
                + self._obstacle_yield_offsets[obstacle_index, env_ids]
            )
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
            social_retreat_velocity = torch.zeros_like(local_xy)
            if self.cfg.dynamic_obstacle_social_retreat_speed_mps > 0.0:
                robot_axis_position = torch.sum(
                    (robot_local_xy - self._obstacle_centres[obstacle_index, env_ids])
                    * self._obstacle_axes[obstacle_index, env_ids],
                    dim=1,
                )
                away_sign = torch.sign(lateral - robot_axis_position)
                away_sign = torch.where(
                    away_sign == 0.0, torch.ones_like(away_sign), away_sign
                )
                self._obstacle_yield_offsets[obstacle_index, env_ids] += (
                    yielding.float()
                    * away_sign
                    * self.cfg.dynamic_obstacle_social_retreat_speed_mps
                    * self.step_dt
                )
                self._obstacle_yield_offsets[obstacle_index, env_ids].clamp_(
                    -self.cfg.dynamic_obstacle_social_retreat_maximum_m,
                    self.cfg.dynamic_obstacle_social_retreat_maximum_m,
                )
                lateral = (
                    self._obstacle_half_spans[obstacle_index, env_ids]
                    * torch.sin(phase)
                    + self._obstacle_yield_offsets[obstacle_index, env_ids]
                )
                local_xy = (
                    self._obstacle_centres[obstacle_index, env_ids]
                    + self._obstacle_axes[obstacle_index, env_ids]
                    * lateral.unsqueeze(-1)
                )
                social_retreat_velocity = (
                    self._obstacle_axes[obstacle_index, env_ids]
                    * away_sign.unsqueeze(-1)
                    * self.cfg.dynamic_obstacle_social_retreat_speed_mps
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
            velocity[~active] = 0.0
            velocity[yielding, :2] = social_retreat_velocity[yielding]
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
        self._apply_tight_door_motion_envelope()
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
        # A continuous presentation mission advances the route segment without
        # resetting the environment. Re-sample the crossing population at that
        # hand-off so every eligible leg receives its own seeded live scenario.
        route_chain = self.extras.get("route_chain")
        if self.cfg.route_chain_mode and route_chain:
            advance = route_chain["waypoint_reached"] & (
                route_chain["reached_segment_id"] < len(ROUTE_SEGMENTS) - 1
            )
            advance_ids = torch.nonzero(advance, as_tuple=False).squeeze(-1)
            if len(advance_ids) > 0:
                self._sample_dynamic_obstacles(advance_ids)
                self._update_dynamic_obstacles(advance_ids)
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
        self._tight_door_slow_zone_active[env_ids] = False
        self._tight_door_no_rotation_active[env_ids] = False
        self._tight_door_alignment_hold_active[env_ids] = False
        self.extras["domain_randomization"] = {
            "curriculum_strength": self._curriculum_strength(),
            "action_latency_steps": self._action_latency_steps.clone(),
            "motor_strength": self._motor_strength.clone(),
            "wheel_radius_scale": self._wheel_radius_scale.clone(),
            "wheel_track_scale": self._wheel_track_scale.clone(),
            "base_mass_scale": self._mass_scale.clone(),
            "static_friction": self._static_friction.clone(),
            "dynamic_friction": self._dynamic_friction.clone(),
            "contact_material_randomization_enabled": self.cfg.randomize_contact_materials,
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
        # These actions are copied into mutable episode buffers. ``no_grad``
        # keeps inference cheap while returning ordinary tensors that can be
        # reset in place after Gym auto-resets an environment.
        with torch.no_grad():
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
class AishaPhase3DynamicSafetyEnvCfg(AishaPhase3TargetedRecoveryEnvCfg):
    """Train a 360-degree slow/stop layer outside the frozen Phase 3M stack."""

    frozen_recovery_checkpoint = str(PHASE3M_FROZEN_RECOVERY_CHECKPOINT)
    frozen_recovery_checkpoint_sha256 = PHASE3M_FROZEN_RECOVERY_CHECKPOINT_SHA256
    action_space = 1

    # A safety intervention is allowed to wait for a person. Keep the Phase 3M
    # controller untouched but add a 20-second evaluation/training patience
    # budget so a collision avoided near the end of a long return leg is not
    # automatically reclassified as a navigation failure.
    episode_length_s = 120.0
    dynamic_obstacle_social_retreat_speed_mps = 0.30

    # The complete Phase 3M runtime stack stays active and immutable. The new
    # policy receives the same 36-bin 360-degree LiDAR observation and may only
    # remove translation after Phase 3M has acted.
    recovery_supervisor_enabled = True
    safety_dynamic_segment_ids = (0, 1, 2, 5, 7, 10, 11)
    segment_sampling_weights = (
        16.0,
        16.0,
        16.0,
        4.0,
        4.0,
        16.0,
        4.0,
        16.0,
        4.0,
        4.0,
        16.0,
        16.0,
    )

    # The recurrent learner infers closing motion from successive full-ring
    # scans. Shaping is sensor-derived; pedestrian identity/velocity remains
    # evaluation truth and is never appended to the policy observation.
    safety_ring_closing_distance_m = 1.20
    safety_ring_clear_distance_m = 1.60
    safety_ring_closing_delta_m = 0.005
    safety_ring_low_clearance_m = 0.35
    reward_ring_brake_while_closing = 0.20
    penalty_ring_unmitigated_closing = -0.45
    penalty_ring_low_clearance = -0.10
    penalty_ring_unnecessary_brake = -0.015

    # The outer guard can be enabled for threshold experiments, but remains
    # disabled in the accepted training/runtime contract. A zero-output outer
    # actor must reproduce Phase 3M exactly, including its existing projected
    # footprint gate and independent front protective stop. The learned policy
    # still observes and is rewarded from the complete 360-degree ring.
    safety_emergency_guard_enabled = False
    safety_emergency_front_half_angle_rad = math.radians(90.0)
    safety_emergency_forward_trigger_clearance_m = 0.08
    safety_emergency_rotation_trigger_clearance_m = 0.04

    # Empty preserves the accepted Phase 3N contract. High-speed continuation
    # opts in only verified open, straight hallway legs; all other route legs
    # remain at the accepted 0.50 m/s learned-command ceiling.
    high_speed_segment_ids: tuple[int, ...] = ()
    non_high_speed_maximum_mps = 0.50
    high_speed_maximum_mps = 0.50
    measured_route_scoped_phase3n_thresholds_enabled = False


@configclass
class AishaPhase3DynamicSafetyStaticRegressionEnvCfg(
    AishaPhase3DynamicSafetyEnvCfg
):
    """Exercise the frozen stack with full DR but no pedestrian crossings."""

    dynamic_obstacle_activation_probability = 0.0
    dynamic_obstacle_social_retreat_speed_mps = 0.0


@configclass
class AishaPhase6HighSpeed65SafetyEnvCfg(AishaPhase3DynamicSafetyEnvCfg):
    """First adaptation tier for the unchanged robot at 0.65 m/s in open halls."""

    # Preserve the complete frozen stack's accepted 0.50 m/s normalization.
    # Only the final wheel mapping expands on declared straight hallway legs.
    high_speed_maximum_mps = 0.65
    curriculum_warmup_policy_steps = 0
    curriculum_ramp_policy_steps = 4_800
    curriculum_minimum_strength = 0.35
    high_speed_segment_ids = (1, 5)
    non_high_speed_maximum_mps = 0.50
    # Put most continuation experience on the two expanded-speed hallway
    # directions while retaining non-zero rehearsal for every frozen route leg.
    segment_sampling_weights = (
        2.0,
        36.0,
        2.0,
        1.0,
        1.0,
        36.0,
        1.0,
        2.0,
        1.0,
        1.0,
        2.0,
        2.0,
    )

    # The flat-floor 0.80 m/s gate measured a 0.638 m controlled stop. Expand
    # learned authority upstream before the target tier while retaining the
    # brake-only output boundary and full 360-degree observation contract.
    safety_ring_closing_distance_m = 1.50
    safety_ring_clear_distance_m = 1.90
    safety_ring_low_clearance_m = 0.40
    safety_ring_closing_delta_m = 0.006


@configclass
class AishaPhase6HighSpeed65StaticRegressionEnvCfg(
    AishaPhase6HighSpeed65SafetyEnvCfg
):
    """Static-route retention gate for the 0.65 m/s adaptation tier."""

    dynamic_obstacle_activation_probability = 0.0
    dynamic_obstacle_social_retreat_speed_mps = 0.0


@configclass
class AishaPhase6HighSpeed80SafetyEnvCfg(AishaPhase6HighSpeed65SafetyEnvCfg):
    """Target 0.80 m/s open-hall tier after 0.65 m/s adaptation succeeds."""

    high_speed_maximum_mps = 0.80
    curriculum_ramp_policy_steps = 6_400
    curriculum_minimum_strength = 0.65
    safety_ring_closing_distance_m = 1.80
    safety_ring_clear_distance_m = 2.20
    safety_ring_low_clearance_m = 0.45
    safety_ring_closing_delta_m = 0.007


@configclass
class AishaPhase6HighSpeed80StaticRegressionEnvCfg(
    AishaPhase6HighSpeed80SafetyEnvCfg
):
    """Static-route retention gate for the 0.80 m/s target tier."""

    dynamic_obstacle_activation_probability = 0.0
    dynamic_obstacle_social_retreat_speed_mps = 0.0


class AishaPhase3DynamicSafetyEnv(AishaPhase3TargetedRecoveryEnv):
    """Frozen Phase 3M navigation plus a trainable 360-degree safety residual."""

    cfg: AishaPhase3DynamicSafetyEnvCfg

    def _ensure_frozen_command_buffers(self) -> None:
        """Keep the one-action Gym API separate from the two-wheel command state."""
        if hasattr(self, "_actions") and self._actions.shape[-1] == 1:
            self._actions = torch.zeros((self.num_envs, 2), device=self.device)
        if hasattr(self, "_previous_actions") and self._previous_actions.shape[-1] == 1:
            self._previous_actions = torch.zeros(
                (self.num_envs, 2), device=self.device
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # DirectRLEnv sizes its generic action history from the one-dimensional
        # outer API before the frozen two-command stack is constructed.
        self._ensure_frozen_command_buffers()
        return super()._get_observations()

    def __init__(
        self,
        cfg: AishaPhase3DynamicSafetyEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_frozen_command_buffers()
        checkpoint_path = Path(self.cfg.frozen_recovery_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"missing frozen Phase 3M recovery checkpoint: {checkpoint_path}"
            )
        checkpoint_sha256 = _sha256(checkpoint_path)
        if checkpoint_sha256 != self.cfg.frozen_recovery_checkpoint_sha256:
            raise RuntimeError(
                "frozen Phase 3M recovery checkpoint hash mismatch: "
                f"{checkpoint_sha256} != {self.cfg.frozen_recovery_checkpoint_sha256}"
            )
        state = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )["model_state_dict"]
        # Module constructors initialize weights before the frozen state is
        # loaded. Preserve the environment RNG stream so adding this wrapper
        # does not silently change the matched domain-randomization sequence.
        torch_device = torch.device(self.device)
        rng_devices = (
            [torch_device.index if torch_device.index is not None else 0]
            if torch_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            frozen_recovery_memory = nn.GRU(
                input_size=46,
                hidden_size=64,
                num_layers=1,
            ).to(self.device)
            frozen_recovery_actor = nn.Sequential(
                nn.Linear(64, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 2),
            ).to(self.device)
        self._frozen_recovery_memory = frozen_recovery_memory
        memory_state = {
            key.removeprefix("memory_a.rnn."): value
            for key, value in state.items()
            if key.startswith("memory_a.rnn.")
        }
        self._frozen_recovery_memory.load_state_dict(memory_state, strict=True)
        self._frozen_recovery_actor = frozen_recovery_actor
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        }
        self._frozen_recovery_actor.load_state_dict(actor_state, strict=True)
        for module in (self._frozen_recovery_memory, self._frozen_recovery_actor):
            module.eval()
            module.requires_grad_(False)
        self._frozen_recovery_obs_mean = state["actor_obs_normalizer._mean"].to(
            self.device
        )
        self._frozen_recovery_obs_std = state["actor_obs_normalizer._std"].to(
            self.device
        )
        self._frozen_recovery_hidden = torch.zeros(
            (1, self.num_envs, 64), device=self.device
        )
        self._frozen_recovery_checkpoint_path = checkpoint_path
        self._frozen_recovery_checkpoint_actual_sha256 = checkpoint_sha256

        dynamic_ids = tuple(int(value) for value in self.cfg.safety_dynamic_segment_ids)
        if dynamic_ids != tuple(int(value) for value in self.cfg.dynamic_obstacle_segment_ids):
            raise ValueError(
                "360-degree safety authority must match the declared pedestrian segments"
            )
        self._safety_dynamic_segment_ids = torch.tensor(
            dynamic_ids, dtype=torch.long, device=self.device
        )
        self._safety_ray_angles = torch.deg2rad(
            torch.arange(-180.0, 180.0, 10.0, device=self.device)
        )
        self._safety_front_ray_mask = (
            torch.abs(self._safety_ray_angles)
            <= self.cfg.safety_emergency_front_half_angle_rad
        )
        self._safety_action_history = torch.zeros(
            (self.num_envs, 3, 1), device=self.device
        )
        self._safety_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._applied_safety_actions = torch.zeros_like(self._safety_actions)
        self._frozen_recovery_actions = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._frozen_stack_actions = torch.zeros_like(
            self._frozen_recovery_actions
        )
        # Optional integration seam for a conventional global/local planner.
        # ``None`` preserves the accepted Phase 3N training/runtime contract:
        # frozen route actor -> frozen Phase 3M recovery stack -> Phase 3N.
        # A bridge may supply normalized differential-drive commands here so
        # the same learned 360-degree safety actor is the final authority over
        # live Nav2 output.  This mode is deliberately explicit; it must never
        # be enabled by a training configuration accidentally.
        self._external_navigation_actions: torch.Tensor | None = None
        self._safety_brake_fraction = torch.zeros(self.num_envs, device=self.device)
        self._safety_angular_attenuation = torch.zeros(
            self.num_envs, device=self.device
        )
        self._safety_authority_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._safety_segment_scope = torch.zeros_like(
            self._safety_authority_active
        )
        self._high_speed_segment_scope = torch.zeros_like(
            self._safety_authority_active
        )
        self._safety_emergency_forward = torch.zeros_like(
            self._safety_authority_active
        )
        self._safety_emergency_rotation = torch.zeros_like(
            self._safety_authority_active
        )
        self._previous_ring_clearance = torch.full(
            (self.num_envs,), self.cfg.lidar_max_range_m, device=self.device
        )
        self._previous_safety_scan_clearance = torch.full(
            (self.num_envs,), self.cfg.lidar_max_range_m, device=self.device
        )
        self._episode_safety_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_safety_authority_steps = torch.zeros_like(
            self._episode_safety_steps
        )
        self._episode_safety_brake_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_safety_emergency_forward_steps = torch.zeros_like(
            self._episode_safety_steps
        )
        self._episode_safety_emergency_rotation_steps = torch.zeros_like(
            self._episode_safety_steps
        )
        self._episode_minimum_ring_clearance = torch.full(
            (self.num_envs,), self.cfg.lidar_max_range_m, device=self.device
        )
        self._episode_maximum_forward_speed = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_maximum_high_speed_segment_speed = torch.zeros(
            self.num_envs, device=self.device
        )
        for name in (
            "ring_brake_while_closing",
            "ring_unmitigated_closing",
            "ring_low_clearance",
            "ring_unnecessary_brake",
        ):
            self._episode_sums[name] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )

    def set_external_navigation_actions(
        self, actions: torch.Tensor | None
    ) -> None:
        """Select or clear externally supplied base commands for safety arbitration."""
        if actions is None:
            self._external_navigation_actions = None
            return
        if actions.shape != (self.num_envs, 2):
            raise ValueError(
                "external navigation actions must have shape "
                f"({self.num_envs}, 2), got {tuple(actions.shape)}"
            )
        self._external_navigation_actions = actions.detach().to(
            device=self.device, dtype=torch.float32
        ).clamp(-1.0, 1.0)

    def _frozen_phase3m_actions(self) -> torch.Tensor:
        if not hasattr(self, "_last_policy_observation"):
            self._last_policy_observation = super()._get_observations()[
                "policy"
            ].detach()
        normalized = (
            self._last_policy_observation - self._frozen_recovery_obs_mean
        ) / (self._frozen_recovery_obs_std + 1.0e-2)
        # Do not use inference_mode here: the returned action is stored in a
        # mutable reset buffer, and PyTorch forbids later in-place updates to an
        # inference tensor outside the inference context.
        with torch.no_grad():
            memory_output, hidden = self._frozen_recovery_memory(
                normalized.unsqueeze(0), self._frozen_recovery_hidden
            )
            self._frozen_recovery_hidden.copy_(hidden)
            return self._frozen_recovery_actor(memory_output.squeeze(0)).clamp(
                -1.0, 1.0
            )

    def _compose_dynamic_safety(
        self,
        frozen_actions: torch.Tensor,
        safety_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove motion authority without changing route or steering sign."""
        safety_actions = safety_actions.clamp(-1.0, 1.0)
        brake_fraction = torch.relu(-safety_actions[:, 0])
        angular_attenuation = torch.zeros_like(brake_fraction)
        forward_fraction = ((frozen_actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        combined = torch.stack(
            (
                forward_fraction * (1.0 - brake_fraction) * 2.0 - 1.0,
                frozen_actions[:, 1],
            ),
            dim=1,
        )
        return combined, brake_fraction, angular_attenuation

    def _write_safety_wheel_targets(self) -> None:
        minimum, maximum = self.cfg.linear_velocity_range_mps
        route_maximum = self._route_scoped_maximum_speed()
        linear = minimum + (self._actions[:, 0] + 1.0) * 0.5 * (
            route_maximum - minimum
        )
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

    def _route_scoped_maximum_speed(self) -> torch.Tensor:
        """Return the physical speed ceiling selected for each route leg."""
        return torch.where(
            self._high_speed_segment_scope,
            torch.full(
                (self.num_envs,),
                float(self.cfg.high_speed_maximum_mps),
                device=self.device,
            ),
            torch.full(
                (self.num_envs,),
                float(self.cfg.non_high_speed_maximum_mps),
                device=self.device,
            ),
        )

    def _apply_segment_speed_envelope(self) -> None:
        """Select high-speed legs without changing frozen-stack normalization."""
        segment_ids = tuple(int(value) for value in self.cfg.high_speed_segment_ids)
        if not segment_ids:
            self._high_speed_segment_scope.zero_()
            return
        declared = torch.tensor(segment_ids, dtype=torch.long, device=self.device)
        self._high_speed_segment_scope = torch.any(
            self._segment_ids.unsqueeze(1) == declared.unsqueeze(0), dim=1
        )
        minimum, accepted_maximum = self.cfg.linear_velocity_range_mps
        if not math.isclose(
            float(self.cfg.non_high_speed_maximum_mps),
            float(accepted_maximum),
            abs_tol=1.0e-9,
        ):
            raise ValueError("non-high-speed mapping must equal the frozen-stack ceiling")
        if not minimum < accepted_maximum <= self.cfg.high_speed_maximum_mps:
            raise ValueError("high-speed ceiling must expand the accepted action range")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._safety_actions = actions.clone().clamp(-1.0, 1.0)
        if self._external_navigation_actions is None:
            self._frozen_recovery_actions = self._frozen_phase3m_actions()
            # This call executes the hash-locked route actor, Phase 3M residual,
            # rectangular projection, stop latch, office-pivot supervisor, and
            # torque schedule without exposing any of them to the new optimizer.
            super()._pre_physics_step(self._frozen_recovery_actions)
        else:
            # Nav2 has already selected the requested translation and heading
            # rate. Keep the learned actor outside that command exactly as it is
            # outside the frozen Phase 3M stack. Dynamic obstacle motion remains
            # part of the environment and wheel targets are written below only
            # after safety arbitration.
            self._update_dynamic_obstacles()
            self._frozen_recovery_actions.zero_()
            self._actions = self._external_navigation_actions.clone()
        self._apply_segment_speed_envelope()
        self._frozen_stack_actions.copy_(self._actions)

        self._safety_action_history[:, 2] = self._safety_action_history[:, 1]
        self._safety_action_history[:, 1] = self._safety_action_history[:, 0]
        self._safety_action_history[:, 0] = self._safety_actions
        gather_index = self._action_latency_steps.view(-1, 1, 1).expand(-1, 1, 1)
        self._applied_safety_actions = torch.gather(
            self._safety_action_history, 1, gather_index
        ).squeeze(1)
        self._safety_segment_scope = torch.any(
            self._segment_ids.unsqueeze(1)
            == self._safety_dynamic_segment_ids.unsqueeze(0),
            dim=1,
        )
        requested, brake_fraction, angular_attenuation = self._compose_dynamic_safety(
            self._actions, self._applied_safety_actions
        )
        exact_ranges = self._lidar_ranges()
        clearances = exact_ranges - self._lidar_envelope_ranges.unsqueeze(0)
        front_clearance = torch.amin(
            clearances[:, self._safety_front_ray_mask], dim=1
        )
        ring_clearance = torch.amin(clearances, dim=1)
        safety_scan_closing_delta = (
            self._previous_safety_scan_clearance - ring_clearance
        ).clamp_min(0.0)
        closing_distance = torch.full_like(
            ring_clearance, float(self.cfg.safety_ring_closing_distance_m)
        )
        closing_delta = torch.full_like(
            ring_clearance, float(self.cfg.safety_ring_closing_delta_m)
        )
        if self.cfg.measured_route_scoped_phase3n_thresholds_enabled:
            closing_distance = torch.where(
                self._high_speed_segment_scope,
                closing_distance,
                torch.full_like(closing_distance, 1.20),
            )
            closing_delta = torch.where(
                self._high_speed_segment_scope,
                closing_delta,
                torch.full_like(closing_delta, 0.005),
            )
        self._safety_authority_active = (
            self._safety_segment_scope
            & (ring_clearance < closing_distance)
            & (
                safety_scan_closing_delta
                > closing_delta
            )
        )
        self._previous_safety_scan_clearance.copy_(ring_clearance)
        self._safety_brake_fraction = torch.where(
            self._safety_authority_active,
            brake_fraction,
            torch.zeros_like(brake_fraction),
        )
        self._safety_angular_attenuation = torch.where(
            self._safety_authority_active,
            angular_attenuation,
            torch.zeros_like(angular_attenuation),
        )
        applied = torch.where(
            self._safety_authority_active.unsqueeze(1), requested, self._actions
        )
        moving_forward = applied[:, 0] > -1.0 + 1.0e-6
        rotating = torch.abs(applied[:, 1]) > 1.0e-6
        self._safety_emergency_forward = (
            self.cfg.safety_emergency_guard_enabled
            & moving_forward
            & (
                front_clearance
                <= self.cfg.safety_emergency_forward_trigger_clearance_m
            )
        )
        self._safety_emergency_rotation = (
            self.cfg.safety_emergency_guard_enabled
            & rotating
            & (
                ring_clearance
                <= self.cfg.safety_emergency_rotation_trigger_clearance_m
            )
        )
        applied[:, 0] = torch.where(
            self._safety_emergency_forward,
            torch.full_like(applied[:, 0], -1.0),
            applied[:, 0],
        )
        applied[:, 1] = torch.where(
            self._safety_emergency_rotation,
            torch.zeros_like(applied[:, 1]),
            applied[:, 1],
        )
        self._actions = applied.clamp(-1.0, 1.0)
        self._write_safety_wheel_targets()

        self._episode_safety_steps += 1
        self._episode_safety_authority_steps += self._safety_authority_active.long()
        self._episode_safety_brake_sum += self._safety_brake_fraction
        self._episode_safety_emergency_forward_steps += (
            self._safety_emergency_forward.long()
        )
        self._episode_safety_emergency_rotation_steps += (
            self._safety_emergency_rotation.long()
        )
        self._episode_minimum_ring_clearance.copy_(
            torch.minimum(self._episode_minimum_ring_clearance, ring_clearance)
        )
        forward_speed = torch.abs(self._robot.data.root_lin_vel_b[:, 0])
        self._episode_maximum_forward_speed.copy_(
            torch.maximum(self._episode_maximum_forward_speed, forward_speed)
        )
        self._episode_maximum_high_speed_segment_speed.copy_(
            torch.where(
                self._high_speed_segment_scope,
                torch.maximum(
                    self._episode_maximum_high_speed_segment_speed, forward_speed
                ),
                self._episode_maximum_high_speed_segment_speed,
            )
        )

    def _get_rewards(self) -> torch.Tensor:
        rewards = super()._get_rewards()
        ring_clearance = torch.amin(
            self._lidar_ranges() - self._lidar_envelope_ranges.unsqueeze(0), dim=1
        )
        closing_delta = (
            self._previous_ring_clearance - ring_clearance
        ).clamp_min(0.0)
        scope = self._safety_segment_scope.float()
        closing = (
            (ring_clearance < self.cfg.safety_ring_closing_distance_m)
            & (closing_delta > self.cfg.safety_ring_closing_delta_m)
        ).float() * scope
        clear = (
            ring_clearance > self.cfg.safety_ring_clear_distance_m
        ).float() * scope
        normalized_forward = ((self._actions[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        low_clearance_fraction = (
            torch.relu(self.cfg.safety_ring_low_clearance_m - ring_clearance)
            / self.cfg.safety_ring_low_clearance_m
        ).clamp(0.0, 1.0) * scope

        shaped_rewards = (
            (
                "ring_brake_while_closing",
                closing
                * self._safety_brake_fraction
                * self.cfg.reward_ring_brake_while_closing,
            ),
            (
                "ring_unmitigated_closing",
                closing
                * (1.0 - self._safety_brake_fraction)
                * normalized_forward
                * self.cfg.penalty_ring_unmitigated_closing,
            ),
            (
                "ring_low_clearance",
                low_clearance_fraction * self.cfg.penalty_ring_low_clearance,
            ),
            (
                "ring_unnecessary_brake",
                clear
                * self._safety_brake_fraction
                * self.cfg.penalty_ring_unnecessary_brake,
            ),
        )
        for name, value in shaped_rewards:
            self._episode_sums[name] += value
            rewards += value
        self._previous_ring_clearance.copy_(ring_clearance)
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if hasattr(self, "_episode_safety_steps"):
            self.extras["episode_outcomes"].update(
                {
                    "safety_steps": self._episode_safety_steps.clone(),
                    "safety_authority_steps": (
                        self._episode_safety_authority_steps.clone()
                    ),
                    "safety_brake_fraction_sum": (
                        self._episode_safety_brake_sum.clone()
                    ),
                    "safety_emergency_forward_steps": (
                        self._episode_safety_emergency_forward_steps.clone()
                    ),
                    "safety_emergency_rotation_steps": (
                        self._episode_safety_emergency_rotation_steps.clone()
                    ),
                    "minimum_ring_clearance_m": (
                        self._episode_minimum_ring_clearance.clone()
                    ),
                    "maximum_forward_speed_mps": (
                        self._episode_maximum_forward_speed.clone()
                    ),
                    "maximum_high_speed_segment_speed_mps": (
                        self._episode_maximum_high_speed_segment_speed.clone()
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
        if not hasattr(self, "_safety_action_history"):
            return
        self._frozen_recovery_hidden[:, env_ids] = 0.0
        self._safety_action_history[env_ids] = 0.0
        self._safety_actions[env_ids] = 0.0
        self._applied_safety_actions[env_ids] = 0.0
        self._frozen_recovery_actions[env_ids] = 0.0
        self._frozen_stack_actions[env_ids] = 0.0
        self._safety_brake_fraction[env_ids] = 0.0
        self._safety_angular_attenuation[env_ids] = 0.0
        self._safety_authority_active[env_ids] = False
        self._safety_segment_scope[env_ids] = False
        self._high_speed_segment_scope[env_ids] = False
        self._safety_emergency_forward[env_ids] = False
        self._safety_emergency_rotation[env_ids] = False
        ring_clearance = torch.amin(
            self._lidar_ranges()[env_ids]
            - self._lidar_envelope_ranges.unsqueeze(0),
            dim=1,
        )
        self._previous_ring_clearance[env_ids] = ring_clearance
        self._previous_safety_scan_clearance[env_ids] = ring_clearance
        self._episode_safety_steps[env_ids] = 0
        self._episode_safety_authority_steps[env_ids] = 0
        self._episode_safety_brake_sum[env_ids] = 0.0
        self._episode_safety_emergency_forward_steps[env_ids] = 0
        self._episode_safety_emergency_rotation_steps[env_ids] = 0
        self._episode_minimum_ring_clearance[env_ids] = self.cfg.lidar_max_range_m
        self._episode_maximum_forward_speed[env_ids] = 0.0
        self._episode_maximum_high_speed_segment_speed[env_ids] = 0.0
        self.extras["dynamic_safety"] = {
            "architecture": "outer_recurrent_360_degree_lidar_safety_residual",
            "frozen_recovery_checkpoint": str(
                self._frozen_recovery_checkpoint_path
            ),
            "frozen_recovery_checkpoint_sha256": (
                self._frozen_recovery_checkpoint_actual_sha256
            ),
            "frozen_route_checkpoint": str(self._frozen_route_checkpoint_path),
            "frozen_route_checkpoint_sha256": (
                self._frozen_route_checkpoint_actual_sha256
            ),
            "policy_authority": (
                "braking only while 360-degree LiDAR clearance is closing on "
                "declared pedestrian segments"
            ),
            "lidar_horizontal_field_of_view_deg": 360.0,
            "lidar_bins": self.cfg.lidar_training_bins,
            "may_increase_speed": False,
            "may_reverse": False,
            "may_flip_steering_sign": False,
            "duplicate_outer_emergency_guard_enabled": (
                self.cfg.safety_emergency_guard_enabled
            ),
            "high_speed_segment_ids": tuple(self.cfg.high_speed_segment_ids),
            "non_high_speed_maximum_mps": self.cfg.non_high_speed_maximum_mps,
            "high_speed_maximum_mps": self.cfg.high_speed_maximum_mps,
        }


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
