#!/usr/bin/env python3
"""Expose the live administration physics scene to ROS 2/Nav2.

This is the simulation motion boundary: Nav2 may command the articulated Rev D
differential drive through /cmd_vel while Isaac Sim publishes /clock, /odom,
/tf, /scan and /front_scan. With ``--phase3n-safety-checkpoint``, the accepted
learned 360-degree safety actor becomes the final authority over Nav2's base
command before wheel targets are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AISHA-Administration-Live-Direct-v0")
parser.add_argument("--max-steps", type=int, default=0, help="zero runs until the app closes")
parser.add_argument(
    "--self-test",
    action="store_true",
    help="inject a short 0.05 m/s command without requiring Nav2",
)
parser.add_argument("--output-report", type=Path)
parser.add_argument(
    "--phase3n-safety-checkpoint",
    type=Path,
    help="load a Phase 3N checkpoint and arbitrate every Nav2 command through it",
)
parser.add_argument(
    "--mapped-safety-overlay",
    type=Path,
    help="enable the measured-doorway and central-polygon presentation guard",
)
parser.add_argument(
    "--mapped-safety-site-config",
    type=Path,
    help="site geometry containing mapped door centres and wall orientations",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
import aisha_isaaclab.tasks  # noqa: E402,F401
from aisha_isaaclab.tasks.office_nav.mapped_nav2_safety import (  # noqa: E402
    MappedNav2SafetyGuard,
)


PHASE3N_PRESENTATION_TASK = (
    "Isaac-AISHA-Administration-Live-Phase3-DynamicSafety-Presentation-Direct-v0"
)
PHASE3N_MEASURED_NAV2_TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-DynamicSafety-Direct-v0"
)
PHASE3N_COMPATIBLE_TASKS = {
    PHASE3N_PRESENTATION_TASK,
    PHASE3N_MEASURED_NAV2_TASK,
}
ACCEPTED_PHASE3N_CHECKPOINT = "aisha_phase3n_dynamic_safety_model_50.pt"
ACCEPTED_PHASE3N_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ros_stamp(simulation_time_s: float):
    from builtin_interfaces.msg import Time

    stamp = Time()
    stamp.sec = int(simulation_time_s)
    stamp.nanosec = int(round((simulation_time_s - stamp.sec) * 1_000_000_000))
    if stamp.nanosec >= 1_000_000_000:
        stamp.sec += 1
        stamp.nanosec -= 1_000_000_000
    return stamp


class AishaSimulationBridge:
    """Small direct ROS publisher/subscriber around the Isaac Lab scene."""

    def __init__(
        self,
        raw_env,
        control_period_s: float,
        sensor_positions: dict[str, list[float]],
        mapped_guard: MappedNav2SafetyGuard | None = None,
    ):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Bool
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self.rclpy = rclpy
        self.raw_env = raw_env
        self.control_period_s = control_period_s
        self.mapped_guard = mapped_guard
        self.node = rclpy.create_node(
            "aisha_isaac_administration_bridge",
            parameter_overrides=[],
            automatically_declare_parameters_from_overrides=True,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.clock_publisher = self.node.create_publisher(Clock, "/clock", 10)
        self.odom_publisher = self.node.create_publisher(Odometry, "/odom", 10)
        self.crown_publisher = self.node.create_publisher(LaserScan, "/scan", sensor_qos)
        self.front_publisher = self.node.create_publisher(
            LaserScan, "/front_scan", sensor_qos
        )
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self.node)
        self.subscription = self.node.create_subscription(
            Twist, "/cmd_vel", self._command_callback, 10
        )
        self.mission_complete_subscription = self.node.create_subscription(
            Bool, "/aisha/mission_complete", self._mission_complete_callback, 10
        )
        self.command_linear_mps = 0.0
        self.command_angular_rad_s = 0.0
        self.command_received_at_s = -math.inf
        self.simulation_time_s = 0.0
        self.stop_latched = False
        self.front_stop_trigger_m = 0.60
        self.front_stop_release_m = 0.75
        self.front_stop_half_angle_deg = 10.0
        self.command_watchdog_s = 0.30
        self.maximum_forward_mps = 0.30
        self.maximum_angular_rad_s = 0.55
        self.minimum_in_place_angular_rad_s = 0.30
        self.minimum_rotation_clearance_m = 0.08
        self.rejected_reverse_commands = 0
        self.rejected_lateral_commands = 0
        self.watchdog_stops = 0
        self.obstacle_stops = 0
        self.in_place_deadband_compensations = 0
        self.front_latch_rotation_escape_steps = 0
        self.rotation_guard_stops = 0
        self.exit_requested = False
        self.messages = {
            "clock": 0,
            "odom": 0,
            "tf": 0,
            "tf_static": 0,
            "scan": 0,
            "front_scan": 0,
        }
        self.minimum_front_range_m = math.inf
        self.minimum_central_front_range_m = math.inf
        self.minimum_ring_clearance_m = math.inf
        self.publish_static_transforms(sensor_positions)

    def _mission_complete_callback(self, message) -> None:
        if bool(message.data):
            self.exit_requested = True

    def publish_static_transforms(self, sensor_positions: dict[str, list[float]]) -> None:
        from geometry_msgs.msg import TransformStamped

        transforms = []
        for child, position in sensor_positions.items():
            transform = TransformStamped()
            transform.header.stamp = ros_stamp(0.0)
            transform.header.frame_id = "base_link"
            transform.child_frame_id = child
            transform.transform.translation.x = position[0]
            transform.transform.translation.y = position[1]
            transform.transform.translation.z = position[2]
            transform.transform.rotation.w = 1.0
            transforms.append(transform)
        self.static_tf_broadcaster.sendTransform(transforms)
        self.messages["tf_static"] = len(transforms)

    def _command_callback(self, message) -> None:
        if message.linear.x < 0.0:
            self.rejected_reverse_commands += 1
        if abs(message.linear.y) > 1.0e-6:
            self.rejected_lateral_commands += 1
        self.command_linear_mps = min(self.maximum_forward_mps, max(0.0, message.linear.x))
        self.command_angular_rad_s = min(
            self.maximum_angular_rad_s,
            max(-self.maximum_angular_rad_s, message.angular.z),
        )
        self.command_received_at_s = self.simulation_time_s

    def set_self_test_command(self, linear_mps: float, angular_rad_s: float = 0.0) -> None:
        self.command_linear_mps = min(self.maximum_forward_mps, max(0.0, linear_mps))
        self.command_angular_rad_s = min(
            self.maximum_angular_rad_s, max(-self.maximum_angular_rad_s, angular_rad_s)
        )
        self.command_received_at_s = self.simulation_time_s

    def spin(self) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def front_ranges(self) -> torch.Tensor:
        sensor = self.raw_env.scene.sensors["front_lidar"]
        vectors = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
        ranges = torch.linalg.norm(vectors, dim=-1)
        return torch.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=0.12).clamp(
            0.12, 10.0
        )

    def command_action(self) -> torch.Tensor:
        front_ranges = self.front_ranges()[0]
        front_min = float(torch.amin(front_ranges).item())
        centre_index = int(front_ranges.numel() // 2)
        half_width = max(1, round(self.front_stop_half_angle_deg / 5.0))
        central_front_min = float(
            torch.amin(
                front_ranges[
                    max(0, centre_index - half_width) : centre_index + half_width + 1
                ]
            ).item()
        )
        ring_clearance = float(
            torch.amin(
                self.raw_env._lidar_ranges()[0]
                - self.raw_env._lidar_envelope_ranges
            ).item()
        )
        self.minimum_front_range_m = min(self.minimum_front_range_m, front_min)
        self.minimum_central_front_range_m = min(
            self.minimum_central_front_range_m, central_front_min
        )
        self.minimum_ring_clearance_m = min(
            self.minimum_ring_clearance_m, ring_clearance
        )
        if self.stop_latched:
            self.stop_latched = central_front_min < self.front_stop_release_m
        elif central_front_min <= self.front_stop_trigger_m:
            self.stop_latched = True
            self.obstacle_stops += 1

        linear = self.command_linear_mps
        angular = self.command_angular_rad_s
        if self.mapped_guard is not None:
            local_xy = self.raw_env._local_xy()[0]
            quaternion = self.raw_env._robot.data.root_quat_w[0]
            yaw = math.atan2(
                2.0
                * (
                    float(quaternion[0] * quaternion[3])
                    + float(quaternion[1] * quaternion[2])
                ),
                1.0
                - 2.0
                * (
                    float(quaternion[2] * quaternion[2])
                    + float(quaternion[3] * quaternion[3])
                ),
            )
            guard_result = self.mapped_guard.apply(
                x_m=float(local_xy[0].item()),
                y_m=float(local_xy[1].item()),
                yaw_rad=yaw,
                yaw_rate_rad_s=float(
                    self.raw_env._robot.data.root_ang_vel_b[0, 2].item()
                ),
                forward_speed_mps=float(
                    self.raw_env._robot.data.root_lin_vel_b[0, 0].item()
                ),
                requested_linear_mps=linear,
                requested_angular_rad_s=angular,
            )
            linear = guard_result.linear_mps
            angular = guard_result.angular_rad_s

        stale = self.simulation_time_s - self.command_received_at_s > self.command_watchdog_s
        if stale and (self.command_linear_mps != 0.0 or self.command_angular_rad_s != 0.0):
            self.watchdog_stops += 1
        linear = 0.0 if stale or self.stop_latched else linear
        angular = 0.0 if stale else angular
        if self.stop_latched and abs(angular) > 1.0e-3:
            # A front-only latch that also suppresses yaw can deadlock at a
            # doorway edge: DWB needs to rotate away before the front beam can
            # reach its release distance. Permit zero-translation escape yaw
            # only while the exact 360-degree footprint clearance remains above
            # the conservative rotation guard.
            if ring_clearance > self.minimum_rotation_clearance_m:
                self.front_latch_rotation_escape_steps += 1
            else:
                angular = 0.0
                self.rotation_guard_stops += 1
        if (
            linear < 1.0e-3
            and 1.0e-3 < abs(angular) < self.minimum_in_place_angular_rad_s
        ):
            # The raw Isaac velocity drives do not include the low-level
            # friction compensation present on a physical motor controller.
            # Preserve direction but lift a requested pivot out of the USD
            # furnishing/floor static-friction deadband.
            angular = math.copysign(self.minimum_in_place_angular_rad_s, angular)
            self.in_place_deadband_compensations += 1

        action = torch.zeros((1, 2), device=self.raw_env.device)
        # The frozen environment maps [-1,+1] to [0,0.50] m/s and angular
        # action directly to [-1,+1] rad/s. Keep that established contract.
        action[0, 0] = 2.0 * linear / self.raw_env.cfg.linear_velocity_range_mps[1] - 1.0
        action[0, 1] = angular / self.raw_env.cfg.angular_velocity_max_rad_s
        return action.clamp(-1.0, 1.0)

    def _publish_scan(
        self,
        publisher,
        frame_id: str,
        ranges: torch.Tensor,
        angle_min: float,
        angle_max: float,
        scan_time_s: float,
        message_key: str,
    ) -> None:
        from sensor_msgs.msg import LaserScan

        values = ranges.detach().cpu().tolist()
        message = LaserScan()
        message.header.stamp = ros_stamp(self.simulation_time_s)
        message.header.frame_id = frame_id
        message.angle_min = angle_min
        message.angle_max = angle_max
        message.angle_increment = (angle_max - angle_min) / max(1, len(values) - 1)
        message.time_increment = scan_time_s / max(1, len(values))
        message.scan_time = scan_time_s
        message.range_min = 0.12
        message.range_max = 10.0
        message.ranges = values
        publisher.publish(message)
        self.messages[message_key] += 1

    def publish(self, step_index: int) -> None:
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock

        stamp = ros_stamp(self.simulation_time_s)
        clock = Clock()
        clock.clock = stamp
        self.clock_publisher.publish(clock)
        self.messages["clock"] += 1

        robot = self.raw_env._robot
        origin = self.raw_env.scene.env_origins[0]
        position = robot.data.root_pos_w[0] - origin
        quaternion = robot.data.root_quat_w[0]
        linear = robot.data.root_lin_vel_b[0]
        angular = robot.data.root_ang_vel_b[0]

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(position[0].item())
        odom.pose.pose.position.y = float(position[1].item())
        odom.pose.pose.position.z = float(position[2].item())
        odom.pose.pose.orientation.w = float(quaternion[0].item())
        odom.pose.pose.orientation.x = float(quaternion[1].item())
        odom.pose.pose.orientation.y = float(quaternion[2].item())
        odom.pose.pose.orientation.z = float(quaternion[3].item())
        odom.twist.twist.linear.x = float(linear[0].item())
        odom.twist.twist.linear.y = float(linear[1].item())
        odom.twist.twist.angular.z = float(angular[2].item())
        self.odom_publisher.publish(odom)
        self.messages["odom"] += 1

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = odom.pose.pose.position.z
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)
        self.messages["tf"] += 1

        if step_index % max(1, round(0.10 / self.control_period_s)) == 0:
            self._publish_scan(
                self.crown_publisher,
                "lidar_link",
                self.raw_env._lidar_ranges()[0],
                -math.pi,
                math.pi - math.radians(10.0),
                0.10,
                "scan",
            )
        if step_index % max(1, round(0.05 / self.control_period_s)) == 0:
            self._publish_scan(
                self.front_publisher,
                "front_lidar_link",
                self.front_ranges()[0],
                -math.radians(60.0),
                math.radians(60.0),
                0.05,
                "front_scan",
            )

    def close(self) -> None:
        self.node.destroy_node()


def main() -> int:
    output = args.output_report or (
        PACKAGE_ROOT / "results" / "administration_nav2_bridge_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    import rclpy
    import yaml

    if not rclpy.ok():
        rclpy.init()
    learned_safety_enabled = args.phase3n_safety_checkpoint is not None
    mapped_guard = None
    mapped_overlay_path = None
    mapped_site_config_path = None
    if args.mapped_safety_overlay is not None:
        if args.mapped_safety_site_config is None:
            raise ValueError(
                "--mapped-safety-overlay requires --mapped-safety-site-config"
            )
        mapped_overlay_path = args.mapped_safety_overlay.expanduser().resolve()
        mapped_site_config_path = args.mapped_safety_site_config.expanduser().resolve()
        if not mapped_overlay_path.is_file():
            raise FileNotFoundError(mapped_overlay_path)
        if not mapped_site_config_path.is_file():
            raise FileNotFoundError(mapped_site_config_path)
        mapped_guard = MappedNav2SafetyGuard.from_site_configs(
            yaml.safe_load(mapped_site_config_path.read_text(encoding="utf-8")),
            yaml.safe_load(mapped_overlay_path.read_text(encoding="utf-8")),
        )
    task = args.task
    checkpoint = None
    checkpoint_sha256 = None
    if learned_safety_enabled:
        checkpoint = args.phase3n_safety_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)
        if task == parser.get_default("task"):
            task = PHASE3N_PRESENTATION_TASK
        if task not in PHASE3N_COMPATIBLE_TASKS:
            raise ValueError(
                "--phase3n-safety-checkpoint requires a deterministic compatible "
                f"Phase 3N task: {sorted(PHASE3N_COMPATIBLE_TASKS)}"
            )

    cfg = parse_env_cfg(task, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = 3600.0
    cfg.route_chain_mode = True
    # Nav2 and AMCL are initialized at the disclosed map origin.  Disable the
    # training-only reset jitter so odom, map, and the mission start agree.
    cfg.start_lateral_jitter_m = 0.0
    cfg.start_yaw_jitter_rad = 0.0
    cfg.goal_jitter_m = 0.0
    # Nav2 owns mission completion in bridge mode. Keep the DirectRLEnv's
    # training success threshold effectively unreachable so it cannot auto-
    # reset the robot between Nav2 action success and the completion signal.
    cfg.goal_tolerance_m = 0.001
    cfg.goal_tolerance_m_by_segment = tuple(
        0.001 for _ in cfg.goal_tolerance_m_by_segment
    )
    if learned_safety_enabled:
        # Isolate command/safety behavior from training-time domain randomization
        # for this reproducible integration gate. The checkpoint itself remains
        # unchanged and hash-verified; its formal randomized evaluations remain
        # separate evidence.
        deterministic_overrides = {
            "dynamic_obstacle_activation_probability": 0.0,
            "action_latency_steps_range": (0, 0),
            "motor_strength_scale_range": (1.0, 1.0),
            "wheel_radius_scale_range": (1.0, 1.0),
            "wheel_track_scale_range": (1.0, 1.0),
            "drive_joint_damping_range": (120.0, 120.0),
            "base_mass_scale_range": (1.0, 1.0),
            "robot_static_friction_range": (0.60, 0.60),
            "robot_dynamic_friction_range": (0.50, 0.50),
            "observation_lidar_noise_std_m": 0.0,
            "observation_lidar_dropout_probability": 0.0,
            "lidar_episode_bias_range_m": (0.0, 0.0),
            "lidar_episode_scale_range": (1.0, 1.0),
        }
        for name, value in deterministic_overrides.items():
            if hasattr(cfg, name):
                setattr(cfg, name, value)

    # Isaac Sim 5.1's Replicator global seeding path conflicts with the ROS
    # bridge's headless stage initialization. Seed Torch directly instead; all
    # remaining task randomization is pinned to deterministic ranges above.
    torch.manual_seed(6084)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(6084)
    raw_gym_env = gym.make(task, cfg=cfg)
    raw_env = raw_gym_env.unwrapped
    env = raw_gym_env
    observations = None
    policy = None
    policy_network = None
    if learned_safety_enabled:
        if not hasattr(raw_env, "set_external_navigation_actions"):
            raise TypeError(f"task {task} does not expose learned safety arbitration")
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        device = args.device or "cuda:0"
        agent_cfg.seed = 6084
        agent_cfg.device = device
        env = RslRlVecEnvWrapper(raw_gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=raw_env.device)
        try:
            policy_network = runner.alg.policy
        except AttributeError:
            policy_network = runner.alg.actor_critic
        observations = env.get_observations()

    control_period_s = float(cfg.sim.dt * cfg.decimation)
    sensor_config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "sensors.yaml").read_text(encoding="utf-8")
    )
    sensor_positions = {
        sensor_config["frames"][name]["prim_name"]: sensor_config["frames"][name][
            "position_m"
        ]
        for name in ("crown_lidar", "front_lidar", "imu")
    }
    bridge = AishaSimulationBridge(
        raw_env,
        control_period_s,
        sensor_positions,
        mapped_guard=mapped_guard,
    )
    steps_completed = 0
    reset_detected = False
    safety_authority_steps = 0
    safety_brake_steps = 0
    safety_brake_sum = 0.0
    maximum_safety_brake_fraction = 0.0
    minimum_ring_clearance_m = math.inf
    last_pre_step_position_xy_m = None
    last_pre_step_minimum_ring_clearance_m = None
    try:
        if not learned_safety_enabled:
            env.reset()
        while (
            simulation_app.is_running()
            and not bridge.exit_requested
            and (args.max_steps <= 0 or steps_completed < args.max_steps)
        ):
            bridge.spin()
            if args.self_test:
                if 15 <= steps_completed < 75:
                    bridge.set_self_test_command(0.05)
                elif steps_completed == 75:
                    bridge.set_self_test_command(0.0)
            navigation_action = bridge.command_action()
            pre_step_position = raw_env._local_xy()[0].detach().cpu().tolist()
            pre_step_ring_clearance = float(
                torch.amin(
                    raw_env._lidar_ranges()[0] - raw_env._lidar_envelope_ranges
                ).item()
            )
            last_pre_step_position_xy_m = [
                round(float(value), 6) for value in pre_step_position
            ]
            last_pre_step_minimum_ring_clearance_m = round(
                pre_step_ring_clearance, 6
            )
            if learned_safety_enabled:
                raw_env.set_external_navigation_actions(navigation_action)
                with torch.inference_mode():
                    safety_action = policy(observations)
                    observations, _, dones, _ = env.step(safety_action)
                    policy_network.reset(dones)
                terminated_or_truncated = dones
                authority = bool(raw_env._safety_authority_active[0].item())
                brake_fraction = float(raw_env._safety_brake_fraction[0].item())
                ring_clearance = float(
                    torch.amin(
                        raw_env._lidar_ranges()[0] - raw_env._lidar_envelope_ranges
                    ).item()
                )
                safety_authority_steps += int(authority)
                safety_brake_steps += int(brake_fraction > 1.0e-6)
                safety_brake_sum += brake_fraction
                maximum_safety_brake_fraction = max(
                    maximum_safety_brake_fraction, brake_fraction
                )
                minimum_ring_clearance_m = min(
                    minimum_ring_clearance_m, ring_clearance
                )
            else:
                _, _, terminated, truncated, _ = env.step(navigation_action)
                terminated_or_truncated = terminated | truncated
            bridge.simulation_time_s += control_period_s
            bridge.publish(steps_completed)
            steps_completed += 1
            if bool(torch.any(terminated_or_truncated).item()):
                reset_detected = True
                break
    finally:
        position = raw_env._local_xy()[0].detach().cpu().tolist()
        exact_ranges = raw_env._lidar_ranges()[0]
        exact_clearances = exact_ranges - raw_env._lidar_envelope_ranges
        minimum_clearance_index = int(torch.argmin(exact_clearances).item())
        minimum_hit = raw_env.scene.sensors["crown_lidar"].data.ray_hits_w[
            0, minimum_clearance_index
        ]
        episode_outcomes = raw_env.extras.get("episode_outcomes", {})
        termination_diagnostics = {
            "minimum_clearance_ray_index": minimum_clearance_index,
            "minimum_clearance_ray_angle_deg": -180.0
            + 10.0 * minimum_clearance_index,
            "minimum_exact_range_m": round(
                float(exact_ranges[minimum_clearance_index].item()), 6
            ),
            "minimum_exact_clearance_m": round(
                float(exact_clearances[minimum_clearance_index].item()), 6
            ),
            "minimum_hit_world_xyz_m": [
                round(float(value), 6) for value in minimum_hit.detach().cpu().tolist()
            ],
            "episode_outcomes": {
                key: bool(value[0].item())
                for key, value in episode_outcomes.items()
                if isinstance(value, torch.Tensor)
                and value.dtype == torch.bool
                and value.numel() >= 1
            },
        }
        report = {
            "report_type": "administration_nav2_bridge",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "bridge_smoke_completed" if steps_completed > 0 else "bridge_not_exercised",
            "task": task,
            "steps_completed": steps_completed,
            "control_period_s": control_period_s,
            "topics": {
                "subscribed": ["/cmd_vel"],
                "published": [
                    "/clock",
                    "/odom",
                    "/tf",
                    "/tf_static",
                    "/scan",
                    "/front_scan",
                ],
                "message_counts": bridge.messages,
            },
            "command_constraints": {
                "maximum_forward_mps": bridge.maximum_forward_mps,
                "reverse_allowed": False,
                "lateral_motion_allowed": False,
                "maximum_angular_rad_s": bridge.maximum_angular_rad_s,
                "minimum_in_place_angular_rad_s": bridge.minimum_in_place_angular_rad_s,
                "command_watchdog_s": bridge.command_watchdog_s,
                "front_stop_trigger_m": bridge.front_stop_trigger_m,
                "front_stop_release_m": bridge.front_stop_release_m,
                "front_stop_half_angle_deg": bridge.front_stop_half_angle_deg,
                "minimum_rotation_clearance_m": bridge.minimum_rotation_clearance_m,
            },
            "events": {
                "rejected_reverse_commands": bridge.rejected_reverse_commands,
                "rejected_lateral_commands": bridge.rejected_lateral_commands,
                "watchdog_stops": bridge.watchdog_stops,
                "obstacle_stops": bridge.obstacle_stops,
                "in_place_deadband_compensations": bridge.in_place_deadband_compensations,
                "front_latch_rotation_escape_steps": bridge.front_latch_rotation_escape_steps,
                "rotation_guard_stops": bridge.rotation_guard_stops,
                "mission_complete_signal_received": bridge.exit_requested,
                "episode_reset_gate_detected": reset_detected,
            },
            "termination_diagnostics": termination_diagnostics,
            "last_pre_step_snapshot": {
                "position_xy_m": last_pre_step_position_xy_m,
                "minimum_ring_clearance_m": last_pre_step_minimum_ring_clearance_m,
            },
            "learned_safety": {
                "enabled": learned_safety_enabled,
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_is_accepted_phase3n": bool(
                    checkpoint is not None
                    and checkpoint.name == ACCEPTED_PHASE3N_CHECKPOINT
                    and checkpoint_sha256 == ACCEPTED_PHASE3N_SHA256
                ),
                "base_command_source": "nav2_cmd_vel" if learned_safety_enabled else None,
                "authority_steps": safety_authority_steps,
                "brake_steps": safety_brake_steps,
                "mean_brake_fraction_while_active": (
                    safety_brake_sum / safety_authority_steps
                    if safety_authority_steps > 0
                    else 0.0
                ),
                "maximum_brake_fraction": maximum_safety_brake_fraction,
                "minimum_360_clearance_m": (
                    round(minimum_ring_clearance_m, 5)
                    if math.isfinite(minimum_ring_clearance_m)
                    else None
                ),
                "deterministic_static_integration_gate": learned_safety_enabled,
            },
            "mapped_site_safety": (
                {
                    **mapped_guard.report(),
                    "overlay": str(mapped_overlay_path),
                    "overlay_sha256": sha256_file(mapped_overlay_path),
                    "site_config": str(mapped_site_config_path),
                    "site_config_sha256": sha256_file(mapped_site_config_path),
                }
                if mapped_guard is not None
                else {"enabled": False}
            ),
            "final_position_xy_m": [round(float(value), 5) for value in position],
            "minimum_front_range_m": (
                round(bridge.minimum_front_range_m, 5)
                if math.isfinite(bridge.minimum_front_range_m)
                else None
            ),
            "minimum_central_front_range_m": (
                round(bridge.minimum_central_front_range_m, 5)
                if math.isfinite(bridge.minimum_central_front_range_m)
                else None
            ),
            "minimum_ring_clearance_m": (
                round(bridge.minimum_ring_clearance_m, 5)
                if math.isfinite(bridge.minimum_ring_clearance_m)
                else None
            ),
            "learned_policy_coupled": learned_safety_enabled,
            "learned_360_safety_coupled": learned_safety_enabled,
            "central_drop_safety_routing": {
                "learned_crown_scan_excludes_navigation_barrier": (
                    task == PHASE3N_MEASURED_NAV2_TASK
                ),
                "occupancy_map_keeps_navigation_barrier": True,
                "physics_collider_kept": True,
                "mapped_full_footprint_guard_required": (
                    task == PHASE3N_MEASURED_NAV2_TASK
                ),
            },
            "measured_nav2_termination_envelope": {
                "lidar_collision_margin_m": float(cfg.lidar_collision_margin_m),
                "nav2_footprint_padding_m": 0.03 if mapped_guard is not None else None,
                "physical_geometry_changed": False,
                "physical_safety_credit": False,
            },
            "frozen_phase3m_local_navigation_coupled": False,
            "physical_release": False,
            "claim_boundary": (
                "With a Phase 3N checkpoint this run proves live Nav2-to-learned-safety "
                "arbitration in the Isaac scene. When enabled, the mapped-site guard adds "
                "simulation-only doorway alignment/speed and central-drop constraints before "
                "the learned brake actor. It does not couple the frozen "
                "Phase 3M local navigator, replace the formal dynamic-obstacle evaluation, "
                "prove stopping distance or sim-to-real performance, or authorize physical "
                "deployment. Without a checkpoint it proves Isaac physics/ROS exchange only."
            ),
        }
        report["passed"] = (
            steps_completed > 0
            and all(count > 0 for count in bridge.messages.values())
            and not reset_detected
            and (
                task != PHASE3N_MEASURED_NAV2_TASK or mapped_guard is not None
            )
            and (
                not learned_safety_enabled
                or report["learned_safety"]["checkpoint_is_accepted_phase3n"]
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"AISHA_NAV2_BRIDGE passed={report['passed']} steps={steps_completed} "
            f"report={output}"
        )
        bridge.close()
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
