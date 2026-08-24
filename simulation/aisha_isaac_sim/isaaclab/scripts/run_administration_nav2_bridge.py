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
import time
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
parser.add_argument(
    "--fallback-learned-safety-checkpoint",
    type=Path,
    help=(
        "optional accepted Phase 3N checkpoint used outside declared Phase 6 "
        "high-speed route segments"
    ),
)
parser.add_argument("--output-report", type=Path)
parser.add_argument(
    "--phase3n-safety-checkpoint",
    "--learned-safety-checkpoint",
    dest="learned_safety_checkpoint",
    type=Path,
    help=(
        "load a compatible Phase 3N/Phase 6 checkpoint and arbitrate every "
        "Nav2 command through it"
    ),
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
PHASE6_MEASURED_NAV2_TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase6-"
    "HighSpeed80-DynamicSafety-Direct-v0"
)
PHASE7_MEASURED_NAV2_DYNAMIC_TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7-"
    "DynamicCrossing-Safety-Direct-v0"
)
PHASE7B_MEASURED_NAV2_BLOCKED_ROUTE_TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7B-"
    "BlockedRoute-Replanning-Safety-Direct-v0"
)
PHASE7D_ADMINISTRATION_NATIVE_COSTMAP_TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7D-"
    "NativeCostmap-SafeWait-Safety-Direct-v0"
)
PHASE7C_NATIVE_COSTMAP_DETOUR_TASK = (
    "Isaac-AISHA-Phase7C-NativeCostmap-Detour-Safety-Direct-v0"
)
PHASE3N_COMPATIBLE_TASKS = {
    PHASE3N_PRESENTATION_TASK,
    PHASE3N_MEASURED_NAV2_TASK,
    PHASE6_MEASURED_NAV2_TASK,
    PHASE7_MEASURED_NAV2_DYNAMIC_TASK,
    PHASE7B_MEASURED_NAV2_BLOCKED_ROUTE_TASK,
    PHASE7D_ADMINISTRATION_NATIVE_COSTMAP_TASK,
    PHASE7C_NATIVE_COSTMAP_DETOUR_TASK,
}
ACCEPTED_PHASE3N_CHECKPOINT = "aisha_phase3n_dynamic_safety_model_50.pt"
ACCEPTED_PHASE3N_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)
ACCEPTED_PHASE6_CHECKPOINT = "aisha_phase6_high_speed_080_model_223.pt"
ACCEPTED_PHASE6_SHA256 = (
    "e49767507925548aa0086c38e764c43037f25734943b2c5712cb58eecb0b6318"
)
PHASE6_NAV2_TASKS = {
    PHASE6_MEASURED_NAV2_TASK,
    PHASE7_MEASURED_NAV2_DYNAMIC_TASK,
    PHASE7B_MEASURED_NAV2_BLOCKED_ROUTE_TASK,
    PHASE7D_ADMINISTRATION_NATIVE_COSTMAP_TASK,
}
MEASURED_NAV2_TASKS = {PHASE3N_MEASURED_NAV2_TASK, *PHASE6_NAV2_TASKS}


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
        maximum_forward_mps: float = 0.30,
        non_high_speed_navigation_maximum_mps: float = 0.30,
        publish_ground_truth_map_to_odom: bool = False,
    ):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan, PointCloud2
        from std_msgs.msg import Bool, UInt32
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
        self.phase7b_front_points_publisher = self.node.create_publisher(
            PointCloud2, "/aisha/phase7b/front_points", sensor_qos
        )
        self.blockage_state_publisher = self.node.create_publisher(
            Bool, "/aisha/blocked_route_active", 10
        )
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self.node)
        self.subscription = self.node.create_subscription(
            Twist, "/cmd_vel", self._command_callback, 10
        )
        self.mission_complete_subscription = self.node.create_subscription(
            Bool, "/aisha/mission_complete", self._mission_complete_callback, 10
        )
        self.route_segment_subscription = self.node.create_subscription(
            UInt32,
            "/aisha/route_segment_id",
            self._route_segment_callback,
            10,
        )
        self.blockage_release_subscription = self.node.create_subscription(
            Bool,
            "/aisha/clear_blocked_route",
            self._blockage_release_callback,
            10,
        )
        self.blockage_activation_subscription = self.node.create_subscription(
            Bool,
            "/aisha/activate_blocked_route",
            self._blockage_activation_callback,
            10,
        )
        self.command_linear_mps = 0.0
        self.command_angular_rad_s = 0.0
        self.command_received_at_s = -math.inf
        self.simulation_time_s = 0.0
        self.stop_latched = False
        self.front_stop_trigger_m = 0.60
        self.front_stop_release_m = 0.75
        self.ring_stop_release_clearance_m: float | None = None
        self.front_stop_half_angle_deg = 10.0
        self.command_watchdog_s = 0.30
        self.crown_scan_period_s = 0.10
        self.front_scan_period_s = 0.05
        self.publish_phase7b_front_points = False
        self.publish_blockage_state = False
        self.maximum_forward_mps = maximum_forward_mps
        self.non_high_speed_navigation_maximum_mps = (
            non_high_speed_navigation_maximum_mps
        )
        self.publish_ground_truth_map_to_odom = publish_ground_truth_map_to_odom
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
        self.current_front_range_m = math.inf
        self.current_central_front_range_m = math.inf
        self.current_ring_clearance_m = math.inf
        self.current_route_segment_id = 0
        self.route_segment_messages = 0
        self.invalid_route_segment_messages = 0
        self.blockage_active = False
        self.blockage_release_requests = 0
        self.blockage_activation_requests = 0
        self.maximum_requested_linear_mps_by_segment: dict[int, float] = {}
        self.maximum_guarded_linear_mps_by_segment: dict[int, float] = {}
        self.maximum_observed_linear_mps_by_segment: dict[int, float] = {}
        self.publish_static_transforms(sensor_positions)

    def _mission_complete_callback(self, message) -> None:
        if bool(message.data):
            self.exit_requested = True

    def _route_segment_callback(self, message) -> None:
        segment_id = int(message.data)
        segment_count = int(self.raw_env._segment_goals.shape[0])
        if not 0 <= segment_id < segment_count:
            self.invalid_route_segment_messages += 1
            return
        self.current_route_segment_id = segment_id
        self.route_segment_messages += 1
        self.raw_env._segment_ids[0] = segment_id
        self.raw_env._goal_w[0] = (
            self.raw_env.scene.env_origins[0, :2]
            + self.raw_env._segment_goals[segment_id]
        )
        if hasattr(self.raw_env, "_apply_segment_speed_envelope"):
            self.raw_env._apply_segment_speed_envelope()

    def _blockage_release_callback(self, message) -> None:
        if not bool(message.data):
            return
        self.blockage_release_requests += 1
        if hasattr(self.raw_env, "release_temporary_blockage"):
            self.raw_env.release_temporary_blockage()

    def _blockage_activation_callback(self, message) -> None:
        if not bool(message.data):
            return
        self.blockage_activation_requests += 1
        if hasattr(self.raw_env, "activate_temporary_blockage"):
            self.raw_env.activate_temporary_blockage()

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

    def route_navigation_maximum_mps(self) -> float:
        high_speed_ids = tuple(int(value) for value in self.raw_env.cfg.high_speed_segment_ids)
        if self.current_route_segment_id in high_speed_ids:
            return self.maximum_forward_mps
        return min(
            self.maximum_forward_mps,
            self.non_high_speed_navigation_maximum_mps,
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
        self.current_front_range_m = front_min
        self.current_central_front_range_m = central_front_min
        self.current_ring_clearance_m = ring_clearance
        if self.stop_latched:
            central_blocked = central_front_min < self.front_stop_release_m
            ring_blocked = (
                self.ring_stop_release_clearance_m is not None
                and ring_clearance < self.ring_stop_release_clearance_m
            )
            self.stop_latched = central_blocked or ring_blocked
        elif central_front_min <= self.front_stop_trigger_m:
            self.stop_latched = True
            self.obstacle_stops += 1

        requested_linear = min(
            self.command_linear_mps, self.route_navigation_maximum_mps()
        )
        previous_requested = self.maximum_requested_linear_mps_by_segment.get(
            self.current_route_segment_id, 0.0
        )
        self.maximum_requested_linear_mps_by_segment[self.current_route_segment_id] = max(
            previous_requested, requested_linear
        )
        linear = requested_linear
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

        previous_guarded = self.maximum_guarded_linear_mps_by_segment.get(
            self.current_route_segment_id, 0.0
        )
        self.maximum_guarded_linear_mps_by_segment[self.current_route_segment_id] = max(
            previous_guarded, linear
        )

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
        # External Nav2 commands are physical m/s. Convert them against the
        # route-scoped final wheel mapping so a 0.30 m/s request remains 0.30
        # on a Phase 6 leg rather than being unintentionally expanded. The
        # learned actor still sees and brakes the same normalized command.
        if hasattr(self.raw_env, "_apply_segment_speed_envelope"):
            self.raw_env._apply_segment_speed_envelope()
        if hasattr(self.raw_env, "_route_scoped_maximum_speed"):
            route_mapping_maximum = float(
                self.raw_env._route_scoped_maximum_speed()[0].item()
            )
        else:
            route_mapping_maximum = float(
                self.raw_env.cfg.linear_velocity_range_mps[1]
            )
        action[0, 0] = 2.0 * linear / route_mapping_maximum - 1.0
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
        # Isaac's ray caster returns one simultaneous physics snapshot, not a
        # mechanically swept sequence.  Advertising a per-beam time offset
        # makes laser_geometry request transforms in the future and can cause
        # Nav2's costmap to discard an otherwise valid obstacle scan.
        message.time_increment = 0.0
        message.scan_time = scan_time_s
        message.range_min = 0.12
        message.range_max = 10.0
        message.ranges = values
        publisher.publish(message)
        self.messages[message_key] += 1

    def _publish_front_points(
        self,
        ranges: torch.Tensor,
    ) -> None:
        """Publish map-registered points generated only from live LiDAR hits."""
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header

        valid = torch.isfinite(ranges) & (ranges >= 0.12) & (ranges < 9.999)
        sensor = self.raw_env.scene.sensors["front_lidar"]
        origin = self.raw_env.scene.env_origins[0]
        # MultiMeshRayCaster already supplies the actual world-space hit for
        # every ray. Register those hits into the map/odom-coincident frame at
        # acquisition time, exactly as a point-cloud registration node would.
        # No blocker pose or scenario geometry is used here.
        points = sensor.data.ray_hits_w[0, valid] - origin.unsqueeze(0)
        header = Header()
        header.stamp = ros_stamp(self.simulation_time_s)
        header.frame_id = "map"
        message = point_cloud2.create_cloud_xyz32(
            header, points.detach().cpu().tolist()
        )
        self.phase7b_front_points_publisher.publish(message)
        self.messages["phase7b_front_points"] += 1

    def publish(self, step_index: int) -> None:
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from std_msgs.msg import Bool

        stamp = ros_stamp(self.simulation_time_s)
        clock = Clock()
        clock.clock = stamp
        self.clock_publisher.publish(clock)
        self.messages["clock"] += 1

        if self.publish_blockage_state:
            blockage_state = Bool()
            blockage_state.data = self.blockage_active
            self.blockage_state_publisher.publish(blockage_state)
            self.messages["blocked_route_active"] += 1

        robot = self.raw_env._robot
        origin = self.raw_env.scene.env_origins[0]
        position = robot.data.root_pos_w[0] - origin
        quaternion = robot.data.root_quat_w[0]
        linear = robot.data.root_lin_vel_b[0]
        angular = robot.data.root_ang_vel_b[0]
        observed_linear_mps = abs(float(linear[0].item()))
        previous_observed = self.maximum_observed_linear_mps_by_segment.get(
            self.current_route_segment_id, 0.0
        )
        self.maximum_observed_linear_mps_by_segment[self.current_route_segment_id] = max(
            previous_observed, observed_linear_mps
        )

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
        transforms = [transform]
        if self.publish_ground_truth_map_to_odom:
            # The measured Phase 6 gate evaluates navigation and safety, not
            # scan-matching. At 0.8 m/s the presentation scene's assumed
            # furniture can make AMCL diverge from the exact Isaac pose. Keep
            # map and odom coincident so Nav2 consumes deterministic simulator
            # ground truth, with the limitation explicitly reported.
            map_to_odom = TransformStamped()
            map_to_odom.header.stamp = stamp
            map_to_odom.header.frame_id = "map"
            map_to_odom.child_frame_id = "odom"
            map_to_odom.transform.rotation.w = 1.0
            transforms.append(map_to_odom)
        self.tf_broadcaster.sendTransform(transforms)
        self.messages["tf"] += len(transforms)

        if step_index % max(
            1, round(self.crown_scan_period_s / self.control_period_s)
        ) == 0:
            self._publish_scan(
                self.crown_publisher,
                "lidar_link",
                self.raw_env._lidar_ranges()[0],
                -math.pi,
                math.pi - math.radians(10.0),
                self.crown_scan_period_s,
                "scan",
            )
        if step_index % max(
            1, round(self.front_scan_period_s / self.control_period_s)
        ) == 0:
            front_ranges = self.front_ranges()[0]
            self._publish_scan(
                self.front_publisher,
                "front_lidar_link",
                front_ranges,
                -math.radians(60.0),
                math.radians(60.0),
                self.front_scan_period_s,
                "front_scan",
            )
            if self.publish_phase7b_front_points:
                self._publish_front_points(front_ranges)

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
    learned_safety_enabled = args.learned_safety_checkpoint is not None
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
    accepted_checkpoint_profile = None
    fallback_checkpoint = None
    fallback_checkpoint_sha256 = None
    if learned_safety_enabled:
        checkpoint = args.learned_safety_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)
        if task == parser.get_default("task"):
            task = PHASE3N_PRESENTATION_TASK
        if task not in PHASE3N_COMPATIBLE_TASKS:
            raise ValueError(
                "--learned-safety-checkpoint requires a deterministic compatible "
                f"learned-safety task: {sorted(PHASE3N_COMPATIBLE_TASKS)}"
            )
        if (
            checkpoint.name == ACCEPTED_PHASE3N_CHECKPOINT
            and checkpoint_sha256 == ACCEPTED_PHASE3N_SHA256
        ):
            accepted_checkpoint_profile = "phase3n_dynamic_safety"
        elif (
            checkpoint.name == ACCEPTED_PHASE6_CHECKPOINT
            and checkpoint_sha256 == ACCEPTED_PHASE6_SHA256
        ):
            accepted_checkpoint_profile = "phase6_high_speed_080"
        if (
            task in PHASE6_NAV2_TASKS
            or task == PHASE7C_NATIVE_COSTMAP_DETOUR_TASK
        ) and accepted_checkpoint_profile != "phase6_high_speed_080":
            raise ValueError(
                "Phase 6 measured Nav2 task requires the accepted Phase 6 checkpoint"
            )
    if args.fallback_learned_safety_checkpoint is not None:
        fallback_checkpoint = (
            args.fallback_learned_safety_checkpoint.expanduser().resolve()
        )
        if not fallback_checkpoint.is_file():
            raise FileNotFoundError(fallback_checkpoint)
        fallback_checkpoint_sha256 = sha256_file(fallback_checkpoint)
        if task not in PHASE6_NAV2_TASKS:
            raise ValueError(
                "fallback learned safety is supported only by the Phase 6 "
                "measured Nav2 task"
            )
        if not (
            fallback_checkpoint.name == ACCEPTED_PHASE3N_CHECKPOINT
            and fallback_checkpoint_sha256 == ACCEPTED_PHASE3N_SHA256
        ):
            raise ValueError(
                "Phase 6 measured fallback must be the accepted Phase 3N checkpoint"
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
            "dynamic_obstacle_activation_probability": (
                1.0
                if task
                in {
                    PHASE7_MEASURED_NAV2_DYNAMIC_TASK,
                    PHASE7B_MEASURED_NAV2_BLOCKED_ROUTE_TASK,
                    PHASE7D_ADMINISTRATION_NATIVE_COSTMAP_TASK,
                }
                else 0.0
            ),
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
    fallback_policy = None
    fallback_policy_network = None
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
        if fallback_checkpoint is not None:
            fallback_runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=device
            )
            fallback_runner.load(str(fallback_checkpoint))
            fallback_policy = fallback_runner.get_inference_policy(
                device=raw_env.device
            )
            try:
                fallback_policy_network = fallback_runner.alg.policy
            except AttributeError:
                fallback_policy_network = fallback_runner.alg.actor_critic
        observations, _ = env.reset()

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
    phase6_high_speed_replay = (
        task in PHASE6_NAV2_TASKS
        or task == PHASE7C_NATIVE_COSTMAP_DETOUR_TASK
    )
    dynamic_crossing_replay = task == PHASE7_MEASURED_NAV2_DYNAMIC_TASK
    blocked_route_replay = task in {
        PHASE7B_MEASURED_NAV2_BLOCKED_ROUTE_TASK,
        PHASE7D_ADMINISTRATION_NATIVE_COSTMAP_TASK,
    }
    administration_native_costmap_replay = (
        task == PHASE7D_ADMINISTRATION_NATIVE_COSTMAP_TASK
    )
    native_detour_replay = task == PHASE7C_NATIVE_COSTMAP_DETOUR_TASK
    blockage_replay = blocked_route_replay or native_detour_replay
    if phase6_high_speed_replay and mapped_guard is not None:
        # The furnished wheel contacts need the same proven 0.42 rad/s
        # breakaway command used by the office pivots. Translation remains
        # held at zero while this open-approach alignment is active.
        mapped_guard.breakaway_angular_rad_s = 0.42
    bridge = AishaSimulationBridge(
        raw_env,
        control_period_s,
        sensor_positions,
        mapped_guard=mapped_guard,
        maximum_forward_mps=0.80 if phase6_high_speed_replay else 0.30,
        non_high_speed_navigation_maximum_mps=0.30,
        publish_ground_truth_map_to_odom=phase6_high_speed_replay,
    )
    if dynamic_crossing_replay:
        # Begin the presentation-only protective stop earlier than the static
        # obstacle default so the 0.8 m/s approach remains outside the frozen
        # rectangular LiDAR envelope while the pedestrian completes crossing.
        # This is a simulation gate and provides no physical stopping-distance
        # credit.
        bridge.front_stop_trigger_m = 0.90
        bridge.front_stop_release_m = 1.10
        bridge.ring_stop_release_clearance_m = 0.35
    if blockage_replay:
        # Match temporary-blockage sensing to a conservative 2 Hz processing
        # rate so the executor's TF filter consumes current samples rather than
        # accumulating a stale high-rate queue. This is required by the large
        # administration map and retained for the compact reproducibility gate.
        bridge.crown_scan_period_s = 0.50
        bridge.front_scan_period_s = 0.50
        bridge.publish_blockage_state = True
        bridge.messages["blocked_route_active"] = 0
    if blocked_route_replay and not administration_native_costmap_replay:
        bridge.publish_phase7b_front_points = True
        bridge.messages["phase7b_front_points"] = 0
    steps_completed = 0
    reset_detected = False
    safety_authority_steps = 0
    safety_brake_steps = 0
    safety_brake_sum = 0.0
    maximum_safety_brake_fraction = 0.0
    minimum_ring_clearance_m = math.inf
    primary_policy_steps = 0
    fallback_policy_steps = 0
    dynamic_triggered = False
    dynamic_crossing_completed = False
    dynamic_trigger_step = None
    dynamic_completion_step = None
    dynamic_crossing_steps = 0
    dynamic_policy_handoff_steps = 0
    dynamic_stop_latched_steps = 0
    dynamic_authority_steps = 0
    dynamic_brake_steps = 0
    dynamic_minimum_centre_distance_m = math.inf
    dynamic_minimum_central_front_range_m = math.inf
    dynamic_minimum_ring_clearance_m = math.inf
    dynamic_minimum_forward_speed_mps = math.inf
    dynamic_maximum_pre_trigger_speed_mps = 0.0
    dynamic_maximum_recovery_speed_mps = 0.0
    dynamic_stop_observed = False
    dynamic_recovery_observed = False
    dynamic_trace = []
    blockage_triggered = False
    blockage_cleared = False
    blockage_trigger_step = None
    blockage_clear_step = None
    blockage_active_steps = 0
    blockage_maximum_robot_speed_mps = 0.0
    blockage_minimum_central_front_range_m = math.inf
    blockage_start_robot_xy_m = None
    blockage_clear_robot_xy_m = None
    blockage_position_xy_m = None
    last_pre_step_position_xy_m = None
    last_pre_step_minimum_ring_clearance_m = None
    phase7b_pacing_start_wall_s = time.monotonic()
    try:
        if not learned_safety_enabled:
            env.reset()
        while (
            simulation_app.is_running()
            and not bridge.exit_requested
            and (args.max_steps <= 0 or steps_completed < args.max_steps)
        ):
            bridge.spin()
            authority = False
            brake_fraction = 0.0
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
                    primary_safety_action = policy(observations)
                    if fallback_policy is not None:
                        fallback_safety_action = fallback_policy(observations)
                        use_primary = bridge.current_route_segment_id in tuple(
                            int(value) for value in raw_env.cfg.high_speed_segment_ids
                        )
                        dynamic_policy_handoff = (
                            dynamic_crossing_replay
                            and bridge.current_route_segment_id
                            == int(raw_env.cfg.showcase_segment_id)
                            and bridge.current_central_front_range_m <= 1.50
                        )
                        if dynamic_policy_handoff:
                            use_primary = False
                            dynamic_policy_handoff_steps += 1
                        safety_action = (
                            primary_safety_action
                            if use_primary
                            else fallback_safety_action
                        )
                        primary_policy_steps += int(use_primary)
                        fallback_policy_steps += int(not use_primary)
                    else:
                        safety_action = primary_safety_action
                        primary_policy_steps += 1
                    observations, _, dones, _ = env.step(safety_action)
                    policy_network.reset(dones)
                    if fallback_policy_network is not None:
                        fallback_policy_network.reset(dones)
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
            if dynamic_crossing_replay and hasattr(raw_env, "showcase_state"):
                state = raw_env.showcase_state()
                triggered = bool(state["triggered"][0].item())
                progress = float(state["crossing_progress"][0].item())
                robot_xy = raw_env._local_xy()[0]
                person_xy = state["person_position_xy_m"][0]
                centre_distance = float(
                    torch.linalg.norm(robot_xy - person_xy).item()
                )
                forward_speed = abs(
                    float(raw_env._robot.data.root_lin_vel_b[0, 0].item())
                )
                crossing_segment_id = int(raw_env.cfg.showcase_segment_id)
                on_crossing_segment = (
                    bridge.current_route_segment_id == crossing_segment_id
                )
                if on_crossing_segment and not triggered:
                    dynamic_maximum_pre_trigger_speed_mps = max(
                        dynamic_maximum_pre_trigger_speed_mps, forward_speed
                    )
                if triggered and on_crossing_segment:
                    if not dynamic_triggered:
                        dynamic_triggered = True
                        dynamic_trigger_step = steps_completed
                    dynamic_crossing_steps += 1
                    dynamic_minimum_centre_distance_m = min(
                        dynamic_minimum_centre_distance_m, centre_distance
                    )
                    dynamic_minimum_central_front_range_m = min(
                        dynamic_minimum_central_front_range_m,
                        bridge.current_central_front_range_m,
                    )
                    dynamic_minimum_ring_clearance_m = min(
                        dynamic_minimum_ring_clearance_m,
                        bridge.current_ring_clearance_m,
                    )
                    dynamic_minimum_forward_speed_mps = min(
                        dynamic_minimum_forward_speed_mps, forward_speed
                    )
                    dynamic_stop_latched_steps += int(bridge.stop_latched)
                    dynamic_authority_steps += int(authority)
                    dynamic_brake_steps += int(brake_fraction > 1.0e-6)
                    dynamic_stop_observed |= forward_speed <= 0.05
                    if progress >= 0.999 and not dynamic_crossing_completed:
                        dynamic_crossing_completed = True
                        dynamic_completion_step = steps_completed
                    if dynamic_crossing_completed:
                        dynamic_maximum_recovery_speed_mps = max(
                            dynamic_maximum_recovery_speed_mps, forward_speed
                        )
                        dynamic_recovery_observed |= forward_speed >= 0.30
                    if (
                        steps_completed % 12 == 0
                        or steps_completed == dynamic_completion_step
                    ):
                        dynamic_trace.append(
                            {
                                "step": steps_completed,
                                "segment_id": bridge.current_route_segment_id,
                                "progress": round(progress, 5),
                                "robot_xy_m": [
                                    round(float(value), 5)
                                    for value in robot_xy.detach().cpu().tolist()
                                ],
                                "pedestrian_xy_m": [
                                    round(float(value), 5)
                                    for value in person_xy.detach().cpu().tolist()
                                ],
                                "centre_distance_m": round(centre_distance, 5),
                                "forward_speed_mps": round(forward_speed, 5),
                                "central_front_range_m": round(
                                    bridge.current_central_front_range_m, 5
                                ),
                                "ring_clearance_m": round(
                                    bridge.current_ring_clearance_m, 5
                                ),
                                "front_stop_latched": bridge.stop_latched,
                                "learned_authority": authority,
                                "learned_brake_fraction": round(
                                    brake_fraction, 6
                                ),
                            }
                        )
            if blockage_replay and hasattr(raw_env, "blockage_state"):
                state = raw_env.blockage_state()
                triggered = bool(state["triggered"][0].item())
                active = bool(state["active"][0].item())
                cleared = bool(state["cleared"][0].item())
                bridge.blockage_active = active
                robot_xy = raw_env._local_xy()[0]
                blocker_xy = state["blocker_position_xy_m"][0]
                blockage_position_xy_m = [
                    round(float(value), 5)
                    for value in blocker_xy.detach().cpu().tolist()
                ]
                if triggered and not blockage_triggered:
                    blockage_triggered = True
                    blockage_trigger_step = steps_completed
                    blockage_start_robot_xy_m = [
                        round(float(value), 6)
                        for value in robot_xy.detach().cpu().tolist()
                    ]
                if active:
                    blockage_active_steps += 1
                    speed = abs(
                        float(raw_env._robot.data.root_lin_vel_b[0, 0].item())
                    )
                    blockage_maximum_robot_speed_mps = max(
                        blockage_maximum_robot_speed_mps, speed
                    )
                    blockage_minimum_central_front_range_m = min(
                        blockage_minimum_central_front_range_m,
                        bridge.current_central_front_range_m,
                    )
                if cleared and not blockage_cleared:
                    blockage_cleared = True
                    blockage_clear_step = steps_completed
                    blockage_clear_robot_xy_m = [
                        round(float(value), 6)
                        for value in robot_xy.detach().cpu().tolist()
                    ]
            bridge.simulation_time_s += control_period_s
            bridge.publish(steps_completed)
            steps_completed += 1
            if blockage_replay:
                # Nav2 consumes the LiDAR through timestamped TF filters.  Keep
                # the Phase 7B bridge at or below wall-clock rate so a fast
                # workstation cannot advance /clock beyond the transform cache
                # before the costmap has processed the corresponding scan.
                pacing_deadline_s = (
                    phase7b_pacing_start_wall_s + bridge.simulation_time_s
                )
                pacing_delay_s = pacing_deadline_s - time.monotonic()
                if pacing_delay_s > 0.0:
                    time.sleep(pacing_delay_s)
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
                "subscribed": [
                    "/cmd_vel",
                    "/aisha/route_segment_id",
                ]
                + (
                    [
                        "/aisha/activate_blocked_route",
                        "/aisha/clear_blocked_route",
                    ]
                    if blockage_replay
                    else []
                ),
                "published": [
                    "/clock",
                    "/odom",
                    "/tf",
                    "/tf_static",
                    "/scan",
                    "/front_scan",
                ]
                + (
                    ["/aisha/blocked_route_active"]
                    + (
                        ["/aisha/phase7b/front_points"]
                        if blocked_route_replay
                        and not administration_native_costmap_replay
                        else []
                    )
                    if blockage_replay
                    else []
                ),
                "message_counts": bridge.messages,
            },
            "localization": {
                "nav2_global_pose_source": (
                    "isaac_ground_truth_odom_with_identity_map_to_odom"
                    if bridge.publish_ground_truth_map_to_odom
                    else "amcl_map_to_odom"
                ),
                "bridge_publishes_map_to_odom": (
                    bridge.publish_ground_truth_map_to_odom
                ),
                "amcl_tf_broadcast_required": (
                    not bridge.publish_ground_truth_map_to_odom
                ),
                "presentation_simulation_only": True,
                "physical_localization_credit": False,
            },
            "command_constraints": {
                "maximum_forward_mps": bridge.maximum_forward_mps,
                "non_high_speed_navigation_maximum_mps": (
                    bridge.non_high_speed_navigation_maximum_mps
                ),
                "high_speed_route_segment_ids": list(
                    raw_env.cfg.high_speed_segment_ids
                ),
                "route_scoped_phase3n_thresholds_enabled": bool(
                    raw_env.cfg.measured_route_scoped_phase3n_thresholds_enabled
                ),
                "reverse_allowed": False,
                "lateral_motion_allowed": False,
                "maximum_angular_rad_s": bridge.maximum_angular_rad_s,
                "minimum_in_place_angular_rad_s": bridge.minimum_in_place_angular_rad_s,
                "command_watchdog_s": bridge.command_watchdog_s,
                "front_stop_trigger_m": bridge.front_stop_trigger_m,
                "front_stop_release_m": bridge.front_stop_release_m,
                "ring_stop_release_clearance_m": (
                    bridge.ring_stop_release_clearance_m
                ),
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
                "route_segment_messages": bridge.route_segment_messages,
                "invalid_route_segment_messages": (
                    bridge.invalid_route_segment_messages
                ),
                "blockage_release_requests": bridge.blockage_release_requests,
                "blockage_activation_requests": (
                    bridge.blockage_activation_requests
                ),
            },
            "route_scoped_speed_evidence": {
                "maximum_requested_linear_mps_by_segment": {
                    str(key): value
                    for key, value in sorted(
                        bridge.maximum_requested_linear_mps_by_segment.items()
                    )
                },
                "maximum_guarded_linear_mps_by_segment": {
                    str(key): value
                    for key, value in sorted(
                        bridge.maximum_guarded_linear_mps_by_segment.items()
                    )
                },
                "maximum_observed_linear_mps_by_segment": {
                    str(key): value
                    for key, value in sorted(
                        bridge.maximum_observed_linear_mps_by_segment.items()
                    )
                },
                "mission_segment_source": "/aisha/route_segment_id",
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
                    accepted_checkpoint_profile == "phase3n_dynamic_safety"
                ),
                "checkpoint_is_accepted_phase6": bool(
                    accepted_checkpoint_profile == "phase6_high_speed_080"
                ),
                "accepted_checkpoint_profile": accepted_checkpoint_profile,
                "fallback_checkpoint": (
                    str(fallback_checkpoint)
                    if fallback_checkpoint is not None
                    else None
                ),
                "fallback_checkpoint_sha256": fallback_checkpoint_sha256,
                "fallback_checkpoint_is_accepted_phase3n": bool(
                    fallback_checkpoint is not None
                    and fallback_checkpoint_sha256 == ACCEPTED_PHASE3N_SHA256
                ),
                "policy_selection": (
                    (
                        "phase6_on_declared_high_speed_segments_phase3n_elsewhere_"
                        "plus_front_scan_scoped_phase3n_dynamic_handoff"
                    )
                    if dynamic_crossing_replay
                    else "phase6_on_declared_high_speed_segments_phase3n_elsewhere"
                    if fallback_checkpoint is not None
                    else "single_checkpoint"
                ),
                "dynamic_handoff_sensor": (
                    "central_front_scan_at_or_below_1.50_m_on_crossing_segment"
                    if dynamic_crossing_replay
                    else None
                ),
                "primary_policy_steps": primary_policy_steps,
                "fallback_policy_steps": fallback_policy_steps,
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
                "deterministic_static_integration_gate": (
                    learned_safety_enabled
                    and not dynamic_crossing_replay
                    and not blockage_replay
                ),
            },
            "dynamic_obstacle": {
                "enabled": dynamic_crossing_replay,
                "scenario": (
                    "single_sensed_pedestrian_crossing_on_high_speed_segment_1"
                    if dynamic_crossing_replay
                    else None
                ),
                "stylized_proxy_not_human_model": dynamic_crossing_replay,
                "pedestrian_state_exposed_to_policy": False,
                "crossing_segment_id": (
                    int(cfg.showcase_segment_id)
                    if dynamic_crossing_replay
                    else None
                ),
                "triggered": dynamic_triggered,
                "crossing_completed": dynamic_crossing_completed,
                "trigger_step": dynamic_trigger_step,
                "completion_step": dynamic_completion_step,
                "crossing_steps": dynamic_crossing_steps,
                "sensor_scoped_phase3n_handoff_steps": (
                    dynamic_policy_handoff_steps
                ),
                "front_stop_latched_steps": dynamic_stop_latched_steps,
                "learned_authority_steps_during_encounter": dynamic_authority_steps,
                "learned_brake_steps_during_encounter": dynamic_brake_steps,
                "minimum_robot_pedestrian_centre_distance_m": (
                    round(dynamic_minimum_centre_distance_m, 5)
                    if math.isfinite(dynamic_minimum_centre_distance_m)
                    else None
                ),
                "minimum_central_front_range_m": (
                    round(dynamic_minimum_central_front_range_m, 5)
                    if math.isfinite(dynamic_minimum_central_front_range_m)
                    else None
                ),
                "minimum_360_clearance_m": (
                    round(dynamic_minimum_ring_clearance_m, 5)
                    if math.isfinite(dynamic_minimum_ring_clearance_m)
                    else None
                ),
                "maximum_pre_trigger_forward_speed_mps": round(
                    dynamic_maximum_pre_trigger_speed_mps, 5
                ),
                "minimum_encounter_forward_speed_mps": (
                    round(dynamic_minimum_forward_speed_mps, 5)
                    if math.isfinite(dynamic_minimum_forward_speed_mps)
                    else None
                ),
                "maximum_post_crossing_recovery_speed_mps": round(
                    dynamic_maximum_recovery_speed_mps, 5
                ),
                "controlled_stop_observed": dynamic_stop_observed,
                "post_crossing_recovery_observed": dynamic_recovery_observed,
                "trace": dynamic_trace,
            },
            "blocked_route": {
                "enabled": blockage_replay,
                "scenario": (
                    "temporary_top_branch_blockage_with_bottom_branch_detour"
                    if native_detour_replay
                    else "temporary_full_width_single_path_hallway_blockage"
                    if blocked_route_replay
                    else None
                ),
                "planner_observation_source": (
                    "native_nav2_obstacle_layer_from_live_isaac_laserscan"
                    if native_detour_replay or administration_native_costmap_replay
                    else "registered_front_lidar_supervisory_path_validator"
                    if blocked_route_replay
                    else None
                ),
                "simulation_time_paced_to_wall_clock": blockage_replay,
                "sensor_publish_rate_hz": (
                    {
                        "crown_lidar": round(1.0 / bridge.crown_scan_period_s, 3),
                        "front_lidar": round(1.0 / bridge.front_scan_period_s, 3),
                    }
                    if blockage_replay
                    else None
                ),
                "registered_pointcloud_topic": (
                    "/aisha/phase7b/front_points"
                    if blocked_route_replay
                    and not administration_native_costmap_replay
                    else None
                ),
                "blocker_state_exposed_to_policy": False,
                "coordination_state_exposed_to_mission_only": blockage_replay,
                "route_topology": (
                    "two_route_loop_spatial_detour_available"
                    if native_detour_replay
                    else (
                        "map_connected_detour_available_but_outside_"
                        "mission_authorized_east_hallway"
                    )
                    if administration_native_costmap_replay
                    else "single_path_safe_wait_required_no_detour_available"
                    if blocked_route_replay
                    else None
                ),
                "blockage_segment_id": (
                    int(cfg.temporary_blockage_segment_id)
                    if blockage_replay
                    else None
                ),
                "blockage_size_xyz_m": (
                    list(cfg.temporary_blockage_size_xyz_m)
                    if blockage_replay
                    else None
                ),
                "blockage_position_xy_m": blockage_position_xy_m,
                "triggered": blockage_triggered,
                "active_steps": blockage_active_steps,
                "trigger_step": blockage_trigger_step,
                "cleared": blockage_cleared,
                "clear_step": blockage_clear_step,
                "release_requests": bridge.blockage_release_requests,
                "robot_xy_at_trigger_m": blockage_start_robot_xy_m,
                "robot_xy_at_clear_m": blockage_clear_robot_xy_m,
                "maximum_robot_speed_while_active_mps": round(
                    blockage_maximum_robot_speed_mps, 6
                ),
                "minimum_central_front_range_while_active_m": (
                    round(blockage_minimum_central_front_range_m, 5)
                    if math.isfinite(blockage_minimum_central_front_range_m)
                    else None
                ),
                "physical_safety_credit": False,
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
                    task in MEASURED_NAV2_TASKS
                ),
                "occupancy_map_keeps_navigation_barrier": True,
                "physics_collider_kept": True,
                "mapped_full_footprint_guard_required": (
                    task in MEASURED_NAV2_TASKS
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
                task not in MEASURED_NAV2_TASKS or mapped_guard is not None
            )
            and (
                not learned_safety_enabled
                or accepted_checkpoint_profile is not None
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
