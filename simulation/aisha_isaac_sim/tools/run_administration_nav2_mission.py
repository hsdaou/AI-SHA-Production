#!/usr/bin/env python3
"""Plan and execute the administration waypoint chain through Nav2 actions."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.action import ActionClient
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, UInt32


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PHASE6_CHECKPOINT = (
    PACKAGE_ROOT
    / "isaaclab/checkpoints/aisha_phase6_high_speed_080_model_223.pt"
)
OFFICE_STAGING_OFFSETS_M = {
    # DWB hands a position-only waypoint back near the edge of its 0.40 m
    # tolerance.  These interior staging offsets make that accepted pose land
    # on the disclosed presentation stop, with room left for the departure
    # pivot.  They are simulation-only and explicitly reported below.
    "vice_principal": (0.0, -0.25),
    "principal": (0.17, -0.19),
}
POST_VISIT_PIVOTS = {"vice_principal", "principal"}
PHASE6_PRE_DOOR_ALIGNMENT_WAYPOINTS = {
    "vice_principal_approach",
    "principal_approach",
}
APPROACH_DOOR_BY_WAYPOINT = {
    "vice_principal_approach": "vice_principal",
    "principal_approach": "principal",
}
PHASE6_CONTROL_STACKS = {
    "nav2_mapped_doorway_phase6_high_speed_safety",
    "nav2_mapped_doorway_phase7_dynamic_crossing_safety",
    "nav2_mapped_doorway_phase7b_blocked_route_replanning_safety",
    "nav2_mapped_doorway_phase7d_native_costmap_safe_wait_safety",
}
PHASE7B_CONTROL_STACK = (
    "nav2_mapped_doorway_phase7b_blocked_route_replanning_safety"
)
PHASE7D_CONTROL_STACK = (
    "nav2_mapped_doorway_phase7d_native_costmap_safe_wait_safety"
)


def yaw_quaternion(yaw_radians: float) -> tuple[float, float]:
    return math.sin(yaw_radians / 2.0), math.cos(yaw_radians / 2.0)


def path_length(path) -> float:
    return sum(
        math.hypot(
            second.pose.position.x - first.pose.position.x,
            second.pose.position.y - first.pose.position.y,
        )
        for first, second in zip(path.poses, path.poses[1:])
    )


class MissionNode:
    def __init__(self) -> None:
        self.node = rclpy.create_node(
            "aisha_administration_nav2_mission",
            parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)],
        )
        self.compute_client = ActionClient(
            self.node, ComputePathToPose, "/compute_path_to_pose"
        )
        self.follow_client = ActionClient(self.node, FollowPath, "/follow_path")
        self.planner_state_client = self.node.create_client(
            GetState, "/planner_server/get_state"
        )
        self.controller_state_client = self.node.create_client(
            GetState, "/controller_server/get_state"
        )
        self.global_costmap_clear_client = self.node.create_client(
            ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap"
        )
        self.initial_pose_publisher = self.node.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.stop_publisher = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.completion_publisher = self.node.create_publisher(
            Bool, "/aisha/mission_complete", 10
        )
        self.route_segment_publisher = self.node.create_publisher(
            UInt32, "/aisha/route_segment_id", 10
        )
        self.blockage_release_publisher = self.node.create_publisher(
            Bool, "/aisha/clear_blocked_route", 10
        )
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.node.create_subscription(OccupancyGrid, "/map", self._map, map_qos)
        self.node.create_subscription(Odometry, "/odom", self._odom, 10)
        self.node.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_pose, 10
        )
        self.node.create_subscription(
            Bool,
            "/aisha/blocked_route_active",
            self._blocked_route_state,
            10,
        )
        self.node.create_subscription(
            Costmap,
            "/global_costmap/costmap_raw",
            self._global_costmap,
            10,
        )
        self.node.create_subscription(
            PointCloud2,
            "/aisha/phase7b/front_points",
            self._registered_front_points,
            qos_profile_sensor_data,
        )
        self.map_received = False
        self.amcl_pose_samples = 0
        self.odom_samples: list[tuple[float, float, float, float, float]] = []
        self.command_samples: list[tuple[float, float]] = []
        self.follow_feedback_count = 0
        self.current_route_segment_id = 0
        self.trace_enabled = False
        self.trace_control_mode = "nav2"
        self.pose_trace: list[dict[str, object]] = []
        self._trace_odom_count = 0
        self._trace_start_sim_s: float | None = None
        self.blockage_active = False
        self.blockage_ever_active = False
        self.blockage_state_messages = 0
        self.blockage_transitions: list[dict[str, object]] = []
        self.global_costmap: Costmap | None = None
        self.global_costmap_samples = 0
        self.registered_front_points: list[tuple[float, float, float]] = []
        self.registered_front_pointcloud_samples = 0
        self.node.create_subscription(Twist, "/cmd_vel", self._command, 10)

    def _map(self, _message: OccupancyGrid) -> None:
        self.map_received = True

    def _amcl_pose(self, _message: PoseWithCovarianceStamped) -> None:
        self.amcl_pose_samples += 1

    def _odom(self, message: Odometry) -> None:
        quaternion = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )
        self.odom_samples.append(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.twist.twist.linear.x,
                yaw,
                message.twist.twist.angular.z,
            )
        )
        if not self.trace_enabled:
            return
        simulation_time_s = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) / 1_000_000_000.0
        )
        if self._trace_start_sim_s is None:
            self._trace_start_sim_s = simulation_time_s
        self._trace_odom_count += 1
        if self._trace_odom_count % 3 != 0:
            return
        self.pose_trace.append(
            {
                "step": self._trace_odom_count,
                "elapsed_s": round(
                    simulation_time_s - self._trace_start_sim_s, 6
                ),
                "x_m": round(float(message.pose.pose.position.x), 6),
                "y_m": round(float(message.pose.pose.position.y), 6),
                "yaw_rad": round(yaw, 7),
                "segment_id": self.current_route_segment_id,
                "control_mode": self.trace_control_mode,
                "linear_velocity_mps": round(
                    float(message.twist.twist.linear.x), 6
                ),
                "yaw_rate_rad_s": round(
                    float(message.twist.twist.angular.z), 6
                ),
            }
        )

    def _command(self, message: Twist) -> None:
        self.command_samples.append((message.linear.x, message.angular.z))

    def _blocked_route_state(self, message: Bool) -> None:
        active = bool(message.data)
        self.blockage_state_messages += 1
        self.blockage_ever_active |= active
        if active == self.blockage_active and self.blockage_transitions:
            return
        self.blockage_active = active
        pose = self.odom_samples[-1] if self.odom_samples else None
        self.blockage_transitions.append(
            {
                "active": active,
                "odometry_sample": len(self.odom_samples),
                "robot_xy_m": list(pose[:2]) if pose is not None else None,
                "robot_linear_velocity_mps": pose[2] if pose is not None else None,
            }
        )

    def _global_costmap(self, message: Costmap) -> None:
        self.global_costmap = message
        self.global_costmap_samples += 1

    def _registered_front_points(self, message: PointCloud2) -> None:
        points = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        self.registered_front_points = [
            (float(point[0]), float(point[1]), float(point[2]))
            for point in points
        ]
        self.registered_front_pointcloud_samples += 1

    def costmap_blockage_profile(
        self, x_m: float = 10.62, y_min_m: float = -1.45, y_max_m: float = 1.45
    ) -> dict[str, object]:
        message = self.global_costmap
        if message is None:
            return {"available": False}
        metadata = message.metadata
        resolution = float(metadata.resolution)
        origin_x = float(metadata.origin.position.x)
        origin_y = float(metadata.origin.position.y)
        size_x = int(metadata.size_x)
        size_y = int(metadata.size_y)
        x_index = int(math.floor((x_m - origin_x) / resolution))
        samples = []
        y_m = y_min_m
        while y_m <= y_max_m + resolution * 0.5:
            y_index = int(math.floor((y_m - origin_y) / resolution))
            if 0 <= x_index < size_x and 0 <= y_index < size_y:
                value = int(message.data[y_index * size_x + x_index])
                samples.append((round(y_m, 3), value))
            y_m += 0.05
        return {
            "available": True,
            "topic_samples_received": self.global_costmap_samples,
            "sample_x_m": x_m,
            "sample_count": len(samples),
            "maximum_cost": max((value for _, value in samples), default=None),
            "lethal_or_inscribed_samples": sum(
                value >= 253 for _, value in samples
            ),
            "nonzero_samples": sum(value > 0 for _, value in samples),
            "samples_y_cost": [list(item) for item in samples],
        }

    @staticmethod
    def path_blockage_profile(path, centre_xy_m: tuple[float, float]) -> dict:
        if path is None:
            return {"available": False}
        centre_x, centre_y = centre_xy_m
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in path.poses
        ]
        nearest = min(
            points,
            key=lambda point: math.hypot(
                point[0] - centre_x, point[1] - centre_y
            ),
        )
        near_face = [
            point for point in points if abs(point[0] - centre_x) <= 0.30
        ]
        return {
            "available": True,
            "minimum_centre_distance_m": math.hypot(
                nearest[0] - centre_x, nearest[1] - centre_y
            ),
            "nearest_xy_m": list(nearest),
            "near_blockage_point_count": len(near_face),
            "near_blockage_y_range_m": (
                [
                    min(point[1] for point in near_face),
                    max(point[1] for point in near_face),
                ]
                if near_face
                else None
            ),
        }

    def registered_lidar_path_profile(
        self, path, required_clearance_m: float = 0.46
    ) -> dict[str, object]:
        """Validate a global path against the latest registered LiDAR returns."""
        if path is None:
            return {"available": False, "reason": "no_candidate_path"}
        if not self.registered_front_points:
            return {"available": False, "reason": "no_registered_lidar_points"}
        path_points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in path.poses
        ]
        minimum_distance = math.inf
        nearest_path = None
        nearest_obstacle = None
        violating_path_pose_count = 0
        for path_x, path_y in path_points:
            nearest_for_pose, obstacle = min(
                (
                    (
                        math.hypot(path_x - obstacle_x, path_y - obstacle_y),
                        (obstacle_x, obstacle_y),
                    )
                    for obstacle_x, obstacle_y, _ in self.registered_front_points
                ),
                key=lambda item: item[0],
            )
            if nearest_for_pose <= required_clearance_m:
                violating_path_pose_count += 1
            if nearest_for_pose < minimum_distance:
                minimum_distance = nearest_for_pose
                nearest_path = (path_x, path_y)
                nearest_obstacle = obstacle
        return {
            "available": True,
            "pointcloud_samples_received": self.registered_front_pointcloud_samples,
            "registered_obstacle_point_count": len(self.registered_front_points),
            "required_radial_clearance_m": required_clearance_m,
            "minimum_path_to_obstacle_distance_m": minimum_distance,
            "nearest_path_xy_m": list(nearest_path) if nearest_path else None,
            "nearest_obstacle_xy_m": (
                list(nearest_obstacle) if nearest_obstacle else None
            ),
            "violating_path_pose_count": violating_path_pose_count,
            "candidate_rejected": violating_path_pose_count > 0,
        }

    def wait_for_map_and_odometry(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.10)
            if self.map_received and self.odom_samples:
                return True
        return False

    def wait_for_action_servers(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.10)
            if not (
                self.compute_client.server_is_ready()
                and self.follow_client.server_is_ready()
                and self.planner_state_client.service_is_ready()
                and self.controller_state_client.service_is_ready()
            ):
                continue
            planner_future = self.planner_state_client.call_async(GetState.Request())
            controller_future = self.controller_state_client.call_async(GetState.Request())
            query_deadline = time.monotonic() + 1.0
            while (
                time.monotonic() < query_deadline
                and (not planner_future.done() or not controller_future.done())
            ):
                rclpy.spin_once(self.node, timeout_sec=0.05)
            if (
                planner_future.done()
                and controller_future.done()
                and planner_future.result().current_state.id == 3
                and controller_future.result().current_state.id == 3
            ):
                return True
        return False

    def pose(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z, pose.pose.orientation.w = yaw_quaternion(
            math.radians(yaw_deg)
        )
        return pose

    def publish_initial_pose(self, x: float, y: float, yaw_deg: float) -> bool:
        message = PoseWithCovarianceStamped()
        message.header.frame_id = "map"
        message.pose.pose = self.pose(x, y, yaw_deg).pose
        message.pose.covariance[0] = 0.01
        message.pose.covariance[7] = 0.01
        message.pose.covariance[35] = math.radians(2.0) ** 2
        discovery_deadline = time.monotonic() + 5.0
        while (
            self.initial_pose_publisher.get_subscription_count() == 0
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(self.node, timeout_sec=0.10)
        publish_deadline = time.monotonic() + 5.0
        while self.amcl_pose_samples == 0 and time.monotonic() < publish_deadline:
            message.header.stamp = self.node.get_clock().now().to_msg()
            self.initial_pose_publisher.publish(message)
            rclpy.spin_once(self.node, timeout_sec=0.10)
        return self.amcl_pose_samples > 0

    def wait_future(self, future, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.10)
        return future.done()

    def compute_path(self, waypoint: dict, timeout_s: float):
        goal = ComputePathToPose.Goal()
        goal.goal = self.pose(
            float(waypoint["x_m"]),
            float(waypoint["y_m"]),
            float(waypoint["yaw_deg"]),
        )
        goal.use_start = False
        goal.planner_id = "GridBased"
        send_future = self.compute_client.send_goal_async(goal)
        if not self.wait_future(send_future, timeout_s):
            return None, "compute_goal_response_timeout"
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return None, "compute_goal_rejected"
        result_future = handle.get_result_async()
        if not self.wait_future(result_future, timeout_s):
            handle.cancel_goal_async()
            return None, "compute_result_timeout"
        wrapped = result_future.result()
        if wrapped is None or wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            status = wrapped.status if wrapped is not None else None
            return None, f"compute_failed_status_{status}"
        return wrapped.result.path, "succeeded"

    def _feedback(self, _feedback) -> None:
        self.follow_feedback_count += 1

    def follow_path(
        self, path, timeout_s: float, goal_checker_id: str
    ) -> tuple[bool, str]:
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = "FollowPath"
        goal.goal_checker_id = goal_checker_id
        send_future = self.follow_client.send_goal_async(
            goal, feedback_callback=self._feedback
        )
        if not self.wait_future(send_future, timeout_s):
            return False, "follow_goal_response_timeout"
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False, "follow_goal_rejected"
        result_future = handle.get_result_async()
        if not self.wait_future(result_future, timeout_s):
            cancel = handle.cancel_goal_async()
            self.wait_future(cancel, 5.0)
            return False, "follow_result_timeout"
        wrapped = result_future.result()
        if wrapped is None or wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            status = wrapped.status if wrapped is not None else None
            return False, f"follow_failed_status_{status}"
        return True, "succeeded"

    def stop(self) -> None:
        if rclpy.ok():
            self.stop_publisher.publish(Twist())

    def signal_completion(self) -> None:
        if not rclpy.ok():
            return
        message = Bool()
        message.data = True
        for _ in range(5):
            self.completion_publisher.publish(message)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def publish_route_segment(self, segment_id: int) -> None:
        self.current_route_segment_id = segment_id
        message = UInt32()
        message.data = segment_id
        discovery_deadline = time.monotonic() + 2.0
        while (
            self.route_segment_publisher.get_subscription_count() == 0
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        for _ in range(3):
            self.route_segment_publisher.publish(message)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_for_blockage_state(self, active: bool, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.blockage_active == active and (
                active or self.blockage_ever_active
            ):
                return True
        return False

    def hold_stopped(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.stop()
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def request_blockage_clear(self) -> None:
        message = Bool()
        message.data = True
        discovery_deadline = time.monotonic() + 2.0
        while (
            self.blockage_release_publisher.get_subscription_count() == 0
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        for _ in range(5):
            self.blockage_release_publisher.publish(message)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def exercise_blocked_route_replan(
        self, waypoint: dict, planning_timeout_s: float
    ):
        """Reject a LiDAR-conflicting candidate, wait, then validate a fresh path."""
        started = time.monotonic()
        active_seen = self.wait_for_blockage_state(True, 15.0)
        result: dict[str, object] = {
            "scenario": "temporary_full_width_single_path_hallway_blockage",
            "blockage_active_seen": active_seen,
            "topology": "single_path_safe_wait_no_detour_available",
            "observation_source": "registered_front_lidar_supervisory_path_validator",
            "nav2_dynamic_costmap_marking_credit": False,
            "scenario_state_used_for_test_synchronization_only": True,
            "scenario_state_exposed_to_policy": False,
            "attempts": [],
        }
        if not active_seen:
            result["passed"] = False
            result["status"] = "blockage_active_not_observed"
            return None, result["status"], result

        # Accumulate several independent 2 Hz registered LiDAR snapshots before
        # allowing the route supervisor to evaluate Nav2's candidate path.
        self.hold_stopped(2.5)
        start_odom_index = len(self.odom_samples)
        start_pose = self.odom_samples[-1] if self.odom_samples else None
        blocked_candidate, blocked_candidate_status = self.compute_path(
            waypoint, planning_timeout_s
        )
        blocked_sensor_profile = self.registered_lidar_path_profile(
            blocked_candidate
        )
        sensor_rejected_candidate = bool(
            blocked_sensor_profile.get("candidate_rejected", False)
        )
        blocked_status = (
            "rejected_by_registered_lidar_path_validator"
            if sensor_rejected_candidate
            else blocked_candidate_status
        )
        blocked_path = None if sensor_rejected_candidate else blocked_candidate
        result["global_costmap_during_blockage"] = self.costmap_blockage_profile()
        result["attempts"].append(
            {
                "attempt": 1,
                "phase": "barrier_active",
                "planning_status": blocked_status,
                "nav2_candidate_planning_status": blocked_candidate_status,
                "candidate_path_pose_count": (
                    len(blocked_candidate.poses)
                    if blocked_candidate is not None
                    else 0
                ),
                "path_pose_count": (
                    len(blocked_path.poses) if blocked_path is not None else 0
                ),
                "path_length_m": (
                    round(path_length(blocked_candidate), 4)
                    if blocked_candidate is not None
                    else None
                ),
                "path_blockage_profile": self.path_blockage_profile(
                    blocked_candidate, (10.80, 0.0)
                ),
                "registered_lidar_validation": blocked_sensor_profile,
            }
        )
        blocked_plan_rejected = (
            blocked_candidate is None or sensor_rejected_candidate
        )
        result["blocked_plan_rejected"] = blocked_plan_rejected
        result["rejection_authority"] = (
            "registered_lidar_supervisory_path_validator"
            if sensor_rejected_candidate
            else "nav2_global_planner"
        )

        self.request_blockage_clear()
        cleared_seen = self.wait_for_blockage_state(False, 15.0)
        result["clearance_requested"] = True
        result["blockage_cleared_seen"] = cleared_seen
        if cleared_seen:
            self.hold_stopped(2.5)

        end_pose = self.odom_samples[-1] if self.odom_samples else None
        wait_samples = self.odom_samples[start_odom_index:]
        wait_displacement = (
            math.hypot(end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
            if start_pose is not None and end_pose is not None
            else None
        )
        result["safe_wait"] = {
            "measurement_start": "after_initial_costmap_and_stop_settle",
            "odometry_samples": len(wait_samples),
            "maximum_absolute_linear_velocity_mps": max(
                (abs(sample[2]) for sample in wait_samples), default=0.0
            ),
            "displacement_m": wait_displacement,
            "elapsed_wall_s": round(time.monotonic() - started, 3),
        }
        if not blocked_plan_rejected:
            result["passed"] = False
            result["status"] = "blocked_candidate_not_rejected_by_lidar_validator"
            return None, result["status"], result
        if not cleared_seen:
            result["passed"] = False
            result["status"] = "blockage_clear_not_observed"
            return None, result["status"], result
        fresh_candidate, fresh_candidate_status = self.compute_path(
            waypoint, planning_timeout_s
        )
        fresh_sensor_profile = self.registered_lidar_path_profile(fresh_candidate)
        fresh_sensor_rejected = bool(
            fresh_sensor_profile.get("candidate_rejected", False)
        )
        fresh_path = None if fresh_sensor_rejected else fresh_candidate
        fresh_status = (
            "rejected_by_registered_lidar_path_validator"
            if fresh_sensor_rejected
            else fresh_candidate_status
        )
        result["attempts"].append(
            {
                "attempt": 2,
                "phase": "barrier_cleared",
                "planning_status": fresh_status,
                "nav2_candidate_planning_status": fresh_candidate_status,
                "candidate_path_pose_count": (
                    len(fresh_candidate.poses) if fresh_candidate is not None else 0
                ),
                "path_pose_count": (
                    len(fresh_path.poses) if fresh_path is not None else 0
                ),
                "path_length_m": (
                    round(path_length(fresh_candidate), 4)
                    if fresh_candidate is not None
                    else None
                ),
                "registered_lidar_validation": fresh_sensor_profile,
            }
        )
        result["fresh_path_computed_after_clearance"] = fresh_path is not None
        result["planner_attempt_count"] = 2
        result["replan_mode"] = "fresh_compute_path_to_pose_after_safe_wait"
        result["passed"] = fresh_path is not None
        result["status"] = (
            "succeeded" if fresh_path is not None else fresh_status
        )
        return fresh_path, fresh_status, result

    def wait_for_native_costmap_profile(
        self,
        *,
        minimum_lethal_samples: int,
        maximum_lethal_samples: int | None,
        timeout_s: float,
    ) -> dict[str, object]:
        """Wait for the live global obstacle layer to reach a required state."""
        deadline = time.monotonic() + timeout_s
        last_profile: dict[str, object] = {"available": False}
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.10)
            last_profile = self.costmap_blockage_profile(
                x_m=10.62, y_min_m=-0.80, y_max_m=0.80
            )
            if last_profile.get("available") is not True:
                continue
            lethal = int(last_profile.get("lethal_or_inscribed_samples", 0))
            if lethal < minimum_lethal_samples:
                continue
            if maximum_lethal_samples is not None and lethal > maximum_lethal_samples:
                continue
            return last_profile
        return last_profile

    @staticmethod
    def east_hallway_route_profile(path) -> dict[str, object]:
        """Check a plan against the mission-authorized east hallway envelope."""
        if path is None:
            return {"available": False, "authorized": False}
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in path.poses
        ]
        violations = [
            point
            for point in points
            if not (3.80 <= point[0] <= 17.50 and -0.90 <= point[1] <= 0.90)
        ]
        return {
            "available": True,
            "authorization_source": "approved_mission_east_hallway_route_envelope",
            "authorised_x_range_m": [3.80, 17.50],
            "authorised_y_range_m": [-0.90, 0.90],
            "path_pose_count": len(points),
            "violating_pose_count": len(violations),
            "minimum_y_m": min((point[1] for point in points), default=None),
            "maximum_y_m": max((point[1] for point in points), default=None),
            "authorized": not violations,
        }

    def clear_native_global_costmap(self, timeout_s: float = 10.0) -> bool:
        """Clear stale obstacle marks after the test barrier is physically removed."""
        if not self.global_costmap_clear_client.wait_for_service(timeout_sec=timeout_s):
            return False
        future = self.global_costmap_clear_client.call_async(ClearEntireCostmap.Request())
        return self.wait_future(future, timeout_s) and future.result() is not None

    def exercise_native_costmap_blocked_route_replan(
        self,
        waypoint: dict,
        planning_timeout_s: float,
        clear_baseline: dict[str, object],
        baseline_path,
        baseline_path_status: str,
    ):
        """Mark the blockage, reject unapproved detours, wait, clear and replan."""
        started = time.monotonic()
        active_seen = self.wait_for_blockage_state(True, 15.0)
        result: dict[str, object] = {
            "scenario": "temporary_full_width_single_path_hallway_blockage",
            "blockage_active_seen": active_seen,
            "topology": (
                "mission_authorized_single_path_with_unapproved_map_detour_available"
            ),
            "observation_source": "native_nav2_obstacle_layer_from_live_isaac_laserscan",
            "nav2_dynamic_costmap_marking_credit": False,
            "scenario_state_used_for_test_synchronization_only": True,
            "scenario_state_exposed_to_policy": False,
            "costmap_before_activation": clear_baseline,
            "baseline": {
                "planning_status": baseline_path_status,
                "path_pose_count": (
                    len(baseline_path.poses) if baseline_path is not None else 0
                ),
                "path_length_m": (
                    round(path_length(baseline_path), 4)
                    if baseline_path is not None
                    else None
                ),
                "route_authorization": self.east_hallway_route_profile(baseline_path),
            },
            "attempts": [],
        }
        if not active_seen:
            result["passed"] = False
            result["status"] = "blockage_active_not_observed"
            return None, result["status"], result

        blocked_profile = self.wait_for_native_costmap_profile(
            minimum_lethal_samples=20,
            maximum_lethal_samples=None,
            timeout_s=15.0,
        )
        start_odom_index = len(self.odom_samples)
        start_pose = self.odom_samples[-1] if self.odom_samples else None
        blocked_candidate, blocked_candidate_status = self.compute_path(
            waypoint, planning_timeout_s
        )
        blocked_route_profile = self.east_hallway_route_profile(blocked_candidate)
        result["native_global_costmap_during_blockage"] = blocked_profile
        result["attempts"].append(
            {
                "attempt": 1,
                "phase": "barrier_active",
                "planning_status": blocked_candidate_status,
                "path_pose_count": (
                    len(blocked_candidate.poses)
                    if blocked_candidate is not None
                    else 0
                ),
                "path_length_m": (
                    round(path_length(blocked_candidate), 4)
                    if blocked_candidate is not None
                    else None
                ),
                "path_blockage_profile": self.path_blockage_profile(
                    blocked_candidate, (10.80, 0.0)
                ),
                "route_authorization": blocked_route_profile,
            }
        )
        blocked_plan_rejected = (
            blocked_candidate is None
            or blocked_route_profile.get("authorized") is False
        )
        result["blocked_plan_rejected"] = blocked_plan_rejected
        result["rejection_authority"] = (
            "nav2_native_global_costmap"
            if blocked_candidate is None
            else "native_costmap_plan_plus_mission_route_authorization"
        )

        self.request_blockage_clear()
        cleared_seen = self.wait_for_blockage_state(False, 15.0)
        result["clearance_requested"] = True
        result["blockage_cleared_seen"] = cleared_seen
        # Replace the final obstacle-bearing observation buffers with several
        # post-removal scans before clearing the costmap layers. Without this
        # settle interval, an in-flight 2 Hz sample can immediately re-mark the
        # barrier after the clear service returns.
        if cleared_seen:
            self.hold_stopped(2.5)
        explicit_clear_succeeded = (
            self.clear_native_global_costmap() if cleared_seen else False
        )
        result["explicit_global_costmap_clear_succeeded"] = explicit_clear_succeeded
        cleared_profile = self.wait_for_native_costmap_profile(
            minimum_lethal_samples=0,
            maximum_lethal_samples=0,
            timeout_s=15.0,
        ) if explicit_clear_succeeded else {"available": False}
        result["native_global_costmap_after_clearance"] = cleared_profile

        end_pose = self.odom_samples[-1] if self.odom_samples else None
        wait_samples = self.odom_samples[start_odom_index:]
        wait_displacement = (
            math.hypot(end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
            if start_pose is not None and end_pose is not None
            else None
        )
        result["safe_wait"] = {
            "measurement_start": "after_native_costmap_marking",
            "odometry_samples": len(wait_samples),
            "maximum_absolute_linear_velocity_mps": max(
                (abs(sample[2]) for sample in wait_samples), default=0.0
            ),
            "displacement_m": wait_displacement,
            "elapsed_wall_s": round(time.monotonic() - started, 3),
        }
        before_lethal = int(clear_baseline.get("lethal_or_inscribed_samples", -1))
        blocked_lethal = int(blocked_profile.get("lethal_or_inscribed_samples", 0))
        cleared_lethal = int(cleared_profile.get("lethal_or_inscribed_samples", -1))
        native_marking_credit = (
            clear_baseline.get("available") is True
            and before_lethal == 0
            and blocked_lethal >= 20
            and cleared_profile.get("available") is True
            and cleared_lethal == 0
        )
        result["nav2_dynamic_costmap_marking_credit"] = native_marking_credit
        baseline_authorized = result["baseline"]["route_authorization"].get(
            "authorized"
        ) is True
        if not baseline_authorized:
            result["passed"] = False
            result["status"] = "clear_baseline_path_not_authorized"
            return None, result["status"], result
        if not blocked_plan_rejected:
            result["passed"] = False
            result["status"] = "blocked_candidate_remained_mission_authorized"
            return None, result["status"], result
        if not cleared_seen or not native_marking_credit:
            result["passed"] = False
            result["status"] = "native_costmap_did_not_clear"
            return None, result["status"], result

        fresh_path, fresh_status = self.compute_path(waypoint, planning_timeout_s)
        fresh_route_profile = self.east_hallway_route_profile(fresh_path)
        result["attempts"].append(
            {
                "attempt": 2,
                "phase": "barrier_cleared",
                "planning_status": fresh_status,
                "path_pose_count": len(fresh_path.poses) if fresh_path is not None else 0,
                "path_length_m": (
                    round(path_length(fresh_path), 4)
                    if fresh_path is not None
                    else None
                ),
                "route_authorization": fresh_route_profile,
            }
        )
        fresh_path_accepted = (
            fresh_path is not None and fresh_route_profile.get("authorized") is True
        )
        result["fresh_path_computed_after_clearance"] = fresh_path_accepted
        result["planner_attempt_count"] = 2
        result["replan_mode"] = "fresh_compute_path_to_pose_after_native_clearance"
        result["passed"] = fresh_path_accepted
        result["status"] = (
            "succeeded"
            if fresh_path_accepted
            else "fresh_path_outside_mission_authorized_route"
            if fresh_path is not None
            else fresh_status
        )
        return (fresh_path if fresh_path_accepted else None), result["status"], result

    def pivot_in_place(
        self,
        direction: float = 1.0,
        target_radians: float = math.pi,
        angular_rad_s: float = 0.42,
        timeout_s: float = 90.0,
    ) -> dict:
        if not self.odom_samples:
            return {"passed": False, "status": "no_odometry", "accumulated_rad": 0.0}
        started = time.monotonic()
        previous_yaw = self.odom_samples[-1][3]
        accumulated = 0.0
        command = Twist()
        command.angular.z = math.copysign(abs(angular_rad_s), direction)
        while time.monotonic() - started < timeout_s and accumulated < target_radians - 0.12:
            self.stop_publisher.publish(command)
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if not self.odom_samples:
                continue
            current_yaw = self.odom_samples[-1][3]
            delta = math.atan2(
                math.sin(current_yaw - previous_yaw),
                math.cos(current_yaw - previous_yaw),
            )
            accumulated += max(0.0, direction * delta)
            previous_yaw = current_yaw
        self.stop()
        passed = accumulated >= target_radians - 0.12
        return {
            "passed": passed,
            "status": "succeeded" if passed else "timeout",
            "direction": "counterclockwise" if direction > 0.0 else "clockwise",
            "requested_angular_rad_s": abs(angular_rad_s),
            "target_rad": target_radians,
            "accumulated_rad": accumulated,
            "elapsed_wall_s": round(time.monotonic() - started, 3),
        }

    def align_to_yaw(
        self,
        target_yaw_deg: float,
        tolerance_deg: float = 2.9,
        angular_rad_s: float = 0.42,
        timeout_s: float = 20.0,
    ) -> dict:
        """Align in the open approach corridor before a straight door crossing."""
        if not self.odom_samples:
            return {"passed": False, "status": "no_odometry"}
        started = time.monotonic()
        target = math.radians(target_yaw_deg)
        tolerance = math.radians(tolerance_deg)
        final_error = math.inf
        while time.monotonic() - started < timeout_s:
            current_yaw = self.odom_samples[-1][3]
            final_error = math.atan2(
                math.sin(target - current_yaw), math.cos(target - current_yaw)
            )
            if abs(final_error) <= tolerance:
                self.stop()
                return {
                    "passed": True,
                    "status": "succeeded",
                    "target_yaw_deg": target_yaw_deg,
                    "tolerance_deg": tolerance_deg,
                    "final_error_deg": math.degrees(final_error),
                    "elapsed_wall_s": round(time.monotonic() - started, 3),
                    "rotation_location": "open_approach_corridor_before_doorway",
                }
            command = Twist()
            command.angular.z = math.copysign(abs(angular_rad_s), final_error)
            self.stop_publisher.publish(command)
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.stop()
        return {
            "passed": False,
            "status": "timeout",
            "target_yaw_deg": target_yaw_deg,
            "tolerance_deg": tolerance_deg,
            "final_error_deg": math.degrees(final_error),
            "elapsed_wall_s": round(time.monotonic() - started, 3),
            "rotation_location": "open_approach_corridor_before_doorway",
        }

    def converge_to_stage(
        self,
        target_x_m: float,
        target_y_m: float,
        tolerance_m: float = 0.015,
        linear_mps: float = 0.18,
        angular_rad_s: float = 0.42,
        heading_tolerance_deg: float = 2.5,
        timeout_s: float = 30.0,
    ) -> dict:
        """Close the coarse Nav2 handoff onto the measured pre-door centre."""
        if not self.odom_samples:
            return {"passed": False, "status": "no_odometry"}
        started = time.monotonic()
        heading_tolerance = math.radians(heading_tolerance_deg)
        final_distance = math.inf
        maximum_linear_command = 0.0
        maximum_angular_command = 0.0
        while time.monotonic() - started < timeout_s:
            x_m, y_m, _, yaw_rad, _ = self.odom_samples[-1]
            dx = target_x_m - x_m
            dy = target_y_m - y_m
            final_distance = math.hypot(dx, dy)
            if final_distance <= tolerance_m:
                self.stop()
                return {
                    "passed": True,
                    "status": "succeeded",
                    "target_xy_m": [target_x_m, target_y_m],
                    "tolerance_m": tolerance_m,
                    "final_distance_m": final_distance,
                    "maximum_linear_command_mps": maximum_linear_command,
                    "maximum_angular_command_rad_s": maximum_angular_command,
                    "elapsed_wall_s": round(time.monotonic() - started, 3),
                }
            desired_yaw = math.atan2(dy, dx)
            heading_error = math.atan2(
                math.sin(desired_yaw - yaw_rad),
                math.cos(desired_yaw - yaw_rad),
            )
            command = Twist()
            if abs(heading_error) <= heading_tolerance:
                # The furnished USD needs the mapped guard's established
                # traction ceiling to overcome translational static friction.
                # The guard closes the loop on actual speed and retains its
                # independent 0.095 m/s doorway overspeed stop.
                command.linear.x = linear_mps
            else:
                command.angular.z = math.copysign(abs(angular_rad_s), heading_error)
            maximum_linear_command = max(maximum_linear_command, command.linear.x)
            maximum_angular_command = max(
                maximum_angular_command, abs(command.angular.z)
            )
            self.stop_publisher.publish(command)
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.stop()
        return {
            "passed": False,
            "status": "timeout",
            "target_xy_m": [target_x_m, target_y_m],
            "tolerance_m": tolerance_m,
            "final_distance_m": final_distance,
            "maximum_linear_command_mps": maximum_linear_command,
            "maximum_angular_command_rad_s": maximum_angular_command,
            "elapsed_wall_s": round(time.monotonic() - started, 3),
        }

    def close(self) -> None:
        self.stop()
        self.node.destroy_node()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config" / "administration_assumptions.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "results" / "administration_nav2_provisional_mission.json",
    )
    parser.add_argument("--ready-timeout-s", type=float, default=45.0)
    parser.add_argument("--planning-timeout-s", type=float, default=20.0)
    parser.add_argument("--waypoint-timeout-s", type=float, default=120.0)
    parser.add_argument("--stop-after-waypoint", type=int, default=0)
    parser.add_argument(
        "--control-stack",
        choices=(
            "nav2",
            "nav2_phase3n_safety",
            "nav2_mapped_doorway_phase3n_safety",
            "nav2_mapped_doorway_phase6_high_speed_safety",
            "nav2_mapped_doorway_phase7_dynamic_crossing_safety",
            PHASE7B_CONTROL_STACK,
            PHASE7D_CONTROL_STACK,
        ),
        default="nav2",
        help="declare the independently launched bridge command-arbitration mode",
    )
    parser.add_argument(
        "--site-profile",
        choices=("provisional", "measured_presentation"),
        default="provisional",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    waypoints = [dict(waypoint) for waypoint in config["route"]["waypoints"]]
    if args.stop_after_waypoint > 0:
        waypoints = waypoints[: args.stop_after_waypoint + 1]

    rclpy.init()
    mission = MissionNode()
    legs = []
    failure = None
    started_at = time.monotonic()
    try:
        transport_ready = mission.wait_for_map_and_odometry(args.ready_timeout_s)
        if not transport_ready:
            failure = "map_or_bridge_not_ready"
        else:
            home = waypoints[0]
            initial_pose_accepted = mission.publish_initial_pose(
                float(home["x_m"]), float(home["y_m"]), float(home["yaw_deg"])
            )
            if not initial_pose_accepted:
                failure = "amcl_did_not_accept_initial_pose"
            # Allow AMCL to publish map -> odom after accepting /initialpose.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                rclpy.spin_once(mission.node, timeout_sec=0.10)
            actions_ready = (
                failure is None
                and mission.wait_for_action_servers(args.ready_timeout_s)
            )
            if not actions_ready:
                failure = "nav2_action_servers_not_ready_after_initial_pose"
            mission.trace_enabled = failure is None
            mission.trace_control_mode = (
                "nav2_phase6_and_phase3n_learned_safety_with_dynamic_crossing"
                if args.control_stack
                == "nav2_mapped_doorway_phase7_dynamic_crossing_safety"
                else "nav2_phase6_and_phase3n_learned_safety_with_blocked_route_replanning"
                if args.control_stack == PHASE7B_CONTROL_STACK
                else "nav2_phase6_and_phase3n_learned_safety_with_native_costmap_safe_wait"
                if args.control_stack == PHASE7D_CONTROL_STACK
                else "nav2_phase6_and_phase3n_learned_safety"
                if args.control_stack in PHASE6_CONTROL_STACKS
                else args.control_stack
            )
            for index, waypoint in enumerate(waypoints[1:], start=1):
                if failure is not None:
                    break
                leg_started = time.monotonic()
                route_segment_id = index - 1
                clear_costmap_baseline = None
                clear_path_baseline = None
                clear_path_baseline_status = "not_requested"
                navigation_waypoint = dict(waypoint)
                offset = OFFICE_STAGING_OFFSETS_M.get(waypoint["id"], (0.0, 0.0))
                navigation_waypoint["x_m"] = float(waypoint["x_m"]) + offset[0]
                navigation_waypoint["y_m"] = float(waypoint["y_m"]) + offset[1]
                mapped_stage = None
                if (
                    args.control_stack in PHASE6_CONTROL_STACKS
                    and waypoint["id"] in PHASE6_PRE_DOOR_ALIGNMENT_WAYPOINTS
                ):
                    door_name = APPROACH_DOOR_BY_WAYPOINT[waypoint["id"]]
                    door = config["doors"][door_name]
                    wall_angle = math.radians(float(door["wall_rotation_deg"]))
                    normal_x = -math.sin(wall_angle)
                    normal_y = math.cos(wall_angle)
                    centre_x, centre_y = (
                        float(value) for value in door["centre_xy_m"]
                    )
                    waypoint_normal_coordinate = (
                        (float(waypoint["x_m"]) - centre_x) * normal_x
                        + (float(waypoint["y_m"]) - centre_y) * normal_y
                    )
                    crossing_sign = -math.copysign(
                        1.0, waypoint_normal_coordinate
                    )
                    stage_x = centre_x - crossing_sign * normal_x
                    stage_y = centre_y - crossing_sign * normal_y
                    mapped_stage = {
                        "door": door_name,
                        "xy_m": [stage_x, stage_y],
                        "distance_before_door_m": 1.0,
                    }
                    navigation_waypoint["x_m"] = stage_x
                    navigation_waypoint["y_m"] = stage_y
                    offset = (
                        stage_x - float(waypoint["x_m"]),
                        stage_y - float(waypoint["y_m"]),
                    )
                if (
                    args.control_stack == PHASE7D_CONTROL_STACK
                    and route_segment_id == 1
                ):
                    mission.hold_stopped(1.5)
                    clear_costmap_baseline = mission.wait_for_native_costmap_profile(
                        minimum_lethal_samples=0,
                        maximum_lethal_samples=0,
                        timeout_s=5.0,
                    )
                    clear_path_baseline, clear_path_baseline_status = (
                        mission.compute_path(
                            navigation_waypoint, args.planning_timeout_s
                        )
                    )
                mission.publish_route_segment(route_segment_id)
                blocked_route_replanning = None
                if (
                    args.control_stack in {PHASE7B_CONTROL_STACK, PHASE7D_CONTROL_STACK}
                    and route_segment_id == 1
                ):
                    if args.control_stack == PHASE7D_CONTROL_STACK:
                        path, planning_status, blocked_route_replanning = (
                            mission.exercise_native_costmap_blocked_route_replan(
                                navigation_waypoint,
                                args.planning_timeout_s,
                                clear_costmap_baseline or {"available": False},
                                clear_path_baseline,
                                clear_path_baseline_status,
                            )
                        )
                    else:
                        path, planning_status, blocked_route_replanning = (
                            mission.exercise_blocked_route_replan(
                                navigation_waypoint, args.planning_timeout_s
                            )
                        )
                else:
                    path, planning_status = mission.compute_path(
                        navigation_waypoint, args.planning_timeout_s
                    )
                record = {
                    "index": index,
                    "route_segment_id": route_segment_id,
                    "waypoint_id": waypoint["id"],
                    "goal_xy_yaw": [
                        float(waypoint["x_m"]),
                        float(waypoint["y_m"]),
                        float(waypoint["yaw_deg"]),
                    ],
                    "navigation_goal_xy_yaw": [
                        float(navigation_waypoint["x_m"]),
                        float(navigation_waypoint["y_m"]),
                        float(navigation_waypoint["yaw_deg"]),
                    ],
                    "simulation_staging_offset_xy_m": list(offset),
                    "mapped_pre_door_stage": mapped_stage,
                    "planning_status": planning_status,
                    "path_pose_count": len(path.poses) if path is not None else 0,
                    "path_length_m": round(path_length(path), 4) if path is not None else None,
                    "blocked_route_replanning": blocked_route_replanning,
                }
                if path is None:
                    record["execution_status"] = "not_started"
                    record["elapsed_wall_s"] = round(time.monotonic() - leg_started, 3)
                    legs.append(record)
                    failure = f"{waypoint['id']}:{planning_status}"
                    break
                goal_checker_id = (
                    "predoor_goal_checker"
                    if mapped_stage is not None
                    else "position_goal_checker"
                )
                executed, execution_status = mission.follow_path(
                    path, args.waypoint_timeout_s, goal_checker_id
                )
                record["goal_checker_id"] = goal_checker_id
                record["execution_status"] = execution_status
                record["elapsed_wall_s"] = round(time.monotonic() - leg_started, 3)
                legs.append(record)
                print(
                    f"AISHA_NAV2_LEG waypoint={waypoint['id']} "
                    f"plan={planning_status} execute={execution_status} "
                    f"length_m={record['path_length_m']}"
                )
                if not executed:
                    failure = f"{waypoint['id']}:{execution_status}"
                    break
                if (
                    args.control_stack in PHASE6_CONTROL_STACKS
                    and waypoint["id"] in PHASE6_PRE_DOOR_ALIGNMENT_WAYPOINTS
                ):
                    stage_convergence = mission.converge_to_stage(
                        float(mapped_stage["xy_m"][0]),
                        float(mapped_stage["xy_m"][1]),
                    )
                    record["pre_door_stage_convergence"] = stage_convergence
                    print(
                        f"AISHA_NAV2_STAGE waypoint={waypoint['id']} "
                        f"status={stage_convergence['status']} "
                        f"distance_m={stage_convergence.get('final_distance_m', float('nan')):.4f}"
                    )
                    if not stage_convergence["passed"]:
                        failure = (
                            f"{waypoint['id']}:pre_door_stage_"
                            f"{stage_convergence['status']}"
                        )
                        break
                    alignment = mission.align_to_yaw(float(waypoint["yaw_deg"]))
                    alignment["mapped_door"] = mapped_stage["door"]
                    alignment["mapped_stage_xy_m"] = mapped_stage["xy_m"]
                    record["pre_door_alignment"] = alignment
                    print(
                        f"AISHA_NAV2_ALIGNMENT waypoint={waypoint['id']} "
                        f"status={alignment['status']} "
                        f"error_deg={alignment.get('final_error_deg', float('nan')):.3f}"
                    )
                    if not alignment["passed"]:
                        failure = (
                            f"{waypoint['id']}:pre_door_alignment_"
                            f"{alignment['status']}"
                        )
                        break
                if waypoint["id"] in POST_VISIT_PIVOTS:
                    pivot = mission.pivot_in_place()
                    record["post_visit_pivot"] = pivot
                    print(
                        f"AISHA_NAV2_PIVOT waypoint={waypoint['id']} "
                        f"status={pivot['status']} accumulated_rad={pivot['accumulated_rad']:.3f}"
                    )
                    if not pivot["passed"]:
                        failure = f"{waypoint['id']}:post_visit_pivot_{pivot['status']}"
                        break
    finally:
        mission.stop()
        end_position = mission.odom_samples[-1][:2] if mission.odom_samples else None
        start_position = mission.odom_samples[0][:2] if mission.odom_samples else None
        displacement = (
            math.hypot(
                end_position[0] - start_position[0],
                end_position[1] - start_position[1],
            )
            if start_position is not None and end_position is not None
            else None
        )
        expected_legs = max(0, len(waypoints) - 1)
        completed_legs = sum(
            item["execution_status"] == "succeeded" for item in legs
        )
        mission_passed = failure is None and len(legs) == expected_legs
        report = {
            "report_type": (
                "administration_nav2_measured_presentation_mission"
                if args.site_profile == "measured_presentation"
                else "administration_nav2_provisional_mission"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed" if failure is None else "failed",
            "route_status": config["route"]["status"],
            "simulation_route_support": {
                "office_staging_offsets_xy_m": {
                    key: list(value) for key, value in OFFICE_STAGING_OFFSETS_M.items()
                },
                "post_visit_pivots": sorted(POST_VISIT_PIVOTS),
                "phase6_pre_door_alignment_waypoints": sorted(
                    PHASE6_PRE_DOOR_ALIGNMENT_WAYPOINTS
                ),
                "pre_door_rotation_location": (
                    "open_approach_corridor_before_doorway"
                ),
                "pre_door_stage_position_tolerance_m": 0.015,
                "pivot_direction": "counterclockwise",
                "reverse_motion_used": False,
            },
            "legs": legs,
            "expected_legs": expected_legs,
            "completed_legs": completed_legs,
            "failure": failure,
            "elapsed_wall_s": round(time.monotonic() - started_at, 3),
            "odometry_samples": len(mission.odom_samples),
            "start_xy_m": list(start_position) if start_position is not None else None,
            "end_xy_m": list(end_position) if end_position is not None else None,
            "net_displacement_m": displacement,
            "follow_feedback_count": mission.follow_feedback_count,
            "final_odometry_yaw_deg": (
                math.degrees(mission.odom_samples[-1][3])
                if mission.odom_samples
                else None
            ),
            "final_odometry_angular_rad_s": (
                mission.odom_samples[-1][4] if mission.odom_samples else None
            ),
            "commands": {
                "samples": len(mission.command_samples),
                "maximum_linear_mps": max(
                    (sample[0] for sample in mission.command_samples), default=0.0
                ),
                "maximum_absolute_angular_rad_s": max(
                    (abs(sample[1]) for sample in mission.command_samples), default=0.0
                ),
                "last_20": [list(sample) for sample in mission.command_samples[-20:]],
            },
            "outcome": "success" if mission_passed else "failed",
            "waypoints_completed": completed_legs,
            "completed_steps": mission._trace_odom_count,
            "duration_s": (
                mission.pose_trace[-1]["elapsed_s"] if mission.pose_trace else 0.0
            ),
            "pose_trace_interval_steps": 3,
            "pose_trace": mission.pose_trace,
            "control_steps": {
                mission.trace_control_mode: mission._trace_odom_count,
            },
            "route_control": mission.trace_control_mode,
            "checkpoint": (
                str(PHASE6_CHECKPOINT.resolve())
                if args.control_stack in PHASE6_CONTROL_STACKS
                else None
            ),
            "seed": 6084,
            "root_transform_animation": False,
            "policy_architecture": (
                "nav2_global_and_dwb_plus_phase6_phase3n_learned_brake_safety"
                if args.control_stack in PHASE6_CONTROL_STACKS
                else args.control_stack
            ),
            "map_status": (
                "measured_site_presentation_candidate"
                if args.site_profile == "measured_presentation"
                else "provisional_plan_derived_not_measured"
            ),
            "site_profile": args.site_profile,
            "control_stack": args.control_stack,
            "learned_policy_coupled": args.control_stack
            in {
                "nav2_phase3n_safety",
                "nav2_mapped_doorway_phase3n_safety",
                "nav2_mapped_doorway_phase6_high_speed_safety",
                "nav2_mapped_doorway_phase7_dynamic_crossing_safety",
                PHASE7B_CONTROL_STACK,
                PHASE7D_CONTROL_STACK,
            },
            "learned_360_safety_coupled": (
                args.control_stack
                in {
                    "nav2_phase3n_safety",
                    "nav2_mapped_doorway_phase3n_safety",
                    "nav2_mapped_doorway_phase6_high_speed_safety",
                    "nav2_mapped_doorway_phase7_dynamic_crossing_safety",
                    PHASE7B_CONTROL_STACK,
                    PHASE7D_CONTROL_STACK,
                }
            ),
            "mapped_doorway_safety_coupled": (
                args.control_stack
                in {
                    "nav2_mapped_doorway_phase3n_safety",
                    "nav2_mapped_doorway_phase6_high_speed_safety",
                    "nav2_mapped_doorway_phase7_dynamic_crossing_safety",
                    PHASE7B_CONTROL_STACK,
                    PHASE7D_CONTROL_STACK,
                }
            ),
            "phase6_high_speed_safety_coupled": (
                args.control_stack in PHASE6_CONTROL_STACKS
            ),
            "phase7_dynamic_crossing_safety_coupled": (
                args.control_stack
                == "nav2_mapped_doorway_phase7_dynamic_crossing_safety"
            ),
            "phase7b_blocked_route_replanning_coupled": (
                args.control_stack == PHASE7B_CONTROL_STACK
            ),
            "phase7d_administration_native_costmap_coupled": (
                args.control_stack == PHASE7D_CONTROL_STACK
            ),
            "blocked_route_replanning": next(
                (
                    leg.get("blocked_route_replanning")
                    for leg in legs
                    if leg.get("blocked_route_replanning") is not None
                ),
                None,
            ),
            "blocked_route_coordination": {
                "state_topic": "/aisha/blocked_route_active",
                "release_topic": "/aisha/clear_blocked_route",
                "state_messages": mission.blockage_state_messages,
                "transitions": mission.blockage_transitions,
                "used_for_scenario_synchronization_only": True,
                "exposed_to_policy": False,
            },
            "frozen_phase3m_local_navigation_coupled": False,
            "physical_release": False,
            "claim_boundary": (
                "This is a live Nav2/Isaac physics mission. The "
                "control-stack declaration is verified against the paired bridge report by "
                "the corresponding integration validator. A measured_presentation profile "
                "uses the reported minimum and disclosed assumptions; it does not prove "
                "measured-site clearance, stopping distance, sim-to-real performance, or "
                "physical safety."
            ),
        }
        report["passed"] = mission_passed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"AISHA_NAV2_MISSION passed={report['passed']} "
            f"legs={report['completed_legs']}/{expected_legs} report={args.output}"
        )
        mission.signal_completion()
        mission.close()
        rclpy.shutdown()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
