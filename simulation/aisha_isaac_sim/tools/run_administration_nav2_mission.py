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
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OFFICE_STAGING_OFFSETS_M = {
    # DWB hands a position-only waypoint back near the edge of its 0.40 m
    # tolerance.  These interior staging offsets make that accepted pose land
    # on the disclosed presentation stop, with room left for the departure
    # pivot.  They are simulation-only and explicitly reported below.
    "vice_principal": (0.0, -0.25),
    "principal": (0.17, -0.19),
}
POST_VISIT_PIVOTS = {"vice_principal", "principal"}


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
        self.initial_pose_publisher = self.node.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.stop_publisher = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.completion_publisher = self.node.create_publisher(
            Bool, "/aisha/mission_complete", 10
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
        self.map_received = False
        self.amcl_pose_samples = 0
        self.odom_samples: list[tuple[float, float, float, float, float]] = []
        self.command_samples: list[tuple[float, float]] = []
        self.follow_feedback_count = 0
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

    def _command(self, message: Twist) -> None:
        self.command_samples.append((message.linear.x, message.angular.z))

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
        choices=("nav2", "nav2_phase3n_safety"),
        default="nav2",
        help="declare the independently launched bridge command-arbitration mode",
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
            for index, waypoint in enumerate(waypoints[1:], start=1):
                if failure is not None:
                    break
                leg_started = time.monotonic()
                navigation_waypoint = dict(waypoint)
                offset = OFFICE_STAGING_OFFSETS_M.get(waypoint["id"], (0.0, 0.0))
                navigation_waypoint["x_m"] = float(waypoint["x_m"]) + offset[0]
                navigation_waypoint["y_m"] = float(waypoint["y_m"]) + offset[1]
                path, planning_status = mission.compute_path(
                    navigation_waypoint, args.planning_timeout_s
                )
                record = {
                    "index": index,
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
                    "planning_status": planning_status,
                    "path_pose_count": len(path.poses) if path is not None else 0,
                    "path_length_m": round(path_length(path), 4) if path is not None else None,
                }
                if path is None:
                    record["execution_status"] = "not_started"
                    record["elapsed_wall_s"] = round(time.monotonic() - leg_started, 3)
                    legs.append(record)
                    failure = f"{waypoint['id']}:{planning_status}"
                    break
                goal_checker_id = "position_goal_checker"
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
        report = {
            "report_type": "administration_nav2_provisional_mission",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed" if failure is None else "failed",
            "route_status": config["route"]["status"],
            "simulation_route_support": {
                "office_staging_offsets_xy_m": {
                    key: list(value) for key, value in OFFICE_STAGING_OFFSETS_M.items()
                },
                "post_visit_pivots": sorted(POST_VISIT_PIVOTS),
                "pivot_direction": "counterclockwise",
                "reverse_motion_used": False,
            },
            "legs": legs,
            "expected_legs": expected_legs,
            "completed_legs": sum(
                item["execution_status"] == "succeeded" for item in legs
            ),
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
            "map_status": "provisional_plan_derived_not_measured",
            "control_stack": args.control_stack,
            "learned_policy_coupled": args.control_stack == "nav2_phase3n_safety",
            "learned_360_safety_coupled": (
                args.control_stack == "nav2_phase3n_safety"
            ),
            "frozen_phase3m_local_navigation_coupled": False,
            "physical_release": False,
            "claim_boundary": (
                "This is a live Nav2/Isaac physics mission on provisional geometry. The "
                "control-stack declaration is verified against the paired bridge report by "
                "validate_administration_nav2_phase3n_integration.py. It does not prove "
                "measured-site clearance, stopping distance, sim-to-real performance, or "
                "physical safety."
            ),
        }
        report["passed"] = failure is None and len(legs) == expected_legs
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
