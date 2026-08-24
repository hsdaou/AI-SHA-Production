#!/usr/bin/env python3
"""Prove native Nav2 marking and execute the alternate Phase 7C branch."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from std_msgs.msg import Bool

from run_administration_nav2_mission import MissionNode, path_length


ROOT = Path(__file__).resolve().parents[1]
START_XY_M = (1.30, 0.45)
GOAL = {"x_m": 10.70, "y_m": 0.45, "yaw_deg": 0.0}
BLOCKER_CENTRE_XY_M = (6.00, 2.10)


def branch_profile(path) -> dict[str, object]:
    if path is None:
        return {"available": False}
    points = [
        (float(pose.pose.position.x), float(pose.pose.position.y))
        for pose in path.poses
    ]
    island_span = [point for point in points if 4.25 <= point[0] <= 7.75]
    if not island_span:
        return {"available": False, "reason": "no_path_points_at_island"}
    mean_y = sum(point[1] for point in island_span) / len(island_span)
    branch = "top" if mean_y > 0.80 else "bottom" if mean_y < -0.80 else "centre_invalid"
    return {
        "available": True,
        "branch": branch,
        "mean_island_span_y_m": mean_y,
        "minimum_island_span_y_m": min(point[1] for point in island_span),
        "maximum_island_span_y_m": max(point[1] for point in island_span),
        "island_span_pose_count": len(island_span),
        "pose_count": len(points),
        "path_length_m": path_length(path),
    }


class Phase7CMission(MissionNode):
    def __init__(self) -> None:
        super().__init__()
        self.activation_publisher = self.node.create_publisher(
            Bool, "/aisha/activate_blocked_route", 10
        )

    def request_blockage_activation(self) -> None:
        message = Bool()
        message.data = True
        deadline = time.monotonic() + 3.0
        while (
            self.activation_publisher.get_subscription_count() == 0
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        for _ in range(5):
            self.activation_publisher.publish(message)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def blockage_costmap_profile(self) -> dict[str, object]:
        message = self.global_costmap
        if message is None:
            return {"available": False}
        metadata = message.metadata
        resolution = float(metadata.resolution)
        origin_x = float(metadata.origin.position.x)
        origin_y = float(metadata.origin.position.y)
        values = []
        sample_cells = []
        x_m = 5.80
        y_m = 1.45
        while y_m <= 2.75 + 1.0e-6:
            x_index = int(math.floor((x_m - origin_x) / resolution))
            y_index = int(math.floor((y_m - origin_y) / resolution))
            if 0 <= x_index < metadata.size_x and 0 <= y_index < metadata.size_y:
                value = int(message.data[y_index * metadata.size_x + x_index])
                values.append(value)
                sample_cells.append([round(x_m, 3), round(y_m, 3), value])
            y_m += 0.05
        centre_x_index = int(math.floor((5.80 - origin_x) / resolution))
        centre_y_index = int(math.floor((2.10 - origin_y) / resolution))
        centre_cost = int(
            message.data[centre_y_index * metadata.size_x + centre_x_index]
        )
        return {
            "available": True,
            "costmap_samples_received": self.global_costmap_samples,
            "sample_x_m": 5.80,
            "sample_y_range_m": [1.45, 2.75],
            "sample_count": len(values),
            "centre_cost": centre_cost,
            "maximum_cost": max(values, default=None),
            "lethal_or_inscribed_samples": sum(value >= 253 for value in values),
            "nonzero_samples": sum(value > 0 for value in values),
            "samples_xy_cost": sample_cells,
        }

    def wait_for_native_marking(self, timeout_s: float) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        latest = {"available": False}
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.10)
            latest = self.blockage_costmap_profile()
            if (
                latest.get("centre_cost", 0) >= 253
                and latest.get("lethal_or_inscribed_samples", 0) >= 8
            ):
                return latest
        return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase7c_native_costmap_detour_mission.json",
    )
    parser.add_argument("--planning-timeout-s", type=float, default=30.0)
    parser.add_argument("--execution-timeout-s", type=float, default=180.0)
    args = parser.parse_args()

    rclpy.init()
    mission = Phase7CMission()
    started = time.monotonic()
    report: dict[str, object] = {
        "report_type": "phase7c_native_costmap_detour_mission",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "top_branch_blocked_bottom_branch_available",
        "map": "maps/phase7c_native_detour_loop/phase7c_native_detour_loop.yaml",
        "start_xy_m": list(START_XY_M),
        "goal_xy_m": [GOAL["x_m"], GOAL["y_m"]],
        "blocker_centre_xy_m": list(BLOCKER_CENTRE_XY_M),
        "scenario_state_used_for_test_synchronization_only": True,
        "scenario_state_exposed_to_policy": False,
        "native_nav2_dynamic_costmap_marking_credit": False,
        "spatial_detour_credit": False,
    }
    passed = False
    try:
        if not mission.wait_for_map_and_odometry(25.0):
            report["failure"] = "map_or_odometry_timeout"
            return_code = 1
        elif not mission.publish_initial_pose(*START_XY_M, 0.0):
            report["failure"] = "initial_pose_not_accepted"
            return_code = 1
        elif not mission.wait_for_action_servers(35.0):
            report["failure"] = "nav2_action_servers_timeout"
            return_code = 1
        else:
            mission.publish_route_segment(0)
            mission.trace_enabled = True
            mission.trace_control_mode = "nav2_phase7c_native_detour"
            mission.hold_stopped(1.5)
            report["costmap_before_activation"] = mission.blockage_costmap_profile()

            baseline_path, baseline_status = mission.compute_path(
                GOAL, args.planning_timeout_s
            )
            baseline_profile = branch_profile(baseline_path)
            report["baseline"] = {
                "planning_status": baseline_status,
                **baseline_profile,
            }

            mission.request_blockage_activation()
            active_seen = mission.wait_for_blockage_state(True, 10.0)
            report["blockage_active_seen"] = active_seen
            marked_profile = mission.wait_for_native_marking(12.0)
            report["native_global_costmap_marking"] = marked_profile
            native_marked = bool(
                marked_profile.get("centre_cost", 0) >= 253
                and marked_profile.get("lethal_or_inscribed_samples", 0) >= 8
            )
            report["native_nav2_dynamic_costmap_marking_credit"] = native_marked

            detour_path, detour_status = mission.compute_path(
                GOAL, args.planning_timeout_s
            )
            detour_profile = branch_profile(detour_path)
            report["detour"] = {
                "planning_status": detour_status,
                **detour_profile,
            }
            spatial_detour = bool(
                baseline_profile.get("branch") == "top"
                and detour_profile.get("branch") == "bottom"
            )
            report["spatial_detour_credit"] = spatial_detour

            execution_ok = False
            execution_status = "not_attempted"
            if active_seen and native_marked and spatial_detour and detour_path is not None:
                execution_ok, execution_status = mission.follow_path(
                    detour_path,
                    args.execution_timeout_s,
                    "general_goal_checker",
                )
            mission.stop()
            report["execution"] = {
                "attempted": detour_path is not None and native_marked and spatial_detour,
                "succeeded": execution_ok,
                "status": execution_status,
            }
            final_pose = mission.odom_samples[-1] if mission.odom_samples else None
            final_distance = (
                math.hypot(final_pose[0] - GOAL["x_m"], final_pose[1] - GOAL["y_m"])
                if final_pose is not None
                else None
            )
            report["final_xy_m"] = list(final_pose[:2]) if final_pose else None
            report["final_goal_distance_m"] = final_distance
            report["blockage_active_at_execution_end"] = mission.blockage_active
            report["odometry_samples"] = len(mission.odom_samples)
            report["command_samples"] = len(mission.command_samples)
            report["maximum_commanded_linear_mps"] = max(
                (sample[0] for sample in mission.command_samples), default=0.0
            )
            report["pose_trace"] = mission.pose_trace
            passed = bool(
                execution_ok
                and active_seen
                and native_marked
                and spatial_detour
                and mission.blockage_active
                and final_distance is not None
                and final_distance <= 0.32
            )
            report["failure"] = None if passed else "phase7c_acceptance_not_met"
            return_code = 0 if passed else 1
    finally:
        report["elapsed_wall_s"] = round(time.monotonic() - started, 3)
        report["passed"] = passed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        mission.stop()
        mission.signal_completion()
        mission.close()
        rclpy.shutdown()

    print(
        "AISHA_PHASE7C_MISSION "
        f"passed={report['passed']} "
        f"native_marking={report['native_nav2_dynamic_costmap_marking_credit']} "
        f"detour={report['spatial_detour_credit']} report={args.output}"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
