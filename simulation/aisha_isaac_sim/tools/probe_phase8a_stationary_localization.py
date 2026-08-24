#!/usr/bin/env python3
"""Observe the Phase 8A ROS graph without publishing any motion command."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "config/phase8a_physical_localization_preflight.yaml",
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase8a_stationary_localization_probe.json",
    )
    return parser.parse_args()


class StationaryLocalizationProbe(Node):
    """Read-only topic/TF observer. Deliberately has no publishers or services."""

    def __init__(self, profile: dict[str, Any]) -> None:
        super().__init__("phase8a_stationary_localization_probe")
        self.profile = profile
        self.samples: dict[str, list[float]] = {
            "scan": [],
            "imu": [],
            "wheel_odom": [],
            "filtered_odom": [],
        }
        self.frames: dict[str, set[str]] = {key: set() for key in self.samples}
        self.scan_finite_fraction_min = 1.0
        self.imu_quaternion_norm_error_max = 0.0
        self.stationary_linear_speed_max = 0.0
        self.stationary_angular_speed_max = 0.0
        contract = profile["localization_contract"]

        self.create_subscription(
            LaserScan,
            contract["crown_scan_topic"],
            self._scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            contract["imu_topic"],
            self._imu,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            contract["raw_wheel_odometry_topic"],
            self._wheel_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            contract["filtered_odometry_topic"],
            self._filtered_odom,
            qos_profile_sensor_data,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _record(self, key: str, frame_id: str) -> None:
        self.samples[key].append(time.monotonic())
        self.frames[key].add(frame_id)

    def _scan(self, message: LaserScan) -> None:
        self._record("scan", message.header.frame_id)
        if message.ranges:
            finite = sum(math.isfinite(value) for value in message.ranges)
            self.scan_finite_fraction_min = min(
                self.scan_finite_fraction_min, finite / len(message.ranges)
            )

    def _imu(self, message: Imu) -> None:
        self._record("imu", message.header.frame_id)
        orientation = message.orientation
        norm = math.sqrt(
            orientation.x**2 + orientation.y**2 + orientation.z**2 + orientation.w**2
        )
        self.imu_quaternion_norm_error_max = max(
            self.imu_quaternion_norm_error_max, abs(norm - 1.0)
        )

    def _record_odom(self, key: str, message: Odometry) -> None:
        self._record(key, f"{message.header.frame_id}->{message.child_frame_id}")
        linear = message.twist.twist.linear
        angular = message.twist.twist.angular
        self.stationary_linear_speed_max = max(
            self.stationary_linear_speed_max,
            math.sqrt(linear.x**2 + linear.y**2 + linear.z**2),
        )
        self.stationary_angular_speed_max = max(
            self.stationary_angular_speed_max,
            math.sqrt(angular.x**2 + angular.y**2 + angular.z**2),
        )

    def _wheel_odom(self, message: Odometry) -> None:
        self._record_odom("wheel_odom", message)

    def _filtered_odom(self, message: Odometry) -> None:
        self._record_odom("filtered_odom", message)

    def observed_rate(self, key: str) -> float:
        samples = self.samples[key]
        if len(samples) < 2 or samples[-1] <= samples[0]:
            return 0.0
        return (len(samples) - 1) / (samples[-1] - samples[0])

    def transform_available(self, parent: str, child: str) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                parent, child, Time(), timeout=Duration(seconds=0.25)
            )
            return True
        except TransformException:
            return False


def main() -> int:
    args = parse_args()
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    configured_duration = float(profile["stationary_runtime_gate"]["observation_duration_s"])
    duration = configured_duration if args.duration is None else max(0.25, args.duration)

    rclpy.init()
    node = StationaryLocalizationProbe(profile)
    deadline = time.monotonic() + duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.10, max(0.0, deadline - time.monotonic())))

        sensor = profile["sensor_contract"]
        localization = profile["localization_contract"]
        required_transforms = profile["stationary_runtime_gate"]["required_transforms"]
        transform_checks = {
            f"{parent}->{child}": node.transform_available(parent, child)
            for parent, child in required_transforms
        }
        cmd_publishers = node.get_publishers_info_by_topic("/cmd_vel")
        rates = {key: node.observed_rate(key) for key in node.samples}
        has_scan = bool(node.samples["scan"])
        has_imu = bool(node.samples["imu"])
        has_both_odometry_streams = bool(node.samples["wheel_odom"]) and bool(
            node.samples["filtered_odom"]
        )
        checks = {
            "crown_scan_rate": rates["scan"] >= sensor["crown_lidar"]["minimum_observed_rate_hz"],
            "imu_rate": rates["imu"] >= sensor["imu"]["minimum_observed_rate_hz"],
            "raw_wheel_odometry_rate": rates["wheel_odom"]
            >= sensor["wheel_odometry"]["minimum_observed_rate_hz"],
            "filtered_odometry_rate": rates["filtered_odom"]
            >= sensor["wheel_odometry"]["minimum_observed_rate_hz"],
            "crown_scan_frame": node.frames["scan"] == {localization["crown_lidar_frame"]},
            "imu_frame": node.frames["imu"] == {localization["imu_frame"]},
            "raw_wheel_odometry_frames": node.frames["wheel_odom"]
            == {f'{localization["odom_frame"]}->{localization["base_frame"]}'},
            "filtered_odometry_frames": node.frames["filtered_odom"]
            == {f'{localization["odom_frame"]}->{localization["base_frame"]}'},
            "scan_contains_finite_returns": has_scan and node.scan_finite_fraction_min >= 0.10,
            "imu_quaternion_is_normalized": has_imu
            and node.imu_quaternion_norm_error_max <= 0.05,
            "robot_remained_stationary_linear": has_both_odometry_streams
            and node.stationary_linear_speed_max
            <= sensor["wheel_odometry"]["maximum_stationary_linear_speed_mps"],
            "robot_remained_stationary_angular": has_both_odometry_streams
            and node.stationary_angular_speed_max
            <= sensor["wheel_odometry"]["maximum_stationary_angular_speed_rad_s"],
            "required_tf_chain_available": all(transform_checks.values()),
            "no_cmd_vel_publisher_present": len(cmd_publishers) == 0,
        }
        passed = all(checks.values())
        report = {
            "report_type": "phase8a_stationary_localization_runtime_probe",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "status": "stationary_runtime_gate_passed" if passed else "stationary_runtime_gate_blocked",
            "duration_s": duration,
            "checks": checks,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "observed_rates_hz": rates,
            "observed_frames": {key: sorted(value) for key, value in node.frames.items()},
            "transform_checks": transform_checks,
            "scan_finite_fraction_min": node.scan_finite_fraction_min,
            "imu_quaternion_norm_error_max": node.imu_quaternion_norm_error_max,
            "stationary_linear_speed_max_mps": node.stationary_linear_speed_max,
            "stationary_angular_speed_max_rad_s": node.stationary_angular_speed_max,
            "cmd_vel_publisher_count": len(cmd_publishers),
            "motion_command_published_by_probe": False,
            "stationary_runtime_gate_passed": passed,
            "physical_runtime_ready": False,
            "motion_authorized": False,
            "physical_release": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "AISHA_PHASE8A_STATIONARY_PROBE "
            f"passed={passed} checks={report['checks_passed']}/{report['checks_total']} "
            f"report={args.output.resolve()}"
        )
        return 0 if passed else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
