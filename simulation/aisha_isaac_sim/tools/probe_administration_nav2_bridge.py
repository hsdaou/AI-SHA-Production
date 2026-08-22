#!/usr/bin/env python3
"""Probe the running Isaac bridge from the separate system ROS 2 environment."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan


class ProbeNode:
    def __init__(self) -> None:
        self.node = rclpy.create_node("aisha_administration_bridge_probe")
        self.odom_samples: list[tuple[float, float, float]] = []
        self.clock_samples = 0
        self.crown_scan_samples = 0
        self.front_scan_samples = 0
        self.front_ranges_finite = True
        self.front_range_count = 0
        self.node.create_subscription(Odometry, "/odom", self._odom, 10)
        self.node.create_subscription(Clock, "/clock", self._clock, 10)
        self.node.create_subscription(
            LaserScan, "/scan", self._crown, qos_profile_sensor_data
        )
        self.node.create_subscription(
            LaserScan, "/front_scan", self._front, qos_profile_sensor_data
        )
        self.command_publisher = self.node.create_publisher(Twist, "/cmd_vel", 10)

    def _odom(self, message: Odometry) -> None:
        self.odom_samples.append(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.twist.twist.linear.x,
            )
        )

    def _clock(self, _message: Clock) -> None:
        self.clock_samples += 1

    def _crown(self, _message: LaserScan) -> None:
        self.crown_scan_samples += 1

    def _front(self, message: LaserScan) -> None:
        self.front_scan_samples += 1
        self.front_range_count = len(message.ranges)
        self.front_ranges_finite &= all(math.isfinite(value) for value in message.ranges)

    def publish_command(self, linear_mps: float) -> None:
        message = Twist()
        message.linear.x = linear_mps
        self.command_publisher.publish(message)

    def spin_for(self, duration_s: float, command_mps: float | None = None) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if command_mps is not None:
                self.publish_command(command_mps)
            rclpy.spin_once(self.node, timeout_sec=0.025)

    def close(self) -> None:
        self.node.destroy_node()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery-timeout-s", type=float, default=10.0)
    parser.add_argument("--command-duration-s", type=float, default=2.0)
    parser.add_argument("--command-mps", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.0 < args.command_mps <= 0.30:
        parser.error("--command-mps must be in (0, 0.30]")

    rclpy.init()
    probe = ProbeNode()
    try:
        deadline = time.monotonic() + args.discovery_timeout_s
        while time.monotonic() < deadline and not (
            probe.odom_samples
            and probe.clock_samples
            and probe.crown_scan_samples
            and probe.front_scan_samples
        ):
            rclpy.spin_once(probe.node, timeout_sec=0.05)
        start = probe.odom_samples[-1] if probe.odom_samples else None
        probe.spin_for(args.command_duration_s, command_mps=args.command_mps)
        probe.spin_for(0.40, command_mps=0.0)
        end = probe.odom_samples[-1] if probe.odom_samples else None
        displacement = None
        maximum_forward_velocity = max(
            (sample[2] for sample in probe.odom_samples), default=0.0
        )
        if start is not None and end is not None:
            displacement = math.hypot(end[0] - start[0], end[1] - start[1])
        topics = {name for name, _types in probe.node.get_topic_names_and_types()}
        required_topics = {
            "/clock",
            "/cmd_vel",
            "/odom",
            "/tf",
            "/tf_static",
            "/scan",
            "/front_scan",
        }
        checks = {
            "required_topics_discovered": required_topics.issubset(topics),
            "clock_received": probe.clock_samples > 0,
            "odometry_received": len(probe.odom_samples) > 1,
            "crown_scan_received": probe.crown_scan_samples > 0,
            "front_scan_received": probe.front_scan_samples > 0,
            "front_scan_has_25_rays": probe.front_range_count == 25,
            "front_scan_ranges_finite": probe.front_ranges_finite,
            "external_cmd_vel_moved_physics_robot": displacement is not None
            and (displacement > 0.002 or maximum_forward_velocity > 0.01),
        }
        report = {
            "report_type": "administration_nav2_bridge_external_probe",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "passed": all(checks.values()),
            "observed_topics": sorted(topics),
            "samples": {
                "clock": probe.clock_samples,
                "odometry": len(probe.odom_samples),
                "crown_scan": probe.crown_scan_samples,
                "front_scan": probe.front_scan_samples,
                "front_range_count": probe.front_range_count,
            },
            "command": {
                "linear_mps": args.command_mps,
                "duration_wall_s": args.command_duration_s,
                "start_xy_m": start[:2] if start is not None else None,
                "end_xy_m": end[:2] if end is not None else None,
                "observed_displacement_m": displacement,
                "maximum_observed_forward_velocity_mps": maximum_forward_velocity,
            },
            "physical_release": False,
            "claim_boundary": (
                "This probe proves bidirectional ROS 2 message exchange and simulated wheel-driven "
                "motion. It is not a Nav2 mission, learned-policy integration, or physical safety test."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"AISHA_BRIDGE_EXTERNAL_PROBE passed={report['passed']} "
            f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
        )
        return 0 if report["passed"] else 1
    finally:
        probe.publish_command(0.0)
        probe.close()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
