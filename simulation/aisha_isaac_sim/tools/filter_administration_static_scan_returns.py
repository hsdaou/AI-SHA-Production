#!/usr/bin/env python3
"""Mask mapped-static LaserScan endpoints from Nav2's dynamic marking stream.

Raw scans remain connected to costmap ray clearing. The filtered outputs retain
only returns whose endpoints are not represented by the static occupancy map,
so mapped walls are not inflated a second time while real temporary obstacles
in mapped-free space remain native Nav2 obstacle marks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    PACKAGE_ROOT / "results" / "administration_nav2_phase7e_static_scan_fusion.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StaticScanReturnFilter(Node):
    def __init__(self, static_match_tolerance_m: float) -> None:
        super().__init__(
            "aisha_administration_static_scan_return_filter",
            parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)],
        )
        self.static_match_tolerance_m = static_match_tolerance_m
        self.static_map: OccupancyGrid | None = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_callback, map_qos)
        self.crown_publisher = self.create_publisher(
            LaserScan, "/aisha/static_fused/scan_dynamic", scan_qos
        )
        self.front_publisher = self.create_publisher(
            LaserScan, "/aisha/static_fused/front_scan_dynamic", scan_qos
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            lambda message: self._scan_callback(
                "crown", message, self.crown_publisher
            ),
            scan_qos,
        )
        self.create_subscription(
            LaserScan,
            "/front_scan",
            lambda message: self._scan_callback(
                "front", message, self.front_publisher
            ),
            scan_qos,
        )
        self.statistics = {
            "map_messages": 0,
            "crown": self._empty_scan_statistics(),
            "front": self._empty_scan_statistics(),
        }

    @staticmethod
    def _empty_scan_statistics() -> dict[str, int]:
        return {
            "messages": 0,
            "published_messages": 0,
            "finite_returns": 0,
            "static_returns_masked": 0,
            "mapped_free_returns_preserved": 0,
            "map_unavailable_messages": 0,
            "transform_failures": 0,
        }

    def _map_callback(self, message: OccupancyGrid) -> None:
        self.static_map = message
        self.statistics["map_messages"] += 1

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
        return math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )

    def _static_occupied_near(self, x_m: float, y_m: float) -> bool:
        message = self.static_map
        if message is None:
            return False
        metadata = message.info
        resolution = float(metadata.resolution)
        origin_x = float(metadata.origin.position.x)
        origin_y = float(metadata.origin.position.y)
        x_index = int(math.floor((x_m - origin_x) / resolution))
        y_index = int(math.floor((y_m - origin_y) / resolution))
        radius_cells = int(math.ceil(self.static_match_tolerance_m / resolution))
        width = int(metadata.width)
        height = int(metadata.height)
        for candidate_y in range(y_index - radius_cells, y_index + radius_cells + 1):
            if not 0 <= candidate_y < height:
                continue
            cell_y = origin_y + (candidate_y + 0.5) * resolution
            for candidate_x in range(x_index - radius_cells, x_index + radius_cells + 1):
                if not 0 <= candidate_x < width:
                    continue
                cell_x = origin_x + (candidate_x + 0.5) * resolution
                if math.hypot(cell_x - x_m, cell_y - y_m) > self.static_match_tolerance_m:
                    continue
                if int(message.data[candidate_y * width + candidate_x]) >= 50:
                    return True
        return False

    def _scan_callback(self, name: str, message: LaserScan, publisher) -> None:
        stats = self.statistics[name]
        stats["messages"] += 1
        if self.static_map is None:
            stats["map_unavailable_messages"] += 1
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            stats["transform_failures"] += 1
            return
        translation = transform.transform.translation
        sensor_yaw = self._yaw_from_quaternion(transform.transform.rotation)
        output = copy.deepcopy(message)
        filtered_ranges = list(message.ranges)
        angle = float(message.angle_min)
        for index, value in enumerate(message.ranges):
            range_m = float(value)
            if math.isfinite(range_m) and message.range_min <= range_m <= message.range_max:
                stats["finite_returns"] += 1
                beam_yaw = sensor_yaw + angle
                endpoint_x = float(translation.x) + math.cos(beam_yaw) * range_m
                endpoint_y = float(translation.y) + math.sin(beam_yaw) * range_m
                if self._static_occupied_near(endpoint_x, endpoint_y):
                    filtered_ranges[index] = math.nan
                    stats["static_returns_masked"] += 1
                else:
                    stats["mapped_free_returns_preserved"] += 1
            angle += float(message.angle_increment)
        output.ranges = filtered_ranges
        publisher.publish(output)
        stats["published_messages"] += 1

    def report(self, map_yaml: Path) -> dict:
        return {
            "report_type": "administration_nav2_phase7e_static_scan_fusion",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "architecture": {
                "raw_scan_role": "native_costmap_clearing_only",
                "filtered_scan_role": "native_costmap_dynamic_marking_only",
                "static_map_role": "mapped_static_obstacle_authority",
                "static_match_tolerance_m": self.static_match_tolerance_m,
                "masked_returns_remain_represented_by_static_layer": True,
                "mapped_free_obstacle_returns_preserved": True,
            },
            "map": {
                "yaml": str(map_yaml.resolve()),
                "sha256": sha256(map_yaml),
            },
            "statistics": self.statistics,
            "physical_release": False,
            "claim_boundary": (
                "This map-aware fusion gate prevents duplicate inflation of known "
                "static geometry in the presentation twin. It does not validate "
                "localization error, an unmeasured real office, or physical safety."
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "maps"
            / "administration_measured_presentation_1cm"
            / "administration_measured_presentation_1cm.yaml"
        ),
    )
    parser.add_argument("--static-match-tolerance-m", type=float, default=0.035)
    args = parser.parse_args()
    if args.static_match_tolerance_m < 0.0:
        parser.error("--static-match-tolerance-m must be non-negative")
    rclpy.init()
    node = StaticScanReturnFilter(args.static_match_tolerance_m)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(node.report(args.map_yaml), indent=2) + "\n",
            encoding="utf-8",
        )
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
