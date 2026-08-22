#!/usr/bin/env python3
"""Expose the live administration physics scene to ROS 2/Nav2.

This is the simulation motion boundary: Nav2 may command the articulated Rev D
differential drive through /cmd_vel while Isaac Sim publishes /clock, /odom,
/tf, /scan and /front_scan. The frozen learned policy is intentionally not
coupled here; that arbitration remains a separately reported gate.
"""

from __future__ import annotations

import argparse
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
import aisha_isaaclab.tasks  # noqa: E402,F401


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

    def __init__(self, raw_env, control_period_s: float, sensor_positions: dict[str, list[float]]):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self.rclpy = rclpy
        self.raw_env = raw_env
        self.control_period_s = control_period_s
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
        self.command_linear_mps = 0.0
        self.command_angular_rad_s = 0.0
        self.command_received_at_s = -math.inf
        self.simulation_time_s = 0.0
        self.stop_latched = False
        self.front_stop_trigger_m = 0.60
        self.front_stop_release_m = 0.75
        self.command_watchdog_s = 0.30
        self.maximum_forward_mps = 0.30
        self.maximum_angular_rad_s = 0.55
        self.rejected_reverse_commands = 0
        self.rejected_lateral_commands = 0
        self.watchdog_stops = 0
        self.obstacle_stops = 0
        self.messages = {
            "clock": 0,
            "odom": 0,
            "tf": 0,
            "tf_static": 0,
            "scan": 0,
            "front_scan": 0,
        }
        self.minimum_front_range_m = math.inf
        self.publish_static_transforms(sensor_positions)

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
        front_min = float(torch.amin(self.front_ranges()[0]).item())
        self.minimum_front_range_m = min(self.minimum_front_range_m, front_min)
        if self.stop_latched:
            self.stop_latched = front_min < self.front_stop_release_m
        elif front_min <= self.front_stop_trigger_m:
            self.stop_latched = True
            self.obstacle_stops += 1

        stale = self.simulation_time_s - self.command_received_at_s > self.command_watchdog_s
        if stale and (self.command_linear_mps != 0.0 or self.command_angular_rad_s != 0.0):
            self.watchdog_stops += 1
        linear = 0.0 if stale or self.stop_latched else self.command_linear_mps
        angular = 0.0 if stale or self.stop_latched else self.command_angular_rad_s

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
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = 3600.0
    cfg.route_chain_mode = True
    env = gym.make(args.task, cfg=cfg)
    raw_env = env.unwrapped
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
    bridge = AishaSimulationBridge(raw_env, control_period_s, sensor_positions)
    steps_completed = 0
    reset_detected = False
    try:
        env.reset()
        while simulation_app.is_running() and (args.max_steps <= 0 or steps_completed < args.max_steps):
            bridge.spin()
            if args.self_test:
                if 15 <= steps_completed < 75:
                    bridge.set_self_test_command(0.05)
                elif steps_completed == 75:
                    bridge.set_self_test_command(0.0)
            action = bridge.command_action()
            _, _, terminated, truncated, _ = env.step(action)
            bridge.simulation_time_s += control_period_s
            bridge.publish(steps_completed)
            steps_completed += 1
            if bool(torch.any(terminated | truncated).item()):
                reset_detected = True
                break
    finally:
        position = raw_env._local_xy()[0].detach().cpu().tolist()
        report = {
            "report_type": "administration_nav2_bridge",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "bridge_smoke_completed" if steps_completed > 0 else "bridge_not_exercised",
            "task": args.task,
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
                "command_watchdog_s": bridge.command_watchdog_s,
                "front_stop_trigger_m": bridge.front_stop_trigger_m,
                "front_stop_release_m": bridge.front_stop_release_m,
            },
            "events": {
                "rejected_reverse_commands": bridge.rejected_reverse_commands,
                "rejected_lateral_commands": bridge.rejected_lateral_commands,
                "watchdog_stops": bridge.watchdog_stops,
                "obstacle_stops": bridge.obstacle_stops,
                "episode_reset_gate_detected": reset_detected,
            },
            "final_position_xy_m": [round(float(value), 5) for value in position],
            "minimum_front_range_m": (
                round(bridge.minimum_front_range_m, 5)
                if math.isfinite(bridge.minimum_front_range_m)
                else None
            ),
            "learned_policy_coupled": False,
            "physical_release": False,
            "claim_boundary": (
                "This bridge run proves Isaac physics/ROS topic exchange only. It does not prove "
                "a live Nav2 mission, frozen-policy arbitration, protective safety coverage, "
                "stopping distance, sim-to-real performance, or physical deployment readiness."
            ),
        }
        report["passed"] = (
            steps_completed > 0
            and all(count > 0 for count in bridge.messages.values())
            and not reset_detected
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
