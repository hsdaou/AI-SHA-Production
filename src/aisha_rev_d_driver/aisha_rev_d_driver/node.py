"""ROS 2 encoder-odometry adapter for replay or read-only ZLAC telemetry.

No command subscription, control-register write, motor-enable operation or TF
broadcast exists in this node.  Physical odometry publication stays disabled
until the wheel scale, rolling radius and both encoder signs are verified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from .odometry import DifferentialGeometry, EncoderOdometry
from .transport import EncoderSample, ReadOnlyRs485Transport, ReplayTransport


class RevDEncoderAdapter(Node):
    def __init__(self) -> None:
        super().__init__("aisha_rev_d_encoder_adapter")
        self.declare_parameter("transport", "replay")
        self.declare_parameter("replay_path", "")
        self.declare_parameter("replay_period_s", 0.05)
        self.declare_parameter("serial_port", "/dev/aisha_zlac")
        self.declare_parameter("unit_address", 1)
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("serial_timeout_s", 0.10)
        self.declare_parameter("wheel_radius_m", 0.100)
        self.declare_parameter("wheel_track_m", 0.720)
        self.declare_parameter("encoder_counts_per_rev", 16384)
        self.declare_parameter("left_encoder_sign", 1)
        self.declare_parameter("right_encoder_sign", 1)
        self.declare_parameter("encoder_scale_verified", False)
        self.declare_parameter("rolling_radius_verified", False)
        self.declare_parameter("encoder_signs_verified", False)
        self.declare_parameter("hardware_label_verified", False)
        self.declare_parameter("motor_leads_isolated", False)
        self.declare_parameter("external_estop_verified", False)
        self.declare_parameter("publish_odom", True)
        self.declare_parameter("odom_topic", "/phase8b/replay/wheel_odom_raw")
        self.declare_parameter("telemetry_topic", "/wheel/encoder_telemetry")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self._transport_name = str(self.get_parameter("transport").value)
        if self._transport_name not in {"replay", "rs485_read_only"}:
            raise RuntimeError("transport must be replay or rs485_read_only")

        geometry = DifferentialGeometry(
            wheel_radius_m=float(self.get_parameter("wheel_radius_m").value),
            wheel_track_m=float(self.get_parameter("wheel_track_m").value),
            encoder_counts_per_rev=int(self.get_parameter("encoder_counts_per_rev").value),
            left_encoder_sign=int(self.get_parameter("left_encoder_sign").value),
            right_encoder_sign=int(self.get_parameter("right_encoder_sign").value),
        )
        self._odometry = EncoderOdometry(geometry)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._publish_odom_requested = bool(self.get_parameter("publish_odom").value)
        if self._transport_name == "replay" and self._odom_topic == "/wheel/odom_raw":
            raise RuntimeError(
                "Replay odometry must use an isolated topic and cannot feed /wheel/odom_raw"
            )
        calibration_verified = all(
            bool(self.get_parameter(name).value)
            for name in (
                "encoder_scale_verified",
                "rolling_radius_verified",
                "encoder_signs_verified",
            )
        )
        self._publish_odom = self._publish_odom_requested and (
            self._transport_name == "replay" or calibration_verified
        )
        if self._transport_name == "rs485_read_only":
            physical_read_gate = all(
                bool(self.get_parameter(name).value)
                for name in (
                    "hardware_label_verified",
                    "motor_leads_isolated",
                    "external_estop_verified",
                )
            )
            if not physical_read_gate:
                raise RuntimeError(
                    "RS485 read-only mode requires hardware_label_verified, "
                    "motor_leads_isolated and external_estop_verified"
                )
            if self._publish_odom_requested and not calibration_verified:
                self.get_logger().warning(
                    "Physical odometry publication suppressed: encoder scale, rolling "
                    "radius and signs are not all verified"
                )

        self._telemetry_publisher = self.create_publisher(
            String,
            str(self.get_parameter("telemetry_topic").value),
            10,
        )
        self._odom_publisher = None
        if self._publish_odom:
            self._odom_publisher = self.create_publisher(
                Odometry,
                self._odom_topic,
                20,
            )

        period_s = float(self.get_parameter("replay_period_s").value)
        if period_s <= 0.0:
            raise RuntimeError("replay_period_s must be positive")
        if self._transport_name == "replay":
            replay_path = str(self.get_parameter("replay_path").value)
            if not replay_path:
                replay_path = str(
                    Path(get_package_share_directory("aisha_rev_d_driver"))
                    / "config"
                    / "phase8b_encoder_replay.jsonl"
                )
            self._replay_iterator = iter(ReplayTransport(Path(replay_path)).samples())
            self._serial_transport = None
        else:
            self._replay_iterator = None
            self._serial_transport = ReadOnlyRs485Transport(
                str(self.get_parameter("serial_port").value),
                unit=int(self.get_parameter("unit_address").value),
                baud_rate=int(self.get_parameter("baud_rate").value),
                timeout_s=float(self.get_parameter("serial_timeout_s").value),
            )
        self._timer = self.create_timer(period_s, self._poll)
        self.get_logger().info(
            f"Phase 8B adapter transport={self._transport_name}; "
            f"odom_publish={self._publish_odom}; motor_write_available=false"
        )

    def destroy_node(self) -> bool:
        if self._serial_transport is not None:
            self._serial_transport.close()
        return super().destroy_node()

    def _poll(self) -> None:
        try:
            if self._replay_iterator is not None:
                sample = next(self._replay_iterator)
            else:
                assert self._serial_transport is not None
                sample = self._serial_transport.sample()
        except StopIteration:
            self._timer.cancel()
            self.get_logger().info("Phase 8B replay completed")
            return
        except Exception as exc:  # hardware polling must fail closed
            self.get_logger().error(f"encoder telemetry rejected: {exc}")
            return
        self._publish_sample(sample)

    def _publish_sample(self, sample: EncoderSample) -> None:
        telemetry = String()
        telemetry.data = json.dumps(
            {
                "source": sample.source,
                "left_count": sample.left_count,
                "right_count": sample.right_count,
                "left_rpm": sample.left_rpm,
                "right_rpm": sample.right_rpm,
                "status_word": sample.status_word,
                "left_fault": sample.left_fault,
                "right_fault": sample.right_fault,
                "motor_write_available": False,
            },
            separators=(",", ":"),
        )
        self._telemetry_publisher.publish(telemetry)
        estimate = self._odometry.update(
            sample.stamp_s,
            sample.left_count,
            sample.right_count,
        )
        if estimate is None or self._odom_publisher is None:
            return
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._odom_frame
        message.child_frame_id = self._base_frame
        message.pose.pose.position.x = estimate.pose.x_m
        message.pose.pose.position.y = estimate.pose.y_m
        half_yaw = 0.5 * estimate.pose.yaw_rad
        message.pose.pose.orientation.z = math.sin(half_yaw)
        message.pose.pose.orientation.w = math.cos(half_yaw)
        message.twist.twist.linear.x = estimate.linear_mps
        message.twist.twist.angular.z = estimate.angular_rad_s
        message.pose.covariance[0] = 0.02
        message.pose.covariance[7] = 0.02
        message.pose.covariance[35] = 0.04
        message.twist.covariance[0] = 0.03
        message.twist.covariance[35] = 0.05
        self._odom_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RevDEncoderAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
