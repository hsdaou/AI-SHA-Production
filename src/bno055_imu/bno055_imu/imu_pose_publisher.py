#!/usr/bin/env python3
"""
IMU Pose Publisher for RViz Visualization
Converts IMU data to PoseStamped for easy visualization
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu


class IMUPosePublisher(Node):
    def __init__(self):
        super().__init__('imu_pose_publisher')

        self.get_logger().info('IMU Pose Publisher: Starting...')

        # Subscribe to standard IMU message
        self.subscription = self.create_subscription(
            Imu,
            'imu/data_raw',
            self.imu_callback,
            10
        )

        # Publish pose for RViz
        self.publisher = self.create_publisher(
            PoseStamped,
            'imu/pose',
            10
        )

        self.get_logger().info('IMU Pose Publisher: Ready (/imu/data_raw -> /imu/pose)')

    def imu_callback(self, msg):
        """Convert IMU orientation to PoseStamped"""
        pose_msg = PoseStamped()

        # Copy header
        pose_msg.header = msg.header

        # Set position at origin (we only care about orientation)
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0

        # Copy orientation from IMU
        pose_msg.pose.orientation = msg.orientation

        # Publish
        self.publisher.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = IMUPosePublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
