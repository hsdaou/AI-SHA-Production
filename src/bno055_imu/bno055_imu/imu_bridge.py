#!/usr/bin/env python3
"""
Bridge node to convert BNO055Data to standard sensor_msgs/Imu for RViz2
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from bno055_imu.msg import BNO055Data


class IMUBridge(Node):
    def __init__(self):
        super().__init__('imu_bridge')

        self.get_logger().info('IMU Bridge: Starting...')

        # Subscribe to custom BNO055 data
        self.subscription = self.create_subscription(
            BNO055Data,
            'imu/data',
            self.data_callback,
            10
        )

        # Publish standard IMU message
        self.publisher = self.create_publisher(
            Imu,
            'imu/data_raw',
            10
        )

        self.get_logger().info('IMU Bridge: Ready (converting /imu/data -> /imu/data_raw)')

    def data_callback(self, msg):
        """Convert BNO055Data to sensor_msgs/Imu"""
        imu_msg = Imu()

        # Copy header
        imu_msg.header = msg.header

        # Copy orientation (quaternion)
        imu_msg.orientation = msg.orientation

        # Copy angular velocity
        imu_msg.angular_velocity = msg.angular_velocity

        # Copy linear acceleration
        imu_msg.linear_acceleration = msg.linear_acceleration

        # Set covariance matrices (unknown, set first element to -1 or use small values)
        # For now, using small diagonal values indicating good confidence
        imu_msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        imu_msg.angular_velocity_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        imu_msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        # Publish
        self.publisher.publish(imu_msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = IMUBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
