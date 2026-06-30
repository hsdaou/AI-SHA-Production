#!/usr/bin/env python3
"""Throttle laser scans to reduce SLAM processing load."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanThrottle(Node):
    def __init__(self):
        super().__init__('scan_throttle')

        self.declare_parameter('rate', 2.0)  # 2 Hz output

        rate = self.get_parameter('rate').value
        self.period = 1.0 / rate

        self.sub = self.create_subscription(
            LaserScan, '/scan_raw', self.scan_callback, 10)
        self.pub = self.create_publisher(LaserScan, '/scan', 10)

        self.last_pub_time = self.get_clock().now()
        self.get_logger().info(f'Throttling scans to {rate} Hz')

    def scan_callback(self, msg):
        now = self.get_clock().now()
        elapsed = (now - self.last_pub_time).nanoseconds / 1e9

        if elapsed >= self.period:
            self.pub.publish(msg)
            self.last_pub_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ScanThrottle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
