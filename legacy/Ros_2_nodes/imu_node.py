#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Float32

import board
import busio
import adafruit_bno055


class BNO055Node(Node):

    def __init__(self):
        super().__init__('imu_node')

        # I2C + sensor init
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

        # Publishers
        self.temp_pub = self.create_publisher(Float32, 'imu/temperature', 10)
        self.accel_pub = self.create_publisher(Imu, 'imu/accel', 10)
        self.gyro_pub = self.create_publisher(Imu, 'imu/gyro', 10)
        self.mag_pub = self.create_publisher(MagneticField, 'imu/mag', 10)

        # Timer
        self.timer = self.create_timer(0.05, self.publish_data)  # 20 Hz

        self.get_logger().info('IMU node started')

    def publish_data(self):
        # Temperature
        if self.sensor.temperature is not None:
            temp_msg = Float32()
            temp_msg.data = float(self.sensor.temperature)
            self.temp_pub.publish(temp_msg)

        # Accelerometer
        if self.sensor.acceleration is not None:
            accel_msg = Imu()
            accel_msg.linear_acceleration.x = self.sensor.acceleration[0]
            accel_msg.linear_acceleration.y = self.sensor.acceleration[1]
            accel_msg.linear_acceleration.z = self.sensor.acceleration[2]
            self.accel_pub.publish(accel_msg)

        # Gyroscope
        if self.sensor.gyro is not None:
            gyro_msg = Imu()
            gyro_msg.angular_velocity.x = self.sensor.gyro[0]
            gyro_msg.angular_velocity.y = self.sensor.gyro[1]
            gyro_msg.angular_velocity.z = self.sensor.gyro[2]
            self.gyro_pub.publish(gyro_msg)

        # Magnetometer
        if self.sensor.magnetic is not None:
            mag_msg = MagneticField()
            mag_msg.magnetic_field.x = self.sensor.magnetic[0]
            mag_msg.magnetic_field.y = self.sensor.magnetic[1]
            mag_msg.magnetic_field.z = self.sensor.magnetic[2]
            self.mag_pub.publish(mag_msg)


def main():
    rclpy.init()
    node = BNO055Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
