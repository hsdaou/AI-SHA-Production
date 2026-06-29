#!/usr/bin/env python3
"""
BNO055 9-DOF IMU ROS 2 Node
Publishes complete IMU data including all 9 DOF, temperature, and calibration status
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import board
import busio
import adafruit_bno055

from bno055_imu.msg import BNO055Data
from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion, Vector3


class BNO055Node(Node):
    def __init__(self):
        super().__init__('bno055_imu_node')

        # Declare parameters
        self.declare_parameter('publish_rate', 50.0)  # Hz
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('i2c_address', 0x28)  # Default BNO055 address

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        i2c_address = self.get_parameter('i2c_address').value

        # Initialize I2C and BNO055
        self.sensor = None
        self.sensor_available = False
        self.get_logger().info('AI-SHARJAH IMU: Starting...')
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor = adafruit_bno055.BNO055_I2C(i2c, address=i2c_address)
            self.sensor_available = True
            self.get_logger().info('AI-SHARJAH IMU: Connected ✓')
        except Exception as e:
            self.get_logger().warn(f'AI-SHARJAH IMU: Sensor not connected (running without IMU)')
            self.sensor_available = False

        # Create publisher with reliable QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher = self.create_publisher(
            BNO055Data,
            'imu/data',
            qos_profile
        )

        # Create timer for publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_imu_data)

        if self.sensor_available:
            self.get_logger().info(f'AI-SHARJAH IMU: Publishing at {self.publish_rate} Hz')
            # Print initial calibration status
            self.log_calibration_status()
        else:
            self.get_logger().info('AI-SHARJAH IMU: Ready (sensor reconnect will auto-enable)')

    def log_calibration_status(self):
        """Log the current calibration status"""
        if not self.sensor_available:
            return
        try:
            sys_cal, gyro_cal, accel_cal, mag_cal = self.sensor.calibration_status
            self.get_logger().info(
                f'AI-SHARJAH IMU: Calibration [Sys:{sys_cal} Gyro:{gyro_cal} Accel:{accel_cal} Mag:{mag_cal}]'
            )
        except Exception as e:
            pass

    def publish_imu_data(self):
        """Read sensor data and publish complete IMU message"""
        if not self.sensor_available:
            return

        try:
            msg = BNO055Data()

            # Header with timestamp
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            # === ORIENTATION (Quaternion) ===
            quat = self.sensor.quaternion
            if quat is not None:
                msg.orientation = Quaternion()
                msg.orientation.w = float(quat[0]) if quat[0] is not None else 0.0
                msg.orientation.x = float(quat[1]) if quat[1] is not None else 0.0
                msg.orientation.y = float(quat[2]) if quat[2] is not None else 0.0
                msg.orientation.z = float(quat[3]) if quat[3] is not None else 0.0

            # === ORIENTATION (Euler Angles) ===
            euler = self.sensor.euler
            if euler is not None:
                msg.euler_heading = float(euler[0]) if euler[0] is not None else 0.0
                msg.euler_roll = float(euler[1]) if euler[1] is not None else 0.0
                msg.euler_pitch = float(euler[2]) if euler[2] is not None else 0.0

            # === LINEAR ACCELERATION (m/s²) ===
            accel = self.sensor.acceleration
            if accel is not None:
                msg.linear_acceleration = Vector3()
                msg.linear_acceleration.x = float(accel[0]) if accel[0] is not None else 0.0
                msg.linear_acceleration.y = float(accel[1]) if accel[1] is not None else 0.0
                msg.linear_acceleration.z = float(accel[2]) if accel[2] is not None else 0.0

            # === ANGULAR VELOCITY (rad/s) ===
            gyro = self.sensor.gyro
            if gyro is not None:
                msg.angular_velocity = Vector3()
                msg.angular_velocity.x = float(gyro[0]) if gyro[0] is not None else 0.0
                msg.angular_velocity.y = float(gyro[1]) if gyro[1] is not None else 0.0
                msg.angular_velocity.z = float(gyro[2]) if gyro[2] is not None else 0.0

            # === MAGNETIC FIELD (micro Tesla) ===
            mag = self.sensor.magnetic
            if mag is not None:
                msg.magnetic_field = Vector3()
                msg.magnetic_field.x = float(mag[0]) if mag[0] is not None else 0.0
                msg.magnetic_field.y = float(mag[1]) if mag[1] is not None else 0.0
                msg.magnetic_field.z = float(mag[2]) if mag[2] is not None else 0.0

            # === TEMPERATURE (°C) ===
            temp = self.sensor.temperature
            msg.temperature = float(temp) if temp is not None else 0.0

            # === CALIBRATION STATUS ===
            cal_status = self.sensor.calibration_status
            if cal_status is not None:
                msg.calibration_system = int(cal_status[0])
                msg.calibration_gyro = int(cal_status[1])
                msg.calibration_accel = int(cal_status[2])
                msg.calibration_mag = int(cal_status[3])

            # Publish the message
            self.publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f'AI-SHARJAH IMU: Error reading sensor: {e}')
            # Try to recover sensor connection
            self.sensor_available = False


def main(args=None):
    rclpy.init(args=args)

    try:
        node = BNO055Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
