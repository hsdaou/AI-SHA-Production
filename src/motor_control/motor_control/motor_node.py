#!/usr/bin/env python3
"""
ROS2 Motor Driver Node - Controls 4 DC motors via 2x HC-160A S2 drivers.
Uses PID with encoder RPM feedback from the encoder_node.
Displays a live-updating table in the terminal.

Subscribes to:
    /cmd_rpm      (Float64MultiArray) - target RPM for [motor1, motor2, motor3, motor4]
    /encoders/rpm (Float64MultiArray) - measured RPM from encoder_node

Publishes:
    /motor_status (std_msgs/String) - JSON status of all motors

Parameters:
    pid_kp, pid_ki, pid_kd - PID gains
    pwm_frequency          - PWM frequency in Hz (default 20000)
    control_rate           - PID loop rate in Hz (default 50)
    max_rpm                - Maximum allowed RPM (default 5000)
"""

import json
import sys
import time

import pigpio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


MOTOR_PINS = [
    {'dir_a': 22, 'dir_b': 24, 'pwm': 13},  # Motor 1 - Driver 1, Ch A
    {'dir_a': 6,  'dir_b': 4,  'pwm': 10},  # Motor 2 - Driver 1, Ch B
    {'dir_a': 11, 'dir_b': 26, 'pwm': 12},  # Motor 3 - Driver 2, Ch A
    {'dir_a': 9,  'dir_b': 20, 'pwm': 16},  # Motor 4 - Driver 2, Ch B
]


class MotorChannel:
    """Controls a single motor channel on an HC-160A S2 driver."""

    def __init__(self, pi, pins, pwm_freq):
        self.pi = pi
        self.dir_a = pins['dir_a']
        self.dir_b = pins['dir_b']
        self.pwm_pin = pins['pwm']

        pi.set_mode(self.dir_a, pigpio.OUTPUT)
        pi.set_mode(self.dir_b, pigpio.OUTPUT)
        pi.write(self.dir_a, 0)
        pi.write(self.dir_b, 0)

        pi.set_PWM_frequency(self.pwm_pin, pwm_freq)
        pi.set_PWM_range(self.pwm_pin, 1000)
        pi.set_PWM_dutycycle(self.pwm_pin, 0)

        self.target_rpm = 0.0
        self.measured_rpm = 0.0
        self.duty = 0.0
        self._integral = 0.0
        self._prev_error = 0.0

    def set_target(self, rpm):
        self.target_rpm = rpm

    def set_measured_rpm(self, rpm):
        self.measured_rpm = rpm

    def update_pid(self, kp, ki, kd, dt, max_rpm):
        target = max(min(self.target_rpm, max_rpm), -max_rpm)
        error = target - self.measured_rpm

        self._integral += error * dt
        self._integral = max(min(self._integral, 1000.0), -1000.0)

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = kp * error + ki * self._integral + kd * derivative
        self.duty = max(min(abs(output), 1000.0), 0.0)

        if target > 0:
            self.pi.write(self.dir_a, 1)
            self.pi.write(self.dir_b, 0)
        elif target < 0:
            self.pi.write(self.dir_a, 0)
            self.pi.write(self.dir_b, 1)
        else:
            self.pi.write(self.dir_a, 0)
            self.pi.write(self.dir_b, 0)
            self.duty = 0.0
            self._integral = 0.0

        self.pi.set_PWM_dutycycle(self.pwm_pin, int(self.duty))

    def stop(self):
        self.pi.set_PWM_dutycycle(self.pwm_pin, 0)
        self.pi.write(self.dir_a, 0)
        self.pi.write(self.dir_b, 0)

    def get_status(self):
        return {
            'target_rpm': round(self.target_rpm, 1),
            'measured_rpm': round(self.measured_rpm, 1),
            'duty': round(self.duty, 1),
        }


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__('motor_driver_node')

        self.declare_parameter('pid_kp', 1.0)
        self.declare_parameter('pid_ki', 0.5)
        self.declare_parameter('pid_kd', 0.01)
        self.declare_parameter('pwm_frequency', 20000)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('max_rpm', 5000.0)

        pwm_freq = self.get_parameter('pwm_frequency').value
        self.control_rate = self.get_parameter('control_rate').value

        self.pi = pigpio.pi()
        if not self.pi.connected:
            self.get_logger().fatal('Cannot connect to pigpio daemon! Run: sudo pigpiod')
            raise RuntimeError('pigpio daemon not running')

        self.declare_parameter('default_rpm', 50.0)
        default_rpm = self.get_parameter('default_rpm').value

        self.motors = []
        for i, pins in enumerate(MOTOR_PINS):
            motor = MotorChannel(self.pi, pins, pwm_freq)
            motor.set_target(default_rpm)
            self.motors.append(motor)

        self.create_subscription(
            Float64MultiArray, '/cmd_rpm', self._on_cmd_rpm, 10
        )
        self.create_subscription(
            Float64MultiArray, '/encoders/rpm', self._on_encoder_rpm, 10
        )

        self.pub_status = self.create_publisher(String, '/motor_status', 10)

        self._last_time = time.monotonic()
        period = 1.0 / self.control_rate
        self.timer = self.create_timer(period, self._control_loop)
        self._status_counter = 0

        # Display at ~10 Hz
        self._display_divider = max(1, int(self.control_rate / 10))
        self._display_cycle = 0
        self._print_header(pwm_freq)

    def _print_header(self, pwm_freq):
        sys.stdout.write('\033[2J\033[H')  # clear screen, cursor home
        sys.stdout.write(
            '\033[1;32m'
            '  Motor Driver Node  |  PID @ {} Hz  |  PWM @ {} Hz'
            '\033[0m\n'.format(int(self.control_rate), pwm_freq)
        )
        sys.stdout.write(
            '\033[1m'
            ' {:<10s} {:>12s} {:>12s} {:>10s} {:>8s}'
            '\033[0m\n'.format('Motor', 'Target RPM', 'Actual RPM', 'Duty %', 'Dir')
        )
        sys.stdout.write('\033[90m' + '-' * 56 + '\033[0m\n')
        # Row 1: title, Row 2: columns, Row 3: separator
        # Data starts at row 4
        sys.stdout.flush()

    def _update_display(self):
        for i, m in enumerate(self.motors):
            duty_pct = m.duty / 10.0
            if m.target_rpm > 0:
                direction = 'FWD'
            elif m.target_rpm < 0:
                direction = 'REV'
            else:
                direction = '---'
            sys.stdout.write('\033[{};1H'.format(4 + i))  # absolute row
            sys.stdout.write(
                ' {:<10s} {:>12.1f} {:>12.1f} {:>9.1f}% {:>8s}\033[K'.format(
                    'Motor {}'.format(i + 1),
                    m.target_rpm,
                    m.measured_rpm,
                    duty_pct,
                    direction,
                )
            )
        sys.stdout.write('\033[{};1H'.format(8))  # row after 4 motors
        sys.stdout.write('\033[90m' + '-' * 56 + '\033[0m\033[K')
        sys.stdout.flush()

    def _on_cmd_rpm(self, msg):
        max_rpm = self.get_parameter('max_rpm').value
        for i, motor in enumerate(self.motors):
            if i < len(msg.data):
                motor.set_target(max(min(msg.data[i], max_rpm), -max_rpm))
            else:
                motor.set_target(0.0)

    def _on_encoder_rpm(self, msg):
        for i, motor in enumerate(self.motors):
            if i < len(msg.data):
                motor.set_measured_rpm(msg.data[i])

    def _control_loop(self):
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0:
            return

        kp = self.get_parameter('pid_kp').value
        ki = self.get_parameter('pid_ki').value
        kd = self.get_parameter('pid_kd').value
        max_rpm = self.get_parameter('max_rpm').value

        for motor in self.motors:
            motor.update_pid(kp, ki, kd, dt, max_rpm)

        # Publish JSON status at ~5 Hz
        self._status_counter += 1
        if self._status_counter >= 10:
            self._status_counter = 0
            status_msg = String()
            status_msg.data = json.dumps({
                f'motor_{i+1}': m.get_status() for i, m in enumerate(self.motors)
            })
            self.pub_status.publish(status_msg)

        # Update terminal display at ~10 Hz
        self._display_cycle += 1
        if self._display_cycle >= self._display_divider:
            self._display_cycle = 0
            self._update_display()

    def destroy_node(self):
        sys.stdout.write('\n\n')
        self.get_logger().info('Shutting down motor driver node...')
        for motor in self.motors:
            motor.stop()
        self.pi.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
