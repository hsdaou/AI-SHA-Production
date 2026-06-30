#!/usr/bin/env python3
"""
ROS2 Encoder Node - Reads 4 quadrature encoders via pigpio.
Displays a live-updating table in the terminal.

Publishes:
    /encoders/rpm      (Float64MultiArray) - [rpm1, rpm2, rpm3, rpm4]
    /encoders/position (Float64MultiArray) - [rad1, deg1, rad2, deg2, rad3, deg3, rad4, deg4]

Parameters:
    publish_rate - Publishing rate in Hz (default 50)
    encoder_ppr  - Pulses per revolution per channel (default 600)
"""

import math
import sys
import time
import threading

import pigpio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


ENCODER_PINS = [
    {'enc_a': 7,  'enc_b': 8},   # Encoder 1
    {'enc_a': 15, 'enc_b': 14},  # Encoder 2
    {'enc_a': 17, 'enc_b': 27},  # Encoder 3
    {'enc_a': 5,  'enc_b': 19},  # Encoder 4
]


class QuadratureEncoder:
    """Reads a single quadrature encoder using pigpio callbacks."""

    FORWARD = {0b0001, 0b0111, 0b1110, 0b1000}
    REVERSE = {0b0010, 0b1011, 0b1101, 0b0100}

    def __init__(self, pi, enc_a, enc_b):
        self.pi = pi
        self.enc_a = enc_a
        self.enc_b = enc_b

        pi.set_mode(enc_a, pigpio.INPUT)
        pi.set_mode(enc_b, pigpio.INPUT)
        pi.set_pull_up_down(enc_a, pigpio.PUD_UP)
        pi.set_pull_up_down(enc_b, pigpio.PUD_UP)

        self._lock = threading.Lock()
        self._position = 0
        self._last_a = pi.read(enc_a)
        self._last_b = pi.read(enc_b)
        self._prev_position = 0
        self._rpm = 0.0

        self._cb_a = pi.callback(enc_a, pigpio.EITHER_EDGE, self._on_edge)
        self._cb_b = pi.callback(enc_b, pigpio.EITHER_EDGE, self._on_edge)

    def _on_edge(self, gpio, level, tick):
        a = self.pi.read(self.enc_a)
        b = self.pi.read(self.enc_b)
        last_state = (self._last_a << 1) | self._last_b
        curr_state = (a << 1) | b
        transition = (last_state << 2) | curr_state

        with self._lock:
            if transition in self.FORWARD:
                self._position += 1
            elif transition in self.REVERSE:
                self._position -= 1

        self._last_a = a
        self._last_b = b

    def get_position(self):
        with self._lock:
            return self._position

    def compute_rpm(self, dt, cpr):
        pos = self.get_position()
        delta = pos - self._prev_position
        self._prev_position = pos
        if dt > 0:
            self._rpm = (delta / cpr / dt) * 60.0
        return self._rpm

    def cancel(self):
        self._cb_a.cancel()
        self._cb_b.cancel()


class EncoderNode(Node):
    def __init__(self):
        super().__init__('encoder_node')

        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('encoder_ppr', 600)

        publish_rate = self.get_parameter('publish_rate').value
        self.ppr = self.get_parameter('encoder_ppr').value
        self.cpr = self.ppr * 4

        self.pi = pigpio.pi()
        if not self.pi.connected:
            self.get_logger().fatal('Cannot connect to pigpio daemon! Run: sudo pigpiod')
            raise RuntimeError('pigpio daemon not running')

        self.encoders = []
        for i, pins in enumerate(ENCODER_PINS):
            enc = QuadratureEncoder(self.pi, pins['enc_a'], pins['enc_b'])
            self.encoders.append(enc)

        # Publishers
        self.pub_rpm = self.create_publisher(Float64MultiArray, '/encoders/rpm', 10)
        self.pub_pos = self.create_publisher(Float64MultiArray, '/encoders/position', 10)

        self._last_time = time.monotonic()
        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self._publish)

        # Display at 10 Hz
        self._display_divider = max(1, int(publish_rate / 10))
        self._cycle = 0

        self._rpms = [0.0] * 4
        self._rads = [0.0] * 4
        self._degs = [0.0] * 4
        self._counts = [0] * 4

        self._print_header()

    def _print_header(self):
        sys.stdout.write('\033[2J\033[H')  # clear screen, cursor home
        sys.stdout.write(
            '\033[1;36m'
            '  Encoder Node  |  PPR: {}  |  CPR: {}'
            '\033[0m\n'.format(self.ppr, self.cpr)
        )
        sys.stdout.write(
            '\033[1m'
            ' {:<10s} {:>10s} {:>12s} {:>12s} {:>10s}'
            '\033[0m\n'.format('Motor', 'RPM', 'Radians', 'Degrees', 'Counts')
        )
        sys.stdout.write('\033[90m' + '-' * 58 + '\033[0m\n')
        # Row 1: title, Row 2: columns, Row 3: separator
        # Data starts at row 4
        sys.stdout.flush()

    def _update_display(self):
        for i in range(4):
            sys.stdout.write('\033[{};1H'.format(4 + i))  # absolute row
            sys.stdout.write(
                ' {:<10s} {:>10.1f} {:>12.4f} {:>12.2f} {:>10d}\033[K'.format(
                    'Motor {}'.format(i + 1),
                    self._rpms[i],
                    self._rads[i],
                    self._degs[i],
                    self._counts[i],
                )
            )
        sys.stdout.write('\033[{};1H'.format(8))  # row after 4 motors
        sys.stdout.write('\033[90m' + '-' * 58 + '\033[0m\033[K')
        sys.stdout.flush()

    def _publish(self):
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0:
            return

        rpms = []
        position_data = []

        for i, enc in enumerate(self.encoders):
            rpm = enc.compute_rpm(dt, self.cpr)
            rpms.append(rpm)

            pos_counts = enc.get_position()
            angle_revs = pos_counts / self.cpr
            angle_rad = angle_revs * 2.0 * math.pi
            angle_deg = angle_revs * 360.0

            position_data.append(angle_rad)
            position_data.append(angle_deg)

            self._rpms[i] = rpm
            self._rads[i] = angle_rad
            self._degs[i] = angle_deg
            self._counts[i] = pos_counts

        rpm_msg = Float64MultiArray()
        rpm_msg.data = rpms
        self.pub_rpm.publish(rpm_msg)

        pos_msg = Float64MultiArray()
        pos_msg.data = position_data
        self.pub_pos.publish(pos_msg)

        self._cycle += 1
        if self._cycle >= self._display_divider:
            self._cycle = 0
            self._update_display()

    def destroy_node(self):
        sys.stdout.write('\n\n')
        self.get_logger().info('Shutting down encoder node...')
        for enc in self.encoders:
            enc.cancel()
        self.pi.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EncoderNode()
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
