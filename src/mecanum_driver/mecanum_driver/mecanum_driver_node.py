#!/usr/bin/env python3
"""
Mecanum Robot Driver Node (ROS2 Humble)

Subscribes to /cmd_vel and converts to 4 mecanum wheel speeds
using inverse kinematics, then sends to Arduino over serial.

When encoder feedback is available (serial line "E <fl> <fr> <rl> <rr>"),
applies Mecanum forward kinematics to compute robot velocity and integrates
into a cumulative pose, publishing nav_msgs/Odometry and odom→base_link TF.

Mecanum Inverse Kinematics (cmd_vel → wheel speeds):
  v_fl = vx - vy - (lx + ly) * wz
  v_fr = vx + vy + (lx + ly) * wz
  v_rl = vx + vy - (lx + ly) * wz
  v_rr = vx - vy + (lx + ly) * wz

Mecanum Forward Kinematics (wheel speeds → robot velocity):
  vx = (R/4)( ωfl + ωfr + ωrl + ωrr)
  vy = (R/4)(-ωfl + ωfr + ωrl - ωrr)
  wz = (R/4(lx+ly))(-ωfl + ωfr - ωrl + ωrr)
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster
import serial
import time
import threading


class MecanumDriverNode(Node):
    def __init__(self):
        super().__init__('mecanum_driver')

        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_radius', 0.03)
        self.declare_parameter('robot_width', 0.20)
        self.declare_parameter('robot_length', 0.15)
        self.declare_parameter('max_motor_pwm', 255)
        # max_wheel_speed in rad/s — default 1.0 m/s / 0.03 m ≈ 33.3 rad/s
        self.declare_parameter('max_wheel_speed', 33.3)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('serial_timeout', 0.1)
        # Right-side motors are physically mounted 180° opposite on most
        # chassis.  Set to True to negate FR/RR PWM in software (instead of
        # swapping IN1/IN2 wires at the motor driver).
        self.declare_parameter('invert_right_side', False)
        # Minimum PWM below which motors stall due to static friction.
        # Nav stack micro-adjustments (PWM 5-30) will just make motors whine.
        # Set via hardware testing: slowly raise PWM until wheels just start
        # turning under load.  0 disables deadband compensation.
        self.declare_parameter('min_motor_pwm', 0)
        # Encoder ticks per full wheel revolution (after 4× quadrature decoding).
        # Set to 0 to disable encoder odometry (e.g. if encoders are not installed).
        # The existing encoder_node.py uses PPR=600 → CPR=2400 (4× quadrature).
        # If Arduino sends raw quadrature-decoded ticks, use 2400.
        # If Arduino sends per-channel pulses, use 600 and let this node multiply by 4.
        self.declare_parameter('encoder_cpr', 2400)
        # Enable/disable publishing odom from encoder data.  When False, the
        # encoder parser still runs (for logging), but no Odometry messages or
        # TF transforms are published.  Use False when running rf2o or dummy_odom
        # as the odom source to avoid conflicting odom→base_link transforms.
        self.declare_parameter('publish_odom', False)

        # Read parameters — type-cast every numeric value to guard against
        # LaunchConfiguration passing strings.  ROS 2 launch substitutions
        # always resolve to strings; without explicit casts, rclpy may throw
        # ParameterException or silently store the wrong type.
        self.serial_port = str(self.get_parameter('serial_port').value)
        self.baud_rate = int(self.get_parameter('baud_rate').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.robot_width = float(self.get_parameter('robot_width').value)
        self.robot_length = float(self.get_parameter('robot_length').value)
        self.max_pwm = int(self.get_parameter('max_motor_pwm').value)
        self.max_wheel_speed = float(self.get_parameter('max_wheel_speed').value)
        self.cmd_vel_timeout = float(self.get_parameter('cmd_vel_timeout').value)
        self.serial_timeout = float(self.get_parameter('serial_timeout').value)
        self.invert_right = -1 if self.get_parameter('invert_right_side').value else 1
        self.min_pwm = int(self.get_parameter('min_motor_pwm').value)
        self.encoder_cpr = int(self.get_parameter('encoder_cpr').value)
        self.publish_odom = bool(self.get_parameter('publish_odom').value)

        # Half-widths for kinematics
        self.lx = self.robot_width / 2.0
        self.ly = self.robot_length / 2.0

        # ── Odometry publisher + TF broadcaster (encoder-based) ───────────
        # Only active when publish_odom=True AND encoder_cpr > 0.
        # Publishes nav_msgs/Odometry on /odom and broadcasts odom→base_link TF.
        # When disabled, rf2o_laser_odometry or dummy_odom provides the transform.
        self._odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        # Cumulative pose from forward kinematics integration
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._prev_encoder_ticks = None  # Set on first encoder reading
        self._last_encoder_time = None

        # Serial connection
        self.ser = None
        self.serial_lock = threading.Lock()
        self._reconnecting = False  # True while background reconnect is in progress
        self.connect_serial()

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # ── Alternative encoder source: RPi encoder_node via ROS topic ────
        # The Arduino Uno has no free pins for encoders (all 12 digital pins
        # are used by motor drivers).  encoder_node.py on the RPi reads
        # quadrature encoders via pigpio and publishes to /encoders/position
        # as Float64MultiArray: [rad1, deg1, rad2, deg2, rad3, deg3, rad4, deg4].
        # This subscription converts those radians to equivalent tick counts
        # and feeds them into the same FK pipeline as the serial "E" protocol.
        # Both sources can coexist — whichever arrives first initializes the
        # baseline; subsequent readings from either source update odometry.
        if self.publish_odom and self.encoder_cpr > 0:
            from std_msgs.msg import Float64MultiArray as F64MA
            self._encoder_pos_sub = self.create_subscription(
                F64MA, '/encoders/position', self._on_encoder_position, 10)

        # Publisher (debug)
        self.wheel_pub = self.create_publisher(
            Float32MultiArray, 'wheel_speeds', 10)

        # ── Serial rate-limiter ─────────────────────────────────────────────
        # cmd_vel_callback stores the latest PWM values but does NOT write
        # to serial directly.  A 20 Hz timer flushes the latest command to
        # the Arduino.  This prevents flooding the UART when nav2 publishes
        # cmd_vel at 30-50 Hz — the Arduino can't parse commands that fast,
        # causing buffer overflows, garbled frames, and erratic motor behaviour.
        # 20 Hz is well above human-perceptible latency (~50 ms) and matches
        # the typical Arduino PID loop rate.
        self._latest_pwm = (0, 0, 0, 0)  # (FL, FR, RL, RR)
        self._pwm_dirty = False  # True when cmd_vel has new data to flush
        self._serial_flush_timer = self.create_timer(
            1.0 / 20.0, self._flush_serial)  # 20 Hz

        # Watchdog timer
        self.last_cmd_time = time.time()
        self.is_moving = False
        # Redundant stop counter: send stop N times after timeout to survive
        # dropped USB serial packets, then stop spamming.  10 ticks × 0.1s = 1s
        # of redundant stops — enough to guarantee at least one gets through.
        self._stop_sends_remaining = 0
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        # Non-blocking serial reconnect timer (1 Hz) — only active when disconnected.
        # This replaces the old blocking connect_serial() call inside
        # send_motor_command(), which could freeze the ROS 2 executor for up to 15s.
        # 1 Hz is sufficient: Arduino bootloader needs ~2s after port open anyway.
        self._reconnect_timer = self.create_timer(1.0, self._reconnect_tick)

        # Encoder state — updated by _parse_encoder_line() in serial_reader thread.
        # Will hold cumulative tick counts once Arduino firmware supports encoders.
        self._last_encoder_ticks = (0, 0, 0, 0)  # (FL, FR, RL, RR)

        # Serial reader thread
        self.running = True
        self.reader_thread = threading.Thread(
            target=self.serial_reader, daemon=True)
        self.reader_thread.start()

        self.get_logger().info(
            f'Mecanum driver started on {self.serial_port} @ {self.baud_rate}')
        self.get_logger().info(
            f'Robot dims: W={self.robot_width}m, L={self.robot_length}m, '
            f'wheel_r={self.wheel_radius}m')

    def connect_serial(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.ser = serial.Serial(
                    port=self.serial_port,
                    baudrate=self.baud_rate,
                    timeout=self.serial_timeout)
                time.sleep(2.0)
                self.ser.reset_input_buffer()
                self.get_logger().info('Serial connection established')
                return
            except serial.SerialException as e:
                self.get_logger().warn(
                    f'Serial attempt {attempt+1}/{max_retries} failed: {e}')
                time.sleep(1.0)
        self.get_logger().error(
            'Could not open serial port! Motors will not work.')

    def cmd_vel_callback(self, msg: Twist):
        self.last_cmd_time = time.time()
        self.is_moving = True
        # Cancel any pending watchdog stop commands — a new velocity command
        # supersedes the previous timeout.  Without this, the stale counter
        # persists (though the watchdog's outer timeout check prevents it from
        # firing while cmd_vel is active, this makes the intent explicit).
        self._stop_sends_remaining = 0

        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        # Mecanum Inverse Kinematics
        k = self.lx + self.ly
        v_fl = vx - vy - k * wz
        v_fr = vx + vy + k * wz
        v_rl = vx + vy - k * wz
        v_rr = vx - vy + k * wz

        # Convert linear surface speed (m/s) → angular velocity (rad/s)
        # so that speed_to_pwm maps physical motor effort, not just m/s.
        omega_fl = v_fl / self.wheel_radius
        omega_fr = v_fr / self.wheel_radius
        omega_rl = v_rl / self.wheel_radius
        omega_rr = v_rr / self.wheel_radius

        # Apply right-side motor inversion if chassis mirrors the motors
        omega_fr *= self.invert_right
        omega_rr *= self.invert_right

        # Normalize all wheels proportionally if any exceeds max_wheel_speed.
        # Independent clamping in speed_to_pwm would destroy the speed ratios
        # between wheels, causing the robot to veer off-course at high speeds.
        max_abs = max(abs(omega_fl), abs(omega_fr), abs(omega_rl), abs(omega_rr))
        if max_abs > self.max_wheel_speed:
            scale = self.max_wheel_speed / max_abs
            omega_fl *= scale
            omega_fr *= scale
            omega_rl *= scale
            omega_rr *= scale

        # Convert to PWM — store for the 20 Hz serial flush timer.
        # Do NOT call send_motor_command() here; writing to serial on every
        # cmd_vel would flood the Arduino UART at nav2's publish rate (30-50 Hz).
        pwm_fl = self.speed_to_pwm(omega_fl)
        pwm_fr = self.speed_to_pwm(omega_fr)
        pwm_rl = self.speed_to_pwm(omega_rl)
        pwm_rr = self.speed_to_pwm(omega_rr)

        self._latest_pwm = (pwm_fl, pwm_fr, pwm_rl, pwm_rr)
        self._pwm_dirty = True

        # Publish debug
        wheel_msg = Float32MultiArray()
        wheel_msg.data = [float(v_fl), float(v_fr),
                          float(v_rl), float(v_rr)]
        self.wheel_pub.publish(wheel_msg)

    def speed_to_pwm(self, speed_rad_s: float) -> int:
        """Map angular velocity (rad/s) to a PWM value in [-255, 255].

        Applies deadband compensation: if min_motor_pwm > 0, the lowest
        non-zero output is min_motor_pwm (not 1).  This prevents the nav
        stack's micro-adjustments from stalling the motors due to static
        friction, which ruins goal-reaching precision.
        """
        if abs(speed_rad_s) < 0.01:
            return 0
        speed_rad_s = max(-self.max_wheel_speed,
                          min(self.max_wheel_speed, speed_rad_s))
        # Map to [min_pwm, max_pwm] range to skip the motor deadband
        usable_range = self.max_pwm - self.min_pwm
        pwm = int((abs(speed_rad_s) / self.max_wheel_speed) * usable_range) + self.min_pwm
        pwm = min(255, pwm)
        return pwm if speed_rad_s > 0 else -pwm

    def _flush_serial(self):
        """20 Hz timer: send the latest PWM command to the Arduino.

        Only writes if cmd_vel_callback flagged new data (_pwm_dirty).
        This decouples the nav2 publish rate (30-50 Hz) from the serial
        write rate (20 Hz), preventing Arduino UART buffer overflows.
        """
        if not self._pwm_dirty:
            return
        self._pwm_dirty = False
        fl, fr, rl, rr = self._latest_pwm
        self.send_motor_command(fl, fr, rl, rr)

    def _reconnect_tick(self):
        """Non-blocking reconnect attempt — runs on a ROS 2 Timer (1 Hz).

        Tries a single serial open per tick.  Avoids blocking sleeps so
        the SingleThreadedExecutor stays responsive for cmd_vel and the
        watchdog.  The Arduino bootloader resets on port open (~2s); we
        simply allow the first few writes to fail silently — the watchdog
        will send stop commands once the port stabilises on a later tick.

        Thread safety: This callback, cmd_vel_callback, and watchdog_callback
        all run on the same SingleThreadedExecutor — no lock needed between
        them.  The only concurrent thread is serial_reader, which uses a
        local snapshot (``ser = self.ser``) to safely handle self.ser being
        reassigned mid-read; AttributeError/SerialException are caught.
        """
        if self.ser is not None and self.ser.is_open:
            return  # already connected
        if self._reconnecting:
            return  # another tick is already trying
        self._reconnecting = True
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.serial_timeout)
            # No blocking sleep — the Arduino bootloader takes ~2s after
            # port open, but write failures during that window are harmless
            # (send_motor_command catches SerialException and sets ser=None).
            # The next successful tick will find ser.is_open and return.
            self.ser.reset_input_buffer()
            self.get_logger().info('Serial reconnected')
        except serial.SerialException:
            self.ser = None  # will retry next tick
        finally:
            self._reconnecting = False

    def send_motor_command(self, fl: int, fr: int, rl: int, rr: int):
        # Snapshot to local var — _reconnect_tick can set self.ser = None
        # between the check and .write(), causing AttributeError.
        ser = self.ser
        if ser is None or not ser.is_open:
            # Don't block — _reconnect_timer will restore the connection.
            return
        cmd = f"M {fl} {fr} {rl} {rr}\n"
        try:
            with self.serial_lock:
                ser.write(cmd.encode('utf-8'))
        except (serial.SerialException, AttributeError, OSError) as e:
            self.get_logger().error(f'Serial write failed: {e}')
            self.ser = None

    def send_stop(self):
        ser = self.ser
        if ser and ser.is_open:
            try:
                with self.serial_lock:
                    ser.write(b"S\n")
            except (serial.SerialException, AttributeError, OSError):
                pass

    def watchdog_callback(self):
        if time.time() - self.last_cmd_time > self.cmd_vel_timeout:
            if self.is_moving:
                # Arm the redundant stop counter on transition from moving→stopped.
                # 10 ticks × 0.1s = 1s of stop commands — enough to survive
                # multiple dropped USB serial packets without spamming forever.
                self._stop_sends_remaining = 10
                self.is_moving = False
                # Clear stale velocity state — without this, _flush_serial
                # could resurrect a pre-timeout cmd_vel command after the
                # watchdog sends stop (the "zombie command" race).
                self._latest_pwm = (0, 0, 0, 0)
                self._pwm_dirty = False
            if self._stop_sends_remaining > 0:
                self.send_motor_command(0, 0, 0, 0)
                self._stop_sends_remaining -= 1

    def _parse_encoder_line(self, line: str) -> bool:
        """Parse an encoder feedback line and compute odometry.

        ── Serial Protocol Contract ──────────────────────────────────────
        The Arduino firmware MUST send encoder tick counts in this format:

            E <fl_ticks> <fr_ticks> <rl_ticks> <rr_ticks>\\n

        Where:
          - "E" is the message type prefix (Encoder)
          - Each value is a signed integer: cumulative encoder ticks since
            Arduino boot (or last reset).  Signed because mecanum wheels
            can rotate in either direction.
          - Fields are space-separated, terminated by newline.
          - Example: "E 1024 -980 1015 -1002\\n"

        Other recognized prefixes:
          - "OK" — command acknowledgment (filtered before reaching here)
          - "ERR <msg>" — Arduino-side error (logged as warning)

        Returns True if the line was recognized and parsed, False otherwise.
        """
        parts = line.split()
        if not parts:
            return False

        if parts[0] == 'E' and len(parts) == 5:
            try:
                ticks = (int(parts[1]), int(parts[2]),
                         int(parts[3]), int(parts[4]))
            except ValueError:
                self.get_logger().warn(f'Malformed encoder line: {line}')
                return True  # recognized prefix, just bad data

            self._last_encoder_ticks = ticks
            self.get_logger().debug(
                f'Encoder ticks: FL={ticks[0]} FR={ticks[1]} '
                f'RL={ticks[2]} RR={ticks[3]}')

            # Compute and publish odometry if enabled
            if self.publish_odom and self.encoder_cpr > 0:
                self._update_odometry(ticks)

            return True

        if parts[0] == 'ERR':
            self.get_logger().warn(f'Arduino error: {line}')
            return True

        return False

    def _update_odometry(self, ticks: tuple):
        """Apply Mecanum forward kinematics and publish odom + TF.

        Converts delta encoder ticks → wheel angular displacements →
        robot-frame velocity (vx, vy, wz) → integrated pose (x, y, θ).

        Called from the serial_reader thread.  Publishing from a non-executor
        thread is safe in rclpy — publishers are thread-safe.  The only shared
        state is self._odom_{x,y,theta} and self._prev_encoder_ticks, which are
        only accessed from this thread (serial_reader), so no lock is needed.

        Mecanum Forward Kinematics:
          vx = (R/4)( ωfl + ωfr + ωrl + ωrr)
          vy = (R/4)(-ωfl + ωfr + ωrl - ωrr)
          wz = (R/4(lx+ly))(-ωfl + ωfr - ωrl + ωrr)
        """
        now = time.monotonic()

        # First reading — just store the baseline, can't compute deltas yet
        if self._prev_encoder_ticks is None:
            self._prev_encoder_ticks = ticks
            self._last_encoder_time = now
            return

        dt = now - self._last_encoder_time
        if dt <= 0:
            return
        self._last_encoder_time = now

        # Delta ticks since last reading
        d_fl = ticks[0] - self._prev_encoder_ticks[0]
        d_fr = ticks[1] - self._prev_encoder_ticks[1]
        d_rl = ticks[2] - self._prev_encoder_ticks[2]
        d_rr = ticks[3] - self._prev_encoder_ticks[3]
        self._prev_encoder_ticks = ticks

        # Convert tick deltas to wheel angular displacement (radians)
        # Each tick = (2π / encoder_cpr) radians
        rad_per_tick = (2.0 * math.pi) / self.encoder_cpr
        w_fl = d_fl * rad_per_tick
        w_fr = d_fr * rad_per_tick
        w_rl = d_rl * rad_per_tick
        w_rr = d_rr * rad_per_tick

        # Mecanum forward kinematics: wheel displacements → robot displacement
        R = self.wheel_radius
        k = self.lx + self.ly  # half-width + half-length

        # Robot-frame displacement (meters, radians)
        dx_robot = (R / 4.0) * (w_fl + w_fr + w_rl + w_rr)
        dy_robot = (R / 4.0) * (-w_fl + w_fr + w_rl - w_rr)
        dtheta = (R / (4.0 * k)) * (-w_fl + w_fr - w_rl + w_rr)

        # Integrate into world-frame pose
        # Use midpoint theta for better accuracy during rotation
        mid_theta = self._odom_theta + dtheta / 2.0
        cos_t = math.cos(mid_theta)
        sin_t = math.sin(mid_theta)
        self._odom_x += dx_robot * cos_t - dy_robot * sin_t
        self._odom_y += dx_robot * sin_t + dy_robot * cos_t
        self._odom_theta += dtheta

        # Robot-frame velocities (for Odometry twist)
        vx = dx_robot / dt
        vy = dy_robot / dt
        wz = dtheta / dt

        # Publish odom→base_link TF
        ros_now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = ros_now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self._odom_x
        t.transform.translation.y = self._odom_y
        t.transform.translation.z = 0.0
        # Quaternion from yaw (2D robot, roll=pitch=0)
        half_theta = self._odom_theta / 2.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(half_theta)
        t.transform.rotation.w = math.cos(half_theta)
        self._tf_broadcaster.sendTransform(t)

        # Publish nav_msgs/Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = ros_now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        # Pose
        odom_msg.pose.pose.position.x = self._odom_x
        odom_msg.pose.pose.position.y = self._odom_y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = t.transform.rotation
        # Twist (robot-frame velocities)
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.angular.z = wz
        self._odom_pub.publish(odom_msg)

    def _on_encoder_position(self, msg):
        """Convert RPi encoder_node position data to tick-equivalent for FK.

        encoder_node.py publishes Float64MultiArray on /encoders/position:
          [rad1, deg1, rad2, deg2, rad3, deg3, rad4, deg4]
        We extract the radian values (indices 0, 2, 4, 6) and convert to
        equivalent tick counts so _update_odometry() can process them
        identically to serial "E" lines.
        """
        if len(msg.data) < 8:
            return
        # Convert radians back to tick counts: ticks = radians / (2π / cpr)
        ticks_per_rad = self.encoder_cpr / (2.0 * math.pi)
        ticks = (
            int(msg.data[0] * ticks_per_rad),  # FL radians → ticks
            int(msg.data[2] * ticks_per_rad),  # FR radians → ticks
            int(msg.data[4] * ticks_per_rad),  # RL radians → ticks
            int(msg.data[6] * ticks_per_rad),  # RR radians → ticks
        )
        self._last_encoder_ticks = ticks
        self._update_odometry(ticks)

    def serial_reader(self):
        """Read Arduino feedback without holding serial_lock.

        Serial ports are full-duplex — reads and writes can happen
        concurrently without corruption.  Holding serial_lock during
        the blocking readline() would stall send_motor_command() if a
        partial line arrives (up to serial_timeout seconds).
        """
        while self.running:
            # Snapshot to local var — another thread may set self.ser = None
            # between the check and .in_waiting, causing AttributeError.
            ser = self.ser
            if ser and ser.is_open:
                try:
                    if ser.in_waiting:
                        line = ser.readline().decode(
                            'utf-8', errors='ignore').strip()
                        if not line or line.startswith('OK'):
                            continue
                        # Try structured protocol parsing first
                        if not self._parse_encoder_line(line):
                            self.get_logger().debug(
                                f'Arduino: {line}')
                except (serial.SerialException, AttributeError, OSError):
                    pass
            time.sleep(0.01)

    def destroy_node(self):
        self.running = False
        self.send_stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MecanumDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
