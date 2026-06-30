#!/usr/bin/env python3
"""
Mecanum Robot Driver Node (ROS2 Humble)

Subscribes to /cmd_vel and converts to 4 mecanum wheel speeds using
inverse kinematics, then sends them to the Arduino Mega over USB serial.

In the Two-Tier SBC + MCU architecture this node also unpacks the
unified Arduino telemetry packet — encoders + BNO055 IMU stamped from a
single MCU loop — and republishes the IMU half as ``sensor_msgs/Imu`` on
``/imu/data`` while feeding the encoder half into the existing Mecanum
forward-kinematics estimator that publishes ``nav_msgs/Odometry`` on
``/odom`` and broadcasts the ``odom→base_link`` TF.

Serial telemetry lines understood by the parser:

  ODOM <fl> <fr> <rl> <rr>
       <qw> <qx> <qy> <qz>
       <gx> <gy> <gz>       # deg/s, converted to rad/s here
       <ax> <ay> <az>       # m/s^2

  E    <fl> <fr> <rl> <rr>  # legacy encoder-only fallback (no BNO055)

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
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster
import serial
import time
import threading


# BNO055 gyro vectors come out of the Adafruit library in deg/s; sensor_msgs/Imu
# expects rad/s for angular_velocity.
_DEG2RAD = math.pi / 180.0


class MecanumDriverNode(Node):
    def __init__(self):
        super().__init__('mecanum_driver')

        # Declare parameters with explicit ParameterDescriptor types.
        # rclpy rejects mismatched types at declaration time, preventing
        # silent failures when LaunchConfiguration passes strings or when
        # YAML values are parsed with the wrong scalar type.
        self.declare_parameter('serial_port', '/dev/aisha_arduino',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Persistent udev symlink for Arduino Mega (see scripts/arduino_mega.rules)'))
        self.declare_parameter('baud_rate', 115200,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Serial baud rate'))
        self.declare_parameter('wheel_radius', 0.03,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Wheel radius in metres'))
        self.declare_parameter('robot_width', 0.20,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Track width (left-right wheel centre distance) in metres'))
        self.declare_parameter('robot_length', 0.15,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Wheelbase (front-rear wheel centre distance) in metres'))
        self.declare_parameter('max_motor_pwm', 255,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum PWM value sent to BTS7960 (0-255)'))
        # max_wheel_speed in rad/s — default 1.0 m/s / 0.03 m ≈ 33.3 rad/s
        self.declare_parameter('max_wheel_speed', 33.3,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Maximum wheel angular velocity in rad/s'))
        self.declare_parameter('cmd_vel_timeout', 0.5,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Seconds of silence before watchdog stops motors'))
        self.declare_parameter('serial_timeout', 0.1,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Serial read timeout in seconds'))
        # Right-side motors are physically mounted 180° opposite on most
        # chassis.  Set to True to negate FR/RR PWM in software (instead of
        # swapping RPWM/LPWM wires at the BTS7960 driver).
        self.declare_parameter('invert_right_side', False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Negate right-side PWM for reversed motor mounting'))
        # Minimum PWM below which motors stall due to static friction.
        # Nav stack micro-adjustments (PWM 5-30) will just make motors whine.
        # Set via hardware testing: slowly raise PWM until wheels just start
        # turning under load.  0 disables deadband compensation.
        self.declare_parameter('min_motor_pwm', 0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Dead-band PWM threshold; 0 disables compensation'))
        # Encoder ticks per full wheel revolution (after 4× quadrature decoding).
        # Set to 0 to disable encoder odometry (e.g. if encoders are not installed).
        # Arduino Mega encoder decoding: CHANGE on Channel A = 2x decoding.
        # A 600 PPR encoder yields 1200 counts/rev.  If using the RPi pigpio
        # encoder_node (full 4x decoding), set encoder_cpr=2400 instead.
        self.declare_parameter('encoder_cpr', 1200,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Encoder counts per revolution (2x Arduino / 4x RPi)'))
        # Enable/disable publishing odom from encoder data.  When False, the
        # encoder parser still runs (for logging), but no Odometry messages or
        # TF transforms are published.  Use False when running rf2o or dummy_odom
        # as the odom source to avoid conflicting odom→base_link transforms.
        self.declare_parameter('publish_odom', False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Enable encoder-based odometry publishing'))
        # Separate TF broadcast control.  When using robot_localization EKF,
        # the EKF node publishes the odom→base_link TF.  If this driver also
        # broadcasts it, both nodes fight over the TF tree, causing the robot's
        # pose to flicker between the two estimates and breaking Nav2.
        # Set publish_odom=True + publish_odom_tf=False to feed raw encoder
        # odometry to the EKF without conflicting TF broadcasts.
        self.declare_parameter('publish_odom_tf', True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Broadcast odom→base_link TF (disable when EKF is active)'))
        # Frame names for the IMU half of the unified ODOM packet.  The BNO055
        # is mounted on the chassis next to the Arduino; the URDF should
        # define an `imu_link` joint, but the static_transform_publisher in
        # rpi_launch.py also covers the case where the URDF is not loaded.
        self.declare_parameter('imu_frame_id', 'imu_link',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='frame_id stamped on sensor_msgs/Imu messages'))

        # Read parameters — ParameterDescriptor guarantees correct types at
        # declaration time, so the typed accessors below will never mismatch.
        self.serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        self.baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.robot_width = self.get_parameter('robot_width').get_parameter_value().double_value
        self.robot_length = self.get_parameter('robot_length').get_parameter_value().double_value
        self.max_pwm = self.get_parameter('max_motor_pwm').get_parameter_value().integer_value
        self.max_wheel_speed = self.get_parameter('max_wheel_speed').get_parameter_value().double_value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').get_parameter_value().double_value
        self.serial_timeout = self.get_parameter('serial_timeout').get_parameter_value().double_value
        self.invert_right = -1 if self.get_parameter('invert_right_side').get_parameter_value().bool_value else 1
        self.min_pwm = self.get_parameter('min_motor_pwm').get_parameter_value().integer_value
        self.encoder_cpr = self.get_parameter('encoder_cpr').get_parameter_value().integer_value
        self.publish_odom = self.get_parameter('publish_odom').get_parameter_value().bool_value
        self.publish_odom_tf = self.get_parameter('publish_odom_tf').get_parameter_value().bool_value
        self.imu_frame_id = self.get_parameter('imu_frame_id').get_parameter_value().string_value

        # Half-widths for kinematics
        self.lx = self.robot_width / 2.0
        self.ly = self.robot_length / 2.0

        # ── Odometry publisher + TF broadcaster (encoder-based) ───────────
        # Only active when publish_odom=True AND encoder_cpr > 0.
        # Publishes nav_msgs/Odometry on /odom and broadcasts odom→base_link TF.
        # When disabled, rf2o_laser_odometry or dummy_odom provides the transform.
        self._odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── IMU publisher (BNO055 via Arduino) ────────────────────────────
        # The Arduino streams the BNO055 fused quaternion + raw gyro/accel in
        # the same line as the encoder ticks, so the IMU and wheel-odom data
        # share a single MCU-side acquisition timestamp.  We re-stamp with
        # the ROS clock here so downstream consumers (slam_toolbox, Nav2,
        # robot_localization EKF) see a coherent timeline across topics.
        self._imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        # Cumulative pose from forward kinematics integration.
        # Protected by _odom_lock because _update_odometry() is called from
        # both the serial_reader thread (Arduino "E" lines) and the ROS
        # executor thread (_on_encoder_position callback).
        self._odom_lock = threading.Lock()
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
        # The Arduino Mega has 6 interrupt pins available for quadrature
        # encoders (pins 18-21 + 2-3).  When USE_ENCODERS is defined in the
        # Arduino firmware, encoder ticks arrive via serial "E" protocol.
        # Alternatively, encoder_node.py on the RPi reads quadrature encoders
        # via pigpio and publishes to /encoders/position as Float64MultiArray:
        # [rad1, deg1, rad2, deg2, rad3, deg3, rad4, deg4].
        # This subscription converts those radians to equivalent tick counts
        # and feeds them into the same FK pipeline as the serial "E" protocol.
        # WARNING: Do NOT run both Arduino serial encoders (USE_ENCODERS) and
        # RPi encoder_node simultaneously — their tick counts are in different
        # coordinate systems, causing massive delta jumps.  Use one or the
        # other.  _odom_lock prevents data corruption but not logic errors.
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

        # Watchdog timer — uses ROS clock for sim-time compatibility
        self.last_cmd_time = self.get_clock().now()
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
                self.get_logger().warning(
                    f'Serial attempt {attempt+1}/{max_retries} failed: {e}')
                time.sleep(1.0)
        self.get_logger().error(
            'Could not open serial port! Motors will not work.')

    def cmd_vel_callback(self, msg: Twist):
        self.last_cmd_time = self.get_clock().now()
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
        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if elapsed > self.cmd_vel_timeout:
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
        """Parse a telemetry line from the Arduino.

        ── Serial Protocol Contract ──────────────────────────────────────
        Two telemetry packet flavours are accepted:

        (1) Unified ODOM packet — Arduino has the BNO055 IMU on I2C:

            ODOM <fl> <fr> <rl> <rr>
                 <qw> <qx> <qy> <qz>
                 <gx> <gy> <gz>   # deg/s
                 <ax> <ay> <az>   # m/s^2  (linear accel WITH gravity)

        (2) Legacy encoder-only packet — bench setup without IMU:

            E <fl> <fr> <rl> <rr>

        Each tick value is a signed integer: cumulative encoder ticks since
        Arduino boot.  IMU fields are floats; the gyro vector is converted
        from deg/s to rad/s before publishing because sensor_msgs/Imu
        expects SI radians.

        Other recognized prefixes:
          - "OK"     — command acknowledgment (filtered before reaching here)
          - "ERR …"  — Arduino-side error (logged as warning)

        Returns True if the line was recognised (whether or not the data
        was usable), False if the line should fall through to debug logging.
        """
        parts = line.split()
        if not parts:
            return False

        # ── Unified encoder + IMU telemetry ───────────────────────────────
        if parts[0] == 'ODOM' and len(parts) == 15:
            try:
                ticks = (int(parts[1]), int(parts[2]),
                         int(parts[3]), int(parts[4]))
                qw, qx, qy, qz = (float(parts[5]), float(parts[6]),
                                   float(parts[7]), float(parts[8]))
                gx, gy, gz = (float(parts[9]), float(parts[10]),
                               float(parts[11]))
                ax, ay, az = (float(parts[12]), float(parts[13]),
                               float(parts[14]))
            except ValueError:
                self.get_logger().warning(f'Malformed ODOM line: {line}')
                return True

            # Unified ROS timestamp — every consumer (slam_toolbox, Nav2,
            # robot_localization) sees encoder and IMU samples with the
            # exact same stamp, eliminating fusion-time mismatch errors.
            stamp = self.get_clock().now().to_msg()

            self._last_encoder_ticks = ticks
            self._publish_imu(stamp, qw, qx, qy, qz, gx, gy, gz, ax, ay, az)

            if self.publish_odom and self.encoder_cpr > 0:
                self._update_odometry(ticks)

            return True

        # ── Legacy encoder-only fallback (no BNO055) ──────────────────────
        if parts[0] == 'E' and len(parts) == 5:
            try:
                ticks = (int(parts[1]), int(parts[2]),
                         int(parts[3]), int(parts[4]))
            except ValueError:
                self.get_logger().warning(f'Malformed encoder line: {line}')
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
            self.get_logger().warning(f'Arduino error: {line}')
            return True

        return False

    def _publish_imu(self, stamp, qw, qx, qy, qz, gx_dps, gy_dps, gz_dps,
                     ax, ay, az):
        """Publish a sensor_msgs/Imu from BNO055 fields in an ODOM packet.

        The BNO055 outputs a fused orientation quaternion plus raw gyro
        (deg/s) and linear acceleration (m/s^2, gravity-included).  We
        convert the gyro to rad/s and stamp the message with the unified
        ROS clock value captured at parse time.

        Covariance matrices: the BNO055 datasheet does not specify formal
        per-axis covariance, so we use empirically-reasonable diagonal
        values.  All-zero matrices would make robot_localization treat the
        sensor as perfectly accurate, which is dangerous.
        """
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame_id

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        msg.angular_velocity.x = gx_dps * _DEG2RAD
        msg.angular_velocity.y = gy_dps * _DEG2RAD
        msg.angular_velocity.z = gz_dps * _DEG2RAD

        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        # Diagonal covariances.  Numbers are conservative but non-zero so
        # downstream EKF/UKF nodes do not divide by zero on this source.
        msg.orientation_covariance = [
            0.01, 0.0,  0.0,
            0.0,  0.01, 0.0,
            0.0,  0.0,  0.01,
        ]
        msg.angular_velocity_covariance = [
            0.005, 0.0,   0.0,
            0.0,   0.005, 0.0,
            0.0,   0.0,   0.005,
        ]
        msg.linear_acceleration_covariance = [
            0.05, 0.0,  0.0,
            0.0,  0.05, 0.0,
            0.0,  0.0,  0.05,
        ]

        self._imu_pub.publish(msg)

    def _update_odometry(self, ticks: tuple):
        """Apply Mecanum forward kinematics and publish odom + TF.

        Converts delta encoder ticks → wheel angular displacements →
        robot-frame velocity (vx, vy, wz) → integrated pose (x, y, θ).

        Called from BOTH the serial_reader thread (Arduino "E" lines) and the
        ROS executor thread (_on_encoder_position callback).  All shared
        odometry state is guarded by _odom_lock.  Publishing from a
        non-executor thread is safe in rclpy — publishers are thread-safe.

        Mecanum Forward Kinematics:
          vx = (R/4)( ωfl + ωfr + ωrl + ωrr)
          vy = (R/4)(-ωfl + ωfr + ωrl - ωrr)
          wz = (R/4(lx+ly))(-ωfl + ωfr - ωrl + ωrr)
        """
        ros_now = self.get_clock().now()
        now_sec = ros_now.nanoseconds / 1e9

        with self._odom_lock:
            # First reading — just store the baseline, can't compute deltas yet
            if self._prev_encoder_ticks is None:
                self._prev_encoder_ticks = ticks
                self._last_encoder_time = now_sec
                return

            dt = now_sec - self._last_encoder_time
            if dt <= 0:
                return
            self._last_encoder_time = now_sec

            # Delta ticks since last reading.
            # Right-side motors are physically mirrored, so their encoders
            # report negated ticks when invert_right_side is active.  Apply
            # the same inversion used in the inverse kinematics (cmd_vel→PWM)
            # so forward kinematics (encoder→odometry) stays consistent.
            d_fl = ticks[0] - self._prev_encoder_ticks[0]
            d_fr = (ticks[1] - self._prev_encoder_ticks[1]) * self.invert_right
            d_rl = ticks[2] - self._prev_encoder_ticks[2]
            d_rr = (ticks[3] - self._prev_encoder_ticks[3]) * self.invert_right
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

            # Snapshot pose for publishing (still under lock)
            odom_x = self._odom_x
            odom_y = self._odom_y
            odom_theta = self._odom_theta

        # ── Publish outside lock (publishers are thread-safe) ─────────
        ros_stamp = ros_now.to_msg()

        # Compute quaternion from yaw (2D robot, roll=pitch=0)
        half_theta = odom_theta / 2.0
        odom_quat = Quaternion()
        odom_quat.x = 0.0
        odom_quat.y = 0.0
        odom_quat.z = math.sin(half_theta)
        odom_quat.w = math.cos(half_theta)

        # Broadcast odom→base_link TF only if publish_odom_tf is True.
        # When the EKF is active, it owns this transform — broadcasting
        # from both nodes causes TF tree flickering that breaks Nav2.
        if self.publish_odom_tf:
            t = TransformStamped()
            t.header.stamp = ros_stamp
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = odom_x
            t.transform.translation.y = odom_y
            t.transform.translation.z = 0.0
            t.transform.rotation = odom_quat
            self._tf_broadcaster.sendTransform(t)

        # Publish nav_msgs/Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = ros_stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        # Pose
        odom_msg.pose.pose.position.x = odom_x
        odom_msg.pose.pose.position.y = odom_y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = odom_quat
        # Twist (robot-frame velocities)
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.angular.z = wz

        # Covariance matrices (6×6, row-major).  robot_localization's EKF
        # interprets all-zero covariance as infinite certainty, producing a
        # singular matrix during Kalman gain inversion.  Provide baseline
        # uncertainty for the 2D states we actually observe (x, y, yaw).
        # Unobserved states (z, roll, pitch) get a large value so the EKF
        # effectively ignores them from this source.
        #   Index map: 0=x, 7=y, 14=z, 21=roll, 28=pitch, 35=yaw
        pose_cov = [0.0] * 36
        pose_cov[0] = 0.01    # x
        pose_cov[7] = 0.01    # y
        pose_cov[14] = 1e6    # z        (unobserved — large uncertainty)
        pose_cov[21] = 1e6    # roll     (unobserved)
        pose_cov[28] = 1e6    # pitch    (unobserved)
        pose_cov[35] = 0.05   # yaw
        odom_msg.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = 0.01   # vx
        twist_cov[7] = 0.01   # vy
        twist_cov[14] = 1e6   # vz       (unobserved)
        twist_cov[21] = 1e6   # vroll    (unobserved)
        twist_cov[28] = 1e6   # vpitch   (unobserved)
        twist_cov[35] = 0.05  # vyaw
        odom_msg.twist.covariance = twist_cov

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
