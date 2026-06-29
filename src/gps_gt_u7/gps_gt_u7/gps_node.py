#!/usr/bin/env python3
"""
ROS2 driver for Goouuu Tech GT-U7 GPS module.
Reads NMEA 0183 sentences from serial port and publishes:
  - /gps/fix        (sensor_msgs/NavSatFix)     -- position + accuracy
  - /gps/vel        (geometry_msgs/TwistStamped) -- speed + heading
  - /gps/time_ref   (sensor_msgs/TimeReference)  -- GPS time
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import serial
import pynmea2

from sensor_msgs.msg import NavSatFix, NavSatStatus, TimeReference
from geometry_msgs.msg import TwistStamped


# QoS matching common GPS drivers (best effort, volatile)
GPS_QOS = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class GpsNode(Node):
    def __init__(self):
        super().__init__('gps_gt_u7_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 9600)
        self.declare_parameter('timeout', 1.0)
        self.declare_parameter('frame_id', 'gps_link')

        self.port      = self.get_parameter('port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.timeout   = self.get_parameter('timeout').value
        self.frame_id  = self.get_parameter('frame_id').value

        # ── Publishers ───────────────────────────────────────────────────────
        self.fix_pub  = self.create_publisher(NavSatFix,     'gps/fix',      GPS_QOS)
        self.vel_pub  = self.create_publisher(TwistStamped,  'gps/vel',      GPS_QOS)
        self.time_pub = self.create_publisher(TimeReference, 'gps/time_ref', GPS_QOS)

        # ── State shared between sentence callbacks ───────────────────────────
        self._fix_msg  = NavSatFix()
        self._fix_msg.status.status        = NavSatStatus.STATUS_NO_FIX
        self._fix_msg.status.service       = NavSatStatus.SERVICE_GPS
        self._fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self._fix_msg.position_covariance  = [0.0] * 9

        self._hdop   = float('nan')   # updated by GPGSA / GNGSA
        self._vdop   = float('nan')

        # ── Open serial port ─────────────────────────────────────────────────
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
            )
            self.get_logger().info(
                f'Opened GPS on {self.port} @ {self.baud_rate} baud'
            )
        except serial.SerialException as e:
            self.get_logger().error(f'Cannot open {self.port}: {e}')
            raise SystemExit(1)

        # ── Read loop timer (runs as fast as data arrives) ───────────────────
        self.create_timer(0.0, self._read_once)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _stamp(self):
        return self.get_clock().now().to_msg()

    @staticmethod
    def _covariance_from_dop(hdop, vdop):
        """
        Build a diagonal covariance matrix from DOP values.
        Uses the rule-of-thumb: 1σ ≈ DOP × 3 m (UERE ≈ 3 m for single GPS).
        """
        UERE = 3.0   # metres, typical user equivalent range error
        h_var = (hdop * UERE) ** 2 if math.isfinite(hdop) else 0.0
        v_var = (vdop * UERE) ** 2 if math.isfinite(vdop) else 0.0
        return [
            h_var, 0.0,   0.0,
            0.0,   h_var, 0.0,
            0.0,   0.0,   v_var,
        ]

    # ── Sentence parsers ──────────────────────────────────────────────────────

    def _handle_gga(self, msg):
        """GPGGA / GNGGA — position, altitude, fix quality, satellites."""
        if msg.gps_qual == 0:
            self._fix_msg.status.status = NavSatStatus.STATUS_NO_FIX
            return

        self._fix_msg.header.stamp    = self._stamp()
        self._fix_msg.header.frame_id = self.frame_id
        self._fix_msg.latitude        = msg.latitude
        self._fix_msg.longitude       = msg.longitude

        # Altitude: MSL + geoid separation = ellipsoidal height
        try:
            alt_msl   = float(msg.altitude)
            geoid_sep = float(msg.geo_sep) if msg.geo_sep else 0.0
            self._fix_msg.altitude = alt_msl + geoid_sep
        except (TypeError, ValueError):
            self._fix_msg.altitude = float('nan')

        # Fix status
        if msg.gps_qual == 1:
            self._fix_msg.status.status = NavSatStatus.STATUS_FIX
        elif msg.gps_qual in (2, 4, 5):
            self._fix_msg.status.status = NavSatStatus.STATUS_SBAS_FIX
        else:
            self._fix_msg.status.status = NavSatStatus.STATUS_FIX

        # HDOP from GGA field (fallback if GSA not available)
        try:
            self._hdop = float(msg.horizontal_dil)
        except (TypeError, ValueError):
            pass

        # Update covariance
        cov_type = (
            NavSatFix.COVARIANCE_TYPE_APPROXIMATED
            if math.isfinite(self._hdop) else
            NavSatFix.COVARIANCE_TYPE_UNKNOWN
        )
        self._fix_msg.position_covariance_type = cov_type
        self._fix_msg.position_covariance = self._covariance_from_dop(
            self._hdop, self._vdop
        )

        self.fix_pub.publish(self._fix_msg)

        self.get_logger().debug(
            f'Fix  lat={msg.latitude:.7f}  lon={msg.longitude:.7f}  '
            f'alt={self._fix_msg.altitude:.2f} m  '
            f'qual={msg.gps_qual}  sats={msg.num_sats}  hdop={self._hdop:.1f}'
        )

    def _handle_rmc(self, msg):
        """GPRMC / GNRMC — speed over ground, course over ground, date/time."""
        if msg.status != 'A':          # 'A' = Active (valid fix)
            return

        now = self._stamp()

        # Velocity
        try:
            sog_ms  = float(msg.spd_over_grnd) * 0.51444   # knots → m/s
            cog_rad = math.radians(float(msg.true_course))
            vel = TwistStamped()
            vel.header.stamp    = now
            vel.header.frame_id = self.frame_id
            vel.twist.linear.x  = sog_ms * math.cos(cog_rad)   # East
            vel.twist.linear.y  = sog_ms * math.sin(cog_rad)   # North
            vel.twist.linear.z  = 0.0
            self.vel_pub.publish(vel)
        except (TypeError, ValueError):
            pass

        # GPS time reference
        try:
            import datetime
            if msg.datestamp and msg.timestamp:
                dt = datetime.datetime.combine(msg.datestamp, msg.timestamp)
                t_ref = TimeReference()
                t_ref.header.stamp  = now
                t_ref.header.frame_id = self.frame_id
                t_ref.time_ref.sec  = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
                t_ref.time_ref.nanosec = 0
                t_ref.source = 'gps'
                self.time_pub.publish(t_ref)
        except Exception:
            pass

    def _handle_gsa(self, msg):
        """GPGSA / GNGSA — DOP values for covariance matrix."""
        try:
            self._hdop = float(msg.hdop)
            self._vdop = float(msg.vdop)
        except (TypeError, ValueError):
            pass

    # ── Main read loop ────────────────────────────────────────────────────────

    def _read_once(self):
        """Called repeatedly by timer; reads one line and dispatches."""
        try:
            raw = self._serial.readline()
            if not raw:
                return
            line = raw.decode('ascii', errors='replace').strip()
        except serial.SerialException as e:
            self.get_logger().error(f'Serial read error: {e}')
            return

        if not line.startswith('$'):
            return

        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            return

        sentence = type(msg).__name__  # e.g. 'GGA', 'RMC', 'GSA'

        if sentence == 'GGA':
            self._handle_gga(msg)
        elif sentence == 'RMC':
            self._handle_rmc(msg)
        elif sentence == 'GSA':
            self._handle_gsa(msg)

    def destroy_node(self):
        if hasattr(self, '_serial') and self._serial.is_open:
            self._serial.close()
            self.get_logger().info('Serial port closed.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
