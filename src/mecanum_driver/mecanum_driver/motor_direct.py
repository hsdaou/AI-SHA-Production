import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


class MotorDirectNode(Node):
    def __init__(self):
        super().__init__('motor_direct')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)

        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value

        self.ser = None
        self.connect_serial()

        self.subscription = self.create_subscription(
            Int16MultiArray,
            'mecanum/motor_cmd',
            self.cmd_callback,
            10
        )

        self.get_logger().info(
            f'motor_direct ready — listening on mecanum/motor_cmd, '
            f'serial: {self.serial_port} @ {self.baud_rate}'
        )

    def connect_serial(self):
        if not HAS_SERIAL:
            self.get_logger().warn('pyserial not installed — running in dry-run mode')
            return

        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            self.get_logger().info(f'Connected to Arduino on {self.serial_port}')
        except serial.SerialException as e:
            self.get_logger().warn(
                f'Could not open {self.serial_port}: {e} — running in dry-run mode'
            )
            self.ser = None

    def cmd_callback(self, msg):
        if len(msg.data) != 4:
            self.get_logger().warn(f'Expected 4 motor values, got {len(msg.data)}')
            return

        m1, m2, m3, m4 = msg.data

        # Protocol: "<m1,m2,m3,m4>\n"
        # Each value is -255 to 255; Arduino parses and drives BTS7960 accordingly
        cmd_str = f'<{m1},{m2},{m3},{m4}>\n'

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd_str.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f'Serial write error: {e}')
                self.ser = None
        else:
            self.get_logger().info(f'[dry-run] {cmd_str.strip()}')


def main(args=None):
    rclpy.init(args=args)
    node = MotorDirectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop on shutdown
        if node.ser and node.ser.is_open:
            node.ser.write(b'<0,0,0,0>\n')
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
