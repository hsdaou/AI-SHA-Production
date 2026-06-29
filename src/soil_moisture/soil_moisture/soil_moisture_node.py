import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32, Bool
import serial
import threading


class SoilMoistureNode(Node):
    def __init__(self):
        super().__init__('soil_moisture_node')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 9600)
        self.declare_parameter('publish_rate', 1.0)

        port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud = self.get_parameter('baud_rate').get_parameter_value().integer_value
        rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.pub_raw = self.create_publisher(Int32, 'soil_moisture/raw', 10)
        self.pub_moisture = self.create_publisher(Float32, 'soil_moisture/moisture', 10)
        self.pub_dry = self.create_publisher(Bool, 'soil_moisture/dry', 10)

        try:
            self.ser = serial.Serial(port, baud, timeout=2.0)
            self.get_logger().info(f'Opened serial port {port} at {baud} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {port}: {e}')
            raise

        self.ser.reset_input_buffer()

        self._lock = threading.Lock()
        self._latest_raw = None
        self._latest_dry = None

        self._reader_thread = threading.Thread(target=self._read_serial, daemon=True)
        self._reader_thread.start()

        self.create_timer(1.0 / rate, self._publish)

    def _read_serial(self):
        while rclpy.ok():
            try:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 2:
                    continue
                raw = int(parts[0])
                dry = int(parts[1])
                with self._lock:
                    self._latest_raw = raw
                    self._latest_dry = dry
            except (ValueError, serial.SerialException) as e:
                self.get_logger().warning(f'Serial read error: {e}')

    def _publish(self):
        with self._lock:
            raw = self._latest_raw
            dry = self._latest_dry

        if raw is None:
            return

        msg = Int32(); msg.data = raw
        self.pub_raw.publish(msg)

        msg = Float32(); msg.data = float((1023 - raw) / 1023 * 100)
        self.pub_moisture.publish(msg)

        msg = Bool(); msg.data = bool(dry)
        self.pub_dry.publish(msg)

        self.get_logger().debug(
            f'raw={raw}  moisture={float((1023-raw)/1023*100):.1f}%  dry={bool(dry)}'
        )

    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SoilMoistureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
