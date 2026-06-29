import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure, Temperature
from std_msgs.msg import Float64
import smbus2
import time
import struct
import math


BMP180_ADDR = 0x77
REG_CHIP_ID  = 0xD0
REG_CTRL     = 0xF4
REG_DATA     = 0xF6
CMD_TEMP     = 0x2E
CMD_PRESS    = 0x34  # OSS=0 (single sample)
OSS          = 0     # oversampling: 0=1x, 1=2x, 2=4x, 3=8x


class BMP180Node(Node):
    def __init__(self):
        super().__init__('bmp180_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('publish_rate', 1.0)   # Hz
        # Standard sea-level pressure for Sharjah, UAE (≈ sea level, 101325 Pa)
        # Tune this to your local weather station QNH for best accuracy
        self.declare_parameter('sea_level_pressure', 101325.0)  # Pa

        bus_num       = self.get_parameter('i2c_bus').value
        rate_hz       = self.get_parameter('publish_rate').value
        self.p0       = self.get_parameter('sea_level_pressure').value

        self.bus = smbus2.SMBus(bus_num)
        self._verify_chip()
        self.cal = self._read_calibration()
        self.get_logger().info('BMP180 calibration loaded')

        self.pub_pressure = self.create_publisher(FluidPressure, 'bmp180/pressure',    10)
        self.pub_temp     = self.create_publisher(Temperature,   'bmp180/temperature', 10)
        self.pub_altitude = self.create_publisher(Float64,        'bmp180/altitude',    10)

        self.create_timer(1.0 / rate_hz, self._timer_cb)
        self.get_logger().info(
            f'BMP180 node started — bus={bus_num}, rate={rate_hz} Hz'
        )

    # ------------------------------------------------------------------ setup

    def _verify_chip(self):
        chip_id = self.bus.read_byte_data(BMP180_ADDR, REG_CHIP_ID)
        if chip_id != 0x55:
            raise RuntimeError(f'Unexpected BMP180 chip ID: 0x{chip_id:02X} (expected 0x55)')
        self.get_logger().info(f'BMP180 found at 0x{BMP180_ADDR:02X}, chip_id=0x{chip_id:02X}')

    def _read_calibration(self):
        """Read 22 bytes of factory calibration from 0xAA–0xBF."""
        raw = self.bus.read_i2c_block_data(BMP180_ADDR, 0xAA, 22)
        keys = ['AC1', 'AC2', 'AC3', 'AC4', 'AC5', 'AC6',
                'B1',  'B2',  'MB',  'MC',  'MD']
        # First 6 values: AC1-AC3 signed, AC4-AC6 unsigned
        signed_mask = [True, True, True, False, False, False,
                       True, True, True, True,  True]
        cal = {}
        for i, (k, signed) in enumerate(zip(keys, signed_mask)):
            fmt = '>h' if signed else '>H'
            cal[k] = struct.unpack_from(fmt, bytes(raw), i * 2)[0]
        return cal

    # ---------------------------------------------------------------- reading

    def _read_raw_temp(self):
        self.bus.write_byte_data(BMP180_ADDR, REG_CTRL, CMD_TEMP)
        time.sleep(0.0045)  # 4.5 ms
        msb, lsb = self.bus.read_i2c_block_data(BMP180_ADDR, REG_DATA, 2)
        return (msb << 8) + lsb

    def _read_raw_pressure(self):
        self.bus.write_byte_data(BMP180_ADDR, REG_CTRL, CMD_PRESS + (OSS << 6))
        time.sleep(0.005 + 0.003 * (2 ** OSS))  # wait per datasheet
        data = self.bus.read_i2c_block_data(BMP180_ADDR, REG_DATA, 3)
        return ((data[0] << 16) + (data[1] << 8) + data[2]) >> (8 - OSS)

    def _compensate(self, UT, UP):
        """Apply BMP180 datasheet compensation formulas.
        Returns (temperature_C, pressure_Pa).
        """
        c = self.cal

        # --- temperature ---
        X1 = (UT - c['AC6']) * c['AC5'] >> 15
        X2 = (c['MC'] << 11) // (X1 + c['MD'])
        B5 = X1 + X2
        temp_C = (B5 + 8) / 160.0  # 0.1 °C units → °C

        # --- pressure ---
        B6 = B5 - 4000
        X1 = (c['B2'] * (B6 * B6 >> 12)) >> 11
        X2 = (c['AC2'] * B6) >> 11
        X3 = X1 + X2
        B3 = (((c['AC1'] * 4 + X3) << OSS) + 2) >> 2
        X1 = (c['AC3'] * B6) >> 13
        X2 = (c['B1'] * (B6 * B6 >> 12)) >> 16
        X3 = ((X1 + X2) + 2) >> 2
        B4 = c['AC4'] * (X3 + 32768) >> 15
        B7 = (UP - B3) * (50000 >> OSS)

        p = (B7 * 2) // B4 if B7 < 0x80000000 else (B7 // B4) * 2

        X1 = (p >> 8) ** 2
        X1 = (X1 * 3038) >> 16
        X2 = (-7357 * p) >> 16
        pressure_Pa = p + ((X1 + X2 + 3791) >> 4)

        return temp_C, float(pressure_Pa)

    def _pressure_to_altitude(self, pressure_Pa):
        """International barometric formula (ICAO standard atmosphere).
        altitude = 44330 * (1 - (P / P0) ^ (1/5.255))
        Accurate to ±1 m when P0 matches local QNH.
        """
        return 44330.0 * (1.0 - math.pow(pressure_Pa / self.p0, 1.0 / 5.255))

    # --------------------------------------------------------------- callback

    def _timer_cb(self):
        try:
            UT = self._read_raw_temp()
            UP = self._read_raw_pressure()
            temp_C, pressure_Pa = self._compensate(UT, UP)
        except Exception as e:
            self.get_logger().error(f'BMP180 read error: {e}')
            return

        altitude_m = self._pressure_to_altitude(pressure_Pa)
        now = self.get_clock().now().to_msg()

        press_msg = FluidPressure()
        press_msg.header.stamp = now
        press_msg.header.frame_id = 'bmp180'
        press_msg.fluid_pressure = pressure_Pa  # Pa
        press_msg.variance = 0.0
        self.pub_pressure.publish(press_msg)

        temp_msg = Temperature()
        temp_msg.header.stamp = now
        temp_msg.header.frame_id = 'bmp180'
        temp_msg.temperature = temp_C  # °C
        temp_msg.variance = 0.0
        self.pub_temp.publish(temp_msg)

        alt_msg = Float64()
        alt_msg.data = altitude_m  # metres above sea level
        self.pub_altitude.publish(alt_msg)

        self.get_logger().info(
            f'P={pressure_Pa:.1f} Pa ({pressure_Pa/100:.2f} hPa)  '
            f'T={temp_C:.2f} °C  Alt={altitude_m:.1f} m',
            throttle_duration_sec=5.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = BMP180Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.bus.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
