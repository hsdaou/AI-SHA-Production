#!/usr/bin/env python3
"""
ROS2 node: read up to FOUR quadrature encoders wired DIRECTLY to the Pi's GPIO
via pigpio. This is a 4-encoder variant of pi_encoder_node (which is unchanged).

Encoders: E38S6G5 (600 PPR), NPN open-collector A/B phases.
Wiring convention (BCM numbering) used here:
    The TWO pins given per encoder are "white, green".
    white (A phase) -> first  pin
    green (B phase) -> second pin
    VCC -> 5V        GND -> GND
    External pull-ups (~2.2k) on A/B to 3.3V, and pigpiod runs at -s 1 (1us).
    Both are needed above ~2600 RPM: the pull-ups sharpen the open-collector
    edges, and 1us sampling stops pigpio coalescing the A/B edges.

Current wiring:
    Encoder 1: A(white)=12, B(green)=20
    Encoder 2: A(white)=7,  B(green)=8
    Encoder 3: A(white)=10, B(green)=9
    Encoder 4: A(white)=26, B(green)=19
An encoder whose gpio_a or gpio_b is < 0 is skipped: no decoder process and no
topics are created for it.

WHY A SEPARATE PROCESS: at motor-shaft speed each encoder emits >100k edges/s.
If the pigpio callbacks run in the same process as rclpy, ROS work steals GIL
time from the callback thread in bursts and edges are dropped (RPM jitters,
counts under-read). So each encoder's decoding runs in its OWN process that does
nothing else, writing the live count to shared memory; this node just reads
those ints at the publish rate. Each decoder process captures ~100% of edges.

Publishes (n = 1..4, only for enabled encoders):
    /encoder_<n>/rpm        (Float64) - signed revolutions per minute
    /encoder_<n>/count      (Int64)   - raw cumulative quadrature count
    /encoder_<n>/angle_rad  (Float64) - cumulative angle in radians
    /encoder_<n>/angle_deg  (Float64) - cumulative angle in degrees

Parameters:
    gpio_a_1, gpio_b_1 - BCM pins for encoder 1 A/B (default 12, 20)
    gpio_a_2, gpio_b_2 - BCM pins for encoder 2 A/B (default  7,  8)
    gpio_a_3, gpio_b_3 - BCM pins for encoder 3 A/B (default 10, 9)
    gpio_a_4, gpio_b_4 - BCM pins for encoder 4 A/B (default 26, 19)
    encoder_ppr    - pulses per rev per channel (default 600; cpr = 4*ppr)
    publish_rate   - publish rate in Hz (default 20)
    display_rate   - terminal refresh in Hz (default 4)
    rpm_window     - sliding window for RPM in seconds (default 0.3)
    rpm_ema        - extra smoothing 0..1, 1 = off (default 0.5)
"""

import math
import sys
import time
import multiprocessing as mp
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int64


# Quadrature decode table. Index = (prevState << 2) | newState,
# state = (A << 1) | B; value is the count delta (+1/-1, 0 = no-op/invalid).
_QDEC = (
    0, -1,  1,  0,
    1,  0,  0, -1,
   -1,  0,  0,  1,
    0,  1, -1,  0,
)


def _decoder_process(gpio_a, gpio_b, shared, stop_evt):
    """Runs in its own process: pigpio quadrature decode -> shared int count.

    The callback only updates a local int (the proven-fast path); a slow loop
    copies it to shared memory. Nothing else competes for this process's GIL,
    so the callback thread keeps up with >100k edges/s.
    """
    import pigpio
    pi = pigpio.pi()
    if not pi.connected:
        shared.value = 0
        return
    for g in (gpio_a, gpio_b):
        pi.set_mode(g, pigpio.INPUT)
        pi.set_pull_up_down(g, pigpio.PUD_UP)

    st = {'a': pi.read(gpio_a), 'b': pi.read(gpio_b), 'pos': 0}
    st['last'] = (st['a'] << 1) | st['b']

    def cb(gpio, level, tick):
        if level == 2:
            return
        if gpio == gpio_a:
            st['a'] = level
        else:
            st['b'] = level
        state = (st['a'] << 1) | st['b']
        st['pos'] += _QDEC[(st['last'] << 2) | state]
        st['last'] = state

    cb1 = pi.callback(gpio_a, pigpio.EITHER_EDGE, cb)
    cb2 = pi.callback(gpio_b, pigpio.EITHER_EDGE, cb)

    while not stop_evt.is_set():
        shared.value = st['pos']
        time.sleep(0.005)

    cb1.cancel()
    cb2.cancel()
    pi.stop()


class _Encoder:
    """Per-encoder state: decoder process + publishers + RPM filtering."""

    def __init__(self, node, index, gpio_a, gpio_b, cpr, rpm_window, rpm_ema):
        self.index = index
        self.gpio_a = gpio_a
        self.gpio_b = gpio_b
        self.cpr = cpr
        self.rpm_window = rpm_window
        self.rpm_ema = rpm_ema

        self.shared = mp.Value('q', 0)
        self.stop = mp.Event()
        self.proc = mp.Process(
            target=_decoder_process,
            args=(gpio_a, gpio_b, self.shared, self.stop),
            daemon=True)
        self.proc.start()

        ns = '/encoder_{}'.format(index)
        self.pub_rpm = node.create_publisher(Float64, ns + '/rpm', 10)
        self.pub_cnt = node.create_publisher(Int64, ns + '/count', 10)
        self.pub_rad = node.create_publisher(Float64, ns + '/angle_rad', 10)
        self.pub_deg = node.create_publisher(Float64, ns + '/angle_deg', 10)

        self._win = deque()        # (monotonic_time, count) for windowed RPM
        self._rpm_filt = 0.0

        # last computed values, for the shared terminal display
        self.rpm = 0.0
        self.count = 0
        self.rad = 0.0
        self.deg = 0.0

    def update(self, now):
        count = self.shared.value
        rpm = self._windowed_rpm(now, count)
        revs = count / self.cpr if self.cpr else 0.0
        rad = revs * 2.0 * math.pi
        deg = revs * 360.0

        self.pub_rpm.publish(Float64(data=rpm))
        self.pub_cnt.publish(Int64(data=int(count)))
        self.pub_rad.publish(Float64(data=rad))
        self.pub_deg.publish(Float64(data=deg))

        self.rpm, self.count, self.rad, self.deg = rpm, count, rad, deg

    def _windowed_rpm(self, now, count):
        self._win.append((now, count))
        cutoff = now - self.rpm_window
        while len(self._win) > 2 and self._win[0][0] < cutoff:
            self._win.popleft()
        t0, c0 = self._win[0]
        dt = now - t0
        if dt <= 0 or self.cpr <= 0:
            return self._rpm_filt
        raw = (count - c0) / self.cpr / dt * 60.0
        a = self.rpm_ema if 0.0 < self.rpm_ema <= 1.0 else 1.0
        self._rpm_filt = a * raw + (1.0 - a) * self._rpm_filt
        return self._rpm_filt

    def shutdown(self):
        try:
            self.stop.set()
            self.proc.join(timeout=2.0)
            if self.proc.is_alive():
                self.proc.terminate()
        except Exception:
            pass


class PiEncoderNode4(Node):
    def __init__(self):
        super().__init__('pi_encoder_node_4')

        # Per-encoder pins. Encoders 1 & 2 are wired; 3 & 4 default to -1
        # (disabled) until their pins are supplied.
        defaults = {
            1: (12, 20),
            2: (7, 8),
            3: (10, 9),
            4: (26, 19),
        }
        for n, (a, b) in defaults.items():
            self.declare_parameter('gpio_a_{}'.format(n), a)
            self.declare_parameter('gpio_b_{}'.format(n), b)

        self.declare_parameter('encoder_ppr', 600)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('display_rate', 4.0)
        self.declare_parameter('rpm_window', 0.3)
        self.declare_parameter('rpm_ema', 0.5)

        self.ppr = self.get_parameter('encoder_ppr').value
        publish_rate = self.get_parameter('publish_rate').value
        display_rate = self.get_parameter('display_rate').value
        rpm_window = self.get_parameter('rpm_window').value
        rpm_ema = self.get_parameter('rpm_ema').value
        self.cpr = self.ppr * 4   # decoder is always X4

        # Verify the pigpio daemon is reachable before spawning any decoder.
        import pigpio
        probe = pigpio.pi()
        if not probe.connected:
            self.get_logger().fatal('Cannot connect to pigpio daemon! Run: sudo pigpiod')
            raise RuntimeError('pigpio daemon not running')
        probe.stop()

        self.encoders = []
        for n in range(1, 5):
            a = self.get_parameter('gpio_a_{}'.format(n)).value
            b = self.get_parameter('gpio_b_{}'.format(n)).value
            if a < 0 or b < 0:
                self.get_logger().info(
                    'Encoder {} disabled (pins {}, {}); skipping.'.format(n, a, b))
                continue
            self.get_logger().info(
                'Encoder {}: A(white)=GPIO{}  B(green)=GPIO{}'.format(n, a, b))
            self.encoders.append(
                _Encoder(self, n, a, b, self.cpr, rpm_window, rpm_ema))

        if not self.encoders:
            self.get_logger().fatal('No encoders enabled! Set gpio_a_N/gpio_b_N.')
            raise RuntimeError('no encoders enabled')

        self.timer = self.create_timer(1.0 / max(1.0, publish_rate), self._update)

        self._display_divider = max(1, round(publish_rate / max(0.1, display_rate)))
        self._tick = 0
        self._header_printed = False

    def _update(self):
        now = time.monotonic()
        for enc in self.encoders:
            enc.update(now)

        self._tick += 1
        if self._tick >= self._display_divider:
            self._tick = 0
            self._display()

    def _display(self):
        if not self._header_printed:
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.write(
                '\033[1;36m  Pi Encoder x4  |  PPR {} X4 -> CPR {}\033[0m\n'.format(
                    self.ppr, self.cpr))
            sys.stdout.write('\033[90m' + '-' * 64 + '\033[0m\n')
            sys.stdout.write(
                '\033[1m {:<6s}{:>6s}{:>12s}{:>12s}{:>12s}{:>14s}\033[0m\n'.format(
                    'Enc', 'A/B', 'RPM', 'rad', 'deg', 'Counts'))
            self._header_printed = True
        # Table body starts on terminal row 4.
        sys.stdout.write('\033[4;1H')
        for enc in self.encoders:
            sys.stdout.write(
                ' {:<6d}{:>6s}{:>12.2f}{:>12.4f}{:>12.2f}{:>14d}\033[K\n'.format(
                    enc.index,
                    '{}/{}'.format(enc.gpio_a, enc.gpio_b),
                    enc.rpm, enc.rad, enc.deg, int(enc.count)))
        sys.stdout.flush()

    def destroy_node(self):
        for enc in self.encoders:
            enc.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiEncoderNode4()
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
