#!/usr/bin/env python3
"""AI-SHA LiDAR bring-up for the Raspberry Pi 5.

The unit on /dev/ttyUSB0 is an LDROBOT LD06/LD19-family scanner: it free-runs
(no command protocol -- an RPLIDAR GET_INFO gets no descriptor back) and streams
47-byte packets at 230400 baud, header 0x54 0x2C, 12 measurement points each.

One process does two jobs on purpose:

  * publishes sensor_msgs/LaserScan on /scan for the Pi's own ROS 2 (Jazzy) graph,
  * serves the same sweep as JSON on :8090 for the Jetson console.

The JSON feed is not a shortcut -- data does NOT flow from Jazzy to the Jetson's
Humble over DDS (Iron+ added type hashes Humble doesn't supply, so real
subscriptions match on discovery but never receive). Plain HTTP is immune to that.
"""
import json, math, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import glob
import serial

# Resolved fresh on every (re)connect, never cached. A reboot or a USB glitch moves
# this scanner between /dev/ttyUSB0 and ttyUSB1, and reading a device that has gone
# away returns b"" forever instead of raising -- so a hardcoded path strands the
# driver in a silent spin holding a sweep that never updates.
BY_ID_GLOB = "/dev/serial/by-id/*CP2102*"
FALLBACK_GLOB = "/dev/ttyUSB*"
STALL_SECS = 3.0              # no valid packet for this long => reopen the port
BAUD     = 230400
HTTP_PORT = 8090
HDR, VERLEN = 0x54, 0x2C
PKT = 47                      # 2 hdr + 2 speed + 2 start + 12*3 + 2 end + 2 ts + 1 crc
BINS = 360                    # 1 degree bins; the LD06 gives ~450 points/rev


def _crc8(data: bytes) -> int:
    """LD06 CRC: poly 0x4D, init 0, no reflection."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class Scanner:
    """Reads the serial stream and keeps the most recent full revolution."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sweep = {}                  # bin -> (dist_mm, intensity)
        self.latest = []                 # list of [angle_deg, dist_mm, intensity]
        self.rpm = 0.0
        self.stamp = 0.0
        self.pkts = self.bad = 0
        self.last_angle = None

    def run(self):
        while True:
            try:
                self._read_forever()
            except Exception as e:
                print(f"[lidar] serial error: {e}; retrying in 2s", flush=True)
                time.sleep(2)

    def _find_port(self):
        for pat in (BY_ID_GLOB, FALLBACK_GLOB):
            hits = sorted(glob.glob(pat))
            if hits:
                return hits[0]
        raise IOError("no CP2102 serial device present")

    def _read_forever(self):
        dev = self._find_port()
        ser = serial.Serial(dev, BAUD, timeout=1)
        print(f"[lidar] open {dev} @ {BAUD}", flush=True)
        buf = bytearray()
        last_ok = time.time()
        while True:
            chunk = ser.read(512)
            # A vanished or wedged device reads as empty, not as an error. Treat a
            # silent port as a fault so the outer loop re-resolves the device path.
            if time.time() - last_ok > STALL_SECS:
                ser.close()
                raise IOError(f"no valid packet for {STALL_SECS}s on {dev}")
            if not chunk:
                continue
            before = self.pkts
            buf += chunk
            # Resync on the 0x54 0x2C header rather than assuming alignment --
            # a dropped byte would otherwise corrupt every packet from then on.
            i = 0
            while len(buf) - i >= PKT:
                if buf[i] != HDR or buf[i + 1] != VERLEN:
                    i += 1
                    continue
                pkt = bytes(buf[i:i + PKT])
                if _crc8(pkt[:-1]) != pkt[-1]:
                    self.bad += 1
                    i += 1
                    continue
                self._packet(pkt)
                self.pkts += 1
                i += PKT
            del buf[:i]
            if len(buf) > 4096:
                del buf[:-PKT]
            if self.pkts > before:
                last_ok = time.time()

    def _packet(self, p: bytes):
        speed = int.from_bytes(p[2:4], "little")            # deg/s
        start = int.from_bytes(p[4:6], "little") / 100.0    # deg
        end   = int.from_bytes(p[42:44], "little") / 100.0
        span  = (end - start) % 360.0
        step  = span / 11.0 if span else 0.0

        for n in range(12):
            off = 6 + n * 3
            dist = int.from_bytes(p[off:off + 2], "little")   # mm, 0 = invalid
            inten = p[off + 2]
            ang = (start + step * n) % 360.0
            if dist == 0:
                continue
            b = int(ang) % BINS
            with self.lock:
                self.sweep[b] = (dist, inten)

            # A wrap past 0 deg means the revolution finished -- publish it.
            if self.last_angle is not None and ang < self.last_angle - 180.0:
                self._finish(speed)
            self.last_angle = ang

    def _finish(self, speed):
        with self.lock:
            pts = [[b, d, i] for b, (d, i) in sorted(self.sweep.items())]
            self.latest = pts
            self.rpm = speed / 6.0            # deg/s -> rpm
            self.stamp = time.time()
            self.sweep = {}

    def snapshot(self):
        with self.lock:
            return {
                "ts": self.stamp,
                "age": round(time.time() - self.stamp, 3) if self.stamp else None,
                "rpm": round(self.rpm, 1),
                "packets": self.pkts,
                "crc_errors": self.bad,
                "count": len(self.latest),
                "points": self.latest,
            }


SCAN = Scanner()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/scan.json"):
            body = json.dumps(SCAN.snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def ros_publisher():
    """Publish /scan locally. Optional -- the console feed must not depend on it."""
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import LaserScan
    except Exception as e:
        print(f"[lidar] ROS unavailable ({e}); JSON feed only", flush=True)
        return

    rclpy.init()
    node = Node("ld06_lidar")
    pub = node.create_publisher(LaserScan, "/scan", 10)
    inc = 2 * math.pi / BINS

    def tick():
        snap = SCAN.snapshot()
        if not snap["points"]:
            return
        ranges = [float("inf")] * BINS
        inten = [0.0] * BINS
        for b, d, it in snap["points"]:
            ranges[b] = d / 1000.0
            inten[b] = float(it)
        m = LaserScan()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = "laser"
        m.angle_min, m.angle_max, m.angle_increment = 0.0, 2 * math.pi - inc, inc
        m.range_min, m.range_max = 0.02, 12.0
        m.scan_time = 0.1
        m.ranges, m.intensities = ranges, inten
        pub.publish(m)

    node.create_timer(0.1, tick)
    print("[lidar] publishing /scan", flush=True)
    rclpy.spin(node)


def main():
    threading.Thread(target=SCAN.run, daemon=True).start()
    threading.Thread(
        target=lambda: ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler).serve_forever(),
        daemon=True).start()
    print(f"[lidar] JSON feed at http://{socket.gethostname()}:{HTTP_PORT}/scan.json", flush=True)
    ros_publisher()
    while True:                      # if ROS is absent, keep the feed alive
        time.sleep(3600)


if __name__ == "__main__":
    main()
