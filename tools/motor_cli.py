#!/usr/bin/env python3
"""
Motor Control CLI — easy interactive control for 4 mecanum motors via Arduino.

Usage: python3 motor_cli.py
"""

import serial
import time
import sys

PORT = '/dev/ttyACM0'
BAUD = 115200

# Current motor speeds
motors = {'fl': 0, 'fr': 0, 'bl': 0, 'br': 0}

HELP = """
  Individual motors:
    fl <speed>          Front-left  (top-left)
    fr <speed>          Front-right (top-right)
    bl <speed>          Back-left   (bottom-left)
    br <speed>          Back-right  (bottom-right)
    all <speed>         All motors same speed

  Set all at once:
    set <fl> <fr> <bl> <br>

  Directions:
    fwd <speed>         Drive forward
    back <speed>        Drive backward
    left <speed>        Strafe left
    right <speed>       Strafe right
    cw <speed>          Spin clockwise
    ccw <speed>         Spin counter-clockwise

  Other:
    stop / s            Stop all motors
    status              Show current motor speeds
    help / h            Show this help
    quit / q            Stop motors and exit

  Speed range: -255 to 255
"""


def connect(port, baud):
    print(f"Connecting to Arduino on {port}...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # wait for Arduino reset
        ser.reset_input_buffer()
        # ping
        ser.write(b'P\n')
        time.sleep(0.3)
        resp = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
        if 'PONG' in resp:
            print("Connected! Arduino responded.")
        else:
            print(f"Connected but unexpected response: {resp!r}")
        return ser
    except serial.SerialException as e:
        print(f"ERROR: Could not connect: {e}")
        sys.exit(1)


def send(ser):
    """Send current motor state to Arduino."""
    cmd = f"M {motors['fl']} {motors['fr']} {motors['bl']} {motors['br']}\n"
    try:
        ser.write(cmd.encode())
        time.sleep(0.05)
        if ser.in_waiting:
            resp = ser.readline().decode('utf-8', errors='ignore').strip()
            return resp
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    return None


def clamp(val):
    return max(-255, min(255, int(val)))


def set_all(speed):
    for k in motors:
        motors[k] = clamp(speed)


def stop_all():
    for k in motors:
        motors[k] = 0


def show_status():
    print(f"  FL (top-left):     {motors['fl']:>4}")
    print(f"  FR (top-right):    {motors['fr']:>4}")
    print(f"  BL (bottom-left):  {motors['bl']:>4}")
    print(f"  BR (bottom-right): {motors['br']:>4}")


def main():
    ser = connect(PORT, BAUD)
    stop_all()
    send(ser)

    print("\nMotor Control CLI — type 'help' for commands\n")

    while True:
        try:
            raw = input("motor> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        try:
            if cmd in ('q', 'quit', 'exit'):
                break

            elif cmd in ('h', 'help'):
                print(HELP)
                continue

            elif cmd in ('s', 'stop'):
                stop_all()

            elif cmd == 'status':
                show_status()
                continue

            elif cmd in ('fl', 'fr', 'bl', 'br'):
                if len(parts) < 2:
                    print(f"  Usage: {cmd} <speed>")
                    continue
                motors[cmd] = clamp(parts[1])

            elif cmd == 'all':
                if len(parts) < 2:
                    print("  Usage: all <speed>")
                    continue
                set_all(int(parts[1]))

            elif cmd == 'set':
                if len(parts) < 5:
                    print("  Usage: set <fl> <fr> <bl> <br>")
                    continue
                motors['fl'] = clamp(parts[1])
                motors['fr'] = clamp(parts[2])
                motors['bl'] = clamp(parts[3])
                motors['br'] = clamp(parts[4])

            elif cmd == 'fwd':
                spd = clamp(parts[1]) if len(parts) > 1 else 150
                motors['fl'] = spd
                motors['fr'] = spd
                motors['bl'] = spd
                motors['br'] = spd

            elif cmd == 'back':
                spd = clamp(parts[1]) if len(parts) > 1 else 150
                motors['fl'] = -spd
                motors['fr'] = -spd
                motors['bl'] = -spd
                motors['br'] = -spd

            elif cmd == 'left':
                spd = clamp(parts[1]) if len(parts) > 1 else 150
                motors['fl'] = -spd
                motors['fr'] = spd
                motors['bl'] = spd
                motors['br'] = -spd

            elif cmd == 'right':
                spd = clamp(parts[1]) if len(parts) > 1 else 150
                motors['fl'] = spd
                motors['fr'] = -spd
                motors['bl'] = -spd
                motors['br'] = spd

            elif cmd == 'cw':
                spd = clamp(parts[1]) if len(parts) > 1 else 150
                motors['fl'] = spd
                motors['fr'] = -spd
                motors['bl'] = spd
                motors['br'] = -spd

            elif cmd == 'ccw':
                spd = clamp(parts[1]) if len(parts) > 1 else 150
                motors['fl'] = -spd
                motors['fr'] = spd
                motors['bl'] = -spd
                motors['br'] = spd

            else:
                print(f"  Unknown command: {cmd}  (type 'help')")
                continue

            resp = send(ser)
            show_status()
            if resp:
                print(f"  Arduino: {resp}")

        except (ValueError, IndexError):
            print("  Invalid input. Speed must be a number (-255 to 255)")

    # cleanup
    print("Stopping motors...")
    stop_all()
    send(ser)
    ser.close()
    print("Done.")


if __name__ == '__main__':
    main()
