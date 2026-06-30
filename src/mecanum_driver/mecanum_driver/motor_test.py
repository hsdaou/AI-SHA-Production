import os
import sys
import termios
import tty
import select
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

# ANSI helpers
ESC = '\033['
RESET = ESC + '0m'
BOLD = ESC + '1m'
DIM = ESC + '2m'
GREEN = ESC + '92m'
RED = ESC + '91m'
YELLOW = ESC + '93m'
CYAN = ESC + '96m'
MAGENTA = ESC + '95m'
WHITE = ESC + '97m'
BG_SEL = ESC + '48;5;236m'

BAR_HALF = 20  # chars per side of the gauge

MOTOR_NAMES = ['FRONT LEFT ', 'FRONT RIGHT', 'REAR LEFT  ', 'REAR RIGHT ']


class MotorTestNode(Node):
    def __init__(self):
        super().__init__('motor_test')
        self.pub = self.create_publisher(Int16MultiArray, 'mecanum/motor_cmd', 10)
        self.rpms = [0, 0, 0, 0]
        self.sel = 0          # 0 = ALL, 1-4 = single motor
        self.step = 10
        self.dirty = False
        self.timer = self.create_timer(0.02, self.tick)
        self.draw_timer = self.create_timer(0.1, self.draw_tick)

        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write(ESC + '?25l' + ESC + '2J')  # hide cursor, clear screen
        self.draw()

    # ---------- rendering ----------

    def gauge(self, val):
        """Two-sided bar: reverse fills left of center, forward fills right."""
        n = round(abs(val) * BAR_HALF / 255)
        if val > 0:
            left = DIM + '·' * BAR_HALF + RESET
            right = GREEN + '█' * n + RESET + DIM + '·' * (BAR_HALF - n) + RESET
        elif val < 0:
            left = DIM + '·' * (BAR_HALF - n) + RESET + RED + '█' * n + RESET
            right = DIM + '·' * BAR_HALF + RESET
        else:
            left = DIM + '·' * BAR_HALF + RESET
            right = DIM + '·' * BAR_HALF + RESET
        return left + WHITE + BOLD + '│' + RESET + right

    def motor_row(self, i):
        selected = (self.sel == 0) or (self.sel == i + 1)
        val = self.rpms[i]
        if val > 0:
            color, arrow = GREEN, '▲ FWD'
        elif val < 0:
            color, arrow = RED, '▼ REV'
        else:
            color, arrow = DIM, '■ STOP'
        mark = CYAN + BOLD + ' ▶ ' + RESET if selected else '   '
        bg = BG_SEL if selected else ''
        name = (BOLD if selected else DIM) + f'{i + 1} {MOTOR_NAMES[i]}' + RESET
        return (f'{mark}{bg}{name} {self.gauge(val)}{bg} '
                f'{color}{BOLD}{val:+4d}{RESET}{bg} {color}{arrow:<6}{RESET}'
                + ESC + 'K')

    def draw(self):
        sel_label = 'ALL MOTORS' if self.sel == 0 else f'MOTOR {self.sel} ONLY'
        moving = any(self.rpms)
        status = (YELLOW + BOLD + '● MOVING' if moving else DIM + '○ idle') + RESET
        w = 78
        out = [
            ESC + 'H',  # cursor home
            CYAN + BOLD + '╔' + '═' * w + '╗' + RESET,
            CYAN + BOLD + '║' + RESET
            + BOLD + '   🛞  MECANUM MOTOR TEST '.ljust(w - 24) + RESET
            + status.ljust(len(status) + 4)
            + CYAN + BOLD + ' ║' + RESET + ESC + 'K',
            CYAN + BOLD + '╚' + '═' * w + '╝' + RESET,
            '',
            f'   {MAGENTA}{BOLD}CONTROLLING: {sel_label}{RESET}'
            f'   {DIM}step size:{RESET} {BOLD}{self.step}{RESET}' + ESC + 'K',
            '',
            self.motor_row(0),
            self.motor_row(1),
            self.motor_row(2),
            self.motor_row(3),
            '',
            DIM + '   ' + '─' * w + RESET,
            f'   {BOLD}{CYAN}↑ / ↓{RESET}  faster / slower      '
            f'{BOLD}{CYAN}SPACE{RESET}  ⛔ STOP EVERYTHING' + ESC + 'K',
            f'   {BOLD}{CYAN}1 2 3 4{RESET}  pick one motor      '
            f'{BOLD}{CYAN}5{RESET}      control all 4 together' + ESC + 'K',
            f'   {BOLD}{CYAN}+ / -{RESET}  bigger/smaller steps  '
            f'{BOLD}{CYAN}q{RESET}      quit (stops motors)' + ESC + 'K',
            DIM + '   ' + '─' * w + RESET,
        ]
        sys.stdout.write('\n'.join(out) + '\n')
        sys.stdout.flush()

    # ---------- input ----------

    def read_keys(self):
        """Grab every buffered byte at once and parse into key events."""
        data = b''
        while select.select([sys.stdin], [], [], 0.0)[0]:
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                break
            data += chunk
        keys = []
        i = 0
        while i < len(data):
            if data[i:i + 3] == b'\x1b[A':
                keys.append('UP')
                i += 3
            elif data[i:i + 3] == b'\x1b[B':
                keys.append('DOWN')
                i += 3
            elif data[i] == 0x1b:
                i += 1  # other escape sequence intro — skip
            else:
                keys.append(chr(data[i]))
                i += 1
        return keys

    def adjust(self, delta):
        idxs = range(4) if self.sel == 0 else [self.sel - 1]
        for i in idxs:
            self.rpms[i] = max(-255, min(255, self.rpms[i] + delta))

    def publish(self):
        msg = Int16MultiArray()
        msg.data = list(self.rpms)
        self.pub.publish(msg)

    def tick(self):
        changed = False
        for key in self.read_keys():
            changed = True
            if key == 'UP':
                self.adjust(self.step)
            elif key == 'DOWN':
                self.adjust(-self.step)
            elif key == ' ':
                self.rpms = [0, 0, 0, 0]
            elif key in '1234':
                self.sel = int(key)
            elif key == '5':
                self.sel = 0
            elif key in '+=':
                self.step = min(255, self.step + 5)
            elif key == '-':
                self.step = max(1, self.step - 5)
            elif key in 'qQ':
                self.rpms = [0, 0, 0, 0]
                self.publish()
                raise SystemExit
        # Heartbeat: publish every tick so the Arduino's 1s safety
        # timeout never trips while this node is running
        self.publish()
        if changed:
            self.dirty = True

    def draw_tick(self):
        # Redraw at most 10x/sec so terminal output never blocks input
        if self.dirty:
            self.dirty = False
            self.draw()

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        sys.stdout.write(ESC + '?25h' + RESET + '\n')  # show cursor again
        sys.stdout.flush()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorTestNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.rpms = [0, 0, 0, 0]
        node.publish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
