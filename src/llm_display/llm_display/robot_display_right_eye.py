#!/usr/bin/env python3
"""
Robot Speech Display - Animated single-panel display
Shows STT input and LLM responses with smooth transitions
Includes pause button to interrupt TTS
"""

import sys
import signal
import threading
import math
import random
import subprocess
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String, Empty, Bool

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsOpacityEffect, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QPropertyAnimation,
    QEasingCurve, QSize
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPainter, QPen, QBrush,
    QPainterPath, QRadialGradient
)


class SignalBridge(QObject):
    """Bridge for thread-safe GUI updates"""
    user_message = pyqtSignal(str)
    robot_message = pyqtSignal(str)
    pause_response = pyqtSignal(bool)
    deactivated = pyqtSignal()
    audio_playing = pyqtSignal(bool)
    blink_trigger = pyqtSignal()


class RobotDisplayNode(Node):
    def __init__(self, signal_bridge):
        super().__init__('robot_display_right_eye')
        self.signal_bridge = signal_bridge

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.stt_sub = self.create_subscription(
            String, '/speech/text', self.stt_callback, qos_profile)
        self.tts_sub = self.create_subscription(
            String, '/tts_text', self.tts_callback, qos_profile)

        # Subscriber for session deactivation (STT idle for 25s)
        self.deactivated_sub = self.create_subscription(
            Empty, '/session/deactivated', self.deactivated_callback, qos_profile)

        # Subscriber for audio playing status - controls when to return to eyes
        self.audio_playing_sub = self.create_subscription(
            Bool, '/audio/playing', self.audio_playing_callback, qos_profile)

        # Subscriber for synchronized eye blinks (Boolean state-based)
        blink_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.blink_sub = self.create_subscription(
            Bool, '/eye/blink_state', self.blink_state_callback, blink_qos)

        # Publisher for synchronized blinks (right eye is master)
        self.blink_pub = self.create_publisher(Bool, '/eye/blink_state', blink_qos)

        # Publisher for pause command
        self.pause_pub = self.create_publisher(Empty, '/pause', qos_profile)

        # Publisher to command TTS to speak a specific message
        self.tts_speak_pub = self.create_publisher(String, '/tts_speak', qos_profile)

        # Publisher to unmute the microphone on STT node
        self.mic_unmute_pub = self.create_publisher(Empty, '/mic/unmute', qos_profile)

        self.get_logger().info('Robot Display started - listening on /speech/text and /tts_text')
        self.get_logger().info('Pause publisher ready on /pause')
        self.get_logger().info('TTS speak publisher ready on /tts_speak')
        self.get_logger().info('Mic unmute publisher ready on /mic/unmute')

    def stt_callback(self, msg):
        # RIGHT EYE: Ignore STT messages
        self.get_logger().debug("STT message ignored on right eye")

    def tts_callback(self, msg):
        self.signal_bridge.robot_message.emit(msg.data)

    def deactivated_callback(self, msg):
        self.get_logger().info('Session deactivated - switching to eye mode')
        self.signal_bridge.deactivated.emit()

    def audio_playing_callback(self, msg):
        """Handle audio playing status - return to eyes when audio stops"""
        is_playing = msg.data
        self.get_logger().info(f'Audio playing status: {is_playing}')
        if not is_playing:
            # Audio stopped - return to eye mode
            self.signal_bridge.audio_playing.emit(False)

    def blink_state_callback(self, msg):
        """Trigger synchronized blink when state becomes True"""
        import time
        should_blink = msg.data
        if should_blink:
            self.get_logger().info(f'[RIGHT EYE] Blink TRUE at {time.time():.3f}')
            self.signal_bridge.blink_trigger.emit()
        else:
            self.get_logger().debug(f'[RIGHT EYE] Blink FALSE at {time.time():.3f}')

    def publish_blink(self):
        """Publish blink state TRUE for synchronized blinking"""
        msg = Bool()
        msg.data = True
        self.blink_pub.publish(msg)
        self.get_logger().info('Published blink state: TRUE')

    def publish_blink_done(self):
        """Publish blink state FALSE after blink completes"""
        msg = Bool()
        msg.data = False
        self.blink_pub.publish(msg)
        self.get_logger().debug('Published blink state: FALSE')

    def publish_pause(self):
        """Publish pause command to stop TTS/LLM with multiple fallback mechanisms"""
        # Method 1: Publish to ROS2 topic
        msg = Empty()
        self.pause_pub.publish(msg)
        self.get_logger().info('Pause command sent to /pause topic')

        # Method 2: SSH fallback to create pause signal file on Pi5 (TTS node)
        # This bypasses potential DDS discovery issues between Humble and Jazzy
        def ssh_pause_pi5():
            try:
                cmd = [
                    'sshpass', '-p', 'aisharjah123',
                    'ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=2',
                    'pi5@pi5.local',
                    'touch /tmp/tts_pause_signal'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=3)
                if result.returncode == 0:
                    self.get_logger().info('SSH pause signal file created on Pi5')
                else:
                    self.get_logger().warn(f'SSH pause to Pi5 failed: {result.stderr.decode()}')
            except Exception as e:
                self.get_logger().warn(f'SSH pause fallback failed: {e}')

        # Run SSH in background thread to not block
        threading.Thread(target=ssh_pause_pi5, daemon=True).start()

        self.signal_bridge.pause_response.emit(True)

    def publish_tts_speak(self, text):
        """Command TTS node to speak a specific message"""
        msg = String()
        msg.data = text
        self.tts_speak_pub.publish(msg)
        self.get_logger().info(f'TTS speak command sent: {text}')

    def publish_mic_unmute(self):
        """Send unmute signal to STT/mic node"""
        msg = Empty()
        self.mic_unmute_pub.publish(msg)
        self.get_logger().info('Mic unmute signal sent to /mic/unmute')


class PulsingDotsWidget(QWidget):
    """Animated pulsing dots indicator"""
    def __init__(self, parent=None, color="#00d9ff", num_dots=3):
        super().__init__(parent)
        self.color = QColor(color)
        self.num_dots = num_dots
        self.phase = 0
        self.setFixedSize(120, 40)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)

    def start(self):
        self.timer.start(80)
        self.show()

    def stop(self):
        self.timer.stop()
        self.hide()

    def set_color(self, color):
        self.color = QColor(color)

    def animate(self):
        self.phase += 0.3
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        dot_spacing = 30
        start_x = (self.width() - (self.num_dots - 1) * dot_spacing) // 2
        center_y = self.height() // 2

        for i in range(self.num_dots):
            # Calculate size based on phase offset
            offset = i * 0.8
            scale = 0.5 + 0.5 * math.sin(self.phase + offset)
            radius = int(6 + 6 * scale)

            # Calculate opacity
            opacity = 0.4 + 0.6 * scale
            color = QColor(self.color)
            color.setAlphaF(opacity)

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)

            x = start_x + i * dot_spacing
            painter.drawEllipse(x - radius, center_y - radius, radius * 2, radius * 2)


class SpinningStarsWidget(QWidget):
    """Animated spinning stars indicator"""
    def __init__(self, parent=None, color="#00ff88"):
        super().__init__(parent)
        self.color = QColor(color)
        self.angle = 0
        self.setFixedSize(60, 60)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)

    def start(self):
        self.timer.start(50)
        self.show()

    def stop(self):
        self.timer.stop()
        self.hide()

    def set_color(self, color):
        self.color = QColor(color)

    def animate(self):
        self.angle += 5
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)

        # Draw 4 points like a star/sparkle
        for i in range(4):
            painter.rotate(90)
            opacity = 0.5 + 0.5 * math.sin(math.radians(self.angle + i * 90))
            color = QColor(self.color)
            color.setAlphaF(opacity)

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)

            # Draw elongated diamond shape
            points = [
                (-3, 0), (0, -20), (3, 0), (0, -8)
            ]
            from PyQt5.QtGui import QPolygon
            from PyQt5.QtCore import QPoint
            polygon = QPolygon([QPoint(int(p[0]), int(p[1])) for p in points])
            painter.drawPolygon(polygon)


class BlinkingEyeWidget(QWidget):
    """Full-screen blinking eye widget for deactivated/idle state"""
    blink_started = pyqtSignal()  # Signal emitted when internal blink starts

    def __init__(self, parent=None, side='left', use_internal_timer=False, ros_node=None):
        super().__init__(parent)
        self.side = side
        self.use_internal_timer = use_internal_timer
        self.ros_node = ros_node
        self.blink_progress = 0.0  # 0.0 = fully open, 1.0 = fully closed
        self.is_blinking = False
        self.blink_direction = 1  # 1 = closing, -1 = opening
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #000000;")

        # Animation timer (~60fps)
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._animate)

        # Blink trigger timer (only used if use_internal_timer=True)
        self.blink_trigger = QTimer()
        self.blink_trigger.setSingleShot(True)
        self.blink_trigger.timeout.connect(self._start_blink)

    def start(self):
        self.blink_progress = 0.0
        self.is_blinking = False
        self.anim_timer.start(16)
        if self.use_internal_timer:
            self._schedule_blink()
        self.show()

    def stop(self):
        self.anim_timer.stop()
        self.blink_trigger.stop()
        self.hide()

    def _schedule_blink(self):
        if self.use_internal_timer:
            delay = random.randint(2500, 5500)
            self.blink_trigger.start(delay)

    def _start_blink(self):
        # If this is the master (right eye), ONLY publish - don't blink yet
        # Both eyes will blink when they receive the /eye/blink message
        if self.use_internal_timer and self.ros_node:
            self.ros_node.publish_blink()
        else:
            # Follower (left eye without timer) - blink immediately
            self.is_blinking = True
            self.blink_direction = 1

    def trigger_blink(self):
        """Externally trigger a blink (for synchronized blinking)"""
        if not self.is_blinking:
            self.is_blinking = True
            self.blink_direction = 1

    def _animate(self):
        if self.is_blinking:
            speed = 0.19  # Faster, snappier blink (MUST match Pi4 exactly)
            self.blink_progress += speed * self.blink_direction
            if self.blink_progress >= 1.0:
                self.blink_progress = 1.0
                self.blink_direction = -1
            elif self.blink_progress <= 0.0:
                self.blink_progress = 0.0
                self.is_blinking = False
                # Notify that blink completed (for master to publish FALSE)
                if self.use_internal_timer and self.ros_node:
                    self.ros_node.publish_blink_done()
                self._schedule_blink()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2
        cy = h / 2

        # Black background
        painter.fillRect(self.rect(), QColor('#000000'))

        # Eye dimensions - rounded square (squircle), nearly 1:1 aspect
        eye_size = min(w, h) * 0.65
        eye_w = eye_size
        eye_h_full = eye_size * 0.85
        corner_radius = eye_size * 0.3

        # Apply blink - shrink height vertically, stay centered
        open_amount = 1.0 - self.blink_progress
        eye_h = eye_h_full * max(open_amount, 0.0)

        if eye_h < 2:
            # Fully closed - draw thin cyan line
            painter.setPen(QPen(QColor('#00e0ff'), 3, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(int(cx - eye_w / 2), int(cy),
                             int(cx + eye_w / 2), int(cy))
            return

        # Center the eye
        eye_x = cx - eye_w / 2
        eye_y = cy - eye_h / 2

        r = min(corner_radius, eye_h / 2)

        # Subtle outer glow
        for i in range(3, 0, -1):
            glow_alpha = 25 * (4 - i)
            glow_pen = QPen(QColor(0, 224, 255, glow_alpha), i * 4)
            painter.setPen(glow_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(int(eye_x - i * 2), int(eye_y - i * 2),
                                    int(eye_w + i * 4), int(eye_h + i * 4),
                                    r + i * 2, r + i * 2)

        # Radial gradient fill: bright cyan center, darker teal edges
        gradient = QRadialGradient(cx, cy, max(eye_w, eye_h) * 0.6)
        gradient.setColorAt(0.0, QColor('#00f5ff'))
        gradient.setColorAt(0.5, QColor('#00d8ee'))
        gradient.setColorAt(0.8, QColor('#00a0b0'))
        gradient.setColorAt(1.0, QColor('#006570'))

        # Dark teal border
        painter.setPen(QPen(QColor('#005060'), 5))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(int(eye_x), int(eye_y),
                                int(eye_w), int(eye_h), r, r)


class WordByWordLabel(QLabel):
    """Label that displays text word by word with animation"""
    finished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_text = ""
        self.displayed_words = []
        self.word_index = 0
        self.char_index = 0

        self.word_timer = QTimer()
        self.word_timer.timeout.connect(self.show_next_word)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)

    def set_text_animated(self, text, word_delay=80):
        """Start word-by-word animation"""
        self.word_timer.stop()
        self.full_text = text
        self.words = text.split()
        self.word_index = 0
        self.setText("")

        if self.words:
            self.word_timer.start(word_delay)

    def show_next_word(self):
        if self.word_index < len(self.words):
            self.word_index += 1
            display_text = ' '.join(self.words[:self.word_index])
            self.setText(display_text)
        else:
            self.word_timer.stop()
            self.finished.emit()

    def set_text_instant(self, text):
        """Set text immediately without animation"""
        self.word_timer.stop()
        self.full_text = text
        self.setText(text)

    def stop_animation(self):
        self.word_timer.stop()
        self.setText(self.full_text)

    def fade_in(self, duration=300):
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(duration)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.start()

    def fade_out(self, duration=300):
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(duration)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.start()


class RobotDisplayWindow(QMainWindow):
    # Display modes
    MODE_IDLE = 0
    MODE_STT = 1  # User speaking
    MODE_TTS = 2  # Robot responding
    MODE_EYE = 3  # Deactivated - showing blinking eye

    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.current_mode = self.MODE_IDLE
        self.tts_active = False

        self.setWindowTitle("Robot Assistant - RIGHT EYE")
        self.showFullScreen()

        # Main widget
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.main_layout = QVBoxLayout(self.central)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(20)

        # Header with status indicator
        self.setup_header()

        # Main content area (single panel with scroll)
        self.setup_content()

        # Pause button (hidden by default)
        self.setup_pause_button()

        # Footer
        self.setup_footer()

        # Animation timers (must be created before apply_idle_style)
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.pulse_animation)
        self.pulse_value = 0

        # Idle timeout - return to idle after inactivity
        self.idle_timer = QTimer()
        self.idle_timer.timeout.connect(self.return_to_idle)
        self.idle_timer.setSingleShot(True)

        # Eye widget for deactivated mode (overlays everything)
        # Right eye is master - uses internal timer and publishes blinks
        self.eye_widget = BlinkingEyeWidget(parent=self.central, side='right', use_internal_timer=True, ros_node=self.ros_node)
        self.eye_widget.hide()

        # Apply initial style
        self.apply_idle_style()

    def setup_header(self):
        header = QHBoxLayout()

        # Mode indicator with animation widget
        self.mode_container = QWidget()
        mode_layout = QHBoxLayout(self.mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(10)

        # Mode icon
        self.mode_icon = QLabel()
        self.mode_icon.setFixedSize(50, 50)
        self.mode_icon.setAlignment(Qt.AlignCenter)
        self.mode_icon.setStyleSheet("font-size: 36px; background: transparent;")
        self.mode_icon.setText("🤖")
        mode_layout.addWidget(self.mode_icon)

        # Pulsing dots for listening
        self.listening_dots = PulsingDotsWidget(color="#00d9ff")
        self.listening_dots.hide()
        mode_layout.addWidget(self.listening_dots)

        # Spinning stars for speaking
        self.speaking_stars = SpinningStarsWidget(color="#00ff88")
        self.speaking_stars.hide()
        mode_layout.addWidget(self.speaking_stars)

        header.addWidget(self.mode_container)

        # Title
        self.title_label = QLabel("Robot Response")
        self.title_label.setStyleSheet("""
            color: white;
            font-size: 28px;
            font-weight: bold;
            background: transparent;
        """)
        header.addWidget(self.title_label)

        header.addStretch()

        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            color: #00ff88;
            font-size: 18px;
            font-weight: bold;
            background: transparent;
            padding: 10px 20px;
            border-radius: 20px;
            background-color: rgba(0, 255, 136, 0.1);
        """)
        header.addWidget(self.status_label)

        self.main_layout.addLayout(header)

    def setup_content(self):
        # Scroll area for long text
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.1);
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Content widget inside scroll area
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)

        # Main text display with word-by-word animation
        self.main_text = WordByWordLabel()
        self.main_text.setWordWrap(True)
        self.main_text.setAlignment(Qt.AlignCenter)
        self.main_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_text.setMinimumHeight(200)
        self.main_text.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 48px;
                font-weight: 500;
                padding: 40px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
            }
        """)
        self.main_text.setText("Waiting for input...")
        content_layout.addWidget(self.main_text)

        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area, 1)

    def setup_pause_button(self):
        # Pause button REMOVED from right eye (Pi5) - now on left eye (Pi4) only
        # Create empty container for compatibility
        self.pause_container = QWidget()
        self.pause_container.hide()
        # Don't add to layout - pause button not needed on right eye

    def setup_footer(self):
        self.footer = QLabel("ESC = toggle fullscreen  |  SPACE = pause  |  Ctrl+Q = quit")
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setStyleSheet("""
            color: #666666;
            font-size: 14px;
            padding: 10px;
            background: transparent;
        """)
        self.main_layout.addWidget(self.footer)

    def apply_idle_style(self):
        """Apply idle/waiting style"""
        self.central.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e,
                    stop:1 #16213e
                );
            }
        """)
        self.mode_icon.setText("🤖")
        self.listening_dots.stop()
        self.speaking_stars.stop()
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("""
            color: #00ff88;
            font-size: 18px;
            font-weight: bold;
            background: transparent;
            padding: 10px 20px;
            border-radius: 20px;
            background-color: rgba(0, 255, 136, 0.1);
        """)
        self.main_text.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 36px;
                font-weight: 400;
                padding: 40px;
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 20px;
            }
        """)
        self.main_text.set_text_instant("Waiting for input...")
        self.pause_container.hide()
        self.pulse_timer.stop()

    def apply_stt_style(self):
        """Apply STT (user speaking) style - cyan/blue theme"""
        self.central.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a1628,
                    stop:1 #162447
                );
            }
        """)
        self.mode_icon.setText("🎤")
        self.speaking_stars.stop()
        self.listening_dots.set_color("#00d9ff")
        self.listening_dots.start()

        self.status_label.setText("Listening...")
        self.status_label.setStyleSheet("""
            color: #00d9ff;
            font-size: 18px;
            font-weight: bold;
            background: transparent;
            padding: 10px 20px;
            border-radius: 20px;
            background-color: rgba(0, 217, 255, 0.15);
        """)
        self.main_text.setStyleSheet("""
            QLabel {
                color: #00d9ff;
                font-size: 52px;
                font-weight: 600;
                padding: 50px;
                background-color: rgba(0, 217, 255, 0.08);
                border-radius: 25px;
                border: 2px solid rgba(0, 217, 255, 0.3);
            }
        """)
        self.pause_container.hide()
        self.pulse_timer.start(50)

    def apply_tts_style(self):
        """Apply TTS (robot speaking) style - green theme"""
        self.central.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a2818,
                    stop:1 #1a4730
                );
            }
        """)
        self.mode_icon.setText("🤖")
        self.listening_dots.stop()
        self.speaking_stars.set_color("#00ff88")
        self.speaking_stars.start()

        self.status_label.setText("Speaking...")
        self.status_label.setStyleSheet("""
            color: #00ff88;
            font-size: 18px;
            font-weight: bold;
            background: transparent;
            padding: 10px 20px;
            border-radius: 20px;
            background-color: rgba(0, 255, 136, 0.15);
        """)
        self.main_text.setStyleSheet("""
            QLabel {
                color: #00ff88;
                font-size: 42px;
                font-weight: 500;
                padding: 50px;
                background-color: rgba(0, 255, 136, 0.08);
                border-radius: 25px;
                border: 2px solid rgba(0, 255, 136, 0.3);
            }
        """)
        self.pause_container.hide()  # No pause button on right eye
        self.tts_active = True
        self.pulse_timer.start(50)

    def pulse_animation(self):
        """Create subtle pulsing effect on status"""
        self.pulse_value += 0.1
        opacity = 0.7 + 0.3 * math.sin(self.pulse_value)

        if self.current_mode == self.MODE_STT:
            color = f"rgba(0, 217, 255, {opacity})"
        elif self.current_mode == self.MODE_TTS:
            color = f"rgba(0, 255, 136, {opacity})"
        else:
            return

        self.status_label.setStyleSheet(f"""
            color: {color};
            font-size: 18px;
            font-weight: bold;
            background: transparent;
            padding: 10px 20px;
            border-radius: 20px;
            background-color: rgba(255, 255, 255, 0.05);
        """)

    def show_user_message(self, text):
        """Display user's speech (STT)"""
        if self.current_mode == self.MODE_EYE:
            self.eye_widget.stop()
        self.idle_timer.stop()

        if self.current_mode != self.MODE_STT:
            self.current_mode = self.MODE_STT
            self.main_text.fade_out(150)
            QTimer.singleShot(150, lambda: self._update_stt_text(text))
        else:
            self._update_stt_text(text)

        # Don't use idle timer - wait for audio_playing to become False

    def _update_stt_text(self, text):
        self.apply_stt_style()
        # For STT, show text instantly (it's what they said)
        self.main_text.set_text_instant(f'"{text}"')
        self.main_text.fade_in(200)
        # Scroll to top
        self.scroll_area.verticalScrollBar().setValue(0)

    def show_robot_message(self, text):
        """Display robot's response (TTS)"""
        if self.current_mode == self.MODE_EYE:
            self.eye_widget.stop()
        self.idle_timer.stop()

        if self.current_mode != self.MODE_TTS:
            self.current_mode = self.MODE_TTS
            self.main_text.fade_out(150)
            QTimer.singleShot(150, lambda: self._update_tts_text(text))
        else:
            self._update_tts_text(text)

        # Don't use idle timer - wait for audio_playing to become False

    def _update_tts_text(self, text):
        self.apply_tts_style()
        # For TTS, animate word by word
        self.main_text.set_text_animated(text, word_delay=100)
        self.main_text.fade_in(200)
        # Auto-scroll as text appears
        self.main_text.word_timer.timeout.connect(self._auto_scroll)

    def _auto_scroll(self):
        """Auto-scroll to bottom as words appear"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def return_to_idle(self):
        """Return to idle state after inactivity"""
        self.current_mode = self.MODE_IDLE
        self.tts_active = False
        self.main_text.stop_animation()
        self.main_text.fade_out(300)
        QTimer.singleShot(300, self.apply_idle_style)
        QTimer.singleShot(350, lambda: self.main_text.fade_in(300))

    def show_eye_mode(self):
        """Switch to blinking eye display when session is deactivated"""
        if self.current_mode == self.MODE_EYE:
            return
        self.idle_timer.stop()
        self.pulse_timer.stop()
        self.listening_dots.stop()
        self.speaking_stars.stop()
        self.main_text.stop_animation()
        self.tts_active = False
        self.current_mode = self.MODE_EYE

        # Resize and show eye widget over everything
        self.eye_widget.setGeometry(self.central.rect())
        self.eye_widget.raise_()
        self.eye_widget.start()

    def on_audio_stopped(self, is_playing):
        """Return to eye mode when audio stops playing (after 2 second delay)"""
        if not is_playing:
            # Wait 2 seconds before returning to eye mode
            QTimer.singleShot(2000, self.show_eye_mode)

    def on_blink_trigger(self):
        """Trigger synchronized blink immediately"""
        if self.current_mode == self.MODE_EYE:
            # Blink immediately when message received - no artificial delay
            self.eye_widget.trigger_blink()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_mode == self.MODE_EYE:
            self.eye_widget.setGeometry(self.central.rect())

    def on_pause_clicked(self):
        """Handle pause button click"""
        if self.ros_node and self.tts_active:
            self.ros_node.publish_pause()
            self.show_pause_feedback()
            # Tell TTS to play "Go ahead, I'm listening"
            self.ros_node.publish_tts_speak("Go ahead, I'm listening")
            # After 1.5 seconds, send signal to unmute the mic
            QTimer.singleShot(1500, self.ros_node.publish_mic_unmute)

    def show_pause_feedback(self):
        """Show visual feedback when pause is triggered"""
        self.main_text.stop_animation()
        self.speaking_stars.stop()

        self.pause_btn.setText("⏹  PAUSED")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffa502;
                color: white;
                font-size: 22px;
                font-weight: bold;
                border: none;
                border-radius: 32px;
            }
        """)
        self.status_label.setText("Paused")
        self.tts_active = False

        # Reset after delay
        QTimer.singleShot(2000, self.reset_pause_button)
        QTimer.singleShot(3000, self.return_to_idle)

    def reset_pause_button(self):
        """Reset pause button to default state"""
        self.pause_btn.setText("⏸  PAUSE")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4757;
                color: white;
                font-size: 22px;
                font-weight: bold;
                border: none;
                border-radius: 32px;
            }
            QPushButton:hover {
                background-color: #ff6b7a;
            }
            QPushButton:pressed {
                background-color: #ee3344;
            }
        """)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_Space:
            if self.tts_active:
                self.on_pause_clicked()
        elif event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Sans Serif", 12)
    app.setFont(font)

    # Signal bridge for thread-safe updates
    bridge = SignalBridge()

    # Initialize ROS2
    rclpy.init()
    node = RobotDisplayNode(bridge)

    # Create window with ROS node reference
    window = RobotDisplayWindow(node)

    # Connect signals
    bridge.user_message.connect(window.show_user_message)
    bridge.robot_message.connect(window.show_robot_message)
    bridge.deactivated.connect(window.show_eye_mode)
    bridge.audio_playing.connect(window.on_audio_stopped)
    bridge.blink_trigger.connect(window.on_blink_trigger)

    window.show()

    # ROS2 spin in separate thread
    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    ros_thread.start()

    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
