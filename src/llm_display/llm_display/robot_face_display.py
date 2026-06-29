#!/usr/bin/env python3
"""
Robot Face Display with Animations
Visual robot interface with animated face, status indicators, and dashboard
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFrame, QGridLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsEllipseItem)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QRectF, QPointF
from PyQt5.QtGui import QFont, QPainter, QColor, QBrush, QPen, QLinearGradient, QRadialGradient
import sys
from datetime import datetime
import math


class AnimatedEye(QGraphicsEllipseItem):
    """Animated robot eye"""
    def __init__(self, x, y, size, parent=None):
        super().__init__(0, 0, size, size, parent)
        self.base_x = x
        self.base_y = y
        self.size = size
        self.setPos(x, y)

        # Eye colors
        gradient = QRadialGradient(size/2, size/2, size/2)
        gradient.setColorAt(0, QColor(100, 200, 255))
        gradient.setColorAt(0.7, QColor(50, 150, 255))
        gradient.setColorAt(1, QColor(0, 100, 200))
        self.setBrush(QBrush(gradient))
        self.setPen(QPen(QColor(255, 255, 255), 3))

        # Pupil
        self.pupil = QGraphicsEllipseItem(size*0.3, size*0.3, size*0.4, size*0.4, self)
        self.pupil.setBrush(QBrush(QColor(0, 0, 50)))
        self.pupil_offset_x = 0
        self.pupil_offset_y = 0

        # Animation state
        self.blink_state = 0  # 0 = open, 1 = closing, 2 = opening
        self.original_rect = QRectF(0, 0, size, size)

    def blink(self):
        """Blink animation"""
        # Animate by changing the rect height
        self.blink_timer = QTimer()
        self.blink_step = 0
        self.blink_timer.timeout.connect(self._animate_blink)
        self.blink_timer.start(20)

    def _animate_blink(self):
        """Animate blink step"""
        self.blink_step += 1
        if self.blink_step <= 5:
            # Closing
            scale = 1.0 - (self.blink_step / 5.0) * 0.9
            self.setRect(0, self.size * (1 - scale) / 2, self.size, self.size * scale)
        elif self.blink_step <= 10:
            # Opening
            scale = 0.1 + ((self.blink_step - 5) / 5.0) * 0.9
            self.setRect(0, self.size * (1 - scale) / 2, self.size, self.size * scale)
        else:
            # Done
            self.setRect(self.original_rect)
            self.blink_timer.stop()
            self.blink_step = 0

    def look_at(self, direction):
        """Move pupil to look in direction"""
        offset = self.size * 0.1
        if direction == "left":
            self.pupil.setPos(self.size*0.2, self.size*0.3)
        elif direction == "right":
            self.pupil.setPos(self.size*0.4, self.size*0.3)
        elif direction == "up":
            self.pupil.setPos(self.size*0.3, self.size*0.2)
        elif direction == "down":
            self.pupil.setPos(self.size*0.3, self.size*0.4)
        else:  # center
            self.pupil.setPos(self.size*0.3, self.size*0.3)


class WaveformWidget(QWidget):
    """Animated waveform for audio visualization"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.bars = [0] * 30
        self.active = False
        self.animation_step = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        bar_width = width / len(self.bars)

        # Background
        painter.fillRect(0, 0, width, height, QColor(26, 26, 46))

        # Draw bars
        for i, bar_height in enumerate(self.bars):
            x = i * bar_width
            bar_h = bar_height * height

            # Color gradient
            if self.active:
                color = QColor(46, 204, 113)  # Green when active
            else:
                color = QColor(52, 152, 219)  # Blue when inactive

            painter.fillRect(int(x + 2), int(height - bar_h),
                           int(bar_width - 4), int(bar_h), color)

    def animate(self):
        """Animate waveform"""
        if self.active:
            self.animation_step += 1
            for i in range(len(self.bars)):
                # Create wave pattern
                self.bars[i] = abs(math.sin((i + self.animation_step) * 0.3)) * 0.8
        else:
            # Decay to zero
            for i in range(len(self.bars)):
                self.bars[i] *= 0.9

        self.update()

    def set_active(self, active):
        """Set waveform active state"""
        self.active = active


class StatusIndicator(QFrame):
    """Animated status indicator"""
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.status_label = label
        self.is_active = False
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout()

        # Status LED
        self.led = QLabel("●")
        self.led.setFont(QFont("Ubuntu", 24))
        self.led.setStyleSheet("color: #7f8c8d;")

        # Label
        label = QLabel(self.status_label)
        label.setFont(QFont("Ubuntu", 14))
        label.setStyleSheet("color: white;")

        layout.addWidget(self.led)
        layout.addWidget(label)
        layout.addStretch()

        self.setLayout(layout)
        self.setStyleSheet("""
            background-color: rgba(52, 73, 94, 0.6);
            border-radius: 10px;
            padding: 10px;
        """)

    def set_active(self, active):
        """Set indicator state"""
        self.is_active = active
        if active:
            self.led.setStyleSheet("color: #2ecc71;")  # Green
        else:
            self.led.setStyleSheet("color: #7f8c8d;")  # Gray


class StatCard(QFrame):
    """Dashboard stat card"""
    def __init__(self, title, value="0", unit="", color="#3498db", parent=None):
        super().__init__(parent)
        self.title = title
        self.value_text = value
        self.unit_text = unit
        self.color = color
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Ubuntu", 11))
        title_label.setStyleSheet(f"color: {self.color};")
        title_label.setAlignment(Qt.AlignCenter)

        # Value
        self.value_label = QLabel(self.value_text)
        self.value_label.setFont(QFont("Ubuntu", 22, QFont.Bold))
        self.value_label.setStyleSheet("color: white;")
        self.value_label.setAlignment(Qt.AlignCenter)

        # Unit
        self.unit_label = QLabel(self.unit_text)
        self.unit_label.setFont(QFont("Ubuntu", 10))
        self.unit_label.setStyleSheet("color: #95a5a6;")
        self.unit_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        if self.unit_text:
            layout.addWidget(self.unit_label)

        self.setLayout(layout)
        self.setStyleSheet(f"""
            background-color: rgba(52, 73, 94, 0.7);
            border-radius: 12px;
            border: 2px solid {self.color};
            padding: 15px;
        """)
        self.setMinimumHeight(120)

    def update_value(self, value, unit=""):
        self.value_label.setText(str(value))
        if unit:
            self.unit_label.setText(unit)


class RobotFaceDisplay(QMainWindow):
    """Main robot face display"""
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.message_count = 0
        self.is_listening = False
        self.is_speaking = False
        self.start_time = datetime.now()
        self.last_message = ""
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Robot Display")
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0f1e, stop:0.5 #1a1a2e, stop:1 #16213e);
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Main content area
        content_layout = QHBoxLayout()

        # Left side - Robot Face
        face_widget = self.create_face()
        content_layout.addWidget(face_widget, stretch=3)

        # Right side - Dashboard
        dashboard = self.create_dashboard()
        content_layout.addWidget(dashboard, stretch=2)

        main_layout.addLayout(content_layout)

        # Status indicators
        status_row = self.create_status_indicators()
        main_layout.addWidget(status_row)

        # Waveform
        self.waveform = WaveformWidget()
        main_layout.addWidget(self.waveform)

        # Message display
        self.message_label = QLabel("Waiting for interaction...")
        self.message_label.setFont(QFont("Ubuntu", 14))
        self.message_label.setStyleSheet("""
            color: #95a5a6;
            background-color: rgba(26, 26, 46, 0.8);
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
        """)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        main_layout.addWidget(self.message_label)

        # Fullscreen
        self.showFullScreen()

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.start(50)  # 20 FPS

        # Blink timer
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_eyes)
        self.blink_timer.start(3000)  # Blink every 3 seconds

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(1000)  # Update every second

    def create_header(self):
        """Create header"""
        header = QFrame()
        header.setStyleSheet("""
            background-color: rgba(15, 15, 30, 0.9);
            border-bottom: 3px solid #3498db;
            padding: 20px;
        """)
        header.setMinimumHeight(100)

        layout = QHBoxLayout()

        # Title
        title_layout = QVBoxLayout()
        title = QLabel("AI ROBOT INTERFACE")
        title.setFont(QFont("Ubuntu", 32, QFont.Bold))
        title.setStyleSheet("color: #3498db;")

        subtitle = QLabel("Neural Processing Unit • Active")
        subtitle.setFont(QFont("Ubuntu", 12))
        subtitle.setStyleSheet("color: #2ecc71;")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        # Time
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Ubuntu", 20))
        self.time_label.setStyleSheet("color: white;")

        layout.addLayout(title_layout)
        layout.addStretch()
        layout.addWidget(self.time_label)

        header.setLayout(layout)
        return header

    def create_face(self):
        """Create animated robot face"""
        face_frame = QFrame()
        face_frame.setStyleSheet("""
            background-color: rgba(15, 15, 30, 0.7);
            border-radius: 20px;
            border: 2px solid #3498db;
        """)
        layout = QVBoxLayout()

        # Graphics view for eyes
        view = QGraphicsView()
        view.setStyleSheet("background: transparent; border: none;")
        self.scene = QGraphicsScene()
        view.setScene(self.scene)
        view.setRenderHint(QPainter.Antialiasing)

        # Create eyes
        eye_size = 120
        eye_spacing = 200
        center_x = 200
        center_y = 150

        self.left_eye = AnimatedEye(center_x - eye_spacing//2, center_y, eye_size)
        self.right_eye = AnimatedEye(center_x + eye_spacing//2, center_y, eye_size)

        self.scene.addItem(self.left_eye)
        self.scene.addItem(self.right_eye)

        self.scene.setSceneRect(0, 0, 600, 400)

        layout.addWidget(view)

        face_frame.setLayout(layout)
        return face_frame

    def create_dashboard(self):
        """Create dashboard with stats"""
        dashboard = QFrame()
        dashboard.setStyleSheet("""
            background-color: rgba(15, 15, 30, 0.7);
            border-radius: 20px;
            border: 2px solid #2ecc71;
            padding: 20px;
        """)
        layout = QVBoxLayout()

        # Title
        title = QLabel("SYSTEM DASHBOARD")
        title.setFont(QFont("Ubuntu", 18, QFont.Bold))
        title.setStyleSheet("color: #2ecc71;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Stats grid
        stats_grid = QGridLayout()

        self.messages_card = StatCard("Messages", "0", "", "#3498db")
        self.interactions_card = StatCard("Interactions", "0", "", "#9b59b6")
        self.uptime_card = StatCard("Uptime", "00:00", "hh:mm", "#e74c3c")
        self.status_card = StatCard("Status", "READY", "", "#2ecc71")

        stats_grid.addWidget(self.messages_card, 0, 0)
        stats_grid.addWidget(self.interactions_card, 0, 1)
        stats_grid.addWidget(self.uptime_card, 1, 0)
        stats_grid.addWidget(self.status_card, 1, 1)

        layout.addLayout(stats_grid)
        layout.addStretch()

        dashboard.setLayout(layout)
        return dashboard

    def create_status_indicators(self):
        """Create status indicators"""
        status_frame = QFrame()
        layout = QHBoxLayout()

        self.listening_indicator = StatusIndicator("LISTENING")
        self.processing_indicator = StatusIndicator("PROCESSING")
        self.speaking_indicator = StatusIndicator("SPEAKING")

        layout.addWidget(self.listening_indicator)
        layout.addWidget(self.processing_indicator)
        layout.addWidget(self.speaking_indicator)

        status_frame.setLayout(layout)
        return status_frame

    def animate(self):
        """Main animation loop"""
        # Update waveform
        self.waveform.animate()

        # Update time
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)

    def blink_eyes(self):
        """Make eyes blink"""
        self.left_eye.blink()
        self.right_eye.blink()

    def update_stats(self):
        """Update dashboard statistics"""
        # Calculate uptime
        uptime = datetime.now() - self.start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        self.uptime_card.update_value(f"{hours:02d}:{minutes:02d}", "hh:mm")

    def on_user_message(self, message):
        """Handle user message"""
        self.message_count += 1
        self.is_listening = True

        # Update UI
        self.listening_indicator.set_active(True)
        self.processing_indicator.set_active(False)
        self.speaking_indicator.set_active(False)

        self.waveform.set_active(True)
        self.message_label.setText(f"USER: {message}")
        self.message_label.setStyleSheet("""
            color: white;
            background-color: rgba(52, 152, 219, 0.8);
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
        """)

        # Eyes look at user
        self.left_eye.look_at("center")
        self.right_eye.look_at("center")

        # Update stats
        self.messages_card.update_value(self.message_count)
        self.status_card.update_value("LISTENING")

        # Turn off after delay
        QTimer.singleShot(2000, lambda: self.listening_indicator.set_active(False))
        QTimer.singleShot(2000, lambda: self.waveform.set_active(False))

    def on_ai_message(self, message):
        """Handle AI message"""
        self.message_count += 1
        self.is_speaking = True

        # Update UI
        self.listening_indicator.set_active(False)
        self.processing_indicator.set_active(True)
        self.speaking_indicator.set_active(True)

        self.waveform.set_active(True)
        self.message_label.setText(f"AI: {message}")
        self.message_label.setStyleSheet("""
            color: white;
            background-color: rgba(46, 204, 113, 0.8);
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
        """)

        # Eyes animate
        self.blink_eyes()

        # Update stats
        self.messages_card.update_value(self.message_count)
        interaction_count = self.message_count // 2
        self.interactions_card.update_value(interaction_count)
        self.status_card.update_value("SPEAKING")

        # Turn off after delay
        QTimer.singleShot(3000, lambda: self.processing_indicator.set_active(False))
        QTimer.singleShot(3000, lambda: self.speaking_indicator.set_active(False))
        QTimer.singleShot(3000, lambda: self.waveform.set_active(False))
        QTimer.singleShot(3000, lambda: self.status_card.update_value("READY"))


class RobotDisplayNode(Node):
    """ROS2 node for robot display"""
    def __init__(self, gui):
        super().__init__('robot_display_node')
        self.gui = gui

        # Subscribe to topics
        self.user_subscription = self.create_subscription(
            String,
            '/speech_rec',
            self.user_callback,
            10
        )

        self.ai_subscription = self.create_subscription(
            String,
            '/speech/text',
            self.ai_callback,
            10
        )

        self.get_logger().info('Robot Display Node started')

    def user_callback(self, msg):
        """Handle user messages"""
        self.gui.on_user_message(msg.data)
        self.get_logger().info(f'User: {msg.data}')

    def ai_callback(self, msg):
        """Handle AI messages"""
        self.gui.on_ai_message(msg.data)
        self.get_logger().info(f'AI: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create GUI
    gui = RobotFaceDisplay(None)

    # Create ROS2 node
    node = RobotDisplayNode(gui)
    gui.node = node

    # Timer to process ROS2 callbacks
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
    timer.start(10)

    # Show GUI
    gui.show()

    # Start event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
