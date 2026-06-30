#!/usr/bin/env python3
"""
Robot Speech Display - Shows STT input and LLM responses
Subscribes to /speech/text and /tts_text ROS2 topics
"""

import sys
import signal
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette


class SignalBridge(QObject):
    """Bridge for thread-safe GUI updates"""
    user_message = pyqtSignal(str)
    robot_message = pyqtSignal(str)


class RobotDisplayNode(Node):
    def __init__(self, signal_bridge):
        super().__init__('robot_display')
        self.signal_bridge = signal_bridge

        # QoS profile to match publishers
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.stt_sub = self.create_subscription(
            String, '/speech/text', self.stt_callback, qos_profile)

        self.tts_sub = self.create_subscription(
            String, '/tts_text', self.tts_callback, qos_profile)

        self.get_logger().info('Robot Display started - listening on /speech/text and /tts_text')

    def stt_callback(self, msg):
        self.signal_bridge.user_message.emit(msg.data)

    def tts_callback(self, msg):
        self.signal_bridge.robot_message.emit(msg.data)


class RobotDisplayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Assistant")
        self.setStyleSheet(self.get_stylesheet())

        # Fullscreen
        self.showFullScreen()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QHBoxLayout()
        title = QLabel("🤖 Robot Assistant")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.status = QLabel("● Connected")
        self.status.setObjectName("status")
        header.addWidget(self.status)
        layout.addLayout(header)

        # Content panels
        content = QHBoxLayout()
        content.setSpacing(20)

        # User panel
        user_panel = self.create_panel("🎤 You Said", "#00d9ff", "#16213e")
        self.user_text = user_panel.findChild(QTextEdit)
        content.addWidget(user_panel)

        # Robot panel
        robot_panel = self.create_panel("🤖 Robot Says", "#00ff88", "#0f3460")
        self.robot_text = robot_panel.findChild(QTextEdit)
        content.addWidget(robot_panel)

        layout.addLayout(content, 1)

        # Footer
        footer = QLabel("ESC = exit fullscreen  |  F11 = toggle  |  Ctrl+Q = quit")
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

    def create_panel(self, title, accent, bg):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setStyleSheet(f"#panel {{ background-color: {bg}; border-radius: 10px; }}")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)

        header = QLabel(title)
        header.setStyleSheet(f"color: {accent}; font-size: 20px; font-weight: bold;")
        layout.addWidget(header)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("""
            QTextEdit {
                background-color: #0d1b2a;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 18px;
            }
        """)
        layout.addWidget(text)

        return panel

    def add_user_message(self, text):
        self._add_message(self.user_text, text)

    def add_robot_message(self, text):
        self._add_message(self.robot_text, text)

    def _add_message(self, widget, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        current = widget.toPlainText()
        if current:
            widget.append("\n" + "─" * 30 + "\n")
        widget.append(f"[{timestamp}]\n{text}")
        widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())

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
        elif event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
            self.close()
        else:
            super().keyPressEvent(event)

    def get_stylesheet(self):
        return """
            QMainWindow, QWidget {
                background-color: #1a1a2e;
            }
            #title {
                color: white;
                font-size: 28px;
                font-weight: bold;
            }
            #status {
                color: #00ff88;
                font-size: 16px;
            }
            #footer {
                color: #666666;
                font-size: 14px;
            }
        """


def main():
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    # Signal bridge for thread-safe updates
    bridge = SignalBridge()

    # Create window
    window = RobotDisplayWindow()
    bridge.user_message.connect(window.add_user_message)
    bridge.robot_message.connect(window.add_robot_message)
    window.show()

    # ROS2 in separate thread
    rclpy.init()
    node = RobotDisplayNode(bridge)

    ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    ros_thread.start()

    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
