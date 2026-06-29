#!/usr/bin/env python3
"""
Simple Dashboard Display for Robot
Shows LLM responses and system stats in a clean, readable layout
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFrame, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import sys
from datetime import datetime


class DashboardDisplay(QMainWindow):
    """Simple fullscreen dashboard"""
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.message_count = 0
        self.start_time = datetime.now()
        self.current_message = "Waiting for robot to speak..."
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Robot Dashboard")

        # Dark background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        central_widget.setLayout(main_layout)

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Main message display - takes most of the space
        message_frame = self.create_message_display()
        main_layout.addWidget(message_frame, stretch=3)

        # Stats bar at bottom
        stats_bar = self.create_stats_bar()
        main_layout.addWidget(stats_bar)

        # Fullscreen
        self.showFullScreen()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)

    def create_header(self):
        """Create header bar"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        header.setFixedHeight(100)

        layout = QHBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)

        # Title
        title = QLabel("ROBOT DISPLAY")
        title.setFont(QFont("Ubuntu", 36, QFont.Bold))
        title.setStyleSheet("color: #00d4ff;")

        # Time and date
        time_layout = QVBoxLayout()
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Ubuntu", 28, QFont.Bold))
        self.time_label.setStyleSheet("color: white;")
        self.time_label.setAlignment(Qt.AlignRight)

        self.date_label = QLabel()
        self.date_label.setFont(QFont("Ubuntu", 14))
        self.date_label.setStyleSheet("color: #7f8c8d;")
        self.date_label.setAlignment(Qt.AlignRight)

        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.date_label)

        layout.addWidget(title)
        layout.addStretch()
        layout.addLayout(time_layout)

        header.setLayout(layout)
        return header

    def create_message_display(self):
        """Create main message display area"""
        message_frame = QFrame()
        message_frame.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-radius: 20px;
                border: 3px solid #00d4ff;
                padding: 40px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        # Label
        label = QLabel("ROBOT SPEAKING:")
        label.setFont(QFont("Ubuntu", 18, QFont.Bold))
        label.setStyleSheet("color: #00d4ff; padding-bottom: 20px;")
        layout.addWidget(label)

        # Message text
        self.message_text = QLabel(self.current_message)
        self.message_text.setFont(QFont("Ubuntu", 32))
        self.message_text.setStyleSheet("""
            color: white;
            padding: 20px;
            background-color: rgba(0, 212, 255, 0.1);
            border-radius: 15px;
        """)
        self.message_text.setWordWrap(True)
        self.message_text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.message_text)

        message_frame.setLayout(layout)
        return message_frame

    def create_stats_bar(self):
        """Create stats bar at bottom"""
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        stats_frame.setFixedHeight(150)

        layout = QGridLayout()
        layout.setSpacing(20)

        # Messages stat
        messages_widget = self.create_stat_widget("MESSAGES", "0", "#3498db")
        layout.addWidget(messages_widget, 0, 0)

        # Status stat
        self.status_widget = self.create_stat_widget("STATUS", "READY", "#2ecc71")
        layout.addWidget(self.status_widget, 0, 1)

        # Uptime stat
        self.uptime_widget = self.create_stat_widget("UPTIME", "00:00:00", "#e74c3c")
        layout.addWidget(self.uptime_widget, 0, 2)

        # Last updated
        self.last_updated_widget = self.create_stat_widget("LAST UPDATE", "Never", "#9b59b6")
        layout.addWidget(self.last_updated_widget, 0, 3)

        stats_frame.setLayout(layout)
        return stats_frame

    def create_stat_widget(self, title, value, color):
        """Create a stat widget"""
        widget = QFrame()
        widget.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(52, 73, 94, 0.5);
                border-radius: 10px;
                border: 2px solid {color};
                padding: 15px;
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Ubuntu", 12, QFont.Bold))
        title_label.setStyleSheet(f"color: {color};")
        title_label.setAlignment(Qt.AlignCenter)

        # Value
        value_label = QLabel(value)
        value_label.setFont(QFont("Ubuntu", 22, QFont.Bold))
        value_label.setStyleSheet("color: white;")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setObjectName("value")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        widget.setLayout(layout)
        return widget

    def update_display(self):
        """Update time and stats"""
        # Update time
        now = datetime.now()
        self.time_label.setText(now.strftime("%H:%M:%S"))
        self.date_label.setText(now.strftime("%A, %B %d, %Y"))

        # Update uptime
        uptime = now - self.start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        seconds = int(uptime.total_seconds() % 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        uptime_value = self.uptime_widget.findChild(QLabel, "value")
        if uptime_value:
            uptime_value.setText(uptime_str)

    def on_message_received(self, message, is_user=False):
        """Handle new message"""
        self.message_count += 1
        self.current_message = message

        # Update message display
        self.message_text.setText(message)

        # Update messages count
        messages_value = self.stats_frame.layout().itemAt(0).widget().findChild(QLabel, "value")
        if messages_value:
            messages_value.setText(str(self.message_count))

        # Update status
        status_value = self.status_widget.findChild(QLabel, "value")
        if status_value:
            if is_user:
                status_value.setText("LISTENING")
                self.status_widget.setStyleSheet("""
                    QFrame {
                        background-color: rgba(52, 73, 94, 0.5);
                        border-radius: 10px;
                        border: 2px solid #f39c12;
                        padding: 15px;
                    }
                """)
            else:
                status_value.setText("SPEAKING")
                self.status_widget.setStyleSheet("""
                    QFrame {
                        background-color: rgba(52, 73, 94, 0.5);
                        border-radius: 10px;
                        border: 2px solid #2ecc71;
                        padding: 15px;
                    }
                """)

        # Update last updated
        last_updated_value = self.last_updated_widget.findChild(QLabel, "value")
        if last_updated_value:
            last_updated_value.setText(datetime.now().strftime("%H:%M:%S"))

        # Reset status after delay
        QTimer.singleShot(3000, self.reset_status)

    def reset_status(self):
        """Reset status to ready"""
        status_value = self.status_widget.findChild(QLabel, "value")
        if status_value:
            status_value.setText("READY")
            self.status_widget.setStyleSheet("""
                QFrame {
                    background-color: rgba(52, 73, 94, 0.5);
                    border-radius: 10px;
                    border: 2px solid #2ecc71;
                    padding: 15px;
                }
            """)

    @property
    def stats_frame(self):
        """Get stats frame widget"""
        return self.centralWidget().layout().itemAt(2).widget()


class DashboardNode(Node):
    """ROS2 node for dashboard"""
    def __init__(self, gui):
        super().__init__('dashboard_node')
        self.gui = gui

        # Subscribe to user input
        self.user_subscription = self.create_subscription(
            String,
            '/speech_rec',
            self.user_callback,
            10
        )

        # Subscribe to AI output
        self.ai_subscription = self.create_subscription(
            String,
            '/speech/text',
            self.ai_callback,
            10
        )

        self.get_logger().info('Dashboard Display Node started')

    def user_callback(self, msg):
        """Handle user messages"""
        self.gui.on_message_received(f"USER: {msg.data}", is_user=True)
        self.get_logger().info(f'User: {msg.data}')

    def ai_callback(self, msg):
        """Handle AI messages"""
        self.gui.on_message_received(msg.data, is_user=False)
        self.get_logger().info(f'AI: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create GUI
    gui = DashboardDisplay(None)

    # Create ROS2 node
    node = DashboardNode(gui)
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
