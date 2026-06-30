#!/usr/bin/env python3
"""
LLM Display Node for Raspberry Pi 5
Beautiful animated display with dashboard for LLM interactions
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QScrollArea, QFrame, QPushButton,
                             QGridLayout, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QPoint
from PyQt5.QtGui import QFont, QPalette, QColor, QPainter, QLinearGradient
import sys
from datetime import datetime
from collections import deque


class AnimatedLabel(QLabel):
    """Label with fade-in animation"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._opacity = 0.0

    def get_opacity(self):
        return self._opacity

    def set_opacity(self, value):
        self._opacity = value
        self.setStyleSheet(f"""
            background-color: rgba(40, 44, 52, {int(value * 255)});
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
        """)

    opacity = pyqtProperty(float, get_opacity, set_opacity)

    def fade_in(self):
        self.animation = QPropertyAnimation(self, b"opacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()


class MessageBubble(QFrame):
    """Animated message bubble for chat"""
    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setup_ui(text)

    def setup_ui(self, text):
        layout = QVBoxLayout()

        # Message label
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Ubuntu", 14))

        if self.is_user:
            self.label.setStyleSheet("""
                color: white;
                background-color: #3498db;
                border-radius: 15px;
                padding: 15px 20px;
            """)
            self.label.setAlignment(Qt.AlignRight)
        else:
            self.label.setStyleSheet("""
                color: white;
                background-color: #2ecc71;
                border-radius: 15px;
                padding: 15px 20px;
            """)
            self.label.setAlignment(Qt.AlignLeft)

        # Timestamp
        self.time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        self.time_label.setFont(QFont("Ubuntu", 10))
        self.time_label.setStyleSheet("color: #7f8c8d; padding: 5px;")

        if self.is_user:
            self.time_label.setAlignment(Qt.AlignRight)
            layout.addWidget(self.label, alignment=Qt.AlignRight)
            layout.addWidget(self.time_label, alignment=Qt.AlignRight)
        else:
            self.time_label.setAlignment(Qt.AlignLeft)
            layout.addWidget(self.label, alignment=Qt.AlignLeft)
            layout.addWidget(self.time_label, alignment=Qt.AlignLeft)

        self.setLayout(layout)
        self.setStyleSheet("background: transparent;")

        # Fade in animation
        self.setWindowOpacity(0)
        self.fade_in()

    def fade_in(self):
        self.animation = QPropertyAnimation(self, b"pos")
        self.animation.setDuration(300)
        start_pos = self.pos()
        self.animation.setStartValue(QPoint(start_pos.x(), start_pos.y() + 20))
        self.animation.setEndValue(start_pos)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()


class StatusCard(QFrame):
    """Status card for dashboard"""
    def __init__(self, title, value="0", color="#3498db", parent=None):
        super().__init__(parent)
        self.title = title
        self.value_text = value
        self.color = color
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Ubuntu", 12, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.color};")
        title_label.setAlignment(Qt.AlignCenter)

        # Value
        self.value_label = QLabel(self.value_text)
        self.value_label.setFont(QFont("Ubuntu", 28, QFont.Bold))
        self.value_label.setStyleSheet("color: white;")
        self.value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

        self.setLayout(layout)
        self.setStyleSheet(f"""
            background-color: rgba(52, 73, 94, 0.8);
            border-radius: 15px;
            border: 2px solid {self.color};
            padding: 20px;
        """)

    def update_value(self, value):
        self.value_label.setText(str(value))


class LLMDisplayGUI(QMainWindow):
    """Main display window"""
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.messages = deque(maxlen=50)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("AI Assistant Display")
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # Central widget with stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Stacked widget for switching between chat and dashboard
        self.stacked_widget = QStackedWidget()

        # Chat view
        self.chat_widget = self.create_chat_view()
        self.stacked_widget.addWidget(self.chat_widget)

        # Dashboard view
        self.dashboard_widget = self.create_dashboard()
        self.stacked_widget.addWidget(self.dashboard_widget)

        main_layout.addWidget(self.stacked_widget)

        # Navigation buttons
        nav_bar = self.create_navigation()
        main_layout.addWidget(nav_bar)

        # Status bar
        self.status_label = QLabel("Waiting for messages...")
        self.status_label.setStyleSheet("""
            color: #95a5a6;
            padding: 10px;
            font-size: 12px;
        """)
        main_layout.addWidget(self.status_label)

        # Fullscreen
        self.showFullScreen()

    def create_header(self):
        """Create header with title and time"""
        header = QFrame()
        header.setStyleSheet("""
            background-color: rgba(26, 26, 46, 0.9);
            border-bottom: 3px solid #3498db;
            padding: 15px;
        """)

        layout = QHBoxLayout()

        # Title
        title = QLabel("AI Assistant")
        title.setFont(QFont("Ubuntu", 24, QFont.Bold))
        title.setStyleSheet("color: #3498db;")

        # Subtitle
        subtitle = QLabel("Powered by Llama 3.2")
        subtitle.setFont(QFont("Ubuntu", 12))
        subtitle.setStyleSheet("color: #7f8c8d;")

        # Time
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Ubuntu", 16))
        self.time_label.setStyleSheet("color: white;")
        self.update_time()

        # Timer for updating time
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)

        title_layout = QVBoxLayout()
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        layout.addLayout(title_layout)
        layout.addStretch()
        layout.addWidget(self.time_label)

        header.setLayout(layout)
        return header

    def create_chat_view(self):
        """Create chat interface"""
        chat_widget = QWidget()
        layout = QVBoxLayout()

        # Scroll area for messages
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #2c3e50;
                width: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #3498db;
                border-radius: 7px;
            }
        """)

        # Messages container
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout()
        self.messages_layout.addStretch()
        self.messages_widget.setLayout(self.messages_layout)
        self.messages_widget.setStyleSheet("background: transparent;")

        scroll.setWidget(self.messages_widget)
        layout.addWidget(scroll)

        chat_widget.setLayout(layout)
        return chat_widget

    def create_dashboard(self):
        """Create dashboard with stats"""
        dashboard = QWidget()
        layout = QVBoxLayout()

        # Dashboard title
        title = QLabel("System Dashboard")
        title.setFont(QFont("Ubuntu", 20, QFont.Bold))
        title.setStyleSheet("color: white; padding: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Stats grid
        stats_layout = QGridLayout()

        self.messages_card = StatusCard("Total Messages", "0", "#3498db")
        self.user_card = StatusCard("User Messages", "0", "#9b59b6")
        self.ai_card = StatusCard("AI Responses", "0", "#2ecc71")
        self.uptime_card = StatusCard("Uptime", "00:00:00", "#e74c3c")

        stats_layout.addWidget(self.messages_card, 0, 0)
        stats_layout.addWidget(self.user_card, 0, 1)
        stats_layout.addWidget(self.ai_card, 1, 0)
        stats_layout.addWidget(self.uptime_card, 1, 1)

        layout.addLayout(stats_layout)
        layout.addStretch()

        dashboard.setLayout(layout)
        return dashboard

    def create_navigation(self):
        """Create navigation bar"""
        nav_bar = QFrame()
        nav_bar.setStyleSheet("""
            background-color: rgba(26, 26, 46, 0.9);
            border-top: 2px solid #3498db;
            padding: 10px;
        """)

        layout = QHBoxLayout()

        # Chat button
        chat_btn = QPushButton("Chat")
        chat_btn.setFont(QFont("Ubuntu", 14, QFont.Bold))
        chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 40px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        chat_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        # Dashboard button
        dashboard_btn = QPushButton("Dashboard")
        dashboard_btn.setFont(QFont("Ubuntu", 14, QFont.Bold))
        dashboard_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 40px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        dashboard_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        # Clear button
        clear_btn = QPushButton("Clear Chat")
        clear_btn.setFont(QFont("Ubuntu", 14, QFont.Bold))
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 40px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        clear_btn.clicked.connect(self.clear_chat)

        layout.addWidget(chat_btn)
        layout.addWidget(dashboard_btn)
        layout.addStretch()
        layout.addWidget(clear_btn)

        nav_bar.setLayout(layout)
        return nav_bar

    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)

    def add_user_message(self, text):
        """Add user message to chat"""
        bubble = MessageBubble(text, is_user=True)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
        self.messages.append(('user', text))
        self.update_stats()
        self.scroll_to_bottom()
        self.status_label.setText("User: " + text[:50] + "...")

    def add_ai_message(self, text):
        """Add AI message to chat"""
        bubble = MessageBubble(text, is_user=False)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
        self.messages.append(('ai', text))
        self.update_stats()
        self.scroll_to_bottom()
        self.status_label.setText("AI: " + text[:50] + "...")

    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        QTimer.singleShot(100, lambda: self.chat_widget.findChild(QScrollArea).verticalScrollBar().setValue(
            self.chat_widget.findChild(QScrollArea).verticalScrollBar().maximum()
        ))

    def update_stats(self):
        """Update dashboard statistics"""
        total = len(self.messages)
        user_count = sum(1 for msg_type, _ in self.messages if msg_type == 'user')
        ai_count = sum(1 for msg_type, _ in self.messages if msg_type == 'ai')

        self.messages_card.update_value(total)
        self.user_card.update_value(user_count)
        self.ai_card.update_value(ai_count)

    def clear_chat(self):
        """Clear all messages"""
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.messages.clear()
        self.update_stats()
        self.status_label.setText("Chat cleared")


class LLMDisplayNode(Node):
    """ROS2 node for LLM display"""
    def __init__(self, gui):
        super().__init__('llm_display_node')
        self.gui = gui

        # Subscribe to LLM responses
        self.llm_subscription = self.create_subscription(
            String,
            '/speech/text',
            self.llm_callback,
            10
        )

        # Subscribe to user input
        self.user_subscription = self.create_subscription(
            String,
            '/speech_rec',
            self.user_callback,
            10
        )

        self.get_logger().info('LLM Display Node started')

    def llm_callback(self, msg):
        """Handle LLM responses"""
        self.gui.add_ai_message(msg.data)
        self.get_logger().info(f'AI: {msg.data}')

    def user_callback(self, msg):
        """Handle user input"""
        self.gui.add_user_message(msg.data)
        self.get_logger().info(f'User: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create GUI
    gui = LLMDisplayGUI(None)

    # Create ROS2 node
    node = LLMDisplayNode(gui)
    gui.node = node

    # Timer to process ROS2 callbacks
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
    timer.start(10)  # 100Hz

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
