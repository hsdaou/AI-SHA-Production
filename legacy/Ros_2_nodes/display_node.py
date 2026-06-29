import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QObject

class Bridge(QObject):
    """Bridge to pass ROS data to the Qt Thread"""
    image_received = pyqtSignal(QImage)
    text_received = pyqtSignal(str)
    status_received = pyqtSignal(str)

class DisplayManagerNode(Node):
    def __init__(self, bridge):
        super().__init__('display_manager_node')
        self.bridge = bridge
        self.cv_bridge = CvBridge()

        # 1. Subscriptions
        self.create_subscription(Image, '/camera/color/image_raw', self.img_callback, 10)
        self.create_subscription(String, '/llm/response', self.llm_callback, 10)
        self.create_subscription(String, '/robot_state', self.state_callback, 10)
        self.create_subscription(String, '/detected_objects', self.obj_callback, 10)

    def img_callback(self, msg):
        cv_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        # Resize to fit the left side of our dashboard
        cv_img = cv2.resize(cv_img, (640, 480))
        qt_img = self.convert_cv_qt(cv_img)
        self.bridge.image_received.emit(qt_img)

    def llm_callback(self, msg):
        self.bridge.text_received.emit(msg.data)

    def state_callback(self, msg):
        self.bridge.status_received.emit(f"STATE: {msg.data}")

    def obj_callback(self, msg):
        # You could parse this JSON/String and update a list
        pass

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        return convert_to_Qt_format

class MainWindow(QWidget):
    def __init__(self, bridge):
        super().__init__()
        self.setWindowTitle("Robot OS Dashboard")
        self.setFixedSize(1024, 600) # ELECROW native resolution
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        # Layouts
        main_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # Video Feed (Left)
        self.video_label = QLabel("Waiting for Camera...")
        self.video_label.setFixedSize(640, 480)
        left_col.addWidget(self.video_label)

        # LLM Text Box (Bottom Left)
        self.llm_label = QLabel("System Ready...")
        self.llm_label.setWordWrap(True)
        self.llm_label.setStyleSheet("font-size: 18px; background: #2d2d2d; padding: 10px;")
        left_col.addWidget(self.llm_label)

        # Robot State (Right Top)
        self.state_label = QLabel("STATE: IDLE")
        self.state_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff00;")
        right_col.addWidget(self.state_label)

        # Objects List (Right Bottom)
        self.obj_label = QLabel("Objects:\n- None")
        right_col.addWidget(self.obj_label)

        main_layout.addLayout(left_col, stretch=2)
        main_layout.addLayout(right_col, stretch=1)
        self.setLayout(main_layout)

        # Connect Signals
        bridge.image_received.connect(self.update_image)
        bridge.text_received.connect(self.llm_label.setText)
        bridge.status_received.connect(self.state_label.setText)

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

def main():
    app = QApplication(sys.argv)
    rclpy.init()
    
    bridge = Bridge()
    node = DisplayManagerNode(bridge)
    window = MainWindow(bridge)
    window.show()

    # Use a timer to spin ROS 2 within the Qt Event Loop
    import threading
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
