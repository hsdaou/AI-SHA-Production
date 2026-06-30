#!/usr/bin/env python3

"""
ROS2 Node: vision_pipeline_node
YOLOv8-m Vision Pipeline

Subscribes:
  /camera/color/image_raw   (sensor_msgs/Image)
  /camera/depth/image_raw   (sensor_msgs/Image)

Publishes:
  /detected_objects         (vision_msgs/Detection2DArray)
  /segmentation_masks       (vision_msgs/Detection2DArray)  [placeholder]
  /tracking/targets         (vision_msgs/Detection2DArray)  [placeholder]

Assumptions:
- ROS2 Humble
- YOLOv8-m (.engine or .pt) available
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header

from cv_bridge import CvBridge
import numpy as np

from ultralytics import YOLO


class VisionPipeline(Node):
    def __init__(self):
        super().__init__('vision_pipeline_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('model_path', 'yolov8m.engine')
        self.declare_parameter('confidence', 0.4)
        self.declare_parameter('camera_frame', 'camera_color_frame')

        self.model_path = self.get_parameter('model_path').value
        self.confidence = self.get_parameter('confidence').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # ---------------- YOLO ----------------
        self.get_logger().info(f'Loading YOLOv8-m model: {self.model_path}')
        self.model = YOLO(self.model_path)

        # ---------------- ROS ----------------
        self.bridge = CvBridge()
        self.latest_depth = None

        # Subscriptions
        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )

        self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # Publications
        self.det_pub = self.create_publisher(
            Detection2DArray,
            '/detected_objects',
            10
        )

        self.seg_pub = self.create_publisher(
            Detection2DArray,
            '/segmentation_masks',
            10
        )

        self.track_pub = self.create_publisher(
            Detection2DArray,
            '/tracking/targets',
            10
        )

        self.get_logger().info('Vision pipeline node started')

    # ---------------- Callbacks ----------------
    def depth_callback(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough'
        )

    def color_callback(self, msg: Image):
        if self.latest_depth is None:
            return

        color_img = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )

        results = self.model(
            color_img,
            device=0,
            conf=self.confidence
        )

        det_array = Detection2DArray()
        det_array.header = Header()
        det_array.header.stamp = self.get_clock().now().to_msg()
        det_array.header.frame_id = self.camera_frame

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                depth = float(self.latest_depth[cy, cx]) / 1000.0  # mm → m

                det = Detection2D()
                det.bbox.center.x = (x1 + x2) / 2.0
                det.bbox.center.y = (y1 + y2) / 2.0
                det.bbox.size_x = (x2 - x1)
                det.bbox.size_y = (y2 - y1)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(box.cls))
                hyp.hypothesis.score = float(box.conf)
                hyp.pose.pose.position.z = depth

                det.results.append(hyp)
                det_array.detections.append(det)

        # Publish according to contract
        self.det_pub.publish(det_array)
        self.track_pub.publish(det_array)
        self.seg_pub.publish(Detection2DArray())  # placeholder


def main(args=None):
    rclpy.init(args=args)
    node = VisionPipeline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
