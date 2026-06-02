#!/usr/bin/env python3
"""
Standalone plant disease classifier ROS2 node.
Subscribes directly to camera RGB, runs TensorRT disease classifier on full frame.
Uses HSV green-ratio check to reject non-plant frames before classification.
"""

import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from yolov8_ros.plant_disease_engine import PlantDiseaseEngine


def is_plant_frame(frame_bgr, min_green_ratio=0.15):
    """Check if enough of the frame contains green (plant-like) pixels."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Broad green range in HSV
    lower_green = np.array([25, 30, 30], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / mask.size
    return green_ratio >= min_green_ratio, green_ratio


class PlantDiseaseNode(Node):
    def __init__(self):
        super().__init__('plant_disease_node')

        self.declare_parameter('engine_path',
            '/home/orin-robot/plant_disease_models/plant_disease_classifier.engine')
        self.declare_parameter('class_mapping_path',
            '/home/orin-robot/plant_disease_models/class_mapping.json')
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('min_green_ratio', 0.15)

        engine_path = self.get_parameter('engine_path').value
        mapping_path = self.get_parameter('class_mapping_path').value
        threshold = self.get_parameter('confidence_threshold').value
        camera_topic = self.get_parameter('camera_topic').value
        self.min_green_ratio = self.get_parameter('min_green_ratio').value

        self.get_logger().info(f'Loading disease engine: {engine_path}')
        self.engine = PlantDiseaseEngine(engine_path, mapping_path,
                                         confidence_threshold=threshold)
        self.get_logger().info('Disease engine loaded!')

        self.bridge = CvBridge()
        self.pub = self.create_publisher(String, '/plant_disease/top', 10)

        self.sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 1)

        self.get_logger().info(
            f'Subscribed to {camera_topic} — min_green_ratio={self.min_green_ratio}')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Reject frames that don't contain enough green (not a plant)
        plant_detected, green_ratio = is_plant_frame(frame, self.min_green_ratio)
        if not plant_detected:
            out = {
                'species': 'none',
                'disease': 'none',
                'confidence': 0.0,
                'is_healthy': False,
                'below_threshold': True,
                'top3': [],
                'green_ratio': round(green_ratio, 3),
                'plant_detected': False,
            }
            msg_out = String()
            msg_out.data = json.dumps(out)
            self.pub.publish(msg_out)
            return

        preprocessed = self.engine.preprocess(frame)
        if preprocessed is None:
            return

        results = self.engine.infer_batch([preprocessed])
        r = results[0]

        out = {
            'species': r['species'],
            'disease': r['disease'],
            'confidence': round(r['confidence'], 3),
            'is_healthy': r['is_healthy'],
            'below_threshold': r['below_threshold'],
            'top3': [(name, round(conf, 3)) for name, conf in r['top3']],
            'green_ratio': round(green_ratio, 3),
            'plant_detected': True,
        }

        msg_out = String()
        msg_out.data = json.dumps(out)
        self.pub.publish(msg_out)

        if not r['below_threshold']:
            self.get_logger().info(
                f"{r['species']} - {r['disease']} ({r['confidence']:.1%}) "
                f"[green: {green_ratio:.0%}]",
                throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = PlantDiseaseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
