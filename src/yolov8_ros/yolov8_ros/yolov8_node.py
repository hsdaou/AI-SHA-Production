#!/usr/bin/env python3
"""
Ultra-optimized detection node for Jetson Orin with:
- YOLO object detection (TensorRT accelerated)
- MediaPipe Hands (gesture recognition)
- MediaPipe Face Detection
- EasyOCR text recognition
- Parallel processing pipeline
- CUDA acceleration throughout
- IMPROVED: Working depth estimation with camera info display
- IMPROVED: Concise ROS message format

Target: 30+ FPS with all features enabled.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32, Bool
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

import os
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import time
import json

# Lazy import for OCR
easyocr = None


@dataclass
class DetectionResult:
    class_id: str
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    depth: Optional[float] = None
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0


@dataclass
class FrameData:
    frame_bgr: np.ndarray
    frame_rgb: np.ndarray
    depth_image: Optional[np.ndarray]
    header: any
    width: int
    height: int
    timestamp: float


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # Declare all parameters
        self.declare_parameter('model_path', '~/robot_ws/yolov8m.engine')
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('show_window', True)
        self.declare_parameter('max_det', 100)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('target_fps', 30)
        self.declare_parameter('enable_ocr', True)  # Enabled by default
        self.declare_parameter('enable_faces', True)  # Enabled by default
        self.declare_parameter('enable_gestures', True)  # Enabled by default
        self.declare_parameter('ocr_interval', 10)
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        # Depth parameters
        self.declare_parameter('depth_min_valid', 0.1)
        self.declare_parameter('depth_max_valid', 10.0)
        self.declare_parameter('depth_percentile', 50)
        self.declare_parameter('depth_sample_ratio', 0.5)

        # Load parameters
        model_path = os.path.expanduser(self.get_parameter('model_path').value)
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.img_size = self.get_parameter('image_size').value
        self.use_depth = self.get_parameter('use_depth').value
        self.show_window = self.get_parameter('show_window').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.target_fps = self.get_parameter('target_fps').value
        self.enable_ocr = self.get_parameter('enable_ocr').value
        self.enable_faces = self.get_parameter('enable_faces').value
        self.enable_gestures = self.get_parameter('enable_gestures').value
        self.ocr_interval = self.get_parameter('ocr_interval').value
        self.frame_id = self.get_parameter('frame_id').value
        self.depth_min = self.get_parameter('depth_min_valid').value * 1000.0
        self.depth_max = self.get_parameter('depth_max_valid').value * 1000.0
        self.depth_percentile = self.get_parameter('depth_percentile').value
        self.depth_sample_ratio = self.get_parameter('depth_sample_ratio').value

        self.bridge = CvBridge()
        self.min_frame_time = 1.0 / self.target_fps

        # Initialize YOLO
        self.get_logger().info(f"Loading YOLO: {model_path}")
        self.model = YOLO(model_path, task='detect')
        if model_path.endswith('.pt'):
            self.model.to(self.device)
            self.model.fuse()

        # CUDA optimizations
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Note: Removed set_per_process_memory_fraction to allow dynamic allocation
        # This prevents OOM errors when other processes (brain LLM) use GPU

        # Warmup YOLO
        self.get_logger().info("Warming up YOLO...")
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for _ in range(10):
            self.model.predict(dummy, verbose=False, conf=self.conf_threshold,
                             imgsz=self.img_size, half=True)
        torch.cuda.synchronize()
        self.get_logger().info("YOLO ready")

        # Initialize MediaPipe
        if self.enable_gestures:
            self.get_logger().info("=" * 60)
            self.get_logger().info("🤚 Initializing MediaPipe Hands (Gesture Recognition)...")
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,  # Lowered for better detection
                    min_tracking_confidence=0.5,
                    model_complexity=0
                )
                self.get_logger().info("✓ MediaPipe Hands ready - Gesture detection ENABLED")
                self.get_logger().info("  Detectable gestures: fist, open, peace, thumbs_up, etc.")
                self.get_logger().info("=" * 60)
            except Exception as e:
                self.get_logger().error(f"❌ Failed to initialize MediaPipe Hands: {e}")
                self.hands = None
                self.enable_gestures = False
        else:
            self.hands = None
            self.get_logger().warn("⚠ Gesture detection DISABLED")

        if self.enable_faces:
            self.get_logger().info("=" * 60)
            self.get_logger().info("👤 Initializing MediaPipe Face Detection...")
            try:
                self.mp_face = mp.solutions.face_detection
                self.face_detector = self.mp_face.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.3  # Lowered for better detection
                )
                self.get_logger().info("✓ MediaPipe Face Detection ready - Face detection ENABLED")
                self.get_logger().info("=" * 60)
            except Exception as e:
                self.get_logger().error(f"❌ Failed to initialize MediaPipe Face: {e}")
                self.face_detector = None
                self.enable_faces = False
        else:
            self.face_detector = None
            self.get_logger().warn("⚠ Face detection DISABLED")

        # Initialize OCR
        self.ocr_reader = None
        self.ocr_results_cache = []
        self.ocr_frame_count = 0

        if self.enable_ocr:
            self.get_logger().info("=" * 60)
            self.get_logger().info("📝 OCR (EasyOCR) will be initialized on first use")
            self.get_logger().info(f"  OCR runs every {self.ocr_interval} frames")
            self.get_logger().info("=" * 60)
        else:
            self.get_logger().warn("⚠ OCR detection DISABLED")
        if self.enable_ocr:
            self.get_logger().info("OCR will be initialized on first use...")

        # State management
        self.frame_lock = Lock()
        self.latest_frame_data: Optional[FrameData] = None
        self.camera_intrinsics = None
        self.running = True
        self.depth_sync_lock = Lock()
        self.latest_depth_image = None
        self.depth_frame_count = 0
        self.rgb_frame_count = 0

        # Results storage
        self.results_lock = Lock()
        self.latest_objects: List[DetectionResult] = []
        self.latest_faces: List[DetectionResult] = []
        self.latest_gestures: List[DetectionResult] = []
        self.latest_ocr: List[DetectionResult] = []

        # Performance tracking
        self.fps_deque = deque(maxlen=60)
        self.timing_stats = {'yolo': deque(maxlen=30), 'face': deque(maxlen=30),
                            'gesture': deque(maxlen=30), 'ocr': deque(maxlen=30)}

        # Thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.rgb_callback, sensor_qos)

        if self.use_depth:
            self.depth_sub = self.create_subscription(
                Image, '/camera/camera/aligned_depth_to_color/image_raw',
                self.depth_callback, sensor_qos)
            self.camera_info_sub = self.create_subscription(
                CameraInfo, '/camera/camera/color/camera_info',
                self.camera_info_callback, 10)

        # Publishers - Simple/concise format (NEW)
        self.objects_simple_pub = self.create_publisher(String, '/detection/objects_simple', reliable_qos)
        self.faces_simple_pub = self.create_publisher(String, '/detection/faces_simple', reliable_qos)
        self.gestures_simple_pub = self.create_publisher(String, '/detection/gestures_simple', reliable_qos)
        self.ocr_simple_pub = self.create_publisher(String, '/detection/ocr_simple', reliable_qos)

        # Publishers - Visualization
        self.annotated_pub = self.create_publisher(Image, '/detection/image_annotated', sensor_qos)
        self.depth_overlay_pub = self.create_publisher(Image, '/detection/depth_overlay', sensor_qos)
        
        # Publishers - Other
        self.closest_object_pub = self.create_publisher(String, '/detection/closest_object', reliable_qos)
        self.fps_pub = self.create_publisher(Float32, '/detection/fps', reliable_qos)
        self.stats_pub = self.create_publisher(String, '/detection/stats', reliable_qos)

        # ── GPU multiplexing (ADR 0001): pause inference on demand ──────────
        # When paused, process_loop skips predict() so no new GPU buffers are
        # allocated and any in-flight work drains — freeing the GPU for a
        # CONVERSING-state llama (num_gpu=99) WITHOUT unloading the engine.
        # The engine stays resident (~700MB, harmless), so resume is instant.
        self.inference_enabled = True
        self._pause_service = self.create_service(
            SetBool, '~/pause_inference', self._pause_inference_cb)
        # Latched state topic so the arbiter / tools can observe inference state.
        latched_qos = QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.inference_state_pub = self.create_publisher(
            Bool, '/detection/inference_active', latched_qos)
        self._publish_inference_state()

        # Start processing thread
        self.process_thread = Thread(target=self.process_loop, daemon=True)
        self.process_thread.start()

        self.get_logger().info("=" * 50)
        self.get_logger().info("Detection Node Ready")
        self.get_logger().info(f"  Target FPS: {self.target_fps}")
        self.get_logger().info(f"  YOLO: {model_path}")
        self.get_logger().info(f"  Depth: {'ON' if self.use_depth else 'OFF'}")
        self.get_logger().info(f"  Faces: {'ON' if self.enable_faces else 'OFF'}")
        self.get_logger().info(f"  Gestures: {'ON' if self.enable_gestures else 'OFF'}")
        self.get_logger().info(f"  OCR: {'ON' if self.enable_ocr else 'OFF'}")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Concise topics (use these for echo):")
        self.get_logger().info("  /detection/objects_simple")
        self.get_logger().info("  /detection/faces_simple")
        self.get_logger().info("  /detection/gestures_simple")
        self.get_logger().info("  /detection/ocr_simple")
        self.get_logger().info("  /detection/closest_object")
        self.get_logger().info("  /detection/stats")
        self.get_logger().info("=" * 50)

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[5],
                'width': msg.width, 'height': msg.height
            }
            self.get_logger().info(f"Camera intrinsics: fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, "
                                 f"cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}")

    def depth_callback(self, msg):
        """Improved depth callback with better synchronization"""
        with self.depth_sync_lock:
            try:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.depth_frame_count += 1
                
                if self.depth_frame_count % 30 == 0:
                    # Check depth data validity
                    valid_pixels = np.sum((self.latest_depth_image > self.depth_min) & 
                                        (self.latest_depth_image < self.depth_max))
                    total_pixels = self.latest_depth_image.size
                    valid_pct = (valid_pixels / total_pixels) * 100
                    
                    self.get_logger().info(
                        f"Depth frame {self.depth_frame_count}: "
                        f"{valid_pct:.1f}% valid pixels, "
                        f"range: {self.latest_depth_image.min()}-{self.latest_depth_image.max()}mm",
                        throttle_duration_sec=5.0
                    )
            except Exception as e:
                self.get_logger().error(f"Depth callback error: {e}", throttle_duration_sec=1.0)

    def rgb_callback(self, msg):
        with self.frame_lock:
            try:
                frame_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = frame_bgr.shape[:2]

                # Get latest depth image with sync lock
                depth_image = None
                with self.depth_sync_lock:
                    if self.latest_depth_image is not None:
                        depth_image = self.latest_depth_image.copy()

                self.latest_frame_data = FrameData(
                    frame_bgr=frame_bgr,
                    frame_rgb=frame_rgb,
                    depth_image=depth_image,
                    header=msg.header,
                    width=w,
                    height=h,
                    timestamp=time.perf_counter()
                )
                
                self.rgb_frame_count += 1
                
            except Exception as e:
                self.get_logger().error(f"RGB callback error: {e}", throttle_duration_sec=1.0)

    def _publish_inference_state(self):
        msg = Bool()
        msg.data = self.inference_enabled
        self.inference_state_pub.publish(msg)

    def _pause_inference_cb(self, request, response):
        """SetBool service: request.data=True -> PAUSE, False -> RESUME.

        Pausing stops process_loop from issuing predict(), draining GPU work
        so a CONVERSING-state llama can take the GPU. The TensorRT engine is
        NOT unloaded (in-process release frees ~0 MB anyway; see ADR 0001),
        so resume is immediate.
        """
        should_pause = request.data
        self.inference_enabled = not should_pause
        if should_pause:
            # Let any in-flight predict() finish before we report drained.
            time.sleep(0.15)
        self._publish_inference_state()
        state = 'PAUSED' if should_pause else 'RESUMED'
        response.success = True
        response.message = f'YOLO inference {state} (engine stays resident)'
        self.get_logger().info(f'[gpu-mux] {response.message}')
        return response

    def process_loop(self):
        while self.running and rclpy.ok():
            # GPU multiplexing gate: when paused (CONVERSING state) skip all
            # inference so no GPU work is issued. Camera subs keep running;
            # we just drop frames until resumed.
            if not self.inference_enabled:
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()

            with self.frame_lock:
                if self.latest_frame_data is None:
                    time.sleep(0.001)
                    continue
                frame_data = self.latest_frame_data
                self.latest_frame_data = None

            try:
                self.process_frame(frame_data)
            except Exception as e:
                self.get_logger().error(f"Process error: {e}", throttle_duration_sec=1.0)

            elapsed = time.perf_counter() - loop_start
            if elapsed < self.min_frame_time:
                time.sleep(self.min_frame_time - elapsed)

    def process_frame(self, data: FrameData):
        t_start = time.perf_counter()

        futures = {}
        futures['yolo'] = self.thread_pool.submit(
            self.detect_objects, data.frame_bgr, data.depth_image)

        if self.enable_faces and self.face_detector:
            futures['face'] = self.thread_pool.submit(
                self.detect_faces, data.frame_rgb, data.depth_image, data.width, data.height)

        if self.enable_gestures and self.hands:
            futures['gesture'] = self.thread_pool.submit(
                self.detect_gestures, data.frame_rgb, data.depth_image, data.width, data.height)

        self.ocr_frame_count += 1
        if self.enable_ocr and (self.ocr_frame_count % self.ocr_interval == 0):
            futures['ocr'] = self.thread_pool.submit(
                self.detect_text, data.frame_bgr, data.depth_image)

        objects = futures['yolo'].result()
        faces = futures.get('face')
        faces = faces.result() if faces else []
        gestures = futures.get('gesture')
        gestures = gestures.result() if gestures else []
        ocr = futures.get('ocr')
        if ocr:
            self.ocr_results_cache = ocr.result()
        ocr_results = self.ocr_results_cache

        # Log detections for debugging (throttled to avoid spam)
        if faces and len(faces) > 0:
            self.get_logger().info(f"👤 Detected {len(faces)} face(s)", throttle_duration_sec=2.0)
        if gestures and len(gestures) > 0:
            gesture_types = [g.class_id for g in gestures]
            self.get_logger().info(f"🤚 Detected gestures: {gesture_types}", throttle_duration_sec=2.0)
        if ocr_results and len(ocr_results) > 0:
            ocr_texts = [o.class_id[:20] for o in ocr_results]  # Show first 20 chars
            self.get_logger().info(f"📝 Detected text: {ocr_texts}", throttle_duration_sec=2.0)

        with self.results_lock:
            self.latest_objects = objects
            self.latest_faces = faces
            self.latest_gestures = gestures
            self.latest_ocr = ocr_results

        self.publish_detections(data.header, objects, faces, gestures, ocr_results)

        t_end = time.perf_counter()
        fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
        self.fps_deque.append(fps)
        avg_fps = np.mean(self.fps_deque)

        fps_msg = Float32()
        fps_msg.data = float(avg_fps)
        self.fps_pub.publish(fps_msg)

        if self.show_window:
            vis_frame = self.visualize(data.frame_bgr.copy(), objects, faces, gestures,
                                       ocr_results, data.depth_image, avg_fps)
            try:
                cv2.namedWindow("YOLO Detection - RealSense D435", cv2.WINDOW_NORMAL)
                cv2.imshow("YOLO Detection - RealSense D435", vis_frame)
                key = cv2.waitKey(1) & 0xFF

                # Keyboard controls
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    self.get_logger().info("User requested shutdown via keyboard")
                    self.running = False
                elif key == ord('s'):  # 's' to save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"/tmp/detection_{timestamp}.png"
                    cv2.imwrite(filename, vis_frame)
                    self.get_logger().info(f"Screenshot saved: {filename}")
                elif key == ord('h'):  # 'h' for help
                    self.get_logger().info("Keyboard controls: [q/ESC] quit | [s] save screenshot | [h] help")

            except cv2.error as e:
                self.get_logger().warn(f"Display not available: {e}", throttle_duration_sec=5.0)
                self.show_window = False

            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis_frame, 'bgr8')
                vis_msg.header = data.header
                vis_msg.header.frame_id = self.frame_id
                self.annotated_pub.publish(vis_msg)
            except Exception:
                pass

        if data.depth_image is not None:
            self.publish_depth_overlay(data.depth_image, data.header, objects)

    def detect_objects(self, frame: np.ndarray, depth_image: Optional[np.ndarray]) -> List[DetectionResult]:
        t0 = time.perf_counter()
        results = []

        try:
            preds = self.model.predict(
                frame, conf=self.conf_threshold, iou=self.iou_threshold,
                imgsz=self.img_size, verbose=False, max_det=self.max_det,
                half=True, agnostic_nms=False, stream=False
            )

            if preds and preds[0].boxes is not None and len(preds[0].boxes) > 0:
                boxes = preds[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[i]
                    w, h = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    depth = self.get_depth_improved(depth_image, int(x1), int(y1), int(w), int(h))

                    results.append(DetectionResult(
                        class_id=self.model.names[cls[i]],
                        confidence=float(conf[i]),
                        cx=float(cx), cy=float(cy),
                        width=float(w), height=float(h),
                        depth=depth,
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2)
                    ))

        except Exception as e:
            self.get_logger().error(f"YOLO error: {e}", throttle_duration_sec=1.0)

        self.timing_stats['yolo'].append(time.perf_counter() - t0)
        return results

    def detect_faces(self, frame_rgb: np.ndarray, depth_image: Optional[np.ndarray],
                     w: int, h: int) -> List[DetectionResult]:
        t0 = time.perf_counter()
        results = []

        try:
            mp_results = self.face_detector.process(frame_rgb)
            if mp_results.detections:
                for det in mp_results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    cx = x1 + bw / 2.0
                    cy = y1 + bh / 2.0
                    conf = det.score[0]
                    depth = self.get_depth_improved(depth_image, x1, y1, bw, bh)

                    results.append(DetectionResult(
                        class_id='face', confidence=float(conf),
                        cx=float(cx), cy=float(cy),
                        width=float(bw), height=float(bh),
                        depth=depth,
                        x1=float(x1), y1=float(y1),
                        x2=float(x1 + bw), y2=float(y1 + bh)
                    ))
        except Exception as e:
            self.get_logger().error(f"Face error: {e}", throttle_duration_sec=1.0)

        self.timing_stats['face'].append(time.perf_counter() - t0)
        return results

    def detect_gestures(self, frame_rgb: np.ndarray, depth_image: Optional[np.ndarray],
                        w: int, h: int) -> List[DetectionResult]:
        t0 = time.perf_counter()
        results = []

        try:
            mp_results = self.hands.process(frame_rgb)
            if mp_results.multi_hand_landmarks and mp_results.multi_handedness:
                for landmarks, handedness in zip(mp_results.multi_hand_landmarks,
                                                  mp_results.multi_handedness):
                    xs = [lm.x * w for lm in landmarks.landmark]
                    ys = [lm.y * h for lm in landmarks.landmark]
                    x1, x2 = int(min(xs)) - 10, int(max(xs)) + 10
                    y1, y2 = int(min(ys)) - 10, int(max(ys)) + 10
                    bw, bh = x2 - x1, y2 - y1

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    conf = handedness.classification[0].score
                    hand_side = handedness.classification[0].label
                    gesture = self.classify_gesture(landmarks.landmark)
                    depth = self.get_depth_improved(depth_image, x1, y1, bw, bh)

                    results.append(DetectionResult(
                        class_id=f"{hand_side}_{gesture}",
                        confidence=float(conf),
                        cx=float(cx), cy=float(cy),
                        width=float(bw), height=float(bh),
                        depth=depth,
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2)
                    ))
        except Exception as e:
            self.get_logger().error(f"Gesture error: {e}", throttle_duration_sec=1.0)

        self.timing_stats['gesture'].append(time.perf_counter() - t0)
        return results

    def classify_gesture(self, landmarks) -> str:
        WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
        THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 3, 6, 10, 14, 18
        INDEX_MCP = 5

        def finger_up(tip, pip):
            return landmarks[tip].y < landmarks[pip].y

        def thumb_up():
            return abs(landmarks[THUMB_TIP].x - landmarks[WRIST].x) > \
                   abs(landmarks[THUMB_IP].x - landmarks[WRIST].x)

        thumb = thumb_up()
        index = finger_up(INDEX_TIP, INDEX_PIP)
        middle = finger_up(MIDDLE_TIP, MIDDLE_PIP)
        ring = finger_up(RING_TIP, RING_PIP)
        pinky = finger_up(PINKY_TIP, PINKY_PIP)

        fingers = [thumb, index, middle, ring, pinky]
        count = sum(fingers)

        if count == 0: return 'fist'
        if count == 5: return 'open'
        if index and not middle and not ring and not pinky:
            return 'gun' if thumb else 'point'
        if index and middle and not ring and not pinky:
            return 'peace'
        if thumb and not index and not middle and not ring and not pinky:
            if landmarks[THUMB_TIP].y < landmarks[INDEX_MCP].y:
                return 'thumbs_up'
            return 'thumbs_down'
        if thumb and pinky and not index and not middle and not ring:
            return 'call'
        if not thumb and not index and not ring and middle and not pinky:
            return 'middle_finger'
        if index and middle and ring and not pinky:
            return 'three'
        if index and middle and ring and pinky and not thumb:
            return 'four'
        if not thumb and index and not middle and not ring and pinky:
            return 'rock'

        return 'hand'

    def detect_text(self, frame: np.ndarray, depth_image: Optional[np.ndarray]) -> List[DetectionResult]:
        global easyocr
        t0 = time.perf_counter()
        results = []

        try:
            if self.ocr_reader is None:
                self.get_logger().info("Initializing EasyOCR (first use)...")
                import easyocr as easyocr_module
                easyocr = easyocr_module
                self.ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                self.get_logger().info("EasyOCR ready")

            scale = 0.5
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            ocr_results = self.ocr_reader.readtext(small, paragraph=False)

            for (bbox, text, conf) in ocr_results:
                if conf < 0.3 or len(text.strip()) < 2:
                    continue

                pts = np.array(bbox) / scale
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                depth = self.get_depth_improved(depth_image, int(x1), int(y1), int(w), int(h))

                results.append(DetectionResult(
                    class_id=text.strip(), confidence=float(conf),
                    cx=float(cx), cy=float(cy),
                    width=float(w), height=float(h),
                    depth=depth,
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2)
                ))

        except ImportError:
            self.get_logger().warn("EasyOCR not installed", throttle_duration_sec=10.0)
            self.enable_ocr = False
        except Exception as e:
            self.get_logger().error(f"OCR error: {e}", throttle_duration_sec=1.0)

        self.timing_stats['ocr'].append(time.perf_counter() - t0)
        return results

    def get_depth_improved(self, depth_image: Optional[np.ndarray], x: int, y: int,
                          w: int, h: int) -> Optional[float]:
        """Improved multi-zone depth estimation"""
        if depth_image is None or w <= 0 or h <= 0:
            return None
        
        try:
            img_h, img_w = depth_image.shape
            
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            margin_ratio = (1.0 - self.depth_sample_ratio) / 2.0
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)
            
            zones = []
            
            # Zone 1: Center
            cx1 = max(x1, x1 + margin_x)
            cy1 = max(y1, y1 + margin_y)
            cx2 = min(x2, x2 - margin_x)
            cy2 = min(y2, y2 - margin_y)
            if cx2 > cx1 and cy2 > cy1:
                zones.append(depth_image[cy1:cy2, cx1:cx2])
            
            # Zone 2: Upper-center
            uy1 = max(y1, y1 + int(margin_y * 0.5))
            uy2 = min(y2, y1 + h // 2)
            if uy2 > uy1 and cx2 > cx1:
                zones.append(depth_image[uy1:uy2, cx1:cx2])
            
            # Zone 3: Full bbox
            zones.append(depth_image[y1:y2, x1:x2])
            
            for zone in zones:
                if zone.size == 0:
                    continue
                
                valid = zone[(zone > self.depth_min) & (zone < self.depth_max)]
                
                if len(valid) >= 5:
                    depth_mm = float(np.percentile(valid, self.depth_percentile))
                    depth_m = depth_mm / 1000.0
                    
                    if 0.1 <= depth_m <= 10.0:
                        return depth_m
            
        except Exception as e:
            self.get_logger().debug(f"Depth error: {e}", throttle_duration_sec=5.0)
        
        return None

    def publish_detections(self, header, objects: List[DetectionResult],
                          faces: List[DetectionResult], gestures: List[DetectionResult],
                          ocr: List[DetectionResult]):
        
        def make_simple_msg(detections: List[DetectionResult]) -> str:
            if not detections:
                return "[]"
            
            items = []
            for det in detections:
                item = {
                    'class': det.class_id,
                    'conf': round(det.confidence, 2),
                    'center': [round(det.cx, 1), round(det.cy, 1)],
                }
                if det.depth is not None:
                    item['depth'] = round(det.depth, 2)
                items.append(item)
            
            return json.dumps(items, separators=(',', ':'))

        # Publish simple/concise versions
        self.objects_simple_pub.publish(String(data=make_simple_msg(objects)))
        self.faces_simple_pub.publish(String(data=make_simple_msg(faces)))
        self.gestures_simple_pub.publish(String(data=make_simple_msg(gestures)))
        self.ocr_simple_pub.publish(String(data=make_simple_msg(ocr)))

        # Closest object
        all_dets = objects + faces + gestures
        closest = None
        closest_dist = float('inf')
        for det in all_dets:
            if det.depth is not None and 0 < det.depth < closest_dist:
                closest_dist = det.depth
                closest = det

        if closest:
            close_msg = String()
            close_msg.data = json.dumps({
                'class': closest.class_id,
                'distance': round(closest.depth, 2),
                'confidence': round(closest.confidence, 2)
            })
            self.closest_object_pub.publish(close_msg)

        # Stats
        stats = {
            'fps': round(float(np.mean(self.fps_deque)), 1) if len(self.fps_deque) > 0 else 0,
            'counts': {
                'objects': len(objects),
                'faces': len(faces),
                'gestures': len(gestures),
                'ocr': len(ocr)
            },
            'depth_available': any(d.depth is not None for d in all_dets),
            'timings_ms': {
                'yolo': round(np.mean(self.timing_stats['yolo']) * 1000, 1) if self.timing_stats['yolo'] else 0,
                'face': round(np.mean(self.timing_stats['face']) * 1000, 1) if self.timing_stats['face'] else 0,
                'gesture': round(np.mean(self.timing_stats['gesture']) * 1000, 1) if self.timing_stats['gesture'] else 0,
                'ocr': round(np.mean(self.timing_stats['ocr']) * 1000, 1) if self.timing_stats['ocr'] else 0,
            }
        }
        self.stats_pub.publish(String(data=json.dumps(stats)))

    def visualize(self, frame: np.ndarray, objects: List[DetectionResult],
                  faces: List[DetectionResult], gestures: List[DetectionResult],
                  ocr: List[DetectionResult], depth_image: Optional[np.ndarray],
                  fps: float) -> np.ndarray:

        h, w = frame.shape[:2]

        # Create side panel for detailed info (300px wide)
        panel_width = 320
        canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = (40, 40, 40)  # Dark gray panel

        # Draw semi-transparent overlay for legend
        legend_overlay = canvas.copy()

        # Helper function to draw enhanced bounding box with center marker
        def draw_detection(img, det, color, label_prefix=""):
            x1, y1 = int(det.x1), int(det.y1)
            x2, y2 = int(det.x2), int(det.y2)
            cx, cy = int(det.cx), int(det.cy)

            # Draw bounding box with rounded corners effect (multi-line)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 1)

            # Draw center marker (crosshair)
            marker_size = 8
            cv2.line(img, (cx - marker_size, cy), (cx + marker_size, cy), color, 2)
            cv2.line(img, (cx, cy - marker_size), (cx, cy + marker_size), color, 2)
            cv2.circle(img, (cx, cy), 3, color, -1)
            cv2.circle(img, (cx, cy), 4, (255, 255, 255), 1)

            # Build label with all info
            label_parts = []
            if label_prefix:
                label_parts.append(label_prefix)
            label_parts.append(f"{det.class_id}")
            label_parts.append(f"{det.confidence:.2%}")
            if det.depth is not None:
                label_parts.append(f"{det.depth:.2f}m")

            label = " | ".join(label_parts)

            # Draw label background with padding
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label above box, or below if too close to top
            if y1 > label_h + 10:
                label_y = y1 - 5
                bg_y1, bg_y2 = y1 - label_h - 8, y1 - 2
            else:
                label_y = y2 + label_h + 5
                bg_y1, bg_y2 = y2 + 2, y2 + label_h + 8

            # Draw label background
            cv2.rectangle(img, (x1, bg_y1), (x1 + label_w + 8, bg_y2), color, -1)
            cv2.rectangle(img, (x1, bg_y1), (x1 + label_w + 8, bg_y2), (255, 255, 255), 1)

            # Draw label text
            cv2.putText(img, label, (x1 + 4, label_y), font, font_scale, (0, 0, 0), 2)
            cv2.putText(img, label, (x1 + 4, label_y), font, font_scale, (255, 255, 255), 1)

        # Draw all detections
        for det in objects:
            draw_detection(canvas, det, (0, 255, 0))  # Green

        for det in faces:
            draw_detection(canvas, det, (255, 100, 0), "FACE")  # Blue

        for det in gestures:
            draw_detection(canvas, det, (0, 255, 255), "GESTURE")  # Yellow

        for det in ocr:
            draw_detection(canvas, det, (255, 0, 255), "TEXT")  # Magenta

        # ===== SIDE PANEL INFO =====
        panel_x = w + 10
        panel_y = 20
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        cv2.putText(canvas, "DETECTION INFO", (panel_x, panel_y),
                   font, 0.6, (255, 255, 255), 2)
        panel_y += 30

        # FPS and performance
        cv2.rectangle(canvas, (panel_x - 5, panel_y - 18), (panel_x + 300, panel_y + 2), (60, 60, 60), -1)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (panel_x, panel_y),
                   font, 0.7, (0, 255, 255), 2)
        panel_y += 30

        # Detection counts with color coding
        counts = [
            (f"Objects: {len(objects)}", (0, 255, 0)),
            (f"Faces: {len(faces)}", (255, 100, 0)),
            (f"Gestures: {len(gestures)}", (0, 255, 255)),
            (f"OCR Texts: {len(ocr)}", (255, 0, 255))
        ]

        for text, color in counts:
            cv2.rectangle(canvas, (panel_x - 3, panel_y - 15), (panel_x + 6, panel_y - 5), color, -1)
            cv2.putText(canvas, text, (panel_x + 15, panel_y), font, 0.5, color, 1)
            panel_y += 22

        panel_y += 10
        cv2.line(canvas, (panel_x, panel_y), (panel_x + 290, panel_y), (100, 100, 100), 1)
        panel_y += 20

        # Camera info
        if self.camera_intrinsics:
            cv2.putText(canvas, "CAMERA", (panel_x, panel_y), font, 0.5, (200, 200, 200), 1)
            panel_y += 18
            cv2.putText(canvas, f"{self.camera_intrinsics['width']}x{self.camera_intrinsics['height']}",
                       (panel_x, panel_y), font, 0.45, (150, 150, 150), 1)
            panel_y += 16
            cv2.putText(canvas, f"fx:{self.camera_intrinsics['fx']:.0f} fy:{self.camera_intrinsics['fy']:.0f}",
                       (panel_x, panel_y), font, 0.4, (150, 150, 150), 1)
            panel_y += 16
            cv2.putText(canvas, f"cx:{self.camera_intrinsics['cx']:.0f} cy:{self.camera_intrinsics['cy']:.0f}",
                       (panel_x, panel_y), font, 0.4, (150, 150, 150), 1)
            panel_y += 20

        # Depth info
        if depth_image is not None:
            valid_pixels = np.sum((depth_image > self.depth_min) & (depth_image < self.depth_max))
            total = depth_image.size
            validity_pct = (valid_pixels / total) * 100

            cv2.putText(canvas, "DEPTH", (panel_x, panel_y), font, 0.5, (0, 255, 255), 1)
            panel_y += 18
            cv2.putText(canvas, f"Valid: {validity_pct:.1f}%", (panel_x, panel_y),
                       font, 0.45, (0, 255, 255) if validity_pct > 50 else (0, 165, 255), 1)
            panel_y += 16
            cv2.putText(canvas, f"Range: {self.depth_min:.1f}-{self.depth_max:.0f}m",
                       (panel_x, panel_y), font, 0.4, (150, 150, 150), 1)
            panel_y += 20
        else:
            cv2.putText(canvas, "DEPTH: NO DATA", (panel_x, panel_y),
                       font, 0.5, (0, 0, 255), 1)
            panel_y += 25

        # Detailed object list
        panel_y += 10
        cv2.line(canvas, (panel_x, panel_y), (panel_x + 290, panel_y), (100, 100, 100), 1)
        panel_y += 20

        cv2.putText(canvas, "DETECTIONS", (panel_x, panel_y), font, 0.5, (255, 255, 255), 1)
        panel_y += 20

        # Show detailed list (limited to fit panel)
        all_detections = []
        all_detections.extend([("OBJ", det, (0, 255, 0)) for det in objects])
        all_detections.extend([("FACE", det, (255, 100, 0)) for det in faces])
        all_detections.extend([("GEST", det, (0, 255, 255)) for det in gestures])
        all_detections.extend([("TEXT", det, (255, 0, 255)) for det in ocr])

        # Sort by distance (closest first)
        all_detections.sort(key=lambda x: x[1].depth if x[1].depth is not None else 999.0)

        max_list_items = min(15, len(all_detections))
        for i, (dtype, det, color) in enumerate(all_detections[:max_list_items]):
            if panel_y > h - 25:
                break

            # Type indicator
            cv2.rectangle(canvas, (panel_x - 3, panel_y - 10), (panel_x + 25, panel_y), color, -1)
            cv2.putText(canvas, dtype, (panel_x, panel_y - 2), font, 0.35, (0, 0, 0), 1)

            # Object name (truncate if too long)
            name = det.class_id[:12] if len(det.class_id) > 12 else det.class_id
            cv2.putText(canvas, name, (panel_x + 32, panel_y - 2), font, 0.4, (200, 200, 200), 1)

            # Depth
            if det.depth is not None:
                depth_text = f"{det.depth:.2f}m"
                cv2.putText(canvas, depth_text, (panel_x + 150, panel_y - 2),
                           font, 0.4, (0, 255, 255), 1)

            # Confidence
            conf_text = f"{det.confidence:.0%}"
            cv2.putText(canvas, conf_text, (panel_x + 230, panel_y - 2),
                       font, 0.35, (150, 150, 150), 1)

            panel_y += 16

        # Show overflow indicator
        if len(all_detections) > max_list_items:
            remaining = len(all_detections) - max_list_items
            cv2.putText(canvas, f"+ {remaining} more...", (panel_x, panel_y),
                       font, 0.4, (100, 100, 100), 1)

        # Add legend at bottom of main frame
        legend_y = h - 60
        cv2.rectangle(canvas, (5, legend_y - 5), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.rectangle(canvas, (5, legend_y - 5), (w - 5, h - 5), (255, 255, 255), 1)

        cv2.putText(canvas, "LEGEND:", (15, legend_y + 12), font, 0.45, (255, 255, 255), 1)

        legend_items = [
            ("Objects", (0, 255, 0)),
            ("Faces", (255, 100, 0)),
            ("Gestures", (0, 255, 255)),
            ("Text/OCR", (255, 0, 255))
        ]

        legend_x = 100
        for label, color in legend_items:
            cv2.rectangle(canvas, (legend_x, legend_y + 2), (legend_x + 15, legend_y + 14), color, -1)
            cv2.rectangle(canvas, (legend_x, legend_y + 2), (legend_x + 15, legend_y + 14), (255, 255, 255), 1)
            cv2.putText(canvas, label, (legend_x + 20, legend_y + 12), font, 0.4, (200, 200, 200), 1)
            legend_x += 120

        return canvas

    def publish_depth_overlay(self, depth_image: np.ndarray, header, objects: List[DetectionResult]):
        try:
            depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            for det in objects:
                x1, y1 = int(det.x1), int(det.y1)
                x2, y2 = int(det.x2), int(det.y2)
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
                if det.depth is not None:
                    label = f"{det.class_id}: {det.depth:.2f}m"
                    cv2.putText(depth_color, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            depth_msg = self.bridge.cv2_to_imgmsg(depth_color, 'bgr8')
            depth_msg.header = header
            depth_msg.header.frame_id = self.frame_id
            self.depth_overlay_pub.publish(depth_msg)
        except Exception as e:
            self.get_logger().error(f"Depth overlay error: {e}", throttle_duration_sec=5.0)

    def __del__(self):
        self.running = False
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        if hasattr(node, 'show_window') and node.show_window:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()