#!/usr/bin/env python3
"""
Project Cerebro - General Purpose Robot Brain
Jetson Orin Nano - LLM text generator with optional vision context
Acts as the "mouth of the robot" - generates responses to any query
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import threading
import time
import os
from typing import Optional, Dict, List
import numpy as np

# LLM imports
from llama_cpp import Llama


class RobotBrain(Node):
    def __init__(self):
        super().__init__('robot_brain')

        # Parameters
        self.declare_parameter('local_model_path', '/home/orin-robot/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf')
        self.declare_parameter('prompt_file', '/home/orin-robot/robot_ws/robot_prompt.txt')
        self.declare_parameter('n_ctx', 2048)
        self.declare_parameter('n_gpu_layers', -1)
        self.declare_parameter('temperature', 0.7)
        self.declare_parameter('max_tokens', 150)

        # Get parameters
        self.local_model_path = self.get_parameter('local_model_path').value
        self.prompt_file = self.get_parameter('prompt_file').value
        self.n_ctx = self.get_parameter('n_ctx').value
        self.n_gpu_layers = self.get_parameter('n_gpu_layers').value
        self.temperature = self.get_parameter('temperature').value
        self.max_tokens = self.get_parameter('max_tokens').value

        # Load system prompt from file
        self.system_prompt = self._load_prompt_file()

        # State
        self.latest_detections = []
        self.detections_lock = threading.Lock()

        # Model loading
        self.local_llm = None
        self.local_llm_loaded = False
        self.llm_lock = threading.Lock()

        # Publishers
        self.response_pub = self.create_publisher(String, '/speech/text', 10)

        # Subscribers
        self.speech_sub = self.create_subscription(
            String, '/speech_rec', self.speech_callback, 10)

        self.detection_sub = self.create_subscription(
            String, '/detection/objects_simple', self.detection_callback, 10)

        # Load local LLM in background
        self.get_logger().info('=' * 60)
        self.get_logger().info('PROJECT CEREBRO - Robot Brain')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Loading LLM: {os.path.basename(self.local_model_path)}')
        threading.Thread(target=self._load_local_llm, daemon=True).start()

        # Give status update
        self.create_timer(5.0, self._status_update)
        self.startup_logged = False

    def _load_prompt_file(self) -> str:
        """Load system prompt from file"""
        try:
            if os.path.exists(self.prompt_file):
                with open(self.prompt_file, 'r') as f:
                    prompt = f.read().strip()
                self.get_logger().info(f'✓ Loaded prompt from: {self.prompt_file}')
                return prompt
            else:
                self.get_logger().warn(f'Prompt file not found: {self.prompt_file}')
                return "You are a helpful robot assistant. Answer questions briefly and clearly."
        except Exception as e:
            self.get_logger().error(f'Failed to load prompt file: {e}')
            return "You are a helpful robot assistant. Answer questions briefly and clearly."

    def _status_update(self):
        """Periodic status update"""
        if not self.startup_logged:
            with self.llm_lock:
                if self.local_llm_loaded:
                    self.get_logger().info('=' * 60)
                    self.get_logger().info('✓ System Ready - General Purpose Text Generator')
                    self.get_logger().info('  Listening: /speech_rec')
                    self.get_logger().info('  Publishing: /speech/text')
                    self.get_logger().info('  Mode: Responds to any query (vision-enhanced)')
                    self.get_logger().info('=' * 60)
                    self.startup_logged = True

    def _load_local_llm(self):
        """Load local Llama model"""
        try:
            self.local_llm = Llama(
                model_path=self.local_model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            with self.llm_lock:
                self.local_llm_loaded = True
            self.get_logger().info('✓ LLM loaded successfully (GPU accelerated)')
        except Exception as e:
            self.get_logger().error(f'✗ LLM load failed: {e}')

    def detection_callback(self, msg):
        """Store latest YOLO detections"""
        with self.detections_lock:
            try:
                self.latest_detections = json.loads(msg.data)
            except:
                self.latest_detections = []

    def speech_callback(self, msg):
        """Handle incoming speech - IMMEDIATE RESPONSE"""
        query = msg.data.strip()

        if not query:
            return

        self.get_logger().info(f'🎤 Query: "{query}"')

        # Process immediately in separate thread
        threading.Thread(target=self._handle_query, args=(query,), daemon=True).start()

    def _handle_query(self, query: str):
        """Process query and generate response - GENERAL PURPOSE"""
        start_time = time.time()

        # Check if model loaded
        with self.llm_lock:
            if not self.local_llm_loaded:
                self.get_logger().warn('Model still loading...')
                response = "My AI brain is still initializing. Please wait a moment."
                self._publish_response(response)
                return

        # Check if query seems vision-related
        vision_keywords = ['see', 'what', 'where', 'how many', 'show', 'look', 'find', 'detect', 'there', 'identify', 'spot', 'notice']
        is_vision_query = any(keyword in query.lower() for keyword in vision_keywords)

        # Get detections only if vision-related
        vision_context = None
        if is_vision_query:
            with self.detections_lock:
                detections = self.latest_detections.copy()
            if detections:
                vision_context = self._build_context(detections)
                self.get_logger().info(f'🧠 Vision query: {len(detections)} objects in view')
            else:
                vision_context = "I don't see anything in my current field of view."
                self.get_logger().info('🧠 Vision query: no objects detected')
        else:
            self.get_logger().info('🧠 General query (no vision context)')

        # Generate response
        response = self._query_llm(query, vision_context)

        elapsed = time.time() - start_time

        if response:
            self.get_logger().info(f'✓ Response ({elapsed:.1f}s): "{response}"')
            self._publish_response(response)
        else:
            self.get_logger().error('✗ No response generated')

    def _build_context(self, detections: List[Dict]) -> str:
        """Build context string from detections"""
        if not detections:
            return "I currently see nothing in my field of view."

        # Aggregate by class
        class_info = {}
        for det in detections:
            cls = det.get('class', 'object')
            depth = det.get('depth')

            if cls not in class_info:
                class_info[cls] = {'count': 0, 'depths': []}

            class_info[cls]['count'] += 1
            if depth:
                class_info[cls]['depths'].append(depth)

        # Build description
        items = []
        for cls, info in class_info.items():
            count = info['count']
            depths = info['depths']

            if count == 1:
                item = f"a {cls}"
            else:
                item = f"{count} {cls}s"

            if depths:
                avg_depth = np.mean(depths)
                item += f" at {avg_depth:.1f}m"

            items.append(item)

        return "I can see: " + ", ".join(items) + "."

    def _query_llm(self, query: str, vision_context: Optional[str] = None) -> Optional[str]:
        """Query local LLM - general purpose with optional vision context"""
        try:
            # Build prompt with optional vision context
            if vision_context:
                user_message = f"Visual Context: {vision_context}\n\nQuestion: {query}"
            else:
                user_message = f"Question: {query}"

            prompt = f"""<|start_header_id|>system<|end_header_id|>

{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

            # Generate - with lock to prevent concurrent access (llama_cpp not thread-safe)
            with self.llm_lock:
                response = self.local_llm(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False
                )

            return response['choices'][0]['text'].strip()

        except Exception as e:
            self.get_logger().error(f'LLM error: {e}')
            return "I'm having trouble processing that right now."

    def _publish_response(self, text: str):
        """Publish response to TTS"""
        msg = String()
        msg.data = text
        self.response_pub.publish(msg)
        self.get_logger().info(f'📢 Published to /speech/text')


def main(args=None):
    # Set RMW for compatibility - MUST match other nodes
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    os.environ['ROS_DOMAIN_ID'] = '0'  # Default domain

    rclpy.init(args=args)

    try:
        brain = RobotBrain()
        print(f"Brain node starting: {brain.get_name()}")
        print(f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID')}")
        print(f"RMW: {os.environ.get('RMW_IMPLEMENTATION')}")
        rclpy.spin(brain)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
