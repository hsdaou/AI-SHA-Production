#!/usr/bin/env python3
"""
Project Cerebro - Unified Launch File
Jetson Orin Nano Complete System Launch

Launches:
- RealSense D435 camera (RGB-D)
- YOLO object detection (TensorRT)
- STT node (Whisper speech recognition)
- Robot brain (LLM decision engine)

Usage:
    ros2 launch robot_bringup cerebro.launch.py

Optional parameters:
    stt_model_size:=tiny|base|small    (default: tiny)
    yolo_model_path:=/path/to/model    (default: ~/robot_ws/yolov8m.engine)
    enable_faces:=true|false            (default: false)
    enable_gestures:=true|false         (default: false)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Common DDS configuration for all nodes
    dds_config = {
        'ROS_DOMAIN_ID': '0',
        'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp'
    }
    # Declare launch arguments
    declare_stt_model = DeclareLaunchArgument(
        'stt_model_size',
        default_value='tiny',
        description='Whisper model size: tiny, base, small, medium (tiny recommended for memory)'
    )

    declare_yolo_model = DeclareLaunchArgument(
        'yolo_model_path',
        default_value=os.path.expanduser('~/robot_ws/yolov8m.engine'),
        description='Path to YOLO TensorRT engine'
    )

    declare_enable_faces = DeclareLaunchArgument(
        'enable_faces',
        default_value='false',
        description='Enable MediaPipe face detection'
    )

    declare_enable_gestures = DeclareLaunchArgument(
        'enable_gestures',
        default_value='false',
        description='Enable MediaPipe gesture detection'
    )

    declare_enable_ocr = DeclareLaunchArgument(
        'enable_ocr',
        default_value='false',
        description='Enable EasyOCR text recognition'
    )

    # RealSense Camera Node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'align_depth.enable': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'depth_module.depth_profile': '848x480x30',
            'rgb_camera.color_profile': '640x480x30',
        }],
        output='screen',
        emulate_tty=True,
        additional_env=dds_config
    )

    # YOLO Detection Node (delayed start to let camera initialize)
    yolo_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='yolov8_ros',
                executable='yolov8_node',
                name='detection_node',
                parameters=[{
                    'model_path': LaunchConfiguration('yolo_model_path'),
                    'enable_faces': LaunchConfiguration('enable_faces'),
                    'enable_gestures': LaunchConfiguration('enable_gestures'),
                    'enable_ocr': LaunchConfiguration('enable_ocr'),
                    'use_depth': True,
                    'show_window': False,  # No display on headless system
                    'confidence_threshold': 0.4,
                    'target_fps': 60,
                }],
                output='screen',
                emulate_tty=True,
                additional_env=dds_config
            )
        ]
    )

    # STT Node (Speech Recognition)
    stt_node = Node(
        package='stt_node',
        executable='stt_node',
        name='stt_node',
        parameters=[{
            'model_size': LaunchConfiguration('stt_model_size'),
            'sample_rate': 16000,
            'channels': 1,
            'silence_threshold': 0.015,
            'language': 'en',
        }],
        output='screen',
        emulate_tty=True,
        additional_env=dds_config
    )

    # Robot Brain (LLM Text Generator) - delayed to let everything initialize
    brain_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='robot_brain',
                executable='robot_brain',
                name='robot_brain',
                parameters=[{
                    'local_model_path': os.path.expanduser('~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf'),
                    'prompt_file': os.path.expanduser('~/robot_ws/robot_prompt.txt'),
                    'n_gpu_layers': 20,  # Reduced from -1 to leave GPU memory for YOLO (~1.5GB for LLM, rest for YOLO)
                    'temperature': 0.7,
                    'max_tokens': 150,
                }],
                output='screen',
                emulate_tty=True,
                additional_env=dds_config
            )
        ]
    )

    # Startup message
    startup_msg = LogInfo(
        msg=[
            '\n',
            '=' * 70, '\n',
            '  PROJECT CEREBRO - Multimodal Robot System\n',
            '=' * 70, '\n',
            '  Jetson Orin Nano - Unified Launch\n',
            '\n',
            '  Components:\n',
            '    - RealSense D435 Camera (RGB-D)\n',
            '    - YOLO Object Detection (TensorRT)\n',
            '    - STT Node (Whisper GPU/CPU)\n',
            '    - Robot Brain (Llama-3.2-3B LLM)\n',
            '\n',
            '  Topics:\n',
            '    /speech_rec         - Speech input (from STT)\n',
            '    /speech/text        - Speech output (to TTS)\n',
            '    /detection/objects_simple - Detected objects\n',
            '    /camera/camera/color/image_raw - RGB feed\n',
            '    /camera/camera/aligned_depth_to_color/image_raw - Depth\n',
            '\n',
            '  Usage:\n',
            '    ros2 topic echo /speech/text     # Monitor brain responses\n',
            '    ros2 topic echo /detection/objects_simple  # See detections\n',
            '\n',
            '=' * 70, '\n'
        ]
    )

    return LaunchDescription([
        # Arguments
        declare_stt_model,
        declare_yolo_model,
        declare_enable_faces,
        declare_enable_gestures,
        declare_enable_ocr,

        # Startup message
        startup_msg,

        # Nodes (sequential with delays)
        realsense_node,
        stt_node,
        yolo_node,
        brain_node,
    ])
