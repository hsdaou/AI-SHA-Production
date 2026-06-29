#!/usr/bin/env python3
"""
AI-SHA Farm Brain Launch File

Launches the fully autonomous farm monitoring pipeline:
  1. RealSense D435 (RGB-D perception)
  2. YOLOv8 detection + plant disease classifier
  3. LiDAR driver (LD-19)
  4. STT node (voice override channel)
  5. LLM node (intent parsing)
  6. Farm Brain orchestrator (autonomous -- starts patrol automatically)

The robot operates autonomously by default. Voice commands are an override.

Usage:
    ros2 launch robot_brain farm_brain.launch.py

    # With Isaac Sim RL policy:
    ros2 launch robot_brain farm_brain.launch.py \
        isaac_model_path:=/path/to/policy.engine

    # With custom farm layout:
    ros2 launch robot_brain farm_brain.launch.py \
        farm_locations_file:=/path/to/locations.json
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    dds_env = {
        'ROS_DOMAIN_ID': '0',
        'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
    }

    # ── Launch arguments ──────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('nav2_enabled', default_value='true'),
        DeclareLaunchArgument('obstacle_avoidance_enabled', default_value='true'),
        DeclareLaunchArgument('auto_start', default_value='true',
                              description='Begin autonomous patrol on startup'),
        DeclareLaunchArgument('farm_locations_file', default_value=''),
        DeclareLaunchArgument('isaac_model_path', default_value=''),
        DeclareLaunchArgument('patrol_interval_sec', default_value='300.0'),
        DeclareLaunchArgument(
            'yolo_model_path',
            default_value=os.path.expanduser('~/robot_ws/yolov8m.engine')),
        DeclareLaunchArgument('stt_model_size', default_value='tiny'),
    ]

    # ── RealSense Camera ─────────────────────────────────────────────
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
        additional_env=dds_env,
    )

    # ── YOLOv8 Detection + Plant Disease ─────────────────────────────
    yolo_node = TimerAction(
        period=3.0,
        actions=[Node(
            package='yolov8_ros',
            executable='yolov8_node',
            name='detection_node',
            parameters=[{
                'model_path': LaunchConfiguration('yolo_model_path'),
                'use_depth': True,
                'show_window': False,
                'confidence_threshold': 0.4,
                'target_fps': 30,
                'enable_disease_classifier': True,
            }],
            output='screen',
            emulate_tty=True,
            additional_env=dds_env,
        )],
    )

    # ── STT (voice override channel) ─────────────────────────────────
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
        additional_env=dds_env,
    )

    # ── LLM Node (intent parsing for voice overrides) ────────────────
    llm_node = TimerAction(
        period=5.0,
        actions=[Node(
            package='robot_brain',
            executable='robot_brain',
            name='robot_brain',
            parameters=[{
                'local_model_path': os.path.expanduser(
                    '~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf'),
                'n_gpu_layers': 20,
                'temperature': 0.7,
                'max_tokens': 150,
            }],
            output='screen',
            emulate_tty=True,
            additional_env=dds_env,
        )],
    )

    # ── Farm Brain Orchestrator (autonomous) ─────────────────────────
    farm_brain_node = TimerAction(
        period=8.0,
        actions=[Node(
            package='robot_brain',
            executable='farm_brain',
            name='farm_brain',
            parameters=[{
                'auto_start': LaunchConfiguration('auto_start'),
                'nav2_enabled': LaunchConfiguration('nav2_enabled'),
                'obstacle_avoidance_enabled': LaunchConfiguration(
                    'obstacle_avoidance_enabled'),
                'patrol_interval_sec': LaunchConfiguration(
                    'patrol_interval_sec'),
                'farm_locations_file': LaunchConfiguration(
                    'farm_locations_file'),
                'isaac_model_path': LaunchConfiguration('isaac_model_path'),
            }],
            output='screen',
            emulate_tty=True,
            additional_env=dds_env,
        )],
    )

    # ── Startup banner ───────────────────────────────────────────────
    banner = LogInfo(msg=[
        '\n',
        '=' * 70, '\n',
        '  AI-SHA FARM BRAIN - Fully Autonomous Agricultural Robot\n',
        '=' * 70, '\n',
        '  Mode: AUTONOMOUS (voice = override only)\n',
        '\n',
        '  Nodes:\n',
        '    RealSense D435 .... RGB-D perception\n',
        '    YOLOv8 + Disease .. Plant disease detection (TensorRT)\n',
        '    STT ............... Voice override channel (Whisper GPU)\n',
        '    LLM ............... Intent parsing (Llama 3.2)\n',
        '    Farm Brain ........ Autonomous orchestrator\n',
        '\n',
        '  Autonomous Behaviour:\n',
        '    Patrol all sections -> deploy probes -> measure sensors\n',
        '    -> inspect for disease -> retract probes -> analyse\n',
        '    -> auto-water if dry -> advance -> return home\n',
        '\n',
        '  Voice Overrides: stop, pause, resume, skip, report,\n',
        '                   go to <location>, water, sow\n',
        '\n',
        '=' * 70, '\n',
    ])

    return LaunchDescription([
        *args,
        banner,
        realsense_node,
        stt_node,
        yolo_node,
        llm_node,
        farm_brain_node,
    ])
