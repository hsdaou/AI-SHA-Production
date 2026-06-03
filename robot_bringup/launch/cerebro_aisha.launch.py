#!/usr/bin/env python3
"""
Project Cerebro (AI-SHA edition) — Jetson Orin Nano launch.

Same hardware/vision/STT stack as cerebro.launch.py, but the reasoning
brain is swapped from the old single robot_brain (local GGUF Llama) to the
AI-SHA brain stack: brain_node (intent router) + admin_node (RAG knowledge
base) + action_node, all served by a local Ollama instance.

Topic graph (unchanged interfaces):
    stt_node          --> /speech/text              (speech in)
    yolo detection    --> /detection/objects_simple (vision context)
    brain_node        --> /robot_speech             (speech out, to TTS)
    brain_node        --> /admin_task --> admin_node --> /admin_response
    brain_node forwards /admin_response & /action_response to /robot_speech

Prerequisites:
    - Ollama running on the Jetson at 127.0.0.1:11434 with the RAG model pulled:
        ollama pull llama3.2:1b     (admin_node RAG)
      (brain_node routing is rule-based — no LLM router model needed.)
    - Knowledge base built at src/aisha_brain/aisha_knowledge_db
      (jetson_launch.py auto-syncs it into the install share dir on start).
    - TTS runs on the RPi5 and MUST subscribe to /robot_speech
      (the new topic) instead of the old /speech/text.

Usage:
    ros2 launch robot_bringup cerebro_aisha.launch.py
    ros2 launch robot_bringup cerebro_aisha.launch.py llm_model:=llama3.2:1b enable_stt:=false
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Common DDS configuration for all nodes (matches cerebro.launch.py)
    dds_config = {
        'ROS_DOMAIN_ID': '0',
        'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
    }

    declare_stt_model = DeclareLaunchArgument(
        'stt_model_size', default_value='tiny',
        description='Whisper model size: tiny, base, small, medium')
    declare_yolo_model = DeclareLaunchArgument(
        'yolo_model_path',
        default_value=os.path.expanduser('~/robot_ws/yolov8m.engine'),
        description='Path to YOLO TensorRT engine')
    declare_enable_faces = DeclareLaunchArgument(
        'enable_faces', default_value='false')
    declare_enable_gestures = DeclareLaunchArgument(
        'enable_gestures', default_value='false')
    declare_enable_ocr = DeclareLaunchArgument(
        'enable_ocr', default_value='false')
    # Toggle the Jetson-side STT. Set false if STT runs on the RPi instead
    # (aisha_brain rpi_launch.py), to avoid two nodes publishing /speech/text.
    declare_enable_stt = DeclareLaunchArgument(
        'enable_stt', default_value='true',
        description='Run STT on the Jetson (publishes /speech/text)')
    # Ollama model selection, forwarded to aisha_brain jetson_launch.py.
    # llama3.2:1b (CPU) ~8-15s/answer vs ~60s for the 3B on this 8GB Jetson.
    # Override with llm_model:=llama3.2 for better quality at higher latency.
    declare_llm_model = DeclareLaunchArgument(
        'llm_model', default_value='llama3.2:1b')
    # GPU layers for the admin RAG model. 0 = CPU-only (safe with YOLO up).
    # Partial offload (e.g. 10) trims ~30% latency but contends for VRAM with
    # the YOLO TensorRT engine — only raise while load-testing the full stack.
    declare_llm_num_gpu = DeclareLaunchArgument(
        'llm_num_gpu', default_value='0')
    # GPU arbiter (ADR 0001): time-multiplex the GPU between vision (NAVIGATING)
    # and the LLM (CONVERSING). When true, the arbiter OWNS yolov8_node (spawns
    # /kills it to free the GPU) so the standalone YOLO node is NOT launched.
    # Set false to fall back to the old always-on-vision behaviour.
    declare_enable_arbiter = DeclareLaunchArgument(
        'enable_gpu_arbiter', default_value='true',
        description='Run gpu_arbiter (manages yolov8_node); disables standalone YOLO')
    # LD19 LiDAR + slam_toolbox mapping (includes slam.launch.py). Set false
    # when the LiDAR isn't connected (otherwise the lidar node errors on the
    # missing /dev/ttyUSB0). SLAM runs on CPU — no GPU contention with vision.
    declare_enable_slam = DeclareLaunchArgument(
        'enable_slam', default_value='true',
        description='Bring up LD19 LiDAR + slam_toolbox (needs LiDAR on /dev/ttyUSB0)')

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
        additional_env=dds_config,
    )

    # YOLO Detection Node — provides /detection/objects_simple to brain_node
    yolo_node = Node(
        package='yolov8_ros',
        executable='yolov8_node',
        name='detection_node',
        parameters=[{
            'model_path': LaunchConfiguration('yolo_model_path'),
            'enable_faces': LaunchConfiguration('enable_faces'),
            'enable_gestures': LaunchConfiguration('enable_gestures'),
            'enable_ocr': LaunchConfiguration('enable_ocr'),
            'use_depth': True,
            'show_window': False,
            'confidence_threshold': 0.4,
            'target_fps': 60,
        }],
        output='screen',
        emulate_tty=True,
        additional_env=dds_config,
    )

    # STT Node (existing package) — publishes /speech/text, consumed by brain_node
    stt_node = Node(
        package='stt_node',
        executable='stt_node',
        name='stt_node',
        condition=IfCondition(LaunchConfiguration('enable_stt')),
        parameters=[{
            'model_size': LaunchConfiguration('stt_model_size'),
            'sample_rate': 16000,
            'channels': 1,
            'silence_threshold': 0.015,
            'language': 'en',
        }],
        output='screen',
        emulate_tty=True,
        additional_env=dds_config,
    )

    # AI-SHA brain stack (brain_node + admin_node + action_node).
    # jetson_launch.py also kills stale AI nodes and syncs the KB into share/.
    aisha_brain = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('aisha_brain'), 'launch', 'jetson_launch.py',
        ])),
        launch_arguments={
            'llm_model': LaunchConfiguration('llm_model'),
            'llm_num_gpu': LaunchConfiguration('llm_num_gpu'),
        }.items(),
    )

    # LD19 LiDAR + slam_toolbox (LiDAR -> scan_raw -> throttle 2Hz -> scan ->
    # slam_toolbox; dummy_odom + static base_link->laser TF). Reuses the
    # existing slam.launch.py so there's one source of truth for the SLAM graph.
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('robot_bringup'), 'launch', 'slam.launch.py',
        ])),
        condition=IfCondition(LaunchConfiguration('enable_slam')),
    )

    # Vision selection (resolved at launch): either the gpu_arbiter (which
    # spawns/kills yolov8_node to time-multiplex the GPU) OR the standalone
    # YOLO node. They are mutually exclusive — running both would start two
    # detection_node instances competing for the camera + GPU.
    def _vision(context, *_args, **_kw):
        use_arbiter = context.perform_substitution(
            LaunchConfiguration('enable_gpu_arbiter')).strip().lower() == 'true'
        if not use_arbiter:
            return [yolo_node]
        # Build the command the arbiter uses to (re)spawn YOLO, reproducing the
        # standalone node's params. Node default name is 'detection_node', so
        # the pause service stays /detection_node/pause_inference.
        model_path = context.perform_substitution(
            LaunchConfiguration('yolo_model_path'))
        faces = context.perform_substitution(LaunchConfiguration('enable_faces'))
        gestures = context.perform_substitution(
            LaunchConfiguration('enable_gestures'))
        ocr = context.perform_substitution(LaunchConfiguration('enable_ocr'))
        yolo_cmd = [
            'ros2', 'run', 'yolov8_ros', 'yolov8_node', '--ros-args',
            '-p', f'model_path:={model_path}',
            '-p', f'enable_faces:={faces}',
            '-p', f'enable_gestures:={gestures}',
            '-p', f'enable_ocr:={ocr}',
            '-p', 'use_depth:=true',
            '-p', 'show_window:=false',
            '-p', 'confidence_threshold:=0.4',
            '-p', 'target_fps:=60',
        ]
        gpu_arbiter = Node(
            package='aisha_brain',
            executable='gpu_arbiter',
            name='gpu_arbiter',
            parameters=[{
                'manage_yolo': True,
                'yolo_cmd': yolo_cmd,
                'llm_model': context.perform_substitution(
                    LaunchConfiguration('llm_model')),
            }],
            output='screen',
            emulate_tty=True,
            additional_env=dds_config,
        )
        return [gpu_arbiter]

    vision = OpaqueFunction(function=_vision)

    startup_msg = LogInfo(msg=[
        '\n', '=' * 70, '\n',
        '  PROJECT CEREBRO — AI-SHA edition (Ollama brain)\n',
        '=' * 70, '\n',
        '  Vision: RealSense D435 + YOLO  |  STT -> /speech/text\n',
        '  Brain:  brain_node (rule-based router) + admin_node (RAG) + action_node\n',
        '  GPU:    gpu_arbiter time-multiplexes vision <-> LLM (ADR 0001)\n',
        '  SLAM:   LD19 LiDAR + slam_toolbox -> /map  (enable_slam:=false to skip)\n',
        '  Speech out: /robot_speech  (TTS must subscribe to this)\n',
        '  Requires: ollama serve + model llama3.2:1b\n',
        '=' * 70, '\n',
    ])

    return LaunchDescription([
        declare_stt_model,
        declare_yolo_model,
        declare_enable_faces,
        declare_enable_gestures,
        declare_enable_ocr,
        declare_enable_stt,
        declare_llm_model,
        declare_llm_num_gpu,
        declare_enable_arbiter,
        declare_enable_slam,
        startup_msg,
        realsense_node,
        stt_node,
        vision,
        aisha_brain,
        slam,
    ])
