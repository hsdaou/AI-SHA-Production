#!/usr/bin/env python3
"""
Launch file for STT node with configurable parameters
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'sample_rate',
            default_value='16000',
            description='Audio sample rate in Hz'
        ),
        DeclareLaunchArgument(
            'chunk_duration',
            default_value='3.0',
            description='Audio chunk duration in seconds'
        ),
        DeclareLaunchArgument(
            'silence_threshold',
            default_value='0.01',
            description='Voice activity detection threshold'
        ),
        DeclareLaunchArgument(
            'device_index',
            default_value='None',
            description='Audio device index (None for auto-detect)'
        ),
        DeclareLaunchArgument(
            'model_name',
            default_value='nvidia/canary-1b-v2',
            description='HuggingFace model ID'
        ),

        # STT node
        Node(
            package='stt_node',
            executable='stt_node',
            name='stt_node',
            output='screen',
            parameters=[{
                'sample_rate': LaunchConfiguration('sample_rate'),
                'chunk_duration': LaunchConfiguration('chunk_duration'),
                'silence_threshold': LaunchConfiguration('silence_threshold'),
                'device_index': LaunchConfiguration('device_index'),
                'model_name': LaunchConfiguration('model_name'),
            }],
            remappings=[
                # Uncomment to remap to LLM node topic
                # ('/speech_rec', '/speech/text'),
            ]
        ),
    ])
