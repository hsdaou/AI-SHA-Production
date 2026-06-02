#!/usr/bin/env python3
"""Launch file for Plant Health LLM Node"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    backend = LaunchConfiguration('backend')
    prompt_type = LaunchConfiguration('prompt_type')
    max_tokens = LaunchConfiguration('max_tokens')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'prompt_type',
            default_value='concise',
            description='Prompt type: "system" (full) or "concise"'
        ),

        DeclareLaunchArgument(
            'max_tokens',
            default_value='200',
            description='Maximum tokens in response'
        ),

        DeclareLaunchArgument(
            'backend',
            default_value='local',
            description='LLM backend: "local" (Llama) or "gemini" (cloud API)'
        ),

        # Local Llama LLM Node
        Node(
            package='llm_node',
            executable='llm_node',
            name='llm_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", backend, "' == 'local'"])
            ),
            parameters=[{
                'model_path': '/home/orin-robot/models/qwen2.5-1.5b-instruct-q4_k_m.gguf',
                'system_prompt_path': [
                    '/home/orin-robot/robot_ws/src/llm_node/prompts/sabis_robot_',
                    prompt_type,
                    '.txt'
                ],
                'n_ctx': 2048,
                'n_gpu_layers': -1,
                'temperature': 0.7,
                'max_tokens': max_tokens,
            }],
        ),

        # Gemini Cloud LLM Node
        Node(
            package='llm_node',
            executable='llm_node_gemini',
            name='llm_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", backend, "' == 'gemini'"])
            ),
            parameters=[{
                'system_prompt_path': [
                    '/home/orin-robot/robot_ws/src/llm_node/prompts/sabis_robot_',
                    prompt_type,
                    '.txt'
                ],
                'temperature': 0.4,
                'max_tokens': max_tokens,
            }],
        ),
    ])
