#!/usr/bin/env python3
"""Launch file for Plant Health LLM Node — DEPRECATED.

DEPRECATED: This is the legacy "Plant Health Robot" brain. It is NOT part of
the AI-SHA production stack and must NOT be run alongside it.

The current robot is launched via `robot_bringup cerebro_aisha.launch.py`,
where LLM inference is owned exclusively by `aisha_brain admin_node` (RAG over
the school knowledge base). This llm_node:
  * subscribes to /speech/text (same topic admin_node routes from), so running
    it concurrently triggers a SECOND local LLM inference per utterance —
    on an 8GB Jetson Orin Nano that can exceed unified memory and OOM-crash;
  * loads a large plant-disease system prompt (prompts/sabis_robot_*.txt) and
    subscribes to 8 agricultural sensor topics that no longer exist on the
    chassis — pure overhead and wrong-identity responses.

Kept only for rollback/reference. Do not add to any production launch.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    import sys
    print(
        '\n'
        '╔══════════════════════════════════════════════════════════════════╗\n'
        '║  DEPRECATED: llm_node/llm_sabis.launch.py (legacy Plant Health)   ║\n'
        '║  NOT for production. Do NOT run alongside cerebro_aisha.launch.py ║\n'
        '║  — concurrent local LLM inference can OOM-crash the Jetson.       ║\n'
        '║  Production brain = aisha_brain admin_node (via cerebro_aisha).   ║\n'
        '╚══════════════════════════════════════════════════════════════════╝\n',
        file=sys.stderr,
    )

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
