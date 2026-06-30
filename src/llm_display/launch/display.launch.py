from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='llm_display',
            executable='llm_display',
            name='llm_display_node',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        ),
    ])
