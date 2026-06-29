from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='soil_moisture',
            executable='soil_moisture_node',
            name='soil_moisture_node',
            parameters=[{
                'serial_port': '/dev/ttyACM0',
                'baud_rate': 9600,
                'publish_rate': 1.0,
            }],
            output='screen',
        )
    ])
