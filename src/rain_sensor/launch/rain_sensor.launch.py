from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rain_sensor',
            executable='rain_sensor_node',
            name='rain_sensor_node',
            parameters=[{
                'serial_port': '/dev/ttyACM1',
                'baud_rate': 9600,
                'publish_rate': 1.0,
            }],
            output='screen',
        )
    ])
