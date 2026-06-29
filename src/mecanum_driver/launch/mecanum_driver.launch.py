from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('mecanum_driver'),
        'config',
        'mecanum_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='mecanum_driver',
            executable='mecanum_driver',
            name='mecanum_driver',
            parameters=[config],
            output='screen',
            emulate_tty=True,
        ),
    ])
