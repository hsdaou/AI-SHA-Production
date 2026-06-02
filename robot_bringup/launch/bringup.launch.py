from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
import os
import yaml


def generate_launch_description():

    # Robot URDF path
    robot_description_path = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'robot.urdf'
    )

    # Load LiDAR parameters from YAML
    ld19_config_file = os.path.join(
        get_package_share_directory('robot_bringup'),
        'config',
        'ld19.yaml'
    )

    with open(ld19_config_file, 'r') as f:
        ld19_params = yaml.safe_load(f)['ld19_config']

    return LaunchDescription([

        # Robot State Publisher (publishes TF from URDF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command(['cat ', robot_description_path])
            }]
        ),

        # LD-19 LiDAR Node
        Node(
            package='ldlidar_stl_ros2',
            executable='ldlidar_stl_ros2_node',
            name='lidar',
            output='screen',
            parameters=[ld19_params]
        ),

        # RealSense D435 Camera Node
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            output='screen',
            parameters=[{
                'enable_depth': True,
                'enable_color': True,
                'pointcloud.enable': False
            }]
        ),
    ])
