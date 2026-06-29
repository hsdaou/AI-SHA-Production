#!/usr/bin/env python3
"""
Launch file for BNO055 IMU with RViz2 visualization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os


def generate_launch_description():
    # Get the package directory
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rviz_config = os.path.join(pkg_dir, 'config', 'imu_viz.rviz')

    return LaunchDescription([
        # BNO055 IMU Node
        Node(
            package='bno055_imu',
            executable='bno055_node',
            name='bno055_imu_node',
            output='screen',
            parameters=[{
                'publish_rate': 50.0,
                'frame_id': 'imu_link',
                'i2c_address': 0x28,
            }]
        ),

        # Bridge to convert to sensor_msgs/Imu
        Node(
            package='bno055_imu',
            executable='imu_bridge',
            name='imu_bridge',
            output='screen'
        ),

        # Static transform publisher (imu_link to map)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_to_map',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'imu_link']
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
            output='screen'
        ),
    ])
