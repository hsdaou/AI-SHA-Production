#!/usr/bin/env python3
"""
Complete BNO055 IMU Launch File - ALL NODES
Starts all necessary nodes for IMU operation with RViz visualization
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    # Get the package directory for RViz config
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rviz_config = os.path.join(pkg_dir, 'config', 'imu_viz.rviz')

    return LaunchDescription([
        # 1. BNO055 IMU Node - Main sensor driver
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

        # 2. IMU Bridge - Convert to standard sensor_msgs/Imu
        Node(
            package='bno055_imu',
            executable='imu_bridge',
            name='imu_bridge',
            output='screen'
        ),

        # 3. IMU Pose Publisher - Convert to PoseStamped for visualization
        Node(
            package='bno055_imu',
            executable='imu_pose_publisher',
            name='imu_pose_publisher',
            output='screen'
        ),

        # 4. Static TF Publisher - Publish map -> imu_link transform
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_to_map',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'imu_link'],
            output='screen'
        ),

        # 5. RViz2 - Visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
            output='screen'
        ),
    ])
