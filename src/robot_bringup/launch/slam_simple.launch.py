#!/usr/bin/env python3
"""Simplified SLAM launch - using built-in ROS tools only"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bringup_dir = get_package_share_directory('robot_bringup')
    slam_params = os.path.join(bringup_dir, 'config', 'slam_toolbox_simple.yaml')

    # LiDAR
    lidar = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='LD19',
        output='screen',
        parameters=[{
            'product_name': 'LDLiDAR_LD19',
            'topic_name': 'scan',
            'frame_id': 'laser',
            'port_name': '/dev/ttyUSB0',
            'port_baudrate': 230400,
            'laser_scan_dir': True,
            'enable_angle_crop_func': False,
        }]
    )

    # TF: base_link -> laser
    tf_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0.18',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser']
    )

    # Throttle scans using built-in ROS tool
    throttle = ExecuteProcess(
        cmd=['ros2', 'topic', 'hz', '/scan', '--window', '1'],
        output='screen',
        # This doesn't actually throttle, need different approach
    )

    # SLAM with simpler config
    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params]
    )

    slam_delayed = TimerAction(period=5.0, actions=[slam])

    ld = LaunchDescription()
    ld.add_action(lidar)
    ld.add_action(tf_laser)
    ld.add_action(slam_delayed)

    return ld
