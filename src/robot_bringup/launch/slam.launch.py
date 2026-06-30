#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    bringup_dir = get_package_share_directory('robot_bringup')

    # Parameter file
    slam_params_file = os.path.join(bringup_dir, 'config', 'slam_toolbox.yaml')

    # LDROBOT LiDAR publisher node
    ldlidar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='LD19',
        output='screen',
        parameters=[
            {'product_name': 'LDLiDAR_LD19'},
            {'topic_name': 'scan_raw'},  # Publish to scan_raw, throttle to scan
            {'frame_id': 'laser'},
            {'port_name': '/dev/ttyUSB0'},
            {'port_baudrate': 230400},
            {'laser_scan_dir': True},
            {'enable_angle_crop_func': False},
            {'angle_crop_min': 135.0},
            {'angle_crop_max': 225.0}
        ]
    )

    # Dummy odometry publisher - publishes odom->base_link at high frequency
    # This fills TF buffer history for slam_toolbox
    dummy_odom_node = Node(
        package='robot_bringup',
        executable='dummy_odom.py',
        name='dummy_odom_publisher',
        output='screen'
    )

    # Scan throttle - reduce scan rate from 10Hz to 2Hz
    scan_throttle_node = Node(
        package='robot_bringup',
        executable='scan_throttle.py',
        name='scan_throttle',
        parameters=[{'rate': 2.0}],
        output='screen'
    )

    # base_link to laser tf node
    base_link_to_laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_laser',
        arguments=['--x', '0', '--y', '0', '--z', '0.18',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser']
    )

    # SLAM Toolbox node (async mode with throttled scans)
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file]
    )

    # Delay SLAM start to ensure TF publishers are ready and buffer is filled
    slam_toolbox_delayed = TimerAction(
        period=3.0,
        actions=[slam_toolbox_node]
    )

    # Define LaunchDescription
    ld = LaunchDescription()

    # Add all nodes
    ld.add_action(ldlidar_node)
    ld.add_action(scan_throttle_node)  # Throttle 10Hz -> 2Hz
    ld.add_action(dummy_odom_node)  # Dynamic TF publisher (50 Hz)
    ld.add_action(base_link_to_laser_tf_node)
    ld.add_action(slam_toolbox_delayed)  # Start SLAM after 3 seconds

    return ld
