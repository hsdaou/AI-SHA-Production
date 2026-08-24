#!/usr/bin/env python3
"""Stationary-only Rev D localization preflight; this launch cannot command motion."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    map_file = LaunchConfiguration("map")
    amcl_params = LaunchConfiguration("amcl_params")
    ekf_params = LaunchConfiguration("ekf_params")
    robot_urdf = LaunchConfiguration("robot_urdf")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="",
                description=(
                    "Required occupancy-map YAML. The Phase 8A presentation map is "
                    "stationary-observation only and never authorizes motion."
                ),
            ),
            DeclareLaunchArgument(
                "amcl_params",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("robot_bringup"), "config", "amcl_rev_d_preflight.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "ekf_params",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("robot_bringup"), "config", "ekf_rev_d.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "robot_urdf",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("robot_description"),
                        "urdf",
                        "aisha_rev_d_localization.urdf",
                    ]
                ),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="aisha_rev_d_state_publisher",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": False,
                        "robot_description": ParameterValue(
                            Command(["cat ", robot_urdf]), value_type=str
                        ),
                    }
                ],
            ),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[ekf_params],
                remappings=[("odometry/filtered", "/odometry/filtered")],
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[amcl_params, {"yaml_filename": map_file, "use_sim_time": False}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=[amcl_params, {"use_sim_time": False}],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_phase8a_localization",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": False,
                        "autostart": True,
                        "bond_timeout": 8.0,
                        "node_names": ["map_server", "amcl"],
                    }
                ],
            ),
        ]
    )
