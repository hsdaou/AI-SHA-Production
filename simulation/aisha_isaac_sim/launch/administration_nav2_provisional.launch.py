#!/usr/bin/env python3
"""Minimal Nav2 server graph for the AI-SHA Isaac administration mission."""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    map_file = LaunchConfiguration("map")
    maximum_linear_mps = LaunchConfiguration("maximum_linear_mps")
    amcl_tf_broadcast = LaunchConfiguration("amcl_tf_broadcast")
    common = [params_file, {"use_sim_time": True}]
    localization_parameters = [
        *common,
        {
            "tf_broadcast": ParameterValue(
                amcl_tf_broadcast, value_type=bool
            )
        },
    ]
    controller_parameters = [
        *common,
        {
            "FollowPath.max_vel_x": ParameterValue(
                maximum_linear_mps, value_type=float
            ),
            "FollowPath.max_speed_xy": ParameterValue(
                maximum_linear_mps, value_type=float
            ),
        },
    ]
    localization_nodes = ["map_server", "amcl"]
    navigation_nodes = ["planner_server", "controller_server"]
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(PACKAGE_ROOT / "config" / "nav2_sim_params.yaml"),
            ),
            DeclareLaunchArgument(
                "map",
                default_value=str(
                    PACKAGE_ROOT
                    / "maps"
                    / "administration_provisional"
                    / "administration_provisional.yaml"
                ),
            ),
            DeclareLaunchArgument("maximum_linear_mps", default_value="0.30"),
            DeclareLaunchArgument("amcl_tf_broadcast", default_value="true"),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[*common, {"yaml_filename": map_file}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=localization_parameters,
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=common,
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=controller_parameters,
                remappings=[("cmd_vel", "/cmd_vel")],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_localization",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "autostart": True,
                        "bond_timeout": 8.0,
                        "node_names": localization_nodes,
                    }
                ],
            ),
            # Planner and controller costmaps require map -> odom.  Delay their
            # lifecycle manager so the mission client can deliver /initialpose
            # after AMCL is active instead of racing a costmap activation timeout.
            TimerAction(
                period=20.0,
                actions=[
                    Node(
                        package="nav2_lifecycle_manager",
                        executable="lifecycle_manager",
                        name="lifecycle_manager_navigation",
                        output="screen",
                        parameters=[
                            {
                                "use_sim_time": True,
                                "autostart": True,
                                "bond_timeout": 8.0,
                                "node_names": navigation_nodes,
                            }
                        ],
                    )
                ],
            ),
        ]
    )
