#!/usr/bin/env python3
"""Minimal Nav2 server graph for the AI-SHA Isaac administration mission."""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    map_file = LaunchConfiguration("map")
    common = [params_file, {"use_sim_time": True}]
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
                parameters=common,
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
                parameters=common,
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
