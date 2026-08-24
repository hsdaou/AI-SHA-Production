#!/usr/bin/env python3
"""Offline Phase 8B replay; no serial device or motor command path exists."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="aisha_rev_d_driver",
                executable="rev_d_encoder_adapter",
                name="aisha_rev_d_encoder_adapter",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [FindPackageShare("aisha_rev_d_driver"), "config", "phase8b_replay.yaml"]
                    )
                ],
            )
        ]
    )
