#!/bin/bash
# AI-SHARJAH Robot Startup Script
# Launch all nodes in one clean command

set -e  # Exit on error

echo "======================================================================"
echo "  AI-SHARJAH - ISC Sharjah Educational Robot"
echo "======================================================================"

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash
source /home/pi5/ros2_ws/install/setup.bash

# Launch the robot
exec ros2 launch llm_node rpi5_robot_launcher.launch.py
