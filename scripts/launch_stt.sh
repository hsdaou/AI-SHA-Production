#!/bin/bash
# Launch STT Node for Jetson Integration
# This script starts the Speech-to-Text node that publishes to /speech_rec

echo "Starting STT Node (Speech Recognition)..."
echo "Publishing to: /speech_rec"
echo "Press Ctrl+C to stop"
echo ""

# Set environment variables
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export CYCLONEDDS_URI=file:///home/pi5/cyclonedds.xml

# Source ROS2 setup
source /opt/ros/jazzy/setup.bash
source /home/pi5/ros2_ws/install/setup.bash

# Run the STT node
ros2 run speech_recognition stt_node
