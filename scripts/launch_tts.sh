#!/bin/bash
# Launch TTS Node for Jetson Integration
# This script starts the Text-to-Speech node that subscribes to /speech/text

echo "Starting TTS Node (Text-to-Speech)..."
echo "Subscribing to: /speech/text"
echo "Press Ctrl+C to stop"
echo ""

# Set environment variables
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export CYCLONEDDS_URI=file:///home/pi5/cyclonedds.xml

# Source ROS2 setup
source /opt/ros/jazzy/setup.bash
source /home/pi5/ros2_ws/install/setup.bash

# Run the TTS node
ros2 run tts_speaker tts_speaker_node
