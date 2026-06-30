#!/bin/bash
# Launcher script for Robot Display
# Sources ROS2 and starts the display GUI

# Source ROS2
source /opt/ros/humble/setup.bash

# ROS2 network config - connect to Jetson at 10.42.0.33
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/pi4/fastdds.xml

# Source workspace if it exists
if [ -f /home/pi4/ros2_ws/install/setup.bash ]; then
    source /home/pi4/ros2_ws/install/setup.bash
fi

# Set display (detect automatically)
export DISPLAY=${DISPLAY:-:1}

# Run the display
cd /home/pi4
python3 /home/pi4/robot_display.py
