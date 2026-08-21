#!/bin/bash
# AI-SHA IMU bring-up: BNO055 driver + sensor_msgs/Imu bridge.
# Mirrors slam_bringup.sh - starts the children then waits, so systemd owns them.
#
# imu_complete.launch.py is deliberately NOT used: it also starts a static
# map -> imu_link transform, which fights slam_toolbox for the TF tree.
source /opt/ros/jazzy/setup.bash
source /home/pi5/ros2_ws/install/setup.bash

ros2 run bno055_imu bno055_node &
ros2 run bno055_imu imu_bridge &

# Exit as soon as either child dies so systemd Restart=always brings both back.
wait -n
