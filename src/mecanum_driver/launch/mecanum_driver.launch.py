# ─────────────────────────────────────────────────────────────────────────────
# Mecanum Driver Launch — runs on the Raspberry Pi 5
#
# In the Two-Tier SBC + MCU layout the Arduino Mega owns both the encoders
# and the BNO055 IMU and streams them inside the unified ODOM telemetry
# packet.  This launch file is the minimal node-only entry point — for the
# full Pi 5 stack (LiDAR + SLAM + mecanum_driver + TTS + display) use
# `ros2 launch aisha_integration rpi_launch.py` instead.
#
# To enable encoder-based odometry publishing, set `publish_odom: true`
# in mecanum_params.yaml.
# ─────────────────────────────────────────────────────────────────────────────
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('mecanum_driver'),
        'config',
        'mecanum_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='mecanum_driver',
            executable='mecanum_driver',
            name='mecanum_driver',
            parameters=[config],
            output='screen',
            emulate_tty=True,
        ),
    ])
