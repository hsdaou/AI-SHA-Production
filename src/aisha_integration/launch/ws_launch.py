"""AI-SHA Workstation Launch — Dev / simulation / WhatsApp relay.

Runs on the RTX 5080 Linux workstation:
    - whatsapp_listener  : WhatsApp message relay node
    - Isaac Sim bridge   : (optional) ROS2 topic bridge for simulation

Prerequisites:
    export FASTRTPS_DEFAULT_PROFILES_FILE=~/aisha-integration/config/fastdds_ws.xml
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    npx mudslide login   (WhatsApp auth, once)

Usage:
    ros2 launch aisha_integration ws_launch.py
    ros2 launch aisha_integration ws_launch.py wa_number:=+971XXXXXXXXX
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    args = [
        DeclareLaunchArgument('wa_number',   default_value=''),
        DeclareLaunchArgument('wa_jid',      default_value=''),
    ]

    def create_nodes(context):
        from launch_ros.actions import Node

        def arg(name):
            return LaunchConfiguration(name).perform(context)

        wa_node = Node(
            package='aisha_brain',
            executable='whatsapp_listener',
            name='ai_sha_whatsapp',
            output='screen',
            parameters=[{
                'allowed_number': arg('wa_number'),
                'monitored_jid': arg('wa_jid'),
            }],
        )

        exit_handler = RegisterEventHandler(
            OnProcessExit(on_exit=[
                LogInfo(msg='[ws_launch] Node ai_sha_whatsapp has exited. '
                        'Check logs for errors.'),
            ]),
        )

        return [wa_node, exit_handler]

    return LaunchDescription(args + [OpaqueFunction(function=create_nodes)])
