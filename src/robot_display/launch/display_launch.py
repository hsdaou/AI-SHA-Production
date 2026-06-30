from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch argument for eye side
    eye_side_arg = DeclareLaunchArgument(
        'eye_side',
        default_value='both',
        description='Which eye this display represents: left, right, or both'
    )

    # Create display node with parameter
    display_node = Node(
        package='robot_display',
        executable='display_node',
        name='robot_display',
        output='screen',
        parameters=[{
            'eye_side': LaunchConfiguration('eye_side')
        }]
    )

    return LaunchDescription([
        eye_side_arg,
        display_node,
    ])
