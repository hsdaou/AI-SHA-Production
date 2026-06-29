from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    port_arg = DeclareLaunchArgument(
        'port', default_value='/dev/ttyAMA0',
        description='Serial port of the GT-U7 GPS module'
    )
    baud_arg = DeclareLaunchArgument(
        'baud_rate', default_value='9600',
        description='Baud rate (default 9600 for GT-U7)'
    )
    frame_arg = DeclareLaunchArgument(
        'frame_id', default_value='gps_link',
        description='TF frame id for GPS messages'
    )

    params_file = PathJoinSubstitution([
        FindPackageShare('gps_gt_u7'), 'config', 'gps_gt_u7.yaml'
    ])

    gps_node = Node(
        package='gps_gt_u7',
        executable='gps_node',
        name='gps_gt_u7_node',
        output='screen',
        parameters=[
            params_file,
            {
                'port':      LaunchConfiguration('port'),
                'baud_rate': LaunchConfiguration('baud_rate'),
                'frame_id':  LaunchConfiguration('frame_id'),
            },
        ],
    )

    return LaunchDescription([
        port_arg,
        baud_arg,
        frame_arg,
        gps_node,
    ])
