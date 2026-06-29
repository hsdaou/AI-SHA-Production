from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bmp180_pressure',
            executable='bmp180_node',
            name='bmp180_node',
            output='screen',
            parameters=[{
                'i2c_bus': 1,
                'publish_rate': 1.0,    # Hz — increase up to ~10 Hz if needed
                # Sharjah, UAE is at sea level — set to local QNH (Pa) for precision
                # 1 hPa = 100 Pa  e.g. QNH 1013 hPa → 101300.0
                'sea_level_pressure': 101400.0,  # QNH 1014 hPa, Sharjah UAE
            }]
        )
    ])
