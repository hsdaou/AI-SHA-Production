"""AI-SHA Raspberry Pi 5 Launch — Spatial + Interface + Motor-translation tier.

The Pi 5 owns everything that is NOT GPU-bound in the Two-Tier SBC + MCU
architecture:

    Sensors / Perception:
    - ldlidar_stl_ros2     : LD-19 2D LiDAR driver (/dev/ttyUSB0)
    NOTE: the BNO055 IMU is now read by the Arduino Mega over I2C and
    forwarded inside the unified ODOM serial packet; mecanum_driver_node
    publishes the IMU half on /imu/data. No bno055_imu node runs on the
    Pi 5 anymore — that path was prone to I2C clock-stretching stalls.

    Navigation / SLAM (moved here from the Jetson so the GPU board stays
    AI-only):
    - rf2o_laser_odometry  : odom from /scan wall shifts (when odom_source=laser)
    - dummy_odom           : identity odom (when odom_source=dummy)
    - slam_toolbox         : online async SLAM

    Motor translation (absorbed from the retired Pi 4):
    - mecanum_driver_node  : USB serial to Arduino Mega, parses ODOM,
                              publishes /odom + /imu/data

    Audio / Display:
    - tts_node     : Piper TTS (local, offline)
    - llm_display  : Animated face display (PyQt5)

Prerequisites:
    export FASTRTPS_DEFAULT_PROFILES_FILE=~/config/fastdds_rpi.xml
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=42   # MUST match the Jetson

    All SBCs MUST run ROS 2 Humble.  See README "FastDDS cross-distro"
    warning — Nav2 action goals silently fail across Humble↔Jazzy meshes.

Usage:
    ros2 launch aisha_integration rpi_launch.py
    ros2 launch aisha_integration rpi_launch.py enable_display:=false
    ros2 launch aisha_integration rpi_launch.py lidar_port:=/dev/ttyUSB1
    ros2 launch aisha_integration rpi_launch.py enable_slam:=false
    ros2 launch aisha_integration rpi_launch.py odom_source:=dummy
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── ROS_DOMAIN_ID consistency check ──────────────────────────────────────
    # All SBCs MUST use the same ROS_DOMAIN_ID.  See jetson_launch.py for details.
    domain_id = os.environ.get('ROS_DOMAIN_ID', '')
    if not domain_id:
        print(
            '\n'
            '╔═══════════════════════════════════════════════════════════════════╗\n'
            '║  WARNING: ROS_DOMAIN_ID is NOT set!                              ║\n'
            '║  Set it on ALL SBCs before launching:                            ║\n'
            '║    export ROS_DOMAIN_ID=42                                       ║\n'
            '║  Must match the Jetson or topics will be partitioned.            ║\n'
            '╚═══════════════════════════════════════════════════════════════════╝\n',
        )
    else:
        print(f'[rpi_launch] ROS_DOMAIN_ID={domain_id}')

    # ── FastDDS discovery safety check ────────────────────────────────────────
    if not os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE'):
        print(
            '\n'
            '╔═══════════════════════════════════════════════════════════════════╗\n'
            '║  WARNING: FASTRTPS_DEFAULT_PROFILES_FILE is NOT set!             ║\n'
            '║  ROS 2 will fall back to multicast — nodes on other devices      ║\n'
            '║  will be invisible.  Set it before launching:                    ║\n'
            '║    export FASTRTPS_DEFAULT_PROFILES_FILE=~/config/fastdds_rpi.xml║\n'
            '║    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp                    ║\n'
            '╚═══════════════════════════════════════════════════════════════════╝\n',
        )

    args = [
        # ── Sensor args ──────────────────────────────────────────────────
        DeclareLaunchArgument('lidar_port',          default_value='/dev/ttyUSB0'),
        # ── Motor / Arduino args ─────────────────────────────────────────
        # Persistent udev symlink — see src/mecanum_driver/scripts/arduino_mega.rules.
        DeclareLaunchArgument('arduino_serial_port', default_value='/dev/aisha_arduino'),
        DeclareLaunchArgument('arduino_baud_rate',   default_value='115200'),
        # ── Navigation args ──────────────────────────────────────────────
        DeclareLaunchArgument('enable_slam',         default_value='true'),
        # Odometry source for SLAM.  Options:
        #   'laser'  — rf2o_laser_odometry: estimates odom from /scan wall
        #              shifts.  Works without encoders but drifts in featureless
        #              corridors and fails in open spaces with no walls.
        #   'dummy'  — identity TF (odom=base_link at all times).  SLAM relies
        #              entirely on scan-matching, which works for slow mapping
        #              but shears the map under fast Mecanum movement.
        #   'encoder' — encoder-based odom from mecanum_driver_node, which
        #               owns the odom→base_link TF directly (publish_odom and
        #               publish_odom_tf both forced true).  Best raw accuracy
        #               when the wheel-radius / track-width calibration is
        #               trusted, but no IMU heading correction.
        #   'ekf'     — robot_localization EKF fuses encoder /odom + IMU
        #               /imu/data into a smooth odom→base_link (config:
        #               robot_bringup/config/ekf.yaml).  The driver publishes
        #               /odom but yields TF ownership to the EKF
        #               (publish_odom=true, publish_odom_tf=false, set below).
        #               Requires: sudo apt install ros-humble-robot-localization
        DeclareLaunchArgument('odom_source',         default_value='laser'),
        # ── Audio / display args ─────────────────────────────────────────
        DeclareLaunchArgument('enable_display',      default_value='true'),
        DeclareLaunchArgument('piper_binary',        default_value=os.path.expanduser('~/piper/piper')),
        DeclareLaunchArgument('piper_voice',         default_value=os.path.expanduser('~/piper/voices/en_US-amy-low.onnx')),
        # NOTE: USB audio device indexes can shift on reboot (I2S HAT, USB mic).
        # Verify with: arecord -l / aplay -l.
        # STRONGLY RECOMMENDED: Create a persistent ALSA alias in /etc/asound.conf
        # based on hardware ID (not card index) to prevent tts_node crashes:
        #   pcm.aisha_speaker { type hw; card "Device"; }  # match aplay -l name
        # Then set audio_playback_dev:=aisha_speaker instead of plughw:0,0.
        DeclareLaunchArgument('audio_playback_dev',  default_value='plughw:0,0'),
    ]

    def create_nodes(context):
        from launch_ros.actions import Node

        def arg(name):
            return LaunchConfiguration(name).perform(context)

        # Odometry source decides who owns the odom→base_link TF, which in
        # turn decides how mecanum_driver_node is configured (below).  Compute
        # it once here so both the driver params and the SLAM block agree.
        odom_source = arg('odom_source')
        if odom_source == 'ekf':
            # EKF owns the TF: driver emits /odom but not the transform.
            mecanum_odom_overrides = {'publish_odom': True, 'publish_odom_tf': False}
        elif odom_source == 'encoder':
            # Driver owns the TF directly.
            mecanum_odom_overrides = {'publish_odom': True, 'publish_odom_tf': True}
        else:
            # laser/dummy: another node owns odom; leave the YAML defaults
            # (publish_odom: false) untouched.
            mecanum_odom_overrides = {}

        nodes = []
        event_handlers = []

        def _exit_logger(target, label):
            return RegisterEventHandler(
                OnProcessExit(
                    target_action=target,
                    on_exit=[LogInfo(msg=f'[rpi_launch] Node {label} has exited. '
                                     'Check logs for errors.')],
                ),
            )

        # ── LiDAR Driver (LD-19, physically connected to Pi 5) ───────────
        # QoS NOTE: ldlidar_stl_ros2 publishes /scan with SensorDataQoS
        # (Best Effort, Volatile).  slam_toolbox now lives on the same Pi
        # so cross-host QoS mismatches are no longer a concern, but if you
        # ever remote-subscribe to /scan check `ros2 topic info /scan --verbose`.
        lidar_node = Node(
            package='ldlidar_stl_ros2',
            executable='ldlidar_stl_ros2_node',
            name='ldlidar',
            output='screen',
            parameters=[{
                'product_name': 'LDLiDAR_LD19',
                'topic_name': 'scan',
                'port_name': arg('lidar_port'),
                'port_baudrate': 230400,
                'laser_scan_dir': True,
                'enable_angle_crop_func': False,
                'frame_id': 'base_laser',
            }],
        )
        nodes.append(lidar_node)
        event_handlers.append(_exit_logger(lidar_node, 'ldlidar'))

        # ── Mecanum Driver Node (USB serial to Arduino Mega 2560) ─────────
        # Absorbed from the retired Pi 4 motor SBC.  Receives /cmd_vel,
        # forwards M-frames to the Arduino, and parses ODOM telemetry into
        # /imu/data + /odom on the Pi 5 side.
        mecanum_node = Node(
            package='mecanum_driver',
            executable='mecanum_driver',
            name='mecanum_driver',
            output='screen',
            emulate_tty=True,
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('mecanum_driver'),
                    'config', 'mecanum_params.yaml',
                ]),
                # Allow operators to override the serial port from the
                # launch file without re-installing the YAML — keeps the
                # bench/robot configs interchangeable.
                {
                    'serial_port': arg('arduino_serial_port'),
                    'baud_rate': int(arg('arduino_baud_rate')),
                    # Encoder-odom / TF ownership depends on odom_source.
                    **mecanum_odom_overrides,
                },
            ],
        )
        nodes.append(mecanum_node)
        event_handlers.append(_exit_logger(mecanum_node, 'mecanum_driver'))

        # ── Static TF: base_link → imu_link ───────────────────────────────
        # The BNO055 sits on the Arduino-side I2C bus; mecanum_driver_node
        # stamps the IMU messages with frame_id=imu_link by default.  The
        # URDF on the Jetson may also define this joint — if so, remove
        # this publisher to avoid TF chatter (identical static transforms
        # are tolerated by TF but waste broadcast bandwidth).
        imu_tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_static_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'],
            output='screen',
        )
        nodes.append(imu_tf_node)

        # ── Static TF: base_link → base_laser ─────────────────────────────
        # SLAM + Nav2 need this to transform /scan into the robot frame.
        # If robot_state_publisher on the Jetson loads a complete URDF
        # successfully, both will publish the same static transform — TF
        # handles matching values gracefully.
        lidar_tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_static_tf',
            arguments=['0', '0', '0.2', '0', '0', '0', 'base_link', 'base_laser'],
            output='screen',
        )
        nodes.append(lidar_tf_node)

        # ── SLAM Toolbox + odometry fallback ─────────────────────────────
        # Both relocated here from the Jetson so the GPU board stays
        # AI-only and so SLAM consumes /scan from a local DDS endpoint
        # rather than crossing the network.
        if arg('enable_slam') == 'true':
            if odom_source == 'laser':
                # rf2o_laser_odometry: estimates odom→base_link from
                # consecutive /scan frames (wall-shift matching).  Much
                # more accurate than dummy_odom for SLAM — the scan
                # matcher no longer fights a "robot is stationary" prior
                # while LiDAR sees walls moving.
                # Install: sudo apt install ros-humble-rf2o-laser-odometry
                rf2o_node = Node(
                    package='rf2o_laser_odometry',
                    executable='rf2o_laser_odometry_node',
                    name='rf2o_laser_odometry',
                    output='screen',
                    parameters=[{
                        'laser_scan_topic': '/scan',
                        'odom_topic': '/odom_rf2o',
                        'publish_tf': True,
                        'base_frame_id': 'base_link',
                        'odom_frame_id': 'odom',
                        'freq': 20.0,
                    }],
                )
                nodes.append(rf2o_node)
                event_handlers.append(_exit_logger(rf2o_node, 'rf2o_laser_odometry'))
            elif odom_source == 'dummy':
                # Identity odom→base_link at 50 Hz so slam_toolbox always
                # has a TF to operate on.  Shears the map under fast moves.
                dummy_node = Node(
                    package='robot_bringup',
                    executable='dummy_odom',
                    name='dummy_odom_publisher',
                    output='screen',
                )
                nodes.append(dummy_node)
                event_handlers.append(_exit_logger(dummy_node, 'dummy_odom_publisher'))
            elif odom_source == 'ekf':
                # robot_localization EKF fuses encoder /odom + IMU /imu/data
                # into a smooth, drift-corrected odom→base_link at 50 Hz.
                # The driver was set to publish_odom=true / publish_odom_tf=false
                # above, so the EKF is the sole odom→base_link broadcaster.
                # Install: sudo apt install ros-humble-robot-localization
                ekf_node = Node(
                    package='robot_localization',
                    executable='ekf_node',
                    name='ekf_filter_node',
                    output='screen',
                    parameters=[
                        PathJoinSubstitution([
                            FindPackageShare('robot_bringup'),
                            'config', 'ekf.yaml',
                        ])
                    ],
                )
                nodes.append(ekf_node)
                event_handlers.append(_exit_logger(ekf_node, 'ekf_filter_node'))
            # When odom_source=='encoder' the mecanum_driver_node itself
            # owns odom→base_link (publish_odom_tf=true, set above), so no
            # separate odometry node is needed.

            slam_node = Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    PathJoinSubstitution([
                        FindPackageShare('robot_bringup'),
                        'config', 'slam_toolbox.yaml',
                    ])
                ],
                remappings=[('scan', '/scan')],
            )
            nodes.append(slam_node)
            event_handlers.append(_exit_logger(slam_node, 'slam_toolbox'))

        # ── TTS Node (Piper, fully local) ─────────────────────────────────
        tts_node = Node(
            package='aisha_brain',
            executable='tts_node',
            name='ai_sha_tts',
            output='screen',
            parameters=[{
                'piper_binary': arg('piper_binary'),
                'voice_model': arg('piper_voice'),
                'audio_device': arg('audio_playback_dev'),
            }],
        )
        nodes.append(tts_node)
        event_handlers.append(_exit_logger(tts_node, 'ai_sha_tts'))

        # ── Animated Face Display ─────────────────────────────────────────
        if arg('enable_display') == 'true':
            display_node = Node(
                package='llm_display',
                executable='llm_display_node',
                name='ai_sha_display',
                output='screen',
            )
            nodes.append(display_node)
            event_handlers.append(_exit_logger(display_node, 'ai_sha_display'))

        return nodes + event_handlers

    return LaunchDescription(args + [OpaqueFunction(function=create_nodes)])
