"""AI-SHA Jetson Launch — Cognitive + Perception (GPU-only) tier.

In the Two-Tier SBC + MCU layout the Jetson runs *only* the GPU-bound
nodes.  SLAM, Nav2, the motor driver, and the BNO055/LiDAR sensor
drivers all live on the Raspberry Pi 5 (rpi_launch.py); the Arduino
Mega 2560 handles real-time motor PWM and IMU acquisition.

Nodes started here:

  Perception:
    - realsense2_camera  : D435 driver with aligned depth (enable_camera)
    - yolov8_ros         : object + face + gesture + OCR (TensorRT)
    - gpu_arbiter        : time-multiplexes the GPU between vision and the LLM
                           (ADR 0001); OWNS yolov8_node when enable_gpu_arbiter
                           is true (spawns/kills it). Mutually exclusive with
                           the standalone YOLO node.

  TF tree:
    - robot_state_publisher + URDF (xacro)
    NOTE: LiDAR (/scan) and IMU (/imu/data) are published by nodes on
    the Pi 5 and reach the Jetson over FastDDS unicast.

  Cognitive / Administrative (aisha_brain):
    - stt_node           : Faster-Whisper STT (CUDA, fully local)
    - brain_node         : Intent router (ADMIN / NAV / ACTION)
    - admin_node         : RAG knowledge base (Ollama + ChromaDB)
    - action_node        : WhatsApp integration

Cloud nodes intentionally excluded (privacy / offline requirement):
    - llm_node_gemini    (Gemini API — deprecated)
    - stt_assemblyai     (AssemblyAI — deprecated)
    - tts_elevenlabs     (ElevenLabs API — deprecated)

Prerequisites on Jetson:
    ollama serve         (run as systemd service or in separate terminal)
    export FASTRTPS_DEFAULT_PROFILES_FILE=~/config/fastdds_jetson.xml
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=42   # MUST match the Pi 5

Usage:
    ros2 launch aisha_integration jetson_launch.py
    ros2 launch aisha_integration jetson_launch.py llm_model:=llama3.2:1b
    ros2 launch aisha_integration jetson_launch.py wake_word_enabled:=false
    # Vision-only bring-up (always-on YOLO, no arbiter, external camera):
    ros2 launch aisha_integration jetson_launch.py \\
        enable_stt:=false enable_gpu_arbiter:=false enable_camera:=false
"""

import os
import signal
import subprocess
import urllib.request
import urllib.error

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# ── Deprecated cloud node names — killed on startup if somehow running ────────
_DEPRECATED_NODES = [
    'llm_node_gemini',
    'stt_assemblyai',
    'tts_elevenlabs_node',
    'robot_brain',
]


def _kill_deprecated_nodes():
    """Ensure no deprecated cloud-dependent nodes are running.

    Uses exact executable name matching to avoid killing unrelated processes.
    """
    my_pid = os.getpid()
    for node_name in _DEPRECATED_NODES:
        try:
            result = subprocess.run(
                ['pgrep', '-x', '-f', node_name],
                capture_output=True, text=True,
            )
        except FileNotFoundError:
            break
        for line in result.stdout.strip().splitlines():
            pid_str = line.strip().split()[0] if line.strip() else ''
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == my_pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                print(f'[jetson_launch] WARNING: Killed deprecated node '
                      f'(pid={pid}, pattern={node_name}). '
                      f'Cloud nodes must not run — offline/privacy requirement.')
            except ProcessLookupError:
                pass


def _check_ollama(url='http://127.0.0.1:11434'):
    """Check if Ollama is reachable. Returns True if responsive."""
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _load_robot_description_xacro():
    """Load robot URDF via xacro from robot_description package."""
    try:
        from ament_index_python.packages import get_package_share_directory
        import xacro
        pkg_share = get_package_share_directory('robot_description')
        # Try common xacro file locations
        for candidate in ['urdf/robot.urdf.xacro', 'urdf/robot.xacro',
                          'robot.urdf.xacro', 'robot.urdf']:
            xacro_path = os.path.join(pkg_share, candidate)
            if os.path.isfile(xacro_path):
                doc = xacro.process_file(xacro_path)
                return doc.toxml()
        print('[jetson_launch] WARNING: No URDF xacro file found in '
              f'robot_description package at {pkg_share}. '
              'robot_state_publisher will start with an empty description.')
    except Exception as e:
        print(f'[jetson_launch] WARNING: Could not load URDF via xacro: {e}. '
              'robot_state_publisher will start with an empty description.')
    return ''


def _on_node_exit(node_obj, node_name):
    """Create an event handler that logs when a specific node exits.

    Uses target_action to scope the handler to the exact node,
    preventing false 'exited' messages when an unrelated process stops.
    """
    return RegisterEventHandler(
        OnProcessExit(
            target_action=node_obj,
            on_exit=[
                LogInfo(msg=f'[jetson_launch] Node {node_name} has exited. '
                        'Check logs for errors.'),
            ],
        ),
    )


def generate_launch_description():
    # ── ROS_DOMAIN_ID consistency check ──────────────────────────────────────
    # Jetson and Pi 5 MUST use the same ROS_DOMAIN_ID.  If they differ,
    # nodes will discover each other via FastDDS but topics will be silently
    # partitioned — /cmd_vel from Nav2 on the Pi 5 will never reach
    # mecanum_driver on the same Pi (different domain == different bus),
    # and the robot won't move.
    # Default: 42 (arbitrary, but must be identical on both SBCs).
    # Set in each device's ~/.bashrc or systemd unit:
    #   export ROS_DOMAIN_ID=42
    domain_id = os.environ.get('ROS_DOMAIN_ID', '')
    if not domain_id:
        print(
            '\n'
            '╔═══════════════════════════════════════════════════════════════════════╗\n'
            '║  WARNING: ROS_DOMAIN_ID is NOT set!                                  ║\n'
            '║  Defaulting to 0, which may conflict with other ROS 2 systems on     ║\n'
            '║  the same network.  Set it on every SBC before launching:            ║\n'
            '║    export ROS_DOMAIN_ID=42                                           ║\n'
            '║  Jetson and Pi 5 MUST use the SAME value.                            ║\n'
            '╚═══════════════════════════════════════════════════════════════════════╝\n',
        )
    else:
        print(f'[jetson_launch] ROS_DOMAIN_ID={domain_id}')

    # ── FastDDS discovery safety check ────────────────────────────────────────
    # If the FastDDS profile is not set, ROS 2 silently falls back to UDP
    # multicast discovery, which fails on most school/corporate networks.
    # Warn loudly so operators know why nodes are invisible across devices.
    if not os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE'):
        print(
            '\n'
            '╔═══════════════════════════════════════════════════════════════════════╗\n'
            '║  WARNING: FASTRTPS_DEFAULT_PROFILES_FILE is NOT set!                 ║\n'
            '║  ROS 2 will fall back to multicast discovery, which may fail on      ║\n'
            '║  your network.  Set it before launching:                             ║\n'
            '║    export FASTRTPS_DEFAULT_PROFILES_FILE=~/config/fastdds_jetson.xml ║\n'
            '║    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp                        ║\n'
            '╚═══════════════════════════════════════════════════════════════════════╝\n',
        )

    _kill_deprecated_nodes()

    # ── Launch arguments ──────────────────────────────────────────────────────
    # SLAM, Nav2, and the odometry-source switch are NOT declared here:
    # the spatial tier now lives on the Pi 5 (rpi_launch.py).
    args = [
        DeclareLaunchArgument('enable_vision',    default_value='true'),
        DeclareLaunchArgument(
            'enable_camera', default_value='true',
            description='Start the RealSense D435 driver (with aligned depth) that '
                        'yolov8_node consumes. Set false if a camera is launched '
                        'externally (e.g. ros2 launch realsense2_camera rs_launch.py).',
        ),
        DeclareLaunchArgument(
            'enable_gpu_arbiter', default_value='true',
            description='Run gpu_arbiter, which time-multiplexes the GPU between '
                        'vision (NAVIGATING) and the LLM (CONVERSING) per ADR 0001. '
                        'When true the arbiter OWNS yolov8_node (spawns/kills it) and '
                        'the standalone YOLO node is NOT started. Set false for '
                        'always-on vision without GPU multiplexing.',
        ),
        DeclareLaunchArgument('enable_stt',       default_value='true'),
        DeclareLaunchArgument('llm_model',        default_value='llama3.2'),
        DeclareLaunchArgument('whisper_model',    default_value='base'),
        DeclareLaunchArgument('whisper_device',   default_value='cuda'),
        # int8 saves ~500 MB VRAM vs float16 with negligible accuracy loss
        # on the Whisper base model.  On an 8 GB Jetson where VRAM is shared
        # with the CPU, this headroom prevents OOM kills during simultaneous
        # STT + YOLOv8 + Ollama inference.  Use float16 only if you notice
        # transcription quality degradation on accented English.
        DeclareLaunchArgument('whisper_compute',  default_value='int8'),
        # Audio input device for STT (Faster-Whisper).
        # The microphone is physically connected to the Jetson (USB) so
        # that Whisper can run on CUDA.  Audio output (speaker) is on the
        # Pi 5 via tts_node (Piper, CPU-only).  These are separate USB
        # devices — do NOT use a single USB speakerphone split across
        # devices.  Verify with: arecord -l (on Jetson).
        # NOTE: Echo cancellation is handled by a software mute bridge —
        # tts_node publishes /speaker/playing (Bool) before/after speaking,
        # and stt_node drops all audio frames while that flag is True.
        # This prevents the Jetson mic from transcribing the robot's own
        # TTS output from the Pi 5 speaker.
        DeclareLaunchArgument('audio_device',     default_value='plughw:1,0'),
        DeclareLaunchArgument('yolo_confidence',  default_value='0.4'),
        DeclareLaunchArgument('wake_word_enabled', default_value='true'),
        DeclareLaunchArgument('wake_word_timeout', default_value='15.0'),
        DeclareLaunchArgument(
            'enable_disease_classifier', default_value='false',
            description='Enable the plant disease TensorRT classifier in YOLOv8. '
                        'Disabled by default — it consumes ~0.3 GB VRAM and is not '
                        'needed for core school-admin functionality.',
        ),
        # ── Ollama VRAM tuning (Risk: OOM on 8 GB Jetson) ─────────────
        # On an 8 GB Jetson Orin Nano, VRAM is shared with the system.
        # Approximate budget (with int8 Whisper):
        #   ~2.0 GB  OS + CUDA runtime
        #   ~1.0 GB  Faster-Whisper (base model, int8)
        #   ~0.5 GB  YOLOv8 (TensorRT, without disease classifier)
        #   ~4.5 GB  remaining for Ollama
        # Plant disease classifier adds ~0.3 GB; disabled by default.
        # If you see OOM kills, reduce ollama_gpu_layers or ollama_num_ctx.
        # A ZRAM or physical swap file (≥8 GB) is strongly recommended —
        # LLMs will stutter when spilling to swap, but it prevents the
        # Linux OOM killer from terminating nodes mid-conversation.
        # Setup: sudo fallocate -l 8G /swapfile && sudo mkswap /swapfile
        #        && sudo swapon /swapfile  (add to /etc/fstab for persistence)
        DeclareLaunchArgument(
            'ollama_gpu_layers', default_value='999',
            description='OLLAMA_NUM_GPU: number of model layers offloaded to GPU. '
                        'Lower this to reduce VRAM usage (e.g. 20 for partial offload).',
        ),
        DeclareLaunchArgument(
            'ollama_num_ctx', default_value='2048',
            description='OLLAMA_NUM_CTX: context window size. '
                        'Smaller values use less VRAM (default 2048, min 512).',
        ),
    ]

    def create_nodes(context):
        def arg(name):
            return LaunchConfiguration(name).perform(context)

        nodes = []
        event_handlers = []

        # ── Ollama health check ────────────────────────────────────────────
        ollama_ok = _check_ollama()
        if not ollama_ok:
            print('[jetson_launch] WARNING: Ollama is not responding at '
                  'http://127.0.0.1:11434. brain_node and admin_node may '
                  'fail. Ensure "ollama serve" is running.')

        # ── 1. Robot State Publisher (URDF via xacro) ──────────────────────
        robot_description = _load_robot_description_xacro()

        rsp_node = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
            }],
        )
        nodes.append(rsp_node)
        event_handlers.append(_on_node_exit(rsp_node, 'robot_state_publisher'))

        # SLAM Toolbox, rf2o_laser_odometry, and dummy_odom intentionally
        # omitted — they now live in rpi_launch.py on the Pi 5 alongside
        # the LiDAR and motor driver.  Keeping them here would duplicate
        # the odom→base_link transform and break the TF tree.

        # ── 4. RealSense D435 camera (aligned depth for YOLO) ─────────────
        # yolov8_node subscribes to /camera/camera/color/image_raw and
        # /camera/camera/aligned_depth_to_color/image_raw.  We include the
        # vendor rs_launch.py (rather than a bare Node) because it reproduces
        # exactly those nested-namespace topic names AND enables aligned depth
        # — a plain realsense2_camera Node publishes single-level /camera/...
        # with no aligned-depth topic, which YOLO's depth path can't consume.
        # Set enable_camera:=false if you run the camera from another launch.
        if arg('enable_camera') == 'true':
            realsense = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('realsense2_camera'),
                        'launch', 'rs_launch.py',
                    ]),
                ]),
                launch_arguments={
                    'align_depth.enable': 'true',
                    'pointcloud.enable': 'false',
                }.items(),
            )
            nodes.append(realsense)

        # ── 5. YOLOv8 vision — arbiter-managed OR standalone (never both) ──
        # Mutually exclusive (see ADR 0001): two detection_node instances would
        # fight over the camera + GPU.  With the arbiter, YOLO is a managed
        # child it spawns/kills to free VRAM for the LLM during CONVERSING.
        if arg('enable_vision') == 'true':
            if arg('enable_gpu_arbiter') == 'true':
                # The arbiter (re)spawns YOLO via this command, reproducing the
                # standalone node's params.  `ros2 run` gives the node its
                # default name 'detection_node', so the pause service stays
                # /detection_node/pause_inference (the arbiter's default).
                disease = 'true' if arg('enable_disease_classifier') == 'true' else 'false'
                yolo_cmd = [
                    'ros2', 'run', 'yolov8_ros', 'yolov8_node', '--ros-args',
                    '-p', f"confidence_threshold:={arg('yolo_confidence')}",
                    '-p', 'enable_ocr:=true',
                    '-p', 'enable_faces:=true',
                    '-p', 'enable_gestures:=true',
                    '-p', f'enable_disease_classifier:={disease}',
                    '-p', 'target_fps:=30',
                ]
                gpu_arbiter_node = Node(
                    package='aisha_brain',
                    executable='gpu_arbiter',
                    name='gpu_arbiter',
                    output='screen',
                    emulate_tty=True,
                    parameters=[{
                        'manage_yolo': True,
                        'yolo_cmd': yolo_cmd,
                        'llm_model': arg('llm_model'),
                    }],
                )
                nodes.append(gpu_arbiter_node)
                event_handlers.append(_on_node_exit(gpu_arbiter_node, 'gpu_arbiter'))
            else:
                # Always-on vision, no GPU time-multiplexing (higher OOM risk
                # during live Q&A on the 8 GB Jetson).
                vision_node = Node(
                    package='yolov8_ros',
                    executable='yolov8_node',
                    name='yolov8_vision',
                    output='screen',
                    parameters=[{
                        'confidence_threshold': float(arg('yolo_confidence')),
                        'enable_ocr': True,
                        'enable_faces': True,
                        'enable_gestures': True,
                        'enable_disease_classifier': arg('enable_disease_classifier') == 'true',
                        'target_fps': 30,
                    }],
                )
                nodes.append(vision_node)
                event_handlers.append(_on_node_exit(vision_node, 'yolov8_vision'))

        # ── 6. STT Node (Faster-Whisper, fully local, CUDA) ───────────────
        if arg('enable_stt') == 'true':
            stt_node = Node(
                package='aisha_brain',
                executable='stt_node',
                name='ai_sha_stt',
                output='screen',
                parameters=[{
                    'whisper_model': arg('whisper_model'),
                    'whisper_device': arg('whisper_device'),
                    'whisper_compute_type': arg('whisper_compute'),
                    'audio_device': arg('audio_device'),
                    'sample_rate': 16000,
                    'wake_word_enabled': arg('wake_word_enabled') == 'true',
                    'wake_word_timeout': float(arg('wake_word_timeout')),
                }],
            )
            nodes.append(stt_node)
            event_handlers.append(_on_node_exit(stt_node, 'ai_sha_stt'))

        # ── Ollama VRAM env vars (passed to brain + admin nodes) ─────────
        # These environment variables are picked up by the Ollama client
        # to control GPU memory usage.  See launch args for tuning guidance.
        ollama_env = {
            'OLLAMA_NUM_GPU': arg('ollama_gpu_layers'),
            'OLLAMA_NUM_CTX': arg('ollama_num_ctx'),
        }

        # ── 7. Brain Node (keyword intent router) — delayed for STT init ──
        # 5s delay lets the STT/Whisper model load first.  brain_node routes
        # with deterministic keywords (no LLM, no Ollama), so it adds no
        # GPU/VRAM load and needs no router parameters.
        brain_node = Node(
            package='aisha_brain',
            executable='brain_node',
            name='ai_sha_brain',
            output='screen',
        )
        # Stagger cognitive nodes to avoid a "thundering herd" of heavy
        # Python processes initializing simultaneously on the Jetson's ARM
        # cores (Ollama health check, ChromaDB load, embedding model init).
        nodes.append(TimerAction(period=5.0, actions=[brain_node]))
        event_handlers.append(_on_node_exit(brain_node, 'ai_sha_brain'))

        # ── 8. Admin Node (RAG Knowledge Base, local Ollama) ──────────────
        admin_node = Node(
            package='aisha_brain',
            executable='admin_node',
            name='ai_sha_admin',
            output='screen',
            parameters=[{
                'ollama_url': 'http://127.0.0.1:11434',
                'llm_model': arg('llm_model'),
                'llm_timeout': 120.0,
                # 5 chunks fit safely within OLLAMA_NUM_CTX=2048.
                # Budget: ~500 system prompt + ~500 history (5 turns) + ~50
                # user question + 5 chunks × ~150 tokens ≈ 1800 tokens.
                # If you increase chunk_size in build_knowledge.py above
                # ~200 tokens, reduce this to 3 to stay within the context
                # window — otherwise Llama 3.2 silently truncates RAG data.
                'similarity_top_k': 5,
            }],
            additional_env=ollama_env,
        )
        nodes.append(TimerAction(period=8.0, actions=[admin_node]))
        event_handlers.append(_on_node_exit(admin_node, 'ai_sha_admin'))

        # ── 9. Action Node (WhatsApp integration) ─────────────────────────
        action_node = Node(
            package='aisha_brain',
            executable='action_node',
            name='ai_sha_action',
            output='screen',
        )
        nodes.append(TimerAction(period=11.0, actions=[action_node]))
        event_handlers.append(_on_node_exit(action_node, 'ai_sha_action'))

        return nodes + event_handlers

    return LaunchDescription(args + [OpaqueFunction(function=create_nodes)])
