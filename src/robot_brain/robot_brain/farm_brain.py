#!/usr/bin/env python3
"""
AI-SHA Farm Brain - Fully Autonomous Agricultural Robot Orchestrator

This is the central intelligence node for a smart farming robot.  The robot
operates **autonomously by default** -- it patrols farm sections, deploys soil
probes via linear actuators, reads ~16 environmental sensors, detects plant
diseases with vision, waters dry soil, and returns home to recharge.

Human voice commands are a **secondary override** channel (e.g. "stop",
"go to row 3", "skip", "report") -- they do NOT drive the primary behaviour.

Navigation uses a layered approach:
    1. Nav2 (global + local planner)  -- map-based path planning
    2. Isaac Sim RL policy            -- learned obstacle avoidance / locomotion
    3. Reactive obstacle layer        -- LiDAR + depth emergency stop

Sensor suite (~16 sensors):
    Soil moisture, BMP180 (pressure/temp/alt), DHT11 (temp/humidity),
    BNO055 (9-DOF IMU), pH sensor, UV sensor, lux sensor, 4x rotary encoders,
    raindrop sensor, gas/CO2 sensor (MQ-135), GPS, LiDAR, RealSense D435.

Actuators:
    4x mecanum motors (via /cmd_vel), water pump, seed dispenser,
    2x linear actuators (soil probe insertion / retraction).

Runs on: NVIDIA Jetson Orin Nano 8 GB  (ROS 2 Humble)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import String, Bool, Float32, Int32, Float64
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import NavSatFix, LaserScan, FluidPressure, Temperature, Imu
from nav_msgs.msg import Odometry

try:
    from nav2_msgs.action import NavigateToPose
    _NAV2_AVAILABLE = True
except ImportError:
    _NAV2_AVAILABLE = False

import json
import threading
import time
import os
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
# State machine
# ═══════════════════════════════════════════════════════════════════════════
class BrainState(Enum):
    """Primary autonomous state machine."""
    STARTUP          = auto()   # Self-check, wait for sensors
    PATROLLING       = auto()   # Moving through waypoints autonomously
    NAVIGATING       = auto()   # Travelling to a specific waypoint
    DEPLOYING_PROBES = auto()   # Extending linear actuators into soil
    MEASURING        = auto()   # Reading all sensors at current location
    ANALYSING        = auto()   # Evaluating sensor data against thresholds
    WATERING         = auto()   # Pumping water to dry soil
    SOWING           = auto()   # Dispensing seeds
    RETRACTING_PROBES = auto()  # Pulling linear actuators out of soil
    INSPECTING       = auto()   # Vision-based plant disease scan
    RETURNING_HOME   = auto()   # Navigating back to home / charger
    IDLE             = auto()   # Waiting between patrols
    PAUSED           = auto()   # Human-requested pause
    EMERGENCY_STOP   = auto()   # Obstacle too close / fault
    ERROR            = auto()   # Recoverable fault


# ═══════════════════════════════════════════════════════════════════════════
# Full sensor snapshot  (~16 sensors)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SensorSnapshot:
    """Aggregated sensor readings from all ~16 sensors."""
    # ── Soil moisture sensor (Arduino serial) ────────────────────────────
    soil_moisture_pct: float = 0.0
    soil_is_dry: bool = False
    soil_moisture_raw: int = 0

    # ── pH sensor (Arduino ADC) ──────────────────────────────────────────
    soil_ph: float = 7.0                # Neutral default
    soil_ph_raw: int = 0

    # ── BMP180 (I2C -- pressure, temperature, altitude) ──────────────────
    bmp180_temperature_c: float = 0.0
    bmp180_pressure_pa: float = 101325.0
    bmp180_altitude_m: float = 0.0

    # ── DHT11 (GPIO -- temperature, humidity) ────────────────────────────
    dht11_temperature_c: float = 0.0
    dht11_humidity_pct: float = 0.0

    # ── BNO055 9-DOF IMU (I2C) ──────────────────────────────────────────
    imu_orientation_yaw: float = 0.0
    imu_orientation_pitch: float = 0.0
    imu_orientation_roll: float = 0.0
    imu_linear_accel_x: float = 0.0
    imu_linear_accel_y: float = 0.0
    imu_linear_accel_z: float = 0.0

    # ── UV sensor (I2C / analog) ─────────────────────────────────────────
    uv_index: float = 0.0
    uv_raw: int = 0

    # ── Lux / ambient light sensor (I2C) ─────────────────────────────────
    lux: float = 0.0

    # ── Gas / CO2 sensor -- MQ-135 (Arduino ADC) ────────────────────────
    co2_ppm: float = 400.0              # Ambient default
    gas_raw: int = 0

    # ── Raindrop sensor (Arduino serial) ─────────────────────────────────
    rain_intensity_pct: float = 0.0
    is_raining: bool = False
    rain_raw: int = 0

    # ── GPS -- GT-U7 (serial NMEA) ──────────────────────────────────────
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0
    gps_fix: bool = False

    # ── 4x Rotary encoders (GPIO interrupts) ─────────────────────────────
    encoder_fl_rpm: float = 0.0
    encoder_fr_rpm: float = 0.0
    encoder_rl_rpm: float = 0.0
    encoder_rr_rpm: float = 0.0

    # ── Odometry (computed from encoders + IMU) ──────────────────────────
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_yaw: float = 0.0

    # ── LiDAR -- LD-19 (serial, 360 deg) ────────────────────────────────
    lidar_min_range: float = float('inf')
    lidar_min_angle: float = 0.0
    lidar_ranges: List[float] = field(default_factory=list)

    # ── RealSense D435 depth (USB) ──────────────────────────────────────
    depth_min_forward: float = float('inf')  # Nearest obstacle in front

    # ── Vision detections (from YOLOv8 + disease classifier) ─────────────
    detected_objects: List[Dict] = field(default_factory=list)
    detected_diseases: List[Dict] = field(default_factory=list)

    # ── Linear actuator state ────────────────────────────────────────────
    probe_deployed: bool = False

    # ── Timestamps ──────────────────────────────────────────────────────
    last_update: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Measurement record (logged per waypoint)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class WaypointMeasurement:
    """Snapshot taken at a single patrol waypoint."""
    waypoint_name: str = ''
    timestamp: float = 0.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    soil_moisture_pct: float = 0.0
    soil_ph: float = 7.0
    temperature_c: float = 0.0
    humidity_pct: float = 0.0
    pressure_pa: float = 0.0
    co2_ppm: float = 400.0
    uv_index: float = 0.0
    lux: float = 0.0
    rain_intensity_pct: float = 0.0
    is_raining: bool = False
    diseases: List[Dict] = field(default_factory=list)
    action_taken: str = ''  # e.g. "watered", "sowed", "none"


# ═══════════════════════════════════════════════════════════════════════════
# Voice override intent (secondary)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class VoiceOverride:
    """Simple human override parsed from voice."""
    action: str = ''    # stop, pause, resume, skip, go_to, water, sow, report
    target: str = ''    # Location name (for go_to)
    raw: str = ''


# ═══════════════════════════════════════════════════════════════════════════
# Farm layout (template -- configure per farm via JSON)
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_FARM_LOCATIONS = {
    'home':             {'x': 0.0,  'y': 0.0,  'yaw': 0.0,  'type': 'base'},
    'charging_station': {'x': 0.0,  'y': 0.0,  'yaw': 0.0,  'type': 'base'},
    'row_1':            {'x': 2.0,  'y': 0.0,  'yaw': 0.0,  'type': 'crop'},
    'row_2':            {'x': 2.0,  'y': 2.0,  'yaw': 0.0,  'type': 'crop'},
    'row_3':            {'x': 2.0,  'y': 4.0,  'yaw': 0.0,  'type': 'crop'},
    'row_4':            {'x': 2.0,  'y': 6.0,  'yaw': 0.0,  'type': 'crop'},
    'tomato_section':   {'x': 4.0,  'y': 0.0,  'yaw': 0.0,  'type': 'crop'},
    'potato_section':   {'x': 4.0,  'y': 2.0,  'yaw': 0.0,  'type': 'crop'},
    'pepper_section':   {'x': 4.0,  'y': 4.0,  'yaw': 0.0,  'type': 'crop'},
    'water_source':     {'x': -1.0, 'y': 0.0,  'yaw': 3.14, 'type': 'utility'},
    'seed_storage':     {'x': -1.0, 'y': 2.0,  'yaw': 3.14, 'type': 'utility'},
}

DEFAULT_PATROL_ORDER = [
    'row_1', 'row_2', 'row_3', 'row_4',
    'tomato_section', 'potato_section', 'pepper_section',
]

# ═══════════════════════════════════════════════════════════════════════════
# Autonomous decision thresholds
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_THRESHOLDS = {
    'soil_dry_pct':       30.0,     # Below this -> water
    'soil_wet_pct':       80.0,     # Above this -> skip watering
    'ph_low':             5.5,      # Acidic alert
    'ph_high':            8.0,      # Alkaline alert
    'co2_high_ppm':       1000.0,   # Elevated CO2 alert
    'uv_high_index':      8.0,      # High UV warning
    'temp_high_c':        40.0,     # Heat warning
    'temp_low_c':         5.0,      # Frost warning
    'humidity_low_pct':   20.0,     # Low humidity
    'obstacle_stop_m':    0.30,     # Emergency stop distance
    'obstacle_slow_m':    0.80,     # Slow-down distance
    'disease_conf_min':   0.60,     # Min confidence to log disease
}


# ═══════════════════════════════════════════════════════════════════════════
# Main brain node
# ═══════════════════════════════════════════════════════════════════════════
class FarmBrain(Node):
    def __init__(self):
        super().__init__('farm_brain')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('farm_locations_file', '')
        self.declare_parameter('patrol_interval_sec', 300.0)
        self.declare_parameter('probe_deploy_time_sec', 3.0)
        self.declare_parameter('probe_retract_time_sec', 3.0)
        self.declare_parameter('measure_settle_time_sec', 5.0)
        self.declare_parameter('water_duration_sec', 8.0)
        self.declare_parameter('sow_duration_sec', 4.0)
        self.declare_parameter('inspect_duration_sec', 5.0)
        self.declare_parameter('nav2_enabled', True)
        self.declare_parameter('isaac_model_path', '')
        self.declare_parameter('obstacle_avoidance_enabled', True)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('tts_topic', '/tts_text')
        self.declare_parameter('stt_topic', '/speech/text')
        self.declare_parameter('log_file', '/home/orin-robot/farm_brain_log.json')

        # Thresholds
        for key, default in DEFAULT_THRESHOLDS.items():
            self.declare_parameter(f'thresh_{key}', default)

        self.patrol_interval = self.get_parameter('patrol_interval_sec').value
        self.probe_deploy_time = self.get_parameter('probe_deploy_time_sec').value
        self.probe_retract_time = self.get_parameter('probe_retract_time_sec').value
        self.measure_settle = self.get_parameter('measure_settle_time_sec').value
        self.water_duration = self.get_parameter('water_duration_sec').value
        self.sow_duration = self.get_parameter('sow_duration_sec').value
        self.inspect_duration = self.get_parameter('inspect_duration_sec').value
        self.nav2_enabled = self.get_parameter('nav2_enabled').value
        self.isaac_model_path = self.get_parameter('isaac_model_path').value
        self.obstacle_avoidance = self.get_parameter('obstacle_avoidance_enabled').value
        self.auto_start = self.get_parameter('auto_start').value
        self.tts_topic = self.get_parameter('tts_topic').value
        self.stt_topic = self.get_parameter('stt_topic').value
        self.log_file = self.get_parameter('log_file').value

        self.thresholds = {}
        for key in DEFAULT_THRESHOLDS:
            self.thresholds[key] = self.get_parameter(f'thresh_{key}').value

        # ── State ─────────────────────────────────────────────────────────
        self.state = BrainState.STARTUP
        self.state_lock = threading.Lock()
        self.sensors = SensorSnapshot()
        self.sensor_lock = threading.Lock()
        self.patrol_index = 0
        self.last_patrol_time = 0.0
        self.patrol_log: List[WaypointMeasurement] = []
        self.current_waypoint = ''
        self.nav_goal_handle = None
        self.startup_time = time.time()
        self.voice_override: Optional[VoiceOverride] = None
        self.override_lock = threading.Lock()
        self.state_before_pause = BrainState.IDLE
        self.obstacle_stop_active = False

        # Farm locations
        self.farm_locations = dict(DEFAULT_FARM_LOCATIONS)
        self.patrol_order = list(DEFAULT_PATROL_ORDER)
        locations_file = self.get_parameter('farm_locations_file').value
        if locations_file and os.path.exists(locations_file):
            self._load_farm_locations(locations_file)

        # Isaac Sim model
        self.isaac_model = None
        self.isaac_model_loaded = False
        if self.isaac_model_path and os.path.exists(self.isaac_model_path):
            threading.Thread(target=self._load_isaac_model, daemon=True).start()

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tts_pub = self.create_publisher(String, self.tts_topic, 10)
        self.status_pub = self.create_publisher(String, '/farm_brain/status', 10)
        self.sensor_summary_pub = self.create_publisher(
            String, '/farm_brain/sensor_summary', 10)
        self.alert_pub = self.create_publisher(String, '/farm_brain/alerts', 10)
        self.measurement_pub = self.create_publisher(
            String, '/farm_brain/measurement', 10)

        # Actuator command publishers
        self.water_cmd_pub = self.create_publisher(Bool, '/actuators/water_pump', 10)
        self.seed_cmd_pub = self.create_publisher(Bool, '/actuators/seed_dispenser', 10)
        self.probe_left_pub = self.create_publisher(
            Int32, '/actuators/linear_actuator_left', 10)   # +1 extend, -1 retract, 0 stop
        self.probe_right_pub = self.create_publisher(
            Int32, '/actuators/linear_actuator_right', 10)

        # ── Subscribers: ~16 sensors ──────────────────────────────────────

        # 1. Soil moisture (Arduino serial -> RPi)
        self.create_subscription(
            Float32, '/soil_moisture/moisture', self._on_soil_moisture, 10)
        self.create_subscription(
            Bool, '/soil_moisture/dry', self._on_soil_dry, 10)
        self.create_subscription(
            Int32, '/soil_moisture/raw', self._on_soil_moisture_raw, 10)

        # 2. pH sensor (Arduino ADC -> RPi)
        self.create_subscription(
            Float32, '/ph_sensor/ph', self._on_ph, 10)
        self.create_subscription(
            Int32, '/ph_sensor/raw', self._on_ph_raw, 10)

        # 3. BMP180 (I2C on RPi)
        self.create_subscription(
            Temperature, '/bmp180/temperature', self._on_bmp180_temp, 10)
        self.create_subscription(
            FluidPressure, '/bmp180/pressure', self._on_bmp180_pressure, 10)
        self.create_subscription(
            Float64, '/bmp180/altitude', self._on_bmp180_altitude, 10)

        # 4. DHT11 (GPIO on RPi)
        self.create_subscription(
            Float32, '/dht11/temperature', self._on_dht11_temp, 10)
        self.create_subscription(
            Float32, '/dht11/humidity', self._on_dht11_humidity, 10)

        # 5. BNO055 IMU (I2C on RPi)
        self.create_subscription(Imu, '/imu/data', self._on_imu, 10)

        # 6. UV sensor (I2C / analog)
        self.create_subscription(
            Float32, '/uv_sensor/index', self._on_uv_index, 10)
        self.create_subscription(
            Int32, '/uv_sensor/raw', self._on_uv_raw, 10)

        # 7. Lux / ambient light sensor (I2C)
        self.create_subscription(
            Float32, '/lux_sensor/lux', self._on_lux, 10)

        # 8. Gas / CO2 sensor -- MQ-135 (Arduino ADC)
        self.create_subscription(
            Float32, '/gas_sensor/co2_ppm', self._on_co2, 10)
        self.create_subscription(
            Int32, '/gas_sensor/raw', self._on_gas_raw, 10)

        # 9. Raindrop sensor (Arduino serial)
        self.create_subscription(
            Float32, '/rain_sensor/intensity', self._on_rain_intensity, 10)
        self.create_subscription(
            Bool, '/rain_sensor/raining', self._on_raining, 10)
        self.create_subscription(
            Int32, '/rain_sensor/raw', self._on_rain_raw, 10)

        # 10. GPS (serial NMEA)
        gps_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(NavSatFix, '/gps/fix', self._on_gps, gps_qos)

        # 11. 4x Rotary encoders
        self.create_subscription(
            Float32, '/encoders/fl_rpm', self._on_enc_fl, 10)
        self.create_subscription(
            Float32, '/encoders/fr_rpm', self._on_enc_fr, 10)
        self.create_subscription(
            Float32, '/encoders/rl_rpm', self._on_enc_rl, 10)
        self.create_subscription(
            Float32, '/encoders/rr_rpm', self._on_enc_rr, 10)

        # 12. Odometry (fused from encoders + IMU)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        # 13. LiDAR -- LD-19
        self.create_subscription(
            LaserScan, '/scan', self._on_scan,
            QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT))

        # 14. RealSense D435 depth (min forward distance)
        self.create_subscription(
            Float32, '/camera/depth/min_forward', self._on_depth_min, 10)

        # 15-16. Vision (YOLOv8 detections + plant disease)
        self.create_subscription(
            String, '/detection/objects_simple', self._on_detections, 10)
        self.create_subscription(
            String, '/detection/disease_simple', self._on_disease, 10)

        # Voice override (secondary -- human can redirect)
        self.create_subscription(
            String, self.stt_topic, self._on_speech, 10)
        self.create_subscription(
            String, '/farm_brain/intent', self._on_intent, 10)

        # ── Nav2 action client ────────────────────────────────────────────
        self.nav2_client = None
        if _NAV2_AVAILABLE and self.nav2_enabled:
            self.nav2_client = ActionClient(
                self, NavigateToPose, 'navigate_to_pose')

        # ── Timers ────────────────────────────────────────────────────────
        self.create_timer(0.5, self._tick)                  # Main loop 2 Hz
        self.create_timer(0.1, self._obstacle_check)        # Obstacle check 10 Hz
        self.create_timer(10.0, self._publish_sensor_summary)
        self.create_timer(3.0, self._publish_status)

        # ── Startup banner ────────────────────────────────────────────────
        self.get_logger().info('=' * 70)
        self.get_logger().info('  AI-SHA FARM BRAIN - Fully Autonomous Agricultural Monitor')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'  Mode:          AUTONOMOUS (voice = override only)')
        self.get_logger().info(f'  Nav2:          {"enabled" if self.nav2_client else "disabled"}')
        self.get_logger().info(f'  Isaac model:   {self.isaac_model_path or "none"}')
        self.get_logger().info(f'  Obstacle stop: {"ON" if self.obstacle_avoidance else "OFF"}')
        self.get_logger().info(f'  Auto-start:    {"YES" if self.auto_start else "NO"}')
        self.get_logger().info(f'  Patrol every:  {self.patrol_interval}s')
        self.get_logger().info(f'  Locations:     {len(self.farm_locations)} defined')
        self.get_logger().info(f'  Patrol route:  {" -> ".join(self.patrol_order)}')
        self.get_logger().info(f'  Sensors:       ~16 (soil, pH, BMP, DHT, IMU, UV, '
                               f'lux, CO2, rain, GPS, 4x enc, LiDAR, depth, vision)')
        self.get_logger().info(f'  Actuators:     pump, seeder, 2x linear probes')
        self.get_logger().info('=' * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # Config helpers
    # ═══════════════════════════════════════════════════════════════════════
    def _load_farm_locations(self, path: str):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if 'locations' in data:
                self.farm_locations.update(data['locations'])
            else:
                self.farm_locations.update(data)
            if 'patrol_order' in data:
                self.patrol_order = data['patrol_order']
            self.get_logger().info(f'Loaded farm layout from {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load farm locations: {e}')

    def _load_isaac_model(self):
        """
        Load an Isaac Sim trained navigation policy (TensorRT / ONNX).

        The model is an RL policy trained in Isaac Sim that takes:
            Input:  [lidar_ranges(N), imu(6), goal_relative(3)]
            Output: [linear_vel, angular_vel]

        TODO: Replace with actual model loading code.
        Example with TensorRT:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.isaac_model_path, 'rb') as f:
                engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            self.isaac_model = engine
            self.isaac_model_loaded = True

        Example with ONNX Runtime:
            import onnxruntime as ort
            self.isaac_model = ort.InferenceSession(self.isaac_model_path)
            self.isaac_model_loaded = True
        """
        self.get_logger().info(
            f'Isaac Sim model path: {self.isaac_model_path}')
        self.get_logger().info(
            'TODO: Implement actual model loading (TensorRT/ONNX)')
        # self.isaac_model_loaded = True

    # ═══════════════════════════════════════════════════════════════════════
    # Sensor callbacks  (~16 sensors)
    # ═══════════════════════════════════════════════════════════════════════

    # -- 1. Soil moisture --
    def _on_soil_moisture(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.soil_moisture_pct = msg.data
    def _on_soil_dry(self, msg: Bool):
        with self.sensor_lock:
            self.sensors.soil_is_dry = msg.data
    def _on_soil_moisture_raw(self, msg: Int32):
        with self.sensor_lock:
            self.sensors.soil_moisture_raw = msg.data

    # -- 2. pH sensor --
    def _on_ph(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.soil_ph = msg.data
    def _on_ph_raw(self, msg: Int32):
        with self.sensor_lock:
            self.sensors.soil_ph_raw = msg.data

    # -- 3. BMP180 --
    def _on_bmp180_temp(self, msg: Temperature):
        with self.sensor_lock:
            self.sensors.bmp180_temperature_c = msg.temperature
    def _on_bmp180_pressure(self, msg: FluidPressure):
        with self.sensor_lock:
            self.sensors.bmp180_pressure_pa = msg.fluid_pressure
    def _on_bmp180_altitude(self, msg: Float64):
        with self.sensor_lock:
            self.sensors.bmp180_altitude_m = msg.data

    # -- 4. DHT11 --
    def _on_dht11_temp(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.dht11_temperature_c = msg.data
    def _on_dht11_humidity(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.dht11_humidity_pct = msg.data

    # -- 5. BNO055 IMU --
    def _on_imu(self, msg: Imu):
        with self.sensor_lock:
            q = msg.orientation
            # Yaw (Z)
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.sensors.imu_orientation_yaw = math.atan2(siny, cosy)
            # Pitch (Y)
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            self.sensors.imu_orientation_pitch = (
                math.asin(max(-1.0, min(1.0, sinp))))
            # Roll (X)
            sinr = 2.0 * (q.w * q.x + q.y * q.z)
            cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            self.sensors.imu_orientation_roll = math.atan2(sinr, cosr)
            # Linear acceleration
            self.sensors.imu_linear_accel_x = msg.linear_acceleration.x
            self.sensors.imu_linear_accel_y = msg.linear_acceleration.y
            self.sensors.imu_linear_accel_z = msg.linear_acceleration.z

    # -- 6. UV sensor --
    def _on_uv_index(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.uv_index = msg.data
    def _on_uv_raw(self, msg: Int32):
        with self.sensor_lock:
            self.sensors.uv_raw = msg.data

    # -- 7. Lux sensor --
    def _on_lux(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.lux = msg.data

    # -- 8. Gas / CO2 --
    def _on_co2(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.co2_ppm = msg.data
    def _on_gas_raw(self, msg: Int32):
        with self.sensor_lock:
            self.sensors.gas_raw = msg.data

    # -- 9. Raindrop sensor --
    def _on_rain_intensity(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.rain_intensity_pct = msg.data
    def _on_raining(self, msg: Bool):
        with self.sensor_lock:
            self.sensors.is_raining = msg.data
    def _on_rain_raw(self, msg: Int32):
        with self.sensor_lock:
            self.sensors.rain_raw = msg.data

    # -- 10. GPS --
    def _on_gps(self, msg: NavSatFix):
        with self.sensor_lock:
            self.sensors.gps_lat = msg.latitude
            self.sensors.gps_lon = msg.longitude
            self.sensors.gps_alt = msg.altitude
            self.sensors.gps_fix = msg.status.status >= 0

    # -- 11. Rotary encoders --
    def _on_enc_fl(self, msg: Float32):
        with self.sensor_lock: self.sensors.encoder_fl_rpm = msg.data
    def _on_enc_fr(self, msg: Float32):
        with self.sensor_lock: self.sensors.encoder_fr_rpm = msg.data
    def _on_enc_rl(self, msg: Float32):
        with self.sensor_lock: self.sensors.encoder_rl_rpm = msg.data
    def _on_enc_rr(self, msg: Float32):
        with self.sensor_lock: self.sensors.encoder_rr_rpm = msg.data

    # -- 12. Odometry --
    def _on_odom(self, msg: Odometry):
        with self.sensor_lock:
            self.sensors.odom_x = msg.pose.pose.position.x
            self.sensors.odom_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.sensors.odom_yaw = math.atan2(siny, cosy)

    # -- 13. LiDAR --
    def _on_scan(self, msg: LaserScan):
        with self.sensor_lock:
            valid = [r for r in msg.ranges
                     if msg.range_min < r < msg.range_max]
            if valid:
                min_r = min(valid)
                min_idx = msg.ranges.index(min_r)
                self.sensors.lidar_min_range = min_r
                self.sensors.lidar_min_angle = (
                    msg.angle_min + min_idx * msg.angle_increment)
            else:
                self.sensors.lidar_min_range = float('inf')
            self.sensors.lidar_ranges = list(msg.ranges)

    # -- 14. Depth camera --
    def _on_depth_min(self, msg: Float32):
        with self.sensor_lock:
            self.sensors.depth_min_forward = msg.data

    # -- 15-16. Vision --
    def _on_detections(self, msg: String):
        with self.sensor_lock:
            try:
                self.sensors.detected_objects = json.loads(msg.data)
            except (json.JSONDecodeError, TypeError):
                self.sensors.detected_objects = []

    def _on_disease(self, msg: String):
        with self.sensor_lock:
            try:
                data = json.loads(msg.data)
                self.sensors.detected_diseases = (
                    data if isinstance(data, list) else [data])
            except (json.JSONDecodeError, TypeError):
                self.sensors.detected_diseases = []

    # ═══════════════════════════════════════════════════════════════════════
    # Obstacle avoidance (continuous, 10 Hz)
    # ═══════════════════════════════════════════════════════════════════════
    def _obstacle_check(self):
        """Reactive obstacle avoidance running at 10 Hz."""
        if not self.obstacle_avoidance:
            return

        with self.state_lock:
            state = self.state
        # Only intervene while moving
        if state not in (BrainState.NAVIGATING, BrainState.PATROLLING,
                         BrainState.RETURNING_HOME):
            if self.obstacle_stop_active:
                self.obstacle_stop_active = False
            return

        with self.sensor_lock:
            lidar_min = self.sensors.lidar_min_range
            depth_min = self.sensors.depth_min_forward

        nearest = min(lidar_min, depth_min)
        stop_dist = self.thresholds['obstacle_stop_m']
        slow_dist = self.thresholds['obstacle_slow_m']

        if nearest < stop_dist:
            # Emergency stop
            if not self.obstacle_stop_active:
                self.obstacle_stop_active = True
                self.cmd_vel_pub.publish(Twist())  # zero velocity
                self.get_logger().warn(
                    f'OBSTACLE STOP: {nearest:.2f}m < {stop_dist:.2f}m')
                self._alert('obstacle_stop',
                            f'Emergency stop - obstacle at {nearest:.2f}m')
        elif nearest < slow_dist:
            # Slow down (scale velocity by proximity factor)
            if self.obstacle_stop_active:
                self.obstacle_stop_active = False
            # Isaac Sim policy can handle fine-grained avoidance here
            if self.isaac_model_loaded:
                self._apply_isaac_avoidance()
        else:
            if self.obstacle_stop_active:
                self.obstacle_stop_active = False
                self.get_logger().info('Obstacle cleared, resuming')

    def _apply_isaac_avoidance(self):
        """
        Use Isaac Sim RL policy for local obstacle avoidance.

        TODO: Build observation tensor from current sensor state,
        run inference through the TensorRT engine, and publish
        the resulting velocity command.

        Observation vector (example):
            [lidar_ranges_downsampled(36),   # 360deg / 10deg bins
             imu_linear_accel(3),
             imu_angular_vel(3),
             goal_relative_x, goal_relative_y, goal_relative_yaw]

        Action vector:
            [linear_vel_x, angular_vel_z]
        """
        pass  # TODO: implement after Isaac Sim training

    # ═══════════════════════════════════════════════════════════════════════
    # Voice override (secondary interface)
    # ═══════════════════════════════════════════════════════════════════════
    def _on_speech(self, msg: String):
        """Parse voice command as a simple override."""
        raw = msg.data.strip()
        if not raw:
            return
        self.get_logger().info(f'Voice override: "{raw}"')
        override = self._parse_voice_override(raw)
        if override:
            with self.override_lock:
                self.voice_override = override

    def _on_intent(self, msg: String):
        """Accept structured JSON intent from LLM."""
        try:
            data = json.loads(msg.data)
            override = VoiceOverride(
                action=data.get('action', ''),
                target=data.get('target', ''),
                raw=data.get('raw_command', ''),
            )
            with self.override_lock:
                self.voice_override = override
        except (json.JSONDecodeError, TypeError):
            pass

    def _parse_voice_override(self, raw: str) -> Optional[VoiceOverride]:
        cmd = raw.lower()
        ov = VoiceOverride(raw=raw)
        if any(w in cmd for w in ['stop', 'halt', 'emergency']):
            ov.action = 'stop'
        elif any(w in cmd for w in ['pause', 'wait', 'hold']):
            ov.action = 'pause'
        elif any(w in cmd for w in ['resume', 'continue', 'start']):
            ov.action = 'resume'
        elif any(w in cmd for w in ['skip', 'next']):
            ov.action = 'skip'
        elif any(w in cmd for w in ['report', 'status', 'sensor']):
            ov.action = 'report'
        elif any(w in cmd for w in ['go to', 'navigate', 'move to', 'head to']):
            ov.action = 'go_to'
            for loc in self.farm_locations:
                if loc.replace('_', ' ') in cmd:
                    ov.target = loc
                    break
        elif any(w in cmd for w in ['water', 'irrigate']):
            ov.action = 'water'
        elif any(w in cmd for w in ['sow', 'seed', 'plant']):
            ov.action = 'sow'
        elif 'home' in cmd or 'return' in cmd:
            ov.action = 'go_to'
            ov.target = 'home'
        else:
            return None
        return ov

    def _handle_voice_override(self):
        """Process any pending voice override (called from _tick)."""
        with self.override_lock:
            ov = self.voice_override
            self.voice_override = None
        if ov is None:
            return

        self.get_logger().info(f'Processing override: {ov.action}')

        if ov.action == 'stop':
            self._emergency_stop('Voice command')
        elif ov.action == 'pause':
            with self.state_lock:
                self.state_before_pause = self.state
                self.state = BrainState.PAUSED
            self.cmd_vel_pub.publish(Twist())
            self._speak('Paused. Say resume to continue.')
        elif ov.action == 'resume':
            with self.state_lock:
                if self.state == BrainState.PAUSED:
                    self.state = self.state_before_pause
            self._speak('Resuming autonomous operation.')
        elif ov.action == 'skip':
            self._speak('Skipping current waypoint.')
            self._advance_patrol()
        elif ov.action == 'report':
            self._speak_status_report()
        elif ov.action == 'go_to':
            if ov.target in self.farm_locations:
                self._speak(f'Redirecting to {ov.target.replace("_", " ")}.')
                with self.state_lock:
                    self.state = BrainState.NAVIGATING
                self.current_waypoint = ov.target
                self._navigate_to(ov.target)
            else:
                self._speak(f'Unknown location: {ov.target}')
        elif ov.action == 'water':
            self._speak('Watering at current position.')
            threading.Thread(target=self._do_water, daemon=True).start()
        elif ov.action == 'sow':
            self._speak('Sowing at current position.')
            threading.Thread(target=self._do_sow, daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════════
    # Main autonomous tick  (2 Hz)
    # ═══════════════════════════════════════════════════════════════════════
    def _tick(self):
        with self.sensor_lock:
            self.sensors.last_update = time.time()

        # Always handle voice overrides first
        self._handle_voice_override()

        with self.state_lock:
            state = self.state

        if state == BrainState.STARTUP:
            self._tick_startup()
        elif state == BrainState.IDLE:
            self._tick_idle()
        elif state == BrainState.PATROLLING:
            self._tick_patrol()
        elif state == BrainState.NAVIGATING:
            pass  # Nav2 callbacks handle transition
        elif state == BrainState.DEPLOYING_PROBES:
            pass  # handled in thread
        elif state == BrainState.MEASURING:
            pass  # handled in thread
        elif state == BrainState.ANALYSING:
            pass  # handled in thread
        elif state == BrainState.WATERING:
            pass  # handled in thread
        elif state == BrainState.SOWING:
            pass  # handled in thread
        elif state == BrainState.RETRACTING_PROBES:
            pass  # handled in thread
        elif state == BrainState.INSPECTING:
            pass  # handled in thread
        elif state == BrainState.RETURNING_HOME:
            pass  # Nav2 handles
        elif state == BrainState.PAUSED:
            pass  # waiting for resume
        elif state == BrainState.EMERGENCY_STOP:
            pass  # waiting for clear / voice resume
        elif state == BrainState.ERROR:
            pass  # waiting for recovery

    def _tick_startup(self):
        """Wait for sensors to come online, then begin."""
        elapsed = time.time() - self.startup_time
        if elapsed > 10.0 and self.auto_start:
            self.get_logger().info('Startup complete -> beginning patrol')
            self._speak('All systems online. Beginning autonomous farm patrol.')
            with self.state_lock:
                self.state = BrainState.IDLE
            self.last_patrol_time = 0.0  # Trigger immediate patrol

    def _tick_idle(self):
        """Between patrols -- check if it's time to patrol again."""
        now = time.time()
        if now - self.last_patrol_time >= self.patrol_interval:
            self.last_patrol_time = now
            self.patrol_index = 0
            self.patrol_log = []
            with self.state_lock:
                self.state = BrainState.PATROLLING
            self.get_logger().info('Starting new patrol cycle')

    def _tick_patrol(self):
        """Advance through patrol waypoints."""
        if self.obstacle_stop_active:
            return  # Wait for obstacle to clear

        if self.patrol_index >= len(self.patrol_order):
            self._finish_patrol()
            return

        wp_name = self.patrol_order[self.patrol_index]
        if self.current_waypoint == wp_name:
            return  # Already navigating to this one

        self.current_waypoint = wp_name
        self.get_logger().info(
            f'Patrol [{self.patrol_index + 1}/{len(self.patrol_order)}]: {wp_name}')
        with self.state_lock:
            self.state = BrainState.NAVIGATING
        self._navigate_to(wp_name)

    def _advance_patrol(self):
        """Move to next waypoint in patrol."""
        self.patrol_index += 1
        self.current_waypoint = ''
        with self.state_lock:
            self.state = BrainState.PATROLLING

    def _finish_patrol(self):
        """Patrol cycle done -- log, report, return home."""
        self.get_logger().info('Patrol cycle complete')
        self._log_patrol()
        # Publish summary
        summary = self._build_patrol_summary()
        self._speak(summary)
        # Return home
        with self.state_lock:
            self.state = BrainState.RETURNING_HOME
        self.current_waypoint = 'home'
        self._navigate_to('home')

    # ═══════════════════════════════════════════════════════════════════════
    # Navigation (Nav2 + Isaac Sim)
    # ═══════════════════════════════════════════════════════════════════════
    def _navigate_to(self, location_name: str):
        """Send navigation goal via Nav2 or fallback controller."""
        if location_name not in self.farm_locations:
            self.get_logger().error(f'Unknown location: {location_name}')
            self._advance_patrol()
            return

        loc = self.farm_locations[location_name]
        x, y, yaw = loc['x'], loc['y'], loc.get('yaw', 0.0)

        if self.nav2_client:
            self._send_nav2_goal(x, y, yaw)
        else:
            threading.Thread(
                target=self._fallback_navigate,
                args=(x, y, yaw), daemon=True).start()

    def _send_nav2_goal(self, x: float, y: float, yaw: float):
        if not _NAV2_AVAILABLE or not self.nav2_client:
            return
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        if not self.nav2_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 not available, using fallback')
            threading.Thread(
                target=self._fallback_navigate,
                args=(x, y, yaw), daemon=True).start()
            return

        future = self.nav2_client.send_goal_async(
            goal, feedback_callback=self._nav2_feedback)
        future.add_done_callback(self._nav2_goal_response)

    def _nav2_feedback(self, feedback_msg):
        pass  # Nav2 handles path planning + local avoidance

    def _nav2_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 goal rejected')
            self._advance_patrol()
            return
        self.nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav2_result)

    def _nav2_result(self, future):
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info(
                f'Arrived at {self.current_waypoint}')
            self._on_arrival()
        else:
            self.get_logger().warn(
                f'Nav failed (status={result.status}), advancing')
            self._advance_patrol()

    def _fallback_navigate(self, tx, ty, tyaw):
        """Proportional controller with obstacle avoidance."""
        kp_lin, kp_ang = 0.4, 1.2
        rate_hz = 10
        for _ in range(rate_hz * 120):  # max 2 min
            with self.state_lock:
                if self.state not in (BrainState.NAVIGATING,
                                      BrainState.RETURNING_HOME):
                    return
            if self.obstacle_stop_active:
                time.sleep(0.2)
                continue

            with self.sensor_lock:
                dx = tx - self.sensors.odom_x
                dy = ty - self.sensors.odom_y
            dist = math.hypot(dx, dy)
            if dist < 0.15:
                break

            # Isaac Sim policy can replace this controller
            if self.isaac_model_loaded:
                vx, wz = self._query_isaac_policy(tx, ty)
            else:
                goal_yaw = math.atan2(dy, dx)
                with self.sensor_lock:
                    yaw_err = goal_yaw - self.sensors.odom_yaw
                yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
                vx = min(kp_lin * dist, 0.3)
                wz = kp_ang * yaw_err

            twist = Twist()
            twist.linear.x = vx
            twist.angular.z = wz
            self.cmd_vel_pub.publish(twist)
            time.sleep(1.0 / rate_hz)

        self.cmd_vel_pub.publish(Twist())
        self._on_arrival()

    def _on_arrival(self):
        """Called when the robot reaches a waypoint."""
        with self.state_lock:
            st = self.state
        if st == BrainState.RETURNING_HOME:
            self.get_logger().info('Arrived home')
            self._speak('Patrol complete. Home position reached.')
            with self.state_lock:
                self.state = BrainState.IDLE
            return

        # At a crop waypoint -- run full measure-analyse-act sequence
        threading.Thread(
            target=self._waypoint_sequence, daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════════
    # Waypoint sequence:  deploy -> measure -> analyse -> act -> retract
    # ═══════════════════════════════════════════════════════════════════════
    def _waypoint_sequence(self):
        """Full autonomous sequence at each patrol waypoint."""
        wp = self.current_waypoint
        self.get_logger().info(f'=== Waypoint sequence: {wp} ===')

        # 1. Deploy soil probes
        self._set_state(BrainState.DEPLOYING_PROBES)
        self._deploy_probes()
        time.sleep(self.probe_deploy_time)

        # 2. Settle and measure
        self._set_state(BrainState.MEASURING)
        time.sleep(self.measure_settle)
        measurement = self._take_measurement(wp)

        # 3. Visual inspection (concurrent with soil reading)
        self._set_state(BrainState.INSPECTING)
        time.sleep(self.inspect_duration)
        with self.sensor_lock:
            measurement.diseases = list(self.sensors.detected_diseases)

        # 4. Retract probes
        self._set_state(BrainState.RETRACTING_PROBES)
        self._retract_probes()
        time.sleep(self.probe_retract_time)

        # 5. Analyse and act
        self._set_state(BrainState.ANALYSING)
        action = self._analyse_and_act(measurement)
        measurement.action_taken = action

        # 6. Log measurement
        self.patrol_log.append(measurement)
        self._publish_measurement(measurement)

        self.get_logger().info(
            f'=== {wp} done (action={action}) ===')

        # 7. Advance to next waypoint
        self._advance_patrol()

    def _take_measurement(self, wp_name: str) -> WaypointMeasurement:
        """Snapshot all sensors into a measurement record."""
        with self.sensor_lock:
            s = self.sensors
            return WaypointMeasurement(
                waypoint_name=wp_name,
                timestamp=time.time(),
                gps_lat=s.gps_lat,
                gps_lon=s.gps_lon,
                soil_moisture_pct=s.soil_moisture_pct,
                soil_ph=s.soil_ph,
                temperature_c=s.dht11_temperature_c or s.bmp180_temperature_c,
                humidity_pct=s.dht11_humidity_pct,
                pressure_pa=s.bmp180_pressure_pa,
                co2_ppm=s.co2_ppm,
                uv_index=s.uv_index,
                lux=s.lux,
                rain_intensity_pct=s.rain_intensity_pct,
                is_raining=s.is_raining,
            )

    def _analyse_and_act(self, m: WaypointMeasurement) -> str:
        """Evaluate measurement against thresholds and take action."""
        th = self.thresholds
        actions = []

        # Soil too dry -> water (unless raining)
        if (m.soil_moisture_pct < th['soil_dry_pct']
                and not m.is_raining):
            self.get_logger().info(
                f'{m.waypoint_name}: soil dry ({m.soil_moisture_pct:.0f}%) -> watering')
            self._do_water()
            actions.append('watered')

        # pH out of range -> alert
        if m.soil_ph < th['ph_low']:
            self._alert('ph_low',
                        f'{m.waypoint_name}: pH too low ({m.soil_ph:.1f})')
            actions.append('ph_low_alert')
        elif m.soil_ph > th['ph_high']:
            self._alert('ph_high',
                        f'{m.waypoint_name}: pH too high ({m.soil_ph:.1f})')
            actions.append('ph_high_alert')

        # CO2 elevated -> alert
        if m.co2_ppm > th['co2_high_ppm']:
            self._alert('co2_high',
                        f'{m.waypoint_name}: CO2 elevated ({m.co2_ppm:.0f} ppm)')
            actions.append('co2_alert')

        # Temperature extremes
        temp = m.temperature_c
        if temp > th['temp_high_c']:
            self._alert('temp_high', f'{m.waypoint_name}: heat {temp:.1f}C')
            actions.append('heat_alert')
        elif temp < th['temp_low_c']:
            self._alert('temp_low', f'{m.waypoint_name}: frost {temp:.1f}C')
            actions.append('frost_alert')

        # UV too high
        if m.uv_index > th['uv_high_index']:
            self._alert('uv_high',
                        f'{m.waypoint_name}: high UV ({m.uv_index:.1f})')
            actions.append('uv_alert')

        # Disease detected
        for d in m.diseases:
            conf = d.get('conf', 0)
            if conf >= th['disease_conf_min']:
                disease_name = d.get('disease', 'unknown')
                self._alert('disease',
                            f'{m.waypoint_name}: {disease_name} ({conf*100:.0f}%)')
                actions.append(f'disease:{disease_name}')

        return ', '.join(actions) if actions else 'none'

    # ═══════════════════════════════════════════════════════════════════════
    # Actuators
    # ═══════════════════════════════════════════════════════════════════════
    def _deploy_probes(self):
        """Extend both linear actuators to insert soil probes."""
        self.get_logger().info('Deploying soil probes (extending actuators)')
        cmd = Int32()
        cmd.data = 1  # +1 = extend
        self.probe_left_pub.publish(cmd)
        self.probe_right_pub.publish(cmd)
        with self.sensor_lock:
            self.sensors.probe_deployed = True

    def _retract_probes(self):
        """Retract both linear actuators."""
        self.get_logger().info('Retracting soil probes')
        cmd = Int32()
        cmd.data = -1  # -1 = retract
        self.probe_left_pub.publish(cmd)
        self.probe_right_pub.publish(cmd)
        # Stop after retract time (handled by caller sleep)
        time.sleep(self.probe_retract_time)
        cmd.data = 0
        self.probe_left_pub.publish(cmd)
        self.probe_right_pub.publish(cmd)
        with self.sensor_lock:
            self.sensors.probe_deployed = False

    def _do_water(self):
        """Activate water pump for configured duration."""
        self._set_state(BrainState.WATERING)
        self.get_logger().info(f'Watering for {self.water_duration}s')
        msg = Bool()
        msg.data = True
        self.water_cmd_pub.publish(msg)
        time.sleep(self.water_duration)
        msg.data = False
        self.water_cmd_pub.publish(msg)
        self.get_logger().info('Watering complete')

    def _do_sow(self):
        """Activate seed dispenser for configured duration."""
        self._set_state(BrainState.SOWING)
        self.get_logger().info(f'Sowing for {self.sow_duration}s')
        msg = Bool()
        msg.data = True
        self.seed_cmd_pub.publish(msg)
        time.sleep(self.sow_duration)
        msg.data = False
        self.seed_cmd_pub.publish(msg)
        self.get_logger().info('Sowing complete')

    def _emergency_stop(self, reason: str):
        """Full stop -- cancel Nav2, zero velocity, alert."""
        self.get_logger().error(f'EMERGENCY STOP: {reason}')
        with self.state_lock:
            self.state = BrainState.EMERGENCY_STOP
        self.cmd_vel_pub.publish(Twist())
        if (self.nav_goal_handle is not None
                and self.nav2_client):
            try:
                self.nav_goal_handle.cancel_goal_async()
            except Exception:
                pass
        self._retract_probes()
        self._alert('emergency_stop', reason)
        self._speak(f'Emergency stop. {reason}.')

    # ═══════════════════════════════════════════════════════════════════════
    # Isaac Sim RL policy
    # ═══════════════════════════════════════════════════════════════════════
    def _query_isaac_policy(self, goal_x: float, goal_y: float
                            ) -> Tuple[float, float]:
        """
        Query Isaac Sim trained RL policy for velocity commands.

        Observation vector (build from current sensor state):
            - LiDAR ranges downsampled to N bins (e.g. 36 bins for 10deg each)
            - IMU linear acceleration (3)
            - IMU angular velocity (3) -- TODO: subscribe to angular_velocity
            - Relative goal position in robot frame (dx, dy, dyaw)

        Returns:
            (linear_vel_x, angular_vel_z)

        TODO: Implement actual inference once Isaac Sim model is trained.
        The trained policy .onnx/.engine file goes at isaac_model_path.

        Example inference with ONNX Runtime:
            import numpy as np
            obs = self._build_observation(goal_x, goal_y)
            input_name = self.isaac_model.get_inputs()[0].name
            result = self.isaac_model.run(None, {input_name: obs})
            return (float(result[0][0]), float(result[0][1]))

        Example inference with TensorRT:
            import pycuda.driver as cuda
            # Copy observation to GPU, run engine, copy output
        """
        if not self.isaac_model_loaded or self.isaac_model is None:
            return (0.0, 0.0)
        return (0.0, 0.0)

    def _build_observation(self, goal_x: float, goal_y: float):
        """
        Build Isaac Sim observation tensor from current sensor state.

        TODO: Match this to your Isaac Sim training observation space.
        """
        import numpy as np
        with self.sensor_lock:
            s = self.sensors
            # Downsample LiDAR to 36 bins
            ranges = s.lidar_ranges if s.lidar_ranges else [0.0] * 36
            if len(ranges) > 36:
                step = len(ranges) // 36
                lidar_obs = [ranges[i * step] for i in range(36)]
            else:
                lidar_obs = ranges + [0.0] * (36 - len(ranges))
            # Relative goal in robot frame
            dx = goal_x - s.odom_x
            dy = goal_y - s.odom_y
            cos_y = math.cos(-s.odom_yaw)
            sin_y = math.sin(-s.odom_yaw)
            rel_x = dx * cos_y - dy * sin_y
            rel_y = dx * sin_y + dy * cos_y
            goal_dist = math.hypot(dx, dy)

            obs = np.array(
                lidar_obs
                + [s.imu_linear_accel_x, s.imu_linear_accel_y,
                   s.imu_linear_accel_z]
                + [rel_x, rel_y, goal_dist],
                dtype=np.float32
            ).reshape(1, -1)
        return obs

    # ═══════════════════════════════════════════════════════════════════════
    # Publishing helpers
    # ═══════════════════════════════════════════════════════════════════════
    def _set_state(self, new_state: BrainState):
        with self.state_lock:
            self.state = new_state

    def _speak(self, text: str):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
        self.get_logger().info(f'TTS: "{text}"')

    def _alert(self, alert_type: str, message: str):
        msg = String()
        msg.data = json.dumps({
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
            'waypoint': self.current_waypoint,
        })
        self.alert_pub.publish(msg)
        self.get_logger().warn(f'ALERT [{alert_type}]: {message}')

    def _publish_status(self):
        with self.state_lock:
            state_name = self.state.name
        msg = String()
        msg.data = json.dumps({
            'state': state_name,
            'waypoint': self.current_waypoint,
            'patrol_progress': (f'{self.patrol_index}/'
                                f'{len(self.patrol_order)}'),
            'obstacle_stop': self.obstacle_stop_active,
            'timestamp': time.time(),
        })
        self.status_pub.publish(msg)

    def _publish_sensor_summary(self):
        with self.sensor_lock:
            s = self.sensors
            summary = {
                'soil_moisture_pct': round(s.soil_moisture_pct, 1),
                'soil_ph': round(s.soil_ph, 2),
                'bmp180_temp_c': round(s.bmp180_temperature_c, 1),
                'bmp180_pressure_hpa': round(s.bmp180_pressure_pa / 100, 1),
                'bmp180_altitude_m': round(s.bmp180_altitude_m, 1),
                'dht11_temp_c': round(s.dht11_temperature_c, 1),
                'dht11_humidity_pct': round(s.dht11_humidity_pct, 1),
                'uv_index': round(s.uv_index, 1),
                'lux': round(s.lux, 0),
                'co2_ppm': round(s.co2_ppm, 0),
                'rain_intensity_pct': round(s.rain_intensity_pct, 1),
                'is_raining': s.is_raining,
                'gps': {'lat': round(s.gps_lat, 7),
                        'lon': round(s.gps_lon, 7),
                        'fix': s.gps_fix},
                'odom': {'x': round(s.odom_x, 2),
                         'y': round(s.odom_y, 2),
                         'yaw_deg': round(math.degrees(s.odom_yaw), 1)},
                'encoders_rpm': {
                    'fl': round(s.encoder_fl_rpm, 0),
                    'fr': round(s.encoder_fr_rpm, 0),
                    'rl': round(s.encoder_rl_rpm, 0),
                    'rr': round(s.encoder_rr_rpm, 0)},
                'lidar_min_m': (round(s.lidar_min_range, 2)
                                if s.lidar_min_range != float('inf') else None),
                'depth_min_m': (round(s.depth_min_forward, 2)
                                if s.depth_min_forward != float('inf') else None),
                'probe_deployed': s.probe_deployed,
                'detections': len(s.detected_objects),
                'diseases': len(s.detected_diseases),
                'timestamp': time.time(),
            }
        msg = String()
        msg.data = json.dumps(summary)
        self.sensor_summary_pub.publish(msg)

    def _publish_measurement(self, m: WaypointMeasurement):
        msg = String()
        msg.data = json.dumps({
            'waypoint': m.waypoint_name,
            'timestamp': m.timestamp,
            'gps': {'lat': m.gps_lat, 'lon': m.gps_lon},
            'soil_moisture_pct': m.soil_moisture_pct,
            'soil_ph': m.soil_ph,
            'temperature_c': m.temperature_c,
            'humidity_pct': m.humidity_pct,
            'pressure_pa': m.pressure_pa,
            'co2_ppm': m.co2_ppm,
            'uv_index': m.uv_index,
            'lux': m.lux,
            'rain_intensity_pct': m.rain_intensity_pct,
            'is_raining': m.is_raining,
            'diseases': m.diseases,
            'action_taken': m.action_taken,
        })
        self.measurement_pub.publish(msg)

    def _speak_status_report(self):
        with self.sensor_lock:
            s = self.sensors
        report = (
            f'Autonomous farm report. '
            f'Soil moisture {s.soil_moisture_pct:.0f} percent, '
            f'pH {s.soil_ph:.1f}. '
            f'Temperature {s.dht11_temperature_c:.0f} degrees, '
            f'humidity {s.dht11_humidity_pct:.0f} percent. '
            f'CO2 {s.co2_ppm:.0f} ppm. '
            f'UV index {s.uv_index:.1f}. '
            f'{"Raining. " if s.is_raining else ""}'
            f'Patrol progress {self.patrol_index} of {len(self.patrol_order)}. '
        )
        if s.detected_diseases:
            names = [d.get('disease', '?') for d in s.detected_diseases]
            report += f'Active diseases: {", ".join(names)}. '
        self._speak(report)

    def _build_patrol_summary(self) -> str:
        """Build spoken summary of completed patrol."""
        n = len(self.patrol_log)
        watered = sum(1 for m in self.patrol_log if 'watered' in m.action_taken)
        diseases = sum(1 for m in self.patrol_log if m.diseases)
        return (
            f'Patrol summary. Visited {n} sections. '
            f'Watered {watered} sections. '
            f'Found diseases in {diseases} sections. '
            f'Returning home.'
        )

    def _log_patrol(self):
        """Append patrol log to disk."""
        try:
            existing = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    existing = json.load(f)
            log_entry = {
                'patrol_time': time.time(),
                'measurements': [
                    {
                        'waypoint': m.waypoint_name,
                        'timestamp': m.timestamp,
                        'soil_moisture_pct': m.soil_moisture_pct,
                        'soil_ph': m.soil_ph,
                        'temperature_c': m.temperature_c,
                        'humidity_pct': m.humidity_pct,
                        'co2_ppm': m.co2_ppm,
                        'uv_index': m.uv_index,
                        'lux': m.lux,
                        'is_raining': m.is_raining,
                        'diseases': m.diseases,
                        'action_taken': m.action_taken,
                    }
                    for m in self.patrol_log
                ],
            }
            existing.append(log_entry)
            with open(self.log_file, 'w') as f:
                json.dump(existing, f, indent=2)
            self.get_logger().info(f'Patrol log saved to {self.log_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to save patrol log: {e}')

    # ═══════════════════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════════════════
    def destroy_node(self):
        self.cmd_vel_pub.publish(Twist())
        # Ensure probes retracted
        cmd = Int32()
        cmd.data = -1
        self.probe_left_pub.publish(cmd)
        self.probe_right_pub.publish(cmd)
        super().destroy_node()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    os.environ.setdefault('ROS_DOMAIN_ID', '0')

    rclpy.init(args=args)
    try:
        node = FarmBrain()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
