#!/usr/bin/env python3
"""
Generate AI-SHA Farm Brain technical documentation PDF.
Uses fpdf2 for LaTeX-style formatting without requiring a TeX installation.

Usage:
    python3 docs/generate_farm_brain_pdf.py
"""

from fpdf import FPDF
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'FARM_BRAIN_DOCUMENTATION.pdf')


class DocPDF(FPDF):
    """Custom PDF with header/footer and helper methods."""

    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'AI-SHA Farm Brain -- Technical Documentation', align='L')
        self.cell(0, 8, f'Page {self.page_no()}/{{nb}}', align='R', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(180, 180, 180)
        self.line(10, 16, 200, 16)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, 'International School of Choueifat, Sharjah', align='C')

    def title_page(self):
        self.add_page()
        self.ln(50)
        self.set_font('Helvetica', 'B', 28)
        self.set_text_color(30, 30, 30)
        self.cell(0, 15, 'AI-SHA Farm Brain', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, 'Fully Autonomous Agricultural Robot', align='C', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 10, 'Orchestrator Node', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(10)
        self.set_font('Helvetica', '', 12)
        self.cell(0, 8, 'Technical Documentation', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(20)
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(60, 60, 60)
        lines = [
            'Platform: NVIDIA Jetson Orin Nano 8GB',
            'Framework: ROS 2 Humble',
            'Navigation: Nav2 + Isaac Sim RL Policy',
            'Vision: YOLOv8m (TensorRT) + MobileNetV3 Plant Disease',
            'Sensors: ~16 environmental + 2 linear actuators',
            '',
            'Repository: github.com/Ahmed28309/AI-SHA',
        ]
        for line in lines:
            self.cell(0, 7, line, align='C', new_x='LMARGIN', new_y='NEXT')

    def section(self, num, title, level=1):
        self.ln(5)
        if level == 1:
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(0, 70, 150)
            self.cell(0, 10, f'{num}  {title}', new_x='LMARGIN', new_y='NEXT')
            self.set_draw_color(0, 70, 150)
            self.line(10, self.get_y(), 200, self.get_y())
        elif level == 2:
            self.set_font('Helvetica', 'B', 13)
            self.set_text_color(40, 40, 40)
            self.cell(0, 8, f'{num}  {title}', new_x='LMARGIN', new_y='NEXT')
        elif level == 3:
            self.set_font('Helvetica', 'BI', 11)
            self.set_text_color(60, 60, 60)
            self.cell(0, 7, f'{num}  {title}', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        self.set_text_color(30, 30, 30)

    def body(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font('Courier', '', 8.5)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        lines = text.strip().split('\n')
        for line in lines:
            # Truncate long lines
            if len(line) > 100:
                line = line[:97] + '...'
            self.cell(0, 4.5, '  ' + line, fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(3)
        self.set_font('Helvetica', '', 10)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - 20) / len(headers)
            col_widths = [w] * len(headers)
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(0, 70, 150)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()
        # Rows
        self.set_font('Helvetica', '', 8.5)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
            self.set_fill_color(245, 245, 245) if fill else self.set_fill_color(255, 255, 255)
            max_h = 6
            for i, cell_text in enumerate(row):
                self.cell(col_widths[i], max_h, str(cell_text), border=1,
                          fill=fill, align='L')
            self.ln()
            fill = not fill
        self.ln(3)

    def bullet(self, text, indent=10):
        self.set_font('Helvetica', '', 10)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, chr(8226))
        self.multi_cell(0, 5.5, text)


def build_pdf():
    pdf = DocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title Page ────────────────────────────────────────────────────
    pdf.title_page()

    # ── Table of Contents ─────────────────────────────────────────────
    pdf.add_page()
    pdf.section('', 'Table of Contents', level=1)
    toc = [
        '1.  System Overview',
        '2.  Architecture',
        '    2.1  State Machine',
        '    2.2  Navigation Stack',
        '    2.3  Isaac Sim Integration',
        '    2.4  Obstacle Avoidance',
        '3.  Sensor Suite (~16 Sensors)',
        '4.  Actuators',
        '    4.1  Linear Actuators (Soil Probes)',
        '    4.2  Water Pump & Seed Dispenser',
        '5.  ROS 2 Node Details',
        '    5.1  Subscribed Topics',
        '    5.2  Published Topics',
        '    5.3  Parameters',
        '6.  Autonomous Decision Engine',
        '7.  Voice Override (Secondary)',
        '8.  Waypoint Sequence',
        '9.  Configuration',
        '10. Launch',
        '11. Data Logging',
        '12. All ROS 2 Nodes in the System',
    ]
    for item in toc:
        pdf.body(item)

    # ═════════════════════════════════════════════════════════════════
    # 1. System Overview
    # ═════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section('1', 'System Overview')
    pdf.body(
        'AI-SHA is a fully autonomous agricultural robot built on three '
        'compute platforms: an NVIDIA Jetson Orin Nano 8GB (AI processing), '
        'a Raspberry Pi 5 (display and TTS), and a Raspberry Pi 4 (motor '
        'control, encoders, and low-level sensors).\n\n'
        'The Farm Brain node (farm_brain.py) is the central orchestrator '
        'running on the Jetson. It operates autonomously by default -- the '
        'robot patrols pre-defined farm sections, deploys soil probes via '
        'linear actuators, reads ~16 environmental sensors, detects plant '
        'diseases using computer vision, and takes corrective actions such '
        'as watering dry soil.\n\n'
        'Human voice commands serve only as an override channel. The robot '
        'does not require human input to begin or continue its work. Voice '
        'commands like "stop", "pause", "skip", or "go to row 3" override '
        'the autonomous loop when spoken.\n\n'
        'The three core intelligence layers are:\n'
        '  1. LLM (Llama 3.2 3B) -- parses natural language into intents\n'
        '  2. Isaac Sim RL Policy -- trained navigation / obstacle avoidance\n'
        '  3. Farm Brain State Machine -- autonomous decision-making\n'
    )

    # ═════════════════════════════════════════════════════════════════
    # 2. Architecture
    # ═════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section('2', 'Architecture')
    pdf.body(
        'The system is distributed across three boards connected via '
        'Ethernet (ROS 2 DDS). Each board runs its own set of ROS 2 nodes.'
    )
    pdf.code_block(
        'Jetson Orin Nano (ROS 2 Humble)\n'
        '  +-- farm_brain         (orchestrator, this node)\n'
        '  +-- robot_brain        (LLM text generation)\n'
        '  +-- llm_node           (Gemini / Llama 3.2)\n'
        '  +-- stt_node           (Faster-Whisper GPU)\n'
        '  +-- yolov8_ros         (YOLOv8m TensorRT + plant disease)\n'
        '  +-- realsense2_camera  (RGB-D)\n'
        '  +-- ldlidar_stl_ros2   (LD-19 LiDAR)\n'
        '  +-- slam_toolbox       (SLAM / localization)\n'
        '\n'
        'Raspberry Pi 5 (ROS 2 Jazzy)\n'
        '  +-- tts_elevenlabs     (ElevenLabs TTS)\n'
        '  +-- llm_display        (robot face + dashboard)\n'
        '  +-- speaker_monitor    (audio state)\n'
        '  +-- bno055_imu         (9-DOF IMU)\n'
        '\n'
        'Raspberry Pi 4 (ROS 2 Humble)\n'
        '  +-- mecanum_driver     (4x motor control)\n'
        '  +-- motor_control      (encoders, odometry)\n'
        '  +-- soil_moisture      (Arduino serial)\n'
        '  +-- rain_sensor        (Arduino serial)\n'
        '  +-- bmp180_pressure    (I2C)\n'
        '  +-- gps_gt_u7          (serial NMEA)\n'
        '  +-- dht11_node         (GPIO)          [placeholder]\n'
        '  +-- ph_sensor_node     (Arduino ADC)   [placeholder]\n'
        '  +-- uv_sensor_node     (I2C / analog)  [placeholder]\n'
        '  +-- lux_sensor_node    (I2C)           [placeholder]\n'
        '  +-- gas_sensor_node    (Arduino ADC)   [placeholder]\n'
    )

    # 2.1 State Machine
    pdf.section('2.1', 'State Machine', level=2)
    pdf.body(
        'The Farm Brain operates as a finite state machine with the '
        'following states:'
    )
    pdf.table(
        ['State', 'Description'],
        [
            ['STARTUP', 'Wait for sensors to come online (~10s)'],
            ['IDLE', 'Between patrols; check if patrol interval elapsed'],
            ['PATROLLING', 'Iterating through patrol waypoint list'],
            ['NAVIGATING', 'Travelling to a waypoint via Nav2 or fallback'],
            ['DEPLOYING_PROBES', 'Extending linear actuators into soil'],
            ['MEASURING', 'Waiting for sensor readings to stabilise'],
            ['INSPECTING', 'Vision scan for plant diseases'],
            ['ANALYSING', 'Evaluating sensor data against thresholds'],
            ['WATERING', 'Pump activated for dry soil'],
            ['SOWING', 'Seed dispenser activated'],
            ['RETRACTING_PROBES', 'Pulling actuators out of soil'],
            ['RETURNING_HOME', 'Navigating back to home/charger'],
            ['PAUSED', 'Human-requested pause (voice override)'],
            ['EMERGENCY_STOP', 'Obstacle too close or fault detected'],
            ['ERROR', 'Recoverable error state'],
        ],
        col_widths=[42, 148],
    )

    pdf.body(
        'Primary autonomous flow:\n'
        'STARTUP -> IDLE -> PATROLLING -> NAVIGATING -> DEPLOYING_PROBES '
        '-> MEASURING -> INSPECTING -> RETRACTING_PROBES -> ANALYSING '
        '-> [WATERING | SOWING] -> (next waypoint) -> ... -> '
        'RETURNING_HOME -> IDLE'
    )

    # 2.2 Navigation Stack
    pdf.section('2.2', 'Navigation Stack', level=2)
    pdf.body(
        'Navigation uses a layered approach:\n\n'
        'Layer 1 -- Nav2 (primary): ROS 2 Navigation2 stack provides '
        'global path planning (NavFn / Smac) and local trajectory '
        'planning (DWB / MPPI) using the SLAM-generated costmap. The '
        'farm brain sends NavigateToPose goals via the Nav2 action server.\n\n'
        'Layer 2 -- Isaac Sim RL Policy: A reinforcement learning policy '
        'trained in NVIDIA Isaac Sim provides learned obstacle avoidance '
        'and locomotion. The policy takes a downsampled LiDAR scan (36 bins), '
        'IMU data (6 DOF), and relative goal position (3) as input, and '
        'outputs linear and angular velocity commands. This runs as a '
        'local planner complement or fallback.\n\n'
        'Layer 3 -- Reactive safety: A 10 Hz obstacle check monitors '
        'LiDAR minimum range and RealSense forward depth. If an obstacle '
        'is closer than 0.30 m, an emergency stop is triggered. Between '
        '0.30-0.80 m, speed is reduced.'
    )

    # 2.3 Isaac Sim Integration
    pdf.section('2.3', 'Isaac Sim Integration', level=2)
    pdf.body(
        'The Isaac Sim integration follows a sim-to-real transfer pipeline:\n\n'
        '1. Train an RL agent in Isaac Sim using the Omniverse environment '
        '(e.g. Isaac Gym / Orbit). The agent learns to navigate to goals '
        'while avoiding obstacles using LiDAR + IMU observations.\n\n'
        '2. Export the trained policy as ONNX or TensorRT engine.\n\n'
        '3. Load the engine in farm_brain.py via _load_isaac_model().\n\n'
        '4. At runtime, _build_observation() constructs the observation '
        'vector from live sensor data, and _query_isaac_policy() runs '
        'inference to produce velocity commands.\n\n'
        'Observation vector structure (42 dimensions):\n'
        '  [lidar_36_bins, accel_x, accel_y, accel_z, '
        'rel_goal_x, rel_goal_y, goal_dist]\n\n'
        'Action vector (2 dimensions):\n'
        '  [linear_velocity_x, angular_velocity_z]'
    )

    # 2.4 Obstacle Avoidance
    pdf.section('2.4', 'Obstacle Avoidance', level=2)
    pdf.body(
        'Obstacle avoidance runs continuously at 10 Hz (_obstacle_check) '
        'regardless of the autonomous state. It monitors two sources:\n\n'
        '  - LiDAR (LD-19): 360-degree 2D scan, minimum valid range extracted\n'
        '  - RealSense D435: Forward depth minimum (published as Float32)\n\n'
        'The nearest obstacle distance triggers three behaviours:\n'
        '  1. > 0.80 m: normal speed\n'
        '  2. 0.30 - 0.80 m: reduced speed (Isaac Sim policy can fine-tune)\n'
        '  3. < 0.30 m: EMERGENCY STOP -- zero velocity, alert published\n\n'
        'When the obstacle clears, the robot automatically resumes navigation.'
    )

    # ═════════════════════════════════════════════════════════════════
    # 3. Sensor Suite
    # ═════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section('3', 'Sensor Suite (~16 Sensors)')
    pdf.body(
        'The farm brain subscribes to all sensor topics and aggregates '
        'them into a SensorSnapshot dataclass. Sensor data is published '
        'as a JSON summary on /farm_brain/sensor_summary every 10 seconds.'
    )
    pdf.table(
        ['#', 'Sensor', 'Interface', 'ROS Topics', 'Data'],
        [
            ['1', 'Soil Moisture', 'Arduino Serial', '/soil_moisture/*', 'Moisture %, raw ADC, dry bool'],
            ['2', 'pH Sensor', 'Arduino ADC', '/ph_sensor/*', 'pH value (0-14), raw ADC'],
            ['3', 'BMP180', 'I2C (RPi)', '/bmp180/*', 'Temp (C), Pressure (Pa), Alt (m)'],
            ['4', 'DHT11', 'GPIO (RPi)', '/dht11/*', 'Temperature (C), Humidity (%)'],
            ['5', 'BNO055 IMU', 'I2C (RPi)', '/imu/data', 'Quaternion, Accel, Gyro, Mag'],
            ['6', 'UV Sensor', 'I2C / Analog', '/uv_sensor/*', 'UV index, raw ADC'],
            ['7', 'Lux Sensor', 'I2C', '/lux_sensor/lux', 'Ambient light (lux)'],
            ['8', 'Gas / CO2 (MQ-135)', 'Arduino ADC', '/gas_sensor/*', 'CO2 ppm, raw ADC'],
            ['9', 'Raindrop Sensor', 'Arduino Serial', '/rain_sensor/*', 'Intensity %, raining bool'],
            ['10', 'GPS (GT-U7)', 'Serial NMEA', '/gps/fix', 'Lat, Lon, Alt, Fix status'],
            ['11', 'Encoder FL', 'GPIO IRQ', '/encoders/fl_rpm', 'RPM (600 PPR)'],
            ['12', 'Encoder FR', 'GPIO IRQ', '/encoders/fr_rpm', 'RPM (600 PPR)'],
            ['13', 'Encoder RL', 'GPIO IRQ', '/encoders/rl_rpm', 'RPM (600 PPR)'],
            ['14', 'Encoder RR', 'GPIO IRQ', '/encoders/rr_rpm', 'RPM (600 PPR)'],
            ['15', 'LiDAR (LD-19)', 'Serial', '/scan', '360 deg, ~4000 pts, 10 Hz'],
            ['16', 'RealSense D435', 'USB', '/camera/depth/*', 'RGB 640x480, Depth 848x480'],
        ],
        col_widths=[8, 30, 24, 38, 90],
    )

    # ═════════════════════════════════════════════════════════════════
    # 4. Actuators
    # ═════════════════════════════════════════════════════════════════
    pdf.section('4', 'Actuators')

    pdf.section('4.1', 'Linear Actuators (Soil Probes)', level=2)
    pdf.body(
        'Two linear actuators are used to insert soil moisture and pH '
        'probes into the ground at each waypoint. The actuators are '
        'controlled via Int32 commands on dedicated topics:\n\n'
        '  /actuators/linear_actuator_left   (+1=extend, -1=retract, 0=stop)\n'
        '  /actuators/linear_actuator_right  (+1=extend, -1=retract, 0=stop)\n\n'
        'The deploy/retract durations are configurable parameters '
        '(default 3 seconds each). On emergency stop or node shutdown, '
        'probes are automatically retracted for safety.'
    )

    pdf.section('4.2', 'Water Pump & Seed Dispenser', level=2)
    pdf.body(
        'Water pump:      /actuators/water_pump     (Bool: true=on, false=off)\n'
        'Seed dispenser:  /actuators/seed_dispenser  (Bool: true=on, false=off)\n\n'
        'The water pump activates for water_duration_sec (default 8s) when '
        'soil moisture falls below thresh_soil_dry_pct (default 30%) and it '
        'is not raining. The seed dispenser activates for sow_duration_sec '
        '(default 4s) when requested via voice override or when autonomous '
        'sowing logic is triggered.'
    )
    pdf.table(
        ['Actuator', 'Topic', 'Message Type', 'Values'],
        [
            ['Linear Act. Left', '/actuators/linear_actuator_left', 'std_msgs/Int32', '+1=extend, -1=retract, 0=stop'],
            ['Linear Act. Right', '/actuators/linear_actuator_right', 'std_msgs/Int32', '+1=extend, -1=retract, 0=stop'],
            ['Water Pump', '/actuators/water_pump', 'std_msgs/Bool', 'true=on, false=off'],
            ['Seed Dispenser', '/actuators/seed_dispenser', 'std_msgs/Bool', 'true=on, false=off'],
            ['Mecanum Motors', '/cmd_vel', 'geometry_msgs/Twist', 'linear.x, angular.z'],
        ],
        col_widths=[30, 52, 32, 76],
    )

    # ═════════════════════════════════════════════════════════════════
    # 5. ROS 2 Node Details
    # ═════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section('5', 'ROS 2 Node Details')
    pdf.body('Node name: farm_brain\nPackage: robot_brain\nEntry point: farm_brain = robot_brain.farm_brain:main')

    pdf.section('5.1', 'Subscribed Topics (~35 subscriptions)', level=2)
    pdf.table(
        ['Topic', 'Type', 'Source'],
        [
            ['/soil_moisture/moisture', 'Float32', 'RPi4 soil_moisture node'],
            ['/soil_moisture/dry', 'Bool', 'RPi4 soil_moisture node'],
            ['/soil_moisture/raw', 'Int32', 'RPi4 soil_moisture node'],
            ['/ph_sensor/ph', 'Float32', 'RPi4 ph_sensor node'],
            ['/ph_sensor/raw', 'Int32', 'RPi4 ph_sensor node'],
            ['/bmp180/temperature', 'Temperature', 'RPi4 bmp180 node'],
            ['/bmp180/pressure', 'FluidPressure', 'RPi4 bmp180 node'],
            ['/bmp180/altitude', 'Float64', 'RPi4 bmp180 node'],
            ['/dht11/temperature', 'Float32', 'RPi4 dht11 node'],
            ['/dht11/humidity', 'Float32', 'RPi4 dht11 node'],
            ['/imu/data', 'Imu', 'RPi5 bno055 node'],
            ['/uv_sensor/index', 'Float32', 'RPi4 uv_sensor node'],
            ['/uv_sensor/raw', 'Int32', 'RPi4 uv_sensor node'],
            ['/lux_sensor/lux', 'Float32', 'RPi4 lux_sensor node'],
            ['/gas_sensor/co2_ppm', 'Float32', 'RPi4 gas_sensor node'],
            ['/gas_sensor/raw', 'Int32', 'RPi4 gas_sensor node'],
            ['/rain_sensor/intensity', 'Float32', 'RPi4 rain_sensor node'],
            ['/rain_sensor/raining', 'Bool', 'RPi4 rain_sensor node'],
            ['/rain_sensor/raw', 'Int32', 'RPi4 rain_sensor node'],
            ['/gps/fix', 'NavSatFix', 'RPi4 gps_gt_u7 node'],
            ['/encoders/fl_rpm', 'Float32', 'RPi4 motor_control node'],
            ['/encoders/fr_rpm', 'Float32', 'RPi4 motor_control node'],
            ['/encoders/rl_rpm', 'Float32', 'RPi4 motor_control node'],
            ['/encoders/rr_rpm', 'Float32', 'RPi4 motor_control node'],
            ['/odom', 'Odometry', 'RPi4 motor_control node'],
            ['/scan', 'LaserScan', 'Jetson LiDAR driver'],
            ['/camera/depth/min_forward', 'Float32', 'Jetson depth processor'],
            ['/detection/objects_simple', 'String', 'Jetson yolov8_ros'],
            ['/detection/disease_simple', 'String', 'Jetson yolov8_ros'],
            ['/speech/text', 'String', 'Jetson stt_node (voice)'],
            ['/farm_brain/intent', 'String', 'Jetson LLM (parsed JSON)'],
        ],
        col_widths=[52, 26, 112],
    )

    pdf.section('5.2', 'Published Topics', level=2)
    pdf.table(
        ['Topic', 'Type', 'Rate', 'Description'],
        [
            ['/cmd_vel', 'Twist', '10 Hz', 'Velocity commands (fallback nav)'],
            ['/tts_text', 'String', 'Event', 'Spoken announcements to TTS'],
            ['/farm_brain/status', 'String', '0.33 Hz', 'JSON: state, waypoint, progress'],
            ['/farm_brain/sensor_summary', 'String', '0.1 Hz', 'JSON: all sensor readings'],
            ['/farm_brain/alerts', 'String', 'Event', 'JSON: type, message, waypoint'],
            ['/farm_brain/measurement', 'String', 'Event', 'JSON: per-waypoint measurement'],
            ['/actuators/water_pump', 'Bool', 'Event', 'Pump on/off'],
            ['/actuators/seed_dispenser', 'Bool', 'Event', 'Seeder on/off'],
            ['/actuators/linear_actuator_left', 'Int32', 'Event', 'Probe extend/retract/stop'],
            ['/actuators/linear_actuator_right', 'Int32', 'Event', 'Probe extend/retract/stop'],
        ],
        col_widths=[52, 18, 16, 104],
    )

    pdf.section('5.3', 'Parameters', level=2)
    pdf.table(
        ['Parameter', 'Default', 'Description'],
        [
            ['auto_start', 'true', 'Begin autonomous patrol on startup'],
            ['nav2_enabled', 'true', 'Use Nav2 for navigation'],
            ['obstacle_avoidance_enabled', 'true', 'Reactive obstacle layer'],
            ['isaac_model_path', "''", 'Path to TensorRT/ONNX RL policy'],
            ['farm_locations_file', "''", 'JSON file with locations + patrol order'],
            ['patrol_interval_sec', '300.0', 'Seconds between patrol cycles'],
            ['probe_deploy_time_sec', '3.0', 'Linear actuator extend time'],
            ['probe_retract_time_sec', '3.0', 'Linear actuator retract time'],
            ['measure_settle_time_sec', '5.0', 'Sensor stabilisation wait'],
            ['water_duration_sec', '8.0', 'Pump activation time per waypoint'],
            ['sow_duration_sec', '4.0', 'Seed dispenser activation time'],
            ['inspect_duration_sec', '5.0', 'Vision scan duration'],
            ['thresh_soil_dry_pct', '30.0', 'Soil moisture % trigger for watering'],
            ['thresh_ph_low', '5.5', 'Acidic soil alert threshold'],
            ['thresh_ph_high', '8.0', 'Alkaline soil alert threshold'],
            ['thresh_co2_high_ppm', '1000.0', 'CO2 ppm alert threshold'],
            ['thresh_uv_high_index', '8.0', 'UV index alert threshold'],
            ['thresh_temp_high_c', '40.0', 'Heat warning (Celsius)'],
            ['thresh_temp_low_c', '5.0', 'Frost warning (Celsius)'],
            ['thresh_obstacle_stop_m', '0.30', 'Emergency stop distance'],
            ['thresh_obstacle_slow_m', '0.80', 'Slow-down distance'],
            ['thresh_disease_conf_min', '0.60', 'Min disease detection confidence'],
            ['log_file', 'farm_brain_log.json', 'Patrol log output file'],
        ],
        col_widths=[52, 35, 103],
    )

    # ═════════════════════════════════════════════════════════════════
    # 6. Autonomous Decision Engine
    # ═════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section('6', 'Autonomous Decision Engine')
    pdf.body(
        'At each patrol waypoint, after measuring, the brain evaluates '
        'sensor data against configurable thresholds and takes automatic '
        'actions:'
    )
    pdf.table(
        ['Condition', 'Threshold', 'Action'],
        [
            ['Soil moisture < thresh', '< 30%', 'Activate water pump (unless raining)'],
            ['pH too low', '< 5.5', 'Publish acidic soil alert'],
            ['pH too high', '> 8.0', 'Publish alkaline soil alert'],
            ['CO2 elevated', '> 1000 ppm', 'Publish CO2 alert'],
            ['Temperature high', '> 40 C', 'Publish heat warning'],
            ['Temperature low', '< 5 C', 'Publish frost warning'],
            ['UV index high', '> 8.0', 'Publish UV warning'],
            ['Disease detected', 'conf > 60%', 'Log disease + publish alert'],
            ['Obstacle close', '< 0.30 m', 'Emergency stop (continuous 10Hz)'],
        ],
        col_widths=[40, 30, 120],
    )
    pdf.body(
        'All alerts are published on /farm_brain/alerts as JSON with '
        'fields: type, message, timestamp, waypoint. Measurements at '
        'each waypoint are logged to disk and published on '
        '/farm_brain/measurement.'
    )

    # ═════════════════════════════════════════════════════════════════
    # 7. Voice Override
    # ═════════════════════════════════════════════════════════════════
    pdf.section('7', 'Voice Override (Secondary)')
    pdf.body(
        'Voice commands are a secondary override channel. The robot does '
        'not require voice input to operate. When a human speaks, the STT '
        'node publishes text on /speech/text, and the farm brain parses it '
        'as a simple override:'
    )
    pdf.table(
        ['Voice Command', 'Action', 'Effect'],
        [
            ['"Stop" / "Halt"', 'stop', 'Emergency stop, cancel all motion'],
            ['"Pause" / "Wait"', 'pause', 'Pause autonomous loop'],
            ['"Resume" / "Continue"', 'resume', 'Resume from paused state'],
            ['"Skip" / "Next"', 'skip', 'Skip current waypoint, advance'],
            ['"Report" / "Status"', 'report', 'Speak full sensor summary via TTS'],
            ['"Go to row 1"', 'go_to', 'Navigate to named location'],
            ['"Water"', 'water', 'Force watering at current position'],
            ['"Sow" / "Plant"', 'sow', 'Force sowing at current position'],
            ['"Home" / "Return"', 'go_to home', 'Navigate to home position'],
        ],
        col_widths=[45, 25, 120],
    )

    # ═════════════════════════════════════════════════════════════════
    # 8. Waypoint Sequence
    # ═════════════════════════════════════════════════════════════════
    pdf.section('8', 'Waypoint Sequence')
    pdf.body(
        'At each patrol waypoint, the farm brain executes the following '
        'sequence in a dedicated thread:'
    )
    pdf.code_block(
        '1. DEPLOYING_PROBES   -- Extend linear actuators into soil\n'
        '                         Wait probe_deploy_time_sec (3s)\n'
        '\n'
        '2. MEASURING          -- Wait measure_settle_time_sec (5s)\n'
        '                         Take SensorSnapshot (all ~16 sensors)\n'
        '\n'
        '3. INSPECTING         -- Wait inspect_duration_sec (5s)\n'
        '                         YOLOv8 + plant disease classifier scan\n'
        '\n'
        '4. RETRACTING_PROBES  -- Retract linear actuators\n'
        '                         Wait probe_retract_time_sec (3s)\n'
        '\n'
        '5. ANALYSING          -- Compare readings against thresholds\n'
        '                         Trigger actions (water, alerts)\n'
        '\n'
        '6. [WATERING]         -- If soil dry: pump for water_duration_sec\n'
        '\n'
        '7. LOG                -- Save WaypointMeasurement to patrol_log\n'
        '                         Publish on /farm_brain/measurement\n'
        '\n'
        '8. ADVANCE            -- Move patrol_index to next waypoint'
    )

    # ═════════════════════════════════════════════════════════════════
    # 9. Configuration
    # ═════════════════════════════════════════════════════════════════
    pdf.section('9', 'Configuration')
    pdf.body(
        'Farm layout is defined in robot_brain/config/farm_locations.json. '
        'Each location has x, y, yaw (map frame coordinates from SLAM) '
        'and a type (base, crop, utility). The patrol_order array defines '
        'the sequence of waypoints to visit.'
    )
    pdf.code_block(
        '{\n'
        '  "locations": {\n'
        '    "home":           {"x": 0.0, "y": 0.0, "yaw": 0.0, "type": "base"},\n'
        '    "row_1":          {"x": 2.0, "y": 0.0, "yaw": 0.0, "type": "crop"},\n'
        '    "tomato_section": {"x": 4.0, "y": 0.0, "yaw": 0.0, "type": "crop"}\n'
        '  },\n'
        '  "patrol_order": ["row_1", "row_2", "tomato_section"]\n'
        '}'
    )
    pdf.body(
        'All thresholds and timing parameters are configurable via '
        'robot_brain/config/farm_brain_config.yaml or launch arguments.'
    )

    # ═════════════════════════════════════════════════════════════════
    # 10. Launch
    # ═════════════════════════════════════════════════════════════════
    pdf.section('10', 'Launch')
    pdf.code_block(
        '# Full pipeline (camera + YOLO + STT + LLM + brain)\n'
        'ros2 launch robot_brain farm_brain.launch.py\n'
        '\n'
        '# Brain node only (sensors already running)\n'
        'ros2 run robot_brain farm_brain\n'
        '\n'
        '# With Isaac Sim model\n'
        'ros2 launch robot_brain farm_brain.launch.py \\\n'
        '    isaac_model_path:=/path/to/policy.engine\n'
        '\n'
        '# Custom farm layout\n'
        'ros2 launch robot_brain farm_brain.launch.py \\\n'
        '    farm_locations_file:=/path/to/locations.json'
    )

    # ═════════════════════════════════════════════════════════════════
    # 11. Data Logging
    # ═════════════════════════════════════════════════════════════════
    pdf.section('11', 'Data Logging')
    pdf.body(
        'After each complete patrol cycle, all waypoint measurements are '
        'appended to a JSON log file (default: ~/farm_brain_log.json). '
        'Each entry contains:\n\n'
        '  - Patrol timestamp\n'
        '  - Per-waypoint: soil moisture, pH, temp, humidity, CO2, UV, lux, '
        'rain, GPS, detected diseases, and action taken\n\n'
        'This data can be used for trend analysis, ML training, or '
        'farm management dashboards.'
    )

    # ═════════════════════════════════════════════════════════════════
    # 12. All ROS 2 Nodes
    # ═════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section('12', 'All ROS 2 Nodes in the System')
    pdf.body('Complete list of ROS 2 nodes across all three platforms:')

    pdf.section('12.1', 'Jetson Orin Nano', level=2)
    pdf.table(
        ['Node', 'Package', 'Purpose'],
        [
            ['farm_brain', 'robot_brain', 'Autonomous farm orchestrator (this doc)'],
            ['robot_brain', 'robot_brain', 'LLM text generation (Llama 3.2)'],
            ['llm_node', 'llm_node', 'LLM intent parsing (Gemini / Llama)'],
            ['stt_node', 'stt_node', 'Speech-to-text (Faster-Whisper GPU)'],
            ['detection_node', 'yolov8_ros', 'YOLOv8m TensorRT + plant disease'],
            ['camera', 'realsense2_camera', 'Intel RealSense D435 RGB-D'],
            ['lidar', 'ldlidar_stl_ros2', 'LD-19 2D LiDAR driver'],
            ['slam_toolbox', 'slam_toolbox', 'SLAM / localization'],
            ['robot_state_publisher', 'robot_state_publisher', 'TF from URDF'],
        ],
        col_widths=[38, 42, 110],
    )

    pdf.section('12.2', 'Raspberry Pi 5', level=2)
    pdf.table(
        ['Node', 'Package', 'Purpose'],
        [
            ['tts_elevenlabs_node', 'tts_elevenlabs', 'Text-to-speech (ElevenLabs)'],
            ['robot_face_display', 'llm_display', 'Animated robot face (PyQt5)'],
            ['dashboard_display', 'llm_display', 'Sensor dashboard display'],
            ['audio_monitor_node', 'speaker_monitor', 'Audio playback state'],
            ['bno055_imu_node', 'bno055_imu', '9-DOF IMU driver (I2C)'],
        ],
        col_widths=[38, 42, 110],
    )

    pdf.section('12.3', 'Raspberry Pi 4', level=2)
    pdf.table(
        ['Node', 'Package', 'Purpose'],
        [
            ['mecanum_driver', 'mecanum_driver', '4x mecanum wheel control (serial)'],
            ['motor_node', 'motor_control', 'GPIO motor + encoder driver'],
            ['encoder_node', 'motor_control', '4x rotary encoder reader'],
            ['soil_moisture_node', 'soil_moisture', 'Soil moisture (Arduino serial)'],
            ['rain_sensor_node', 'rain_sensor', 'Raindrop sensor (Arduino serial)'],
            ['bmp180_node', 'bmp180_pressure', 'BMP180 pressure/temp (I2C)'],
            ['gps_gt_u7_node', 'gps_gt_u7', 'GPS module (serial NMEA)'],
            ['dht11_node', 'dht11 [placeholder]', 'DHT11 temp/humidity (GPIO)'],
            ['ph_sensor_node', 'ph_sensor [placeholder]', 'pH sensor (Arduino ADC)'],
            ['uv_sensor_node', 'uv_sensor [placeholder]', 'UV sensor (I2C/analog)'],
            ['lux_sensor_node', 'lux_sensor [placeholder]', 'Ambient light (I2C)'],
            ['gas_sensor_node', 'gas_sensor [placeholder]', 'MQ-135 CO2 (Arduino ADC)'],
        ],
        col_widths=[38, 42, 110],
    )

    pdf.body(
        '\nNodes marked [placeholder] have topic interfaces defined in '
        'the farm brain but their ROS 2 packages have not yet been '
        'implemented. The topic names and message types are standardised '
        'so that when the nodes are created, the farm brain will '
        'automatically pick up their data.'
    )

    # ── Save ──────────────────────────────────────────────────────────
    pdf.output(OUTPUT_PATH)
    print(f'PDF generated: {OUTPUT_PATH}')


if __name__ == '__main__':
    build_pdf()
