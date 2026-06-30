# mecanum_driver — ROS2 Humble + Arduino Mega 2560 + BTS7960 + BNO055

Translates `/cmd_vel` into 4-wheel Mecanum PWM commands and forwards the
unified Arduino telemetry (encoders + BNO055 IMU) as ROS `nav_msgs/Odometry`
and `sensor_msgs/Imu`.  Runs on the **Raspberry Pi 5** in the Two-Tier
SBC + MCU layout — the dedicated Pi 4b motor SBC has been retired.

## Deploy to Raspberry Pi 5

```bash
# Copy the whole package into your workspace
scp -r mecanum_driver pi@pi5.local:~/ros2_ws/src/

# SSH in and build
ssh pi@pi5.local
pip3 install pyserial
sudo usermod -aG dialout pi   # log out/in after this

cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mecanum_driver
source install/setup.bash
```

## Arduino Setup

1. Open `arduino/mecanum_motor_control/mecanum_motor_control.ino`
2. Update pin definitions to match your wiring (BTS7960 motor pins and
   BNO055 I2C address — defaults to `0x28`).
3. Install the **Adafruit BNO055** + **Adafruit Unified Sensor** libraries
   via the Arduino Library Manager.
4. Upload to Arduino Mega 2560.
5. Test via Serial Monitor (115200): `M 100 100 100 100`.  At rest you
   should also see periodic `ODOM …` telemetry packets.

## Run

```bash
ros2 launch mecanum_driver mecanum_driver.launch.py

# Test with teleop (new terminal):
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Configure

Edit `config/mecanum_params.yaml` with your robot's actual dimensions.

## Topics

- Subscribes: `/cmd_vel` (geometry_msgs/Twist)
- Publishes:
  - `/wheel_speeds` (std_msgs/Float32MultiArray, debug)
  - `/odom` (nav_msgs/Odometry, encoder fusion)
  - `/imu/data` (sensor_msgs/Imu, BNO055 via Arduino)

## Serial Protocol

- Pi → Arduino: `M <fl> <fr> <rl> <rr>\n` or `S\n` (stop) or `P\n` (ping)
- Arduino → Pi:
  - `OK <fl> <fr> <rl> <rr>\n` per-command ack
  - `STOPPED\n` after `S`, `PONG\n` after `P`
  - Periodic unified telemetry (≈20 Hz):
    `ODOM <fl_ticks> <fr_ticks> <rl_ticks> <rr_ticks> <qw> <qx> <qy> <qz> <gx> <gy> <gz> <ax> <ay> <az>\n`
    where the quaternion is the BNO055 fused orientation, the gyro vector
    is in deg/s (converted to rad/s on the Pi for `sensor_msgs/Imu`), and
    the acceleration vector is in m/s².
