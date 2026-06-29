# mecanum_driver — ROS2 Humble + Arduino Uno

## Deploy to Pi4

```bash
# Copy the whole package into your workspace
scp -r mecanum_driver pi4@pi4.local:~/ros2_ws/src/

# SSH in and build
ssh pi4@pi4.local
pip3 install pyserial
sudo usermod -aG dialout pi4   # log out/in after this

cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mecanum_driver
source install/setup.bash
```

## Arduino Setup

1. Open `arduino/mecanum_motor_control/mecanum_motor_control.ino`
2. Update pin definitions to match your wiring
3. Upload to Arduino Uno
4. Test via Serial Monitor (115200): `M 100 100 100 100`

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
- Publishes: `/wheel_speeds` (std_msgs/Float32MultiArray)

## Serial Protocol

- Pi to Arduino: `M <fl> <fr> <rl> <rr>\n` or `S\n` (stop)
- Arduino to Pi: `OK <fl> <fr> <rl> <rr>\n` or `STOPPED\n`
