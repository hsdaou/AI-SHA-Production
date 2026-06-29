# BNO055 9-DOF IMU ROS 2 Driver

ROS 2 driver for the Bosch BNO055 9-axis absolute orientation sensor.

## Features

- Publishes complete 9-DOF IMU data in a single organized message
- Includes orientation (quaternion and Euler angles)
- 3-axis accelerometer data
- 3-axis gyroscope data
- 3-axis magnetometer data
- Temperature measurement
- Real-time calibration status for all sensors

## Published Topics

- `/imu/data` (`bno055_imu/BNO055Data`) - Complete IMU data at 50 Hz (configurable)

## Message Structure

```
BNO055Data.msg:
├── header (timestamp and frame_id)
├── orientation (quaternion: w, x, y, z)
├── euler_heading, euler_roll, euler_pitch (degrees)
├── linear_acceleration (x, y, z in m/s²)
├── angular_velocity (x, y, z in rad/s)
├── magnetic_field (x, y, z in µT)
├── temperature (°C)
└── calibration_status (system, gyro, accel, mag: 0-3)
```

## Parameters

- `publish_rate` (default: 50.0 Hz) - Publishing frequency
- `frame_id` (default: "imu_link") - TF frame ID
- `i2c_address` (default: 0x28) - I2C address of BNO055

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip3 install adafruit-circuitpython-bno055
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select bno055_imu
source install/setup.bash
```

## Usage

### Run the node

```bash
ros2 run bno055_imu bno055_node
```

### Run with custom parameters

```bash
ros2 run bno055_imu bno055_node --ros-args \
  -p publish_rate:=100.0 \
  -p frame_id:=base_imu \
  -p i2c_address:=0x29
```

### View the data

```bash
# Echo all IMU data
ros2 topic echo /imu/data

# Check publishing rate
ros2 topic hz /imu/data

# View message structure
ros2 interface show bno055_imu/msg/BNO055Data
```

## Hardware Setup

Connect BNO055 to Raspberry Pi:

| BNO055 Pin | Raspberry Pi Pin | GPIO |
|------------|------------------|------|
| VIN | Pin 1 | 3.3V |
| GND | Pin 6 | GND |
| SDA | Pin 3 | GPIO 2 (I2C SDA) |
| SCL | Pin 5 | GPIO 3 (I2C SCL) |

## Calibration

The BNO055 requires calibration for optimal performance:

- **System**: 0-3 (overall calibration)
- **Gyroscope**: 0-3 (place sensor still)
- **Accelerometer**: 0-3 (slowly rotate through all axes)
- **Magnetometer**: 0-3 (move in figure-8 pattern)

Monitor calibration status in the published messages or node logs.

## Troubleshooting

### No I2C device found
```bash
# Check I2C connection
i2cdetect -y 1

# Should show device at 0x28 or 0x29
```

### Permission denied
```bash
# Add user to i2c group
sudo usermod -a -G i2c $USER
newgrp i2c
```

## License

MIT
