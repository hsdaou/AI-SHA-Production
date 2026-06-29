# BNO055 9-DOF IMU Complete Data Message
# All sensor data organized in one message

# Header with timestamp and frame_id
std_msgs/Header header

# === ORIENTATION (Quaternion) ===
# Fused orientation from sensor fusion
geometry_msgs/Quaternion orientation

# === ORIENTATION (Euler Angles in degrees) ===
# Heading, Roll, Pitch
float64 euler_heading      # 0 to 360 degrees
float64 euler_roll         # -180 to 180 degrees
float64 euler_pitch        # -90 to 90 degrees

# === LINEAR ACCELERATION (m/s²) ===
# 3-axis accelerometer data
geometry_msgs/Vector3 linear_acceleration

# === ANGULAR VELOCITY (rad/s) ===
# 3-axis gyroscope data
geometry_msgs/Vector3 angular_velocity

# === MAGNETIC FIELD (micro Tesla) ===
# 3-axis magnetometer data
geometry_msgs/Vector3 magnetic_field

# === TEMPERATURE (°C) ===
float64 temperature

# === CALIBRATION STATUS (0-3, where 3 is fully calibrated) ===
uint8 calibration_system        # Overall system calibration
uint8 calibration_gyro          # Gyroscope calibration
uint8 calibration_accel         # Accelerometer calibration
uint8 calibration_mag           # Magnetometer calibration
