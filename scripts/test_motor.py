#!/usr/bin/env python3
"""Direct GPIO test - bypasses ROS2 and PID entirely."""
import pigpio
import time

pi = pigpio.pi()
if not pi.connected:
    print("ERROR: pigpiod not running! Run: sudo pigpiod")
    exit(1)

# Motor 1 pins
DIR_A = 22
DIR_B = 24
PWM_PIN = 13

print(f"Testing Motor 1: DIR_A=GPIO{DIR_A}, DIR_B=GPIO{DIR_B}, PWM=GPIO{PWM_PIN}")

# Setup
pi.set_mode(DIR_A, pigpio.OUTPUT)
pi.set_mode(DIR_B, pigpio.OUTPUT)
pi.set_mode(PWM_PIN, pigpio.OUTPUT)

# Step 1: Test direction pins
print("\n--- Step 1: Setting DIR_A=HIGH, DIR_B=LOW (forward) ---")
pi.write(DIR_A, 1)
pi.write(DIR_B, 0)
print(f"  DIR_A (GPIO{DIR_A}) = {pi.read(DIR_A)}")
print(f"  DIR_B (GPIO{DIR_B}) = {pi.read(DIR_B)}")
print("  -> Check: does the driver LED turn on?")
time.sleep(3)

# Step 2: Test PWM at 50% duty
print("\n--- Step 2: Setting PWM to 50% duty at 1kHz ---")
pi.set_PWM_frequency(PWM_PIN, 1000)
pi.set_PWM_range(PWM_PIN, 255)
pi.set_PWM_dutycycle(PWM_PIN, 128)  # 50%
actual_freq = pi.get_PWM_frequency(PWM_PIN)
print(f"  PWM frequency: {actual_freq} Hz")
print(f"  PWM duty: 50%")
print("  -> Check: does the motor spin?")
time.sleep(5)

# Step 3: Try 100% duty
print("\n--- Step 3: Setting PWM to 100% duty ---")
pi.set_PWM_dutycycle(PWM_PIN, 255)
print("  PWM duty: 100%")
print("  -> Check: does the motor spin now?")
time.sleep(5)

# Step 4: Try opposite direction
print("\n--- Step 4: Reverse direction (DIR_A=LOW, DIR_B=HIGH) ---")
pi.write(DIR_A, 0)
pi.write(DIR_B, 1)
time.sleep(3)

# Cleanup
print("\n--- Stopping ---")
pi.set_PWM_dutycycle(PWM_PIN, 0)
pi.write(DIR_A, 0)
pi.write(DIR_B, 0)
pi.stop()
print("Done. All pins off.")
