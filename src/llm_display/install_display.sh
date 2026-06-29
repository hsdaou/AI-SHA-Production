#!/bin/bash
# Installation script for LLM Display

set -e

echo "================================"
echo "LLM Display Installation Script"
echo "================================"
echo ""

# Check if running on Raspberry Pi
echo "[1/5] Checking system..."
if [[ $(uname -m) != "aarch64" ]]; then
    echo "Warning: Not running on ARM64 architecture"
fi

# Install dependencies
echo "[2/5] Installing dependencies..."
sudo apt-get update
sudo apt-get install -y unclutter python3-pyqt5 python3-pyqt5.qtsvg

# Build the package
echo "[3/5] Building ROS2 package..."
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select llm_display

# Source the package
echo "[4/5] Sourcing workspace..."
source install/setup.bash

# Make scripts executable
echo "[5/5] Setting permissions..."
chmod +x ~/start_llm_display.sh

echo ""
echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""
echo "To test the display:"
echo "  ros2 run llm_display llm_display"
echo ""
echo "To enable auto-start on boot:"
echo "  sudo cp ~/llm-display.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable llm-display.service"
echo "  sudo systemctl start llm-display.service"
echo ""
echo "To test with sample messages:"
echo "  ./test_display.sh"
echo ""
