#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# FastDDS environment setup for AI-SHA 3-device mesh
# (Jetson Orin Nano + Raspberry Pi 5 + optional dev Workstation)
#
# Source this file in each device's ~/.bashrc (or systemd EnvironmentFile)
# to ensure the generated FastDDS XML profile is loaded by ROS 2.
#
# Usage:
#   # In ~/.bashrc on the Jetson:
#   source /home/orin-robot/robot_ws/config/fastdds_env.sh jetson
#
#   # In ~/.bashrc on RPi 5:
#   source /home/pi/robot_ws/config/fastdds_env.sh rpi
#
#   # On dev workstation:
#   source /path/to/config/fastdds_env.sh ws
# ─────────────────────────────────────────────────────────────────────────────

DEVICE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$DEVICE" ]; then
    echo "[fastdds_env] ERROR: specify device name: jetson | rpi | ws"
    echo "  Usage: source $0 <device>"
    return 1 2>/dev/null || exit 1
fi

XML_FILE="$SCRIPT_DIR/fastdds_${DEVICE}.xml"

if [ ! -f "$XML_FILE" ]; then
    echo "[fastdds_env] ERROR: $XML_FILE not found."
    echo "  Run: bash scripts/generate_fastdds_configs.sh"
    return 1 2>/dev/null || exit 1
fi

# Force FastDDS as the RMW implementation (ROS 2 Humble default, but
# explicit is safer — prevents silent fallback to Cyclone if installed).
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Point FastDDS to the generated unicast discovery profile.
export FASTRTPS_DEFAULT_PROFILES_FILE="$XML_FILE"

# Domain ID must match what generate_fastdds_configs.sh used for port
# calculation.  Default: 42 (matches the generation script default).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

echo "[fastdds_env] Loaded: $DEVICE"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE"
echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
