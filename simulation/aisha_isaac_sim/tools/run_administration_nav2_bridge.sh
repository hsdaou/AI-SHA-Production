#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim 5.1 embeds Python 3.11 and its own Jazzy rclpy. Ubuntu 24.04's
# system Jazzy uses Python 3.12, so sourcing /opt/ros/jazzy before starting
# Isaac causes an ABI mismatch. Start the bridge with NVIDIA's internal ROS
# environment; run Nav2 itself in a separate system-ROS terminal.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$SIM_ROOT/../.." && pwd)
ISAAC_SIM_INSTALL=${ISAAC_SIM_INSTALL:-/home/robot-wst/isaacsim}
ISAAC_LAB_INSTALL=${ISAAC_LAB_INSTALL:-/home/robot-wst/IsaacLab}

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION PYTHONPATH AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH LD_LIBRARY_PATH RMW_IMPLEMENTATION
set +u
source "$ISAAC_SIM_INSTALL/setup_ros_env.sh"
set -u

cd "$REPO_ROOT"
exec "$ISAAC_LAB_INSTALL/isaaclab.sh" -p \
  "$SIM_ROOT/isaaclab/scripts/run_administration_nav2_bridge.py" "$@"
