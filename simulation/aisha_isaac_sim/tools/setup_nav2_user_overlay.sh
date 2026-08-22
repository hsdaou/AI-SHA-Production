#!/usr/bin/env bash
# Source this file from a system-ROS shell to use the sudo-free AI-SHA Nav2 overlay.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

AI_SHA_OVERLAY_ROOT=${AI_SHA_ROS_OVERLAY_ROOT:-/home/robot-wst/.local/share/ai_sha_ros_jazzy_overlay/root}
AI_SHA_ROS_PREFIX="$AI_SHA_OVERLAY_ROOT/opt/ros/jazzy"
if [[ ! -d "$AI_SHA_ROS_PREFIX/share/nav2_controller" ]]; then
  echo "AI-SHA Nav2 user overlay is missing at $AI_SHA_ROS_PREFIX" >&2
  return 2
fi

set +u
source /opt/ros/jazzy/setup.sh
set -u
export AMENT_PREFIX_PATH="$AI_SHA_ROS_PREFIX:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="$AI_SHA_ROS_PREFIX:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$AI_SHA_OVERLAY_ROOT/usr/lib:$AI_SHA_OVERLAY_ROOT/usr/lib/x86_64-linux-gnu:$AI_SHA_ROS_PREFIX/lib:$AI_SHA_ROS_PREFIX/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$AI_SHA_ROS_PREFIX/bin:${PATH:-}"
export PYTHONPATH="$AI_SHA_ROS_PREFIX/lib/python3.12/site-packages:${PYTHONPATH:-}"
