#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_LAUNCHER="${AISHA_ISAACLAB_LAUNCHER:-/home/robot-wst/IsaacLab/isaaclab.sh}"
cd "$PACKAGE_ROOT"

python3 tools/validate_phase7l_nurec_gaussian_twin.py
TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/play_phase7l_nurec_live.py \
  --scene scenes/phase7l_nurec_registered_administration.usda \
  --trajectory-report results/administration_nav2_phase7e_static_fusion_mission.json \
  --segments 6,7,8,9 --camera-start 0 --camera-end 105 \
  --repeat-count 0
