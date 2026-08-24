#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_LAUNCHER="${AISHA_ISAACLAB_LAUNCHER:-/home/robot-wst/IsaacLab/isaaclab.sh}"
cd "$PACKAGE_ROOT"

TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/play_phase7g_presentation_live.py \
  --scene scenes/phase7j_complete_captured_administration.usda \
  --camera-profile config/phase7j_complete_captured_administration_twin.yaml \
  --trajectory-report results/administration_nav2_phase7e_static_fusion_mission.json \
  --session-label PHASE7J --repeat-count 0
