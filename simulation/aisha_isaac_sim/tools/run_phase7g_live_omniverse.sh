#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_LAUNCHER="${AISHA_ISAACLAB_LAUNCHER:-/home/robot-wst/IsaacLab/isaaclab.sh}"

cd "$PACKAGE_ROOT"
python3 tools/validate_phase7g_presentation_freeze.py
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/play_phase7g_presentation_live.py \
  --fps 24 \
  --seconds-per-shot 3 \
  --repeat-count 0 \
  --report tmp/phase7g_live_omniverse_session.json
