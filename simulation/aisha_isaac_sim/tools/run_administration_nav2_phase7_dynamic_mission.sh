#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/setup_nav2_user_overlay.sh"
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

exec python3 "$SCRIPT_DIR/run_administration_nav2_mission.py" \
  --site-profile measured_presentation \
  --control-stack nav2_mapped_doorway_phase7_dynamic_crossing_safety \
  --waypoint-timeout-s 180 \
  --output "$SIM_ROOT/results/administration_nav2_phase7_dynamic_mission.json" \
  "$@"
