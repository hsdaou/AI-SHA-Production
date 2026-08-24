#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CHECKPOINT="$SIM_ROOT/isaaclab/checkpoints/aisha_phase6_high_speed_080_model_223.pt"

exec "$SCRIPT_DIR/run_administration_nav2_bridge.sh" \
  --task "Isaac-AISHA-Phase7C-NativeCostmap-Detour-Safety-Direct-v0" \
  --learned-safety-checkpoint "$CHECKPOINT" \
  --output-report "$SIM_ROOT/results/phase7c_native_costmap_detour_bridge.json" \
  "$@"
