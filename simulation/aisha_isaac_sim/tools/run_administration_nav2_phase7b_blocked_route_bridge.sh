#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CHECKPOINT="$SIM_ROOT/isaaclab/checkpoints/aisha_phase6_high_speed_080_model_223.pt"
FALLBACK_CHECKPOINT="$SIM_ROOT/isaaclab/checkpoints/aisha_phase3n_dynamic_safety_model_50.pt"

exec "$SCRIPT_DIR/run_administration_nav2_bridge.sh" \
  --task "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7B-BlockedRoute-Replanning-Safety-Direct-v0" \
  --learned-safety-checkpoint "$CHECKPOINT" \
  --fallback-learned-safety-checkpoint "$FALLBACK_CHECKPOINT" \
  --mapped-safety-overlay "$SIM_ROOT/config/measured_administration_presentation_2026-08-23.yaml" \
  --mapped-safety-site-config "$SIM_ROOT/config/administration_assumptions.yaml" \
  --output-report "$SIM_ROOT/results/administration_nav2_phase7b_blocked_route_bridge.json" \
  "$@"
