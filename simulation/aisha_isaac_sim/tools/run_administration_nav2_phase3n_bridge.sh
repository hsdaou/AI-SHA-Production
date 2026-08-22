#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CHECKPOINT="$SIM_ROOT/isaaclab/checkpoints/aisha_phase3n_dynamic_safety_model_50.pt"

exec "$SCRIPT_DIR/run_administration_nav2_bridge.sh" \
  --phase3n-safety-checkpoint "$CHECKPOINT" \
  --output-report "$SIM_ROOT/results/administration_nav2_phase3n_bridge.json" \
  "$@"
