#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

exec "$SCRIPT_DIR/run_administration_nav2_servers.sh" \
  params_file:="$SIM_ROOT/config/nav2_sim_tight_door_params.yaml" \
  map:="$SIM_ROOT/maps/administration_measured_presentation_1cm/administration_measured_presentation_1cm.yaml" \
  maximum_linear_mps:=0.80 \
  amcl_tf_broadcast:=false \
  "$@"
