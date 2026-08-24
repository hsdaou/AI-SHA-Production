#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

exec "$SCRIPT_DIR/run_administration_nav2_servers.sh" \
  params_file:="$SIM_ROOT/config/nav2_phase7c_native_detour_params.yaml" \
  map:="$SIM_ROOT/maps/phase7c_native_detour_loop/phase7c_native_detour_loop.yaml" \
  maximum_linear_mps:=0.45 \
  amcl_tf_broadcast:=false \
  "$@"
