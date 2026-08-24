#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MAP_YAML="$SIM_ROOT/maps/administration_measured_presentation_1cm/administration_measured_presentation_1cm.yaml"
RUNTIME_PROFILE=$(mktemp -t aisha-phase7e-nav2-profile-XXXXXX.yaml)
FILTER_PID=""
NAV2_PID=""

cleanup() {
  trap - EXIT INT TERM
  set +e
  if [[ -n "$NAV2_PID" ]] && kill -0 "$NAV2_PID" 2>/dev/null; then
    kill -INT "$NAV2_PID" 2>/dev/null
  fi
  if [[ -n "$FILTER_PID" ]] && kill -0 "$FILTER_PID" 2>/dev/null; then
    kill -INT "$FILTER_PID" 2>/dev/null
  fi
  wait 2>/dev/null
  [[ -f "$RUNTIME_PROFILE" ]] && rm -- "$RUNTIME_PROFILE"
}
trap cleanup EXIT INT TERM

python3 "$SCRIPT_DIR/build_administration_static_fused_nav2_profile.py" \
  --output "$RUNTIME_PROFILE"

source "$SCRIPT_DIR/setup_nav2_user_overlay.sh"
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

python3 "$SCRIPT_DIR/filter_administration_static_scan_returns.py" \
  --map-yaml "$MAP_YAML" \
  --report "${AISHA_PHASE7E_FILTER_REPORT:-$SIM_ROOT/results/administration_nav2_phase7e_static_scan_fusion.json}" &
FILTER_PID=$!

"$SCRIPT_DIR/run_administration_nav2_servers.sh" \
  params_file:="$RUNTIME_PROFILE" \
  map:="$MAP_YAML" \
  maximum_linear_mps:=0.80 \
  amcl_tf_broadcast:=false \
  "$@" &
NAV2_PID=$!

wait "$NAV2_PID"
NAV2_STATUS=$?
NAV2_PID=""
exit "$NAV2_STATUS"
