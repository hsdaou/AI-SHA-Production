#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR=$(mktemp -d -t aisha-nav2-phase7d-XXXXXX)
BRIDGE_PID=""
NAV2_PID=""

cleanup() {
  trap - EXIT INT TERM
  set +e
  if [[ -n "$NAV2_PID" ]] && kill -0 "$NAV2_PID" 2>/dev/null; then
    kill -INT "$NAV2_PID" 2>/dev/null
    for _ in $(seq 1 50); do
      kill -0 "$NAV2_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -0 "$NAV2_PID" 2>/dev/null && kill -TERM "$NAV2_PID" 2>/dev/null
  fi
  if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    kill -INT "$BRIDGE_PID" 2>/dev/null
    for _ in $(seq 1 50); do
      kill -0 "$BRIDGE_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -0 "$BRIDGE_PID" 2>/dev/null && kill -TERM "$BRIDGE_PID" 2>/dev/null
  fi
  wait 2>/dev/null
  if [[ -d "$RUNTIME_DIR" ]]; then
    rm -r -- "$RUNTIME_DIR"
  fi
}
trap cleanup EXIT INT TERM

"$SCRIPT_DIR/run_administration_nav2_phase7d_native_costmap_bridge.sh" \
  --headless >"$RUNTIME_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!

sleep 15
if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
  tail -100 "$RUNTIME_DIR/bridge.log"
  echo "AI-SHA Phase 7D bridge exited before Nav2 startup" >&2
  exit 1
fi

"$SCRIPT_DIR/run_administration_nav2_phase7d_native_costmap_servers.sh" \
  >"$RUNTIME_DIR/nav2.log" 2>&1 &
NAV2_PID=$!

sleep 25
if ! kill -0 "$NAV2_PID" 2>/dev/null; then
  tail -100 "$RUNTIME_DIR/nav2.log"
  echo "AI-SHA Phase 7D Nav2 servers exited before mission startup" >&2
  exit 1
fi

if ! "$SCRIPT_DIR/run_administration_nav2_phase7d_native_costmap_mission.sh"; then
  tail -120 "$RUNTIME_DIR/bridge.log"
  tail -120 "$RUNTIME_DIR/nav2.log"
  exit 1
fi

wait "$BRIDGE_PID"
BRIDGE_PID=""

python3 "$SCRIPT_DIR/validate_administration_nav2_phase7d_native_costmap_integration.py"
