#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR=$(mktemp -d -t aisha-nav2-phase7e-XXXXXX)
BRIDGE_PID=""
NAV2_PID=""

cleanup() {
  trap - EXIT INT TERM
  set +e
  if [[ -n "$NAV2_PID" ]]; then
    kill -INT -- "-$NAV2_PID" 2>/dev/null
  fi
  if [[ -n "$BRIDGE_PID" ]]; then
    kill -INT -- "-$BRIDGE_PID" 2>/dev/null
  fi
  wait 2>/dev/null
  [[ -d "$RUNTIME_DIR" ]] && rm -r -- "$RUNTIME_DIR"
}
trap cleanup EXIT INT TERM

setsid "$SCRIPT_DIR/run_administration_nav2_phase7e_static_fusion_bridge.sh" \
  --headless >"$RUNTIME_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!

sleep 15
if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
  tail -100 "$RUNTIME_DIR/bridge.log"
  echo "AI-SHA Phase 7E bridge exited before Nav2 startup" >&2
  exit 1
fi

setsid "$SCRIPT_DIR/run_administration_nav2_phase7e_static_fusion_servers.sh" \
  >"$RUNTIME_DIR/nav2.log" 2>&1 &
NAV2_PID=$!

sleep 25
if ! kill -0 "$NAV2_PID" 2>/dev/null; then
  tail -100 "$RUNTIME_DIR/nav2.log"
  echo "AI-SHA Phase 7E Nav2/filter stack exited before mission startup" >&2
  exit 1
fi

if ! "$SCRIPT_DIR/run_administration_nav2_phase7e_static_fusion_mission.sh"; then
  tail -120 "$RUNTIME_DIR/bridge.log"
  tail -120 "$RUNTIME_DIR/nav2.log"
  exit 1
fi

wait "$BRIDGE_PID"
BRIDGE_PID=""

# Finish the server group before validation so the filter's shutdown report is
# durably written and available as gate evidence.
kill -INT -- "-$NAV2_PID" 2>/dev/null || true
wait "$NAV2_PID" 2>/dev/null || true
NAV2_PID=""

python3 "$SCRIPT_DIR/validate_administration_nav2_phase7e_static_fusion_integration.py"
