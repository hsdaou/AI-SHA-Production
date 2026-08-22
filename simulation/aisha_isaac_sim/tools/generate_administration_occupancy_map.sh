#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ISAAC_SIM_INSTALL=${ISAAC_SIM_INSTALL:-/home/robot-wst/isaacsim}
USD_LIBS=$(find "$ISAAC_SIM_INSTALL/extscache" -maxdepth 1 -type d -name 'omni.usd.libs-*' -print -quit)
if [[ -z "$USD_LIBS" ]]; then
  echo "Unable to locate the Isaac Sim USD runtime under $ISAAC_SIM_INSTALL/extscache" >&2
  exit 2
fi

export PYTHONPATH="$USD_LIBS:$ISAAC_SIM_INSTALL/exts/omni.pip.compute/pip_prebundle:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$USD_LIBS/bin:${LD_LIBRARY_PATH:-}"
exec "$ISAAC_SIM_INSTALL/kit/python/bin/python3" \
  "$SCRIPT_DIR/generate_administration_occupancy_map.py" "$@"
