#!/usr/bin/env bash
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "${tool_dir}/.." && pwd)"

python3 "${tool_dir}/validate_phase8a_physical_localization_preflight.py"

if [[ "${1:-}" == "--probe-stationary-runtime" ]]; then
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
  fi
  set +e
  python3 "${tool_dir}/probe_phase8a_stationary_localization.py" \
    --output "${package_root}/results/phase8a_stationary_localization_probe.json"
  probe_status=$?
  set -e
  python3 "${tool_dir}/validate_phase8a_physical_localization_preflight.py"
  exit "${probe_status}"
fi

echo "Phase 8A offline preparation passed. Physical runtime and motion remain blocked."
