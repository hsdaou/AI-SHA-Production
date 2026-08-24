#!/usr/bin/env bash
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "${tool_dir}/.." && pwd)"
repo_root="$(cd "${package_root}/../.." && pwd)"

PYTHONPATH="${repo_root}/src/aisha_rev_d_driver${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 "${tool_dir}/validate_phase8b_rev_d_adapter.py" "$@"

PYTHONPATH="${repo_root}/src/aisha_rev_d_driver${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -m pytest -q "${repo_root}/src/aisha_rev_d_driver/test"

echo "Phase 8B offline replay passed. RS485 writes, wheel motion and physical release remain blocked."
