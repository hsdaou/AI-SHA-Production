#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_SIM="${AISHA_ISAAC_SIM_APP:-/home/robot-wst/isaacsim/isaac-sim.sh}"
cd "$PACKAGE_ROOT"

"$ISAAC_SIM" "$(realpath scenes/phase7k_phototextured_photogrammetric_survey.usda)"
