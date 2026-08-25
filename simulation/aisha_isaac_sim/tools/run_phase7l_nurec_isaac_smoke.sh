#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_LAUNCHER="${AISHA_ISAACLAB_LAUNCHER:-/home/robot-wst/IsaacLab/isaaclab.sh}"
cd "$PACKAGE_ROOT"

for asset in \
  tmp/phase7l_nurec_runs/administration_full_nurec.usdz \
  tmp/phase7l_nurec_runs/principal_full_nurec.usdz; do
  [[ -f "$asset" ]] || { echo "missing local privacy-sensitive asset: $asset" >&2; exit 2; }
done

TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/build_phase7l_nurec_composite.py
TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/render_phase7l_nurec_smoke.py \
  --stage scenes/phase7l_nurec_registered_administration.usda \
  --camera /World/Presentation/NuRec/gauss/Cameras/camera_0 \
  --output-dir media/screenshots/phase7l_nurec_composite_smoke \
  --report results/phase7l_nurec_composite_isaac_render.json \
  --frames 0,40,80 --headless
python3 tools/validate_phase7l_nurec_gaussian_twin.py
