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

python3 tools/validate_phase7l_nurec_gaussian_twin.py
TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/render_phase7m_nurec_reel.py \
  --shot-plan config/phase7m_nurec_presentation_reel.yaml \
  --output-dir tmp/phase7m_final_frames \
  --report results/phase7m_nurec_reel_render.json \
  --width 1920 --height 1080 --warmup 2 --headless
python3 tools/encode_phase7m_nurec_reel.py
python3 tools/review_phase7m_nurec_reel.py
python3 tools/validate_phase7m_nurec_presentation.py
