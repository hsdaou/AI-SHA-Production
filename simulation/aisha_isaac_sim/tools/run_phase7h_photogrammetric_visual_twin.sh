#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_LAUNCHER="${AISHA_ISAACLAB_LAUNCHER:-/home/robot-wst/IsaacLab/isaaclab.sh}"

cd "$PACKAGE_ROOT"

python3 tools/validate_administration_replay.py \
  --trajectory-report results/administration_nav2_phase7e_static_fusion_mission.json \
  --output results/administration_nav2_phase7h_replay_validation.json

TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/render_administration_route.py \
  --headless --width 1920 --height 1080 --fps 24 --seconds-per-shot 3 \
  --renderer PathTracing --path-tracing-spp 16 --exposure-bias 0.85 \
  --trajectory-report results/administration_nav2_phase7e_static_fusion_mission.json \
  --presentation-profile config/phase7h_photogrammetric_visual_twin.yaml \
  --frame-directory outputs/phase7h_photogrammetric_rtx_frames \
  --render-report results/administration_nav2_phase7h_rtx_render_report.json

python3 tools/encode_route_video.py \
  --fps 24 --crf 18 --preset slow \
  --frames-dir outputs/phase7h_photogrammetric_rtx_frames \
  --validation results/administration_nav2_phase7h_replay_validation.json \
  --output media/videos/AI-SHA_Phase7H_Photogrammetry_Informed_Omniverse.mp4 \
  --report results/administration_nav2_phase7h_rtx_render_report.json

python3 tools/make_route_contact_sheet.py \
  --render-report results/administration_nav2_phase7h_rtx_render_report.json \
  --frames-dir outputs/phase7h_photogrammetric_rtx_frames \
  --output media/AI-SHA_Phase7H_Photogrammetry_Informed_Omniverse_contact_sheet.jpg

python3 tools/validate_phase7h_photogrammetric_visual_twin.py
