#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_LAUNCHER="${AISHA_ISAACLAB_LAUNCHER:-/home/robot-wst/IsaacLab/isaaclab.sh}"
cd "$PACKAGE_ROOT"

python3 tools/generate_phase7k_capture_materials.py
python3 tools/prepare_phase7k_photogrammetry_assets.py
TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/build_phase7k_photogrammetry_layers.py
TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/build_phase7k_phototextured_survey.py

python3 tools/validate_administration_replay.py \
  --trajectory-report results/administration_nav2_phase7e_static_fusion_mission.json \
  --output results/administration_nav2_phase7k_replay_validation.json

TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/render_phase7k_phototextured_survey.py \
  --headless --renderer PathTracing --width 1920 --height 1080

TERM=xterm "$ISAACLAB_LAUNCHER" -p scripts/render_administration_route.py \
  --headless --width 1920 --height 1080 --fps 24 --seconds-per-shot 2.5 \
  --renderer PathTracing --path-tracing-spp 12 --exposure-bias 0.65 \
  --scene scenes/phase7k_phototextured_presentation.usda \
  --trajectory-report results/administration_nav2_phase7e_static_fusion_mission.json \
  --presentation-profile config/phase7k_phototextured_photogrammetric_survey.yaml \
  --frame-directory outputs/phase7k_phototextured_survey_rtx_frames \
  --render-report results/administration_nav2_phase7k_rtx_render_report.json

python3 tools/encode_route_video.py \
  --fps 24 --crf 18 --preset slow \
  --frames-dir outputs/phase7k_phototextured_survey_rtx_frames \
  --validation results/administration_nav2_phase7k_replay_validation.json \
  --output media/videos/AI-SHA_Phase7K_Phototextured_Photogrammetric_Survey.mp4 \
  --report results/administration_nav2_phase7k_rtx_render_report.json

python3 tools/make_route_contact_sheet.py \
  --render-report results/administration_nav2_phase7k_rtx_render_report.json \
  --frames-dir outputs/phase7k_phototextured_survey_rtx_frames \
  --output media/AI-SHA_Phase7K_Phototextured_Photogrammetric_Survey_contact_sheet.jpg

python3 tools/validate_phase7k_phototextured_survey.py
