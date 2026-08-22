#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/phase2/model_N.pt" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE_ROOT="$(cd "${ISAACLAB_PROJECT}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
CHECKPOINT="$(realpath "$1")"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Phase 2 checkpoint not found: ${CHECKPOINT}" >&2
  exit 3
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 43
fi

cd "${ISAACLAB_PROJECT}"

env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/evaluate.py \
  --task Isaac-AISHA-BlockA-Phase2-Turn-SensorNav-Direct-v0 \
  --checkpoint "${CHECKPOINT}" \
  --output "${BUNDLE_ROOT}/results/phase2_turn_held_out_evaluation.json" \
  --episodes-per-segment 48 --num_envs 96 --seed 6085 \
  --require-acceptance --headless

env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/evaluate.py \
  --task Isaac-AISHA-BlockA-Phase2-EndToEnd-SensorNav-Direct-v0 \
  --checkpoint "${CHECKPOINT}" \
  --output "${BUNDLE_ROOT}/results/phase2_policy_only_route_evaluation.json" \
  --episodes 48 --num_envs 24 --seed 7084 --require-acceptance --headless

env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/play_block_a_route.py \
  --task Isaac-AISHA-BlockA-Phase2-EndToEnd-SensorNav-Direct-v0 \
  --route-control policy-only \
  --checkpoint "${CHECKPOINT}" \
  --output-report "${BUNDLE_ROOT}/results/phase2_policy_only_training_route_report.json" \
  --max-steps 7200 --dwell-seconds 0 --trace-interval 3 \
  --seed 7084 --headless

env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/play_block_a_route.py \
  --task Isaac-AISHA-Administration-Live-Direct-v0 \
  --route-control policy-only \
  --checkpoint "${CHECKPOINT}" \
  --output-report "${BUNDLE_ROOT}/results/phase2_administration_policy_only_video_report.json" \
  --video-folder "${BUNDLE_ROOT}/media/videos/phase2_administration_policy_only" \
  --max-steps 7200 --dwell-seconds 0 --trace-interval 3 \
  --camera-eye -3.8 0.0 2.4 --camera-lookat 0.45 0.0 0.55 \
  --seed 7084 --headless

python3 tools/make_administration_live_policy_presentation_video.py \
  --input "${BUNDLE_ROOT}/media/videos/phase2_administration_policy_only/aisha-block-a-learned-route-step-0.mp4" \
  --run-report "${BUNDLE_ROOT}/results/phase2_administration_policy_only_video_report.json" \
  --output "${BUNDLE_ROOT}/media/videos/AI-SHA_Phase2_Administration_Policy_Only_3x.mp4" \
  --report "${BUNDLE_ROOT}/results/phase2_administration_policy_only_presentation_report.json"

python3 tools/validate_phase2_end_to_end.py
