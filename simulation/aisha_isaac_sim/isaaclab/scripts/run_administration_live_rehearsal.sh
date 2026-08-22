#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
RESUME_RUN="${LIVE_REHEARSAL_RESUME_RUN:-2026-08-22_11-35-30_phase2b_administration_exploration_seed7092}"
RESUME_CHECKPOINT="${LIVE_REHEARSAL_RESUME_CHECKPOINT:-model_2150_rehearsal.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${RESUME_RUN}/${RESUME_CHECKPOINT}"
EXPECTED_SHA256="${LIVE_REHEARSAL_EXPECTED_SHA256:-e4c072c61a8f8f65c58c9a4780600c8b81ce713cc546d2eae8f96374980b0a0f}"
LIVE_REHEARSAL_SEED="${LIVE_REHEARSAL_SEED:-7094}"
LIVE_REHEARSAL_ENVS="${LIVE_REHEARSAL_ENVS:-8}"
LIVE_REHEARSAL_ITERATIONS="${LIVE_REHEARSAL_ITERATIONS:-100}"
LIVE_REHEARSAL_RUN_NAME="${LIVE_REHEARSAL_RUN_NAME:-phase2d_administration_balanced_rehearsal_seed7094}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "LIVE_REHEARSAL_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 71
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "LIVE_REHEARSAL_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 72
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "LIVE_REHEARSAL_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 73
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-Administration-Live-Rehearsal-Direct-v0 \
  --num_envs "${LIVE_REHEARSAL_ENVS}" \
  --max_iterations "${LIVE_REHEARSAL_ITERATIONS}" \
  --seed "${LIVE_REHEARSAL_SEED}" \
  --resume \
  --load_run "${RESUME_RUN}" \
  --checkpoint "${RESUME_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${LIVE_REHEARSAL_RUN_NAME}" \
  --headless
