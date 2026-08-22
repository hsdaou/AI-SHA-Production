#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
RESUME_RUN="${REHEARSAL_RESUME_RUN:-2026-08-22_11-35-30_phase2b_administration_exploration_seed7092}"
RESUME_CHECKPOINT="${REHEARSAL_RESUME_CHECKPOINT:-model_2150_rehearsal.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${RESUME_RUN}/${RESUME_CHECKPOINT}"
EXPECTED_SHA256="${REHEARSAL_EXPECTED_SHA256:-e4c072c61a8f8f65c58c9a4780600c8b81ce713cc546d2eae8f96374980b0a0f}"
REHEARSAL_SEED="${REHEARSAL_SEED:-7093}"
REHEARSAL_ENVS="${REHEARSAL_ENVS:-32}"
REHEARSAL_ITERATIONS="${REHEARSAL_ITERATIONS:-100}"
REHEARSAL_RUN_NAME="${REHEARSAL_RUN_NAME:-phase2c_proxy_rehearsal_seed7093}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "REHEARSAL_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 61
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "REHEARSAL_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 62
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "REHEARSAL_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 63
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase2-Rehearsal-SensorNav-Direct-v0 \
  --num_envs "${REHEARSAL_ENVS}" \
  --max_iterations "${REHEARSAL_ITERATIONS}" \
  --seed "${REHEARSAL_SEED}" \
  --resume \
  --load_run "${RESUME_RUN}" \
  --checkpoint "${RESUME_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${REHEARSAL_RUN_NAME}" \
  --headless
