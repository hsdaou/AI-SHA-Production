#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
RESUME_RUN="${PRINCIPAL_TURN_RESUME_RUN:-2026-08-22_11-35-30_phase2b_administration_exploration_seed7092}"
RESUME_CHECKPOINT="${PRINCIPAL_TURN_RESUME_CHECKPOINT:-model_2150_rehearsal.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${RESUME_RUN}/${RESUME_CHECKPOINT}"
EXPECTED_SHA256="${PRINCIPAL_TURN_EXPECTED_SHA256:-e4c072c61a8f8f65c58c9a4780600c8b81ce713cc546d2eae8f96374980b0a0f}"
PRINCIPAL_TURN_SEED="${PRINCIPAL_TURN_SEED:-7096}"
PRINCIPAL_TURN_ENVS="${PRINCIPAL_TURN_ENVS:-8}"
PRINCIPAL_TURN_ITERATIONS="${PRINCIPAL_TURN_ITERATIONS:-100}"
PRINCIPAL_TURN_RUN_NAME="${PRINCIPAL_TURN_RUN_NAME:-phase2f_administration_principal_turn_seed7096}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "PRINCIPAL_TURN_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 81
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "PRINCIPAL_TURN_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 82
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "PRINCIPAL_TURN_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 83
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-Administration-Live-PrincipalTurn-Direct-v0 \
  --num_envs "${PRINCIPAL_TURN_ENVS}" \
  --max_iterations "${PRINCIPAL_TURN_ITERATIONS}" \
  --seed "${PRINCIPAL_TURN_SEED}" \
  --resume \
  --load_run "${RESUME_RUN}" \
  --checkpoint "${RESUME_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PRINCIPAL_TURN_RUN_NAME}" \
  --headless
