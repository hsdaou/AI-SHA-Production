#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
RESUME_RUN="${LIVE_FINETUNE_RESUME_RUN:-2026-08-22_09-12-57_phase2_moving_transition_seed256}"
RESUME_CHECKPOINT="${LIVE_FINETUNE_RESUME_CHECKPOINT:-model_1850.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${RESUME_RUN}/${RESUME_CHECKPOINT}"
EXPECTED_SHA256="${LIVE_FINETUNE_EXPECTED_SHA256:-3ab596c61259784657b36fe4ee937da8495ce9621a43bd04e2c8a0bf6e0b1880}"
LIVE_FINETUNE_SEED="${LIVE_FINETUNE_SEED:-7091}"
LIVE_FINETUNE_ENVS="${LIVE_FINETUNE_ENVS:-8}"
LIVE_FINETUNE_ITERATIONS="${LIVE_FINETUNE_ITERATIONS:-400}"
LIVE_FINETUNE_RUN_NAME="${LIVE_FINETUNE_RUN_NAME:-phase2b_administration_live_seed7091}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "LIVE_FINETUNE_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 51
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "LIVE_FINETUNE_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 52
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "LIVE_FINETUNE_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 53
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-Administration-Live-FineTune-Direct-v0 \
  --num_envs "${LIVE_FINETUNE_ENVS}" \
  --max_iterations "${LIVE_FINETUNE_ITERATIONS}" \
  --seed "${LIVE_FINETUNE_SEED}" \
  --resume \
  --load_run "${RESUME_RUN}" \
  --checkpoint "${RESUME_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${LIVE_FINETUNE_RUN_NAME}" \
  --headless
