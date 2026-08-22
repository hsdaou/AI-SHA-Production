#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
RESUME_RUN="${PHASE3_RESUME_RUN:-2026-08-22_09-12-57_phase2_moving_transition_seed256}"
RESUME_CHECKPOINT="${PHASE3_RESUME_CHECKPOINT:-model_1850.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${RESUME_RUN}/${RESUME_CHECKPOINT}"
EXPECTED_SHA256="${PHASE3_EXPECTED_SHA256:-3ab596c61259784657b36fe4ee937da8495ce9621a43bd04e2c8a0bf6e0b1880}"
PHASE3_SEED="${PHASE3_SEED:-8701}"
PHASE3_ENVS="${PHASE3_ENVS:-32}"
PHASE3_ITERATIONS="${PHASE3_ITERATIONS:-600}"
PHASE3_RUN_NAME="${PHASE3_RUN_NAME:-phase3h_reciprocal_yield_seed8701}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "PHASE3_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 61
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "PHASE3_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 62
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE3_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 63
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase3-DynamicDR-SensorNav-Direct-v0 \
  --num_envs "${PHASE3_ENVS}" \
  --max_iterations "${PHASE3_ITERATIONS}" \
  --seed "${PHASE3_SEED}" \
  --resume \
  --load_run "${RESUME_RUN}" \
  --checkpoint "${RESUME_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PHASE3_RUN_NAME}" \
  --headless
