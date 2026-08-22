#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
TEACHER_RUN="${PHASE3_TEACHER_RUN:-2026-08-22_09-12-57_phase2_moving_transition_seed256}"
TEACHER_CHECKPOINT="${PHASE3_TEACHER_CHECKPOINT:-model_1850.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${TEACHER_RUN}/${TEACHER_CHECKPOINT}"
EXPECTED_SHA256="${PHASE3_TEACHER_SHA256:-3ab596c61259784657b36fe4ee937da8495ce9621a43bd04e2c8a0bf6e0b1880}"
DISTILL_SEED="${PHASE3_DISTILL_SEED:-8801}"
DISTILL_ENVS="${PHASE3_DISTILL_ENVS:-32}"
DISTILL_ITERATIONS="${PHASE3_DISTILL_ITERATIONS:-200}"
DISTILL_RUN_NAME="${PHASE3_DISTILL_RUN_NAME:-phase3i_recurrent_distillation_seed8801}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "PHASE3_RECURRENT_DISTILL_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 71
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "PHASE3_RECURRENT_DISTILL_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 72
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE3_RECURRENT_DISTILL_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 73
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase3-RecurrentDistill-SensorNav-Direct-v0 \
  --num_envs "${DISTILL_ENVS}" \
  --max_iterations "${DISTILL_ITERATIONS}" \
  --seed "${DISTILL_SEED}" \
  --load_run "${TEACHER_RUN}" \
  --checkpoint "${TEACHER_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${DISTILL_RUN_NAME}" \
  --headless
