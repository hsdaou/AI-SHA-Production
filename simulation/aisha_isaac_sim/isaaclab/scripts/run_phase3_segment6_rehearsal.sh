#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
RESUME_RUN="${PHASE3C_RESUME_RUN:-2026-08-22_14-02-08_phase3b_staged_dynamic_dr_seed8101}"
RESUME_CHECKPOINT="${PHASE3C_RESUME_CHECKPOINT:-model_2525.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${RESUME_RUN}/${RESUME_CHECKPOINT}"
EXPECTED_SHA256="${PHASE3C_EXPECTED_SHA256:-4014edcb7ff8b4664fbd6805865127cdbb64ab60671a15682e8c07ee45b70a4e}"
PHASE3C_SEED="${PHASE3C_SEED:-8201}"
PHASE3C_ENVS="${PHASE3C_ENVS:-32}"
PHASE3C_ITERATIONS="${PHASE3C_ITERATIONS:-300}"
PHASE3C_RUN_NAME="${PHASE3C_RUN_NAME:-phase3c_segment6_retention_seed8201}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "PHASE3C_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 71
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "PHASE3C_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 72
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE3C_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 73
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase3-Segment6Rehearsal-SensorNav-Direct-v0 \
  --num_envs "${PHASE3C_ENVS}" \
  --max_iterations "${PHASE3C_ITERATIONS}" \
  --seed "${PHASE3C_SEED}" \
  --resume \
  --load_run "${RESUME_RUN}" \
  --checkpoint "${RESUME_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PHASE3C_RUN_NAME}" \
  --headless
