#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESUME_RUN="${PHASE2_RESUME_RUN:-2026-08-20_22-33-50_ld19_flush_threshold_v9}"
RESUME_CHECKPOINT="${PHASE2_RESUME_CHECKPOINT:-model_599.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${RESUME_RUN}/${RESUME_CHECKPOINT}"
EXPECTED_SHA256="${PHASE2_EXPECTED_SHA256:-3da826c515d0e58a3c0731dd3b72022208eadda3cc2bd2200a78d92f079bfacf}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
PHASE2_SEED="${PHASE2_SEED:-245}"
PHASE2_ITERATIONS="${PHASE2_ITERATIONS:-1200}"
PHASE2_RUN_NAME="${PHASE2_RUN_NAME:-phase2_transition_turn_seed245}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "PHASE2_PREFLIGHT_FAILED missing checkpoint: ${CHECKPOINT}" >&2
  exit 41
fi

actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "PHASE2_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 42
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE2_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  echo "Boot the known-good 6.17.0-35-generic kernel, then rerun this script." >&2
  exit 43
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase2-Turn-SensorNav-Direct-v0 \
  --num_envs 32 \
  --max_iterations "${PHASE2_ITERATIONS}" \
  --seed "${PHASE2_SEED}" \
  --resume \
  --load_run "${RESUME_RUN}" \
  --checkpoint "${RESUME_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PHASE2_RUN_NAME}" \
  --headless
