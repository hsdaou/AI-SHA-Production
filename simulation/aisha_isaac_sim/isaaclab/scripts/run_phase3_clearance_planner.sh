#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
BOOTSTRAP_RUN="${PHASE3L_BOOTSTRAP_RUN:-phase3l_clearance_planner_bootstrap_seed9601}"
BOOTSTRAP_CHECKPOINT="${PHASE3L_BOOTSTRAP_CHECKPOINT:-model_0.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${BOOTSTRAP_RUN}/${BOOTSTRAP_CHECKPOINT}"
CHECKSUM_FILE="${CHECKPOINT%/*}/checkpoint.sha256"
FROZEN_ROUTE_CHECKPOINT="${ISAACLAB_PROJECT}/checkpoints/aisha_phase3_frozen_route_model_2225.pt"
FROZEN_ROUTE_SHA256="52f0094674dea901b4b7f3d7717bc9c2b014a6dc2d8e22cca768f783f4a9c0c8"
PPO_SEED="${PHASE3L_SEED:-9601}"
PPO_ENVS="${PHASE3L_ENVS:-32}"
PPO_ITERATIONS="${PHASE3L_ITERATIONS:-600}"
PPO_RUN_NAME="${PHASE3L_RUN_NAME:-phase3l_clearance_planner_seed9601}"

if [[ ! -f "${CHECKPOINT}" || ! -f "${CHECKSUM_FILE}" ]]; then
  echo "PHASE3L_PREFLIGHT_FAILED missing bootstrap checkpoint or checksum: ${CHECKPOINT}" >&2
  exit 91
fi
expected_bootstrap_sha256="$(awk '{print $1}' "${CHECKSUM_FILE}")"
actual_bootstrap_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_bootstrap_sha256}" != "${expected_bootstrap_sha256}" ]]; then
  echo "PHASE3L_PREFLIGHT_FAILED bootstrap hash mismatch: ${actual_bootstrap_sha256}" >&2
  exit 92
fi
if [[ ! -f "${FROZEN_ROUTE_CHECKPOINT}" ]]; then
  echo "PHASE3L_PREFLIGHT_FAILED missing frozen route checkpoint: ${FROZEN_ROUTE_CHECKPOINT}" >&2
  exit 93
fi
actual_route_sha256="$(sha256sum "${FROZEN_ROUTE_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_route_sha256}" != "${FROZEN_ROUTE_SHA256}" ]]; then
  echo "PHASE3L_PREFLIGHT_FAILED route checkpoint hash mismatch: ${actual_route_sha256}" >&2
  exit 94
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE3L_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 95
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase3-ClearancePlanner-SensorNav-Direct-v0 \
  --num_envs "${PPO_ENVS}" \
  --max_iterations "${PPO_ITERATIONS}" \
  --seed "${PPO_SEED}" \
  --resume \
  --load_run "${BOOTSTRAP_RUN}" \
  --checkpoint "${BOOTSTRAP_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PPO_RUN_NAME}" \
  --headless
