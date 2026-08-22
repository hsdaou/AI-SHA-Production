#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
BOOTSTRAP_RUN="${PHASE3_RECURRENT_BOOTSTRAP_RUN:-phase3_recurrent_bootstrap_seed8801}"
BOOTSTRAP_CHECKPOINT="${PHASE3_RECURRENT_BOOTSTRAP_CHECKPOINT:-model_0.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${BOOTSTRAP_RUN}/${BOOTSTRAP_CHECKPOINT}"
CHECKSUM_FILE="${CHECKPOINT%/*}/checkpoint.sha256"
PPO_SEED="${PHASE3_RECURRENT_PPO_SEED:-8901}"
PPO_ENVS="${PHASE3_RECURRENT_PPO_ENVS:-32}"
PPO_ITERATIONS="${PHASE3_RECURRENT_PPO_ITERATIONS:-800}"
PPO_RUN_NAME="${PHASE3_RECURRENT_PPO_RUN_NAME:-phase3j_recurrent_ppo_seed8901}"

if [[ ! -f "${CHECKPOINT}" || ! -f "${CHECKSUM_FILE}" ]]; then
  echo "PHASE3_RECURRENT_PPO_PREFLIGHT_FAILED missing bootstrap checkpoint or checksum: ${CHECKPOINT}" >&2
  exit 81
fi

expected_sha256="$(awk '{print $1}' "${CHECKSUM_FILE}")"
actual_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "PHASE3_RECURRENT_PPO_PREFLIGHT_FAILED checkpoint hash mismatch: ${actual_sha256}" >&2
  exit 82
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE3_RECURRENT_PPO_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 83
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase3-RecurrentPPO-SensorNav-Direct-v0 \
  --num_envs "${PPO_ENVS}" \
  --max_iterations "${PPO_ITERATIONS}" \
  --seed "${PPO_SEED}" \
  --resume \
  --load_run "${BOOTSTRAP_RUN}" \
  --checkpoint "${BOOTSTRAP_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PPO_RUN_NAME}" \
  --headless
