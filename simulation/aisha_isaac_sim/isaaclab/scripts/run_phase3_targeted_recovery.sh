#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
BOOTSTRAP_RUN="${PHASE3M_BOOTSTRAP_RUN:-phase3m_corrected_physics_bootstrap_model200}"
BOOTSTRAP_CHECKPOINT="${PHASE3M_BOOTSTRAP_CHECKPOINT:-model_0.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${BOOTSTRAP_RUN}/${BOOTSTRAP_CHECKPOINT}"
CHECKSUM_FILE="${CHECKPOINT%/*}/checkpoint.sha256"
SOURCE_CHECKPOINT="${ISAACLAB_PROJECT}/checkpoints/aisha_phase3l_clearance_planner_model_200.pt"
SOURCE_SHA256="f58951c6e0f3bc7129ff479d925dc75472eca88f0dfec75c5436df12259d53e7"
FROZEN_ROUTE_CHECKPOINT="${ISAACLAB_PROJECT}/checkpoints/aisha_phase3_frozen_route_model_2225.pt"
FROZEN_ROUTE_SHA256="52f0094674dea901b4b7f3d7717bc9c2b014a6dc2d8e22cca768f783f4a9c0c8"
PPO_SEED="${PHASE3M_SEED:-9807}"
PPO_ENVS="${PHASE3M_ENVS:-32}"
PPO_ITERATIONS="${PHASE3M_ITERATIONS:-150}"
PPO_RUN_NAME="${PHASE3M_RUN_NAME:-phase3m_corrected_physics_recovery_seed9807}"

if [[ ! -f "${CHECKPOINT}" || ! -f "${CHECKSUM_FILE}" ]]; then
  if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "PHASE3M_PREFLIGHT_FAILED missing source checkpoint: ${SOURCE_CHECKPOINT}" >&2
    exit 91
  fi
  mkdir -p "${CHECKPOINT%/*}"
  python3 "${SCRIPT_DIR}/prepare_phase3m_recovery_checkpoint.py" \
    --source "${SOURCE_CHECKPOINT}" \
    --output "${CHECKPOINT}" \
    --expected-sha256 "${SOURCE_SHA256}" \
    --brake-std 0.10 \
    --steering-std 0.20 \
    --learning-rate 1.0e-5 \
    --report "${ISAACLAB_PROJECT}/../results/phase3m_corrected_physics_bootstrap_report.json"
fi
expected_bootstrap_sha256="$(awk '{print $1}' "${CHECKSUM_FILE}")"
actual_bootstrap_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_bootstrap_sha256}" != "${expected_bootstrap_sha256}" ]]; then
  echo "PHASE3M_PREFLIGHT_FAILED bootstrap hash mismatch: ${actual_bootstrap_sha256}" >&2
  exit 92
fi
if [[ ! -f "${FROZEN_ROUTE_CHECKPOINT}" ]]; then
  echo "PHASE3M_PREFLIGHT_FAILED missing frozen route checkpoint: ${FROZEN_ROUTE_CHECKPOINT}" >&2
  exit 93
fi
actual_route_sha256="$(sha256sum "${FROZEN_ROUTE_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_route_sha256}" != "${FROZEN_ROUTE_SHA256}" ]]; then
  echo "PHASE3M_PREFLIGHT_FAILED route checkpoint hash mismatch: ${actual_route_sha256}" >&2
  exit 94
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE3M_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 95
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase3-TargetedRecoveryTraining-SensorNav-Direct-v0 \
  --num_envs "${PPO_ENVS}" \
  --max_iterations "${PPO_ITERATIONS}" \
  --seed "${PPO_SEED}" \
  --resume \
  --load_run "${BOOTSTRAP_RUN}" \
  --checkpoint "${BOOTSTRAP_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PPO_RUN_NAME}" \
  --headless
