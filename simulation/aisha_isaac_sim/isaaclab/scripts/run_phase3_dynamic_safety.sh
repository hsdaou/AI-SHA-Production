#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}"
BOOTSTRAP_RUN="${PHASE3N_BOOTSTRAP_RUN:-phase3n_brake_only_bootstrap_seed10017}"
BOOTSTRAP_CHECKPOINT="${PHASE3N_BOOTSTRAP_CHECKPOINT:-model_0.pt}"
CHECKPOINT="${ISAACLAB_PROJECT}/logs/rsl_rl/aisha_block_a_sensor_nav/${BOOTSTRAP_RUN}/${BOOTSTRAP_CHECKPOINT}"
CHECKSUM_FILE="${CHECKPOINT%/*}/checkpoint.sha256"
FROZEN_RECOVERY_CHECKPOINT="${ISAACLAB_PROJECT}/checkpoints/aisha_phase3m_hybrid_recovery_model_125.pt"
FROZEN_RECOVERY_SHA256="bc8727e3ea42c8b29ca74fa5a535fd37b1600633ffd8bf606b02220a557c1a0d"
FROZEN_ROUTE_CHECKPOINT="${ISAACLAB_PROJECT}/checkpoints/aisha_phase3_frozen_route_model_2225.pt"
FROZEN_ROUTE_SHA256="52f0094674dea901b4b7f3d7717bc9c2b014a6dc2d8e22cca768f783f4a9c0c8"
PPO_SEED="${PHASE3N_SEED:-10017}"
PPO_ENVS="${PHASE3N_ENVS:-32}"
PPO_ITERATIONS="${PHASE3N_ITERATIONS:-300}"
PPO_RUN_NAME="${PHASE3N_RUN_NAME:-phase3n_brake_only_social_safety_seed10017}"

if [[ ! -f "${CHECKPOINT}" || ! -f "${CHECKSUM_FILE}" ]]; then
  mkdir -p "${CHECKPOINT%/*}"
  env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p \
    "${SCRIPT_DIR}/bootstrap_safety_residual_ppo.py" \
    --action-count 1 \
    --output-checkpoint "${CHECKPOINT}" \
    --report "${ISAACLAB_PROJECT}/../results/phase3n_dynamic_safety_bootstrap_report.json"
fi

expected_bootstrap_sha256="$(awk '{print $1}' "${CHECKSUM_FILE}")"
actual_bootstrap_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_bootstrap_sha256}" != "${expected_bootstrap_sha256}" ]]; then
  echo "PHASE3N_PREFLIGHT_FAILED bootstrap hash mismatch: ${actual_bootstrap_sha256}" >&2
  exit 91
fi
for checkpoint_contract in \
  "${FROZEN_RECOVERY_CHECKPOINT}:${FROZEN_RECOVERY_SHA256}:recovery" \
  "${FROZEN_ROUTE_CHECKPOINT}:${FROZEN_ROUTE_SHA256}:route"
do
  checkpoint_path="${checkpoint_contract%%:*}"
  remainder="${checkpoint_contract#*:}"
  expected_sha256="${remainder%%:*}"
  label="${remainder##*:}"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "PHASE3N_PREFLIGHT_FAILED missing frozen ${label} checkpoint: ${checkpoint_path}" >&2
    exit 92
  fi
  actual_sha256="$(sha256sum "${checkpoint_path}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "PHASE3N_PREFLIGHT_FAILED frozen ${label} hash mismatch: ${actual_sha256}" >&2
    exit 93
  fi
done
if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE3N_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)." >&2
  exit 94
fi

cd "${ISAACLAB_PROJECT}"
exec env TERM=xterm "${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-Phase3-DynamicSafety-SensorNav-Direct-v0 \
  --num_envs "${PPO_ENVS}" \
  --max_iterations "${PPO_ITERATIONS}" \
  --seed "${PPO_SEED}" \
  --resume \
  --load_run "${BOOTSTRAP_RUN}" \
  --checkpoint "${BOOTSTRAP_CHECKPOINT}" \
  --experiment_name aisha_block_a_sensor_nav \
  --run_name "${PPO_RUN_NAME}" \
  --headless
