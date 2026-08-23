#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ISAACLAB_PROJECT=$(cd -- "$SCRIPT_DIR/.." && pwd)
ISAACLAB_ROOT=${ISAACLAB_ROOT:-/home/robot-wst/IsaacLab}
EXPERIMENT_NAME=aisha_block_a_sensor_nav
STAGE=${1:-65}
if [[ $# -gt 0 ]]; then
  shift
fi

PPO_SEED=${PHASE6_SEED:-10701}
PPO_ENVS=${PHASE6_ENVS:-32}
ACCEPTED_RUN=${PHASE6_ACCEPTED_RUN:-2026-08-22_21-35-00_phase3n_brake_only_social_safety_seed10017}
ACCEPTED_CHECKPOINT=${PHASE6_ACCEPTED_CHECKPOINT:-model_50.pt}
ACCEPTED_SHA256=11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b

case "$STAGE" in
  65|065|0.65)
    TASK=Isaac-AISHA-BlockA-Phase6-HighSpeed65-DynamicSafety-SensorNav-Direct-v0
    LOAD_RUN=$ACCEPTED_RUN
    LOAD_CHECKPOINT=$ACCEPTED_CHECKPOINT
    PPO_ITERATIONS=${PHASE6_ITERATIONS:-75}
    RUN_NAME=${PHASE6_RUN_NAME:-phase6_high_speed_065_seed10701}
    ;;
  80|080|0.80)
    TASK=Isaac-AISHA-BlockA-Phase6-HighSpeed80-DynamicSafety-SensorNav-Direct-v0
    LOAD_RUN=${PHASE6_LOAD_RUN:-.*phase6_high_speed_065_seed10701}
    LOAD_CHECKPOINT=${PHASE6_LOAD_CHECKPOINT:-model_.*.pt}
    PPO_ITERATIONS=${PHASE6_ITERATIONS:-100}
    RUN_NAME=${PHASE6_RUN_NAME:-phase6_high_speed_080_seed10701}
    ;;
  *)
    echo "usage: $0 {65|80}" >&2
    exit 64
    ;;
esac

if ! nvidia-smi >/dev/null 2>&1; then
  echo "PHASE6_PREFLIGHT_FAILED NVIDIA driver unavailable on kernel $(uname -r)" >&2
  exit 94
fi

if [[ "$STAGE" == "65" || "$STAGE" == "065" || "$STAGE" == "0.65" ]]; then
  INPUT_CHECKPOINT="$ISAACLAB_PROJECT/logs/rsl_rl/$EXPERIMENT_NAME/$LOAD_RUN/$LOAD_CHECKPOINT"
  if [[ ! -f "$INPUT_CHECKPOINT" ]]; then
    echo "PHASE6_PREFLIGHT_FAILED missing accepted input: $INPUT_CHECKPOINT" >&2
    exit 92
  fi
  ACTUAL_SHA256=$(sha256sum "$INPUT_CHECKPOINT" | awk '{print $1}')
  if [[ "$ACTUAL_SHA256" != "$ACCEPTED_SHA256" ]]; then
    echo "PHASE6_PREFLIGHT_FAILED accepted checkpoint hash mismatch: $ACTUAL_SHA256" >&2
    exit 93
  fi
fi

cd "$ISAACLAB_PROJECT"
exec env TERM=xterm "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/launch.py train \
  --task "$TASK" \
  --num_envs "$PPO_ENVS" \
  --max_iterations "$PPO_ITERATIONS" \
  --seed "$PPO_SEED" \
  --resume \
  --load_run "$LOAD_RUN" \
  --checkpoint "$LOAD_CHECKPOINT" \
  --experiment_name "$EXPERIMENT_NAME" \
  --run_name "$RUN_NAME" \
  --headless \
  "$@"
