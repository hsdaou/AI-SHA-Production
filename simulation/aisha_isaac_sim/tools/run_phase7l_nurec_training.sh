#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRUT_ROOT="${AISHA_3DGRUT_ROOT:-$PACKAGE_ROOT/tmp/phase7l_3dgrut}"
GRUT_PYTHON="${AISHA_3DGRUT_PYTHON:-$GRUT_ROOT/.venv/bin/python}"

if [[ "${AISHA_ALLOW_NUREC_RETRAIN:-0}" != "1" ]]; then
  echo "Phase 7L retraining is intentionally opt-in (about 10-20 minutes and several GB)."
  echo "Set AISHA_ALLOW_NUREC_RETRAIN=1 after reviewing local privacy/storage requirements."
  exit 2
fi
if [[ ! -x "$GRUT_PYTHON" || ! -f "$GRUT_ROOT/train.py" ]]; then
  echo "NVIDIA 3DGRUT and its prepared environment are required at: $GRUT_ROOT" >&2
  exit 2
fi

cd "$PACKAGE_ROOT"
python3 tools/prepare_phase7l_nurec_dataset.py
python3 tools/prepare_phase7l_nurec_dataset.py \
  --component 2 \
  --output tmp/phase7l_nurec_principal_dataset \
  --report results/phase7l_nurec_principal_dataset_preflight.json
python3 tools/prepare_phase7l_nurec_dataset.py \
  --component 6 \
  --output tmp/phase7l_nurec_principal_connector_dataset \
  --report results/phase7l_nurec_principal_connector_dataset_preflight.json

train_component() {
  local dataset="$1"
  local experiment="$2"
  local asset="$3"
  (
    cd "$GRUT_ROOT"
    "$GRUT_PYTHON" train.py \
      --config-name apps/colmap_3dgut.yaml \
      "path=$PACKAGE_ROOT/$dataset" \
      "out_dir=$PACKAGE_ROOT/tmp/phase7l_nurec_runs" \
      "experiment_name=$experiment" \
      dataset.downsample_factor=2 \
      n_iterations=30000 \
      'checkpoint.iterations=[7000,30000]' \
      test_last=true \
      compute_extra_metrics=true \
      export_usd.enabled=true \
      export_usd.format=nurec \
      "export_usd.path=$PACKAGE_ROOT/$asset" \
      export_usd.half_precision=true \
      num_workers=4
  )
}

train_component \
  tmp/phase7l_nurec_dataset \
  administration_full \
  tmp/phase7l_nurec_runs/administration_full_nurec.usdz
train_component \
  tmp/phase7l_nurec_principal_dataset \
  principal_full \
  tmp/phase7l_nurec_runs/principal_full_nurec.usdz

python3 tools/register_phase7l_nurec_components.py
tools/run_phase7l_nurec_isaac_smoke.sh
