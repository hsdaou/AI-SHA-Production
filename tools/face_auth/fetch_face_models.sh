#!/usr/bin/env bash
# Fetch the ArcFace recognition model (w600k_mbf.onnx) used by the face-auth skill.
# Source: InsightFace 'buffalo_s' model pack (v0.7 release). MobileFaceNet / w600k.
# The face DETECTOR (blaze_face_short_range.tflite) is fetched separately by
# scripts/fetch_mediapipe_task_models.sh (shared with the CP5/CP6 vision stack).
#
# Biometric MODELS are NOT committed to git. Run this once on a new machine.
set -euo pipefail

DATA_DIR="${FACE_AUTH_HOME:-$HOME/face_auth}"
MODELS_DIR="$DATA_DIR/models"
mkdir -p "$MODELS_DIR"

REC="$MODELS_DIR/w600k_mbf.onnx"
ZIP_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
ZIP="$MODELS_DIR/buffalo_s.zip"

if [[ -f "$REC" ]]; then
  echo "[fetch] recogniser already present: $REC"
  exit 0
fi

echo "[fetch] downloading buffalo_s pack ..."
wget --continue -O "$ZIP" "$ZIP_URL"
echo "[fetch] extracting w600k_mbf.onnx ..."
# the pack may unzip either flat or under a 'buffalo_s/' prefix
unzip -o "$ZIP" -d "$MODELS_DIR" >/dev/null
found="$(find "$MODELS_DIR" -name w600k_mbf.onnx | head -1)"
[[ -n "$found" ]] || { echo "[fetch] ERROR: w600k_mbf.onnx not found in pack"; exit 1; }
[[ "$found" == "$REC" ]] || cp "$found" "$REC"
echo "[fetch] OK -> $REC"
