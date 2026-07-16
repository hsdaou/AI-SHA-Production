#!/usr/bin/env bash
# Fetch the mediapipe Tasks API model bundles yolov8_node needs on the Jetson.
# Runs after CP1 on any Jetson bringing up the vision stack; also safe to re-run.
# Files go to <repo>/models/ which is .gitignored (models are artifacts, not code).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p models
FACE_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
for u in "$FACE_URL" "$HAND_URL"; do
  f="models/$(basename "$u")"
  echo "  fetching $(basename "$u") ..."
  wget --continue --tries=0 --timeout=60 --read-timeout=30 --waitretry=5 -q "$u" -O "$f"
done
ls -lh models/
