#!/bin/bash
pkill -9 -f 'aisha_watc[h]'            2>/dev/null || true
pkill -9 -f 'yolov8_nod[e]'            2>/dev/null || true
pkill -9 -f 'realsense2_camera_nod[e]' 2>/dev/null || true
pkill -9 -f 'jetson_launc[h]'          2>/dev/null || true
pkill -9 -f 'stt_nod[e]'            2>/dev/null || true
pkill -9 -f 'brain_node\|admin_node\|action_node\|gpu_arbiter' 2>/dev/null || true
echo "[console] stopped."
