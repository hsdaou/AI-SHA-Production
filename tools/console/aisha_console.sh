#!/bin/bash
# Bring up AI-SHA for the browser console: camera + vision + brain + admin (LLM/RAG),
# NO microphone (you type questions in the browser), NO GPU arbiter (camera stays live).
set -e
source /opt/ros/humble/setup.bash
source ~/robot_ws/install/setup.bash
export ROS_DOMAIN_ID=99
export PATH=$HOME/.local/bin:$PATH

echo "[console] clearing any stale nodes..."
pkill -9 -f 'yolov8_nod[e]'            2>/dev/null || true
pkill -9 -f 'realsense2_camera_nod[e]' 2>/dev/null || true
pkill -9 -f 'jetson_launc[h]'          2>/dev/null || true
pkill -9 -f 'aisha_watc[h]'            2>/dev/null || true
sleep 3

echo "[console] launching stack (camera + vision + brain + LLM, no mic)..."
setsid bash -c "source /opt/ros/humble/setup.bash; source ~/robot_ws/install/setup.bash; \
  export ROS_DOMAIN_ID=99; export PATH=\$HOME/.local/bin:\$PATH; \
  ros2 launch aisha_integration jetson_launch.py \
      enable_stt:=false enable_gpu_arbiter:=false llm_model:=aisha:1b \
  > /tmp/aisha_stack.log 2>&1" </dev/null >/dev/null 2>&1 &

echo "[console] waiting for the camera to publish..."
for i in $(seq 1 40); do
  if ros2 topic hz /camera/camera/color/image_raw --window 5 2>/dev/null | grep -q "average rate"; then
    echo "[console]   camera is up"; break; fi
  sleep 2
done

echo "[console] starting the web console..."
setsid bash -c "source /opt/ros/humble/setup.bash; source ~/robot_ws/install/setup.bash; \
  export ROS_DOMAIN_ID=99; export PATH=\$HOME/.local/bin:\$PATH; \
  python3 ~/aisha_watch.py > /tmp/aisha_watch.log 2>&1" </dev/null >/dev/null 2>&1 &
sleep 4
echo ""
echo "  ===================================================================="
echo "   OPEN IN YOUR WORKSTATION BROWSER:   http://192.168.55.1:8088"
echo "  ===================================================================="
echo "   - the live camera is on the left"
echo "   - type a question on the right, read the answer (nothing is spoken yet)"
echo "   - stop everything with:  ~/aisha_console_stop.sh"
