#!/bin/bash
# AI-SHA browser console WITH voice: the robot's ReSpeaker mic -> Whisper STT -> question.
# Say "Hey Aisha, <your question>" to the robot's microphone; the transcription and the
# answer both appear on http://192.168.55.1:8088 . You can still type as well.
set -e
source /opt/ros/humble/setup.bash
source ~/robot_ws/install/setup.bash
export ROS_DOMAIN_ID=99
export PATH=$HOME/.local/bin:$PATH

echo "[voice] clearing any stale nodes..."
pkill -9 -f 'yolov8_nod[e]'            2>/dev/null || true
pkill -9 -f 'realsense2_camera_nod[e]' 2>/dev/null || true
pkill -9 -f 'jetson_launc[h]'          2>/dev/null || true
pkill -9 -f 'stt_nod[e]'               2>/dev/null || true
pkill -9 -f 'aisha_watc[h]'            2>/dev/null || true
sleep 3

echo "[voice] launching stack WITH the ReSpeaker mic (STT on)..."
setsid bash -c "source /opt/ros/humble/setup.bash; source ~/robot_ws/install/setup.bash; \
  export ROS_DOMAIN_ID=99; export PATH=\$HOME/.local/bin:\$PATH; \
  ros2 launch aisha_integration jetson_launch.py \
      enable_stt:=true audio_device:=ReSpeaker \
      whisper_model:=small whisper_device:=cpu enable_vision:=false enable_gpu_arbiter:=false \
      wake_word_enabled:=true llm_model:=aisha:1b \
  > /tmp/aisha_stack.log 2>&1" </dev/null >/dev/null 2>&1 &

echo "[voice] waiting for the camera..."
for i in $(seq 1 40); do
  if ros2 topic hz /camera/camera/color/image_raw --window 5 2>/dev/null | grep -q "average rate"; then
    echo "[voice]   camera up"; break; fi
  sleep 2
done
echo "[voice] waiting for STT to bind the mic..."
for i in $(seq 1 30); do
  if grep -qiE "wake_word=AISHA|Listening|mic=" /tmp/aisha_stack.log 2>/dev/null; then
    echo "[voice]   STT listening"; break; fi
  sleep 2
done

echo "[voice] starting the web console (voice banner on)..."
setsid bash -c "source /opt/ros/humble/setup.bash; source ~/robot_ws/install/setup.bash; \
  export ROS_DOMAIN_ID=99; export PATH=\$HOME/.local/bin:\$PATH; export AISHA_VOICE=1; \
  python3 ~/aisha_watch.py > /tmp/aisha_watch.log 2>&1" </dev/null >/dev/null 2>&1 &
sleep 4
echo ""
echo "  ===================================================================="
echo "   OPEN:  http://192.168.55.1:8088"
echo "   SAY:   \"Hey Aisha, what are the tuition fees?\"  (to the robot's mic)"
echo "   stop:  ~/aisha_console_stop.sh"
echo "  ===================================================================="
