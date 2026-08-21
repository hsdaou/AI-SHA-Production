#!/bin/bash
# AI-SHA boot entrypoint (systemd: aisha-console.service).
#
# Stays in the FOREGROUND and supervises its children. This matters: if this
# script detaches and exits, systemd considers the service finished, fires
# ExecStop and tears the whole stack down again (that bug killed the camera on
# the first boot test). Exiting non-zero here asks systemd to restart us.
export HOME=/home/hsdaou
source /opt/ros/humble/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=42
export PATH=$HOME/.local/bin:$PATH
export AISHA_VOICE=1

LOG=/tmp/aisha_stack.log
say(){ echo "[boot $(date +%T)] $*"; }

kill_kids(){
  pkill -9 -f 'aisha_watc[h]'            2>/dev/null
  pkill -9 -f 'jetson_launc[h]'          2>/dev/null
  pkill -9 -f 'realsense2_camera_nod[e]' 2>/dev/null
  pkill -9 -f 'stt_nod[e]'               2>/dev/null
  pkill -9 -f 'yolov8_nod[e]'            2>/dev/null
  pkill -9 -f 'brain_node|admin_node|action_node' 2>/dev/null
}
trap 'say "stopping on signal"; kill_kids; exit 0' TERM INT

# ── wait for USB hardware; on a cold boot the camera and mic enumerate late ──
say "waiting for RealSense + ReSpeaker to enumerate..."
for i in $(seq 1 45); do
  cam=$(lsusb 2>/dev/null | grep -ci 'intel' || echo 0)
  mic=$(arecord -l 2>/dev/null | grep -ci respeaker || echo 0)
  [ "$cam" -gt 0 ] && [ "$mic" -gt 0 ] && { say "  hardware present"; break; }
  sleep 2
done
sleep 10          # RealSense needs the bus to settle or it opens then dies

# arecord can see the array before PortAudio/sounddevice can. stt_node matches on
# the SOUNDDEVICE name, so gate on that or STT silently falls back to the dead
# on-board APE ("could not match ReSpeaker" -> device=None).
say "waiting for sounddevice to see the ReSpeaker..."
for i in $(seq 1 30); do
  if python3 -c "import sounddevice as sd,sys; sys.exit(0 if any('ReSpeaker' in d['name'] and d['max_input_channels']>0 for d in sd.query_devices()) else 1)" 2>/dev/null; then
    say "  sounddevice sees the mic"; break
  fi
  sleep 2
done

kill_kids; sleep 3

say "launching stack (camera + mic + LLM/RAG)"
ros2 launch aisha_integration jetson_launch.py \
    enable_stt:=true audio_device:=ReSpeaker \
    whisper_model:=small whisper_device:=cpu \
    enable_vision:=false enable_gpu_arbiter:=false \
    wake_word_enabled:=true wake_word_timeout:=6.0 llm_model:=aisha:1b >> "$LOG" 2>&1 &
STACK=$!

say "waiting for camera frames..."
CAM_OK=0
for i in $(seq 1 60); do
  if timeout 4 ros2 topic hz /camera/camera/color/image_raw --window 3 2>/dev/null | grep -q "average rate"; then
    CAM_OK=1; say "  camera streaming"; break
  fi
  sleep 2
done
[ "$CAM_OK" = "1" ] || say "  WARNING: no camera frames yet (console will still serve)"

say "starting web console on :8088"
python3 "$HOME/aisha_watch.py" >> /tmp/aisha_watch.log 2>&1 &
WATCH=$!
# ── pipeline self-test ────────────────────────────────────────────────────
# On a cold boot brain_node sometimes comes up before discovery settles and its
# /speech/text subscription never fires — the console answers HTTP 200 but no
# question is ever routed. Prove the path works; if not, exit non-zero and let
# systemd restart us (a restart has always cleared it). Bonus: this warms the
# LLM, so the first real question is fast.
# The probe deliberately says "video message": brain_node routes that to
# SKILL_VIDEO and publishes NOTHING, so the self-test proves routing works
# without burning an LLM call or greeting the visitor with a wall of tuition
# figures they never asked for. aisha_watch ignores __selftest__ outright, so
# it does not start a recording either.
say "self-testing the question pipeline..."
BEFORE=$(grep -c 'Route ->' "$LOG" 2>/dev/null || echo 0)
ROUTED=0
for attempt in 1 2; do
  ros2 topic pub --once /speech/text std_msgs/String \
      "{data: '__selftest__ video message'}" >/dev/null 2>&1
  for i in $(seq 1 20); do
    sleep 3
    AFTER=$(grep -c 'Route ->' "$LOG" 2>/dev/null || echo 0)
    if [ "$AFTER" -gt "$BEFORE" ]; then ROUTED=1; break; fi
  done
  [ "$ROUTED" = "1" ] && break
  say "  no routing on attempt $attempt"
done
if [ "$ROUTED" != "1" ]; then
  say "PIPELINE DEAD (brain not routing) — asking systemd to restart"
  kill_kids; exit 1
fi
say "  pipeline OK (brain routed the probe)"

say "READY -> http://192.168.55.1:8088"

# ── supervise: stay alive so systemd keeps the service active ──
while true; do
  sleep 20
  if ! kill -0 "$STACK" 2>/dev/null; then
    say "ros2 launch exited — asking systemd to restart"; kill_kids; exit 1
  fi
  if ! pgrep -f 'realsense2_camera_nod[e]' >/dev/null 2>&1; then
    say "camera node vanished — asking systemd to restart"; kill_kids; exit 1
  fi
  if ! kill -0 "$WATCH" 2>/dev/null; then
    say "web console died — restarting just the console"
    # Clear any half-dead instance first. Without this a console whose ROS thread
    # died but whose HTTP server still holds :8088 survives, the new one cannot
    # bind, and the browser keeps being served a FROZEN camera frame by the old
    # process. Three of them had accumulated that way.
    pkill -9 -f "aisha_watc[h]" 2>/dev/null; sleep 1
    python3 "$HOME/aisha_watch.py" >> /tmp/aisha_watch.log 2>&1 &
    WATCH=$!
  fi
done
