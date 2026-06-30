# AI-SHA — Production (Local School Robot)

ROS 2 Humble workspace for **AI-SHA**, an administrative-assistant robot for the
International School of Choueifat (ISC), Sharjah. It **navigates autonomously and
avoids obstacles**, **listens** and answers school-admin questions from a local RAG
knowledge base, **sees** (RealSense + YOLOv8), and is driven by voice — running
**fully offline / on-device** on an 8 GB **Jetson Orin Nano** (no cloud services).

> This repository is a consolidated **best-of-three** merge (brain + hardware + nav).
> See **[MERGE_NOTES.md](MERGE_NOTES.md)** for what came from where and what still
> needs a `colcon build` pass, and **[NOTICE](NOTICE)** for attribution.

> **📖 Design deep-dive:** the GPU strategy and benchmarks are in
> **[ADR 0001 — Time-multiplex the GPU between NAVIGATING and CONVERSING](src/aisha_brain/docs/adr/0001-gpu-multiplexing-navigating-conversing.md)**.

## What it does

- **Navigates & avoids obstacles** — Nav2 (mecanum-holonomic) + SLAM Toolbox on LD19
  LiDAR, with encoder odometry fused with the BNO055 IMU via an EKF.
- **Listens** continuously (Whisper STT → `/speech/text`).
- **Routes intent** (ADMIN / NAV / ACTION) with a **deterministic keyword/pattern
  classifier** — no LLM is used for routing, so it is instant and the brain node
  has no Ollama dependency.
- **Answers** school-admin questions from a RAG KB (ChromaDB + bge-small + `llama3.2:1b`
  via Ollama), streaming sentence-by-sentence; refuses out-of-scope/academic requests.
- **Sees** (RealSense D435 + YOLOv8 TensorRT).
- **Converses on command** via the wake word **"Hey Aisha stop"** — the robot halts and
  switches into a GPU-accelerated answer mode (~2–6 s replies vs ~16 s on CPU).

Everything runs locally on the Jetson — local LLM (Ollama), local Whisper STT, local
Piper TTS, local embeddings (bge) + ChromaDB. No external/cloud APIs are used.

### Headline design: GPU time-multiplexing (ADR 0001)

The 8 GB shared memory can't hold both YOLO and the LLM on the GPU at once, so a
**`gpu_arbiter`** node time-multiplexes: **NAVIGATING** (YOLO owns the GPU, LLM on CPU,
listening for the wake word) ↔ **CONVERSING** (LLM on GPU, motion-locked, YOLO killed to
free VRAM). Triggered by "Hey Aisha stop"; auto-returns to NAVIGATING after a timeout.

## Workspace layout

```
src/
  aisha_brain/        brain_node (router), admin_node (RAG), action_node, gpu_arbiter,
                      tts_node (Piper, local), stt_node, waypoint_resolver, + ADR
  robot_bringup/      launch + configs: nav2_params.yaml, ekf.yaml, slam_toolbox, ld19
  robot_description/  robot URDF (digital twin)
  yolov8_ros/         vision detection_node (person / face / gesture / OCR) + pause_inference
  stt_node/           Whisper STT (local; mic-mute while speaking)
  mecanum_driver/     holonomic drive + encoder odometry
  motor_control/      low-level motor/encoder serial bridge
  ldlidar_stl_ros2/   LD19 LiDAR driver (C++)
  bno055_imu/         IMU (fused into odometry via EKF)
  speaker_monitor/  robot_display/  llm_display/   audio + face/eyes UI
  aisha_integration/  bringup meta-package
firmware/             Arduino mecanum controller + Pi encoder node
docs/                 hardware setup, Jetson integration, TTS troubleshooting
```

## Quick start

```bash
# Prereqs: Ollama (ollama serve + ollama pull llama3.2:1b); build the KB:
#   python3 -m aisha_brain.build_knowledge      # regenerates aisha_knowledge_db/

colcon build            # NOT --symlink-install (see build_ros.sh)
source install/setup.bash

# Full stack (camera + YOLO + brain + RAG + gpu_arbiter):
ros2 launch robot_bringup cerebro_aisha.launch.py
# Navigation (SLAM + Nav2):
ros2 launch robot_bringup slam.launch.py
```

## Key topics / services

- `/speech/text` — STT output (and wake-word input)
- `/robot_speech` — brain → TTS (speech output)
- `/aisha/mode` — `gpu_arbiter` broadcasts NAVIGATING/CONVERSING
- `/aisha/set_conversing` (`std_srvs/SetBool`) — manual mode trigger
- `/scan`, `/odom`, `/cmd_vel` — LiDAR, fused odometry, mecanum velocity command

See the **[ADR](src/aisha_brain/docs/adr/0001-gpu-multiplexing-navigating-conversing.md)**
for the complete rationale and benchmarks.
