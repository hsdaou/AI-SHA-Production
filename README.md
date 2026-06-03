# AI-SHA — Production Brain (Jetson Orin Nano)

ROS 2 Humble workspace for **AI-SHA**, an administrative-assistant robot for the
International School of Choueifat (ISC), Sharjah. It listens, answers school-admin
questions from a local RAG knowledge base, sees (RealSense + YOLO), and navigates —
all on an 8 GB Jetson Orin Nano.

> **📖 Start here:** the full design + every measurement is in
> **[ADR 0001 — Time-multiplex the GPU between NAVIGATING and CONVERSING](aisha_brain/docs/adr/0001-gpu-multiplexing-navigating-conversing.md)**.
> It explains *why* the system is built the way it is, with live benchmarks.

## What it does

- **Listens** continuously (Whisper STT → `/speech/text`)
- **Routes intent** (ADMIN / NAV / ACTION) with a deterministic **rule-based** classifier
  (the gemma3 LLM router was removed — it scored 0/18 in testing; see ADR / memory)
- **Answers** school-admin questions from a RAG KB (ChromaDB + bge-small + `llama3.2:1b` via Ollama),
  streaming sentence-by-sentence to `/robot_speech`; refuses out-of-scope/academic requests
- **Sees** (RealSense D435 + YOLOv8 TensorRT) and can navigate/act
- **Converses on command** via the wake word **“Hey Aisha stop”** — the robot halts and
  switches into a GPU-accelerated answer mode (~2–6 s replies vs ~16 s on CPU)

### The headline design: GPU time-multiplexing (ADR 0001)

The 8 GB shared memory can't hold both YOLO and the LLM on the GPU at once, so a
**`gpu_arbiter`** node time-multiplexes between two states:

| State | GPU owner | Behaviour |
|---|---|---|
| **NAVIGATING** | YOLO vision | robot roams; LLM on CPU; listens for the wake word |
| **CONVERSING** | LLM (`num_gpu=99`) | robot stationary + motion-locked; YOLO killed to free VRAM; ~2–6 s answers |

Triggered by **“Hey Aisha stop”** (or `ros2 service call /aisha/set_conversing std_srvs/srv/SetBool "{data: true}"`);
auto-returns to NAVIGATING after a timeout (respawns YOLO, ~10 s).

## Quick start

```bash
# Prereqs (once):
#  - Ollama running on the Jetson:  ollama serve  +  ollama pull llama3.2:1b
#  - Build the knowledge base:      python3 -m aisha_brain.build_knowledge   (regenerates aisha_knowledge_db/)
#  - Host config (not in repo):     ~/fastdds.xml referenced by FASTRTPS_DEFAULT_PROFILES_FILE

cd ~/robot_ws
colcon build            # NOT --symlink-install (fails on this setuptools); use plain build
source install/setup.bash

# Bring up the whole stack (camera + YOLO + brain + RAG + gpu_arbiter):
ros2 launch robot_bringup cerebro_aisha.launch.py

# Useful args:
#   enable_gpu_arbiter:=false   # fall back to always-on vision (no GPU multiplexing)
#   enable_stt:=false           # disable Jetson STT (e.g. if STT runs elsewhere)
#   llm_model:=llama3.2         # higher-quality 3B answers (CPU only; ~60 s)
```

## Packages (first-party)

| Package | Role |
|---|---|
| **aisha_brain** | `brain_node` (rule-based router), `admin_node` (RAG), `action_node`, **`gpu_arbiter`**, `tts_node`, `stt_node`, `waypoint_resolver`, `whatsapp_listener` + the ADR & `scripts/gpu_release_probe.py` |
| **robot_bringup** | launch files — `cerebro_aisha.launch.py` is the main entry point |
| **yolov8_ros** | vision (`detection_node`) + the `pause_inference` service |
| **stt_node** | Whisper STT; mutes the mic while the robot speaks (gates on `/speaker/playing` + `/robot_speech`) |
| **robot_description** | robot URDF |
| **robot_brain**, **llm_node** | legacy brains (pre-AI-SHA), kept for rollback |

Vendored ROS packages (realsense-ros, slam_toolbox, ldlidar_stl_ros2, web_video_server)
are **not** tracked here — they carry their own upstream git history.

## Key topics / services

- `/speech/text` — STT output (and wake-word input)
- `/robot_speech` — brain → TTS (speech output)
- `/aisha/mode` — `gpu_arbiter` broadcasts NAVIGATING/CONVERSING; `admin_node` switches GPU↔CPU
- `/aisha/set_conversing` (`std_srvs/SetBool`) — manual mode trigger
- `/detection_node/pause_inference` (`std_srvs/SetBool`) — pause YOLO inference

## Known follow-ups

- **RPi5 TTS (separate machine):** subscribe `/robot_speech` and publish `/speaker/playing` (Bool)
  for precise mic-unmute. Until then the Jetson speaks/mutes via the `/robot_speech` failsafe.
- The ~10 s YOLO respawn on return-to-roam is the floor on 8 GB (Python/torch/CUDA cold start);
  true pause-only (~0 ms) needs a 16 GB board or a C++ vision node. See ADR “yolov8n” section.

See the **[ADR](aisha_brain/docs/adr/0001-gpu-multiplexing-navigating-conversing.md)** for the
complete rationale, benchmarks, and the design decisions (and the dead ends, with data).
