# Merge Notes — best-of-three consolidation (2026-06-29)

This branch (`merge/best-of-three`) consolidates the strongest parts of three
AI-SHA sources into one ROS 2 Humble workspace (`src/` layout), then **pruned to a
fully-local, no-farming school robot** (see *Pruned* below). 13 packages.

## Sources

| Tag | Source | Strength taken |
|---|---|---|
| **brain** | `hsdaou/AI-SHA-Production` (this repo, pre-merge) | Curated brain + perception, tests, ADRs, `build_ros.sh` |
| **robot** | `Ahmed28309/AI-SHA` | Hardware: LiDAR / mecanum / motors, encoder odom + EKF, IMU, firmware, docs (farm/agri parts dropped) |
| **local** | "AI-SHA Reimagined" working tree | Nav2 stack (`nav2_params.yaml`), `aisha_integration` |

## Per-package precedence

| Package / asset | Taken from | Why |
|---|---|---|
| `src/aisha_brain` | **brain** | Dominant: already has the session RAG hardening (grade-filter union, table/subsection parser) **plus** `gpu_arbiter` mode-switching that local lacked |
| `src/stt_node`, `src/yolov8_ros` | **brain** | Curated, brain-consistent (yolov8 = general vision after plant-disease strip) |
| `src/robot_description` | **robot** | URDF with the laser-frame-mismatch fix |
| `src/mecanum_driver`, `src/motor_control`, `src/ldlidar_stl_ros2`, `src/bno055_imu`, displays | **robot** | Only source for the hardware/odometry layer |
| `src/robot_bringup` configs | superset | repo1 configs + **robot**'s `ekf.yaml` + **local**'s `nav2_params.yaml` |
| `src/aisha_integration` | **local** | Bringup meta-package |
| `LICENSE` | **robot** (MIT) | Carried so attribution/licensing is preserved |
| `firmware/`, `docs/`, `scripts/`, `tools/` | **robot** | Hardware firmware + documentation |

## Pruned for local-only + no farming

Removed so the robot runs **entirely on-device** with **no agricultural features**:

* **Cloud-dependent (removed for local-only):** `tts_elevenlabs` (ElevenLabs API),
  `llm_node` (Gemini "Plant Health" LLM), `stt_node/stt_node_api.py` +
  `stt_assemblyai.py` (cloud STT). The stack now uses only local Ollama / Whisper /
  Piper / bge + ChromaDB. (`aisha_integration/jetson_launch.py` keeps a startup
  *kill-list* that terminates any stray cloud node — kept as a local-only guard.)
* **Farming (removed):** `robot_brain` (farm_brain), `soil_moisture`, `rain_sensor`,
  `gps_gt_u7`, `bmp180_pressure`, `plant_disease_training/`, the farm-brain docs, and
  the `plant_disease_*` nodes in `yolov8_ros`. The plant-disease classifier hook was
  also stripped out of `yolov8_node.py` (py_compile-verified); it is now general
  vision (person / face / gesture / OCR). Legacy `cerebro.launch.py` (launched the old
  GGUF `robot_brain`) was removed — use `cerebro_aisha.launch.py`.

## ⚠ NOT YET BUILD-VERIFIED — do this first on the Linux/5080 box

1. **No `colcon build` / `colcon test` has run.** Assembled on a Windows host without
   ROS 2. First action: `colcon build` (not `--symlink-install`) then `colcon test`.
2. **`robot_bringup` launch wiring** — `ekf.yaml` + `nav2_params.yaml` configs are
   present, but the launch that starts encoder odometry + EKF + sensors is not yet
   merged in. Reconcile against `Ahmed28309/AI-SHA` `robot_bringup/launch/*`.
3. **CI** — add a `colcon test` GitHub Action (none exists in-tree).
4. **Duplicate STT path** — `stt_node` exists both as a standalone package and as a
   node inside `aisha_brain`; confirm which is the live one.

## Build

```bash
colcon build            # NOT --symlink-install (see build_ros.sh)
source install/setup.bash
colcon test && colcon test-result --verbose
```
