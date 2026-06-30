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

## Build verification — DONE on the RTX 5080 box (2026-06-30)

Built and tested inside a `ros:humble` container (NOT the workstation's native
Jazzy — see *Build target* below).

1. **✅ `colcon build` (no `--symlink-install`): 13/13 packages clean**, including
   the C++ `ldlidar_stl_ros2` and the `bno055_imu` message package.
2. **✅ `robot_bringup` launch wiring done** — `aisha_integration/rpi_launch.py`
   now has an `odom_source:=ekf` path that starts `robot_localization` with
   `ekf.yaml`, fusing encoder `/odom` + IMU `/imu/data` and owning the
   `odom→base_link` TF (driver set to `publish_odom=true`,
   `publish_odom_tf=false`). `ekf.yaml` made live (odom0 enabled, `imu0` fixed
   to `/imu/data`). Default `odom_source` stays `laser`, so EKF is opt-in.
3. **✅ `colcon test`: no regressions** — reconciled tree vs the pristine local
   baseline give byte-identical `aisha_brain` results. Remaining failures are
   inherited (bare-container missing `requirements.txt` deps + lint/test debt),
   not introduced by the merge.

### Build target — Humble, NOT Jazzy
The deploy target is the Jetson Orin Nano (JetPack / Ubuntu 22.04 = **ROS 2
Humble**). The 5080 workstation is Ubuntu 24.04 (native Jazzy). Build/test in a
`ros:humble` container — Jazzy breakage (e.g. `ldlidar_stl_ros2` C++,
`bno055_imu` msg-gen) is distro-mismatch noise against the wrong baseline, not a
bug to fix.

### Still open
* **CI** — add a `colcon test` GitHub Action (none exists in-tree). The runner
  must `pip install -r src/aisha_brain/requirements.txt` + `rosdep install`, or
  the brain pytest suite errors on missing `requests`/etc. Tracked as a GitHub
  issue.
* **Duplicate STT path — RESOLVED (documented).** Canonical STT is
  **`aisha_brain.stt_node`** (600-line faster-whisper/CUDA node) — the
  production `aisha_integration/jetson_launch.py` and `rpi_launch.py` both spawn
  `package='aisha_brain', executable='stt_node'`. The standalone **`stt_node`
  package** (358-line legacy version) is now marked DEPRECATED in its docstring
  + package.xml; it is retained only because the legacy
  `robot_bringup/cerebro_aisha.launch.py` brain launch still references it. No
  production launch depends on the standalone package. Remove it + rewire
  cerebro_aisha to `aisha_brain` in a later cleanup if cerebro_aisha is retired.

### Resolved in follow-up (2026-06-30)
* **Doc debt — done.** README and ADR 0001 now describe the intent router
  correctly: **LLM-first (`gemma3:270m`) with a keyword fallback**. The ADR's
  gemma3 "removal" wording was clarified to mean the runtime VRAM unload on
  CONVERSING (the router is not deleted).
* **EKF + Nav2 without SLAM — done.** The odom-source selection
  (rf2o / dummy / ekf) was lifted out of the `enable_slam` block in
  `aisha_integration/rpi_launch.py`; it now runs whenever odom is needed, so
  `enable_slam:=false` (AMCL on a saved map) still gets a fused
  `odom→base_link`. Only `slam_toolbox` remains gated by `enable_slam`.

## Build

```bash
colcon build            # NOT --symlink-install (see build_ros.sh)
source install/setup.bash
colcon test && colcon test-result --verbose
```

## Handoff — continuing on the RTX 5080 workstation

State: this is PR #1 (`merge/best-of-three`) — a local-only, no-farming, 13-package
workspace. **Not yet `colcon build`/`test`'d.**

Remaining job is a **reconciliation pass**: the local "AI-SHA Reimagined" tree is
*ahead* of this merge in two packages (audit-confirmed), so this merge would
otherwise regress them:

- `mecanum_driver` — local's node (815 lines) + Arduino firmware (395 lines) beat
  this merge's Ahmed-sourced versions (588 / 145). Reconcile **toward local**.
- `aisha_brain` — local has ~140 lines in `admin_node.py` + ~96 in
  `build_knowledge.py` not present here (parallel evolution). 3-way reconcile.
- Also unique to local: `.env.example`, `config/fastdds_env.sh`,
  `mecanum_driver/scripts/arduino_mega.rules`, fastdds edits, `campus-map.md`.

Local's ahead code is backed up at **hsdaou/aisha-integration**, branch
**`backup/local-wip-2026-06-29`**.

Steps: (1) clone both repos; (2) reconcile `mecanum_driver` + `aisha_brain` toward
local + fold the unique files in; (3) `colcon build` (NOT `--symlink-install`) +
`colcon test`; (4) wire `robot_bringup` launches for EKF/encoder-odom/sensors
(configs already present); (5) update this PR.

### ✅ Reconciliation completed (2026-06-30)
Steps (1)–(4) done; this commit is step (5).
- **`mecanum_driver`** → local (815-line node, 395-line firmware, params,
  launch); folded in `scripts/arduino_mega.rules`; kept merge's
  `motor_test`/`motor_direct`/`mecanum.launch.py` + their entry points, and the
  2nd Arduino sketch `arduino/mecanum_driver/mecanum_driver.ino`.
- **`aisha_brain`** → local for all shared code (verified superset: query_id +
  streaming + RAG table/subsection hardening); kept merge's `jetson_launch.py`
  (GPU-OOM env), `gpu_arbiter` (+ entry point), `gpu_release_probe.py`,
  `fastdds_*.xml`, ADR, `sabis_system.md`; adopted local's `rpi_launch.py`
  deprecation.
- **Folded root files**: `.env.example`, `config/fastdds_env.sh`.
- **CRLF→LF** normalized the whole tree + added `.gitattributes` (the merge was
  assembled on Windows; CRLF was masking the real diffs).
