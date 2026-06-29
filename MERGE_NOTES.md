# Merge Notes — best-of-three consolidation (2026-06-29)

This branch (`merge/best-of-three`) consolidates the strongest parts of three
AI-SHA sources into one full-robot ROS 2 Humble workspace using a `src/` layout.

## Sources

| Tag | Source | Strength taken |
|---|---|---|
| **brain** | `hsdaou/AI-SHA-Production` (this repo, pre-merge) | Curated brain + perception, tests, ADRs, `build_ros.sh` |
| **robot** | `Ahmed28309/AI-SHA` | Full hardware: LiDAR/mecanum/motors/sensors, encoder odom + EKF, firmware, farm brain, docs |
| **local** | "AI-SHA Reimagined" working tree | Nav2 stack (`nav2_params.yaml`), `aisha_integration` |

## Per-package precedence

| Package / asset | Taken from | Why |
|---|---|---|
| `src/aisha_brain` | **brain** | Dominant: already has the session RAG hardening (grade-filter union, table/subsection parser) **plus** `gpu_arbiter` mode-switching that local lacked |
| `src/llm_node`, `src/stt_node`, `src/yolov8_ros` | **brain** | Curated, brain-consistent; yolov8 also the larger variant |
| `src/robot_brain` | **robot** | Farm/agriculture brain (`farm_brain`). NOTE: in the brain source this package name was *legacy* — confirm intended role |
| `src/robot_description` | **robot** | URDF with the laser-frame-mismatch fix |
| `src/mecanum_driver`, `src/motor_control`, `src/ldlidar_stl_ros2`, all `src/*_sensor` / IMU / GPS / display / tts_elevenlabs | **robot** | Only source for the hardware/sensor/odometry layer |
| `src/robot_bringup` configs | superset | repo1 configs + **robot**'s `ekf.yaml` + **local**'s `nav2_params.yaml` |
| `src/aisha_integration` | **local** | Bringup meta-package |
| `LICENSE` | **robot** (MIT) | Carried so attribution/licensing is preserved |
| `firmware/`, `docs/`, `scripts/`, `tools/`, `plant_disease_training/`, `legacy/` | **robot** | Hardware firmware, documentation, training |

## ⚠ NOT YET BUILD-VERIFIED — do this first on the Linux/5080 box

1. **No `colcon build` / `colcon test` has run.** This was assembled on a Windows
   host without ROS 2. First action: `colcon build` (not `--symlink-install`) then
   `colcon test`. Expect to fix package.xml/setup.py and cross-package deps.
2. **`robot_bringup` launch files are from the brain source** (brain-centric). The
   `ekf.yaml` and `nav2_params.yaml` configs are present, but the **launch wiring**
   that starts encoder odometry + EKF + sensors is **not** merged in — reconcile
   against `Ahmed28309/AI-SHA` `robot_bringup/launch/*`.
3. **`llm_node` / `stt_node` / `yolov8_ros`** chose the brain-source versions (very
   close in size to robot-source). Verify no farm-specific features were lost.
4. **`robot_brain` role** — confirm farm brain vs legacy (see precedence note).
5. **`legacy/`** included for completeness; prune if unwanted.
6. **CI** — the brain source reports a GitHub Actions workflow but no workflow file
   exists in-tree. Recommend adding a `colcon test` Action as a follow-up.
7. **Duplicate STT path** — `stt_node` exists both as a standalone package and as a
   node inside `aisha_brain`; confirm which is the live one.

## Build

```bash
colcon build            # NOT --symlink-install (see build_ros.sh)
source install/setup.bash
colcon test && colcon test-result --verbose
```
