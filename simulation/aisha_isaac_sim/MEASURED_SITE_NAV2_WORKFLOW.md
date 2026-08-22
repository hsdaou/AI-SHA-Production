# Measured administration and Nav2 workflow

This workflow replaces presentation geometry assumptions with an iPhone LiDAR /
RoomPlan capture, then runs the Rev D differential-drive robot in the Isaac Sim
administration scene through ROS 2. The scan is survey input for simulation; it
is not a safety certification or a release for physical operation.

## 1. Capture the administration

Preferred delivery is one complete RoomPlan/mesh export in USDZ. If the app
cannot keep one stable scan, make these five overlapping captures:

1. atrium, including the east-hall opening;
2. east hallway and Vice-Principal approach;
3. Vice-Principal office through its doorway;
4. Principal passage from the atrium through its doorway; and
5. Principal office through its doorway.

Keep people, faces, screens, papers, noticeboards and personal information out
of the capture. Keep the original files unchanged. Record the iPhone model,
capture app/version and export coordinate metadata.

The LiDAR mesh is not accurate enough by itself for the route's critical fit.
Measure the following with the Measure app, a laser measure or tape:

- clear width, clear height and frame depth at both office doors;
- threshold height on both sides and the threshold profile at both doors;
- hinge side and swing direction as seen from the hallway;
- east-hall and Principal-passage clear widths;
- the clear length and width of both turn zones;
- ceiling height; and
- atrium column diameter and centre positions, if accessible.

Clear width means the narrowest usable opening with stops and hardware in
place—not the nominal door-leaf size. The simulation gate requires at least
0.928 m for straight transit: 0.768 m physical width plus 0.080 m costmap
padding on each side. In-place turns require a 1.640 m clear diameter.

## 2. Validate and prepare the measured overlay

Copy `config/measured_administration_template.yaml` beside the immutable scan
files and fill it in. For section scans, determine each source-to-stage
translation from their overlap anchors. The target convention is metres, Z up,
+X east, +Y north, with the origin at the atrium centre on finished-floor level.

From `simulation/aisha_isaac_sim`:

```bash
python3 tools/prepare_measured_administration.py \
  --manifest /path/to/capture/administration_manifest.yaml \
  --scan-root /path/to/capture/originals \
  --output results/measured_administration_intake.json \
  --overlay-output config/measured_administration.generated.yaml
```

The tool validates the file headers, records SHA-256 hashes, verifies capture
privacy/coordinate metadata and checks all critical clearances. It writes the
overlay only when the intake is complete. The overlay preserves
`physical_release: false`.

Build the aligned procedural scene candidate with:

```bash
./run_isaac.sh scripts/build_administration.py \
  --headless \
  --plan /home/robot-wst/Downloads/DownloadBuildingRequestApprovedPlan.pdf \
  --measured-geometry config/measured_administration.generated.yaml
```

The generated overlay updates the measured room, doorway, column, ceiling and
route values supplied in the manifest while retaining explicitly disclosed
assumptions for anything not measured. The original scan paths/transforms and
hashes remain in the overlay for the subsequent visual-alignment pass.

## 3. Run the Isaac Sim ROS 2 bridge

Isaac Sim 5.1 embeds Python 3.11 and its own Jazzy `rclpy`, while Ubuntu 24.04's
system Jazzy uses Python 3.12. Do not source `/opt/ros/jazzy` in the Isaac
terminal. The wrapper isolates the two environments correctly:

```bash
tools/run_administration_nav2_bridge.sh --headless
```

For a finite smoke test:

```bash
tools/run_administration_nav2_bridge.sh \
  --headless --max-steps 90 --self-test \
  --output-report results/administration_nav2_bridge_smoke_report.json
```

The bridge subscribes to `/cmd_vel` and publishes `/clock`, `/odom`, `/tf`,
`/tf_static`, `/scan` and `/front_scan`. Static transforms locate the crown
LiDAR, low-front LiDAR and IMU under `base_link`. Commands are applied through
the simulated wheel joints—never by animating the robot root—and have a 0.30 s
watchdog, no reverse or lateral motion, and a non-safety-rated front stop latch.

To verify bidirectional exchange from the separate system-ROS environment while
the bridge is running:

```bash
source /opt/ros/jazzy/setup.bash
python3 tools/probe_administration_nav2_bridge.py \
  --output results/administration_nav2_bridge_external_probe.json
```

## 4. Start Nav2 after its remaining gates are complete

Nav2 packages are not currently installed on this workstation, and a measured
occupancy map cannot be generated until the capture arrives. After those two
gates are resolved, start Nav2 in a separate system-ROS terminal:

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch nav2_bringup bringup_launch.py \
  use_sim_time:=True \
  params_file:=$(pwd)/config/nav2_sim_params.yaml \
  map:=/absolute/path/to/measured_administration_map.yaml
```

`config/nav2_sim_params.yaml` is simulation-only and intentionally separate
from `src/robot_bringup/config/nav2_params.yaml`. That older file is for a 0.60
x 0.50 m holonomic mecanum chassis and is unsafe for the Rev D model.

Check preparation at any time with:

```bash
python3 tools/validate_measured_nav2_preparation.py
```

The next integration gate after a live Nav2 mission is explicit arbitration
between Nav2's velocity command and the frozen Phase 3N learned route/360-degree
safety stack. Until that test passes, neither configuration is represented as a
fully integrated learned-navigation system.
