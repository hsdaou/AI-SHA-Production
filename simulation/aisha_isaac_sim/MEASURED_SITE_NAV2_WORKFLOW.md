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

## 4. Reproduce the Nav2 + Phase 3N integrations

Official ROS 2 Jazzy Nav2 binaries are available in a user-local overlay at
`~/.local/share/ai_sha_ros_jazzy_overlay/root`. This avoids modifying the
workstation while `sudo` requires an interactive password. The overlay is
sourced automatically by the launch wrappers.

The current occupancy map is generated from the same provisional
walkthrough/plan-derived USD used by Isaac Sim:

```bash
tools/generate_administration_occupancy_map.sh
python3 tools/test_administration_occupancy_map.py
```

Run the complete headless integration gate with one command:

```bash
tools/run_administration_nav2_phase3n_integration.sh
```

That command starts the Isaac/ROS bridge, map server, AMCL, Nav2 planner and DWB
controller; executes the 12-leg atrium → Vice-Principal → Principal → atrium
mission; routes each `/cmd_vel` command through the accepted recurrent Phase 3N
360-degree safety actor; and validates the paired reports. The accepted
checkpoint is hash locked to
`11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b`.

The verified run completed all 12 legs and both counterclockwise office pivots
with no episode reset. Its bridge ran 12,101 physics/control steps; the learned
safety authority was eligible for 440 steps and applied nonzero braking on 57.
See `results/administration_nav2_phase3n_integration_gate.json` for the 14/14
combined gate.

To watch it live, use three terminals from `simulation/aisha_isaac_sim`:

```bash
# Terminal 1: omit --headless to open Isaac Sim / Omniverse
tools/run_administration_nav2_phase3n_bridge.sh

# Terminal 2
tools/run_administration_nav2_servers.sh

# Terminal 3, after Nav2 becomes active
tools/run_administration_nav2_mission.sh \
  --control-stack nav2_phase3n_safety \
  --output results/administration_nav2_phase3n_mission.json
```

`config/nav2_sim_params.yaml` is simulation-only and intentionally separate
from `src/robot_bringup/config/nav2_params.yaml`. That older file is for a 0.60
x 0.50 m holonomic mecanum chassis and is unsafe for the Rev D model.

### Measured-presentation static mission

The measured-presentation profile adds the 0.85 m VP door, 0.90 m Principal
door, 0.20 m central drop no-go region and a deterministic two-stage doorway
alignment guard. Run its complete headless gate with:

```bash
tools/run_administration_nav2_measured_integration.sh
```

For a visible Omniverse session, use three terminals:

```bash
# Terminal 1: visible Isaac Sim window
tools/run_administration_nav2_measured_bridge.sh

# Terminal 2
tools/run_administration_nav2_measured_servers.sh

# Terminal 3, after Nav2 becomes active
tools/run_administration_nav2_measured_mission.sh
```

The accepted run completed all 12 legs, both office entries/departures, both
180-degree office pivots and the return home in 261.544 s wall time. The bridge
ran 18,323 control steps with no episode reset. Both doorways were crossed in
both directions; maximum measured doorway body speed was 0.06258 m/s against
the 0.10 m/s limit. The accepted learned Phase 3N layer had authority for 689
steps and applied braking on 192. The aggregate report
`results/administration_nav2_measured_integration_gate.json` passes 27/27.

The baseline hallway command ceiling remains 0.30 m/s, with 0.08 m/s targeted
through the tight openings. The separately trained Phase 6 tier retains the
same geometry and doorway limit and is now accepted for simulation presentation
on straight segments 1 and 5. Reproduce its complete live gate with:

```bash
tools/run_administration_nav2_phase6_integration.sh
```

The accepted run completed all 12 legs and reached 0.74339 m/s or more on both
high-speed legs. `results/administration_nav2_phase6_high_speed_integration_gate.json`
passes 28/28. The Phase 6 bridge publishes an identity map-to-odom transform
from Isaac ground-truth odometry and disables AMCL TF output; this isolates the
navigation/safety presentation from scan-matching errors in assumed furniture
and is not physical localisation evidence.

The final six-shot PathTracing replay uses only the recorded accepted pose
trace and is available at
`media/videos/AI-SHA_Phase6_Nav2_LearnedSafety_RTX_Presentation.mp4`.
`results/administration_nav2_phase6_rtx_presentation_acceptance.json` passes
14/14. It is a cinematic replay of live source motion, not a second live-policy
execution.

## 5. Architecture and claim boundary

Two authentic controller paths are now verified:

1. live Nav2 global/local planning → accepted Phase 3N safety actor → articulated
   wheel physics; and
2. frozen learned route actor → frozen Phase 3M recovery/clearance/pivot stack →
   accepted Phase 3N safety actor → articulated wheel physics in the
   administration scene.

The second path's policy-only administration run completed all 12 waypoints and
is recorded in `results/phase3n_administration_final_omniverse_report.json`.
Nav2 and the frozen learned local navigator are not placed in series because
they are both local motion authorities; doing that without a deliberate
arbitration design would obscure which controller caused a command.

This closes the measured-presentation **static** Nav2 integration gate, not the
blocked-route dynamic-replanning or physical-release gates. The supplied iPhone
mesh sections and manual door dimensions inform the procedural scene, but
native section-to-stage registration and an as-built survey are still
incomplete. Physical work still requires measured clearance/threshold review,
sim-to-real validation, a hardware emergency stop and supervised commissioning.

Check measured-site preparation at any time with:

```bash
python3 tools/validate_measured_nav2_preparation.py
```
