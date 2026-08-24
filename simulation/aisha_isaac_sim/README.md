# AI-SHA - Isaac Sim package (proof of concept)

**Rev X - 2026-08-24 - Phase 8B passive hardware-attachment audit complete**

This package describes the simplified indoor proof-of-concept: two driven hub
wheels on the centre lateral axis, four physical swivel castors, a retained
Prestar NF-301 deck, a guided compliant drive carrier, and a fixed mast
reinforcement spine. It supersedes every earlier 4-wheel skid-steer model.

It is a coherent simulation/design baseline, not a fabrication release or a
safety certification. The unresolved hold points are listed below and in
`config/aisha_drive.yaml`.

## Latest presentation deliverable

The current source is the accepted live Phase 7E ROS 2/Nav2 mission in Isaac
Sim. The native Nav2 costmap uses the static map for known architecture, raw
LiDAR rays for clearing and mapped-free filtered LiDAR endpoints for dynamic
marking. The unchanged Rev D robot and unchanged 0.030 m footprint padding
completed all 12 Principal/Vice-Principal legs, both office pivots, both office
departures and the return home. The live integration gate passes 40/40 without
a collision or episode reset. High-speed legs reached 0.74572 and 0.74342 m/s
from a 0.80 m/s request; doorway speed remained below 0.060 m/s.

Phase 7F turns that accepted motion into the operator-facing film. Omniverse
selects only recorded wheel-physics pose samples from the successful Phase 7E
run; it does not interpolate or invent a second route. Eight fixed,
human-height 11.5-16 mm cameras separately show the atrium departure, high-speed
east hall, Vice-Principal entry and departure, return hall, Principal approach,
Principal visit/departure and home return. The environment remains legible and
AI-SHA does not occupy most of the frame. Every frame says that this is a
visual replay of live source motion, not a second live-policy execution.

Site facts and assumptions remain visible in both the video and reports:

- The Vice-Principal office was locked during capture. Its interior appearance
  is an explicit plan-envelope and adjacent-material assumption.
- The user-reported administration minimum of 0.85 m is conservatively applied
  to the VP door. The Principal door uses a disclosed 0.90 m presentation
  assumption. Both use the reported 2.12 m height.
- The central polygon is modeled as a 0.20 m step-down and a hard mapped no-go
  zone. The robot never enters it.
- Thresholds are assumed flush because they were not measured. The result is an
  Omniverse RTX procedural presentation scene, not a photogrammetric/as-built
  digital twin or a physical-deployment release.

Presentation video:
`media/videos/AI-SHA_Phase7F_Operator_Omniverse_Presentation.mp4` — 24.0 s,
1920 x 1080 at 24 fps, RTX PathTracing at 16 samples per pixel, SHA-256
`6fcc87d6faa91fe45ef8795e8a32e083f68af66f594763351eaaf39e150780e8`.
The presentation gate passes 19/19 and the visual QA contact sheet is
`media/AI-SHA_Phase7F_Operator_Omniverse_contact_sheet.jpg`. The acceptance record is
`results/administration_nav2_phase7f_operator_presentation_acceptance.json`;
the source live-integration record is
`results/administration_nav2_phase7e_static_fusion_integration_gate.json`.
Reproduce the complete replay, encode, contact-sheet and acceptance pipeline
with:

```bash
tools/run_administration_nav2_phase7f_operator_presentation.sh
```

## Phase 8A physical-localization preflight

The first sim-to-real phase now has a fail-safe, localization-only ROS 2 graph
for Rev D. It uses real time, a differential AMCL model, raw encoder odometry on
`/wheel/odom_raw`, BNO055 input on `/imu/data`, filtered output on
`/odometry/filtered`, and explicit single ownership of the `map -> odom ->
base_link` TF chain. The launch file contains no Nav2 controller, planner, motor
node, velocity smoother or `/cmd_vel` publisher.

The offline preparation gate passes 23/23. A two-second host probe correctly
remains blocked at 1/14 because this workstation is not the physical robot and
has no live LD19, BNO055, Rev D encoder odometry or localization TF graph. No
dummy odometry or simulated sensor data was substituted.

The audit also quarantines the production repository's older mecanum driver and
Nav2 profile: they use four-wheel holonomic kinematics and cannot be applied to
the two-wheel Rev D platform. Phase 8B now supplies the missing differential
software foundation, while its physical calibration and runtime gates remain
blocked. The target physical ROS
distribution must also be frozen: simulation is pinned to Jazzy while the
existing Pi/Jetson launch contract requires Humble, and cross-distro Nav2 is not
authorized. See
`PHYSICAL_LOCALIZATION_COMMISSIONING.md` and run:

```bash
tools/run_phase8a_physical_localization_preflight.sh
```

## Phase 8B Rev D differential encoder adapter

The new `src/aisha_rev_d_driver` package passes 30/30 offline acceptance checks
and 13/13 focused unit tests. It implements Rev D differential command math,
signed 32-bit encoder integration with rollover handling, and the verified
supplied ZLAC8015D V4 Series Modbus register layout. Supplier example frames for enable,
positive/negative target velocity and encoder reads reproduce byte-for-byte,
including CRC.

Phase 8B does not command the physical robot. Its default mode is a deterministic
5 RPM replay on isolated `/phase8b/replay/*` topics. The optional physical mode
can read status, fault, position and speed registers with Modbus function `0x03`
only; every outbound frame is checked immediately before transmission. The ROS
node has no `/cmd_vel` subscription, no TF broadcaster and no motor-enable or
target-velocity transport. Run the offline gate with:

```bash
tools/run_phase8b_rev_d_adapter_preflight.sh
```

The read-only transport also passes a Linux pseudo-terminal loopback: it sends
exactly three function-`0x03` requests for the configured position/speed,
status and fault ranges, then decodes signed position and speed values from the
returned byte stream. This is serial-path evidence, not physical-driver
evidence.

The passive hardware-attachment audit opened no serial port and sent zero
Modbus frames. It found only `ZLAC8015D V4.0.zip` in the supplied archive, no
received-unit label photo, no exact V4.2 manual and no stable
`/dev/serial/by-id` USB-RS485 device on this workstation. Follow
`PHASE8B_HARDWARE_ATTACHMENT.md` and rerun the passive audit after those items
are available. A passing audit authorizes only operator review before the
separate guarded read-only probe.

The expected driver label is V4.2 while the supplied communication document is
for the V4 Series, so exact hardware/manual compatibility is not assumed. The
candidate 16384 counts/revolution and 0.100 m radius have no physical odometry
credit until a marked revolution, loaded rolling circumference and both encoder
signs are measured. The future read-only probe blocks before opening the serial
device unless its complete operator checklist is explicitly confirmed. A motor-
write path and the 5 RPM wheels-lifted test remain a later reviewed gate.

Physical doorway motion remains prohibited. The reported 0.85 m opening is
below the Rev D 0.92 m physical minimum and 0.078 m narrower than the 0.928 m
production padded width. The accepted simulation presentation remains valid,
but its 0.030 m padding exception is not transferred to hardware.

## Latest dynamic-autonomy gate

Phase 7A freezes the accepted Phase 6 static mission and adds one separate,
deterministic, sensor-visible pedestrian crossing on high-speed hallway segment
1. The full live Nav2 mission again completed 12/12 legs, both office pivots and
the return home without collision or episode reset; its acceptance report passes
24/24.

The robot reached 0.74538 m/s before the encounter, stopped completely, waited
for the crossing and recovered to 0.73196 m/s. Minimum robot–pedestrian centre
distance was 1.19046 m. Both crown and front LiDAR see the proxy; its position is
evaluation telemetry only and is not supplied to either learned policy. A
front-scan threshold hands the encounter from the Phase 6 actor to the accepted
Phase 3N dynamic actor for 30 steps. That actor had learned authority for 29
encounter steps and requested braking on four; the independent protective latch
held the final stop for 52 steps and used the 360-degree scan to prevent an early
restart as the pedestrian cleared the robot's corner.

This establishes one repeatable stop-wait-resume simulation scenario—not crowd
navigation, a human-behaviour model or physical stopping distance. Reproduce
the retained Phase 7A stage with:

```bash
tools/run_administration_nav2_phase7_dynamic_integration.sh
```

Evidence: `results/administration_nav2_phase7_dynamic_mission.json`,
`results/administration_nav2_phase7_dynamic_bridge.json` and
`results/administration_nav2_phase7_dynamic_integration_gate.json`.

## Latest blocked-route gate

Phase 7B adds a visible 2.90 m full-width temporary barricade to the
mission-authorized east office hallway route. No alternate corridor is approved
for that leg, so the correct behavior is safe wait rather than entering other
administration spaces as an unscheduled detour. Nav2
first generated a 2,554-pose candidate; the independent supervisory validator
rejected it because the registered live front-LiDAR points came within 0.01371 m
of the path against a 0.46 m required radial clearance. The robot then held
stationary for the blockage interval, with 0.000132 m/s maximum measured speed
and less than 0.000001 m displacement.

After the barricade was physically removed, the mission requested a fresh
global path. The new candidate's minimum registered-LiDAR clearance was
1.02065 m with zero violating poses, so it was accepted and executed. The full
Principal/Vice-Principal mission completed 12/12 legs in 482.0 s; the formal
gate passes 26/26 while retaining the Phase 6 28/28 and Phase 7A 24/24 evidence.

The registered point cloud is derived from Isaac's actual front-LiDAR ray hits;
the barricade pose is evaluation/synchronization state only and is not supplied
to either learned policy or the path validator. The installed Nav2 obstacle
layer did not earn dynamic-marking credit, so this result is explicitly a
sensor-supervised candidate rejection, safe wait and fresh-plan acceptance—not
a Navfn infeasibility result, alternate-route proof, physical localization or
physical safety release. Reproduce it with:

```bash
tools/run_administration_nav2_phase7b_blocked_route_integration.sh
```

Evidence: `results/administration_nav2_phase7b_blocked_route_mission.json`,
`results/administration_nav2_phase7b_blocked_route_bridge.json` and
`results/administration_nav2_phase7b_blocked_route_integration_gate.json`.

## Latest native-costmap detour gate

Phase 7C fixes the Nav2 Jazzy observation-source height filter in a separate,
isolated profile. Nav2 had defaulted each source's maximum obstacle height to
0.0 m, silently rejecting valid AI-SHA returns at 0.25 m and 1.86 m despite the
layer-level 2 m limit. The Phase 7C profile explicitly sets 2.00 m for the front
scanner and 2.20 m for the crown scanner; the accepted administration profiles
and their evidence remain frozen.

The compact Isaac loop has two genuine footprint-feasible branches around a
central island. With no blocker, Nav2 selected the 10.13575 m top branch and the
future blocker-centre cost was zero. After a visible, scan-only barricade was
activated, Nav2's own published global costmap changed the centre to cost 253
and all 27 sampled cells across the branch became lethal/inscribed. A fresh
Nav2 request selected the spatially distinct 11.14159 m bottom branch. The
accepted Phase 6 learned 360-degree safety layer remained the final authority
over `/cmd_vel`; AI-SHA executed the detour collision-free and entered the
0.30 m goal disc while the barricade remained present. The formal gate passes
29/29 and retains Phase 6 at 28/28, Phase 7A at 24/24 and Phase 7B at 26/26.

Reproduce it with:

```bash
tools/run_phase7c_native_detour_integration.sh
```

Evidence: `results/phase7c_native_costmap_detour_mission.json`,
`results/phase7c_native_costmap_detour_bridge.json` and
`results/phase7c_native_costmap_detour_integration_gate.json`.

This is an isolated architecture gate. It does not authorize an alternate route
for the administration east-hallway leg and it earns no physical-localization,
stopping-distance, sim-to-real, safety-certification or deployment credit.

## Latest administration native-costmap gate

Phase 7D applies the Phase 7C height-filter correction in a new administration
runtime profile while leaving the accepted profile frozen. Both local and
global costmaps explicitly accept the 2.00 m front and 2.20 m crown scan heights;
`inf_is_valid` also lets no-return rays clear marks after an obstacle is removed.
All accepted footprint, speed, doorway, controller and reverse-disabled settings
are otherwise unchanged.

In the live measured-presentation scene, the 33 sampled east-hallway cells
changed from 0/33 lethal or inscribed to 33/33 after the scan-visible barricade
appeared. Nav2 changed its 12.7977 m direct plan into a 36.4601 m map-connected
detour through administration spaces outside the approved east-hallway mission
envelope. AI-SHA correctly rejected that unscheduled route and waited at no more
than 0.000192 m/s with effectively zero displacement. After physical barrier
removal, post-removal scan settling and an explicit global-costmap clear, the
same cells returned to 0/33; a fresh authorized 12.7977 m plan was executed at
up to 0.74560 m/s. The accepted Phase 6/3N learned safety remained in the loop,
with 0.37798 m minimum 360-degree clearance. The formal gate passes 32/32 and
retains Phase 6 28/28, Phase 7A 24/24, Phase 7B 26/26 and Phase 7C 29/29.

Reproduce it with:

```bash
tools/run_administration_nav2_phase7d_native_costmap_integration.sh
```

Evidence: `results/administration_nav2_phase7d_native_costmap_mission.json`,
`results/administration_nav2_phase7d_native_costmap_bridge.json` and
`results/administration_nav2_phase7d_native_costmap_integration_gate.json`.

The live Phase 7D gate is intentionally scoped to the first two administration
legs. A diagnostic full-mission attempt showed that newly visible office clutter
can make the post-visit Vice-Principal departure infeasible to Navfn. The older
accepted 12-leg Phase 7B gate retains doorway and office-pivot evidence, but it
does not close this new-profile limitation. Full-office native-costmap clutter
classification/refinement is therefore required before the final operator-facing
capture. This earns no physical-localization, stopping-distance, sim-to-real,
safety-certification or deployment credit.

## Latest full-office static-map/live-LiDAR fusion gate

Phase 7E closes the Phase 7D office-departure limitation without changing
AI-SHA's physical footprint, 0.030 m Nav2 padding, the reported 0.85 m minimum
door, or the learned checkpoints. The failure was traced to the same 0.85 m
jamb being inflated once by the static layer and again from dense live returns.
The fused profile now gives raw `/scan` and `/front_scan` clearing-only
authority, while map-filtered outputs give marking-only authority to endpoints
in mapped-free space. Known static returns are never removed from safety: their
obstacles remain present through the static layer.

The accepted live run completed 12/12 legs through the Vice-Principal and
Principal offices and returned home. Both in-office pivots are followed by a
zero-translation departure alignment inside the mapped pivot-clearance zone;
the final heading errors were 1.845 and 1.981 degrees. The full-width temporary
barrier still changed 0/33 east-hallway samples to 33/33 lethal/inscribed. The
36.5018 m unauthorized detour was rejected, safe-wait speed stayed below
0.000125 m/s with negligible displacement, clearing returned the samples to
0/33, and a fresh 12.7877 m direct plan executed. Observed straight-hallway
speeds reached 0.74572 and 0.74342 m/s, while doorway body speed remained below
0.05940 m/s. The formal gate passes 40/40 and retains Phase 6 28/28, Phase 7A
24/24, Phase 7B 26/26, Phase 7C 29/29 and Phase 7D 32/32.

Reproduce it with:

```bash
tools/run_administration_nav2_phase7e_static_fusion_integration.sh
```

Evidence: `results/administration_nav2_phase7e_static_fusion_mission.json`,
`results/administration_nav2_phase7e_static_fusion_bridge.json`,
`results/administration_nav2_phase7e_static_scan_fusion.json` and
`results/administration_nav2_phase7e_static_fusion_integration_gate.json`.

This is presentation-simulation evidence, not proof that an unmeasured physical
site is correctly mapped. It earns no physical localisation, stopping-distance,
sim-to-real, safety-certification or deployment credit. The next simulation
gate is the operator-facing Omniverse/Nav2 capture of this accepted stack.

## Latest live autonomy gate

The unchanged Rev D robot now also passes the measured-presentation static
mission under the coupled live stack:

```text
Nav2 global planner + DWB local controller
  -> measured doorway and central-polygon guard
  -> accepted Phase 3N learned 360-degree brake
  -> articulated wheel/contact physics
```

The verified run completed all 12 mission legs, entered and departed both
offices, performed both in-office 180-degree pivots, and returned home without
an episode reset. The mapped guard recorded two traversals of each doorway. Its
maximum measured doorway body speed was 0.06258 m/s against the 0.10 m/s limit,
maximum doorway tangent offset was 0.02151 m, and minimum predicted full-body
clearance from the inaccessible 0.20 m central drop was 0.27959 m. The learned
Phase 3N layer had authority on 689 control steps and applied braking on 192.
`results/administration_nav2_measured_integration_gate.json` passes 27/27.

The earlier measured-scene gate remains accepted at 0.30 m/s. Phase 6 now also
accepts a **simulation-only 0.80 m/s hallway tier** on straight route segments
1 and 5: 126/128 unseen
episodes succeeded, with one dynamic and one static collision. No chassis,
footprint, URDF, USD, mass, track or sensor geometry changed, and the 0.10 m/s
doorway limit remains fixed. The selected policy and 16/16 acceptance record are
`isaaclab/checkpoints/aisha_phase6_high_speed_080_model_223.pt` and
`results/phase6_high_speed_080_acceptance.json`. Its measured-administration
Nav2 integration and final RTX presentation gates are now complete. Ground
truth Isaac odometry is used for the presentation Nav2 map transform because
the assumed furnishing scene is not an AMCL benchmark; this earns no physical
localisation credit.

## Architecture decision

**Keep four matched swivel castors for this tall PoC**, because their broad
front/rear footprint is useful for static stability. Do not keep either of the
NF-301's fixed-direction castors, and do not bolt all six contacts rigidly to
one chassis.

The two drive wheels need guided, adjustable compliance from the first build:

- target 12-13 kgf normal load on each drive wheel at the 10 kg payload case;
- starting spring rate 20-25 N/mm per side, about 5-6 mm loaded compression;
- at least 3 mm rebound and 4 mm bump travel, with captured guides and stops;
- independent left/right response, or equivalent small roll of the drive beam;
- springs carry vertical load only; guides/stops carry fore/aft and yaw loads.

Battery location does not set wheel load in a six-contact system. Preload and
geometry do. Corner-weigh with both drive wheels on scales and equal-height
plates beneath every other contact; a single scale under one wheel changes the
geometry and gives a misleading result.

## Files

```text
urdf/aisha.urdf               59.25 kg design-empty baseline
urdf/aisha_max_payload.urdf   69.25 kg with the 10 kg PoC payload
config/aisha_drive.yaml       geometry, sources/status, limits, safety and hold points
config/demo_route.yaml        two-office demo route (Vice-Principal, Principal)
config/administration_assumptions.yaml  page-2 trace, appearance and disclosed assumptions
config/training.yaml          pinned Isaac Lab task and claim-boundary contract
scripts/build_administration.py         plan-derived Block A USD builder
scripts/render_administration_route.py  five-shot verified learned-trace Omniverse renderer
isaaclab/                    physics-driven RSL-RL training and evaluation task
isaaclab/tools/build_administration_live_assets.py  live scene/robot composer
isaaclab/scripts/play_block_a_route.py  checkpoint-driven live administration runner
isaaclab/tools/validate_administration_live_policy.py  live evidence validator
isaaclab/tools/make_final_omniverse_presentation_reel.py  final evidence-reel assembler
isaaclab/tools/validate_final_omniverse_presentation_reel.py  final reel validator
isaaclab/tools/validate_measured_administration_presentation.py  measured-scene final gate
tools/encode_route_video.py             verified MP4 encoder for rendered frames
tools/validate_administration_replay.py evidence-chain validation before rendering
tools/generate_aisha_urdf.py  canonical source for both URDFs
tools/validate_urdf.py        mass, inertia, frames, drive and stability checks
```

Edit the generator and regenerate both URDFs; do not hand-edit the XML.

```bash
python3 tools/generate_aisha_urdf.py
python3 tools/validate_urdf.py
```

## What changed from Rev C

| Item | Rev C | Rev D |
|---|---:|---:|
| Empty / loaded design mass | 56.05 / 66.05 kg | **59.25 / 69.25 kg** |
| Battery mass | 13 kg allowance | **11.25 kg datasheet value** |
| Structure mass | 6 kg | **10 kg conservative parts allowance** |
| Tray model | 25 mm solid proxy, 4 kg | **3 mm sheet + posts, 5 kg** |
| Deck envelope | 55 mm | **32 mm from 210-178 mm published heights** |
| Normal wheel effort | 18 N.m peak | **6 N.m rated** |
| Front LiDAR scan height | 0.20 m, intersecting deck | **0.25 m, above deck** |
| Camera frames | body frame only | **body + ROS optical frame** |
| IMU frame | absent | **present; as-built XYZ pending** |
| Overall envelope | incorrectly ~0.91 m long | **0.768 W x 1.180 L x 1.190 H m** |
| Stability | nominal contact locations | **30 mm inward caster-trail allowance** |

The mass increase is not payload growth. It corrects parts previously omitted or
under-counted: the RHS beam/spine, axle and backing plates, compliance hardware,
fasteners, and tray posts.

## Model summary

Convention: **+X forward, +Y left, +Z up**; `base_link` is on the floor at the
deck centrelines.

| Item | Design baseline |
|---|---|
| Drive wheels | V2.18 assumption: Ø200 x 48 mm at x=0, y=±0.360, z=0.100 m |
| Castor swivel axes | x=±0.350, y=±0.255 m; Ø130 mm contacts |
| Deck | 910 x 610 mm; top z=0.210 m |
| Tray | 805 x 610 x 3 mm; surface z=0.530 m |
| Mast / head | mast x=0.420 m; head Ø450 at x=0.500, z=0.925 m |
| Crown LiDAR | scan frame z=1.170 m; housing top z=1.190 m |
| Front LiDAR | scan frame x=0.455, z=0.250 m |
| Envelope | **0.768 m wide, 1.180 m long, 1.190 m high** |

Validated design-mass results:

| | Empty | 10 kg payload |
|---|---:|---:|
| Mass | 59.25 kg | 69.25 kg |
| CG (x, z) | (+0.0886, 0.3330) m | (+0.0682, 0.3723) m |
| Conservative front static tip | 34.8° | 34.1° |
| Conservative lateral static tip | 44.1° | 41.7° |

The tip calculation moves every castor contact 30 mm inward from its swivel
axis. It is still a static estimate from unmeasured masses. Repeat it after the
finished robot is weighed and corner-weighed.

## Simulation abstractions - do not confuse them with hardware

### Castors

The physical robot has four 360-degree swivel castors. The baseline URDF uses
fixed low-friction spheres because detailed two-DOF castors commonly chatter in
PhysX and consume solver iterations. Assign `castor_low_friction` from the YAML.

This proxy is suitable for navigation integration and deterministic demos. It
cannot validate swivel reversal, breakaway torque, flutter, floor marking,
pivot current or odometry transients. Create a separate high-fidelity caster
test scene before claiming sim-to-real performance.

### Drive compliance

The physical drive carrier is guided and compliant; the baseline URDF is rigid.
Flat-floor navigation tests may use the baseline. Threshold, uneven-floor,
contact-load and traction-transfer tests need an articulated Isaac asset with
vertical/roll compliance and the measured spring curve.

### Friction

All friction values are starting assumptions. Measure the delivered tyre and
castor materials on the actual polished tile, then calibrate the USD materials.

## Motor and driver limits

Only `left_wheel_joint` and `right_wheel_joint` are actuated, both about +Y.
Positive velocity must drive toward +X; verify after import.

The URDF effort limit is **6 N.m**, the motor's rated torque. The reported
18 N.m peak must be controller-timed and thermally/current limited; it is not a
continuous physics limit. One ZLAC8015D is provisionally suitable for the two
6 A-rated motors, but the exact V4.2 hardware label, current allocation and
thermal behaviour still require verification.

The official driver manual gives a single 15 A typical / 30 A maximum output
table without identifying a per-channel value. Supplier correspondence recorded
by the project says aggregate. Do not command simultaneous 18 A motor peaks.

## Source control and fabrication hold

The wheel reference PDF in this workspace is explicitly **V2.0**, with Ø206 mm
and 48.8 mm tread. The model is for **V2.18**, using supplier-stated Ø200±2 mm
and 48±1 mm values. Those variants are not interchangeable for axle height or
bracket release.

Before cutting metal:

1. Obtain the exact V2.18 single-shaft dimensioned drawing and exact V4.2 driver manual.
2. Measure both received wheels: free OD, loaded circumference, tread/body width,
   shaft and mounting-face stack.
3. Measure new caster overall height, trail, plate pattern and swivel-axis locations.
4. Inspect and photograph the NF-301 underside. Have the concentrated dynamic
   load path, fasteners and anti-loosening method reviewed by a competent engineer/fabricator.
5. Measure the narrowest door clear width. The 0.768 m robot needs **at least
   0.920 m** for the controlled demo target of roughly 75 mm clearance per side.
6. Weigh the stripped deck, structure, head and finished assembly; update the generator.

Published NF-301 trolley capacity is a manual cargo rating. It does not certify
the modified deck for powered, cantilevered, dynamic wheel loads.

## Import notes

Use `isaac_sim_import` in the YAML:

- floating base; merge fixed joints; import authored inertia;
- collision from collision geometry only; self-collision initially off;
- do not replace wheel cylinders with capsules;
- velocity targets, force drive, zero position stiffness;
- confirm 59.25 / 69.25 kg after import;
- confirm crown scan frame 1.170 m, front scan frame 0.250 m, and the camera
  optical convention (+Z forward, +X right, +Y down);
- confirm wheel contact force is plausible, but do not interpret rigid-URDF
  distribution as proof of physical spring preload.

## Workstation implementation (Isaac Sim 5.1)

The bundle now includes a headless, reproducible first implementation for Isaac
Sim 5.1. Run it with the simulator's Python, not the system interpreter:

```bash
ISAAC_ROOT=/home/robot-wst/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
python3 tools/generate_aisha_urdf.py
python3 tools/validate_urdf.py
python3 tools/inventory_workstation.py
python3 tools/test_controller.py
"$ISAAC_ROOT/python.sh" scripts/import_urdf.py --headless
"$ISAAC_ROOT/python.sh" scripts/build_validation_scenes.py --headless --payload loaded
"$ISAAC_ROOT/python.sh" scripts/run_validation.py --headless --suite smoke --payload loaded
"$ISAAC_ROOT/python.sh" scripts/run_validation.py --headless --suite full --payload loaded
"$ISAAC_ROOT/python.sh" scripts/build_administration.py --headless --payload loaded \
  --plan /path/to/DownloadBuildingRequestApprovedPlan.pdf --presentation-assumptions
"$ISAAC_ROOT/python.sh" scripts/render_administration.py --headless
python3 tools/validate_administration_replay.py
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p scripts/render_administration_route.py \
  --headless --width 1920 --height 1080 --fps 24 --seconds-per-shot 3 \
  --renderer PathTracing --path-tracing-spp 4
python3 tools/encode_route_video.py --fps 24 --crf 18 --preset slow
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/tools/build_administration_live_assets.py
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/play_block_a_route.py \
  --task Isaac-AISHA-Administration-Live-Direct-v0 \
  --checkpoint isaaclab/logs/rsl_rl/aisha_block_a_sensor_nav/2026-08-20_22-33-50_ld19_flush_threshold_v9/model_599.pt \
  --output-report results/administration_live_policy_video_report.json \
  --video-folder media/videos/administration_live_policy \
  --max-steps 5600 --dwell-seconds 1.0 --trace-interval 3 \
  --camera-eye -3.4 0.0 2.15 --camera-lookat 0.35 0.0 0.58 \
  --seed 6084 --headless
python3 isaaclab/tools/make_administration_live_policy_presentation_video.py \
  --input media/videos/administration_live_policy/aisha-block-a-learned-route-step-0.mp4 \
  --run-report results/administration_live_policy_video_report.json \
  --output media/videos/AI-SHA_Administration_Live_Policy_3x.mp4 \
  --report results/administration_live_policy_presentation_video_report.json
python3 isaaclab/tools/validate_administration_live_policy.py
```

Generated evidence is written to `results/`. The import report records the exact
URDF hashes, frame paths, mass attributes, drive settings and importer version.
Validation traces include pose, yaw, base velocity, target/actual wheel velocity,
physics rate, seed, payload, and explicit blocked tests.

The rigid six-contact proxy is statically indeterminate in PhysX. With all six
geometries exactly coplanar, one driven wheel can be left effectively unloaded.
The imported deterministic assets therefore apply a symmetric **1 mm simulation
rest offset to both driven-wheel colliders**. This is a solver seating bias only;
it is not a tyre-radius change, spring preload, or evidence of physical contact
load. It is recorded in `config/physics_materials.yaml` and must remain disclosed.

Isaac Sim 5.1's bundled URDF importer 2.4.30 may print transient unresolved-reference
warnings while it authors mass-only fixed-frame links. `scripts/import_urdf.py`
repairs those generated targets before saving and records the repaired paths in
`results/import_report.json`; the saved assets reopen and validate without those
warnings.

`scenes/validation_thresholds.usd` contains parameterized 5/10/20 mm geometry,
but it is deliberately marked as blocked for contact conclusions. Threshold
validation requires the articulated compliant carrier, measured spring curve,
and measured caster properties.

`scripts/build_administration.py` remains a strict input gate by default. The
explicit `--presentation-assumptions` mode builds `scenes/administration.usd`
from page 2 of the approved ground-floor plan. The central 12.75 m atrium,
2.80 m east hallway, east Vice-Principal placement and angled south-east
Principal placement now control the route topology. Corridor/office finishes,
furniture and lighting follow the supplied walkthrough video. Its original
wide-door mode uses disclosed 1.40 m presentation openings; the newer measured
overlay supersedes those with the reported 0.85 m VP opening and a disclosed
0.90 m Principal assumption. Thresholds remain assumed flush because the rigid
castor proxy cannot support threshold-contact conclusions. Scene metadata and
reports keep physical release false.

The walkthrough-derived atrium columns are also presentation assumptions, not
surveyed structure. Their positions are declared in the scene configuration and
the southeast column is offset from the replay corridor. Validation samples every
interpolated learned-trace segment against a conservative 0.95 m centre-distance
gate (the robot's 0.768 x 1.180 m circumscribed footprint, column radius and a
buffer) before any final render is accepted.

The initial Phase 1 administration replay used five reproducible camera shots:
atrium/east-hall departure, Vice-Principal visit, transfer to the angled
Principal suite, Principal visit, and mission return. The robot was not moved by
a separately authored cinematic route. Each rendered pose was selected from the
1,736 recorded samples in the successful seed-6084 continuous Isaac Lab
wheel-physics run. Its shot segment ranges covered all 12 route segments exactly once.
The overlay identifies learned-policy, physical-turn-supervisor and office-dwell
records and states that this is visual replay rather than live policy execution.
`tools/validate_administration_replay.py` verifies that evidence chain before
rendering. `tools/encode_route_video.py` checks every frame and writes
`media/videos/administration_learned_trajectory_replay.mp4`.

The newer live integration is distinct from that pose replay.
`Isaac-AISHA-Administration-Live-Direct-v0` composes the complete administration
architecture, furniture and lighting into the Isaac Lab environment, deactivates
the replay robot and nested physics scene, and spawns the loaded articulated
AI-SHA with a collisionless presentation shell fixed to its real base link. The
unchanged sensor-policy checkpoint then drives the robot through PhysX in this
scene. Its final seed-6084 run completed all 12 Principal/Vice-Principal route
segments in 5,388 policy ticks (179.6 simulated seconds), with no collision or
reset. The learned policy supplied 4,378 aligned-drive ticks; the disclosed
physical wheel-command supervisor supplied 950 turn ticks and 60 dwell ticks.
No root-transform animation is used. The full 1x capture and labelled 3x cut are
`media/videos/administration_live_policy/aisha-block-a-learned-route-step-0.mp4`
and `media/videos/AI-SHA_Administration_Live_Policy_3x.mp4`.

RTX rendering has a deliberate preflight gate. It initially stopped correctly
when the workstation was booted into kernel `7.0.0-30-generic` without a
matching NVIDIA module. After booting the installed `6.17.0-35-generic` kernel,
the RTX 5080 and driver 580.159.03 were detected and the final capture completed:
360 PathTracing frames at 1920 x 1080, 24 fps and 4 samples per pixel. FFmpeg
7/libx264 encodes the 15.0 s presentation film at CRF 18 and records the final
video hash in `administration_learned_replay_render_report.json`. The GPU gate
may be bypassed only for an environment with another known-valid NVIDIA
discovery path by passing `--skip-gpu-preflight`.

AI-SHA-Production was reviewed at commit `8893535` for sensor and general-system
information only. The approved plan was supplied separately and is recorded by
filename, page and SHA-256 in `config/administration_assumptions.yaml`. The
production repository confirms the deployed LD19, RealSense D435, and BNO055
contracts recorded in `config/sensors.yaml`; its older mecanum chassis footprint
is intentionally not applied to this Rev D differential-drive model. See
`results/PRODUCTION_REPOSITORY_REVIEW.md`.

## Isaac Lab learning foundation

`isaaclab/` registers the original state-observation doorway baseline,
`Isaac-AISHA-BlockA-SensorNav-Direct-v0`, the live
`Isaac-AISHA-Administration-Live-Direct-v0` task, and two Phase 2 training/gate
tasks. The live task keeps the same 36
course ray ranges, ten goal/vehicle terms and two actions while replacing the
simplified training course with the full presentation scene. The policy commands
only the two wheel-joint velocity targets at 30 Hz while PhysX runs at 120 Hz;
direct root-transform motion is forbidden outside normal resets.

The selected seed-144 sensor run trained for 600 PPO iterations across 32
parallel environments: 614,400 simulated policy transitions. Its deterministic
held-out evaluation used the distinct seed 5084 and an equal quota of 48
episodes for every one of the 12 directed route segments. It achieved 576/576
successes, zero collisions and zero timeouts, including every Principal and
Vice-Principal entry/exit leg.

A separate continuous seed-6084 training-course playback chained the same 12
segments in 173.6 simulated seconds. That Phase 1 checkpoint then completed the
full administration scene in 179.6 simulated seconds with disclosed turn and
dwell supervision. Phase 2 supersedes it for the main technical demonstration:
the selected checkpoint supplies every wheel action, including office pivots
and departures, with no turn or dwell overrides and no root-transform motion.

The Phase 2 presentation capture and its 3x edit are in `media/videos/`. These
results support a real Isaac Lab/Omniverse policy-only training demonstration.
The scene remains walkthrough/plan-derived rather than a photogrammetric twin,
ray sensing is not an RTX LD19 model, and Nav2, measured site geometry,
Phase 3 acceptance and sim-to-real release remain open gates.
Reproduction commands and limitations are in `isaaclab/README.md`.

## Phase 2: policy-only turning and full-route control

Phase 2 is checkpoint fine-tuning rather than a new policy trained from zero.
It resumed the exact passing Phase 1 `model_599.pt` checkpoint in 32 parallel
environments. The executed search comprised 1,200 incoming-transition
iterations plus a 300-iteration moving-handoff refinement run: 1,536,000 policy
transitions in total. Starts use the actual incoming route heading with ±15
degree jitter and 0.30-0.50 m/s initial motion, plus observation-level
LD19-style range noise/dropout. A fading route hint resolves only the
left/right ambiguity near an exact 180-degree office reversal; it never
overrides an action.

The selected checkpoint is packaged as
`isaaclab/checkpoints/aisha_phase2_policy_model_1850.pt` (source run
`2026-08-22_09-12-57_phase2_moving_transition_seed256/model_1850.pt`), SHA-256
`3ab596c61259784657b36fe4ee937da8495ce9621a43bd04e2c8a0bf6e0b1880`.
It passed 570/576 balanced held-out segment episodes (98.96%; six collisions,
no timeouts) and 46/48 full chained routes (95.83%; two collisions, no
timeouts). The deterministic live administration evidence then completed all
12 segments in 4,885 policy steps / 162.83 simulated seconds with zero
collisions, zero turn-supervisor steps, zero dwell overrides and no
root-transform animation. `results/phase2_end_to_end_validation.json` passes
all 18 evidence checks.

The post-training pipeline requires 48 balanced transition episodes per
segment, 48 complete chained-route episodes, a policy-only training-course run,
and a policy-only live administration capture.
`isaaclab/scripts/run_phase2_training.sh` performs checkpoint/GPU preflight and
launches fine-tuning; `isaaclab/scripts/run_phase2_gates.sh /path/to/model_N.pt`
runs the declared evaluation and presentation pipeline.

The presentation camera requests a 3.8 m rear distance and 2.4 m height. It is
anchored to each route leg rather than the robot's instantaneous pivot, and a
three-ray visibility fan moves it inward only where a wall or doorway would
occlude the view. The final 162.8 s raw capture has no post-start uniform
camera-occlusion frames. The motion-preserving 3x presentation file is
`media/videos/AI-SHA_Phase2_Administration_Policy_Only_3x.mp4` (54.13 s).

Training and capture ran on the restored `6.17.0-35-generic` kernel with the
NVIDIA 580.159.03 driver and RTX 5080. Passing Phase 2 is simulation evidence,
not a physical-autonomy release: dynamic people/furniture, broader material and
sensor randomization, Nav2, measured doors/thresholds and sim-to-real
commissioning are still required.

### Administration visual upgrade

The first walkthrough-grounded visual upgrade is now complete. The
administration USD uses deterministic procedural PBR textures for polished
terrazzo, dark walnut, light oak and mottled grey finishes, plus a denser
ceiling grid, LED fixtures, vents, glazed office partitions, detailed furniture
and a refined collisionless AI-SHA presentation shell. These additions do not
change the route or collision geometry. The walkthrough controls appearance;
the approved page-2 Block A plan remains the geometry authority.

`scripts/render_administration_route.py` presents the successful Phase 2
policy-only trace through six human-height cameras: atrium departure, east hall,
Vice-Principal visit, Principal-suite turn, Principal visit and atrium return.
It selects only recorded wheel-physics poses; it does not author or interpolate
a cinematic route. The resulting 1280 x 720 PathTracing film contains 240
frames at 20 fps and 8 samples per pixel:

`media/videos/AI-SHA_Phase2_Administration_Visual_Upgrade.mp4`

Its SHA-256 is
`4b02865f831d0d1e0db7bf159d3d1ac09f34e24c2d693eeaf052280817350849`.
`results/phase2_administration_visual_replay_validation.json` passes all 23
scene, checkpoint, policy-only control, route-coverage and clearance checks.
The film is explicitly labelled as an Omniverse visual replay of the verified
live run, not as simultaneous policy execution. The separate raw/live Phase 2
capture remains the evidence for real-time learned-policy inference in the
administration scene.

### Live learned-policy ensemble in the upgraded scene

The current presentation evidence now combines those two previously separate
strengths: simultaneous learned-policy inference and PhysX wheel/contact motion
inside the visually upgraded administration USD. A PPO live-adaptation base
policy controls route segments 0-5 and 7-11. A behavior-cloned specialist,
trained from 16/16 successful pivot-then-drive demonstrations in the same live
physics and observation contract, controls only segment 6 (the hallway return
to Principal-suite turn). The high-level route planner selects between the two
declared learned skills; it never overrides a wheel action.

Both packaged checkpoints and their hashes are recorded in
`isaaclab/checkpoints/administration_policy_ensemble.json`. The final seed-7084
cinematic run completed all 12 waypoints in 5,160 learned-policy steps / 172.0
simulated seconds: 4,730 base-policy steps and 430 specialist steps, with zero
turn-supervisor steps, zero dwell overrides, zero collisions and no root-pose
animation. `results/phase2_administration_live_cinematic_validation.json`
passes all 42 evidence checks, including the six-camera coverage, checkpoint
hashes, action accounting, telemetry, raw video, unchanged-motion edits and the
absence of post-start uniform camera occlusion.

Presentation files:

- `media/videos/AI-SHA_Phase2_Administration_Live_Cinematic_3x.mp4` — 57.2 s
  main film, SHA-256
  `57d68c8d8c980e84424466b2673cede8bbda57405024cd895fc94a754b77ee60`.
- `media/videos/AI-SHA_Phase2_Administration_Live_Cinematic_Teaser_12x.mp4` —
  14.3 s teaser, SHA-256
  `934ab50c7cf44e0fd1f1ce5da61fb391b1accc086ba19d72f12d050d93fcc075`.
- `media/videos/phase2_administration_live_cinematic/aisha-block-a-learned-route-step-0.mp4`
  — complete 172.0 s raw evidence, SHA-256
  `bfa95fc901c35f14fe4142aafcfd1fd1e79d803a9db5be0139f85b32d2567457`.

The overlay explicitly discloses “PPO base + imitation specialist.” This is
authentic simulation training evidence, but the procedural scene is not claimed
as photorealistic or measured as-built, and the ensemble is not a physical
safety release.

## Phase 3: moving obstacles, domain randomization and RTX refinement

Phase 3 ran as conservative fine-tuning from the exact accepted
administration base checkpoint `model_2150_rehearsal.pt` (SHA-256
`e4c072c61a8f8f65c58c9a4780600c8b81ce713cc546d2eae8f96374980b0a0f`).
It preserves the 46-value policy observation and two-wheel action contracts.
Up to two physical, ray-visible person capsules cross selected open route legs;
the policy receives only its existing goal/state terms and 36 range values.

Episode randomization now covers LD19-style observation noise/dropout/bias,
0-2 steps of command latency, motor strength, wheel radius/track, joint damping,
base mass/inertia and robot friction. A four-environment, 180-step Isaac Lab
runtime smoke passed 11/11 checks. The evaluator now explicitly forces full
Phase 3 curriculum strength during fresh evaluation; the earlier 43/48
`model_2525.pt` screen had started at strength zero and is retained only as a
static diagnostic, not dynamic-obstacle evidence.

The revised interaction model places physical person crossings on seven open
route legs, excludes the tight segment-6 U-turn, and pauses a person proxy when
it comes within a disclosed 1.10 m reciprocal-yield radius. The policy still
receives only the same 46-value state/LiDAR frame and controls both wheel
targets. The hash-locked Phase 2 proxy checkpoint consequently retained 32/32
segment-6 successes under full physics and sensor randomization. A clean
600-iteration / 614,400-transition continuation then produced leading screening
candidate `model_2225.pt`: 24/48 full-strength deterministic successes, 12/48
collisions (4 dynamic and 8 static), 12 timeouts, and 4/4 segment-6 successes.
It is not accepted.

The temporal gate has now also been executed. A one-layer 46-unit GRU student
was distilled from the hash-locked Phase 2 teacher for 200 iterations / 204,800
transitions; behavior loss fell from 3.1066 to 0.00458. Its recurrent PPO
continuation then ran 800 iterations / 1,638,400 transitions. Leading checkpoint
`model_700.pt` achieved 26/48 successes, 15 collisions and 7 timeouts. On the
same seed, it completed three more episodes than feed-forward `model_2225.pt`
but incurred three additional dynamic collisions. It is therefore not promoted.

The bounded safety-residual gate has now also been executed. Feed-forward Phase
3 `model_2225.pt` is hash-locked and frozen inside the environment. A new
64-unit GRU may reduce forward speed all the way to zero and attenuate steering
by at most 25%; it cannot increase speed, reverse, increase steering magnitude,
or flip steering sign. The live Isaac runtime smoke passed all 15 action-boundary
checks. Recurrent PPO then ran for 600 iterations / 1,228,800 transitions at
full randomization strength from the first iteration.

Checkpoint `model_175.pt` won a separate seed-9502 selection screen. On the
decisive seed-9501 48-episode comparison it reduced dynamic collisions from 5
to 3, but successes fell from 25 to 21 and static collisions rose from 2 to 8;
total collisions increased from 7 to 11. It is rejected and the presentation
policy remains unchanged. Slow/stop control alone cannot repair static
clearance errors already present in the frozen route actor. The next gate should
retain the hard speed boundary while adding only clearance-verified bounded
lateral correction, or use a map-aware local planner beneath an independent
protective-stop layer.

That follow-up Phase 3L gate has now been executed. The frozen route actor
remains hash locked. A 64-unit recurrent policy may brake and request at most
±0.35 rad/s of steering correction, but its request is sent to the wheels only
when a one-second, five-sample rectangular-footprint forecast preserves both
clearance and route alignment. A separate hysteretic front-LiDAR stop runs
after randomized latency and can always remove forward motion while preserving
steering. The Isaac runtime gate passed 21/21 checks.

Phase 3L PPO ran for 600 iterations / 1,228,800 transitions. `model_200.pt` won
two unseen-seed screens. In the decisive seed-9701 comparison it achieved 29/48
successes, 1 dynamic collision, 4 static collisions and 14 timeouts. The same
frozen route without Phase 3L achieved 19 successes, 13 collisions and 16
timeouts; the zero-output hard-stop controller achieved 19 successes, 6
collisions and 23 timeouts. Model 200 is therefore promoted as the leading
protected-navigation architecture candidate. It is not the accepted
presentation or deployment policy: the result does not meet the declared 90%
success / 2% dynamic / 5% static thresholds or the 64-episode-per-segment
protocol. The next targeted gate is office-departure pivot recovery on segments
4 and 9 plus segment-6 static-clearance recovery.
The frozen route actor and Phase 3L leader are packaged at
`isaaclab/checkpoints/aisha_phase3_frozen_route_model_2225.pt` and
`isaaclab/checkpoints/aisha_phase3l_clearance_planner_model_200.pt`, with the
same hashes used by the runtime and comparison evidence.

Phase 3M has now executed the targeted recovery gate. Diagnosis found that
Phase 3 domain randomization was applying drive-wheel friction to the four
fixed-sphere caster proxies; the targeted task restores their declared
0.15-0.25 static / 0.10-0.20 dynamic low-friction bands. The task also models
the repository-specified 6 Nm continuous / 18 Nm, 3-second peak motor boundary.
The runtime smoke passes 38/38 checks.

PPO fine-tuning ran without deterministic recovery actions for 150 iterations /
307,200 transitions. Checkpoint 125 is
packaged at `isaaclab/checkpoints/aisha_phase3m_hybrid_recovery_model_125.pt`
(SHA-256 `bc8727e3ea42c8b29ca74fa5a535fd37b1600633ffd8bf606b02220a557c1a0d`).
The separate runtime task retains the trained recurrent residual for normal transit and
adds two narrow, auditable recovery primitives: a clearance-projected,
goal-signed 0.55 rad/s stopped pivot only on office-departure segments 4 and 9,
and 0.10/0.08 m/s predictive creep on tight return segments 6, 10 and 11.

On seed 9701 it achieved 46/48 successes, 2 dynamic collisions, no static
collisions and no timeouts; segments 4, 6 and 9 each improved to 4/4. An
independent seed achieved 43/48 with 4 dynamic and 1 static collision. Phase 3M
is therefore a packaged architecture candidate, not a fully accepted or
presentation-replacing policy. Dynamic-person performance remains
seed-sensitive, and the declared 64-episode-per-segment, Phase 2 regression and
12 live-scene gates are still pending. The next gate is a 360-degree
dynamic-obstacle safety layer while freezing the successful pivot/clearance
stack.

Phase 3N has completed that gate. The Phase 3M recurrent controller and its
underlying route actor are loaded from SHA-256-locked checkpoints and kept
non-trainable. A separate 64-unit GRU receives the unchanged 46-value sensor
observation and emits one action: a negative request may only reduce forward
motion when full-ring LiDAR clearance is closing on a declared pedestrian leg.
It cannot steer, accelerate, reverse, or move the robot root, and the duplicate
outer emergency override is disabled. PPO ran for 200 iterations across 32
environments, totaling 409,600 transitions; checkpoint 50 is packaged at
`isaaclab/checkpoints/aisha_phase3n_dynamic_safety_model_50.pt` (SHA-256
`11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b`).

The two-seed matched screen improved the frozen stack from 95/96 successful
episodes with one dynamic collision to 96/96 with none. The declared full gate
then passed: 744/768 successes (96.875%), 8 dynamic contacts (1.042%), 12 static
contacts (1.563%), four timeouts, and a 92.188% minimum success rate on every
pedestrian-enabled segment. The separate full-randomization/no-pedestrian
regression passed 48/48 with zero contacts, and the walkthrough-matched live
administration gate passed 12/12 dynamic scenarios with zero contacts or
timeouts. This accepts the checkpoint for the declared Phase 3 simulation
scope; it is not a physical human-safety claim or deployment release.

The final seed-10201 Omniverse run uses the same packaged checkpoint in the
walkthrough-matched administration USD. It completes all 12 waypoints and both
office visits in 4,941 learned-policy steps / 164.7 simulated seconds, with no
turn supervisor, dwell override, recurrent-state reset, collision, or root
animation. The clean presentation scenario intentionally disables stochastic
person proxies; dynamic safety is evidenced by the separate accepted 768-episode
and 12-live-scenario gates. For film continuity it uses disclosed 0.22 m office
visit stops, a 0.20 m Principal-departure stop, and omits the redundant
segment-10 predictive-creep guard. The 54.77 s 3x cut changes temporal sampling
and labels only; it is available at
`media/videos/AI-SHA_Phase3N_Administration_Final_Omniverse_3x.mp4`.

Phase 4A adds the short dynamic-safety insert that the clean mission film
deliberately omitted. A separate presentation-only task fixes the accepted
Phase 3N checkpoint on segment 7, from the Principal turn to the Principal
office approach, and triggers one deterministic 0.48 m/s pedestrian crossing.
The pedestrian is a stylized kinematic proxy with a torso collision envelope;
its position is evaluation truth and is never added to the checkpoint's
observation. The formal randomized and live Phase 3N gates are unchanged.

Seed 10401 completed the route leg in 395 policy steps with zero pedestrian or
static contacts. During the encounter, the 360-degree closing-clearance gate
granted the learned actor authority on 16 samples, the learned brake output
peaked at 14.32%, the frozen protective-stop stack held the robot for 2.23 s,
and forward speed recovered to 0.463 m/s before the goal. The presentation
overlay labels learned brake authority and the separate protective-stop state
independently; it does not attribute the entire physical stop to the outer
actor. The 13.13 s film is
`media/videos/AI-SHA_Phase4A_Administration_Dynamic_Safety_Showcase.mp4`.
This is live Isaac Sim/Isaac Lab checkpoint evidence for a controlled
presentation scenario, not a human-behaviour model or physical safety release.

The final presentation reel combines those two accepted films without changing
their motion. It includes every frame of the 54.77 s clean mission and every
frame of the 13.13 s pedestrian encounter once and in order, separated by
clearly labeled title cards. The result is 76.9 s, 1280 x 720 at 30 fps, passes
27/27 evidence-chain checks, and is available at
`media/videos/AI-SHA_Final_Omniverse_Administration_Presentation.mp4`. It shows
the complete Vice-Principal and Principal office mission followed by the
stop-wait-resume safety insert. Geometry and door clearances remain disclosed
presentation assumptions; this film is not a physical safety or deployment
release.

That Phase 3N/4A reel remains historical accepted evidence for the original
wide-door scene. The Rev N measured-administration deliverable at the top of
this README supersedes it for the current 0.85/0.90 m presentation geometry.

The administration USD was rebuilt with page-2 Block A printed-dimension anchors
(12.75 m atrium, 2.80 m hall, 7.80 x 6.30 m conference room and 4.73 m
Principal frontage) and tagged `GEOMETRY-RTX-PHASE3-A`. Four procedural finish
families now use albedo, perceptual-roughness and tangent-normal maps; offline
office stills render with RTX PathTracing at 64 samples per pixel. The geometry
and material validator passes 17/17. These are plan-calibrated presentation
assets—not site measurements, photogrammetry or proof of physical clearance.

Reproduce or inspect this phase with:

```bash
isaaclab/scripts/run_phase3_dynamic_training.sh
isaaclab/scripts/run_phase3_recurrent_distillation.sh
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/bootstrap_recurrent_ppo.py \
  --distilled-checkpoint isaaclab/logs/rsl_rl/aisha_block_a_sensor_nav/\
2026-08-22_15-18-36_phase3i_recurrent_distillation_seed8801/model_199.pt \
  --output-checkpoint isaaclab/logs/rsl_rl/aisha_block_a_sensor_nav/\
phase3_recurrent_bootstrap_seed8801/model_0.pt \
  --report results/phase3_recurrent_bootstrap_report.json
isaaclab/scripts/run_phase3_recurrent_ppo.sh
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/bootstrap_safety_residual_ppo.py \
  --output-checkpoint isaaclab/logs/rsl_rl/aisha_block_a_sensor_nav/\
phase3_safety_residual_bootstrap_seed9001/model_0.pt \
  --report results/phase3_safety_residual_bootstrap_report.json
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/smoke_phase3_safety_residual.py --num-envs 4 --steps 90 --headless
isaaclab/scripts/run_phase3_safety_residual.sh
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/bootstrap_clearance_planner_ppo.py \
  --output-checkpoint isaaclab/logs/rsl_rl/aisha_block_a_sensor_nav/\
phase3l_clearance_planner_bootstrap_seed9601/model_0.pt \
  --report results/phase3l_clearance_planner_bootstrap_report.json
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/smoke_phase3_clearance_planner.py \
  --num-envs 4 --steps 120 --headless
isaaclab/scripts/run_phase3_clearance_planner.sh
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/smoke_phase3_clearance_planner.py \
  --task Isaac-AISHA-BlockA-Phase3-TargetedRecovery-SensorNav-Direct-v0 \
  --num-envs 4 --steps 120 --headless \
  --output-report results/phase3m_targeted_recovery_smoke_report.json
isaaclab/scripts/run_phase3_targeted_recovery.sh
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p isaaclab/scripts/evaluate.py \
  --task Isaac-AISHA-BlockA-Phase3-TargetedRecovery-SensorNav-Direct-v0 \
  --checkpoint isaaclab/checkpoints/aisha_phase3m_hybrid_recovery_model_125.pt \
  --output results/phase3m_hybrid_model125_balanced_seed9701.json \
  --episodes 48 --episodes-per-segment 4 --num_envs 48 --seed 9701 --headless
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/smoke_phase3_dynamic_safety.py \
  --num-envs 4 --steps 120 --headless \
  --output-report results/phase3n_dynamic_safety_smoke_report.json
PHASE3N_ITERATIONS=200 isaaclab/scripts/run_phase3_dynamic_safety.sh
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p isaaclab/scripts/evaluate.py \
  --task Isaac-AISHA-BlockA-Phase3-DynamicSafety-SensorNav-Direct-v0 \
  --checkpoint isaaclab/checkpoints/aisha_phase3n_dynamic_safety_model_50.pt \
  --output results/phase3n_model50_full_acceptance_seed10101.json \
  --episodes 768 --episodes-per-segment 64 --num_envs 768 --seed 10101 \
  --require-acceptance --headless
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/smoke_phase3_dynamic.py --num-envs 4 --steps 180 --headless
TERM=xterm /home/robot-wst/isaacsim/python.sh \
  isaaclab/tools/validate_phase3_geometry_rtx.py
```

Reproduce the visual deliverable with:

```bash
python3 tools/generate_administration_textures.py
TERM=xterm /home/robot-wst/isaacsim/python.sh scripts/build_administration.py \
  --headless --payload loaded \
  --plan /home/robot-wst/Downloads/DownloadBuildingRequestApprovedPlan.pdf \
  --presentation-assumptions
python3 tools/validate_administration_replay.py \
  --trajectory-report results/phase2_administration_policy_only_video_report.json \
  --output results/phase2_administration_visual_replay_validation.json
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p scripts/render_administration_route.py \
  --headless --width 1280 --height 720 --fps 20 --seconds-per-shot 2 \
  --renderer PathTracing --path-tracing-spp 8 \
  --trajectory-report results/phase2_administration_policy_only_video_report.json \
  --frame-directory media/phase2_visual_upgrade_frames \
  --render-report results/phase2_administration_visual_upgrade_render_report.json
python3 tools/encode_route_video.py --fps 20 --crf 18 --preset slow \
  --frames-dir media/phase2_visual_upgrade_frames \
  --validation results/phase2_administration_visual_replay_validation.json \
  --output media/videos/AI-SHA_Phase2_Administration_Visual_Upgrade.mp4 \
  --report results/phase2_administration_visual_upgrade_render_report.json
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/play_block_a_route.py \
  --task Isaac-AISHA-Administration-Live-Phase3-DynamicSafety-Presentation-Direct-v0 \
  --checkpoint isaaclab/checkpoints/aisha_phase3n_dynamic_safety_model_50.pt \
  --output-report results/phase3n_administration_final_omniverse_report.json \
  --video-folder media/videos/phase3n_administration_final_omniverse \
  --route-control policy-only --camera-mode cinematic --seed 10201 --headless
python3 isaaclab/tools/make_administration_live_policy_presentation_video.py \
  --input media/videos/phase3n_administration_final_omniverse/\
aisha-block-a-learned-route-step-0.mp4 \
  --run-report results/phase3n_administration_final_omniverse_report.json \
  --output media/videos/AI-SHA_Phase3N_Administration_Final_Omniverse_3x.mp4 \
  --report results/phase3n_administration_final_omniverse_3x_report.json --speed 3
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/play_phase4a_dynamic_showcase.py \
  --checkpoint isaaclab/checkpoints/aisha_phase3n_dynamic_safety_model_50.pt \
  --output-report results/phase4a_administration_dynamic_showcase_report.json \
  --video-folder media/videos/phase4a_administration_dynamic_showcase_final \
  --max-steps 600 --seed 10401 --headless
python3 isaaclab/tools/make_phase4a_dynamic_showcase_video.py \
  --input media/videos/phase4a_administration_dynamic_showcase_final/\
aisha-phase4a-dynamic-showcase-step-0.mp4 \
  --run-report results/phase4a_administration_dynamic_showcase_report.json \
  --output media/videos/AI-SHA_Phase4A_Administration_Dynamic_Safety_Showcase.mp4 \
  --report results/phase4a_dynamic_safety_presentation_video_report.json
python3 isaaclab/tools/validate_phase4a_dynamic_showcase.py \
  --run-report results/phase4a_administration_dynamic_showcase_report.json \
  --video-report results/phase4a_dynamic_safety_presentation_video_report.json \
  --output results/phase4a_dynamic_safety_showcase_acceptance.json
python3 isaaclab/tools/make_final_omniverse_presentation_reel.py \
  --mission-video media/videos/AI-SHA_Phase3N_Administration_Final_Omniverse_3x.mp4 \
  --mission-report results/phase3n_administration_final_omniverse_3x_report.json \
  --safety-video media/videos/AI-SHA_Phase4A_Administration_Dynamic_Safety_Showcase.mp4 \
  --safety-video-report results/phase4a_dynamic_safety_presentation_video_report.json \
  --safety-run-report results/phase4a_administration_dynamic_showcase_report.json \
  --output media/videos/AI-SHA_Final_Omniverse_Administration_Presentation.mp4 \
  --report results/final_omniverse_administration_presentation_report.json
python3 isaaclab/tools/validate_final_omniverse_presentation_reel.py \
  --mission-report results/phase3n_administration_final_omniverse_3x_report.json \
  --safety-video-report results/phase4a_dynamic_safety_presentation_video_report.json \
  --safety-run-report results/phase4a_administration_dynamic_showcase_report.json \
  --reel-report results/final_omniverse_administration_presentation_report.json \
  --output results/final_omniverse_administration_presentation_acceptance.json
```

## Navigation and doorway fit

The measured-site capture and simulation-only ROS 2/Nav2 procedure is documented
in [MEASURED_SITE_NAV2_WORKFLOW.md](MEASURED_SITE_NAV2_WORKFLOW.md). It keeps the
Rev D differential-drive parameters isolated from the older production-repository
mecanum profile and records the remaining runtime gates without overclaiming them.

Use a footprint that contains the outboard wheels and forward head:

```yaml
footprint: "[[-0.455,-0.384],[0.725,-0.384],[0.725,0.384],[-0.455,0.384]]"
```

The head makes the robot 1.180 m long even though the deck is only 0.910 m. The
previous length claim was wrong.

Clear width alone does not release a route. The robot pivots about the
drive-axle midpoint at x=0 with the head 0.725 m ahead, so the furthest
footprint corner sweeps 0.820 m and a turn-in-place needs a **1.640 m clear
circle** — against a 0.768 m transit width. A 0.920 m doorway cannot be pivoted
in. **Plan every rotation in the hallway (2.80 m) or atrium; traverse doorways
on a straight, pre-aligned approach.** Rear corners sweep 0.595 m, so a pivot
also needs that much clearance behind the axle.

For presentation-only simulation, the measured-doorway profile permits the
reported 0.85 m minimum with a padded 0.828 m transit envelope, a straight
centreline approach, no doorway rotation, and a 0.10 m/s speed limit. Its
nominal padded margin is only 11 mm per side, so this acceptance must not be
used as physical clearance approval.

A nominal 900 mm door leaf commonly has less
clear width after stops, hinges and hardware; only an on-site clear measurement
can release the route. For physical operation, if clear width is below 0.920 m,
change the route or narrow the mechanical design instead of tuning Nav2 to
squeeze through.

The measured-administration baseline starts at 0.30 m/s. The separately trained
simulation hallway tier now passes its formal 0.80 m/s gate and the complete
measured-scene Nav2/RTX presentation gates on declared straight segments, while
tight doors remain limited to 0.10 m/s. Physical operation at
0.80 m/s remains prohibited until measured stopping-distance, protective-field
and supervised commissioning tests pass.

## Perception is not the safety system

`lidar_link` at 1.170 m is for localisation/mapping. It cannot be relied on for
chairs, bins or floor hazards. `front_lidar_link` at 0.250 m feeds the Nav2
obstacle layer, but front-only perception does not protect reversing, side-swipe
or pivot motion.

No sensor model, field of view, diagnostic coverage or safety rating has been
provided. Until a risk assessment and all-direction protective-stop design are
complete, operate only on a closed/access-controlled route with a trained
spotter. Reverse must remain disabled in occupied areas.

The real PoC also needs hardwired emergency stops independent of ROS/Jetson/Pi,
physical bumpers, a battery fuse and service disconnect, contactor, charging
interlock, wheel/pinch guards, command watchdog and measured stop performance.
The driver manual explicitly requires an external emergency-stop circuit;
cutting torque may still allow the robot to coast.
