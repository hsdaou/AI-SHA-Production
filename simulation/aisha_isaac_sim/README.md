# AI-SHA - Isaac Sim package (proof of concept)

**Rev M - 2026-08-23 - final Omniverse administration presentation reel accepted**

This package describes the simplified indoor proof-of-concept: two driven hub
wheels on the centre lateral axis, four physical swivel castors, a retained
Prestar NF-301 deck, a guided compliant drive carrier, and a fixed mast
reinforcement spine. It supersedes every earlier 4-wheel skid-steer model.

It is a coherent simulation/design baseline, not a fabrication release or a
safety certification. The unresolved hold points are listed below and in
`config/aisha_drive.yaml`.

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
furniture and lighting follow the supplied walkthrough video. Both office clear
widths are now disclosed 1.40 m presentation assumptions. Both thresholds are
assumed flush because the rigid castor proxy cannot support threshold-contact
conclusions. Neither value is a site measurement; scene metadata and reports
keep physical release false.

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

A nominal 900 mm door leaf commonly has less
clear width after stops, hinges and hardware; only an on-site clear measurement
can release the route. If clear width is below 0.920 m, change the route or
narrow the mechanical design instead of tuning Nav2 to squeeze through.

Start at 0.30 m/s. The controlled demo target is 0.50 m/s. Treat 0.80 m/s as a
design ceiling only after measured stopping-distance and protective-field tests.

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
