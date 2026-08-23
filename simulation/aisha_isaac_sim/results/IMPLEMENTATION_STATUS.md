# AI-SHA Isaac Sim implementation status

**Updated 2026-08-23. The unchanged Rev D robot now passes the measured-presentation 12-leg live Nav2 mission through both offices with the accepted Phase 3N learned 360-degree brake layer. The aggregate gate passes 27/27. Phase 4A and the accepted Omniverse evidence reel remain the dynamic-obstacle presentation evidence. This is not a physical safety or deployment release.**

## Completed

- Workstation inventoried: Ubuntu 24.04.4, RTX 5080 16 GB, driver 580.159.03,
  Isaac Sim 5.1.0, ROS 2 Jazzy and X11. Isaac Lab is linked at
  `/home/robot-wst/IsaacLab`, pinned to commit `80094be3245`.
- Canonical URDF generator and validator pass at 59.25/69.25 kg, 20 links,
  19 joints, and two driven joints.
- Empty and 10 kg-loaded URDFs imported with authored inertia, floating base,
  fixed-joint merge, rated 6 N.m force limit, zero stiffness, and damping 120.
- Required crown LiDAR, low LiDAR, camera, optical, IMU, and payload frames are
  present in both composed assets.
- Mass-only fixed-frame references emitted incorrectly by URDF importer 2.4.30
  are repaired during post-processing; the final composed assets reopen cleanly.
- Drive-wheel and castor physics materials are bound inside each importer's
  reference scope, so the bindings survive USD composition.
- Loaded flat-floor full suite passes:
  - drop/settle remains finite and upright;
  - 0.30 m/s run: 5.325 m, 0.037 deg yaw error, 0.15% steady-speed error;
  - 0.50 m/s run: 5.485 m, 0.016 deg yaw error, 0.09% steady-speed error;
  - both pivot directions pass with about 73.3 deg rotation and under 5 mm drift;
  - stale-command watchdog stops and remains latched until explicit reset.
- Empty and loaded smoke suites pass.
- Parameterized 5/10/20 mm threshold geometry exists, but is correctly marked
  as unsuitable for contact conclusions until the articulated asset is built.
- Page 2 of `DownloadBuildingRequestApprovedPlan.pdf` was reviewed and checksum
  locked. The route-scoped `administration.usd` now uses the plan's Block A
  topology: central 12.75 m atrium, 2.80 m east hallway, Vice-Principal in the
  east room cluster and Principal in the angled south-east suite.
- The walkthrough supplies appearance only: polished terrazzo, dark timber
  slats, warm-white walls, frosted glazing, light-oak offices, suspended LED
  panels, desks, chairs, cabinets and planters.
- A presentation shell makes the engineering URDF read as finished AI-SHA while
  leaving the Rev D collision, mass, drive and sensor model unchanged.
- The earlier primitive four-shot render remains reproducible but is retired as
  final presentation material because it does not match the walkthrough closely
  enough. Walkthrough-grounded photoreal concept stills are kept separate from
  simulation evidence.
- The original wide-door presentation scene used disclosed 1.40 m openings.
  The current measured-presentation scene supersedes them with the reported
  0.85 m minimum VP opening and a disclosed 0.90 m Principal assumption. Both
  thresholds remain assumed flush and neither route is physically released.
- AI-SHA-Production commit `8893535` was reconciled for sensor/system contracts:
  LD19 `/scan`, RealSense D435 aligned depth, and BNO055 `/imu/data` at 50 Hz.
- A real Isaac Lab DirectRLEnv task, `Isaac-AISHA-OfficeNav-Direct-v0`, now
  commands the two imported wheel joints at 30 Hz while PhysX runs at 120 Hz.
  Root-transform movement is prohibited except for normal episode resets.
- The first RSL-RL PPO baseline trained for 200 iterations across 64 parallel
  environments: 307,200 simulated policy transitions using seed 42.
- A deterministic held-out run used seed 1001 and 512 episodes. It recorded
  420 successes (82.03%), 88 collisions (17.19%) and 4 timeouts (0.78%). These
  are state-observation foundation results, not sensor or sim-to-real results.
- A separate 18 s, 1280 x 720, 30 fps Isaac Sim capture shows the learned policy
  physically driving through the training doorway. It is training evidence and
  is intentionally not represented as the photoreal Block A final presentation.
- `Isaac-AISHA-BlockA-SensorNav-Direct-v0` commands the production-derived
  loaded robot using 36 MultiMeshRayCaster ranges and ten goal/vehicle state
  observations over all 12 directed Block A route segments.
- The selected seed-144 PPO run trained for 600 iterations, 32 parallel
  environments and 614,400 policy transitions. The final checkpoint is
  `model_599.pt` from run `2026-08-20_22-33-50_ld19_flush_threshold_v9`.
- The distinct seed-5084 deterministic evaluation enforced 48 completed
  episodes per segment. It passed with 576/576 successes, zero collisions and
  zero timeouts; every individual segment, including the four office entry/exit
  legs, passed 48/48.
- The seed-6084 continuous tour reached all 12 waypoints in 5,209 policy steps
  (173.6 simulated seconds) without collision or reset. The learned policy drove
  4,239 aligned-leg steps; a disclosed physical wheel-command supervisor handled
  910 in-place turn steps and 60 office dwell steps.
- The full 173.6 s, 1280 x 720 Isaac Lab capture and a verified 57.6 s 3x
  presentation cut were encoded. The edit changes crop, labels and temporal
  sampling only; it does not change the learned trajectory.
- `Isaac-AISHA-Administration-Live-Direct-v0` now replaces the simplified
  training course with the complete walkthrough/plan-derived architecture,
  furniture, doors and lighting while preserving the trained task's 46-value
  observation and two-action contracts.
- The live environment excludes the scene's earlier pose-replay robot and
  nested physics scene, then spawns the loaded AI-SHA articulation. Its
  collisionless presentation shell is a fixed link on the real articulated
  base, so it follows Fabric/PhysX instead of a separately animated transform.
- The unchanged `model_599.pt` checkpoint (SHA-256
  `3da826c515d0e58a3c0731dd3b72022208eadda3cc2bd2200a78d92f079bfacf`)
  completed all 12 live administration segments in 5,388 ticks and 179.6
  simulated seconds without collision or reset. The learned policy supplied
  4,378 ticks; the disclosed physical wheel-command turn supervisor supplied
  950 and office dwell supplied 60. No root-transform animation was used.
- The adaptive follow camera uses live rear ray clearance to vary its distance
  from 0.9 to 2.0 m near walls. The final full 1x capture has 5,387 frames at
  1280 x 720 and 30 fps; the motion-preserving labelled 3x presentation cut is
  59.7 s.
- `administration_live_policy_validation.json` passes all 23 declared checks,
  covering the checkpoint and asset hashes, stage composition, complete route,
  control accounting, finite trace, door assumptions, collision result, camera
  visibility, raw video and presentation edit.
- Phase 2 adds two task configurations without changing the Phase 1 evidence:
  `Isaac-AISHA-BlockA-Phase2-Turn-SensorNav-Direct-v0` for efficient
  arbitrary-heading fine-tuning and
  `Isaac-AISHA-BlockA-Phase2-EndToEnd-SensorNav-Direct-v0` for the complete
  policy-only route gate.
- The Phase 2 search resumed the exact Phase 1 checkpoint for 1,200
  incoming-transition iterations and a 300-iteration moving-handoff refinement
  across 32 environments (1,536,000 executed search transitions). It starts
  from physical incoming headings with ±15 degree jitter and 0.30-0.50 m/s
  motion, adds 10 mm range noise and 0.25% ray dropout, and rewards
  heading-error reduction.
- `play_block_a_route.py` now has an explicit `--route-control policy-only`
  mode. In that mode it applies no turn or dwell action overrides and records
  this contract in the run report. The Phase 2 gate permits zero supervisor
  steps.
- Checkpoint `model_1850.pt` from
  `2026-08-22_09-12-57_phase2_moving_transition_seed256` was selected with
  SHA-256
  `3ab596c61259784657b36fe4ee937da8495ce9621a43bd04e2c8a0bf6e0b1880`.
  Later `model_1900.pt` and `model_2097.pt` checkpoints were rejected because
  moving Principal-departure robustness fell to 20/48 and 12/48 respectively.
- The balanced Phase 2 transition gate passed 570/576 episodes (98.96%), with
  six collisions, no timeouts and an 87.5% minimum segment/office-exit rate.
  The 48-episode chained-route gate passed 46/48 (95.83%), with two collisions
  and no timeouts.
- A continuous policy-only training-course run completed 12/12 segments in
  4,752 policy steps / 158.4 simulated seconds. The live administration run
  completed 12/12 in 4,885 policy steps / 162.83 simulated seconds with zero
  collisions, zero turn-supervisor steps, zero dwell steps and no
  root-transform animation.
- The presentation camera now requests 3.8 m behind/2.4 m high and follows the
  stable direction of each route leg rather than the robot's instantaneous
  pivot. A three-ray visibility fan prevents doorway occlusion. The 4,884-frame
  raw capture has no post-start uniform-occlusion interval; its motion-preserving
  3x edit is 54.13 seconds.
- `phase2_end_to_end_validation.json` passes all 18 declared checks, including
  both held-out gates, policy-only control accounting, common checkpoint hash,
  complete raw video, camera visibility and unchanged presentation motion.
- Training and capture executed after returning to kernel
  `6.17.0-35-generic`; NVIDIA driver 580.159.03 and the RTX 5080 were active.
- The first administration visual upgrade is complete. Deterministic procedural
  PBR textures now cover the polished terrazzo, walnut, oak and mottled-grey
  finishes; the scene also adds denser ceiling/lighting detail, vents, office
  furnishings, glazing and a refined AI-SHA presentation shell. These are
  visual-only upgrades and do not change the route or collision geometry.
- Six human-height presentation cameras replace the earlier overhead-heavy
  framing. The robot remains contextual in the atrium/hall views and is clearly
  shown inside both the Vice-Principal and Principal offices.
- The final visual-upgrade film is 240 frames, 1280 x 720, 20 fps, 12.0 s,
  PathTracing at 8 samples per pixel. It replays only the 1,628 recorded
  `learned_sensor_policy` pose samples from the successful 4,885-step Phase 2
  run, with no scripted route interpolation.
- `phase2_administration_visual_replay_validation.json` passes 23/23 checks.
  `AI-SHA_Phase2_Administration_Visual_Upgrade.mp4` has SHA-256
  `4b02865f831d0d1e0db7bf159d3d1ac09f34e24c2d693eeaf052280817350849`.
  It is labelled as a verified Omniverse visual replay; the raw Phase 2 live
  capture remains the proof of simultaneous policy inference and PhysX motion.
- The live visual-scene integration uses a transparent hierarchical learned
  controller: the PPO-adapted `model_2150` base controls segments 0-5 and 7-11,
  while an imitation-trained specialist controls only segment 6. The route
  planner selects the declared learned skill; neither skill receives scripted
  turn, dwell or root-transform actions.
- The Principal-turn specialist was trained from 16/16 successful live-physics
  pivot-then-drive demonstrations and separately passed 8/8 deterministic
  trials with zero collisions and zero timeouts.
- The final seed-7084 cinematic run completed 12/12 in 5,160 learned-policy
  steps / 172.0 simulated seconds: 4,730 base-policy and 430 specialist steps,
  with zero supervisor turns, zero dwell steps, zero collisions and no root
  animation.
- Six static, human-height cameras cover every route segment once. Camera 4 was
  moved into the open hallway after validation caught a uniform gray wall
  occlusion; the corrected raw capture has no non-initial uniform occlusion.
- `phase2_administration_live_cinematic_validation.json` passes 42/42 checks.
  The unchanged-motion main film is 57.2 s (3x) and the teaser is 14.3 s (12x).
  Their overlays accurately disclose `PPO base + imitation specialist`.
- Phase 3 adds up to two moving 1.70 m x 0.48 m person capsules per environment,
  physical collision, tracked multi-mesh LiDAR hits and a forward-clearance
  reward. Crossings are limited to seven open route legs; the exact tight
  segment-6 U-turn is excluded. Person proxies pause inside a disclosed 1.10 m
  reciprocal-yield radius. No privileged person state is added to the 46-value
  policy input.
- Domain randomization now samples lidar noise/dropout/bias/scale, 0-2 policy
  steps of latency, motor strength, wheel radius/track, drive damping, base
  mass/inertia and static/dynamic friction within the declared bounded ranges.
- `phase3_dynamic_yielding_smoke_report.json` passes 11/11 checks across four
  replicated environments and 180 policy steps. The optimizer smoke resumed
  the hash-locked Phase 2 model at iteration 2150 and wrote new PPO checkpoints.
- The first Phase 3 PPO continuation ran 600 iterations / 614,400 transitions
  at full perturbation strength and was rejected after catastrophic forgetting.
  The replacement `2026-08-22_14-02-08_phase3b_staged_dynamic_dr_seed8101`
  run used 100 static-retention iterations, a 350-iteration ramp and 150
  full-strength iterations for another 614,400 transitions.
- The evaluator was corrected to force curriculum strength 1.0 for fresh Phase
  3 evaluation. The earlier `model_2525.pt` 43/48 screen had started at strength
  zero, so it is retained only as a static diagnostic and is not full-strength
  dynamic-obstacle evidence.
- Targeted segment-6 continuation trained for 307,200 transitions, but all
  three screened deterministic actors failed 0/32 without collisions: their
  mean actions had collapsed to reverse-and-turn while stochastic rollouts
  occasionally succeeded. No specialist checkpoint was accepted either.
- The correct proxy-course Phase 2 checkpoint `model_1850.pt` retained 32/32
  segment-6 successes with full physics/sensor randomization after the
  adversarial tight-turn crossing was removed. This isolates the prior failure
  to the interaction design rather than the office U-turn geometry.
- The clean reciprocal-yield run
  `2026-08-22_15-00-21_phase3h_reciprocal_yield_seed8701` completed 600 PPO
  iterations / 614,400 transitions. Leading candidate `model_2225.pt` achieved
  24/48 full-strength deterministic successes with 4 dynamic collisions,
  8 static collisions, 12 timeouts, and 4/4 segment-6 successes. It remains
  unaccepted.
- The temporal gate distilled the hash-locked route teacher into a one-layer
  46-unit GRU for 200 iterations / 204,800 transitions; behavior loss fell from
  3.1066 to 0.00458. The recurrent PPO continuation completed 800 iterations /
  1,638,400 transitions and preserved the unchanged 46-value observation and
  two-wheel action contracts.
- Recurrent checkpoints 475, 600 and 700 were screened at full strength.
  `model_700.pt` led with 26/48 successes, 6 dynamic collisions, 9 static
  collisions and 7 timeouts. On the same seed it gained three successes and
  removed five timeouts versus feed-forward `model_2225.pt`, but added three
  dynamic collisions. It is not accepted. Its recommended bounded recurrent
  residual/safety follow-up is the Phase 3K gate described below.
- The bounded residual environment freezes hash-locked feed-forward Phase 3
  `model_2225.pt`. Its 64-unit GRU can slow or stop and attenuate steering by
  at most 25%, but it cannot increase speed, reverse, increase steering
  magnitude or flip steering sign. The live Isaac runtime gate passed 15/15
  checks, including exact zero-residual pass-through.
- Phase 3K trained that residual for 600 iterations / 1,228,800 transitions at
  full perturbation strength. `model_175.pt` led the independent seed-9502
  selection screen. In the decisive same-seed 48-episode comparison it reduced
  dynamic collisions from 5 to 3, but successes fell from 25 to 21 and static
  collisions rose from 2 to 8; total collisions rose from 7 to 11. It is not
  accepted, and no presentation artifact or accepted policy was replaced.
- The isolated diagnosis is that a slow/stop-only layer cannot repair static
  clearance errors in the frozen route actor. The next gate must preserve the
  hard speed boundary while adding only clearance-verified bounded lateral
  correction, or combine a map-aware local planner with an independent
  protective-stop layer.
- Phase 3L implements that gate. The recurrent layer may brake and request at
  most ±0.35 rad/s around the hash-locked route actor. A one-second,
  five-sample projector checks the robot's rectangular footprint plus 80 mm
  planning margin against uncorrupted LiDAR hit geometry and rejects unsafe or
  route-divergent steering. A separate hysteretic front-sector stop runs after
  latency and can remove forward motion without removing steering.
- `phase3l_clearance_planner_smoke_report.json` passes 21/21 Isaac runtime and
  command-boundary checks. Phase 3L then trained for 600 iterations / 1,228,800
  transitions from an exact zero-output recurrent checkpoint.
- `model_200.pt` won the seed-9702 and seed-9703 selection screens. On the
  decisive seed-9701 48-episode comparison it reached 29 successes, 1 dynamic
  collision, 4 static collisions and 14 timeouts. The original frozen route
  recorded 19/3/10/16; the hard-stop-only controller recorded 19/3/3/23.
  Model 200 is promoted as the leading architecture candidate because it adds
  10 successes while lowering total collisions relative to both controls.
- Phase 3L is not fully accepted and does not replace the presentation policy:
  60.4% success, 2.1% dynamic collision and 8.3% static collision do not pass
  the declared thresholds, and the screen used 4 rather than 64 episodes per
  segment. The next target is office-departure pivot recovery on segments 4
  and 9 plus segment-6 static-clearance recovery.
- Phase 3M executed that targeted gate. Runtime diagnosis found that the Phase
  3 material randomizer had overwritten all four fixed-sphere caster proxies
  with drive-wheel friction. The targeted task restores their disclosed low
  0.15-0.25 static / 0.10-0.20 dynamic friction bands and gates the declared
  6 Nm continuous / 18 Nm peak motor effort to at most three seconds of a
  stationary, large-heading recovery pivot.
- The Phase 3M runtime combines the trained recurrent residual with a narrowly
  scoped hierarchical supervisor: only office departures 4 and 9 receive a
  goal-signed 0.55 rad/s pivot after a one-second rectangular-footprint
  clearance check. Tight return segments 6, 10 and 11 may be capped at
  0.10 m/s, or 0.08 m/s on the dynamic segment-10 crossing. Normal route
  transit remains under learned policy inference. The Isaac runtime contract
  passes 38/38 checks.
- Corrected-physics PPO ran without deterministic recovery actions for 150
  iterations / 307,200 transitions. The runtime task is registered separately
  from that reproducible training curriculum. Model 125 is
  packaged as `aisha_phase3m_hybrid_recovery_model_125.pt` with SHA-256
  `bc8727e3ea42c8b29ca74fa5a535fd37b1600633ffd8bf606b02220a557c1a0d`.
  On seed 9701 it reached 46/48 successes, 2 dynamic collisions, no static
  collisions and no timeouts. Segments 4, 6 and 9 each passed 4/4, versus
  Phase 3L's 0/4, 2/4 and 0/4 on the same seed.
- The independent seed-9702 screen reached 43/48 with 4 dynamic and 1 static
  collision. The candidate therefore fixes the targeted route transitions but
  is not full Phase 3 acceptance, not a replacement presentation policy and
  not a physical release. The next gate freezes this pivot/clearance stack and
  trains a 360-degree dynamic-obstacle safety layer before the declared 64
  episodes per segment, Phase 2 regression and 12 live-scene scenarios.
- Phase 3N executes that gate without modifying either frozen controller. The
  Phase 3M recurrent checkpoint and its underlying route checkpoint are both
  SHA-256 checked and have no trainable parameters. A separate 64-unit GRU has
  one brake-only action: it can remove forward motion when the 36-ray ring is
  closing on a declared pedestrian leg, but cannot steer, accelerate, reverse
  or move the robot root. The runtime contract passes 29/29 checks.
- PPO ran for 200 iterations across 32 environments, totaling 409,600
  transitions. Selected model 50 is packaged as
  `aisha_phase3n_dynamic_safety_model_50.pt`, SHA-256
  `11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b`.
  Across matched seeds 9701/9702 it records 96/96 successes and no contacts;
  the exact frozen-stack control records 95/96 and one dynamic contact.
- The declared randomized gate passes with 744/768 successes (96.875%), 8
  dynamic contacts (1.042%), 12 static contacts (1.563%), four timeouts and a
  92.188% minimum success rate on pedestrian-enabled segments. The separate
  full-randomization/no-pedestrian regression passes 48/48 with zero contacts,
  and the walkthrough-matched live gate passes 12/12 dynamic scenarios with
  zero contacts or timeouts. Full Phase 3 simulation acceptance is therefore
  recorded; site survey, Nav2, sim-to-real and physical safety commissioning
  remain mandatory before physical release.
- The final seed-10201 Omniverse film run uses the packaged Phase 3N checkpoint
  and completes all 12 waypoints in 4,941 learned-policy steps / 164.7 simulated
  seconds. It records zero turn-supervisor steps, zero dwell actions, zero
  collisions, no root animation and continuous recurrent state across both
  office visits and the return. The static presentation scenario, 0.22 m office
  stops, 0.20 m Principal-departure stop and omitted redundant segment-10
  predictive-creep guard are disclosed film-only adaptations; the separate
  accepted dynamic gates remain the safety evidence.
- The raw 1280 x 720 capture has SHA-256
  `fad47145d00e32c492b739cffb190a15e703a625345cc1ee76071e1760b54b8b`.
  Its 54.77 s 3x presentation cut retains motion, adds synchronized controller
  telemetry and labels the frozen Phase 3M plus learned 360-degree brake stack;
  SHA-256 is
  `44cd1e975b6b13a98f3013cdb1f3dc90db52e9d240f763a6f03d843f877e2f13`.
- Phase 4A preserves the accepted checkpoint and all formal Phase 3N gates,
  then adds a deterministic presentation-only crossing on segment 7, the
  Principal-office approach. The stylized kinematic pedestrian proxy has a
  torso collider and remains evaluation truth; no pedestrian position or
  velocity enters the policy observation.
- Seed 10401 passes in 395 policy steps / 13.17 simulated seconds. It records
  16 learned-authority samples during the crossing, a 14.32% peak learned brake
  fraction, a separately measured 2.23 s frozen-stack protective stop, recovery
  to 0.463 m/s, and zero dynamic or static contacts. There is no physics
  supervisor or root-transform animation.
- The 13.13 s, 1280 x 720 presentation video labels learned brake authority and
  protective-stop state independently. Its SHA-256 is
  `556468502d6f5a86a6e13cf5809bd9fd6dba5d3ae76db6d82d3eea64afa77886`;
  `phase4a_dynamic_safety_showcase_acceptance.json` passes 22/22 evidence-chain
  checks. This is a controlled simulation demonstration, not proof of physical
  human safety.
- The final 76.9 s, 1280 x 720, 30 fps reel preserves all 1,643 accepted clean
  mission frames and all 394 accepted dynamic-safety frames once and in order.
  Three labeled cards add 270 frames without changing source motion. The reel's
  SHA-256 is
  `c3218bb4e969c17b9d61be299be8b80e9558775514f0bd097be18c320d18e132`;
  `final_omniverse_administration_presentation_acceptance.json` passes 27/27.
  The ending explicitly retains measured-site, Nav2/sim-to-real and hardware
  safety as future gates.
- Geometry refinement `GEOMETRY-RTX-PHASE3-A` locks the source PDF hash and uses
  printed page-2 Block A dimensions for the 12.75 m atrium, 2.80 m hall,
  7.80 x 6.30 m conference room and 4.73 m Principal frontage. Door widths,
  thresholds, heights and furniture remain explicit presentation assumptions.
- RTX material v2 adds albedo, perceptual roughness and tangent-space normal
  maps for terrazzo, walnut, oak and mottled-grey finishes. Five 1280 x 720
  office/route stills were regenerated with PathTracing at 64 spp;
  `phase3_geometry_rtx_refinement_validation.json` passes 17/17 checks.
- The measured-presentation live autonomy stack now connects Nav2 global/DWB
  planning to a mapped two-stage doorway/central-drop guard, then to the
  accepted Phase 3N learned 360-degree brake and articulated wheel physics.
  The new task, wrappers and evidence keep this path separate from the learned
  route policy because Nav2/DWB and that policy are alternative local motion
  authorities.
- The first accepted measured static mission completed all 12/12 legs, both
  office entries/departures, two in-office 180-degree pivots and the return
  home in 18,323 bridge steps without an episode reset. Door entry counters are
  2/2 for both VP and Principal; maximum measured doorway speed was 0.06258 m/s
  against the 0.10 m/s limit and maximum tangent offset was 0.02151 m.
- The inaccessible 0.20 m central atrium drop remains in the occupancy map and
  as a physics collider. It is removed only from the learned crown-ray target
  set to avoid treating an artificial navigation barrier as a person/object;
  a predicted full-footprint mapped guard retains authority. Minimum predicted
  clearance was 0.27959 m. This routing and the 0.015 m simulation termination
  envelope earn no physical safety credit.
- The accepted Phase 3N checkpoint was loaded by hash. It had authority for 689
  steps and applied braking on 192. The aggregate evidence report
  `administration_nav2_measured_integration_gate.json` passes 27/27.
- Phase 6 now couples the accepted 0.80 m/s brake policy on straight segments
  1 and 5 with the accepted Phase 3N fallback elsewhere. The measured Nav2 run
  completed 12/12 legs, both 180-degree office pivots and both door traversals
  without reset. Minimum observed high-speed peak was 0.74339 m/s; maximum
  doorway speed was 0.05953 m/s and tangent offset was 0.01536 m.
  `administration_nav2_phase6_high_speed_integration_gate.json` passes 28/28.
- The accepted live run contains a 4,760-record pose trace. Its six-shot
  Omniverse replay rendered 240 frames at 1280 x 720 with RTX PathTracing at
  8 spp. `AI-SHA_Phase6_Nav2_LearnedSafety_RTX_Presentation.mp4` is 12.0 s and
  has SHA-256
  `34da79912934b1454a6f6a6b3592c172eadb7d089d6fdec3dda785b6d17c5d07`;
  `administration_nav2_phase6_rtx_presentation_acceptance.json` passes 14/14.
- The presentation integration uses Isaac ground-truth odometry with an
  identity map-to-odom transform. It evaluates Nav2 motion and learned safety,
  not AMCL robustness, and earns no physical localisation or release credit.

## Deterministic contact tuning disclosure

The rigid six-contact proxy is statically indeterminate. Exactly coplanar
contacts allowed PhysX to unload one driven wheel, so the asset applies a
symmetric 1 mm **simulation rest offset** to both drive-wheel colliders. This is
a solver seating bias, not a physical tyre-radius, preload, or contact-load
claim. See `config/physics_materials.yaml`.

## Presentation assumptions and authoritative blockers

- The approved plan is now present and controls printed dimensions, room
  relationship and route topology. PDF trace offsets and final in-room goal
  offsets remain presentation placements rather than survey-grade coordinates.
- The user-reported 0.85 m minimum and 2.12 m door height are applied to the VP
  opening. The Principal opening uses a disclosed 0.90 m presentation
  assumption and the same height. Flush thresholds and exact hinge/leaf
  geometry remain assumptions; neither route is physically released.
- Wall/ceiling height, wall thickness, office furniture offsets and decorative
  detail remain walkthrough-derived or presentation assumptions.
- High-fidelity contact asset: measured carrier spring curve and caster
  trail/inertia/height are absent.
- The LD19, D435 and BNO055 models/interfaces are known, but the low front LiDAR
  model and exact camera intrinsics remain hold points. The trained ray ranges
  are scalable geometric sensing, not a validated RTX LD19 noise model.
- The 12-segment Principal/Vice-Principal sensor policy and declared Phase 3
  dynamic-safety gate pass. The measured-presentation static Nav2 integration
  now also passes. Blocked-route dynamic replanning, broader office-directory
  coverage, native scan registration and sim-to-real commissioning remain
  subsequent gates.
- The administration USD has been rebuilt around the walkthrough's primary
  visual anchors: brighter polished terrazzo and aggregate, white atrium
  columns, dark timber/slatted walls, office glazing, a round timber meeting
  table with black cantilever chairs, suspended ceiling grid/vents, and the
  Principal's timber-backed executive office. These are walkthrough-derived
  visual placements, not measured as-built geometry.
- The southeast walkthrough-derived atrium column was relocated after visual QA
  found it intersecting the recorded return trajectory. All four columns now
  pass a conservative swept-footprint gate against every interpolated trace
  segment. The closest surface clearance is 1.203 m against a required 0.95 m
  trace-centre distance; this is presentation geometry QA, not an as-built
  structural clearance claim.
- The earlier Phase 1 administration replay consumed the successful seed-6084
  pose trace directly. Five shots covered segments 0-11 exactly once; every
  displayed robot pose was selected from a recorded trace sample, with no
  scripted route interpolation. It was explicitly labelled as visual replay,
  not live policy execution.
- `administration_learned_replay_validation.json` passes all evidence-chain
  checks: scene/build/config identity, checkpoint, 12 waypoints, 5,209 steps,
  1,736 finite monotonic trace records, disclosed control modes, complete shot
  coverage, and the four-column conservative swept-clearance gate.
- The workstation was returned to kernel `6.17.0-35-generic`, restoring the
  NVIDIA 580.159.03 driver and RTX 5080. The final PathTracing capture completed
  with 360/360 frames at 1920 x 1080, 24 fps and 4 samples per pixel.
- The final 15.0 s video was encoded with FFmpeg 7/libx264 High Profile at CRF
  18. Its frame count, duration, codec settings, scene/trajectory hashes and
  final SHA-256 are recorded in
  `administration_learned_replay_render_report.json`.
- The newer live administration evidence supersedes that pose replay for the
  main technical demonstration. It is genuine live policy inference with
  wheel/contact physics inside the administration USD, but it remains a
  presentation-quality digital environment rather than a photogrammetric or
  measured as-built digital twin.
- The selected Phase 2 policy is end-to-end over the declared 12-segment
  simulation mission: it supplies every wheel action, including both office
  pivots and departures. This does not imply end-to-end physical autonomy.
  The next learning gate is broader randomized geometry, friction, sensing and
  dynamic obstacles.
- Phase 2 code, checkpoint, gates, live execution and presentation evidence are
  complete. The scene is still a presentation-quality plan/walkthrough model,
  not a photogrammetric or measured as-built digital twin.
- Physical Nav2 release still depends on complete threshold/door hardware data,
  an as-built navigation survey and closed-route commissioning.

## Progress snapshot

These are evidence-based project-management estimates, not safety ratings:

| Workstream | Progress |
|---|---:|
| Presentation-ready Omniverse simulation | 90% |
| Learned navigation and doorway handling | 87% |
| Measured digital twin | 66% |
| Nav2/localisation integration | 72% |
| Dynamic-obstacle simulation safety | 79% |
| Physical deployment readiness | 30% |
| Overall end objective | 74% |

The major increase is Nav2/localisation integration: the full measured static
mission, route-scoped 0.80 m/s simulation tier and final RTX replay are now
accepted. The remaining work is dominated by blocked-route dynamic replanning,
real localisation, native scan/visual refinement, operator-facing capture and
physical sim-to-real commissioning.

## Evidence

- `workstation_inventory.json`
- `import_report.json`
- `scene_build_report.json`
- `validation_smoke_empty.json`
- `validation_smoke_loaded.json`
- `validation_full_loaded.json`
- `administration_build_gate.json`
- `administration_build_report.json`
- `administration_render_report.json`
- `administration_learned_replay_validation.json`
- `administration_learned_replay_render_report.json`
- `administration_live_assets_report.json`
- `administration_live_policy_video_report.json`
- `administration_live_policy_presentation_video_report.json`
- `administration_live_policy_validation.json`
- `phase2_launch_status.json`
- `phase2_turn_held_out_evaluation.json`
- `phase2_policy_only_route_evaluation.json`
- `phase2_policy_only_training_route_report.json`
- `phase2_administration_policy_only_video_report.json`
- `phase2_administration_policy_only_presentation_report.json`
- `phase2_administration_visual_replay_validation.json`
- `phase2_administration_visual_upgrade_render_report.json`
- `phase2_administration_live_cinematic_report.json`
- `phase2_administration_live_cinematic_3x_presentation_report.json`
- `phase2_administration_live_cinematic_teaser_12x_report.json`
- `phase2_administration_live_cinematic_validation.json`
- `phase2_end_to_end_validation.json`
- `phase3_dynamic_dr_smoke_report.json`
- `phase3_dynamic_yielding_smoke_report.json`
- `phase3_dynamic_dr_training_report.json`
- `phase3b_staged_dynamic_dr_training_report.json`
- `phase3b_candidate2525_screen4.json`
- `phase3c_segment6_recovery_training_report.json`
- `phase3c_model2824_segment6_screen32.json`
- `phase3_proxy_model1850_segment6_full_dr_no_crossing_baseline32.json`
- `phase3h_reciprocal_yield_training_report.json`
- `phase3h_model2225_balanced_full_strength_screen4.json`
- `phase3i_recurrent_distillation_report.json`
- `phase3_recurrent_bootstrap_report.json`
- `phase3_recurrent_bootstrap_balanced_full_strength_screen4.json`
- `phase3j_recurrent_ppo_training_report.json`
- `phase3j_recurrent_ppo_training_progress.png`
- `phase3j_model475_balanced_full_strength_screen4.json`
- `phase3j_model600_balanced_full_strength_screen4.json`
- `phase3j_model700_balanced_full_strength_screen4.json`
- `phase3h_model2225_balanced_full_strength_seed9403_screen4.json`
- `phase3_safety_residual_bootstrap_report.json`
- `phase3_safety_residual_smoke_report.json`
- `phase3k_safety_residual_training_report.json`
- `phase3k_safety_residual_training_progress.png`
- `phase3k_zero_residual_base_seed9501_screen4.json`
- `phase3k_model175_balanced_seed9501_screen4.json`
- `phase3k_safety_residual_comparison.json`
- `phase3l_clearance_planner_bootstrap_report.json`
- `phase3l_clearance_planner_smoke_report.json`
- `phase3l_clearance_planner_training_report.json`
- `phase3l_clearance_planner_training_progress.png`
- `phase3l_original_route_control_balanced_seed9701_screen4.json`
- `phase3l_zero_policy_balanced_seed9701.json`
- `phase3l_model200_balanced_seed9701_screen4.json`
- `phase3l_clearance_planner_comparison.json`
- `phase3m_corrected_physics_bootstrap_report.json`
- `phase3m_targeted_recovery_smoke_report.json`
- `phase3m_targeted_recovery_training_report.json`
- `phase3m_targeted_recovery_training_progress.png`
- `phase3m_hybrid_model125_balanced_seed9701.json`
- `phase3m_hybrid_model125_balanced_seed9702.json`
- `phase3m_targeted_recovery_comparison.json`
- `phase3n_dynamic_safety_bootstrap_report.json`
- `phase3n_dynamic_safety_smoke_report.json`
- `phase3n_dynamic_safety_training_report.json`
- `phase3n_dynamic_safety_training_progress.png`
- `phase3n_control_balanced_seed9701.json`
- `phase3n_control_balanced_seed9702.json`
- `phase3n_model50_balanced_seed9701.json`
- `phase3n_model50_balanced_seed9702.json`
- `phase3n_model50_full_acceptance_seed10101.json`
- `phase3n_model50_static_regression_seed10201.json`
- `phase3n_model50_live_administration_dynamic12_seed10301.json`
- `phase3n_dynamic_safety_acceptance.json`
- `phase3n_administration_final_omniverse_report.json`
- `phase3n_administration_final_omniverse_3x_report.json`
- `phase4a_administration_dynamic_showcase_report.json`
- `phase4a_dynamic_safety_presentation_video_report.json`
- `phase4a_dynamic_safety_showcase_acceptance.json`
- `final_omniverse_administration_presentation_report.json`
- `final_omniverse_administration_presentation_acceptance.json`
- `administration_nav2_measured_bridge.json`
- `administration_nav2_measured_mission.json`
- `administration_nav2_measured_integration_gate.json`
- `phase3_geometry_rtx_refinement_validation.json`
- `phase3_launch_status.json`
- `PRODUCTION_REPOSITORY_REVIEW.md`
- `isaaclab_training_report.json`
- `isaaclab_training_progress.png`
- `isaaclab_evaluation_report.json`
- `isaaclab_sensor_training_report.json`
- `isaaclab_sensor_training_progress.png`
- `isaaclab_sensor_evaluation_report.json`
- `isaaclab_learned_route_playback_report.json`
- `isaaclab_learned_route_video_report.json`
- `isaaclab_learned_route_presentation_video_report.json`
- `isaaclab_learned_route_close_video_report.json`
- `isaaclab_learned_route_close_presentation_video_report.json`
- `block_a_flush_threshold_open_loop_audit.json`
- `../media/screenshots/administration_overview.png`
- `../media/screenshots/administration_vice_principal.png`
- `../media/screenshots/administration_principal.png`
- `../media/videos/administration_route.mp4`
- `../media/videos/learned_block_a_route/aisha-block-a-learned-route-step-0.mp4`
- `../media/videos/AI-SHA_IsaacLab_Block_A_Learned_Tour_3x.mp4`
- `../media/videos/AI-SHA_IsaacLab_Block_A_Learned_Tour_Close_3x.mp4`
- `../media/videos/administration_live_policy/aisha-block-a-learned-route-step-0.mp4`
- `../media/videos/AI-SHA_Administration_Live_Policy_3x.mp4`
- `../media/videos/phase2_administration_policy_only/aisha-block-a-learned-route-step-0.mp4`
- `../media/videos/AI-SHA_Phase2_Administration_Policy_Only_3x.mp4`
- `../media/videos/AI-SHA_Phase2_Administration_Visual_Upgrade.mp4`
- `../media/videos/phase2_administration_live_cinematic/aisha-block-a-learned-route-step-0.mp4`
- `../media/videos/AI-SHA_Phase2_Administration_Live_Cinematic_3x.mp4`
- `../media/videos/AI-SHA_Phase2_Administration_Live_Cinematic_Teaser_12x.mp4`
- `../media/videos/phase3n_administration_final_omniverse/aisha-block-a-learned-route-step-0.mp4`
- `../media/videos/AI-SHA_Phase3N_Administration_Final_Omniverse_3x.mp4`
- `../media/videos/phase4a_administration_dynamic_showcase_final/aisha-phase4a-dynamic-showcase-step-0.mp4`
- `../media/videos/AI-SHA_Phase4A_Administration_Dynamic_Safety_Showcase.mp4`
- `../media/videos/AI-SHA_Final_Omniverse_Administration_Presentation.mp4`
