# AI-SHA Isaac Lab training and Block A sensor route

This package registers the AI-SHA Isaac Lab tasks that drive the wheel joints
under PhysX. `Isaac-AISHA-OfficeNav-Direct-v0` is the original
state-observation doorway foundation. `Isaac-AISHA-BlockA-SensorNav-Direct-v0`
uses 36 course ray ranges and ten goal/vehicle state terms to learn all 12
directed segments of the Principal/Vice-Principal presentation route.
`Isaac-AISHA-Administration-Live-Direct-v0` runs the same observation/action
contract and checkpoint in the complete administration scene. The robot is never
moved by setting its root transform outside normal episode resets.

The Phase 2 tasks add arbitrary-heading turn acquisition and policy-only route
control. Phase 3 keeps the same 46-value observation contract while adding
tracked moving-person proxies and bounded sensor/actuation/dynamics
randomization.

The Block A task is a genuine sensor-grounded PPO curriculum, but its
MultiMeshRayCaster is a scalable geometric ray model rather than a validated RTX
LD19 model. The training course is plan-derived, both 1.40 m doors and both flush
thresholds are presentation assumptions, and Nav2/sim-to-real release remains
out of scope.

## Run

The launcher delegates to the installed Isaac Lab scripts while registering the
AI-SHA task from this repository. It defaults to `/home/robot-wst/IsaacLab`; set
`ISAACLAB_ROOT` to override that location.

```bash
cd simulation/aisha_isaac_sim/isaaclab

# Registry/environment smoke check
/home/robot-wst/IsaacLab/isaaclab.sh -p scripts/launch.py list
/home/robot-wst/IsaacLab/isaaclab.sh -p scripts/launch.py random \
  --task Isaac-AISHA-OfficeNav-Direct-v0 --num_envs 4 --headless

# Reproducible PPO smoke training
/home/robot-wst/IsaacLab/isaaclab.sh -p scripts/launch.py train \
  --task Isaac-AISHA-OfficeNav-Direct-v0 --num_envs 64 \
  --max_iterations 25 --seed 42 --headless
```

Training output is written below `logs/rsl_rl/aisha_office_nav/` or
`logs/rsl_rl/aisha_block_a_sensor_nav/` and ignored by Git. Evidence summaries
belong in `../results/`.

## Reproduce the selected Block A sensor run

The selected run used Isaac Sim 5.1, Isaac Lab commit `80094be3245`, an RTX
5080, seed 144, 32 parallel environments and 600 PPO iterations. It generated
614,400 transitions. Build the plan-derived course, train, and evaluate with:

```bash
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  tools/build_block_a_training_course.py

TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p scripts/launch.py train \
  --task Isaac-AISHA-BlockA-SensorNav-Direct-v0 --num_envs 32 \
  --max_iterations 600 --seed 144 --headless

TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p scripts/evaluate.py \
  --task Isaac-AISHA-BlockA-SensorNav-Direct-v0 \
  --checkpoint logs/rsl_rl/aisha_block_a_sensor_nav/<run>/model_599.pt \
  --output ../results/isaaclab_sensor_evaluation_report.json \
  --episodes-per-segment 48 --num_envs 96 --seed 5084 --headless
```

The final held-out run enforced an equal quota for every segment and achieved
576/576 successes, zero collisions and zero timeouts. Every segment passed
48/48. The evaluation seed is distinct from the training seed.

The continuous route player chains the same segments without teleporting. The
learned policy drives aligned legs; a deterministic differential-wheel
supervisor performs in-place turns and office dwell timing. The successful run
completed all 12 waypoints in 173.6 simulated seconds:

```bash
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p scripts/play_block_a_route.py \
  --checkpoint logs/rsl_rl/aisha_block_a_sensor_nav/<run>/model_599.pt \
  --output-report ../results/isaaclab_learned_route_playback_report.json \
  --max-steps 7200 --dwell-seconds 1.0 --trace-interval 3 \
  --seed 6084 --headless

TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p scripts/play_block_a_route.py \
  --checkpoint logs/rsl_rl/aisha_block_a_sensor_nav/<run>/model_599.pt \
  --output-report ../results/isaaclab_learned_route_video_report.json \
  --video-folder ../media/videos/learned_block_a_route \
  --max-steps 5300 --dwell-seconds 1.0 --trace-interval 0 \
  --seed 6084 --headless

python3 tools/make_learned_route_presentation_video.py \
  --input ../media/videos/learned_block_a_route/aisha-block-a-learned-route-step-0.mp4 \
  --output ../media/videos/AI-SHA_IsaacLab_Block_A_Learned_Tour_3x.mp4 \
  --report ../results/isaaclab_learned_route_presentation_video_report.json
```

The continuous player records a time-stamped pose trace for visual replay. That
trace does not make the current administration USD photoreal, and visual replay
must be disclosed separately from live policy physics.

## Run the checkpoint live in the administration scene

First build `../scenes/administration.usd` using the separately supplied plan
and the explicit presentation-assumption flag. Then compose a live environment
that excludes the scene's old pose-replay robot and nested physics scene, and a
loaded articulation whose visual shell follows the physical base:

```bash
cd simulation/aisha_isaac_sim

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

The validated run completed 12/12 route segments in 5,388 ticks and 179.6
simulated seconds without collision or reset. The unchanged `model_599.pt`
checkpoint (SHA-256
`3da826c515d0e58a3c0731dd3b72022208eadda3cc2bd2200a78d92f079bfacf`)
provided 4,378 learned-policy ticks. A deterministic, wheel-physics turn
supervisor provided 950 ticks and office dwell control provided 60; this is not
an end-to-end learned mission and is retained as historical Phase 1 evidence.
That capture used the older 3.4 m by 2.15 m body-follow camera. The validator
checks route/control accounting, checkpoint and asset hashes, live stage state,
finite trace data, assumed doors, video completeness, camera occlusion and the
motion-preserving 3x edit. The Phase 2 evidence below supersedes it for the main
technical presentation.

## Phase 2 fine-tuning and policy-only gate

Phase 2 resumed the selected Phase 1 checkpoint and trained the missing moving
turn/handoff behaviour. The final curriculum starts every segment on its actual
incoming route heading with ±15 degree jitter and 0.30-0.50 m/s initial speed,
adds 10 mm standard-deviation observation noise and 0.25% ray dropout, and
rewards heading-error reduction. The executed search used 1,200
incoming-transition iterations plus 300 moving-handoff refinement iterations
across 32 environments, or 1,536,000 policy transitions:

```bash
cd simulation/aisha_isaac_sim
isaaclab/scripts/run_phase2_training.sh
```

The script validates the checkpoint hash and NVIDIA driver before launching and
never overwrites Phase 1. The selected checkpoint is:

```text
checkpoints/aisha_phase2_policy_model_1850.pt
SHA-256: 3ab596c61259784657b36fe4ee937da8495ce9621a43bd04e2c8a0bf6e0b1880
```

This packaged 1.3 MB artifact is byte-identical to the checkpoint from source
run `2026-08-22_09-12-57_phase2_moving_transition_seed256/model_1850.pt`.

After selecting a candidate checkpoint, run the complete acceptance pipeline:

```bash
isaaclab/scripts/run_phase2_gates.sh \
  "$PWD/isaaclab/checkpoints/aisha_phase2_policy_model_1850.pt"
```

This evaluates 48 randomized incoming-transition episodes per segment, then 48
full chained routes, one policy-only training-course route, and finally one
live administration capture. In `--route-control policy-only` mode every wheel
action comes from the checkpoint: turn and dwell overrides are disabled.

The selected checkpoint passed:

- transition gate: 570/576 (98.96%), six collisions, zero timeouts, minimum
  segment and minimum office-exit rate 87.5%;
- chained-route gate: 46/48 (95.83%), two collisions, zero timeouts;
- deterministic training-course route: 12/12 in 4,752 policy steps;
- deterministic administration route: 12/12 in 4,885 policy steps / 162.83 s,
  zero collisions and zero supervisor steps.

The final validator requires the 3.8 m by 2.4 m route-leg camera, no post-start
camera occlusion, unchanged presentation motion and consistent checkpoint
hashes. All 18 checks pass in `../results/phase2_end_to_end_validation.json`.
The presentation-ready 3x video is
`../media/videos/AI-SHA_Phase2_Administration_Policy_Only_3x.mp4`; the complete
1x evidence capture is in `../media/videos/phase2_administration_policy_only/`.

Training and capture ran on kernel `6.17.0-35-generic`, NVIDIA driver
580.159.03 and the RTX 5080. This gate supports policy-only control in the
declared simulation. Dynamic-obstacle/domain randomization, Nav2, measured site
geometry and sim-to-real commissioning remain outside the claim.

## Current upgraded-scene learned skill ensemble

The visually upgraded administration scene exposed a deterministic stop after
the hallway-return pivot that the earlier single checkpoint did not encounter
in the prior live scene. PPO live-scene curricula acquired the turn
stochastically but did not produce a reliable deterministic mean. The accepted
integration therefore uses a transparent hierarchical learned controller:

- `checkpoints/aisha_phase2_administration_base_model_2150.pt` controls segments
  0-5 and 7-11;
- `checkpoints/aisha_phase2_administration_principal_specialist.pt`, trained by
  behavior cloning from 16/16 successful live-physics demonstrations, controls
  segment 6;
- the route planner selects the declared learned skill but never modifies its
  two wheel actions.

The specialist separately passed 8/8 deterministic trials with zero collisions
or timeouts. The complete cinematic gate then passed 12/12 in 5,160 learned
policy steps with zero supervisor/dwell steps. Reproduce the exact capture and
edits with:

```bash
cd simulation/aisha_isaac_sim
TERM=xterm /home/robot-wst/isaacsim/python.sh \
  isaaclab/scripts/play_block_a_route.py \
  --task Isaac-AISHA-Administration-Live-Direct-v0 \
  --checkpoint isaaclab/checkpoints/aisha_phase2_administration_base_model_2150.pt \
  --segment-policy-checkpoint \
    6=isaaclab/checkpoints/aisha_phase2_administration_principal_specialist.pt \
  --output-report results/phase2_administration_live_cinematic_report.json \
  --video-folder media/videos/phase2_administration_live_cinematic \
  --route-control policy-only --camera-mode cinematic \
  --max-steps 6000 --trace-interval 3 --seed 7084 --headless

/home/robot-wst/isaacsim/python.sh \
  isaaclab/tools/make_administration_live_policy_presentation_video.py \
  --input media/videos/phase2_administration_live_cinematic/aisha-block-a-learned-route-step-0.mp4 \
  --run-report results/phase2_administration_live_cinematic_report.json \
  --output media/videos/AI-SHA_Phase2_Administration_Live_Cinematic_3x.mp4 \
  --report results/phase2_administration_live_cinematic_3x_presentation_report.json \
  --speed 3

/home/robot-wst/isaacsim/python.sh \
  isaaclab/tools/validate_phase2_live_cinematic.py \
  --checkpoint isaaclab/checkpoints/aisha_phase2_administration_base_model_2150.pt
```

The main film is 57.2 seconds and the 12x teaser is 14.3 seconds. The overlay
accurately states `PPO base + imitation specialist`; both are unchanged-motion
edits of the 172.0-second raw live-policy capture. See
`checkpoints/administration_policy_ensemble.json` and
`../results/phase2_administration_live_cinematic_validation.json`.

## Phase 3 dynamic obstacles and domain randomization

Phase 3 resumes the hash-locked accepted administration base checkpoint. Each
replicated course can contain up to two active 1.70 m by 0.48 m kinematic
person capsules crossing selected open route legs. They have physical collision
and are tracked by the same `MultiMeshRayCaster` that supplies the 36 policy
ranges; no obstacle pose or velocity is exposed to the policy.

The episode randomization covers 0-2 policy steps of command latency, motor
strength, wheel radius/track, joint damping, base mass/inertia, friction and
LD19-style noise, dropout, bias and scale. Collision termination always uses
uncorrupted geometric ranges. These ranges are deliberately bounded assumptions
until hardware/site calibration exists.

```bash
cd simulation/aisha_isaac_sim

# Four replicated environments, 180 policy steps, auditable JSON gate.
TERM=xterm /home/robot-wst/IsaacLab/isaaclab.sh -p \
  isaaclab/scripts/smoke_phase3_dynamic.py --num-envs 4 --steps 180 --headless

# Hash-checks model_2150_rehearsal.pt, checks the NVIDIA driver, then resumes PPO.
isaaclab/scripts/run_phase3_dynamic_training.sh

# Plan/hash/USD/material validation after rebuilding the scene.
TERM=xterm /home/robot-wst/isaacsim/python.sh \
  isaaclab/tools/validate_phase3_geometry_rtx.py
```

The 2026-08-22 smoke passed 11/11 runtime checks. An abrupt full-strength run
was rejected after its final training batch fell to 0% success / 100%
collisions. The replacement run uses 100 static-retention iterations, a
350-iteration linear ramp and 150 full-strength iterations and is recorded at
`logs/rsl_rl/aisha_block_a_sensor_nav/2026-08-22_14-02-08_phase3b_staged_dynamic_dr_seed8101`.
Both runs executed 614,400 transitions. Training-batch success is not an
acceptance result. A checkpoint remains
unaccepted until 64 randomized episodes per segment, a static Phase 2 route
regression, and the declared live-administration dynamic scenarios pass.
Dynamic capsules do not establish human-safety behaviour.

The later Phase 3L task adds a bounded recurrent brake/steering-request layer,
a one-second rectangular-footprint clearance projector, and a policy-independent
hysteretic LiDAR stop around the frozen route actor. Its runtime gate passes
21/21 checks. After 600 PPO iterations / 1,228,800 transitions, model 200
achieved 29/48 successes with 5 collisions and 14 timeouts on seed 9701,
compared with 19/48 and 13 collisions for the original route control. This is a
promoted architecture candidate, not full Phase 3 acceptance or physical safety
evidence; see `../results/phase3l_clearance_planner_comparison.json`.

The current screening candidate is `model_2525.pt`: 43/48 deterministic
unseen-seed episodes passed with zero collisions. This is not acceptance: the
screen used only four episodes per segment, and segment 6 timed out 4/4. The
next learning target is segment-6 dynamic rehearsal without losing the other
eleven route skills.

The administration scene is separately tagged `GEOMETRY-RTX-PHASE3-A` and
`administration_rtx_pbr_v2`. It uses the page-2 Block A printed dimensions as
anchors, including 12.75 m atrium, 2.80 m hall and 4.73 m Principal frontage,
and adds albedo, roughness and tangent-normal maps to four finish families.
Offline stills default to RTX PathTracing at 64 samples per pixel. Door widths,
flush thresholds, heights and furniture placement remain disclosed
presentation assumptions—not site measurements or a photogrammetric twin.

## Reproduce the first baseline

The pinned run used Isaac Sim 5.1, Isaac Lab commit `80094be3245`, an RTX 5080,
seed 42, 64 parallel environments and 200 PPO iterations:

```bash
/home/robot-wst/IsaacLab/isaaclab.sh -p scripts/launch.py train \
  --task Isaac-AISHA-OfficeNav-Direct-v0 --num_envs 64 \
  --max_iterations 200 --seed 42 --headless

/home/robot-wst/IsaacLab/isaaclab.sh -p scripts/evaluate.py \
  --checkpoint logs/rsl_rl/aisha_office_nav/<run>/model_199.pt \
  --output ../results/isaaclab_evaluation_report.json \
  --episodes 512 --num_envs 64 --seed 1001 --headless

/home/robot-wst/IsaacLab/isaaclab.sh -p scripts/launch.py play \
  --task Isaac-AISHA-OfficeNav-Direct-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/aisha_office_nav/<run>/model_199.pt \
  --video --video_length 540 --seed 1001 --headless
```

The first held-out unseen-seed baseline achieved 420/512 doorway successes
(82.03%), 88 collisions (17.19%) and 4 timeouts (0.78%). This supports a real
state-observation learning demonstration, not a sensor-policy, Nav2, physical
release or photoreal administration-office claim. See
`../results/isaaclab_evaluation_report.json` for the checkpoint hash and exact
protocol.
