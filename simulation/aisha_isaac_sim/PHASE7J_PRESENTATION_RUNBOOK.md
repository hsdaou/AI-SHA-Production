# Phase 7J complete captured-area presentation runbook

Phase 7J is the presentation-ready Omniverse view of the complete area in the
supplied primary iPhone RoomPlan capture, plus the independently registered
Principal-office supplement. Use the frozen video first and the interactive
Isaac Sim scene second.

## Recommended venue sequence

1. Play the Full HD backup film:

   `media/videos/AI-SHA_Phase7J_Complete_Captured_Administration_Twin.mp4`

2. Explain that the complete primary RoomPlan area is visible at 1:1 metric
scale and that the Principal supplement is registered at its approved
page-2 approach, including explicit floor-level Z registration.
3. Explain that the robot poses came from the accepted 12-leg live
   Nav2/learned-safety mission. The film replays those recorded poses in the
   upgraded environment; it does not run the policy a second time.
4. On the project workstation, start the repeat-until-closed Omniverse player:

   ```bash
   cd simulation/aisha_isaac_sim
   tools/run_phase7j_live_omniverse.sh
   ```

5. Close Isaac Sim or press `Ctrl+C` in the launch terminal when finished.

The interactive scene is
`scenes/phase7j_complete_captured_administration.usda`. The self-contained
complete visual layer is
`scenes/phase7j_complete_captured_administration_visual.usdc`.

The frozen release is 1920 x 1080 at 24 fps, contains 480 PathTracing frames,
and passed 49/49 Phase 7J acceptance checks. The release video SHA-256 is
`49ab74ee97cd9f7b5b51bd115ba8817310673bb298238e0d9145640b0fe7c6fd`.

## Suggested narration

- A1 page 2 Block A controls the global administration topology.
- The supplied RoomPlan scan supplies complete captured semantic geometry,
  including walls, doors, windows and capture-time furniture.
- The independently captured Principal suite is registered at 1:1 scale.
- AI-SHA completes all 12 accepted mission legs, visits the Principal and Vice
  Principal destinations, reverses where required, and returns home.
- The source simulation used Nav2, static-map/live-LiDAR fusion and the learned
  360-degree safety layer. The accepted high-speed hallway legs reached about
  0.745 m/s from the 0.80 m/s command target.
- The central atrium polygon is a mapped hard no-go because it is 0.20 m below
  the robot floor.

## Evidence and claim boundary

It is accurate to call this a complete captured-area **semantic digital twin**
and a real Omniverse/Isaac Sim presentation of an accepted learned-policy
mission. RoomPlan provides metric surfaces and object semantics, not camera
textures. Capture-derived PBR materials improve presentation quality, but the
result is not a complete phototextured photogrammetric or certified as-built
survey.

The long handheld RoomPlan survey contains local drift and records door panels
as closed semantic slabs. Where a captured wall/opening fragment enters the
0.42 m corridor already authorized by the approved plan and accepted collision
layer, that fragment is suppressed only in the presentation composite and is
listed in the build report. The complete source geometry remains present in the
separate survey visual layer.

Movable capture-time furniture uses a conservative 0.85 m radial visual envelope so
the robot's rectangular corners remain clear during turns. Those visibility
overrides are also presentation-only and are enumerated in the build report.

The primary RoomPlan floor surface does not span the complete accepted route.
The composite therefore displays the validated plan-authority floor instead,
including the reported 0.20 m lower atrium no-go polygon. The original RoomPlan
floor remains present in the complete survey visual layer.

The Vice-Principal interior is a presentation assumption because the room was
locked. The reported administration minimum doorway is 0.85 m wide and 2.12 m
high; the VP and Principal presentation openings remain disclosed assumptions
of 0.875 m and 0.90 m. Thresholds remain assumed flush. The visual replay does
not establish physical localization, occupied-building protective safety, or a
physical deployment release.

The Principal-visit shot uses a temporary architectural cutaway so both
correctly elevated RoomPlan wall sets cannot occlude the robot. The exact wall,
door and window visibility states are restored before the final overview.

## Fallback and recovery

If the live Isaac Sim window fails to open or the venue GPU/projector behaves
unreliably, play the frozen MP4. It is the accepted presentation artifact and
does not depend on GUI focus, network access, ROS discovery or attached motor
hardware.

To rebuild and revalidate every Phase 7J artifact:

```bash
cd simulation/aisha_isaac_sim
tools/run_phase7j_complete_captured_twin.sh
```

The acceptance report is
`results/administration_nav2_phase7j_complete_captured_twin_acceptance.json`.
