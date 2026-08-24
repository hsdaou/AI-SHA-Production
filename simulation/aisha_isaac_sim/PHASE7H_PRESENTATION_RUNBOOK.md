# Phase 7H presentation runbook

## Preferred presentation order

1. Play the validated Full HD film first:

   ```bash
   cd /home/robot-wst/Documents/Codex/2026-08-20/continue-the-ai-sha-isaac-sim/work/AI-SHA-Production/simulation/aisha_isaac_sim
   xdg-open media/videos/AI-SHA_Phase7H_Photogrammetry_Informed_Omniverse.mp4
   ```

2. If the venue machine has the RTX 5080/driver available, open the repeat-until-closed Isaac Sim GUI presentation:

   ```bash
   tools/run_phase7h_live_omniverse.sh
   ```

3. Close Isaac Sim normally when finished. The launcher records the live GUI
   session in `tmp/phase7h_live_omniverse_session.json`.

## What to say

- The robot motion came from the accepted live Nav2/learned-safety Isaac
  simulation mission and completed all 12 route legs.
- The Principal geometry is registered at 1:1 scale from the iPhone RoomPlan
  capture, while the approved page-2 plan controls the wider route topology.
- Real dense photogrammetry was computed from the walkthrough: 278,696 corridor
  points and 202,347 Principal points. Because the tour was not captured as a
  controlled photogrammetry orbit, its raw meshes are incomplete and are kept
  as survey evidence rather than unsafe collision geometry.
- The visible Principal floor and walnut finish use privacy-safe PBR maps
  derived from the supplied clean stills.
- The Vice-Principal interior remains an explicit assumption because it was
  locked. The 0.85 m minimum door and 2.12 m height are user-reported; the
  individual 0.875 m VP and 0.90 m Principal openings are presentation
  assumptions.
- This is simulation/presentation evidence, not a physical deployment release.

## Venue fallback

If Isaac Sim does not open, use the MP4. Do not attempt a driver/kernel change
at the venue. The film contains the same wide cameras and verified motion and
does not require Isaac Sim to be running during playback.
