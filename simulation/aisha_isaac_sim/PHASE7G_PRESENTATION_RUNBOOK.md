# Phase 7G Omniverse presentation runbook

Phase 7G freezes the presentation package independently of the physical driver,
encoders and USB-RS485 adapter. It uses the accepted Phase 7E live
Nav2/learned-safety mission as its motion source and the Phase 7F Full HD
PathTracing footage as its wide-camera visual source.

## Recommended presentation sequence

1. Start with the frozen 46-second backup reel:

   `media/videos/AI-SHA_Phase7G_Omniverse_Presentation_Freeze.mp4`

2. Explain that the main mission footage is a Full HD path-traced replay of
   recorded poses from the accepted live simulation source. The dynamic insert
   shows the accepted learned-brake and controlled stop-wait-resume encounter.
3. Launch the GUI Omniverse replay if the venue workstation is stable:

   ```bash
   cd simulation/aisha_isaac_sim
   tools/run_phase7g_live_omniverse.sh
   ```

   The Isaac Sim viewport uses the same eight wide, human-height cameras and
   repeats the 12-leg route until the window is closed or `Ctrl+C` is pressed.
4. If projector, GPU or window-focus behavior is uncertain, use the frozen MP4.
   It is hash-linked to the accepted sources and contains the complete intended
   story without depending on a live GUI.

## Suggested narration

- AI-SHA completes the full administration route and visits the Vice
  Principal's and Principal's offices.
- The source mission ran through the accepted Nav2, static-map/live-LiDAR fusion
  and learned 360-degree safety stack in Isaac Sim/Isaac Lab.
- The dynamic encounter demonstrates learned brake authority followed by a
  controlled protective stop, wait and resume with zero contacts.
- The displayed environment is a plan-, iPhone-capture- and walkthrough-informed
  presentation twin. It is not a photogrammetric or as-built survey.
- The Vice Principal office interior is a disclosed presentation assumption
  because the room was locked during capture.

## Claim boundary

The GUI player is a live Omniverse visualization of recorded successful poses;
it does not execute the policy again. Its source motion did execute the accepted
Nav2 and learned-safety simulation stack. Nothing in this package establishes
physical localization, protective-safety performance or permission to deploy
the robot in an occupied building.
