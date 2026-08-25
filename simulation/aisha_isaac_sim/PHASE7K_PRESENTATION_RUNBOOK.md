# Phase 7K phototextured photogrammetric survey runbook

Phase 7K is a hybrid, metric phototextured administration survey. It combines
the supplied RoomPlan/LiDAR geometry, genuine AliceVision dense reconstruction
clusters and privacy-safe PBR materials derived from the supplied photographs.
Use the frozen video first and the interactive Isaac Sim scene second.

## Recommended venue sequence

1. Show the two raw evidence images in
   `media/screenshots/phase7k_phototextured_survey`. Explain that they contain
   311,750 captured vertices and 625,295 textured faces, but are incomplete
   because the source was a walkthrough rather than a closed-loop survey.
2. Play
   `media/videos/AI-SHA_Phase7K_Phototextured_Photogrammetric_Survey.mp4`.
3. Explain that the clean film hides only the incomplete floating dense
   fragments. It retains the same capture-derived finishes, metric geometry
   and frozen navigation collision layer.
4. Start the repeat-until-closed Omniverse presentation:

   ```bash
   cd simulation/aisha_isaac_sim
   tools/run_phase7k_live_omniverse.sh
   ```

5. To inspect the raw clusters in the full survey stage instead:

   ```bash
   tools/open_phase7k_survey_review.sh
   ```

## Accurate presentation language

Call the result a **hybrid metric phototextured survey** or a
**photogrammetry-backed administration digital twin**. It contains two genuine
dense, camera-textured capture clusters and seven image-derived PBR material
sets. A1 page 2 Block A and the metric RoomPlan/LiDAR scans remain the topology
and scale authority.

Do not call it a seamless or certified as-built photogrammetric survey. The
two capture clusters are independently and provisionally aligned; they are not
falsely welded. The Vice-Principal interior remains an assumption because it
was locked. The robot film replays recorded poses from the accepted live-policy
mission and is not another live policy execution or a physical release.

The raw clusters are visual evidence only. They have no collision API and do
not affect training, clearance, LiDAR, or navigation. The accepted Phase 7I
route-critical collision geometry remains frozen.

## Rebuild and recovery

Run the complete generation, USD build, PathTracing render, video encode and
acceptance pipeline with:

```bash
tools/run_phase7k_phototextured_survey.sh
```

If live Isaac Sim is unreliable at the venue, use the frozen MP4. It does not
depend on network access, ROS discovery, window focus, or attached hardware.
The frozen release is 1920 x 1080 at 24 fps, contains 480 PathTracing frames,
and passed 54/54 Phase 7K acceptance checks. Its video SHA-256 is
`5b823a3bf76619333e1f7b40fdb3b286323e3eae3aa20d2469501315be482c30`.
