# Phase 7M NuRec presentation reel runbook

## Deliverable

Phase 7M is the local Full HD presentation preview of AI-SHA travelling from
the captured atrium approach into the captured Principal-office area. It uses
NVIDIA Isaac Sim's NuRec renderer and the accepted Phase 7E navigation pose
trace.

The reel is 1920×1080, 24 fps and 13.75 seconds. It remains local because the
visual reconstruction comes from a school walkthrough:

```text
media/videos/AI-SHA_Phase7M_NuRec_Principal_Visit_Presentation.mp4
```

## Watch it

From the `simulation/aisha_isaac_sim` directory:

```bash
tools/open_phase7m_nurec_presentation_reel.sh
```

The launcher validates the scene, render, video hash and claim boundary before
opening the video. A presentation operator may also open the MP4 directly.

## What changed in this phase

The first robot-overlay probe found that the PCA gravity axis had the correct
direction but the wrong sign. Phase 7M resolves the sign from COLMAP's physical
camera-up vectors and scales the visual registration from the Principal
doorway-to-turn route anchor. The corrected validation is:

- shared-area median residual: 0.0469 m;
- shared-area p95 residual: 0.1802 m;
- gravity residual: 2.36°;
- Principal doorway-to-turn anchor residual: less than 1 mm; and
- reconstructed ceiling: 2.933 m against the 3.0 m presentation assumption.

The robot is now upright and floor-aligned. Four fixed captured viewpoints keep
the robot readable without letting it fill most of the frame or sending the
camera outside the trained reconstruction.

## Suggested narration

> This is AI-SHA inside a NuRec reconstruction trained from our administration
> walkthrough and rendered in NVIDIA Isaac Sim. The movement comes from the
> accepted Nav2 and learned-safety simulation mission. For a clear presentation,
> recorded poses are replayed and retimed through four captured viewpoints; the
> policy is not executing again during this video.

The reel labels that boundary on screen. Do not describe it as physical
autonomy, certified survey registration or a physical safety release.

## Privacy boundary

The selected camera codes exclude the close certificate and portrait views and
no live people were observed in the final frames. Raw capture, Gaussian assets,
render frames, QA sheets and the encoded video are ignored by Git.

The coding-agent visual selection review is complete, but external distribution
is not approved. Before sending the reel outside the authorized project team,
the user or another authorized school representative must review signage,
documents and institutional imagery.

## Rebuild

Rebuilding reuses the trained local NuRec assets and takes roughly two minutes
on the current RTX 5080 workstation:

```bash
tools/build_phase7m_nurec_presentation_reel.sh
```

NuRec retraining is not required for an ordinary reel rebuild.
