# Phase 7L NuRec presentation runbook

## What is ready

Phase 7L contains two locally trained NVIDIA 3D Gaussian NuRec sectors:

- the captured atrium, reception and administration corridors; and
- the shared atrium approach, doorway and captured Principal's office.

The Principal sector is the default. AI-SHA and the unchanged Phase 7I
navigation/collision world are provisionally registered into that visual sector.
The accepted Phase 7E Principal-route poses are replayed; the presentation does
not execute the policy again.

## Before the presentation

Use the machine on which Phase 7L was built. The privacy-sensitive NuRec assets
must exist locally at:

```text
tmp/phase7l_nurec_runs/administration_full_nurec.usdz
tmp/phase7l_nurec_runs/principal_full_nurec.usdz
```

From the package directory, run the acceptance check:

```bash
python3 tools/validate_phase7l_nurec_gaussian_twin.py
```

Proceed only when it prints `Phase 7L validation: 43/43 checks passed`.

## Start the live Isaac Sim presentation

```bash
tools/run_phase7l_live_omniverse.sh
```

Allow Isaac Sim and the Gaussian asset to load. The viewport uses the real
captured Principal-route camera path while AI-SHA follows selected recorded
poses from mission segments 6-9. The loop continues until Isaac Sim is closed or
the launcher is interrupted with `Ctrl+C`.

If the GUI does not start, run the shorter native smoke first:

```bash
tools/run_phase7l_nurec_isaac_smoke.sh
```

## Suggested narration

> This environment is an NVIDIA 3D Gaussian reconstruction trained from our
> administration walkthrough and rendered natively in Isaac Sim. AI-SHA is
> replaying poses from the accepted learned-safety and Nav2 simulation mission
> inside the separately preserved navigation/collision world. The registration
> is presentation-grade, not certified survey control.

Point out the captured Principal-office wall, cabinetry, doorway approach and
meeting area. Do not describe the locked Vice-Principal interior as captured.

## Claim and privacy boundary

- The Gaussian reconstruction is visual-only; it is not used as collision or
  LiDAR geometry.
- Navigation collision geometry and the accepted policy/safety stack were not
  changed by this visual upgrade.
- The displayed motion is recorded-pose replay, not live policy execution.
- Registration uses shared visual features, gravity alignment and presentation
  assumptions; it is not certified survey control.
- The Vice-Principal interior remains assumed because it was locked.
- This is not physical localization, human-safety certification or permission
  for physical deployment.
- Do not distribute raw screenshots or the NuRec assets externally until school
  signage, portraits and documents have completed a human privacy review.

## Rebuilding

The normal smoke runner reuses the trained local assets. Full training is
intentionally opt-in:

```bash
AISHA_ALLOW_NUREC_RETRAIN=1 tools/run_phase7l_nurec_training.sh
```

Retraining requires the prepared NVIDIA 3DGRUT environment, substantial local
disk space and an NVIDIA GPU. It is unnecessary for normal presentation use.
