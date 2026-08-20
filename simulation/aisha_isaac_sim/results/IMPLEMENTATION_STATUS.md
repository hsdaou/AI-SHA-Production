# AI-SHA Isaac Sim implementation status

**Updated 2026-08-20. Baseline import and deterministic flat-floor validation implemented.**

## Completed

- Workstation inventoried: Ubuntu 24.04.4, RTX 5080 16 GB, driver 580.159.03,
  Isaac Sim 5.1.0, ROS 2 Jazzy, X11. Isaac Lab was not found.
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
- A route-scoped `administration.usd` presentation proxy builds and reopens
  cleanly. It includes the atrium, south-east hallway, both office approaches,
  assumed doors/thresholds, furniture cues, route markers, loaded robot and
  presentation camera.
- The assumed Vice-Principal 1.10 m opening provides 166 mm nominal clearance
  per side; the assumed Principal 1.05 m opening provides 141 mm per side.
- AI-SHA-Production commit `8893535` was reconciled for sensor/system contracts:
  LD19 `/scan`, RealSense D435 aligned depth, and BNO055 `/imu/data` at 50 Hz.

## Deterministic contact tuning disclosure

The rigid six-contact proxy is statically indeterminate. Exactly coplanar
contacts allowed PhysX to unload one driven wheel, so the asset applies a
symmetric 1 mm **simulation rest offset** to both drive-wheel colliders. This is
a solver seating bias, not a physical tyre-radius, preload, or contact-load
claim. See `config/physics_materials.yaml`.

## Presentation assumptions and authoritative blockers

- The A1 page-2 plan is absent from both the supplied bundle and the reviewed
  production repository. The presentation scene is explicitly not plan-confirmed.
- Principal and Vice-Principal goal poses are presentation assumptions, not
  documented site coordinates.
- Door widths (1.10/1.05 m) and thresholds (3/5 mm) are presentation assumptions;
  both measured clear widths and threshold heights remain absent.
- High-fidelity contact asset: measured carrier spring curve and caster
  trail/inertia/height are absent.
- The LD19, D435 and BNO055 models/interfaces are known, but the low front LiDAR
  model and exact camera intrinsics remain hold points.
- Physical Nav2 release still depends on a scaled building map and measured route.

## Evidence

- `workstation_inventory.json`
- `import_report.json`
- `scene_build_report.json`
- `validation_smoke_empty.json`
- `validation_smoke_loaded.json`
- `validation_full_loaded.json`
- `administration_build_gate.json`
- `administration_build_report.json`
- `PRODUCTION_REPOSITORY_REVIEW.md`
- `../media/screenshots/administration_overview.png`
