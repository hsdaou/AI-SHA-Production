# AI-SHA Isaac Sim implementation status

**Updated 2026-08-20. Approved-plan Block A cinematic and deterministic baseline implemented.**

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
- Page 2 of `DownloadBuildingRequestApprovedPlan.pdf` was reviewed and checksum
  locked. The route-scoped `administration.usd` now uses the plan's Block A
  topology: central 12.75 m atrium, 2.80 m east hallway, Vice-Principal in the
  east room cluster and Principal in the angled south-east suite.
- The walkthrough supplies appearance only: polished terrazzo, dark timber
  slats, warm-white walls, frosted glazing, light-oak offices, suspended LED
  panels, desks, chairs, cabinets and planters.
- A presentation shell makes the engineering URDF read as finished AI-SHA while
  leaving the Rev D collision, mass, drive and sensor model unchanged.
- Three ray-traced stills and a 10 s, 1280 x 720, 24 fps four-shot MP4 were
  rendered and visually checked. The MP4 contains 240 verified frames covering
  atrium departure, Vice-Principal visit, plan-derived transfer and Principal visit.
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

- The approved plan is now present and controls printed dimensions, room
  relationship and route topology. PDF trace offsets and final in-room goal
  offsets remain presentation placements rather than survey-grade coordinates.
- Door widths (1.10/1.05 m) and thresholds (3/5 mm) are presentation assumptions;
  both measured clear widths and threshold heights remain absent.
- Wall/ceiling height, wall thickness, office furniture offsets and decorative
  detail remain walkthrough-derived or presentation assumptions.
- High-fidelity contact asset: measured carrier spring curve and caster
  trail/inertia/height are absent.
- The LD19, D435 and BNO055 models/interfaces are known, but the low front LiDAR
  model and exact camera intrinsics remain hold points.
- Physical Nav2 release still depends on measured door/threshold data, an
  as-built navigation survey and closed-route commissioning.

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
- `PRODUCTION_REPOSITORY_REVIEW.md`
- `../media/screenshots/administration_overview.png`
- `../media/screenshots/administration_vice_principal.png`
- `../media/screenshots/administration_principal.png`
- `../media/videos/administration_route.mp4`
