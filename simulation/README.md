# Isaac Sim presentation and validation

`aisha_isaac_sim/` contains the Isaac Sim 5.1 implementation of the audited Rev D
differential-drive AI-SHA proof of concept. It is intentionally separate from
the older mecanum ROS configuration under `src/`; do not mix their footprints or
kinematics.

The package includes:

- generated 59.25 kg empty and 69.25 kg loaded URDF/USD assets;
- deterministic flat-floor and threshold geometry scenes;
- unit, smoke and full dynamic validation evidence;
- a disclosed Block A administration presentation proxy;
- a hash-locked Phase 3M pivot/clearance stack with a trained one-action,
  360-degree Phase 3N dynamic-obstacle brake layer;
- production-reconciled LD19, RealSense D435 and BNO055 contracts; and
- an overview render at `aisha_isaac_sim/media/screenshots/administration_overview.png`.

The approved A1 page-2 plan is not redistributed in this repository. The
administration scene was calibrated against it but retains explicitly tagged
door, threshold, furniture and finish assumptions. It is suitable for a
simulation presentation, not for construction, physical route release, safety
validation, or unsupervised operation. Start with `aisha_isaac_sim/README.md`
and `aisha_isaac_sim/results/PRODUCTION_REPOSITORY_REVIEW.md`.
