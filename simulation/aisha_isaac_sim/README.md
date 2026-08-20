# AI-SHA - Isaac Sim package (proof of concept)

**Rev E - 2026-08-20 - differential-drive design baseline, two-office demo scope**

This package describes the simplified indoor proof-of-concept: two driven hub
wheels on the centre lateral axis, four physical swivel castors, a retained
Prestar NF-301 deck, a guided compliant drive carrier, and a fixed mast
reinforcement spine. It supersedes every earlier 4-wheel skid-steer model.

It is a coherent simulation/design baseline, not a fabrication release or a
safety certification. The unresolved hold points are listed below and in
`config/aisha_drive.yaml`.

## Architecture decision

**Keep four matched swivel castors for this tall PoC**, because their broad
front/rear footprint is useful for static stability. Do not keep either of the
NF-301's fixed-direction castors, and do not bolt all six contacts rigidly to
one chassis.

The two drive wheels need guided, adjustable compliance from the first build:

- target 12-13 kgf normal load on each drive wheel at the 10 kg payload case;
- starting spring rate 20-25 N/mm per side, about 5-6 mm loaded compression;
- at least 3 mm rebound and 4 mm bump travel, with captured guides and stops;
- independent left/right response, or equivalent small roll of the drive beam;
- springs carry vertical load only; guides/stops carry fore/aft and yaw loads.

Battery location does not set wheel load in a six-contact system. Preload and
geometry do. Corner-weigh with both drive wheels on scales and equal-height
plates beneath every other contact; a single scale under one wheel changes the
geometry and gives a misleading result.

## Files

```text
urdf/aisha.urdf               59.25 kg design-empty baseline
urdf/aisha_max_payload.urdf   69.25 kg with the 10 kg PoC payload
config/aisha_drive.yaml       geometry, sources/status, limits, safety and hold points
config/demo_route.yaml        two-office demo route (Vice-Principal, Principal)
tools/generate_aisha_urdf.py  canonical source for both URDFs
tools/validate_urdf.py        mass, inertia, frames, drive and stability checks
```

Edit the generator and regenerate both URDFs; do not hand-edit the XML.

```bash
python3 tools/generate_aisha_urdf.py
python3 tools/validate_urdf.py
```

## What changed from Rev C

| Item | Rev C | Rev D |
|---|---:|---:|
| Empty / loaded design mass | 56.05 / 66.05 kg | **59.25 / 69.25 kg** |
| Battery mass | 13 kg allowance | **11.25 kg datasheet value** |
| Structure mass | 6 kg | **10 kg conservative parts allowance** |
| Tray model | 25 mm solid proxy, 4 kg | **3 mm sheet + posts, 5 kg** |
| Deck envelope | 55 mm | **32 mm from 210-178 mm published heights** |
| Normal wheel effort | 18 N.m peak | **6 N.m rated** |
| Front LiDAR scan height | 0.20 m, intersecting deck | **0.25 m, above deck** |
| Camera frames | body frame only | **body + ROS optical frame** |
| IMU frame | absent | **present; as-built XYZ pending** |
| Overall envelope | incorrectly ~0.91 m long | **0.768 W x 1.180 L x 1.190 H m** |
| Stability | nominal contact locations | **30 mm inward caster-trail allowance** |

The mass increase is not payload growth. It corrects parts previously omitted or
under-counted: the RHS beam/spine, axle and backing plates, compliance hardware,
fasteners, and tray posts.

## Model summary

Convention: **+X forward, +Y left, +Z up**; `base_link` is on the floor at the
deck centrelines.

| Item | Design baseline |
|---|---|
| Drive wheels | V2.18 assumption: Ø200 x 48 mm at x=0, y=±0.360, z=0.100 m |
| Castor swivel axes | x=±0.350, y=±0.255 m; Ø130 mm contacts |
| Deck | 910 x 610 mm; top z=0.210 m |
| Tray | 805 x 610 x 3 mm; surface z=0.530 m |
| Mast / head | mast x=0.420 m; head Ø450 at x=0.500, z=0.925 m |
| Crown LiDAR | scan frame z=1.170 m; housing top z=1.190 m |
| Front LiDAR | scan frame x=0.455, z=0.250 m |
| Envelope | **0.768 m wide, 1.180 m long, 1.190 m high** |

Validated design-mass results:

| | Empty | 10 kg payload |
|---|---:|---:|
| Mass | 59.25 kg | 69.25 kg |
| CG (x, z) | (+0.0886, 0.3330) m | (+0.0682, 0.3723) m |
| Conservative front static tip | 34.8° | 34.1° |
| Conservative lateral static tip | 44.1° | 41.7° |

The tip calculation moves every castor contact 30 mm inward from its swivel
axis. It is still a static estimate from unmeasured masses. Repeat it after the
finished robot is weighed and corner-weighed.

## Simulation abstractions - do not confuse them with hardware

### Castors

The physical robot has four 360-degree swivel castors. The baseline URDF uses
fixed low-friction spheres because detailed two-DOF castors commonly chatter in
PhysX and consume solver iterations. Assign `castor_low_friction` from the YAML.

This proxy is suitable for navigation integration and deterministic demos. It
cannot validate swivel reversal, breakaway torque, flutter, floor marking,
pivot current or odometry transients. Create a separate high-fidelity caster
test scene before claiming sim-to-real performance.

### Drive compliance

The physical drive carrier is guided and compliant; the baseline URDF is rigid.
Flat-floor navigation tests may use the baseline. Threshold, uneven-floor,
contact-load and traction-transfer tests need an articulated Isaac asset with
vertical/roll compliance and the measured spring curve.

### Friction

All friction values are starting assumptions. Measure the delivered tyre and
castor materials on the actual polished tile, then calibrate the USD materials.

## Motor and driver limits

Only `left_wheel_joint` and `right_wheel_joint` are actuated, both about +Y.
Positive velocity must drive toward +X; verify after import.

The URDF effort limit is **6 N.m**, the motor's rated torque. The reported
18 N.m peak must be controller-timed and thermally/current limited; it is not a
continuous physics limit. One ZLAC8015D is provisionally suitable for the two
6 A-rated motors, but the exact V4.2 hardware label, current allocation and
thermal behaviour still require verification.

The official driver manual gives a single 15 A typical / 30 A maximum output
table without identifying a per-channel value. Supplier correspondence recorded
by the project says aggregate. Do not command simultaneous 18 A motor peaks.

## Source control and fabrication hold

The wheel reference PDF in this workspace is explicitly **V2.0**, with Ø206 mm
and 48.8 mm tread. The model is for **V2.18**, using supplier-stated Ø200±2 mm
and 48±1 mm values. Those variants are not interchangeable for axle height or
bracket release.

Before cutting metal:

1. Obtain the exact V2.18 single-shaft dimensioned drawing and exact V4.2 driver manual.
2. Measure both received wheels: free OD, loaded circumference, tread/body width,
   shaft and mounting-face stack.
3. Measure new caster overall height, trail, plate pattern and swivel-axis locations.
4. Inspect and photograph the NF-301 underside. Have the concentrated dynamic
   load path, fasteners and anti-loosening method reviewed by a competent engineer/fabricator.
5. Measure the narrowest door clear width. The 0.768 m robot needs **at least
   0.920 m** for the controlled demo target of roughly 75 mm clearance per side.
6. Weigh the stripped deck, structure, head and finished assembly; update the generator.

Published NF-301 trolley capacity is a manual cargo rating. It does not certify
the modified deck for powered, cantilevered, dynamic wheel loads.

## Import notes

Use `isaac_sim_import` in the YAML:

- floating base; merge fixed joints; import authored inertia;
- collision from collision geometry only; self-collision initially off;
- do not replace wheel cylinders with capsules;
- velocity targets, force drive, zero position stiffness;
- confirm 59.25 / 69.25 kg after import;
- confirm crown scan frame 1.170 m, front scan frame 0.250 m, and the camera
  optical convention (+Z forward, +X right, +Y down);
- confirm wheel contact force is plausible, but do not interpret rigid-URDF
  distribution as proof of physical spring preload.

## Workstation implementation (Isaac Sim 5.1)

The bundle now includes a headless, reproducible first implementation for Isaac
Sim 5.1. Run it with the simulator's Python, not the system interpreter:

```bash
ISAAC_ROOT=/home/robot-wst/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
python3 tools/generate_aisha_urdf.py
python3 tools/validate_urdf.py
python3 tools/inventory_workstation.py
python3 tools/test_controller.py
"$ISAAC_ROOT/python.sh" scripts/import_urdf.py --headless
"$ISAAC_ROOT/python.sh" scripts/build_validation_scenes.py --headless --payload loaded
"$ISAAC_ROOT/python.sh" scripts/run_validation.py --headless --suite smoke --payload loaded
"$ISAAC_ROOT/python.sh" scripts/run_validation.py --headless --suite full --payload loaded
"$ISAAC_ROOT/python.sh" scripts/build_administration.py --headless --payload loaded --presentation-assumptions
"$ISAAC_ROOT/python.sh" scripts/render_administration.py --headless
```

Generated evidence is written to `results/`. The import report records the exact
URDF hashes, frame paths, mass attributes, drive settings and importer version.
Validation traces include pose, yaw, base velocity, target/actual wheel velocity,
physics rate, seed, payload, and explicit blocked tests.

The rigid six-contact proxy is statically indeterminate in PhysX. With all six
geometries exactly coplanar, one driven wheel can be left effectively unloaded.
The imported deterministic assets therefore apply a symmetric **1 mm simulation
rest offset to both driven-wheel colliders**. This is a solver seating bias only;
it is not a tyre-radius change, spring preload, or evidence of physical contact
load. It is recorded in `config/physics_materials.yaml` and must remain disclosed.

Isaac Sim 5.1's bundled URDF importer 2.4.30 may print transient unresolved-reference
warnings while it authors mass-only fixed-frame links. `scripts/import_urdf.py`
repairs those generated targets before saving and records the repaired paths in
`results/import_report.json`; the saved assets reopen and validate without those
warnings.

`scenes/validation_thresholds.usd` contains parameterized 5/10/20 mm geometry,
but it is deliberately marked as blocked for contact conclusions. Threshold
validation requires the articulated compliant carrier, measured spring curve,
and measured caster properties.

`scripts/build_administration.py` remains a strict input gate by default. The
explicit `--presentation-assumptions` mode now builds `scenes/administration.usd`
as a disclosed route-scoped proxy. It uses the recorded 12.75 m atrium and 2.80 m
hallway dimensions, plus assumed 1.10/1.05 m door clearances and 3/5 mm
thresholds. These are presentation values, not survey measurements; the scene's
metadata and report keep physical release false.

AI-SHA-Production was reviewed at commit `8893535`. It does not contain the A1
page-2 plan, scaled building geometry, or non-zero Principal/Vice-Principal goal
poses. It does confirm the deployed LD19, RealSense D435, and BNO055 contracts,
which are recorded in `config/sensors.yaml`. Its older mecanum chassis footprint
is intentionally not applied to this Rev D differential-drive model. See
`results/PRODUCTION_REPOSITORY_REVIEW.md`.

## Navigation and doorway fit

Use a footprint that contains the outboard wheels and forward head:

```yaml
footprint: "[[-0.455,-0.384],[0.725,-0.384],[0.725,0.384],[-0.455,0.384]]"
```

The head makes the robot 1.180 m long even though the deck is only 0.910 m. The
previous length claim was wrong.

Clear width alone does not release a route. The robot pivots about the
drive-axle midpoint at x=0 with the head 0.725 m ahead, so the furthest
footprint corner sweeps 0.820 m and a turn-in-place needs a **1.640 m clear
circle** — against a 0.768 m transit width. A 0.920 m doorway cannot be pivoted
in. **Plan every rotation in the hallway (2.80 m) or atrium; traverse doorways
on a straight, pre-aligned approach.** Rear corners sweep 0.595 m, so a pivot
also needs that much clearance behind the axle.

A nominal 900 mm door leaf commonly has less
clear width after stops, hinges and hardware; only an on-site clear measurement
can release the route. If clear width is below 0.920 m, change the route or
narrow the mechanical design instead of tuning Nav2 to squeeze through.

Start at 0.30 m/s. The controlled demo target is 0.50 m/s. Treat 0.80 m/s as a
design ceiling only after measured stopping-distance and protective-field tests.

## Perception is not the safety system

`lidar_link` at 1.170 m is for localisation/mapping. It cannot be relied on for
chairs, bins or floor hazards. `front_lidar_link` at 0.250 m feeds the Nav2
obstacle layer, but front-only perception does not protect reversing, side-swipe
or pivot motion.

No sensor model, field of view, diagnostic coverage or safety rating has been
provided. Until a risk assessment and all-direction protective-stop design are
complete, operate only on a closed/access-controlled route with a trained
spotter. Reverse must remain disabled in occupied areas.

The real PoC also needs hardwired emergency stops independent of ROS/Jetson/Pi,
physical bumpers, a battery fuse and service disconnect, contactor, charging
interlock, wheel/pinch guards, command watchdog and measured stop performance.
The driver manual explicitly requires an external emergency-stop circuit;
cutting torque may still allow the robot to coast.
