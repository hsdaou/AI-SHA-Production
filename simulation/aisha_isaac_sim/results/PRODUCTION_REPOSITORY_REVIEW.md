# AI-SHA-Production reconciliation

Reviewed 2026-08-20 against `https://github.com/hsdaou/AI-SHA-Production` at
commit `8893535b4043ff766d914e8bfe54a789cf3deba0` (`master`). The default branch,
all remote branches, repository issues, pull requests, releases, and recursive
file trees were checked.

## Plan-source separation

The approved Block A ground-floor plan, page 2, is **not present in the production
repository** and the repository is not used to confirm it. The supplied external
`DownloadBuildingRequestApprovedPlan.pdf` is the geometry source; its reviewed
SHA-256 is recorded in `config/administration_assumptions.yaml`. The repository's
`campus-map.md` describes campus zones in prose and its Principal/Vice-Principal
locations remain zero-valued placeholders.

`administration.usd` now derives the 12.75 m atrium, 2.80 m hallway, room
adjacency and office orientation from the supplied approved plan. The following
route-scoped values remain replaceable presentation assumptions:

- Vice-Principal clear door width: 1.10 m; threshold: 3 mm.
- Principal clear door width: 1.05 m; threshold: 5 mm.
- Ceiling/wall height: 3.00 m.
- Door swings, final goal offsets, ceiling/wall height, furniture and decorative detail: presentation assumptions.

These values pass the 0.920 m presentation width gate but are not measurements,
do not validate threshold contact, and do not release physical or unsupervised
operation.

## Sensor and system information adopted

- LDLiDAR LD19: `/scan`, clockwise, `/dev/ttyUSB0`, 230400 baud. The repository's
  Isaac-style profile specifies 360 degrees, 0.02-12 m hardware range, 10 Hz
  scan rate, 4500 reports/s, 1 mm resolution, and 30 mm accuracy. Its Nav2
  configuration uses 0.12-10 m and marks obstacles to 8 m.
- Intel RealSense D435: vendor launch with aligned depth enabled and point cloud
  disabled. Vision consumes color and aligned-depth topics under
  `/camera/camera/...`; YOLO's configured target is 30 FPS.
- Bosch BNO055: 9-DOF fused IMU, transported through the Arduino Mega unified
  ODOM packet and published as `/imu/data` in `imu_link` at 50 Hz. EKF uses
  orientation and angular velocity, not linear acceleration.
- General topology: Jetson Orin Nano runs RealSense/vision/cognitive workloads;
  Raspberry Pi 5 runs LD19, motor/odometry/IMU bridge, SLAM/Nav2, TTS and display;
  ROS 2 Humble uses FastDDS static-unicast discovery.

## Architecture conflict deliberately not imported

The production repository's checked-in Nav2 footprint and motor configuration
describe an earlier 0.60 x 0.50 m mecanum chassis. The Isaac bundle models the
audited Rev D differential-drive robot with a 1.180 x 0.768 m envelope. The
mecanum footprint, lateral velocity settings, and placeholder goal coordinates
were not copied into the Isaac configuration.
