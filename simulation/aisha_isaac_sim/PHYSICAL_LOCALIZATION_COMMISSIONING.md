# Phase 8A physical localization commissioning

Phase 8A prepares the first real-sensor localization gate for the Rev D AI-SHA.
It is deliberately **stationary and zero-command**. Passing its offline checks
does not authorize motor power, wheel motion, doorway passage, autonomous Nav2,
occupied-building operation, or physical release.

## What is ready

- A Rev D differential-drive AMCL profile uses real time, `/scan`, `map`, `odom`
  and `base_link`.
- A planar `robot_localization` EKF consumes future raw differential encoder
  odometry on `/wheel/odom_raw` plus BNO055 data on `/imu/data`, and publishes
  `/odometry/filtered`. The EKF alone owns `odom -> base_link`; AMCL owns
  `map -> odom`.
- A TF-only localization URDF publishes the Rev D design transforms for
  `lidar_link`, `front_lidar_link` and `imu_link`. These transforms must be
  checked against the built robot before motion.
- The launch file starts only robot-state publication, EKF, map server, AMCL and
  their localization lifecycle manager. It contains no planner, controller,
  velocity smoother, motor node or `/cmd_vel` publisher.
- A read-only runtime observer checks topic rates, message frames, TF ownership,
  scan/IMU sanity, stationary odometry and the absence of `/cmd_vel` publishers.

Run the offline preparation check at any time:

```bash
simulation/aisha_isaac_sim/tools/run_phase8a_physical_localization_preflight.sh
```

## Hard blockers found by the audit

1. The current workstation does not have `nav2_amcl`, `nav2_map_server`,
   `nav2_lifecycle_manager` or `robot_localization` installed. The dependencies
   are now declared by `robot_bringup`; install them through `rosdep` in the
   target ROS 2 workspace.
2. The existing production `mecanum_driver` is not compatible with Rev D. It
   assumes four mecanum wheels, a 0.0695 m radius and holonomic forward/inverse
   kinematics. Rev D has two driven wheels, a 0.100 m design radius, a 0.720 m
   design track and no lateral command. Do not launch the mecanum driver on the
   Rev D robot.
3. A Rev D differential encoder adapter is still required. It must publish raw
   `odom -> base_link` data on `/wheel/odom_raw` without broadcasting TF. Its
   wheel radius and count scale cannot be frozen until the delivered wheels are
   measured under load and one marked revolution resolves 4096 versus 16384
   driver counts.
4. The current administration occupancy map is a plan/RoomPlan/walkthrough-
   informed presentation map, not an as-built map. It may be observed in a
   stationary experiment but cannot authorize physical path planning.
5. The reported 0.85 m doorway is below the Rev D 0.92 m physical minimum and
   is 0.078 m narrower than the 0.928 m production padded transit width. The
   simulation-only 0.030 m padding exception cannot be transferred to hardware.
6. All-direction protective sensing, hardwired emergency stops, bumpers and
   measured stopping distances are not commissioned.
7. The simulation workstation baseline is ROS 2 Jazzy, while the existing
   Pi/Jetson production launch contract says every SBC must run Humble. Freeze
   one target distribution and rebuild/test the entire physical graph on it;
   do not run Nav2 across a Humble/Jazzy split.

## Stationary on-robot procedure

Do this only on the physical robot with an operator and an independent spotter:

1. Keep the motor drive disabled and electrically isolated. Chock the wheels.
   Verify both latching emergency-stop devices before energizing the sensor
   computers. Do not rely on ROS as the emergency stop.
2. Install dependencies and build the two packages:

   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   colcon build --packages-select robot_description robot_bringup
   source install/setup.bash
   ```

3. Start the LD19 as `/scan` with `frame_id:=lidar_link`, and start the BNO055
   source as `/imu/data` with `frame_id:=imu_link`. Start only a verified Rev D
   differential encoder adapter on `/wheel/odom_raw`; do not substitute dummy
   odometry, RF2O, or the mecanum driver.
4. Start the localization-only graph, providing an explicit map path:

   ```bash
   ros2 launch robot_bringup phase8a_localization_preflight.launch.py \
     map:=/absolute/path/to/administration.yaml
   ```

5. Give AMCL one initial pose using RViz. Do not send a navigation goal. Confirm
   that no node publishes `/cmd_vel`.
6. With the robot still chocked and drive-disabled, collect the 20-second gate:

   ```bash
   simulation/aisha_isaac_sim/tools/run_phase8a_physical_localization_preflight.sh \
     --probe-stationary-runtime
   ```

The runtime gate requires at least 8 Hz LD19, 40 Hz BNO055, 20 Hz raw and
filtered odometry, the full `map -> odom -> base_link -> sensor` TF chain,
normalized IMU orientation, stationary velocities below 0.02 m/s and 0.05
rad/s, and zero `/cmd_vel` publishers.

## Go/no-go boundary

A passing stationary report authorizes only progression to a separate
wheels-lifted direction/encoder test. It does not authorize floor motion. The
first floor-motion gate will be tethered, access-controlled, limited to 0.10
m/s in an open test area, and will remain blocked until the differential driver,
encoder scaling, emergency-stop circuit and spotter procedure are verified.
