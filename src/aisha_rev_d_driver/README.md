# AI-SHA Rev D differential adapter

This ROS 2 package replaces the legacy four-wheel mecanum assumption for the
Rev D two-wheel differential chassis. Phase 8B supports deterministic replay
and ZLAC8015D **read-only** RS485 telemetry. It publishes encoder telemetry and,
only under the appropriate scale gate, raw wheel odometry without broadcasting
TF. The EKF remains the sole owner of `odom -> base_link`.

The default replay is safe to run on a workstation:

```bash
ros2 launch aisha_rev_d_driver phase8b_replay.launch.py
```

It publishes replay data on isolated `/phase8b/replay/*` topics. The source-tree
RS485 profile keeps all physical confirmations false and `publish_odom: false`.
The hardware transport permits only Modbus function `0x03`; it cannot set the
control mode, enable the driver, stop it, or write target velocities.

Do not switch to the physical profile until the exact delivered driver label is
matched to its manual, motor leads are isolated, the hardwired emergency stop
is verified, and the operator checklist in
`simulation/aisha_isaac_sim/PHYSICAL_LOCALIZATION_COMMISSIONING.md` is complete.
The 16384-count and 0.100 m values are replay/design candidates, not calibrated
physical odometry.
