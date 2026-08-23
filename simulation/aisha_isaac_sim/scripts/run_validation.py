#!/usr/bin/env python3
"""Run reproducible AI-SHA import, drop, motion, and watchdog validation.

The default smoke suite is an implementation check, not the handoff's final
acceptance campaign. Use --suite full for 5 m straight runs at both accepted
controlled speeds. The isolated --suite high_speed gate measures the proposed
0.80 m/s simulation tier and its controlled stop without changing the accepted
0.30/0.50 m/s reports. It is not a physical stopping-distance test.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--suite", choices=("smoke", "full", "high_speed"), default="smoke"
    )
    parser.add_argument("--payload", choices=("empty", "loaded"), default="loaded")
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

import omni.usd
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from isaacsim.core.version import get_version
from pxr import Usd, UsdPhysics

from aisha_common import (
    CONFIG_DIR,
    RESULTS_DIR,
    SCENES_DIR,
    URDF_DIR,
    USD_DIR,
    DifferentialDriveLimiter,
    ensure_output_dirs,
    load_yaml,
    sha256_file,
    write_json,
)


REQUIRED_FRAMES = (
    "lidar_link",
    "front_lidar_link",
    "front_camera_link",
    "front_camera_optical_frame",
    "imu_link",
    "cargo_payload_frame",
)
DRIVEN_JOINTS = ("left_wheel_joint", "right_wheel_joint")


def progress(message: str) -> None:
    os.write(2, f"[AI-SHA] {message}\n".encode("utf-8"))


def euler_from_wxyz(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def asset_checks(path: str, expected_mass_kg: float) -> dict[str, object]:
    stage = Usd.Stage.Open(path)
    if stage is None:
        return {"passed": False, "errors": [f"could not open {path}"]}
    names: dict[str, list[str]] = {name: [] for name in REQUIRED_FRAMES + DRIVEN_JOINTS}
    mass_values = []
    drive_values: dict[str, dict[str, float | None]] = {}
    articulation_roots = []
    for prim in stage.TraverseAll():
        if prim.GetName() in names:
            names[prim.GetName()].append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        mass_attr = prim.GetAttribute("physics:mass")
        mass = mass_attr.Get() if mass_attr else None
        if mass is not None:
            mass_values.append(float(mass))
        if prim.GetName() in DRIVEN_JOINTS:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            drive_values[prim.GetName()] = {
                "stiffness": drive.GetStiffnessAttr().Get(),
                "damping": drive.GetDampingAttr().Get(),
                "max_force_nm": drive.GetMaxForceAttr().Get(),
                "target_velocity_deg_s": drive.GetTargetVelocityAttr().Get(),
            }
    mass_sum = sum(mass_values)
    errors = []
    for name in REQUIRED_FRAMES:
        if not names[name]:
            errors.append(f"missing frame {name}")
    for name in DRIVEN_JOINTS:
        if len(names[name]) != 1:
            errors.append(f"expected one {name}, found {len(names[name])}")
        values = drive_values.get(name, {})
        if values.get("stiffness") != 0.0:
            errors.append(f"{name} stiffness is not zero")
        if values.get("damping") != 120.0:
            errors.append(f"{name} damping is not 120")
        if values.get("max_force_nm") != 6.0:
            errors.append(f"{name} max force is not rated 6 N.m")
    if abs(mass_sum - expected_mass_kg) > 0.001:
        errors.append(f"mass attribute sum {mass_sum:.6f} differs from {expected_mass_kg:.6f} kg")
    if len(articulation_roots) != 1:
        errors.append(f"expected one articulation root, found {len(articulation_roots)}")
    return {
        "passed": not errors,
        "errors": errors,
        "mass_attribute_sum_kg": round(mass_sum, 6),
        "expected_design_mass_kg": expected_mass_kg,
        "articulation_roots": articulation_roots,
        "frame_prims": {name: names[name] for name in REQUIRED_FRAMES},
        "drives": drive_values,
    }


def locate_articulation_root(stage: Usd.Stage) -> str:
    roots = [str(prim.GetPath()) for prim in stage.TraverseAll() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"expected one articulation root in validation scene, found {roots}")
    return roots[0]


def pose(articulation: Articulation) -> tuple[np.ndarray, np.ndarray]:
    positions, orientations = articulation.get_world_poses(usd=False)
    return np.asarray(positions[0], dtype=float), np.asarray(orientations[0], dtype=float)


def reset_articulation(
    articulation: Articulation,
    simulation: SimulationContext,
    z_m: float,
    wheel_indices: list[int],
    settle_steps: int = 12,
) -> None:
    progress(f"setting articulation pose to z={z_m:.4f}")
    articulation.set_world_poses(
        positions=np.asarray([[0.0, 0.0, z_m]], dtype=np.float32),
        orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    progress("zeroing base linear velocity")
    articulation.set_linear_velocities(np.zeros((1, 3), dtype=np.float32))
    progress("zeroing base angular velocity")
    articulation.set_angular_velocities(np.zeros((1, 3), dtype=np.float32))
    progress("zeroing wheel targets")
    articulation.set_joint_velocity_targets(
        np.zeros((1, 2), dtype=np.float32),
        joint_indices=wheel_indices,
    )
    progress("stepping after reset")
    for _ in range(settle_steps):
        simulation.step(render=False)


def run_motion_command(
    articulation: Articulation,
    simulation: SimulationContext,
    *,
    linear_mps: float,
    angular_rad_s: float,
    duration_s: float,
    physics_hz: int,
    drive: dict[str, object],
    wheel_indices: list[int],
) -> dict[str, object]:
    dt = 1.0 / physics_hz
    limiter = DifferentialDriveLimiter(
        wheel_radius_m=float(drive["geometry"]["wheel_radius_nominal_m"]),
        wheel_track_m=float(drive["geometry"]["wheel_track_design_m"]),
        max_linear_mps=max(
            float(drive["navigation"]["controlled_demo_speed_mps"]),
            abs(linear_mps),
        ),
        max_angular_rad_s=1.0,
        max_acceleration_mps2=max(float(value) for value in drive["navigation"]["acceleration_target_mps2"]),
        max_angular_acceleration_rad_s2=1.0,
        watchdog_timeout_s=0.25,
    )
    limiter.reset(0.0)
    start_position, start_quaternion = pose(articulation)
    start_yaw = euler_from_wxyz(start_quaternion)[2]
    samples = []
    steps = round(duration_s * physics_hz)
    for index in range(steps):
        now_s = index * dt
        limiter.command(linear_mps, angular_rad_s, now_s)
        left, right = limiter.update(now_s, dt)
        articulation.set_joint_velocity_targets(
            np.asarray([[left, right]], dtype=np.float32),
            joint_indices=wheel_indices,
        )
        simulation.step(render=False)
        if index % max(1, physics_hz // 10) == 0 or index == steps - 1:
            position, quaternion = pose(articulation)
            velocity = np.asarray(articulation.get_linear_velocities()[0], dtype=float)
            wheel_velocity = np.asarray(
                articulation.get_joint_velocities(joint_indices=wheel_indices)[0],
                dtype=float,
            )
            samples.append(
                {
                    "time_s": round((index + 1) * dt, 6),
                    "position_m": position.tolist(),
                    "yaw_rad": euler_from_wxyz(quaternion)[2],
                    "linear_velocity_mps": velocity.tolist(),
                    "wheel_target_rad_s": [left, right],
                    "wheel_actual_rad_s": wheel_velocity.tolist(),
                }
            )
    articulation.set_joint_velocity_targets(np.zeros((1, 2), dtype=np.float32), joint_indices=wheel_indices)
    end_position, end_quaternion = pose(articulation)
    end_yaw = euler_from_wxyz(end_quaternion)[2]
    displacement = end_position - start_position
    return {
        "command": {"linear_mps": linear_mps, "angular_rad_s": angular_rad_s, "duration_s": duration_s},
        "start_position_m": start_position.tolist(),
        "end_position_m": end_position.tolist(),
        "displacement_m": displacement.tolist(),
        "distance_xy_m": float(np.linalg.norm(displacement[:2])),
        "yaw_change_deg": math.degrees(angle_delta(end_yaw, start_yaw)),
        "samples": samples,
    }


def run_high_speed_stop_command(
    articulation: Articulation,
    simulation: SimulationContext,
    *,
    target_speed_mps: float,
    physics_hz: int,
    drive: dict[str, object],
    wheel_indices: list[int],
) -> dict[str, object]:
    """Measure a software-limited stop from the proposed hallway speed tier."""
    dt = 1.0 / physics_hz
    deceleration_mps2 = max(
        float(value) for value in drive["navigation"]["acceleration_target_mps2"]
    )
    limiter = DifferentialDriveLimiter(
        wheel_radius_m=float(drive["geometry"]["wheel_radius_nominal_m"]),
        wheel_track_m=float(drive["geometry"]["wheel_track_design_m"]),
        max_linear_mps=target_speed_mps,
        max_angular_rad_s=1.0,
        max_acceleration_mps2=deceleration_mps2,
        max_angular_acceleration_rad_s2=1.0,
        watchdog_timeout_s=0.25,
    )
    limiter.reset(0.0)
    start_position, start_quaternion = pose(articulation)
    start_yaw = euler_from_wxyz(start_quaternion)[2]
    samples: list[dict[str, object]] = []
    peak_speed_mps = 0.0
    acceleration_duration_s = target_speed_mps / deceleration_mps2 + 2.0
    for index in range(round(acceleration_duration_s * physics_hz)):
        now_s = index * dt
        limiter.command(target_speed_mps, 0.0, now_s)
        left, right = limiter.update(now_s, dt)
        articulation.set_joint_velocity_targets(
            np.asarray([[left, right]], dtype=np.float32),
            joint_indices=wheel_indices,
        )
        simulation.step(render=False)
        velocity = np.asarray(articulation.get_linear_velocities()[0], dtype=float)
        peak_speed_mps = max(peak_speed_mps, float(np.linalg.norm(velocity[:2])))

    brake_position, _ = pose(articulation)
    stop_threshold_mps = 0.05
    stopped_after_s = None
    stop_position = brake_position.copy()
    stop_timeout_s = 4.0
    for index in range(round(stop_timeout_s * physics_hz)):
        now_s = acceleration_duration_s + index * dt
        limiter.command(0.0, 0.0, now_s)
        left, right = limiter.update(now_s, dt)
        articulation.set_joint_velocity_targets(
            np.asarray([[left, right]], dtype=np.float32),
            joint_indices=wheel_indices,
        )
        simulation.step(render=False)
        velocity = np.asarray(articulation.get_linear_velocities()[0], dtype=float)
        speed_mps = float(np.linalg.norm(velocity[:2]))
        position, quaternion = pose(articulation)
        if index % max(1, physics_hz // 20) == 0 or speed_mps <= stop_threshold_mps:
            samples.append(
                {
                    "time_after_stop_request_s": round((index + 1) * dt, 6),
                    "speed_mps": speed_mps,
                    "position_m": position.tolist(),
                    "wheel_target_rad_s": [left, right],
                }
            )
        if speed_mps <= stop_threshold_mps:
            stopped_after_s = (index + 1) * dt
            stop_position = position
            break

    articulation.set_joint_velocity_targets(
        np.zeros((1, 2), dtype=np.float32), joint_indices=wheel_indices
    )
    _, end_quaternion = pose(articulation)
    stopping_distance_m = float(np.linalg.norm((stop_position - brake_position)[:2]))
    total_displacement_m = float(np.linalg.norm((stop_position - start_position)[:2]))
    yaw_drift_deg = math.degrees(
        angle_delta(euler_from_wxyz(end_quaternion)[2], start_yaw)
    )
    reached_target = peak_speed_mps >= target_speed_mps * 0.94
    passed = (
        reached_target
        and stopped_after_s is not None
        and stopped_after_s <= 2.25
        and stopping_distance_m <= 1.00
        and abs(yaw_drift_deg) <= 2.0
    )
    return {
        "commanded_speed_mps": target_speed_mps,
        "peak_measured_speed_mps": peak_speed_mps,
        "target_reached_within_6_percent": reached_target,
        "controlled_deceleration_mps2": deceleration_mps2,
        "stop_threshold_mps": stop_threshold_mps,
        "stopped_after_s": stopped_after_s,
        "stopping_distance_m": stopping_distance_m,
        "maximum_stopping_time_s": 2.25,
        "maximum_stopping_distance_m": 1.00,
        "yaw_drift_deg": yaw_drift_deg,
        "total_displacement_m": total_displacement_m,
        "samples": samples,
        "passed": passed,
        "claim_boundary": (
            "software-limited flat-floor Isaac Sim stop only; not an emergency stop, "
            "protective-field validation, or physical stopping-distance result"
        ),
    }


def physics_checks(scene_path: str, physics: dict[str, object], drive: dict[str, object]) -> dict[str, object]:
    progress(f"opening validation scene {scene_path}")
    if not omni.usd.get_context().open_stage(scene_path):
        raise RuntimeError(f"could not open {scene_path}")
    for _ in range(3):
        APP.update()
    stage = omni.usd.get_context().get_stage()
    root_path = locate_articulation_root(stage)
    progress(f"found articulation root {root_path}")
    hz = int(physics["physics"]["physics_hz"])
    simulation = SimulationContext(
        physics_dt=1.0 / hz,
        rendering_dt=1.0 / int(physics["physics"]["render_hz"]),
        stage_units_in_meters=1.0,
    )
    simulation.initialize_physics()
    progress("initialized PhysX")
    articulation = Articulation(root_path)
    articulation.initialize()
    progress("initialized articulation view")
    if not articulation.is_physics_handle_valid():
        raise RuntimeError(f"invalid articulation physics handle at {root_path}")
    wheel_indices = [articulation.get_dof_index(name) for name in DRIVEN_JOINTS]
    progress(f"resolved wheel DOF indices {wheel_indices}")
    simulation.play()
    progress("started simulation")

    reset_articulation(articulation, simulation, 0.15, wheel_indices, settle_steps=1)
    progress("reset for drop test")
    drop_start, _ = pose(articulation)
    drop_samples = []
    for index in range(3 * hz):
        simulation.step(render=False)
        if index % (hz // 10) == 0 or index == 3 * hz - 1:
            position, quaternion = pose(articulation)
            roll, pitch, yaw = euler_from_wxyz(quaternion)
            drop_samples.append(
                {
                    "time_s": round((index + 1) / hz, 6),
                    "position_m": position.tolist(),
                    "rpy_deg": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
                }
            )
    drop_end, drop_quaternion = pose(articulation)
    drop_rpy = euler_from_wxyz(drop_quaternion)
    drop_finite = bool(np.isfinite(drop_end).all() and np.isfinite(drop_quaternion).all())
    drop_passed = drop_finite and abs(math.degrees(drop_rpy[0])) < 2.0 and abs(math.degrees(drop_rpy[1])) < 2.0
    progress(f"drop test complete (passed={drop_passed})")

    tests = []
    if ARGS.suite == "smoke":
        speeds = (0.30,)
    elif ARGS.suite == "full":
        speeds = (0.30, 0.50)
    else:
        speeds = (0.80,)
    for speed in speeds:
        reset_articulation(articulation, simulation, float(drop_end[2]), wheel_indices, settle_steps=2 * hz)
        duration = 3.0 if ARGS.suite == "smoke" else 5.0 / speed + 1.5
        result = run_motion_command(
            articulation,
            simulation,
            linear_mps=speed,
            angular_rad_s=0.0,
            duration_s=duration,
            physics_hz=hz,
            drive=drive,
            wheel_indices=wheel_indices,
        )
        expected = speed * max(0.0, duration - speed / 0.5 / 2.0)
        steady_samples = [
            sample
            for sample in result["samples"]
            if sample["time_s"] >= duration * 0.5
        ]
        steady_speeds = [
            math.hypot(*sample["linear_velocity_mps"][:2])
            for sample in steady_samples
        ]
        steady_speed = float(np.median(steady_speeds)) if steady_speeds else 0.0
        speed_error_pct = abs(steady_speed - speed) / speed * 100.0
        result["expected_distance_approx_m"] = expected
        result["steady_speed_mps"] = steady_speed
        result["steady_speed_error_pct"] = speed_error_pct
        result["passed"] = (
            result["distance_xy_m"] > (0.45 if ARGS.suite == "smoke" else 5.0)
            and abs(result["yaw_change_deg"]) < 2.0
            and (ARGS.suite == "smoke" or speed_error_pct <= 5.0)
        )
        tests.append(result)
        progress(f"straight {speed:.2f} m/s complete (passed={result['passed']})")

    high_speed_stop = None
    if ARGS.suite == "high_speed":
        reset_articulation(
            articulation,
            simulation,
            float(drop_end[2]),
            wheel_indices,
            settle_steps=2 * hz,
        )
        high_speed_stop = run_high_speed_stop_command(
            articulation,
            simulation,
            target_speed_mps=0.80,
            physics_hz=hz,
            drive=drive,
            wheel_indices=wheel_indices,
        )
        progress(
            "high-speed controlled stop complete "
            f"(passed={high_speed_stop['passed']}, "
            f"distance={high_speed_stop['stopping_distance_m']:.3f} m)"
        )

    reset_articulation(articulation, simulation, float(drop_end[2]), wheel_indices, settle_steps=2 * hz)
    pivot = run_motion_command(
        articulation,
        simulation,
        linear_mps=0.0,
        angular_rad_s=0.5,
        duration_s=3.0,
        physics_hz=hz,
        drive=drive,
        wheel_indices=wheel_indices,
    )
    pivot["passed"] = abs(pivot["yaw_change_deg"]) > 30.0 and pivot["distance_xy_m"] < 0.25
    progress(f"pivot test complete (passed={pivot['passed']})")
    pivot_reverse = None
    if ARGS.suite == "full":
        reset_articulation(articulation, simulation, float(drop_end[2]), wheel_indices, settle_steps=2 * hz)
        pivot_reverse = run_motion_command(
            articulation,
            simulation,
            linear_mps=0.0,
            angular_rad_s=-0.5,
            duration_s=3.0,
            physics_hz=hz,
            drive=drive,
            wheel_indices=wheel_indices,
        )
        pivot_reverse["passed"] = (
            abs(pivot_reverse["yaw_change_deg"]) > 30.0
            and pivot_reverse["distance_xy_m"] < 0.25
        )
        progress(f"reverse pivot test complete (passed={pivot_reverse['passed']})")
    simulation.stop()
    return {
        "articulation_root": root_path,
        "physics_hz": hz,
        "drop": {
            "passed": drop_passed,
            "finite": drop_finite,
            "start_position_m": drop_start.tolist(),
            "end_position_m": drop_end.tolist(),
            "end_rpy_deg": [math.degrees(value) for value in drop_rpy],
            "samples": drop_samples,
        },
        "straight": tests,
        "high_speed_stop": high_speed_stop,
        "pivot": pivot,
        "pivot_reverse": pivot_reverse,
    }


def watchdog_check(drive: dict[str, object]) -> dict[str, object]:
    limiter = DifferentialDriveLimiter(
        wheel_radius_m=float(drive["geometry"]["wheel_radius_nominal_m"]),
        wheel_track_m=float(drive["geometry"]["wheel_track_design_m"]),
        max_linear_mps=0.5,
        max_angular_rad_s=1.0,
        max_acceleration_mps2=0.5,
        max_angular_acceleration_rad_s2=1.0,
        watchdog_timeout_s=0.25,
    )
    limiter.reset(0.0)
    limiter.command(0.3, 0.0, 0.0)
    active = limiter.update(0.1, 0.1)
    stopped = limiter.update(0.36, 0.1)
    ignored_before_reset = None
    limiter.command(0.3, 0.0, 0.37)
    ignored_before_reset = limiter.update(0.38, 0.01)
    limiter.reset(0.40)
    limiter.command(0.3, 0.0, 0.41)
    resumed = limiter.update(0.42, 0.01)
    passed = active != (0.0, 0.0) and stopped == (0.0, 0.0) and ignored_before_reset == (0.0, 0.0) and resumed != (0.0, 0.0)
    return {
        "passed": passed,
        "active_wheel_rad_s": active,
        "stale_wheel_rad_s": stopped,
        "latched_command_wheel_rad_s": ignored_before_reset,
        "after_reset_wheel_rad_s": resumed,
        "explicit_reset_required": True,
    }


def main() -> int:
    progress("validation main started")
    ensure_output_dirs()
    drive = load_yaml(CONFIG_DIR / "aisha_drive.yaml")
    physics = load_yaml(CONFIG_DIR / "physics_materials.yaml")
    expected_mass = 59.25 if ARGS.payload == "empty" else 69.25
    asset = USD_DIR / ("aisha_empty.usd" if ARGS.payload == "empty" else "aisha_loaded.usd")
    urdf = URDF_DIR / ("aisha.urdf" if ARGS.payload == "empty" else "aisha_max_payload.urdf")
    scene = SCENES_DIR / "validation_flat.usd"
    for required in (asset, urdf, scene):
        if not required.exists():
            raise FileNotFoundError(required)
    progress("running static asset checks")
    static_asset = asset_checks(str(asset), expected_mass)
    progress(f"static asset checks complete (passed={static_asset['passed']})")
    dynamic_physics = physics_checks(str(scene), physics, drive)
    progress("physics checks complete")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite": ARGS.suite,
        "payload": ARGS.payload,
        "isaac_sim_version": get_version()[0],
        "seed": physics["physics"]["seed"],
        "urdf_sha256": sha256_file(urdf),
        "asset": static_asset,
        "physics": dynamic_physics,
        "watchdog": watchdog_check(drive),
        "blocked": {
            "doorway": "both clear widths and thresholds are unmeasured",
            "administration_route_physical_release": "approved A1 plan page 2 and documented goal poses are absent; presentation proxy uses disclosed assumptions",
            "high_fidelity_contact": "measured spring curve, caster trail/inertia, and articulated asset are absent",
        },
    }
    passes = [report["asset"]["passed"], report["physics"]["drop"]["passed"], report["watchdog"]["passed"]]
    passes.extend(test["passed"] for test in report["physics"]["straight"])
    if report["physics"]["high_speed_stop"] is not None:
        passes.append(report["physics"]["high_speed_stop"]["passed"])
    passes.append(report["physics"]["pivot"]["passed"])
    if report["physics"]["pivot_reverse"] is not None:
        passes.append(report["physics"]["pivot_reverse"]["passed"])
    report["passed"] = all(passes)
    output = RESULTS_DIR / f"validation_{ARGS.suite}_{ARGS.payload}.json"
    write_json(output, report)
    progress(f"wrote {output}")
    print(f"validation {'passed' if report['passed'] else 'failed'}; wrote {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
