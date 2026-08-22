#!/usr/bin/env python3
"""Statically validate measured-site and Nav2 preparation without claiming runtime readiness."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SIM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SIM_ROOT.parents[1]
EXPECTED_FOOTPRINT = [
    [0.725, 0.384],
    [0.725, -0.384],
    [-0.455, -0.384],
    [-0.455, 0.384],
]
REQUIRED_NAV2_PACKAGES = (
    "nav2_bringup",
    "nav2_amcl",
    "nav2_controller",
    "nav2_costmap_2d",
    "nav2_dwb_controller",
    "nav2_navfn_planner",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def close(actual: Any, expected: float, tolerance: float = 1.0e-9) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, abs_tol=tolerance
    )


def footprint_matches(value: Any) -> bool:
    if isinstance(value, str):
        value = yaml.safe_load(value)
    if not isinstance(value, list) or len(value) != len(EXPECTED_FOOTPRINT):
        return False
    return all(
        isinstance(point, list)
        and len(point) == 2
        and all(close(actual, expected) for actual, expected in zip(point, reference))
        for point, reference in zip(value, EXPECTED_FOOTPRINT)
    )


def all_nav2_nodes_use_sim_time(nav2: dict[str, Any]) -> bool:
    checked = 0
    for value in nav2.values():
        if not isinstance(value, dict):
            continue
        params = value.get("ros__parameters")
        if params is not None:
            checked += 1
            if params.get("use_sim_time") is not True:
                return False
            continue
        for nested in value.values():
            if isinstance(nested, dict) and isinstance(nested.get("ros__parameters"), dict):
                checked += 1
                if nested["ros__parameters"].get("use_sim_time") is not True:
                    return False
    return checked > 0


def package_presence(ros_root: Path) -> dict[str, bool]:
    share = ros_root / "share"
    return {name: (share / name).is_dir() for name in REQUIRED_NAV2_PACKAGES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=SIM_ROOT / "config" / "ros2_nav2_sim.yaml"
    )
    parser.add_argument(
        "--nav2-params", type=Path, default=SIM_ROOT / "config" / "nav2_sim_params.yaml"
    )
    parser.add_argument(
        "--measured-manifest",
        type=Path,
        default=SIM_ROOT / "config" / "measured_administration_template.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SIM_ROOT / "results" / "measured_site_nav2_preparation_report.json",
    )
    parser.add_argument(
        "--ros-root",
        type=Path,
        help="explicit ROS prefix; otherwise check system Jazzy and the AI-SHA user overlay",
    )
    parser.add_argument("--strict-runtime", action="store_true")
    args = parser.parse_args()

    contract = load_yaml(args.contract)
    nav2 = load_yaml(args.nav2_params)
    manifest = load_yaml(args.measured_manifest)
    sensors = load_yaml(SIM_ROOT / "config" / "sensors.yaml")
    drive = load_yaml(SIM_ROOT / "config" / "aisha_drive.yaml")
    legacy_path = REPO_ROOT / "src" / "robot_bringup" / "config" / "nav2_params.yaml"
    legacy_text = legacy_path.read_text(encoding="utf-8")
    urdf_path = SIM_ROOT / "urdf" / "aisha.urdf"
    urdf_text = urdf_path.read_text(encoding="utf-8")
    live_env_path = (
        SIM_ROOT
        / "isaaclab"
        / "aisha_isaaclab"
        / "tasks"
        / "office_nav"
        / "administration_live_env.py"
    )
    live_env_text = live_env_path.read_text(encoding="utf-8")

    controller = nav2["controller_server"]["ros__parameters"]["FollowPath"]
    velocity = nav2["velocity_smoother"]["ros__parameters"]
    local = nav2["local_costmap"]["local_costmap"]["ros__parameters"]
    global_costmap = nav2["global_costmap"]["global_costmap"]["ros__parameters"]
    sensor_frames = sensors["frames"]

    contract_checks = {
        "simulation_scope_explicit": contract.get("scope") == "isaac_sim_administration_only",
        "simulation_time_enabled": contract.get("use_sim_time") is True,
        "differential_drive": contract["drive"].get("architecture") == "differential_drive",
        "wheel_radius_matches_rev_d": close(contract["drive"].get("wheel_radius_m"), 0.100),
        "wheel_track_matches_rev_d": close(contract["drive"].get("wheel_track_m"), 0.720),
        "reverse_disabled": close(contract["drive"].get("maximum_reverse_velocity_mps"), 0.0),
        "lateral_motion_disabled": close(
            contract["drive"].get("maximum_lateral_velocity_mps"), 0.0
        ),
        "physical_footprint_matches_rev_d": footprint_matches(
            contract["footprint"].get("physical_xy_m")
        ),
        "footprint_padding_is_80_mm": close(
            contract["footprint"].get("costmap_padding_m"), 0.08
        ),
        "padded_width_is_0_928_m": close(
            contract["footprint"].get("padded_transit_width_m"), 0.928
        ),
        "physical_release_false": contract["integration_boundary"].get("physical_release")
        is False,
        "learned_policy_coupling_not_overclaimed": contract["integration_boundary"].get(
            "learned_policy_coupled_to_nav2"
        )
        is False,
    }
    nav2_checks = {
        "all_nodes_use_sim_time": all_nav2_nodes_use_sim_time(nav2),
        "amcl_uses_differential_model": nav2["amcl"]["ros__parameters"].get(
            "robot_model_type"
        )
        == "nav2_amcl::DifferentialMotionModel",
        "controller_disables_reverse": close(controller.get("min_vel_x"), 0.0),
        "controller_disables_lateral_motion": close(controller.get("min_vel_y"), 0.0)
        and close(controller.get("max_vel_y"), 0.0)
        and int(controller.get("vy_samples", -1)) == 1,
        "controller_limits_forward_speed": close(controller.get("max_vel_x"), 0.30),
        "controller_limits_angular_speed": close(controller.get("max_vel_theta"), 0.55),
        "velocity_smoother_disables_reverse": velocity.get("min_velocity") == [0.0, 0.0, -0.55],
        "velocity_smoother_disables_lateral_motion": velocity.get("max_velocity", [None, None])[1]
        == 0.0,
        "backup_behavior_not_loaded": "backup"
        not in nav2["behavior_server"]["ros__parameters"].get("behavior_plugins", []),
        "local_footprint_matches": footprint_matches(local.get("footprint")),
        "global_footprint_matches": footprint_matches(global_costmap.get("footprint")),
        "costmap_padding_matches_contract": close(local.get("footprint_padding"), 0.08)
        and close(global_costmap.get("footprint_padding"), 0.08),
        "local_costmap_consumes_both_scans": local["obstacle_layer"].get(
            "observation_sources"
        )
        == "crown_scan front_scan"
        and local["obstacle_layer"]["crown_scan"].get("topic") == "/scan"
        and local["obstacle_layer"]["front_scan"].get("topic") == "/front_scan",
        "global_costmap_consumes_both_scans": global_costmap["obstacle_layer"].get(
            "observation_sources"
        )
        == "crown_scan front_scan"
        and global_costmap["obstacle_layer"]["crown_scan"].get("topic") == "/scan"
        and global_costmap["obstacle_layer"]["front_scan"].get("topic") == "/front_scan",
    }
    asset_checks = {
        "drive_source_is_rev_d_differential": drive["model"].get("revision") == "D"
        and drive["model"].get("architecture") == "differential_drive",
        "crown_topic_matches_sensor_contract": sensor_frames["crown_lidar"].get("ros_topic")
        == contract["topics"].get("crown_scan"),
        "front_topic_matches_sensor_contract": sensor_frames["front_lidar"].get("ros_topic")
        == contract["topics"].get("front_scan"),
        "sensor_static_transform_positions_declared": sensor_frames["crown_lidar"].get(
            "position_m"
        )
        == [0.500, 0.0, 1.170]
        and sensor_frames["front_lidar"].get("position_m") == [0.455, 0.0, 0.250]
        and sensor_frames["imu"].get("position_m") == [-0.120, 0.120, 0.230],
        "required_sensor_links_in_urdf": all(
            f'name="{name}"' in urdf_text
            for name in ("lidar_link", "front_lidar_link", "imu_link")
        ),
        "front_lidar_exists_in_live_isaac_scene": "front_lidar = MultiMeshRayCasterCfg("
        in live_env_text,
        "front_lidar_does_not_change_policy_observation": "self._front_lidar"
        not in live_env_text,
        "legacy_mecanum_profile_detected_and_separated": "AI-SHA Mecanum Chassis"
        in legacy_text
        and "max_vel_y: 0.3" in legacy_text
        and args.nav2_params.resolve() != legacy_path.resolve(),
    }
    preparation_checks = {**contract_checks, **nav2_checks, **asset_checks}
    preparation_passed = all(preparation_checks.values())

    overlay_root = Path(
        os.environ.get(
            "AI_SHA_ROS_OVERLAY_ROOT",
            str(Path.home() / ".local/share/ai_sha_ros_jazzy_overlay/root"),
        )
    ) / "opt/ros/jazzy"
    candidate_roots = (
        [args.ros_root]
        if args.ros_root is not None
        else [Path("/opt/ros/jazzy"), overlay_root]
    )
    package_candidates = [
        (root, package_presence(root)) for root in candidate_roots
    ]
    selected_ros_root, packages = next(
        (
            (root, presence)
            for root, presence in package_candidates
            if all(presence.values())
        ),
        package_candidates[0],
    )
    measured_capture_complete = manifest.get("status") not in {None, "awaiting_capture"}
    measured_map_configured = bool(nav2["map_server"]["ros__parameters"].get("yaml_filename"))
    bridge_extension = Path(
        os.environ.get("ISAACSIM_ROOT", "/home/robot-wst/isaacsim")
    ) / "exts" / "isaacsim.ros2.bridge"
    runtime_gates = {
        "nav2_packages_installed": all(packages.values()),
        "isaac_ros2_bridge_extension_present": bridge_extension.is_dir(),
        "measured_capture_complete": measured_capture_complete,
        "measured_occupancy_map_configured": measured_map_configured,
    }
    runtime_ready = preparation_passed and all(runtime_gates.values())
    blockers = []
    if not all(packages.values()):
        blockers.append(
            "Nav2 Jazzy packages were not found in any checked ROS prefix: "
            + ", ".join(str(root) for root in candidate_roots)
        )
    if not measured_capture_complete:
        blockers.append("The measured iPhone LiDAR/RoomPlan capture has not been supplied yet.")
    if not measured_map_configured:
        blockers.append("A measured-site occupancy-map YAML has not been generated or configured yet.")
    if not bridge_extension.is_dir():
        blockers.append("Isaac Sim's isaacsim.ros2.bridge extension was not found.")

    report = {
        "report_type": "measured_site_nav2_preparation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "runtime_ready" if runtime_ready else "preparation_ready_runtime_gated",
        "passed": preparation_passed,
        "runtime_ready": runtime_ready,
        "contract": str(args.contract.resolve()),
        "nav2_params": str(args.nav2_params.resolve()),
        "measured_manifest": str(args.measured_manifest.resolve()),
        "legacy_nav2_profile": {
            "path": str(legacy_path.resolve()),
            "architecture": "mecanum",
            "applicable_to_rev_d_isaac_sim": False,
        },
        "preparation_checks": preparation_checks,
        "preparation_checks_passed": sum(preparation_checks.values()),
        "preparation_checks_total": len(preparation_checks),
        "runtime_gates": runtime_gates,
        "nav2_runtime_root": str(selected_ros_root.resolve()),
        "nav2_package_presence": packages,
        "blockers": blockers,
        "claim_boundary": (
            "Passing this report proves only that the measured-site intake and differential-drive "
            "Nav2 configuration contracts are internally consistent. It does not prove a live "
            "Nav2 run, policy/Nav2 arbitration, physical stopping distance, protective coverage, "
            "sim-to-real transfer, or safe deployment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict_runtime:
        return 0 if runtime_ready else 2
    return 0 if preparation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
