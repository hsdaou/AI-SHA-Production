#!/usr/bin/env python3
"""Validate Phase 8A offline preparation without granting physical motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "config/phase8a_physical_localization_preflight.yaml",
    )
    parser.add_argument(
        "--runtime-probe",
        type=Path,
        default=ROOT / "results/phase8a_stationary_localization_probe.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase8a_physical_localization_preflight.json",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML object: {path}")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_available(name: str) -> bool:
    return (Path("/opt/ros/jazzy/share") / name).is_dir()


def joint_origin(root: ET.Element, child_name: str) -> list[float] | None:
    for joint in root.findall("joint"):
        child = joint.find("child")
        origin = joint.find("origin")
        if child is not None and child.attrib.get("link") == child_name and origin is not None:
            return [float(value) for value in origin.attrib["xyz"].split()]
    return None


def main() -> int:
    args = parse_args()
    profile = load_yaml(args.profile)
    drive_path = ROOT / "config/aisha_drive.yaml"
    drive = load_yaml(drive_path)
    amcl_path = REPO / profile["artifacts"]["amcl_config"]
    ekf_path = REPO / profile["artifacts"]["ekf_config"]
    urdf_path = REPO / profile["artifacts"]["localization_urdf"]
    launch_path = REPO / profile["artifacts"]["launch"]
    probe_path = REPO / profile["artifacts"]["stationary_probe"]
    package_xml_path = REPO / "src/robot_bringup/package.xml"
    legacy_nav_path = REPO / "src/robot_bringup/config/nav2_params.yaml"
    legacy_driver_path = REPO / "src/mecanum_driver/mecanum_driver/mecanum_driver_node.py"
    rpi_launch_path = REPO / "src/aisha_integration/launch/rpi_launch.py"

    amcl = load_yaml(amcl_path)
    ekf = load_yaml(ekf_path)
    launch_source = launch_path.read_text(encoding="utf-8")
    probe_source = probe_path.read_text(encoding="utf-8")
    package_xml = package_xml_path.read_text(encoding="utf-8")
    legacy_nav = legacy_nav_path.read_text(encoding="utf-8")
    legacy_driver = legacy_driver_path.read_text(encoding="utf-8")
    rpi_launch = rpi_launch_path.read_text(encoding="utf-8")
    urdf = ET.parse(urdf_path).getroot()

    robot = profile["robot_contract"]
    localization = profile["localization_contract"]
    authorization = profile["authorization_boundary"]
    map_candidate = profile["map_candidate"]
    doorway = profile["doorway_release_conflict"]
    ros_distribution = profile["ros_distribution_boundary"]
    amcl_params = amcl["amcl"]["ros__parameters"]
    map_params = amcl["map_server"]["ros__parameters"]
    ekf_params = ekf["ekf_filter_node"]["ros__parameters"]
    map_yaml = ROOT / map_candidate["yaml"]
    map_image = ROOT / map_candidate["image"]
    expected_footprint = drive["navigation"]["raw_footprint_xy_m"]

    dependencies = [
        "nav2_amcl",
        "nav2_map_server",
        "nav2_lifecycle_manager",
        "robot_localization",
        "robot_state_publisher",
    ]
    dependency_status = {name: package_available(name) for name in dependencies}
    missing_dependencies = [name for name, present in dependency_status.items() if not present]
    blocker_ids = {item["id"] for item in profile["hard_blockers"]}
    urdf_links = {link.attrib["name"] for link in urdf.findall("link")}
    sequence = profile["commissioning_sequence"]

    checks = {
        "rev_d_differential_contract_matches_design_source": (
            robot["revision"] == drive["model"]["revision"] == "D"
            and robot["architecture"] == drive["model"]["architecture"] == "differential_drive"
            and math.isclose(robot["wheel_radius_design_m"], drive["geometry"]["wheel_radius_nominal_m"])
            and math.isclose(robot["wheel_track_design_m"], drive["geometry"]["wheel_track_design_m"])
        ),
        "physical_footprint_matches_rev_d_source": robot["raw_footprint_xy_m"] == expected_footprint,
        "production_padding_retained": (
            math.isclose(robot["physical_nav2_padding_per_side_m"], 0.080)
            and math.isclose(
                robot["padded_transit_width_m"],
                drive["geometry"]["overall_width_design_m"] + 0.160,
            )
        ),
        "real_time_localization_only": (
            localization["use_sim_time"] is False
            and amcl_params["use_sim_time"] is False
            and map_params["use_sim_time"] is False
            and ekf_params["use_sim_time"] is False
        ),
        "amcl_uses_rev_d_differential_model_and_real_scan": (
            amcl_params["robot_model_type"] == "nav2_amcl::DifferentialMotionModel"
            and amcl_params["scan_topic"] == localization["crown_scan_topic"] == "/scan"
            and amcl_params["base_frame_id"] == localization["base_frame"]
            and amcl_params["odom_frame_id"] == localization["odom_frame"]
            and amcl_params["global_frame_id"] == localization["map_frame"]
        ),
        "ekf_has_single_tf_ownership_and_separate_raw_output_topics": (
            ekf_params["publish_tf"] is True
            and ekf_params["world_frame"] == "odom"
            and ekf_params["odom0"] == localization["raw_wheel_odometry_topic"]
            and ekf_params["imu0"] == localization["imu_topic"]
            and localization["raw_wheel_odometry_topic"]
            != localization["filtered_odometry_topic"]
        ),
        "ekf_is_planar_and_imu_heading_is_relative": (
            ekf_params["two_d_mode"] is True
            and ekf_params["imu0_relative"] is True
            and ekf_params["imu0_remove_gravitational_acceleration"] is True
        ),
        "tf_only_urdf_contains_required_frames": {
            "base_link", "lidar_link", "front_lidar_link", "imu_link"
        }.issubset(urdf_links),
        "tf_only_urdf_matches_sensor_design_coordinates": (
            joint_origin(urdf, "lidar_link") == [0.5, 0.0, 1.17]
            and joint_origin(urdf, "imu_link") == [-0.12, 0.12, 0.23]
        ),
        "launch_contains_localization_nodes_only": (
            all(
                token in launch_source
                for token in (
                    'package="robot_state_publisher"',
                    'package="robot_localization"',
                    'package="nav2_map_server"',
                    'package="nav2_amcl"',
                    'package="nav2_lifecycle_manager"',
                )
            )
            and 'package="nav2_controller"' not in launch_source
            and 'package="nav2_planner"' not in launch_source
            and "cmd_vel" not in launch_source
        ),
        "launch_does_not_start_legacy_mecanum_driver": "mecanum_driver" not in launch_source,
        "runtime_probe_is_read_only": (
            "create_subscription" in probe_source
            and "TransformListener" in probe_source
            and "create_publisher" not in probe_source
            and 'get_publishers_info_by_topic("/cmd_vel")' in probe_source
        ),
        "presentation_map_is_hash_locked": (
            map_yaml.is_file()
            and map_image.is_file()
            and sha256(map_yaml) == map_candidate["yaml_sha256"]
            and sha256(map_image) == map_candidate["image_sha256"]
        ),
        "presentation_map_cannot_authorize_motion": (
            map_candidate["status"] == "presentation_candidate_not_as_built"
            and map_candidate["accepted_for_motion_planning"] is False
            and map_candidate["physical_clearance_authority"] is False
        ),
        "doorway_conflict_is_numerically_explicit": (
            doorway["reported_narrowest_clear_width_m"]
            < doorway["rev_d_physical_minimum_demo_width_m"]
            < doorway["production_padded_transit_width_m"]
            and math.isclose(
                doorway["padded_width_excess_over_reported_door_m"],
                doorway["production_padded_transit_width_m"]
                - doorway["reported_narrowest_clear_width_m"],
            )
            and doorway["physical_passage_authorized"] is False
        ),
        "all_motion_and_release_authority_is_false": (
            authorization["stationary_sensor_and_tf_observation"] is True
            and all(
                value is False
                for key, value in authorization.items()
                if key != "stationary_sensor_and_tf_observation"
            )
        ),
        "commissioning_sequence_is_monotonic_and_starts_stationary": (
            sequence[0]["id"] == "phase8a_offline_contract"
            and sequence[0]["status"] == "prepared"
            and all(stage["motion_allowed"] is False for stage in sequence)
        ),
        "legacy_mecanum_stack_is_detected_and_quarantined": (
            "OmniMotionModel" in legacy_nav
            and "Mecanum forward kinematics" in legacy_driver
            and "rev_d_differential_encoder_adapter_missing" in blocker_ids
        ),
        "missing_ros_dependencies_are_recorded_as_blocker": (
            bool(missing_dependencies)
            and "ros_dependencies_missing_on_current_workstation" in blocker_ids
        ),
        "ros_distribution_split_is_detected_and_blocks_runtime": (
            ros_distribution["simulation_workstation_baseline"] == "jazzy"
            and ros_distribution["existing_pi_jetson_launch_contract"] == "humble"
            and ros_distribution["target_physical_distribution"] == "unresolved"
            and ros_distribution["cross_distribution_nav2_operation_authorized"] is False
            and "All SBCs MUST run ROS 2 Humble" in rpi_launch
            and "ros_distribution_contract_unresolved" in blocker_ids
        ),
        "critical_physical_blockers_are_complete": {
            "rev_d_differential_encoder_adapter_missing",
            "encoder_scale_not_calibrated",
            "map_not_as_built",
            "physical_door_width_conflict",
            "protective_stop_not_commissioned",
            "ros_distribution_contract_unresolved",
        }.issubset(blocker_ids),
        "robot_bringup_declares_localization_dependencies": all(
            f"<exec_depend>{dependency}</exec_depend>" in package_xml
            for dependency in (
                "nav2_amcl",
                "nav2_map_server",
                "nav2_lifecycle_manager",
                "robot_localization",
            )
        ),
        "simulation_success_retained_without_physical_credit": (
            load_yaml(ROOT / "config/phase7e_administration_static_scan_fusion.yaml")
            ["objective"]["physical_release"]
            is False
        ),
    }

    runtime_probe = None
    if args.runtime_probe.is_file():
        runtime_probe = json.loads(args.runtime_probe.read_text(encoding="utf-8"))
    passed = all(checks.values())
    report = {
        "report_type": "phase8a_physical_localization_offline_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "preparation_passed": passed,
        "status": (
            "accepted_offline_preflight_runtime_blocked"
            if passed
            else "offline_preflight_contract_failed"
        ),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "host_dependency_status": dependency_status,
        "missing_runtime_dependencies": missing_dependencies,
        "runtime_probe": runtime_probe,
        "stationary_runtime_gate_passed": bool(runtime_probe and runtime_probe.get("passed")),
        "physical_runtime_ready": False,
        "motion_authorized": False,
        "doorway_traversal_authorized": False,
        "physical_release": False,
        "hard_blockers": profile["hard_blockers"],
        "source_hashes": {
            "profile": sha256(args.profile),
            "drive_contract": sha256(drive_path),
            "amcl_config": sha256(amcl_path),
            "ekf_config": sha256(ekf_path),
            "localization_urdf": sha256(urdf_path),
            "launch": sha256(launch_path),
            "stationary_probe": sha256(probe_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "AISHA_PHASE8A_OFFLINE_PREFLIGHT "
        f"passed={passed} checks={report['checks_passed']}/{report['checks_total']} "
        f"runtime_ready={report['physical_runtime_ready']} report={args.output.resolve()}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
