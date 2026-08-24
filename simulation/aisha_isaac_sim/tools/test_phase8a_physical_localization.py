#!/usr/bin/env python3
"""Contract tests for the Phase 8A stationary physical-localization preflight."""

from __future__ import annotations

import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
PROFILE = ROOT / "config/phase8a_physical_localization_preflight.yaml"
DRIVE = ROOT / "config/aisha_drive.yaml"
AMCL = REPO / "src/robot_bringup/config/amcl_rev_d_preflight.yaml"
EKF = REPO / "src/robot_bringup/config/ekf_rev_d.yaml"
URDF = REPO / "src/robot_description/urdf/aisha_rev_d_localization.urdf"
LAUNCH = REPO / "src/robot_bringup/launch/phase8a_localization_preflight.launch.py"
PROBE = ROOT / "tools/probe_phase8a_stationary_localization.py"
REPORT = ROOT / "results/phase8a_physical_localization_preflight.json"
TRAINING = ROOT / "config/training.yaml"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Phase8APhysicalLocalizationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_yaml(PROFILE)
        cls.drive = load_yaml(DRIVE)
        cls.amcl = load_yaml(AMCL)["amcl"]["ros__parameters"]
        cls.ekf = load_yaml(EKF)["ekf_filter_node"]["ros__parameters"]

    def test_rev_d_geometry_is_preserved_with_production_padding(self) -> None:
        robot = self.profile["robot_contract"]
        self.assertEqual(robot["revision"], "D")
        self.assertEqual(robot["architecture"], "differential_drive")
        self.assertEqual(robot["raw_footprint_xy_m"], self.drive["navigation"]["raw_footprint_xy_m"])
        self.assertEqual(robot["physical_nav2_padding_per_side_m"], 0.080)
        self.assertAlmostEqual(robot["padded_transit_width_m"], 0.928)

    def test_motion_and_physical_release_are_all_fail_safe(self) -> None:
        authorization = self.profile["authorization_boundary"]
        self.assertTrue(authorization["stationary_sensor_and_tf_observation"])
        for name, value in authorization.items():
            if name != "stationary_sensor_and_tf_observation":
                self.assertFalse(value, name)
        self.assertTrue(all(not stage["motion_allowed"] for stage in self.profile["commissioning_sequence"]))

    def test_amcl_and_ekf_are_real_time_differential_contracts(self) -> None:
        localization = self.profile["localization_contract"]
        self.assertFalse(self.amcl["use_sim_time"])
        self.assertEqual(self.amcl["robot_model_type"], "nav2_amcl::DifferentialMotionModel")
        self.assertEqual(self.amcl["scan_topic"], "/scan")
        self.assertFalse(self.ekf["use_sim_time"])
        self.assertTrue(self.ekf["two_d_mode"])
        self.assertTrue(self.ekf["publish_tf"])
        self.assertEqual(self.ekf["odom0"], "/wheel/odom_raw")
        self.assertEqual(self.ekf["imu0"], "/imu/data")
        self.assertEqual(self.ekf["world_frame"], "odom")
        self.assertEqual(localization["filtered_odometry_topic"], "/odometry/filtered")
        self.assertEqual(len(self.ekf["process_noise_covariance"]), 225)
        distro = self.profile["ros_distribution_boundary"]
        self.assertEqual(distro["simulation_workstation_baseline"], "jazzy")
        self.assertEqual(distro["existing_pi_jetson_launch_contract"], "humble")
        self.assertEqual(distro["target_physical_distribution"], "unresolved")
        self.assertFalse(distro["cross_distribution_nav2_operation_authorized"])

    def test_localization_urdf_has_rev_d_sensor_frames(self) -> None:
        root = ET.parse(URDF).getroot()
        links = {link.attrib["name"] for link in root.findall("link")}
        self.assertTrue({"base_link", "lidar_link", "front_lidar_link", "imu_link"}.issubset(links))
        source = URDF.read_text(encoding="utf-8")
        self.assertEqual(root.findall("transmission"), [])
        self.assertEqual(root.findall("ros2_control"), [])
        self.assertNotIn("<plugin", source)

    def test_launch_cannot_start_navigation_or_motion(self) -> None:
        source = LAUNCH.read_text(encoding="utf-8")
        for package in (
            "robot_state_publisher",
            "robot_localization",
            "nav2_map_server",
            "nav2_amcl",
            "nav2_lifecycle_manager",
        ):
            self.assertIn(f'package="{package}"', source)
        self.assertNotIn("nav2_controller", source)
        self.assertNotIn("nav2_planner", source)
        self.assertNotIn("mecanum_driver", source)
        self.assertNotIn("cmd_vel", source)

    def test_runtime_probe_is_observation_only(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("create_subscription", source)
        self.assertIn("TransformListener", source)
        self.assertIn('get_publishers_info_by_topic("/cmd_vel")', source)
        self.assertNotIn("create_publisher", source)
        self.assertIn('"motion_command_published_by_probe": False', source)

    def test_map_and_doorway_are_not_misrepresented_as_physically_released(self) -> None:
        map_candidate = self.profile["map_candidate"]
        map_yaml = ROOT / map_candidate["yaml"]
        map_image = ROOT / map_candidate["image"]
        self.assertEqual(hashlib.sha256(map_yaml.read_bytes()).hexdigest(), map_candidate["yaml_sha256"])
        self.assertEqual(hashlib.sha256(map_image.read_bytes()).hexdigest(), map_candidate["image_sha256"])
        self.assertFalse(map_candidate["accepted_for_motion_planning"])
        doorway = self.profile["doorway_release_conflict"]
        self.assertLess(doorway["reported_narrowest_clear_width_m"], doorway["rev_d_physical_minimum_demo_width_m"])
        self.assertLess(doorway["rev_d_physical_minimum_demo_width_m"], doorway["production_padded_transit_width_m"])
        self.assertFalse(doorway["physical_passage_authorized"])

    def test_generated_offline_report_passes_only_preparation(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["preparation_passed"])
        self.assertEqual(report["checks_passed"], report["checks_total"])
        self.assertFalse(report["physical_runtime_ready"])
        self.assertFalse(report["motion_authorized"])
        self.assertFalse(report["physical_release"])
        release = load_yaml(TRAINING)["release"]
        self.assertTrue(release["phase8a_physical_localization_offline_preflight_passed"])
        self.assertEqual(release["phase8a_physical_localization_offline_checks_passed"], "23/23")
        self.assertFalse(release["phase8a_stationary_runtime_gate_passed"])
        self.assertFalse(release["phase8a_physical_runtime_ready"])
        self.assertFalse(release["physical_robot_release"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
