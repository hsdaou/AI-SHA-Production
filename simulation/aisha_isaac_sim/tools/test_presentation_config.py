#!/usr/bin/env python3
"""Unit checks for presentation assumptions and production sensor contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def load(name: str) -> dict:
    with (ROOT / "config" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class PresentationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = load("administration_assumptions.yaml")
        cls.sensors = load("sensors.yaml")
        cls.refinement = load("geometry_rtx_refinement.yaml")

    def test_door_widths_clear_presentation_gate(self) -> None:
        minimum = self.scene["presentation_release"]["minimum_demo_door_clear_width_m"]
        for name, door in self.scene["doors"].items():
            with self.subTest(door=name):
                self.assertGreaterEqual(door["clear_width_m"], minimum)

    def test_door_values_are_not_mislabelled_as_measured(self) -> None:
        for name, door in self.scene["doors"].items():
            with self.subTest(door=name):
                self.assertIn("assumption", door["width_status"])
                self.assertIn("assumption", door["threshold_status"])
                self.assertEqual(
                    door["swing"],
                    "outward_open_90_deg_for_presentation_route_clearance",
                )
                self.assertIn("presentation_assumption", door["hinge"])

    def test_approved_page_two_is_the_geometry_source(self) -> None:
        provenance = self.scene["provenance"]
        self.assertEqual(provenance["plan_status"], "approved_page_2_reviewed")
        self.assertEqual(provenance["plan_source"]["page"], 2)
        self.assertEqual(
            provenance["plan_source"]["sha256"],
            "4d8698f868c296442ebd6e667d4700f2e9bc7f27b4787639395a370d76779112",
        )
        self.assertEqual(provenance["geometry_status"], "plan_derived_route_scoped")

    def test_office_relationship_matches_block_a_plan(self) -> None:
        cluster = self.scene["plan_geometry"]["south_east_cluster"]
        vice = cluster["vice_principal"]["centre_xy_m"]
        principal = cluster["principal"]["centre_xy_m"]
        self.assertGreater(vice[0], principal[0])
        self.assertGreater(principal[1], -12.80)
        self.assertLess(principal[1], vice[1])

    def test_route_visits_vice_then_principal(self) -> None:
        stops = [
            waypoint["id"]
            for waypoint in self.scene["route"]["waypoints"]
            if waypoint["action"].startswith("presentation_stop")
        ]
        self.assertEqual(stops, ["vice_principal", "principal"])

    def test_pivot_circle_fits_declared_rotation_zones(self) -> None:
        hallway = self.scene["known_dimensions"]["hallway_clear_width_m"]["value"]
        pivot = self.scene["presentation_release"]["pivot_clear_circle_m"]
        self.assertGreaterEqual(hallway, pivot)

    def test_presentation_does_not_release_physical_route(self) -> None:
        release = self.scene["presentation_release"]
        self.assertTrue(release["accepted_for_scripted_presentation"])
        self.assertFalse(release["accepted_for_physical_or_unsupervised_operation"])
        self.assertFalse(release["threshold_contact_claim_allowed"])

    def test_atrium_columns_declare_conservative_trace_clearance(self) -> None:
        columns = self.scene["appearance"]["atrium_columns"]
        release = self.scene["presentation_release"]
        robot_radius = (
            (release["robot_transit_width_m"] / 2.0) ** 2
            + (release["robot_transit_length_m"] / 2.0) ** 2
        ) ** 0.5
        self.assertEqual(len(columns["positions_xy_m"]), 4)
        self.assertGreater(
            columns["minimum_trace_centre_clearance_m"],
            robot_radius + columns["radius_m"],
        )
        self.assertIn("not_surveyed", columns["status"])

    def test_repository_backed_sensor_contracts(self) -> None:
        self.assertEqual(
            self.sensors["source"]["commit"],
            "8893535b4043ff766d914e8bfe54a789cf3deba0",
        )
        frames = self.sensors["frames"]
        self.assertEqual(frames["crown_lidar"]["model"], "LDLiDAR LD19")
        self.assertEqual(frames["front_camera"]["model"], "Intel RealSense D435")
        self.assertEqual(frames["imu"]["model"], "Bosch BNO055")
        self.assertEqual(frames["imu"]["publish_rate_hz"], 50.0)

    def test_phase3_geometry_uses_only_printed_plan_dimensions_as_anchors(self) -> None:
        refinement = self.refinement
        self.assertEqual(
            refinement["source"]["sha256"],
            self.scene["provenance"]["plan_source"]["sha256"],
        )
        printed = refinement["printed_dimensions"]
        self.assertEqual(printed["atrium_diagonal_m"]["value"], 12.75)
        self.assertEqual(printed["administration_hallway_clear_width_m"]["value"], 2.80)
        self.assertEqual(printed["conference_room_size_m"]["value"], [7.80, 6.30])
        self.assertEqual(printed["principal_diagonal_frontage_m"]["value"], 4.73)
        self.assertTrue(
            all(item["confidence"] == "plan_dimension" for item in printed.values())
        )

    def test_phase3_geometry_keeps_site_measurements_blocked(self) -> None:
        unresolved = self.refinement["unresolved_site_measurements"]
        self.assertIn("vice_principal_door_clear_width", unresolved)
        self.assertIn("principal_door_clear_width", unresolved)
        self.assertIn("both_threshold_heights_and_profiles", unresolved)
        self.assertIn("not a site survey", self.refinement["rtx_material_profile"]["claim_boundary"])

    def test_rtx_profile_requires_three_map_pbr_materials(self) -> None:
        profile = self.refinement["rtx_material_profile"]
        self.assertEqual(profile["renderer"], "PathTracing")
        self.assertEqual(profile["offline_samples_per_pixel"], 64)
        self.assertEqual(
            profile["maps"],
            ["albedo", "perceptual_roughness", "tangent_space_normal"],
        )


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)
