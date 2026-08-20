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
            if waypoint["action"] == "presentation_stop"
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


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)
