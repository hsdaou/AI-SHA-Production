#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = PACKAGE_ROOT / "maps" / "administration_provisional"
REPORT_PATH = PACKAGE_ROOT / "results" / "administration_provisional_map_report.json"


class AdministrationOccupancyMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.map_yaml = yaml.safe_load(
            (MAP_DIR / "administration_provisional.yaml").read_text(encoding="utf-8")
        )
        cls.image = Image.open(MAP_DIR / cls.map_yaml["image"])
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_ros_map_contract(self) -> None:
        self.assertEqual(self.map_yaml["mode"], "trinary")
        self.assertEqual(self.map_yaml["resolution"], 0.05)
        self.assertEqual(self.map_yaml["negate"], 0)
        self.assertEqual(len(self.map_yaml["origin"]), 3)

    def test_map_has_unknown_free_and_occupied_cells(self) -> None:
        values = set(self.image.getdata())
        self.assertTrue({0, 205, 254}.issubset(values))

    def test_route_centres_are_free(self) -> None:
        self.assertTrue(self.report["route"]["all_waypoint_centres_free"])
        self.assertTrue(all(item["free"] for item in self.report["route"]["waypoints"]))

    def test_release_boundary_is_explicit(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertFalse(self.report["physical_release"])
        self.assertIn("LiDAR", self.report["replacement_gate"])


if __name__ == "__main__":
    unittest.main()
