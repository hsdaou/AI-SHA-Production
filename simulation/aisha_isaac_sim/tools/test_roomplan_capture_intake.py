#!/usr/bin/env python3
"""Focused regression tests for semantic RoomPlan capture intake."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


TOOL = Path(__file__).with_name("intake_roomplan_capture.py")


def import_tool():
    spec = importlib.util.spec_from_file_location("intake_roomplan_capture", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoomPlanCaptureIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = import_tool()

    def test_clearance_profiles_keep_production_and_presentation_separate(self) -> None:
        production = self.tool.clearance_profile(0.85, 0.08)
        presentation = self.tool.clearance_profile(0.85, 0.03)
        self.assertFalse(production["passes"])
        self.assertEqual(production["required_clear_width_m"], 0.928)
        self.assertTrue(presentation["passes"])
        self.assertEqual(presentation["required_clear_width_m"], 0.828)
        self.assertAlmostEqual(presentation["nominal_margin_per_side_m"], 0.011)

    def test_semantic_asset_extracts_door_dimensions_and_pose(self) -> None:
        text = '''
string Category = "Door(Isopen: True)"
string UUID = "door-test"
point3f[] points = [(-0.425, -1.06, 0), (0.425, 1.06, -0.08)]
matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (2, 0, 3, 1) )
'''
        door = self.tool.semantic_asset("assets/Mesh/Walls/Wall0/Door0.usda", text)
        self.assertIsNotNone(door)
        self.assertEqual(door["clear_width_scan_m"], 0.85)
        self.assertEqual(door["clear_height_scan_m"], 2.12)
        self.assertEqual(door["centre_native_xz_m"], [2.0, 3.0])
        self.assertTrue(door["is_open"])

    def test_door_match_selects_manual_scale_reference(self) -> None:
        doors = [
            {"id": "Door0", "clear_width_scan_m": 0.81, "clear_height_scan_m": 2.14},
            {"id": "Door7", "clear_width_scan_m": 0.854, "clear_height_scan_m": 2.121},
        ]
        match = self.tool.door_match(doors, 0.85, 2.12)
        self.assertEqual(match["id"], "Door7")
        self.assertAlmostEqual(match["width_residual_m"], 0.004)


if __name__ == "__main__":
    unittest.main(verbosity=2)
