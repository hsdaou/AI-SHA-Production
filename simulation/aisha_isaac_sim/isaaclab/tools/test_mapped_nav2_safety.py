#!/usr/bin/env python3
"""Unit tests for the measured Nav2 mapped-site guard."""

from __future__ import annotations

import math
import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aisha_isaaclab"
    / "tasks"
    / "office_nav"
    / "mapped_nav2_safety.py"
)
SPEC = importlib.util.spec_from_file_location("aisha_mapped_nav2_safety_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Doorway = MODULE.Doorway
MappedNav2SafetyGuard = MODULE.MappedNav2SafetyGuard


def guard() -> MappedNav2SafetyGuard:
    return MappedNav2SafetyGuard(
        [
            Doorway("vice_principal", 17.10, -5.05, 1.0, 0.0, 0.85),
            Doorway(
                "principal",
                6.978,
                -7.628,
                math.sqrt(0.5),
                math.sqrt(0.5),
                0.90,
            ),
        ]
    )


class MappedNav2SafetyTests(unittest.TestCase):
    def test_vp_approach_aligns_and_holds_translation(self) -> None:
        result = guard().apply(
            x_m=17.22,
            y_m=-3.80,
            yaw_rad=math.radians(-82.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.02,
            requested_linear_mps=0.10,
            requested_angular_rad_s=0.0,
        )
        self.assertEqual(result.active_door, "vice_principal")
        self.assertTrue(result.doorway_alignment_active)
        self.assertTrue(result.doorway_alignment_hold)
        self.assertEqual(result.linear_mps, 0.0)
        self.assertLess(result.angular_rad_s, 0.0)

    def test_aligned_crossing_is_speed_limited(self) -> None:
        result = guard().apply(
            x_m=17.10,
            y_m=-4.80,
            yaw_rad=math.radians(-90.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.05,
            requested_linear_mps=0.10,
            requested_angular_rad_s=0.4,
        )
        self.assertTrue(result.doorway_alignment_active)
        self.assertFalse(result.doorway_alignment_hold)
        self.assertAlmostEqual(result.linear_mps, 0.10)
        self.assertAlmostEqual(result.angular_rad_s, 0.0, places=6)

    def test_coarse_approach_alignment_does_not_deadlock_progress(self) -> None:
        result = guard().apply(
            x_m=17.006,
            y_m=-3.454,
            yaw_rad=math.radians(-83.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.0,
            requested_linear_mps=0.30,
            requested_angular_rad_s=-0.12,
        )
        self.assertTrue(result.doorway_alignment_active)
        self.assertFalse(result.doorway_alignment_hold)
        self.assertAlmostEqual(result.linear_mps, 0.18)

    def test_post_crossing_guard_keeps_forward_heading(self) -> None:
        candidate = guard()
        candidate.apply(
            x_m=17.10,
            y_m=-4.00,
            yaw_rad=math.radians(-90.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.05,
            requested_linear_mps=0.30,
            requested_angular_rad_s=0.0,
        )
        result = candidate.apply(
            x_m=17.10,
            y_m=-6.10,
            yaw_rad=math.radians(-90.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.05,
            requested_linear_mps=0.30,
            requested_angular_rad_s=0.0,
        )
        self.assertTrue(result.doorway_alignment_active)
        self.assertFalse(result.doorway_alignment_hold)
        self.assertGreater(result.linear_mps, 0.0)
        self.assertAlmostEqual(result.angular_rad_s, 0.0, places=6)

    def test_office_departure_rearms_crossing_direction(self) -> None:
        candidate = guard()
        candidate.apply(
            x_m=17.10,
            y_m=-4.00,
            yaw_rad=math.radians(-90.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.05,
            requested_linear_mps=0.30,
            requested_angular_rad_s=0.0,
        )
        result = candidate.apply(
            x_m=17.10,
            y_m=-6.30,
            yaw_rad=math.radians(90.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.0,
            requested_linear_mps=0.30,
            requested_angular_rad_s=0.0,
        )
        self.assertTrue(result.doorway_alignment_active)
        self.assertFalse(result.doorway_alignment_hold)
        self.assertGreater(result.linear_mps, 0.0)
        self.assertAlmostEqual(result.angular_rad_s, 0.0, places=6)
        self.assertEqual(candidate.report()["doorway_direction_rearms"], 1)

    def test_inside_office_pivot_is_not_suppressed(self) -> None:
        result = guard().apply(
            x_m=17.10,
            y_m=-6.65,
            yaw_rad=math.radians(-90.0),
            yaw_rate_rad_s=0.2,
            forward_speed_mps=0.0,
            requested_linear_mps=0.0,
            requested_angular_rad_s=0.42,
        )
        self.assertFalse(result.doorway_alignment_active)
        self.assertAlmostEqual(result.angular_rad_s, 0.42)

    def test_overspeed_stops_doorway_translation(self) -> None:
        result = guard().apply(
            x_m=17.10,
            y_m=-4.80,
            yaw_rad=math.radians(-90.0),
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.10,
            requested_linear_mps=0.10,
            requested_angular_rad_s=0.0,
        )
        self.assertTrue(result.doorway_overspeed_stop)
        self.assertEqual(result.linear_mps, 0.0)

    def test_polygon_inward_prediction_stops(self) -> None:
        result = guard().apply(
            x_m=3.18,
            y_m=0.0,
            yaw_rad=math.pi,
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.0,
            requested_linear_mps=0.10,
            requested_angular_rad_s=0.0,
        )
        self.assertTrue(result.polygon_no_go_stop)
        self.assertEqual(result.linear_mps, 0.0)

    def test_polygon_outward_motion_is_allowed(self) -> None:
        result = guard().apply(
            x_m=3.18,
            y_m=0.0,
            yaw_rad=0.0,
            yaw_rate_rad_s=0.0,
            forward_speed_mps=0.0,
            requested_linear_mps=0.10,
            requested_angular_rad_s=0.0,
        )
        self.assertFalse(result.polygon_no_go_stop)
        self.assertAlmostEqual(result.linear_mps, 0.10)


if __name__ == "__main__":
    unittest.main()
