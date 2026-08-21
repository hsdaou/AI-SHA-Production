#!/usr/bin/env python3
"""Simulator-independent unit tests for AI-SHA drive limiting and watchdog logic."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from aisha_common import DifferentialDriveLimiter, wheel_speeds_rad_s


class KinematicsTests(unittest.TestCase):
    def test_straight(self) -> None:
        self.assertEqual(wheel_speeds_rad_s(0.5, 0.0, 0.1, 0.72), (5.0, 5.0))

    def test_pivot(self) -> None:
        left, right = wheel_speeds_rad_s(0.0, 0.5, 0.1, 0.72)
        self.assertAlmostEqual(left, -1.8)
        self.assertAlmostEqual(right, 1.8)

    def test_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            wheel_speeds_rad_s(0.1, 0.0, 0.0, 0.72)


class LimiterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = DifferentialDriveLimiter(
            wheel_radius_m=0.1,
            wheel_track_m=0.72,
            max_linear_mps=0.5,
            max_angular_rad_s=1.0,
            max_acceleration_mps2=0.5,
            max_angular_acceleration_rad_s2=1.0,
            watchdog_timeout_s=0.25,
        )
        self.controller.reset(0.0)

    def test_speed_and_acceleration_clamp(self) -> None:
        self.controller.command(2.0, 2.0, 0.0)
        left, right = self.controller.update(0.1, 0.1)
        self.assertAlmostEqual(self.controller.linear, 0.05)
        self.assertAlmostEqual(self.controller.angular, 0.1)
        self.assertTrue(math.isfinite(left) and math.isfinite(right))

    def test_watchdog_latches_until_reset(self) -> None:
        self.controller.command(0.3, 0.0, 0.0)
        self.assertNotEqual(self.controller.update(0.1, 0.1), (0.0, 0.0))
        self.assertEqual(self.controller.update(0.26, 0.1), (0.0, 0.0))
        self.controller.command(0.3, 0.0, 0.27)
        self.assertEqual(self.controller.update(0.28, 0.01), (0.0, 0.0))
        self.controller.reset(0.30)
        self.controller.command(0.3, 0.0, 0.31)
        self.assertNotEqual(self.controller.update(0.32, 0.01), (0.0, 0.0))

    def test_rejects_non_finite_command(self) -> None:
        with self.assertRaises(ValueError):
            self.controller.command(float("nan"), 0.0, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
