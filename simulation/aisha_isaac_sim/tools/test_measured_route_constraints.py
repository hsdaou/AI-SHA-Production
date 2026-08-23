#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_measured_route_constraints.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("audit_measured_route_constraints", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MeasuredRouteConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()
        cls.config = yaml.safe_load(
            (ROOT / "config" / "administration_assumptions.yaml").read_text(encoding="utf-8")
        )
        cls.overlay = yaml.safe_load(
            (
                ROOT
                / "config"
                / "measured_administration_presentation_2026-08-23.yaml"
            ).read_text(encoding="utf-8")
        )

    @staticmethod
    def trace_at(x: float, y: float, yaw: float, segment: int) -> dict:
        return {
            "outcome": "success",
            "waypoints_completed": 12,
            "completed_steps": 2,
            "pose_trace": [
                {"step": 1, "segment_id": segment, "x_m": x, "y_m": y, "yaw_rad": yaw}
            ],
        }

    def test_centred_vp_door_pose_has_eleven_mm_padded_clearance(self) -> None:
        result = self.tool.audit(
            self.config,
            self.overlay,
            self.trace_at(17.10, -5.05, -math.pi / 2.0, 3),
        )
        self.assertAlmostEqual(
            result["doors"]["vice_principal"]["minimum_padded_clearance_m"],
            0.011,
            places=3,
        )
        self.assertTrue(result["doors"]["vice_principal"]["passed"])

    def test_two_centimetre_vp_lateral_offset_fails_aperture(self) -> None:
        result = self.tool.audit(
            self.config,
            self.overlay,
            self.trace_at(17.12, -5.05, -math.pi / 2.0, 3),
        )
        self.assertFalse(result["doors"]["vice_principal"]["passed"])
        self.assertLess(
            result["doors"]["vice_principal"]["minimum_padded_clearance_m"], 0.0
        )

    def test_central_polygon_pose_is_rejected(self) -> None:
        result = self.tool.audit(
            self.config,
            self.overlay,
            self.trace_at(0.0, 0.0, 0.0, 0),
        )
        self.assertFalse(result["checks"]["central_atrium_no_go_respected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
