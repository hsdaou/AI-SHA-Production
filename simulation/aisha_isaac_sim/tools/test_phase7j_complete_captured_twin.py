#!/usr/bin/env python3
"""Contract tests for the Phase 7J complete captured-area twin."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/phase7j_complete_captured_administration_twin.yaml"
BUILD = ROOT / "results/phase7j_complete_captured_administration_build.json"


class Phase7JContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        cls.build = json.loads(BUILD.read_text(encoding="utf-8"))

    def test_complete_capture_and_principal_layers_are_required(self) -> None:
        layer = self.profile["layer_contract"]
        self.assertTrue(layer["complete_primary_capture_included"])
        self.assertTrue(layer["principal_supplement_registered"])
        self.assertTrue(layer["visual_and_collision_layers_separate"])
        self.assertTrue(layer["incomplete_roomplan_floor_hidden_only_in_composite"])

    def test_primary_registration_uses_page2_hallway(self) -> None:
        registration = self.profile["registration"]["primary"]
        self.assertIn("hall", registration["method"])
        self.assertEqual(registration["world_anchor_xy_m"], [4.7, 0.0])
        self.assertEqual(registration["metric_scale"], 1.0)
        self.assertEqual(registration["world_z_offset_m"], 1.3561)

    def test_flattened_visual_layer_has_all_semantic_walls(self) -> None:
        visual = self.build["visual_layer"]
        self.assertTrue(visual["flattened"])
        self.assertFalse(visual["external_roomplan_dependencies"])
        self.assertEqual(visual["category_counts"]["Wall"], 81)

    def test_capture_furniture_is_preserved_but_route_composite_is_cleanable(self) -> None:
        composite = self.build["composite_scene"]
        self.assertTrue(composite["full_capture_layer_retains_all_captured_furniture"])
        self.assertGreaterEqual(composite["presentation_hidden_movable_route_conflict_count"], 1)
        self.assertEqual(
            self.profile["layer_contract"][
                "movable_visual_conflict_corner_swept_envelope_radius_m"
            ],
            0.85,
        )
        self.assertGreaterEqual(
            composite["presentation_hidden_primary_principal_furniture_duplicate_count"], 1
        )

    def test_scan_drift_conflicts_are_recorded_not_deleted_from_survey(self) -> None:
        composite = self.build["composite_scene"]
        conflicts = composite["presentation_hidden_static_visual_route_conflicts"]
        self.assertGreaterEqual(len(conflicts), 1)
        self.assertTrue(
            all(item["full_capture_visual_layer_retains_component"] for item in conflicts)
        )
        self.assertTrue(composite["visible_plan_authority_floor_with_atrium_step_down"])
        self.assertGreaterEqual(composite["presentation_hidden_roomplan_floor_count"], 1)

    def test_render_contract_is_full_hd(self) -> None:
        render = self.profile["render_contract"]
        self.assertEqual(render["resolution"], [1920, 1080])
        self.assertEqual(render["expected_frames"], 480)
        self.assertEqual(len(self.profile["shots"]), 8)
        self.assertTrue(render["robot_should_not_dominate_frame"])

    def test_claim_boundary_is_explicit(self) -> None:
        disclosure = self.profile["presentation_disclosures"]
        self.assertTrue(disclosure["vice_principal_interior_assumed_because_locked"])
        self.assertIn("not a complete phototextured", disclosure["environment"].lower())
        self.assertFalse(disclosure["physical_release"])


if __name__ == "__main__":
    unittest.main()
