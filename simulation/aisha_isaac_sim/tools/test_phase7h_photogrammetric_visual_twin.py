#!/usr/bin/env python3
"""Contract tests for the Phase 7H hybrid visual-twin package."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/phase7h_photogrammetric_visual_twin.yaml"
MATERIAL_REPORT = ROOT / "results/phase7h_photo_materials_report.json"
ACCEPTANCE = ROOT / "results/administration_nav2_phase7h_photogrammetric_acceptance.json"


class Phase7HContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        cls.material_report = json.loads(MATERIAL_REPORT.read_text(encoding="utf-8"))

    def test_reconstruction_evidence_is_real_and_scoped(self) -> None:
        evidence = self.profile["reconstruction_evidence"]
        self.assertGreaterEqual(evidence["atrium_corridor_cluster"]["dense_points"], 250000)
        self.assertGreaterEqual(evidence["principal_office_cluster"]["dense_points"], 200000)
        self.assertFalse(evidence["source_media_committed"])
        self.assertIn("holes", evidence["rejection_reason"])

    def test_navigation_geometry_is_not_replaced_by_raw_mesh(self) -> None:
        hybrid = self.profile["hybrid_visual_contract"]
        self.assertIn("RoomPlan", hybrid["metric_local_geometry_authority"])
        self.assertFalse(hybrid["raw_dense_mesh_collision_enabled"])
        self.assertFalse(hybrid["raw_dense_mesh_in_hero_render"])

    def test_photo_materials_are_privacy_safe_and_present(self) -> None:
        self.assertEqual(self.material_report["status"], "passed")
        self.assertIn("surface-only", self.material_report["source"]["privacy_scope"])
        for relative in self.profile["hybrid_visual_contract"]["capture_derived_materials"]:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_presentation_contract_keeps_wide_full_hd_replay(self) -> None:
        render = self.profile["render_contract"]
        self.assertEqual(render["resolution"], [1920, 1080])
        self.assertEqual(render["expected_frames"], 576)
        self.assertEqual(render["camera_style"], "fixed_human_height_wide_environmental")
        self.assertEqual(len(self.profile["shots"]), 8)
        self.assertTrue(render["robot_should_not_dominate_frame"])

    def test_claim_boundary_is_explicit(self) -> None:
        disclosure = self.profile["presentation_disclosures"]
        self.assertIn("hybrid", disclosure["environment"])
        self.assertIn("Not a complete", disclosure["environment"])
        self.assertTrue(disclosure["vice_principal_interior_assumed_because_locked"])
        self.assertFalse(disclosure["physical_release"])

    def test_acceptance_when_available(self) -> None:
        if not ACCEPTANCE.is_file():
            self.skipTest("Phase 7H final render acceptance has not been generated yet")
        acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        self.assertEqual(acceptance["status"], "accepted")
        self.assertEqual(acceptance["checks_passed"], acceptance["checks_total"])


if __name__ == "__main__":
    unittest.main()
