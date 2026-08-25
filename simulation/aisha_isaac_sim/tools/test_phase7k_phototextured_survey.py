#!/usr/bin/env python3
"""Contract tests for the Phase 7K phototextured survey layers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/phase7k_phototextured_photogrammetric_survey.yaml"


class Phase7KContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        outputs = cls.profile["outputs"]
        cls.material = json.loads((ROOT / outputs["material_report"]).read_text())
        cls.privacy = json.loads((ROOT / outputs["privacy_manifest"]).read_text())
        cls.dense = json.loads((ROOT / outputs["dense_usd_build"]).read_text())
        cls.build = json.loads((ROOT / outputs["survey_build"]).read_text())

    def test_genuine_dense_geometry_is_packaged(self) -> None:
        self.assertEqual(self.dense["total_vertices"], 311750)
        self.assertEqual(self.dense["total_faces"], 625295)
        self.assertEqual(len(self.dense["outputs"]), 2)
        self.assertTrue(self.dense["clusters_kept_separate"])

    def test_dense_capture_cannot_affect_navigation(self) -> None:
        self.assertTrue(all(not item["collision_enabled"] for item in self.dense["outputs"].values()))
        self.assertFalse(self.build["frozen_safety_contract"]["navigation_collision_geometry_changed"])
        self.assertFalse(self.build["frozen_safety_contract"]["raw_dense_mesh_used_for_collision"])

    def test_privacy_screening_contract(self) -> None:
        self.assertTrue(self.privacy["passed"])
        for cluster in self.privacy["clusters"].values():
            self.assertFalse(cluster["source_obj_committed"])
            self.assertFalse(cluster["source_atlases_committed"])
            for atlas in cluster["presentation_atlases"].values():
                self.assertFalse(atlas["readable_ocr_text_after_screen"])

    def test_capture_derived_pbr_sets(self) -> None:
        self.assertEqual(len(self.material["assets"]), 7)
        self.assertFalse(self.material["source_stills_committed"])
        self.assertTrue(all(set(item["maps"]) == {"albedo", "roughness", "normal"} for item in self.material["assets"].values()))

    def test_survey_and_presentation_modes_are_separate(self) -> None:
        layer = self.profile["layer_contract"]
        self.assertTrue(layer["survey_review_dense_clusters_visible"])
        self.assertTrue(layer["presentation_dense_clusters_present_but_hidden"])
        self.assertTrue(self.build["composite_scene"]["clusters_visible"])
        self.assertTrue(self.build["presentation_scene"]["dense_clusters_present_but_hidden"])

    def test_claim_boundary_is_explicit(self) -> None:
        disclosure = self.profile["presentation_disclosures"]
        self.assertFalse(disclosure["complete_monolithic_photogrammetric_mesh"])
        self.assertFalse(disclosure["registration_is_certified_survey_control"])
        self.assertTrue(disclosure["vice_principal_interior_assumed_because_locked"])
        self.assertFalse(disclosure["physical_release"])

    def test_full_hd_route_contract(self) -> None:
        render = self.profile["render_contract"]
        self.assertEqual(render["resolution"], [1920, 1080])
        self.assertEqual(render["expected_frames"], 480)
        self.assertEqual(len(self.profile["shots"]), 8)
        self.assertTrue(render["robot_should_not_dominate_frame"])


if __name__ == "__main__":
    unittest.main()
