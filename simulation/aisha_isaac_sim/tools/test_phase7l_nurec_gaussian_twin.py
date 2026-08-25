#!/usr/bin/env python3
"""Contract tests for the Phase 7L NuRec Gaussian administration twin."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/phase7l_nurec_gaussian_twin.yaml"


class Phase7LContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        outputs = cls.profile["outputs"]
        cls.training = json.loads((ROOT / outputs["training_report"]).read_text())
        cls.registration = json.loads(
            (ROOT / outputs["registration_report"]).read_text()
        )
        cls.build = json.loads((ROOT / outputs["composite_build"]).read_text())
        cls.render = json.loads(
            (ROOT / outputs["composite_isaac_render"]).read_text()
        )

    def test_two_full_nurec_components_are_trained(self) -> None:
        components = self.training["components"]
        self.assertEqual(set(components), {"main_administration", "principal_office"})
        self.assertTrue(all(item["iterations"] == 30000 for item in components.values()))
        self.assertTrue(all(item["gaussian_count"] > 1_000_000 for item in components.values()))

    def test_registration_gate_passes_with_provisional_claim(self) -> None:
        quality = self.registration["validation"]
        self.assertTrue(self.registration["passed"])
        self.assertLess(quality["shared_atrium_world_residual_median_m"], 0.05)
        self.assertLess(quality["shared_atrium_world_residual_p95_m"], 0.20)
        self.assertIn(
            "not a certified survey",
            self.registration["layer_contract"]["registration_classification"],
        )

    def test_native_basis_composition_preserves_gaussian_quality(self) -> None:
        contract = self.build["layer_contract"]
        self.assertEqual(contract["nurec_asset_transform"], "identity_native_training_basis")
        self.assertEqual(contract["metric_world_transform"], "inverse_of_registered_nurec_to_world_sim3")
        self.assertTrue(self.build["checks"]["legacy_render_geometry_hidden"])
        self.assertTrue(self.render["passed"])

    def test_visual_and_navigation_layers_are_separate(self) -> None:
        contract = self.build["layer_contract"]
        self.assertTrue(contract["gaussians_visual_only"])
        self.assertFalse(contract["navigation_collision_geometry_changed"])
        self.assertFalse(contract["raw_gaussians_used_for_lidar_or_collision"])
        self.assertTrue(contract["frozen_collision_hidden_but_active"])

    def test_claim_boundary_is_explicit(self) -> None:
        disclosure = self.profile["presentation_disclosures"]
        self.assertTrue(disclosure["principal_office_captured"])
        self.assertTrue(disclosure["vice_principal_interior_assumed_because_locked"])
        self.assertFalse(disclosure["registration_is_certified_survey_control"])
        self.assertFalse(disclosure["visual_replay_is_live_policy_execution"])
        self.assertFalse(disclosure["physical_release"])

    def test_privacy_sensitive_outputs_stay_local(self) -> None:
        self.assertFalse(self.profile["capture"]["original_video_committed"])
        self.assertFalse(self.profile["capture"]["extracted_images_committed"])
        self.assertFalse(self.profile["capture"]["gaussian_assets_committed"])
        self.assertTrue(
            self.profile["presentation_disclosures"][
                "privacy_review_required_before_external_media_distribution"
            ]
        )


if __name__ == "__main__":
    unittest.main()
