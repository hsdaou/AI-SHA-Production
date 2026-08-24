#!/usr/bin/env python3
"""Contract tests for the Phase 7I scan-matched atrium package."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/phase7i_scan_matched_atrium.yaml"
OVERLAY = ROOT / "config/measured_administration_presentation_2026-08-23.yaml"


class Phase7IContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        cls.overlay = yaml.safe_load(OVERLAY.read_text(encoding="utf-8"))

    def test_authorities_are_explicit(self) -> None:
        contract = self.profile["scan_matched_atrium_contract"]
        self.assertIn("RoomPlan", contract["metric_envelope_authority"])
        self.assertIn("walkthrough", contract["visual_semantics_authority"])
        self.assertIn("page 2", contract["global_topology_authority"])

    def test_incorrect_proxy_is_replaced_without_retraining(self) -> None:
        contract = self.profile["scan_matched_atrium_contract"]
        self.assertTrue(contract["previous_proxy_rejected"])
        self.assertFalse(contract["route_critical_collision_geometry_changed"])
        self.assertFalse(contract["learned_trajectory_changed"])

    def test_atrium_visual_contract_matches_capture_features(self) -> None:
        features = set(self.profile["scan_matched_atrium_contract"]["required_visual_features"])
        self.assertIn("opposing_glazed_walnut_reception_windows", features)
        self.assertIn("paired_slim_black_public_benches", features)
        self.assertIn("three_privacy_safe_display_easels", features)
        self.assertIn("rear_glazed_double_door_and_emblems", features)

    def test_measured_overlay_carries_revision(self) -> None:
        atrium = self.overlay["measured_visual_twin"]["atrium"]
        self.assertEqual(atrium["source_section"], "unidentified4")
        self.assertIn("opposing", atrium["reception_treatment"])
        self.assertFalse(atrium["route_critical_collision_geometry_changed"])

    def test_render_contract_is_full_hd_and_wide(self) -> None:
        render = self.profile["render_contract"]
        self.assertEqual(render["resolution"], [1920, 1080])
        self.assertEqual(render["expected_frames"], 576)
        self.assertTrue(render["robot_should_not_dominate_frame"])
        self.assertEqual(len(self.profile["shots"]), 8)

    def test_claim_boundary_remains_honest(self) -> None:
        disclosures = self.profile["presentation_disclosures"]
        self.assertTrue(disclosures["vice_principal_interior_assumed_because_locked"])
        self.assertFalse(disclosures["physical_release"])


if __name__ == "__main__":
    unittest.main()
