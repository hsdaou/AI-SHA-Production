#!/usr/bin/env python3
"""Contract tests for the Phase 7M NuRec presentation reel."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/phase7m_nurec_presentation_reel.yaml"


class Phase7MPresentationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        outputs = cls.profile["outputs"]
        cls.registration = json.loads((ROOT / outputs["registration"]).read_text())
        cls.render = json.loads((ROOT / outputs["render_report"]).read_text())
        cls.encode = json.loads((ROOT / outputs["encode_report"]).read_text())
        cls.privacy = json.loads((ROOT / outputs["privacy_review"]).read_text())
        cls.acceptance = json.loads((ROOT / outputs["acceptance"]).read_text())

    def test_gravity_sign_and_route_anchor_are_corrected(self) -> None:
        anchor = self.registration["metric_anchor"]
        metrics = self.registration["validation"]
        self.assertTrue(anchor["gravity_axis_sign_resolved"])
        self.assertLess(metrics["principal_turn_anchor_residual_m"], 0.001)
        self.assertLess(metrics["shared_atrium_world_residual_median_m"], 0.05)

    def test_final_render_is_full_hd_and_uses_four_safe_cameras(self) -> None:
        self.assertTrue(self.render["passed"])
        self.assertEqual(self.render["resolution"], [1920, 1080])
        self.assertEqual(len(self.render["frames"]), 198)
        self.assertEqual(sorted(set(self.render["camera_codes_rendered"])), [5, 20, 65, 95])

    def test_motion_and_claim_boundary_are_explicit(self) -> None:
        self.assertTrue(self.render["recorded_pose_selection_without_interpolation"])
        self.assertTrue(self.render["source_motion_was_live_nav2_and_learned_safety"])
        self.assertFalse(self.render["presentation_renderer_executes_policy_live"])
        self.assertTrue(self.profile["motion"]["presentation_retimed"])
        self.assertFalse(self.profile["presentation_disclosures"]["physical_release"])

    def test_encoded_video_is_valid_local_preview(self) -> None:
        media = self.encode["media_probe"]
        self.assertTrue(self.encode["passed"])
        self.assertEqual([media["width"], media["height"]], [1920, 1080])
        self.assertAlmostEqual(media["fps"], 24.0)
        self.assertAlmostEqual(media["duration_s"], 13.75, places=2)
        self.assertTrue(self.privacy["passed_local_preview"])

    def test_external_privacy_and_physical_release_remain_closed(self) -> None:
        self.assertFalse(self.privacy["authorized_human_privacy_review_completed"])
        self.assertFalse(self.privacy["external_distribution_approved"])
        self.assertTrue(self.privacy["external_distribution_requires_user_review"])
        self.assertFalse(self.acceptance["physical_release"])


if __name__ == "__main__":
    unittest.main()
