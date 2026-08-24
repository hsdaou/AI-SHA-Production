#!/usr/bin/env python3
"""Contract tests for the Phase 7F operator-facing Omniverse capture."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "config/phase7f_operator_presentation.yaml"
RENDERER = ROOT / "scripts/render_administration_route.py"
PHASE7E_GATE = ROOT / "results/administration_nav2_phase7e_static_fusion_integration_gate.json"
PHASE7E_MISSION = ROOT / "results/administration_nav2_phase7e_static_fusion_mission.json"
REPLAY_GATE = ROOT / "results/administration_nav2_phase7f_operator_replay_validation.json"
RENDER_REPORT = ROOT / "results/administration_nav2_phase7f_operator_rtx_render_report.json"
ACCEPTANCE = ROOT / "results/administration_nav2_phase7f_operator_presentation_acceptance.json"
VIDEO = ROOT / "media/videos/AI-SHA_Phase7F_Operator_Omniverse_Presentation.mp4"
CONTACT_SHEET = ROOT / "media/AI-SHA_Phase7F_Operator_Omniverse_contact_sheet.jpg"
TRAINING = ROOT / "config/training.yaml"


class Phase7FOperatorPresentationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))

    def test_camera_profile_is_wide_human_height_and_covers_route_once(self) -> None:
        shots = self.profile["shots"]
        contract = self.profile["render_contract"]
        self.assertEqual(len(shots), contract["expected_shots"])
        self.assertEqual(
            [segment for shot in shots for segment in shot["segments"]],
            list(range(12)),
        )
        for shot in shots:
            self.assertLessEqual(shot["focal_length_mm"], contract["focal_length_max_mm"])
            self.assertGreaterEqual(shot["camera"][2], contract["camera_height_range_m"][0])
            self.assertLessEqual(shot["camera"][2], contract["camera_height_range_m"][1])
            source_fraction = shot.get("source_fraction", [0.0, 1.0])
            self.assertEqual(len(source_fraction), 2)
            self.assertGreaterEqual(source_fraction[0], 0.0)
            self.assertLess(source_fraction[0], source_fraction[1])
            self.assertLessEqual(source_fraction[1], 1.0)

    def test_profile_keeps_replay_and_site_limitations_explicit(self) -> None:
        disclosure = self.profile["presentation_disclosures"]
        self.assertIn("not live policy execution", disclosure["overlay"])
        self.assertIn("not photogrammetric", disclosure["environment"])
        self.assertTrue(disclosure["vice_principal_interior_assumed_because_locked"])
        self.assertFalse(disclosure["physical_localization_credit"])
        self.assertFalse(disclosure["physical_safety_credit"])
        self.assertFalse(disclosure["physical_release"])

    def test_renderer_selects_recorded_poses_and_accepts_profile(self) -> None:
        source = RENDERER.read_text(encoding="utf-8")
        self.assertIn('"--presentation-profile"', source)
        self.assertIn("Select recorded poses without interpolating or inventing motion", source)
        self.assertIn('"visual_replay_is_live_policy_execution": False', source)
        self.assertNotIn("np.interp", source)

    def test_phase7e_source_and_phase7f_replay_gate_are_accepted(self) -> None:
        phase7e = json.loads(PHASE7E_GATE.read_text(encoding="utf-8"))
        mission = json.loads(PHASE7E_MISSION.read_text(encoding="utf-8"))
        replay = json.loads(REPLAY_GATE.read_text(encoding="utf-8"))
        self.assertTrue(phase7e["passed"])
        self.assertEqual(phase7e["checks_passed"], phase7e["checks_total"])
        self.assertEqual(phase7e["checks_total"], 40)
        self.assertTrue(mission["passed"])
        self.assertEqual(mission["completed_legs"], 12)
        self.assertEqual(mission["waypoints_completed"], 12)
        self.assertTrue(replay["passed"])

    def test_final_video_and_acceptance_are_hash_linked_and_released(self) -> None:
        report = json.loads(RENDER_REPORT.read_text(encoding="utf-8"))
        acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        training = yaml.safe_load(TRAINING.read_text(encoding="utf-8"))
        video_sha256 = hashlib.sha256(VIDEO.read_bytes()).hexdigest()

        self.assertTrue(VIDEO.is_file())
        self.assertTrue(CONTACT_SHEET.is_file())
        self.assertTrue(acceptance["passed"])
        self.assertEqual(acceptance["checks_passed"], 19)
        self.assertEqual(acceptance["checks_total"], 19)
        self.assertEqual(report["video_sha256"], video_sha256)
        self.assertEqual(acceptance["video"]["sha256"], video_sha256)
        self.assertEqual(report["encoded_frame_count"], 576)
        self.assertEqual(report["resolution"], [1920, 1080])
        release = training["release"]
        self.assertTrue(release["phase7f_operator_omniverse_presentation_available"])
        self.assertTrue(release["phase7f_operator_omniverse_presentation_validated"])
        self.assertEqual(release["phase7f_operator_presentation_checks_passed"], "19/19")
        self.assertFalse(release["physical_robot_release"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
