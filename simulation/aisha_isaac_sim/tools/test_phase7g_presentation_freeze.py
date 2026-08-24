#!/usr/bin/env python3
"""Contract tests for the frozen Phase 7G Omniverse presentation package."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "config/phase7g_presentation_freeze.yaml"
REPORT = ROOT / "results/administration_nav2_phase7g_presentation_freeze_report.json"
ACCEPTANCE = ROOT / "results/administration_nav2_phase7g_presentation_freeze_acceptance.json"
LIVE_SMOKE = ROOT / "results/administration_nav2_phase7g_live_omniverse_smoke.json"
VIDEO = ROOT / "media/videos/AI-SHA_Phase7G_Omniverse_Presentation_Freeze.mp4"
CONTACT = ROOT / "media/AI-SHA_Phase7G_Omniverse_Presentation_Freeze_contact_sheet.jpg"
PLAYER = ROOT / "scripts/play_phase7g_presentation_live.py"
TRAINING = ROOT / "config/training.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase7GPresentationFreezeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        cls.live_smoke = json.loads(LIVE_SMOKE.read_text(encoding="utf-8"))

    def test_freeze_gate_passes_all_23_checks(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(self.acceptance["passed"])
        self.assertEqual(self.acceptance["checks_passed"], 23)
        self.assertEqual(self.acceptance["checks_total"], 23)
        self.assertTrue(all(self.acceptance["checks"].values()))

    def test_final_video_is_hash_linked_full_hd_and_46_seconds(self) -> None:
        self.assertTrue(VIDEO.is_file())
        self.assertGreater(VIDEO.stat().st_size, 5_000_000)
        self.assertEqual(self.report["output"]["video_sha256"], sha256(VIDEO))
        self.assertEqual(self.acceptance["video"]["sha256"], sha256(VIDEO))
        capture = cv2.VideoCapture(str(VIDEO))
        self.assertTrue(capture.isOpened())
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1920)
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1080)
        self.assertTrue(math.isclose(capture.get(cv2.CAP_PROP_FPS), 24.0, abs_tol=0.01))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1107)
        capture.release()

    def test_wide_mission_and_dynamic_evidence_are_preserved(self) -> None:
        contract = self.profile["freeze_contract"]
        self.assertTrue(contract["robot_should_not_dominate_mission_frame"])
        self.assertEqual(contract["camera_style"], "fixed_human_height_wide_environmental")
        self.assertEqual(
            self.report["sources"]["wide_mission"]["frames_retained_once_in_order"], 576
        )
        self.assertFalse(self.report["assembly"]["mission_motion_changed"])
        self.assertFalse(self.report["assembly"]["dynamic_motion_retimed"])
        self.assertTrue(self.report["assembly"]["dynamic_frames_resampled_for_target_rate"])
        self.assertLessEqual(
            self.report["sources"]["dynamic_safety"]["duration_error_s"], 1.0 / 24.0
        )

    def test_live_omniverse_player_completed_a_real_smoke(self) -> None:
        self.assertEqual(self.live_smoke["status"], "completed_requested_loops")
        self.assertEqual(self.live_smoke["renderer"], "RaytracedLighting")
        self.assertEqual(self.live_smoke["loops_completed"], 1)
        self.assertEqual(self.live_smoke["frames_presented"], 16)
        self.assertEqual(set(self.live_smoke["segment_frame_counts"]), {str(i) for i in range(12)})
        source = PLAYER.read_text(encoding="utf-8")
        self.assertIn("SimulationApp", source)
        self.assertIn("set_active_camera", source)
        self.assertIn("wait_for_stage_ready", source)
        self.assertIn('"omni:kit:centerOfInterest"', source)
        self.assertIn("traceback.print_exc", source)
        self.assertIn("Select recorded poses without interpolation", source)
        self.assertNotIn("np.interp", source)

    def test_contact_sheet_and_operator_package_are_present(self) -> None:
        self.assertTrue(CONTACT.is_file())
        self.assertGreater(CONTACT.stat().st_size, 250_000)
        self.assertEqual(self.acceptance["contact_sheet"]["sha256"], sha256(CONTACT))
        package = self.acceptance["operator_package"]
        self.assertTrue(Path(package["live_launcher"]).is_file())
        self.assertTrue(Path(package["runbook"]).is_file())
        self.assertEqual(Path(package["backup_video"]), VIDEO.resolve())

    def test_release_ledger_closes_presentation_only(self) -> None:
        release = yaml.safe_load(TRAINING.read_text(encoding="utf-8"))["release"]
        self.assertTrue(release["phase7g_presentation_freeze_available"])
        self.assertTrue(release["phase7g_presentation_freeze_validated"])
        self.assertEqual(release["phase7g_presentation_freeze_checks_passed"], "23/23")
        self.assertTrue(release["phase7g_live_omniverse_player_smoke_passed"])
        self.assertFalse(release["physical_robot_release"])
        claim = self.acceptance["claim_boundary"]
        self.assertTrue(claim["source_motion_was_live_nav2_and_learned_safety"])
        self.assertFalse(claim["presentation_player_executes_policy_live"])
        self.assertFalse(claim["physical_release"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
