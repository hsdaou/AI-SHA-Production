#!/usr/bin/env python3
"""Validate the frozen Full HD Phase 7G Omniverse presentation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "config/phase7g_presentation_freeze.yaml"
    )
    parser.add_argument(
        "--freeze-report",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7g_presentation_freeze_report.json",
    )
    parser.add_argument(
        "--phase7f-acceptance",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7f_operator_presentation_acceptance.json",
    )
    parser.add_argument(
        "--phase7e-integration",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_fusion_integration_gate.json",
    )
    parser.add_argument(
        "--dynamic-video-report",
        type=Path,
        default=ROOT / "results/phase4a_dynamic_safety_presentation_video_report.json",
    )
    parser.add_argument(
        "--dynamic-run-report",
        type=Path,
        default=ROOT / "results/phase4a_administration_dynamic_showcase_report.json",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "media/videos/AI-SHA_Phase7G_Omniverse_Presentation_Freeze.mp4",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=ROOT / "media/AI-SHA_Phase7G_Omniverse_Presentation_Freeze_contact_sheet.jpg",
    )
    parser.add_argument(
        "--live-smoke-report",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7g_live_omniverse_smoke.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7g_presentation_freeze_acceptance.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"width": 0, "height": 0, "fps": 0.0, "frames": 0, "duration_s": 0.0}
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    result["duration_s"] = result["frames"] / result["fps"] if result["fps"] else 0.0
    return result


def visual_statistics(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    sampled = 0
    minimum_mean = math.inf
    minimum_stddev = math.inf
    for index in range(0, frame_count, 12):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, stddev = cv2.meanStdDev(grey)
        minimum_mean = min(minimum_mean, float(mean[0][0]))
        minimum_stddev = min(minimum_stddev, float(stddev[0][0]))
        sampled += 1
    capture.release()
    return {
        "sampled_frames": sampled,
        "minimum_mean": minimum_mean if sampled else 0.0,
        "minimum_stddev": minimum_stddev if sampled else 0.0,
    }


def main() -> int:
    args = parse_args()
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    freeze = load_json(args.freeze_report)
    phase7f = load_json(args.phase7f_acceptance)
    phase7e = load_json(args.phase7e_integration)
    dynamic_video = load_json(args.dynamic_video_report)
    dynamic_run = load_json(args.dynamic_run_report)
    contract = profile["freeze_contract"]
    disclosure = profile["presentation_disclosures"]
    live = profile["live_omniverse_contract"]
    live_smoke = load_json(args.live_smoke_report)
    metadata = inspect_video(args.video)
    visual = visual_statistics(args.video)
    timeline = freeze.get("timeline", [])
    timeline_contiguous = bool(timeline) and timeline[0].get("start_frame") == 0
    if timeline_contiguous:
        timeline_contiguous = all(
            current.get("start_frame") == previous.get("end_frame") + 1
            for previous, current in zip(timeline, timeline[1:])
        ) and timeline[-1].get("end_frame") == metadata["frames"] - 1
    dynamic_metrics = dynamic_run.get("metrics", {})
    dynamic_termination = dynamic_run.get("termination", {})
    live_player = ROOT / live["player"]
    live_launcher = ROOT / live["launcher"]
    runbook = ROOT / profile["outputs"]["runbook"]
    player_source = live_player.read_text(encoding="utf-8") if live_player.is_file() else ""

    checks = {
        "freeze_builder_passed_all_checks": freeze.get("passed") is True
        and freeze.get("checks_passed") == freeze.get("checks_total"),
        "phase7f_wide_mission_retained_19_of_19": phase7f.get("passed") is True
        and phase7f.get("checks_passed") == phase7f.get("checks_total") == 19,
        "phase7e_live_source_retained_40_of_40": phase7e.get("passed") is True
        and phase7e.get("checks_passed") == phase7e.get("checks_total") == 40,
        "wide_human_height_camera_contract_retained": phase7f.get("checks", {}).get(
            "wide_environmental_lenses_and_human_camera_heights"
        )
        is True
        and contract.get("robot_should_not_dominate_mission_frame") is True
        and contract.get("camera_style") == "fixed_human_height_wide_environmental",
        "path_traced_full_hd_mission_source_retained": phase7f.get("checks", {}).get(
            "path_traced_full_hd_presentation_profile"
        )
        is True,
        "dynamic_video_and_run_are_accepted": dynamic_video.get("passed") is True
        and all(dynamic_video.get("checks", {}).values())
        and dynamic_run.get("passed") is True
        and all(dynamic_run.get("checks", {}).values()),
        "learned_brake_and_protective_stop_observed": int(
            dynamic_metrics.get("encounter_safety_authority_steps", 0)
        )
        > 0
        and float(dynamic_metrics.get("maximum_learned_brake_fraction", 0.0)) >= 0.02
        and float(dynamic_metrics.get("protective_full_stop_duration_s", 0.0)) >= 2.0,
        "dynamic_resume_and_zero_contacts_observed": float(
            dynamic_metrics.get("maximum_resumed_velocity_mps", 0.0)
        )
        >= 0.45
        and dynamic_termination.get("dynamic_obstacle_collision") is False
        and dynamic_termination.get("static_collision") is False,
        "video_hash_and_size_linked": args.video.is_file()
        and args.video.stat().st_size > 5_000_000
        and freeze.get("output", {}).get("video_sha256") == sha256(args.video)
        and freeze.get("output", {}).get("video_size_bytes") == args.video.stat().st_size,
        "full_hd_24fps_46_second_freeze": [metadata["width"], metadata["height"]]
        == contract["resolution"]
        and math.isclose(float(metadata["fps"]), float(contract["fps"]), abs_tol=0.01)
        and metadata["frames"] == contract["expected_total_frames"]
        and math.isclose(
            float(metadata["duration_s"]), float(contract["expected_duration_s"]), abs_tol=0.01
        ),
        "timeline_is_complete_and_contiguous": timeline_contiguous
        and sum(int(item.get("frames", 0)) for item in timeline) == metadata["frames"],
        "wide_mission_frames_retained_once_in_order": freeze.get("sources", {})
        .get("wide_mission", {})
        .get("frames_retained_once_in_order")
        == contract["expected_wide_mission_frames"]
        and freeze.get("assembly", {}).get("mission_motion_changed") is False,
        "dynamic_duration_preserved_without_retiming": freeze.get("sources", {})
        .get("dynamic_safety", {})
        .get("duration_error_s", math.inf)
        <= 1.0 / float(contract["fps"])
        and freeze.get("assembly", {}).get("dynamic_motion_retimed") is False,
        "dynamic_insert_is_deliberately_framed": freeze.get("assembly", {}).get(
            "dynamic_evidence_window_scaled_and_framed"
        )
        is True
        and contract.get("dynamic_insert_style") == "framed_evidence_window_with_metrics_panel",
        "contact_sheet_hash_linked": args.contact_sheet.is_file()
        and args.contact_sheet.stat().st_size > 250_000
        and freeze.get("output", {}).get("contact_sheet_sha256") == sha256(args.contact_sheet),
        "no_black_or_uniform_sampled_frames": visual["sampled_frames"] >= 90
        and visual["minimum_mean"] > 12.0
        and visual["minimum_stddev"] > 7.0,
        "live_omniverse_gui_player_is_packaged": live_player.is_file()
        and live_launcher.is_file()
        and "SimulationApp" in player_source
        and "open_stage" in player_source
        and "set_active_camera" in player_source
        and "set_robot_pose" in player_source,
        "live_player_selects_recorded_poses_without_interpolation": (
            "Select recorded poses without interpolation" in player_source
            and "np.interp" not in player_source
            and live.get("recorded_pose_selection_without_interpolation") is True
        ),
        "live_player_completed_real_isaac_sim_smoke": live_smoke.get("report_type")
        == "phase7g_live_omniverse_presentation_session"
        and live_smoke.get("status") == "completed_requested_loops"
        and live_smoke.get("renderer") == "RaytracedLighting"
        and live_smoke.get("loops_completed") == 1
        and live_smoke.get("frames_presented") == 16
        and set(live_smoke.get("segment_frame_counts", {}))
        == {str(index) for index in range(12)}
        and live_smoke.get("recorded_pose_selection_without_interpolation") is True
        and live_smoke.get("physical_release") is False,
        "operator_runbook_and_backup_are_packaged": runbook.is_file()
        and "run_phase7g_live_omniverse.sh" in runbook.read_text(encoding="utf-8")
        and "AI-SHA_Phase7G_Omniverse_Presentation_Freeze.mp4"
        in runbook.read_text(encoding="utf-8"),
        "site_assumptions_remain_explicit": disclosure.get(
            "vice_principal_interior_assumed_because_locked"
        )
        is True
        and "not a photogrammetric or as-built survey"
        in str(disclosure.get("environment", "")),
        "presentation_player_does_not_claim_live_policy_execution": live.get(
            "source_motion_was_live_nav2_and_learned_safety"
        )
        is True
        and live.get("presentation_player_executes_policy_live") is False,
        "simulation_only_no_physical_credit_or_release": disclosure.get(
            "physical_localization_credit"
        )
        is False
        and disclosure.get("physical_safety_credit") is False
        and disclosure.get("physical_release") is False,
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase7g_presentation_freeze_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": "presentation_frozen_and_operator_ready" if passed else "presentation_freeze_rejected",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "video": {
            "path": str(args.video.resolve()),
            "sha256": sha256(args.video) if args.video.is_file() else None,
            "size_bytes": args.video.stat().st_size if args.video.is_file() else None,
            **metadata,
            "visual_statistics": visual,
        },
        "contact_sheet": {
            "path": str(args.contact_sheet.resolve()),
            "sha256": sha256(args.contact_sheet) if args.contact_sheet.is_file() else None,
        },
        "operator_package": {
            "live_launcher": str(live_launcher.resolve()),
            "live_player": str(live_player.resolve()),
            "live_smoke_report": str(args.live_smoke_report.resolve()),
            "live_smoke_report_sha256": sha256(args.live_smoke_report),
            "runbook": str(runbook.resolve()),
            "backup_video": str(args.video.resolve()),
        },
        "claim_boundary": {
            "supported": (
                "Presentation-ready Full HD Omniverse package using the accepted wide PathTracing "
                "office mission, accepted dynamic obstacle response, and a GUI Omniverse replay."
            ),
            "source_motion_was_live_nav2_and_learned_safety": True,
            "presentation_player_executes_policy_live": False,
            "environment_is_photogrammetric_or_as_built": False,
            "physical_localization_credit": False,
            "physical_safety_credit": False,
            "physical_release": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PHASE7G_PRESENTATION_FREEZE passed={passed} "
        f"checks={report['checks_passed']}/{report['checks_total']} "
        f"video={args.video.resolve()}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
