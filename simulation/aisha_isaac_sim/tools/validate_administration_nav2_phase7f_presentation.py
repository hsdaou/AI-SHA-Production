#!/usr/bin/env python3
"""Validate the operator-facing RTX replay of the accepted Phase 7E mission."""

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
        "--profile", type=Path, default=ROOT / "config/phase7f_operator_presentation.yaml"
    )
    parser.add_argument(
        "--integration",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_fusion_integration_gate.json",
    )
    parser.add_argument(
        "--mission",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_fusion_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_fusion_bridge.json",
    )
    parser.add_argument(
        "--replay-validation",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7f_operator_replay_validation.json",
    )
    parser.add_argument(
        "--render-report",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7f_operator_rtx_render_report.json",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "media/videos/AI-SHA_Phase7F_Operator_Omniverse_Presentation.mp4",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=ROOT / "media/AI-SHA_Phase7F_Operator_Omniverse_contact_sheet.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7f_operator_presentation_acceptance.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visual_frame_statistics(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    minimum_mean = math.inf
    minimum_stddev = math.inf
    sampled = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        # Exclude the deliberate top/bottom evidence overlays so a blocked
        # environment view cannot pass merely because the labels are visible.
        content = frame[120 : max(121, frame.shape[0] - 54), :]
        grey = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        mean, stddev = cv2.meanStdDev(grey)
        minimum_mean = min(minimum_mean, float(mean[0][0]))
        minimum_stddev = min(minimum_stddev, float(stddev[0][0]))
        sampled += 1
    capture.release()
    return {
        "sampled_frames": sampled,
        "minimum_content_mean": minimum_mean if sampled else 0.0,
        "minimum_content_stddev": minimum_stddev if sampled else 0.0,
    }


def main() -> int:
    args = parse_args()
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    integration = load_json(args.integration)
    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    replay = load_json(args.replay_validation)
    render = load_json(args.render_report)
    contract = profile["render_contract"]
    disclosure = profile["presentation_disclosures"]
    expected_resolution = list(contract["resolution"])
    expected_segments = list(range(12))
    rendered_segments = [
        int(segment)
        for shot in render.get("shots", [])
        for segment in shot.get("segment_ids", [])
    ]
    rendered_shots = render.get("shots", [])
    hallway_shot = rendered_shots[1] if len(rendered_shots) > 1 else {}
    hallway_fraction = hallway_shot.get("source_fraction", [])
    hallway_start_fraction = float(hallway_fraction[0]) if hallway_fraction else 0.0
    office_wide_framing = len(rendered_shots) >= 7 and all(
        float(rendered_shots[index].get("focal_length_mm", math.inf)) <= 12.0
        for index in (2, 3, 6)
    )
    principal_approach_fraction = (
        rendered_shots[5].get("source_fraction", []) if len(rendered_shots) > 5 else []
    )
    principal_approach_start = (
        float(principal_approach_fraction[0]) if principal_approach_fraction else 0.0
    )

    capture = cv2.VideoCapture(str(args.video))
    video_open = capture.isOpened()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if video_open else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if video_open else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if video_open else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if video_open else 0
    capture.release()
    visual_stats = visual_frame_statistics(args.video) if video_open else {
        "sampled_frames": 0,
        "minimum_content_mean": 0.0,
        "minimum_content_stddev": 0.0,
    }

    mapped = bridge.get("mapped_site_safety", {})
    learned = bridge.get("learned_safety", {})
    blocked = bridge.get("blocked_route", {})
    route_speed = bridge.get("route_scoped_speed_evidence", {})
    checks = {
        "phase7e_full_office_gate_retained_40_of_40": (
            integration.get("passed") is True
            and integration.get("checks_passed") == integration.get("checks_total") == 40
        ),
        "successful_live_source_completed_all_12_legs": (
            mission.get("passed") is True
            and mission.get("outcome") == "success"
            and mission.get("completed_legs") == mission.get("waypoints_completed") == 12
        ),
        "replay_clearance_evidence_passed": replay.get("passed") is True,
        "render_hash_links_current_phase7e_mission": (
            render.get("trajectory_report_sha256") == sha256(args.mission)
        ),
        "render_hash_links_current_operator_profile": (
            render.get("presentation_profile_sha256") == sha256(args.profile)
        ),
        "eight_human_height_shots_cover_each_leg_once": (
            len(render.get("shots", [])) == int(contract["expected_shots"]) == 8
            and rendered_segments == expected_segments
            and render.get("camera_style") == contract["camera_style"]
        ),
        "wide_environmental_lenses_and_human_camera_heights": all(
            float(shot.get("focal_length_mm", math.inf)) <= float(contract["focal_length_max_mm"])
            and float(contract["camera_height_range_m"][0])
            <= float(shot.get("camera", [0.0, 0.0, math.inf])[2])
            <= float(contract["camera_height_range_m"][1])
            for shot in render.get("shots", [])
        )
        and contract.get("robot_should_not_dominate_frame") is True
        and hallway_start_fraction >= 0.60
        and office_wide_framing
        and principal_approach_start >= 0.50,
        "path_traced_full_hd_presentation_profile": (
            render.get("renderer") == contract["renderer"] == "PathTracing"
            and int(render.get("path_tracing_spp", 0)) >= int(contract["path_tracing_spp"])
            and render.get("resolution") == expected_resolution == [1920, 1080]
        ),
        "encoded_video_matches_render_contract": (
            video_open
            and frames == render.get("frame_count") == render.get("encoded_frame_count")
            == int(contract["expected_frames"])
            and math.isclose(fps, float(contract["fps"]), abs_tol=0.01)
            and [width, height] == expected_resolution
        ),
        "encoded_video_hash_and_size_match": (
            args.video.is_file()
            and args.video.stat().st_size > 2_000_000
            and render.get("video_sha256") == sha256(args.video)
            and render.get("video_size_bytes") == args.video.stat().st_size
        ),
        "visual_qa_contact_sheet_packaged": (
            args.contact_sheet.is_file() and args.contact_sheet.stat().st_size > 200_000
        ),
        "no_black_or_uniform_environment_frames": (
            visual_stats["sampled_frames"] == frames
            and visual_stats["minimum_content_mean"] > 15.0
            and visual_stats["minimum_content_stddev"] > 8.0
        ),
        "both_offices_entered_and_departed": (
            mapped.get("doorway_entries") == {"vice_principal": 2, "principal": 2}
        ),
        "office_departure_alignment_and_doorway_speed_retained": (
            float(mapped.get("maximum_abs_speed_in_doorway_mps", math.inf)) <= 0.100001
            and integration.get("checks", {}).get(
                "both_departure_headings_aligned_in_pivot_zones"
            )
            is True
        ),
        "accepted_hallway_speed_tier_retained": (
            min(
                float(route_speed.get("maximum_observed_linear_mps_by_segment", {}).get(segment, 0.0))
                for segment in ("1", "5")
            )
            >= 0.74
        ),
        "native_lidar_block_wait_clear_replan_retained": (
            blocked.get("enabled") is True
            and blocked.get("triggered") is True
            and blocked.get("cleared") is True
            and float(blocked.get("maximum_robot_speed_while_active_mps", math.inf)) < 0.001
        ),
        "learned_360_safety_was_live_in_source": (
            int(learned.get("primary_policy_steps", 0)) > 0
            and int(learned.get("fallback_policy_steps", 0)) > 0
            and int(learned.get("authority_steps", 0)) > 0
            and int(learned.get("brake_steps", 0)) > 0
        ),
        "replay_and_environment_limitations_are_disclosed": (
            render.get("claim_boundary", {}).get("visual_replay_is_live_policy_execution") is False
            and render.get("claim_boundary", {}).get("source_motion_was_live_policy_execution") is True
            and disclosure.get("vice_principal_interior_assumed_because_locked") is True
            and "not photogrammetric" in str(disclosure.get("environment", ""))
        ),
        "simulation_only_no_physical_release": (
            mission.get("physical_release") is False
            and bridge.get("physical_release") is False
            and disclosure.get("physical_release") is False
            and disclosure.get("physical_safety_credit") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase7f_operator_presentation_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": "accepted_operator_presentation_simulation" if passed else "not_accepted",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "video": {
            "path": str(args.video.resolve()),
            "sha256": sha256(args.video) if args.video.is_file() else None,
            "size_bytes": args.video.stat().st_size if args.video.is_file() else None,
            "frames": frames,
            "fps": fps,
            "resolution": [width, height],
            "duration_s": frames / fps if fps > 0.0 else None,
            "visual_frame_statistics": visual_stats,
        },
        "contact_sheet": {
            "path": str(args.contact_sheet.resolve()),
            "sha256": sha256(args.contact_sheet) if args.contact_sheet.is_file() else None,
        },
        "source_evidence": {
            "integration": str(args.integration.resolve()),
            "mission": str(args.mission.resolve()),
            "bridge": str(args.bridge.resolve()),
            "replay_validation": str(args.replay_validation.resolve()),
            "render_report": str(args.render_report.resolve()),
            "profile": str(args.profile.resolve()),
        },
        "claim_boundary": {
            "supported": (
                "Operator-facing Full HD PathTracing replay of recorded poses from the accepted "
                "Phase 7E live Nav2/learned-safety full-office simulation mission."
            ),
            "visual_replay_is_live_policy_execution": False,
            "source_motion_was_live_policy_execution": True,
            "environment_is_photogrammetric_or_as_built": False,
            "physical_localization_credit": False,
            "physical_safety_credit": False,
            "physical_release": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PHASE7F_OPERATOR_PRESENTATION passed={passed} "
        f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
