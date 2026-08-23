#!/usr/bin/env python3
"""Validate the accepted Phase 6 Nav2 evidence and final RTX presentation cut."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration",
        type=Path,
        default=ROOT / "results/administration_nav2_phase6_high_speed_integration_gate.json",
    )
    parser.add_argument(
        "--mission",
        type=Path,
        default=ROOT / "results/administration_nav2_phase6_high_speed_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "results/administration_nav2_phase6_high_speed_bridge.json",
    )
    parser.add_argument(
        "--replay-validation",
        type=Path,
        default=ROOT / "results/administration_nav2_phase6_rtx_replay_validation.json",
    )
    parser.add_argument(
        "--render-report",
        type=Path,
        default=ROOT / "results/administration_nav2_phase6_rtx_render_report.json",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "media/videos/AI-SHA_Phase6_Nav2_LearnedSafety_RTX_Presentation.mp4",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=ROOT / "media/AI-SHA_Phase6_Nav2_RTX_contact_sheet.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/administration_nav2_phase6_rtx_presentation_acceptance.json",
    )
    args = parser.parse_args()

    integration = load_json(args.integration)
    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    replay = load_json(args.replay_validation)
    render = load_json(args.render_report)
    video = args.video.resolve()
    contact_sheet = args.contact_sheet.resolve()
    capture = cv2.VideoCapture(str(video))
    video_open = capture.isOpened()
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if video_open else 0
    video_fps = float(capture.get(cv2.CAP_PROP_FPS)) if video_open else 0.0
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if video_open else 0
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if video_open else 0
    capture.release()

    trace = mission.get("pose_trace", [])
    segment_ids = {
        int(sample.get("segment_id", -1))
        for sample in trace
        if isinstance(sample, dict)
    }
    mapped = bridge.get("mapped_site_safety", {})
    learned = bridge.get("learned_safety", {})
    localization = bridge.get("localization", {})
    checks = {
        "phase6_live_integration_passed_28_of_28": (
            integration.get("passed") is True
            and integration.get("checks_passed")
            == integration.get("checks_total")
            == 28
        ),
        "trace_source_is_successful_12_leg_live_mission": (
            mission.get("passed") is True
            and mission.get("outcome") == "success"
            and mission.get("completed_legs") == mission.get("waypoints_completed") == 12
        ),
        "trace_covers_all_segments_with_finite_poses": (
            segment_ids == set(range(12))
            and len(trace) > 1000
            and all(
                math.isfinite(float(sample[key]))
                for sample in trace
                for key in ("x_m", "y_m", "yaw_rad")
            )
        ),
        "render_uses_current_trace_by_hash": (
            render.get("trajectory_report_sha256") == sha256_file(args.mission)
        ),
        "trace_clearance_validation_passed": (
            replay.get("passed") is True
            and render.get("clearance_validation_report_sha256")
            == sha256_file(args.replay_validation)
        ),
        "path_tracing_profile_is_presentation_quality": (
            render.get("renderer") == "PathTracing"
            and int(render.get("path_tracing_spp", 0)) >= 8
            and render.get("resolution") == [1280, 720]
        ),
        "six_shots_cover_all_route_segments": (
            len(render.get("shots", [])) == 6
            and [
                segment
                for shot in render.get("shots", [])
                for segment in shot.get("segment_ids", [])
            ]
            == list(range(12))
        ),
        "encoded_video_matches_render_contract": (
            video_open
            and video_frames == render.get("frame_count") == render.get("encoded_frame_count") == 240
            and math.isclose(video_fps, 20.0, abs_tol=0.01)
            and math.isclose(float(render.get("encoded_fps", 0.0)), 20.0, abs_tol=0.01)
            and (video_width, video_height) == (1280, 720)
        ),
        "encoded_video_hash_and_size_match": (
            video.is_file()
            and video.stat().st_size > 1_000_000
            and render.get("video_sha256") == sha256_file(video)
            and render.get("video_size_bytes") == video.stat().st_size
        ),
        "visual_qa_contact_sheet_packaged": (
            contact_sheet.is_file() and contact_sheet.stat().st_size > 100_000
        ),
        "both_offices_and_doorways_exercised": (
            mapped.get("doorway_entries") == {"vice_principal": 2, "principal": 2}
            and float(mapped.get("maximum_abs_speed_in_doorway_mps", float("inf")))
            <= 0.100001
        ),
        "learned_safety_was_live_in_source_run": (
            int(learned.get("primary_policy_steps", 0)) > 0
            and int(learned.get("fallback_policy_steps", 0)) > 0
            and int(learned.get("authority_steps", 0)) > 0
            and int(learned.get("brake_steps", 0)) > 0
        ),
        "ground_truth_localization_limitation_disclosed": (
            localization.get("nav2_global_pose_source")
            == "isaac_ground_truth_odom_with_identity_map_to_odom"
            and localization.get("physical_localization_credit") is False
        ),
        "presentation_remains_simulation_only": (
            mission.get("physical_release") is False
            and bridge.get("physical_release") is False
            and integration.get("claim_boundary", {}).get("physical_release") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase6_rtx_presentation_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": "accepted_presentation_simulation" if passed else "not_accepted",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "video": {
            "path": str(video),
            "sha256": sha256_file(video) if video.is_file() else None,
            "size_bytes": video.stat().st_size if video.is_file() else None,
            "frames": video_frames,
            "fps": video_fps,
            "resolution": [video_width, video_height],
            "duration_s": video_frames / video_fps if video_fps > 0.0 else None,
        },
        "source_evidence": {
            "integration_gate": str(args.integration.resolve()),
            "mission_trace": str(args.mission.resolve()),
            "bridge": str(args.bridge.resolve()),
            "replay_validation": str(args.replay_validation.resolve()),
            "render_report": str(args.render_report.resolve()),
            "trace_records": len(trace),
        },
        "claim_boundary": {
            "supported": (
                "Presentation of the accepted Isaac Sim/Nav2 mission using the "
                "Phase 6 actor on 0.8 m/s hallway legs and the accepted Phase 3N "
                "actor elsewhere, replayed from its recorded wheel-physics poses "
                "in the measured-presentation Omniverse PathTracing scene."
            ),
            "visual_replay_is_live_policy_execution": False,
            "source_motion_was_live_policy_execution": True,
            "vice_principal_interior_assumed_because_locked": True,
            "physical_localization_credit": False,
            "physical_release": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PHASE6_RTX_PRESENTATION passed={passed} "
        f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
