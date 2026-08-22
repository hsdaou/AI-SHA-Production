#!/usr/bin/env python3
"""Validate the final AI-SHA Omniverse administration presentation reel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


EXPECTED_CHECKPOINT_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-report", type=Path, required=True)
    parser.add_argument("--safety-video-report", type=Path, required=True)
    parser.add_argument("--safety-run-report", type=Path, required=True)
    parser.add_argument("--reel-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mission = json.loads(args.mission_report.read_text(encoding="utf-8"))
    safety_video = json.loads(
        args.safety_video_report.read_text(encoding="utf-8")
    )
    safety_run = json.loads(args.safety_run_report.read_text(encoding="utf-8"))
    reel = json.loads(args.reel_report.read_text(encoding="utf-8"))
    mission_video = Path(reel["sources"]["complete_office_mission"]["video"])
    safety_source = Path(reel["sources"]["dynamic_safety_encounter"]["video"])
    output_video = Path(reel["output_video"])
    mission_run_report = Path(mission["source_run_report"])
    if not all(
        path.is_file()
        for path in (mission_video, safety_source, output_video, mission_run_report)
    ):
        raise FileNotFoundError("one or more presentation evidence files are missing")
    mission_run = json.loads(mission_run_report.read_text(encoding="utf-8"))

    capture = cv2.VideoCapture(str(output_video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {output_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    duration_s = frames / fps

    timeline = reel.get("timeline", [])
    contiguous = bool(timeline) and timeline[0].get("start_frame") == 0
    if contiguous:
        for previous, current in zip(timeline, timeline[1:]):
            contiguous = contiguous and (
                current.get("start_frame") == previous.get("end_frame") + 1
            )
        contiguous = contiguous and timeline[-1].get("end_frame") == frames - 1
    timeline_frame_sum = sum(int(section.get("frames", 0)) for section in timeline)
    named_sections = [section.get("section") for section in timeline]
    mission_source = reel["sources"]["complete_office_mission"]
    safety_source_record = reel["sources"]["dynamic_safety_encounter"]
    control_steps = mission_run.get("control_steps", {})
    mission_termination = mission_run.get("termination_details", {})
    safety_metrics = safety_run.get("metrics", {})
    safety_termination = safety_run.get("termination", {})

    checks = {
        "reel_report_type": reel.get("report_type")
        == "final_omniverse_administration_presentation_reel",
        "reel_maker_passed": reel.get("passed") is True
        and all(reel.get("checks", {}).values()),
        "mission_report_type": mission.get("report_type")
        == "administration_live_policy_presentation_video",
        "mission_report_passed": mission.get("passed") is True,
        "mission_motion_unchanged": mission.get("motion_changed") is False
        and mission_source.get("motion_changed_by_assembly") is False,
        "mission_uses_accepted_checkpoint": mission.get("checkpoint_sha256")
        == EXPECTED_CHECKPOINT_SHA256,
        "mission_completed_all_waypoints": mission_run.get("outcome") == "success"
        and mission_run.get("waypoints_completed") == 12
        and mission_run.get("route_segment_count") == 12,
        "mission_policy_only": mission_run.get("route_control") == "policy-only"
        and control_steps.get("physics_supervisor_turn") == 0
        and control_steps.get("presentation_dwell") == 0
        and mission_run.get("root_transform_animation") is False,
        "mission_zero_collisions": mission_termination.get(
            "dynamic_obstacle_collision"
        )
        is False
        and mission_termination.get("static_collision") is False,
        "safety_video_report_type": safety_video.get("report_type")
        == "phase4a_dynamic_safety_presentation_video",
        "safety_video_report_passed": safety_video.get("passed") is True
        and all(safety_video.get("checks", {}).values()),
        "safety_run_report_type": safety_run.get("report_type")
        == "phase4a_live_dynamic_safety_showcase",
        "safety_run_passed": safety_run.get("passed") is True
        and safety_run.get("outcome") == "success"
        and all(safety_run.get("checks", {}).values()),
        "safety_uses_same_accepted_checkpoint": safety_run.get("checkpoint", {}).get(
            "sha256"
        )
        == EXPECTED_CHECKPOINT_SHA256,
        "learned_brake_observed": safety_metrics.get(
            "encounter_safety_authority_steps", 0
        )
        > 0
        and safety_metrics.get("maximum_learned_brake_fraction", 0.0) >= 0.02,
        "protective_stop_and_resume_observed": safety_metrics.get(
            "protective_full_stop_duration_s", 0.0
        )
        >= 0.5
        and safety_metrics.get("maximum_resumed_velocity_mps", 0.0) >= 0.25,
        "dynamic_encounter_zero_collisions": safety_termination.get(
            "dynamic_obstacle_collision"
        )
        is False
        and safety_termination.get("static_collision") is False,
        "mission_video_hash_linked": mission_source.get("video_sha256")
        == mission.get("output_video_sha256")
        == sha256_file(mission_video),
        "mission_report_hash_linked": mission_source.get("report_sha256")
        == sha256_file(args.mission_report),
        "safety_video_hash_linked": safety_source_record.get("video_sha256")
        == safety_video.get("output_video_sha256")
        == sha256_file(safety_source),
        "safety_reports_hash_linked": safety_source_record.get(
            "video_report_sha256"
        )
        == sha256_file(args.safety_video_report)
        and safety_source_record.get("run_report_sha256")
        == sha256_file(args.safety_run_report),
        "reel_video_hash_linked": reel.get("output_video_sha256")
        == sha256_file(output_video),
        "reel_resolution_and_rate": width == 1280
        and height == 720
        and 29.0 <= fps <= 31.0,
        "reel_duration_60_to_90_seconds": 60.0 <= duration_s <= 90.0,
        "timeline_complete_and_contiguous": contiguous
        and timeline_frame_sum == frames
        and named_sections
        == [
            "intro_card",
            "complete_office_mission",
            "dynamic_safety_transition_card",
            "dynamic_safety_encounter",
            "evidence_and_next_gates_card",
        ],
        "accepted_source_frames_preserved": mission_source.get("frames_included")
        == mission.get("presentation_frame_count")
        and safety_source_record.get("frames_included") == safety_video.get("frames")
        and reel.get("motion_changed_by_assembly") is False,
        "claim_boundaries_explicit": "not an as-built survey"
        in reel.get("claim_boundary", "")
        and "physical human-safety claim" in reel.get("claim_boundary", "")
        and "permission to deploy" in reel.get("claim_boundary", ""),
    }
    report = {
        "report_type": "final_omniverse_administration_presentation_acceptance",
        "passed": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "evidence": {
            "reel": str(output_video.resolve()),
            "reel_sha256": sha256_file(output_video),
            "reel_report": str(args.reel_report.resolve()),
            "reel_report_sha256": sha256_file(args.reel_report),
            "mission_report": str(args.mission_report.resolve()),
            "mission_report_sha256": sha256_file(args.mission_report),
            "safety_video_report": str(args.safety_video_report.resolve()),
            "safety_video_report_sha256": sha256_file(args.safety_video_report),
            "safety_run_report": str(args.safety_run_report.resolve()),
            "safety_run_report_sha256": sha256_file(args.safety_run_report),
        },
        "measured_result": {
            "resolution": [width, height],
            "fps": fps,
            "frames": frames,
            "duration_s": duration_s,
            "mission_waypoints_completed": mission_run.get("waypoints_completed"),
            "mission_collisions": int(
                bool(mission_termination.get("dynamic_obstacle_collision"))
                or bool(mission_termination.get("static_collision"))
            ),
            "dynamic_encounter_safety_authority_steps": safety_metrics.get(
                "encounter_safety_authority_steps"
            ),
            "dynamic_encounter_protective_stop_duration_s": safety_metrics.get(
                "protective_full_stop_duration_s"
            ),
            "dynamic_encounter_maximum_resumed_velocity_mps": safety_metrics.get(
                "maximum_resumed_velocity_mps"
            ),
            "dynamic_encounter_collisions": int(
                bool(safety_termination.get("dynamic_obstacle_collision"))
                or bool(safety_termination.get("static_collision"))
            ),
        },
        "decision": (
            "Accept as the final presentation-grade Omniverse reel for the declared "
            "simulation scope. It combines the complete clean office mission with one "
            "accepted deterministic pedestrian encounter without changing source motion. "
            "It is not a physical safety or deployment release."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
