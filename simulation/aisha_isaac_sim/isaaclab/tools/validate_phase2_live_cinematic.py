#!/usr/bin/env python3
"""Validate a policy-only live cinematic administration capture and its edits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from validate_administration_live_policy import (
    sha256_file,
    uniform_occlusion_intervals,
    video_metadata,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-report",
        type=Path,
        default=ROOT / "results" / "phase2_administration_live_cinematic_report.json",
    )
    parser.add_argument(
        "--raw-video",
        type=Path,
        default=(
            ROOT
            / "media"
            / "videos"
            / "phase2_administration_live_cinematic"
            / "aisha-block-a-learned-route-step-0.mp4"
        ),
    )
    parser.add_argument(
        "--presentation-report",
        type=Path,
        default=ROOT / "results" / "phase2_administration_live_cinematic_3x_presentation_report.json",
    )
    parser.add_argument(
        "--presentation-video",
        type=Path,
        default=ROOT / "media" / "videos" / "AI-SHA_Phase2_Administration_Live_Cinematic_3x.mp4",
    )
    parser.add_argument(
        "--teaser-report",
        type=Path,
        default=ROOT / "results" / "phase2_administration_live_cinematic_teaser_12x_report.json",
    )
    parser.add_argument(
        "--teaser-video",
        type=Path,
        default=ROOT / "media" / "videos" / "AI-SHA_Phase2_Administration_Live_Cinematic_Teaser_12x.mp4",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "isaaclab" / "checkpoints" / "aisha_phase2_policy_model_1850.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "phase2_administration_live_cinematic_validation.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    run = read_json(args.run_report)
    presentation = read_json(args.presentation_report)
    teaser = read_json(args.teaser_report)
    build = read_json(ROOT / "results" / "administration_build_report.json")
    live_assets = read_json(ROOT / "results" / "administration_live_assets_report.json")
    raw_metadata = video_metadata(args.raw_video)
    presentation_metadata = video_metadata(args.presentation_video)
    teaser_metadata = video_metadata(args.teaser_video)
    controls = run.get("control_steps", {})
    policy_steps_by_source = controls.get("learned_sensor_policy_by_source", {})
    policy_architecture = run.get("policy_architecture")
    segment_specialists = run.get("segment_policy_checkpoints", {})
    camera = run.get("camera", {})
    shot_definitions = camera.get("shot_definitions", [])
    shot_events = camera.get("shot_events", [])
    pose_trace = run.get("pose_trace", [])
    expected_segments = list(range(12))
    defined_segments = sorted(
        int(segment)
        for shot in shot_definitions
        for segment in shot.get("segments", [])
    )
    event_shots = sorted({int(event.get("shot_index", -1)) for event in shot_events})
    non_initial_occlusions = [
        interval for interval in uniform_occlusion_intervals(args.raw_video) if interval[1] > 0.2
    ]
    scene = ROOT / "scenes" / "administration.usd"
    stage_checks = run.get("live_stage_checks", {})
    telemetry_fields = {
        "linear_velocity_mps",
        "yaw_rate_rad_s",
        "minimum_lidar_range_m",
        "policy_action",
    }
    specialist_hashes_match = bool(segment_specialists) and all(
        Path(details.get("path", "")).is_file()
        and sha256_file(Path(details["path"])) == details.get("sha256")
        for details in segment_specialists.values()
    )

    checks = {
        "upgraded_scene_build_passed": build.get("passed") is True,
        "visual_upgrade_declared": build.get("visual_upgrade", {}).get("version")
        == "administration_walkthrough_procedural_pbr_v1",
        "visual_upgrade_preserves_collision_geometry": build.get("visual_upgrade", {}).get(
            "collision_geometry_changed"
        )
        is False,
        "live_assets_passed": live_assets.get("passed") is True,
        "live_assets_match_current_scene": scene.is_file()
        and live_assets.get("source_scene_sha256") == sha256_file(scene),
        "live_administration_success": run.get("outcome") == "success",
        "all_route_segments_completed": run.get("waypoints_completed") == 12,
        "policy_only_declared": run.get("route_control") == "policy-only",
        "all_actions_from_policy": controls.get("learned_sensor_policy")
        == run.get("completed_steps"),
        "learned_policy_sources_account_for_all_actions": sum(
            int(value) for value in policy_steps_by_source.values()
        )
        == controls.get("learned_sensor_policy"),
        "declared_learned_skill_ensemble": policy_architecture
        == "route_planner_selected_learned_skill_ensemble",
        "segment_specialists_exist_and_match": specialist_hashes_match,
        "zero_turn_supervisor_steps": controls.get("physics_supervisor_turn") == 0,
        "zero_dwell_supervisor_steps": controls.get("presentation_dwell") == 0,
        "no_root_transform_animation": run.get("root_transform_animation") is False,
        "checkpoint_exists_and_matches": args.checkpoint.is_file()
        and sha256_file(args.checkpoint) == run.get("checkpoint_sha256"),
        "live_shell_composed": stage_checks.get("live_shell_body", {}).get("valid") is True
        and stage_checks.get("live_shell_body", {}).get("active") is True,
        "replay_robot_inactive": stage_checks.get("excluded_replay_robot", {}).get("valid") is True
        and stage_checks.get("excluded_replay_robot", {}).get("active") is False,
        "live_robot_active": stage_checks.get("live_robot_base", {}).get("valid") is True
        and stage_checks.get("live_robot_base", {}).get("active") is True,
        "cinematic_camera_mode": camera.get("mode") == "cinematic"
        and run.get("camera_tracking") == "static_segment_cinematic_cameras",
        "six_camera_definitions": len(shot_definitions) == 6,
        "camera_segments_cover_route_once": defined_segments == expected_segments,
        "all_six_cameras_activated": event_shots == [1, 2, 3, 4, 5, 6],
        "telemetry_trace_present": bool(pose_trace)
        and all(telemetry_fields.issubset(sample) for sample in pose_trace),
        "telemetry_trace_policy_only": bool(pose_trace)
        and all(sample.get("control_mode") == "learned_sensor_policy" for sample in pose_trace),
        "telemetry_trace_declares_policy_source": bool(pose_trace)
        and all(sample.get("policy_source") in policy_steps_by_source for sample in pose_trace),
        "raw_video_complete": raw_metadata["resolution"] == [1280, 720]
        and raw_metadata["frame_count"] >= int(run.get("completed_steps", 0)) - 2,
        "no_noninitial_uniform_camera_occlusion": not non_initial_occlusions,
        "presentation_passed": presentation.get("passed") is True,
        "presentation_policy_only": presentation.get("policy_only_control") is True,
        "presentation_live_cinematic_label": presentation.get("camera_mode") == "cinematic",
        "presentation_telemetry_overlay": presentation.get("telemetry_overlay") is True,
        "presentation_source_matches_raw": presentation.get("source_video_sha256")
        == raw_metadata["sha256"],
        "presentation_hash_matches": presentation.get("output_video_sha256")
        == presentation_metadata["sha256"],
        "presentation_motion_unchanged": presentation.get("motion_changed") is False,
        "teaser_passed": teaser.get("passed") is True,
        "teaser_policy_only": teaser.get("policy_only_control") is True,
        "teaser_live_cinematic_label": teaser.get("camera_mode") == "cinematic",
        "teaser_telemetry_overlay": teaser.get("telemetry_overlay") is True,
        "teaser_source_matches_raw": teaser.get("source_video_sha256") == raw_metadata["sha256"],
        "teaser_hash_matches": teaser.get("output_video_sha256") == teaser_metadata["sha256"],
        "teaser_motion_unchanged": teaser.get("motion_changed") is False,
    }
    report = {
        "report_type": "phase2_administration_live_cinematic_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "run_report": str(args.run_report.resolve()),
        "raw_video": raw_metadata,
        "presentation_video": presentation_metadata,
        "teaser_video": teaser_metadata,
        "camera_shot_events": shot_events,
        "non_initial_uniform_occlusion_intervals_s": non_initial_occlusions,
        "claim_boundary": (
            "live learned-skill-ensemble inference, policy-only wheel commands, PhysX contacts and LD19-style "
            "ray observations in the upgraded plan/walkthrough-derived administration USD; not "
            "measured-site, RTX-LD19, Nav2, dynamic-person, sim-to-real or physical-release evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
