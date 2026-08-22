#!/usr/bin/env python3
"""Validate Phase 2 policy-only evaluation and administration evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from validate_administration_live_policy import sha256_file, uniform_occlusion_intervals, video_metadata


ROOT = Path(__file__).resolve().parents[2]
TURN_EVALUATION = ROOT / "results" / "phase2_turn_held_out_evaluation.json"
ROUTE_EVALUATION = ROOT / "results" / "phase2_policy_only_route_evaluation.json"
TRAINING_ROUTE = ROOT / "results" / "phase2_policy_only_training_route_report.json"
LIVE_RUN = ROOT / "results" / "phase2_administration_policy_only_video_report.json"
PRESENTATION_REPORT = ROOT / "results" / "phase2_administration_policy_only_presentation_report.json"
RAW_VIDEO = (
    ROOT
    / "media"
    / "videos"
    / "phase2_administration_policy_only"
    / "aisha-block-a-learned-route-step-0.mp4"
)
PRESENTATION_VIDEO = ROOT / "media" / "videos" / "AI-SHA_Phase2_Administration_Policy_Only_3x.mp4"
PACKAGED_CHECKPOINT = ROOT / "isaaclab" / "checkpoints" / "aisha_phase2_policy_model_1850.pt"
OUTPUT = ROOT / "results" / "phase2_end_to_end_validation.json"


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    turn = read_json(TURN_EVALUATION)
    route = read_json(ROUTE_EVALUATION)
    training_route = read_json(TRAINING_ROUTE)
    live = read_json(LIVE_RUN)
    presentation = read_json(PRESENTATION_REPORT)
    raw_metadata = video_metadata(RAW_VIDEO)
    presentation_metadata = video_metadata(PRESENTATION_VIDEO)
    controls = live.get("control_steps", {})
    recorded_checkpoint = Path(str(live["checkpoint"]))
    checkpoint = recorded_checkpoint if recorded_checkpoint.is_file() else PACKAGED_CHECKPOINT
    non_initial_occlusions = [
        interval for interval in uniform_occlusion_intervals(RAW_VIDEO) if interval[1] > 0.2
    ]

    checks = {
        "turn_gate_passed": turn.get("acceptance_gate", {}).get("passed") is True,
        "full_route_gate_passed": route.get("acceptance_gate", {}).get("passed") is True,
        "training_course_policy_only_success": training_route.get("outcome") == "success"
        and training_route.get("route_control") == "policy-only",
        "live_administration_success": live.get("outcome") == "success",
        "all_live_segments_completed": live.get("waypoints_completed") == 12,
        "policy_only_declared": live.get("route_control") == "policy-only",
        "zero_turn_supervisor_steps": controls.get("physics_supervisor_turn") == 0,
        "zero_dwell_supervisor_steps": controls.get("presentation_dwell") == 0,
        "all_actions_from_policy": controls.get("learned_sensor_policy") == live.get("completed_steps"),
        "no_root_transform_animation": live.get("root_transform_animation") is False,
        "checkpoint_exists_and_matches": checkpoint.is_file()
        and sha256_file(checkpoint) == live.get("checkpoint_sha256"),
        "same_checkpoint_across_gates": turn.get("checkpoint", {}).get("sha256")
        == route.get("checkpoint", {}).get("sha256")
        == live.get("checkpoint_sha256"),
        "wider_camera_requested": live.get("camera", {}).get("eye") == [-3.8, 0.0, 2.4]
        and live.get("camera", {}).get("lookat") == [0.45, 0.0, 0.55]
        and live.get("camera", {}).get("clearance_adaptation")
        == "three_ray_route_leg_visibility_fan_with_0.20_m_buffer",
        "raw_video_complete": raw_metadata["frame_count"] >= live.get("completed_steps", 0) - 2,
        "no_noninitial_uniform_camera_occlusion": not non_initial_occlusions,
        "presentation_passed": presentation.get("passed") is True,
        "presentation_policy_only_label": presentation.get("policy_only_control") is True,
        "presentation_motion_unchanged": presentation.get("motion_changed") is False,
        "presentation_video_hash_matches": presentation.get("output_video_sha256")
        == presentation_metadata["sha256"],
    }
    report = {
        "report_type": "phase2_end_to_end_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "raw_video": raw_metadata,
        "presentation_video": presentation_metadata,
        "recorded_checkpoint": str(recorded_checkpoint),
        "validated_checkpoint": str(checkpoint.resolve()),
        "non_initial_uniform_occlusion_intervals_s": non_initial_occlusions,
        "claim_boundary": (
            "policy-only control in the declared simulation; not Nav2, measured-site, "
            "dynamic-person, RTX-sensor or sim-to-real validation"
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "output": str(OUTPUT)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
