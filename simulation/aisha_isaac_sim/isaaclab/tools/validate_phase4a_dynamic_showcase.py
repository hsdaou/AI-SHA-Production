#!/usr/bin/env python3
"""Validate the evidence chain for the Phase 4A administration showcase."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--video-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_report.read_text(encoding="utf-8"))
    video = json.loads(args.video_report.read_text(encoding="utf-8"))
    raw_video = Path(video["source_video"])
    final_video = Path(video["output_video"])
    if not raw_video.is_file() or not final_video.is_file():
        raise FileNotFoundError("a Phase 4A source or presentation video is missing")

    metrics = run.get("metrics", {})
    termination = run.get("termination") or {}
    policy = run.get("policy_contract", {})
    scenario = run.get("scenario", {})
    checks = {
        "run_report_type": run.get("report_type")
        == "phase4a_live_dynamic_safety_showcase",
        "video_report_type": video.get("report_type")
        == "phase4a_dynamic_safety_presentation_video",
        "run_passed": run.get("passed") is True and run.get("outcome") == "success",
        "run_checks_all_passed": all(run.get("checks", {}).values()),
        "video_passed": video.get("passed") is True
        and all(video.get("checks", {}).values()),
        "accepted_phase3n_checkpoint": run.get("checkpoint", {}).get("sha256")
        == "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b",
        "formal_phase3n_contract_unchanged": scenario.get(
            "formal_phase3n_evaluation_contract_changed"
        )
        is False,
        "segment_7_principal_approach": scenario.get("route_segment_id") == 7,
        "learned_authority_during_crossing": metrics.get(
            "encounter_safety_authority_steps", 0
        )
        > 0,
        "learned_brake_output_observed": metrics.get(
            "maximum_learned_brake_fraction", 0.0
        )
        >= 0.02,
        "physical_stop_observed": metrics.get(
            "protective_full_stop_duration_s", 0.0
        )
        >= 0.5,
        "forward_motion_resumed": metrics.get("maximum_resumed_velocity_mps", 0.0)
        >= 0.25,
        "zero_dynamic_collision": termination.get("dynamic_obstacle_collision")
        is False,
        "zero_static_collision": termination.get("static_collision") is False,
        "no_supervisor": policy.get("physics_supervisor") is False,
        "no_root_animation": policy.get("root_transform_animation") is False,
        "no_privileged_pedestrian_state": policy.get(
            "pedestrian_state_exposed_to_policy"
        )
        is False,
        "run_report_hash_linked": video.get("source_run_report_sha256")
        == sha256_file(args.run_report),
        "raw_video_hash_linked": video.get("source_video_sha256")
        == sha256_file(raw_video),
        "presentation_video_hash_linked": video.get("output_video_sha256")
        == sha256_file(final_video),
        "presentation_resolution_1280x720": video.get("resolution") == [1280, 720],
        "learned_and_protective_states_labeled_separately": (
            video.get("encounter_learned_authority_overlay_frames", 0) > 0
            and video.get("protective_stop_overlay_frames", 0) > 0
            and "does not attribute the entire stop" in video.get(
                "overlay_disclosure", ""
            )
        ),
    }
    report = {
        "report_type": "phase4a_dynamic_safety_showcase_acceptance",
        "passed": all(checks.values()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "evidence": {
            "run_report": str(args.run_report.resolve()),
            "run_report_sha256": sha256_file(args.run_report),
            "video_report": str(args.video_report.resolve()),
            "video_report_sha256": sha256_file(args.video_report),
            "raw_video": str(raw_video.resolve()),
            "raw_video_sha256": sha256_file(raw_video),
            "presentation_video": str(final_video.resolve()),
            "presentation_video_sha256": sha256_file(final_video),
        },
        "measured_result": {
            "completed_steps": run.get("completed_steps"),
            "simulated_duration_s": run.get("duration_s"),
            "encounter_safety_authority_steps": metrics.get(
                "encounter_safety_authority_steps"
            ),
            "maximum_learned_brake_fraction": metrics.get(
                "maximum_learned_brake_fraction"
            ),
            "protective_full_stop_duration_s": metrics.get(
                "protective_full_stop_duration_s"
            ),
            "maximum_resumed_velocity_mps": metrics.get(
                "maximum_resumed_velocity_mps"
            ),
            "minimum_robot_pedestrian_centre_distance_m": metrics.get(
                "minimum_robot_pedestrian_centre_distance_m"
            ),
            "minimum_360_ring_clearance_m": metrics.get(
                "minimum_360_ring_clearance_m"
            ),
            "dynamic_collisions": int(
                bool(termination.get("dynamic_obstacle_collision"))
            ),
            "static_collisions": int(bool(termination.get("static_collision"))),
        },
        "decision": (
            "Accept as a presentation-grade live checkpoint demonstration of one "
            "deterministic pedestrian encounter in the administration scene. This "
            "does not replace randomized Phase 3N evaluation or permit a physical "
            "human-safety claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
