#!/usr/bin/env python3
"""Validate the local-only Phase 7M NuRec presentation reel."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "config/phase7m_nurec_presentation_reel.yaml"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_git_tracked(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.parent.parent)
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=ROOT.parent.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    outputs = profile["outputs"]
    registration = load_json(ROOT / outputs["registration"])
    composite = load_json(ROOT / outputs["composite_build"])
    phase7l_acceptance = load_json(ROOT / "results/phase7l_nurec_gaussian_twin_acceptance.json")
    render_path = ROOT / outputs["render_report"]
    encode_path = ROOT / outputs["encode_report"]
    privacy_path = ROOT / outputs["privacy_review"]
    render = load_json(render_path)
    encode = load_json(encode_path)
    privacy = load_json(privacy_path)
    scene = ROOT / profile["source"]["scene"]
    trajectory = ROOT / profile["source"]["trajectory_report"]
    video = ROOT / encode["output"]
    frames = [ROOT / item["path"] for item in render["frames"]]
    metrics = registration["validation"]
    anchor = registration["metric_anchor"]
    shot_counts = {
        shot["id"]: sum(item["shot_id"] == shot["id"] for item in render["frames"])
        for shot in profile["shots"]
    }
    expected_counts = {shot["id"]: int(shot["frame_count"]) for shot in profile["shots"]}
    expected_codes = sorted(profile["privacy"]["selected_camera_codes"])
    actual_codes = sorted(set(render["camera_codes_rendered"]))
    media = encode["media_probe"]
    checks = {
        "profile_validated_local_preview": profile["status"] == "validated_local_presentation_preview",
        "phase_is_7m": profile["phase"] == "PHASE7M-NUREC-PRESENTATION-REEL",
        "phase7l_baseline_still_accepted_43_of_43": phase7l_acceptance.get("passed") is True and phase7l_acceptance["checks_passed"] == 43,
        "registration_passed": registration.get("passed") is True,
        "gravity_axis_sign_resolved_from_camera_up": anchor.get("gravity_axis_sign_resolved") is True and "physical-up" in anchor.get("gravity_axis_sign_method", ""),
        "shared_atrium_median_under_5cm": metrics["shared_atrium_world_residual_median_m"] < 0.05,
        "shared_atrium_p95_under_20cm": metrics["shared_atrium_world_residual_p95_m"] < 0.20,
        "gravity_residual_under_3deg": metrics["gravity_alignment_residual_deg"] < 3.0,
        "principal_turn_anchor_under_1mm": metrics["principal_turn_anchor_residual_m"] < 0.001,
        "reconstructed_ceiling_within_25cm_assumption": metrics["reconstructed_ceiling_height_residual_m"] < 0.25,
        "route_anchor_sets_absolute_scale": anchor["absolute_scale_method"] == "principal_door_to_turn_metric_anchor",
        "composite_build_passed": composite.get("passed") is True and all(composite["checks"].values()),
        "composite_scene_hash_current": composite["scene_sha256"] == sha256_file(scene),
        "render_passed": render.get("passed") is True,
        "render_uses_current_scene": render["scene_sha256"] == sha256_file(scene),
        "render_uses_accepted_trajectory": render["trajectory_report_sha256"] == sha256_file(trajectory),
        "render_is_full_hd": render["resolution"] == [1920, 1080],
        "render_has_198_frames": len(render["frames"]) == 198,
        "shot_frame_counts_match_profile": shot_counts == expected_counts,
        "privacy_safe_camera_codes_only": actual_codes == expected_codes,
        "all_frames_exist_and_hash_match": all(path.is_file() and sha256_file(path) == item["sha256"] for path, item in zip(frames, render["frames"])),
        "all_frames_are_nonblank": all(max(item["std_rgb"]) > 5.0 for item in render["frames"]),
        "recorded_poses_are_not_interpolated": render["recorded_pose_selection_without_interpolation"] is True,
        "source_motion_came_from_live_nav2_and_learned_safety": render["source_motion_was_live_nav2_and_learned_safety"] is True,
        "renderer_did_not_execute_policy_live": render["presentation_renderer_executes_policy_live"] is False,
        "collision_geometry_unchanged": render["navigation_collision_geometry_changed"] is False,
        "presentation_retiming_disclosed": profile["motion"]["presentation_retimed"] is True and encode["presentation_retimed"] is True,
        "encode_passed": encode.get("passed") is True,
        "encode_uses_current_profile": encode["profile_sha256"] == sha256_file(PROFILE),
        "encode_uses_current_render_report": encode["render_report_sha256"] == sha256_file(render_path),
        "video_exists_and_hash_matches": video.is_file() and sha256_file(video) == encode["output_sha256"],
        "video_is_full_hd_24fps": media["width"] == 1920 and media["height"] == 1080 and abs(media["fps"] - 24.0) < 0.01,
        "video_duration_is_13_75s": abs(media["duration_s"] - 13.75) < 0.05,
        "video_is_local_only_untracked": video.is_file() and not is_git_tracked(video),
        "render_frames_are_local_only_untracked": all(not is_git_tracked(path) for path in frames),
        "privacy_local_preview_passed": privacy["passed_local_preview"] is True,
        "privacy_review_uses_current_encode": privacy["encode_report_sha256"] == sha256_file(encode_path),
        "no_live_people_or_close_sensitive_views_observed": privacy["observations"]["live_people_observed"] is False and privacy["observations"]["close_certificate_views_included"] is False and privacy["observations"]["close_portrait_views_included"] is False,
        "human_external_review_still_required": privacy["authorized_human_privacy_review_completed"] is False and privacy["external_distribution_approved"] is False and privacy["external_distribution_requires_user_review"] is True,
        "nurec_not_used_for_collision_or_lidar": profile["presentation_disclosures"]["nurec_used_for_collision_or_lidar"] is False,
        "vice_principal_not_claimed_captured": profile["presentation_disclosures"]["vice_principal_interior_captured"] is False,
        "physical_release_false": profile["presentation_disclosures"]["physical_release"] is False and encode["physical_release"] is False,
    }
    passed_count = sum(checks.values())
    report = {
        "report_type": "phase7m_nurec_presentation_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_local_preview" if passed_count == len(checks) else "rejected",
        "passed": passed_count == len(checks),
        "checks": checks,
        "checks_passed": passed_count,
        "checks_total": len(checks),
        "profile": "config/phase7m_nurec_presentation_reel.yaml",
        "profile_sha256": sha256_file(PROFILE),
        "scene_sha256": sha256_file(scene),
        "video": encode["output"],
        "video_sha256": encode["output_sha256"],
        "presentation": {
            "resolution_px": [1920, 1080],
            "fps": 24,
            "duration_s": media["duration_s"],
            "recorded_pose_replay": True,
            "source_policy_executed_live_during_render": False,
            "presentation_retimed": True,
        },
        "privacy": {
            "local_preview_accepted": privacy["passed_local_preview"],
            "external_distribution_approved": False,
            "authorized_human_review_required": True,
        },
        "physical_release": False,
    }
    output = ROOT / outputs["acceptance"]
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Phase 7M validation: {passed_count}/{len(checks)} checks passed")
    print(f"wrote {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
