#!/usr/bin/env python3
"""Validate the local-only Phase 7L NuRec Gaussian presentation composite."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "config/phase7l_nurec_gaussian_twin.yaml"


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
    main_preflight = load_json(ROOT / outputs["main_preflight"])
    principal_preflight = load_json(ROOT / outputs["principal_preflight"])
    connector_preflight = load_json(ROOT / outputs["connector_preflight"])
    training = load_json(ROOT / outputs["training_report"])
    registration = load_json(ROOT / outputs["registration_report"])
    build = load_json(ROOT / outputs["composite_build"])
    main_render = load_json(ROOT / outputs["main_isaac_render"])
    principal_render = load_json(ROOT / outputs["principal_isaac_render"])
    composite_render = load_json(ROOT / outputs["composite_isaac_render"])
    live_smoke = load_json(ROOT / outputs["live_player_smoke"])
    scene = ROOT / profile["composition"]["scene"]

    components = training["components"]
    asset_paths = [ROOT / item["asset"]["path"] for item in components.values()]
    checkpoint_paths = [
        ROOT / item["checkpoint"]["path"] for item in components.values()
    ]
    expected_assets = {
        ROOT / item["asset"]["path"]: item["asset"]["sha256"]
        for item in components.values()
    }
    expected_checkpoints = {
        ROOT / item["checkpoint"]["path"]: item["checkpoint"]["sha256"]
        for item in components.values()
    }
    registration_metrics = registration["validation"]
    layer = build["layer_contract"]
    disclosure = profile["presentation_disclosures"]
    scene_text = scene.read_text(encoding="utf-8")

    checks = {
        "profile_ready": profile["status"] == "registered_nurec_composite_validated",
        "phase_is_7l": profile["phase"] == "PHASE7L-NUREC-GAUSSIAN-TWIN",
        "dataset_preflights_passed": all(
            item.get("passed") is True
            for item in (main_preflight, principal_preflight, connector_preflight)
        ),
        "main_dataset_has_355_registered_images": main_preflight["source"]["registered_images"] == 355,
        "principal_dataset_has_122_registered_images": principal_preflight["source"]["registered_images"] == 122,
        "training_passed": training.get("passed") is True,
        "nvidia_3dgrut_commit_recorded": training["engine"]["commit"] == "a37ef721012dea0f29c0fcfff2d525023b4e854a",
        "both_components_trained_30000_iterations": all(
            item["iterations"] == 30000 for item in components.values()
        ),
        "holdout_metrics_are_finite_and_positive": all(
            item["metrics"]["mean_psnr"] > 0
            and 0 < item["metrics"]["mean_ssim"] <= 1
            and item["metrics"]["mean_lpips"] >= 0
            for item in components.values()
        ),
        "local_assets_exist": all(path.is_file() for path in asset_paths),
        "local_asset_hashes_match": all(
            path.is_file() and sha256_file(path) == expected
            for path, expected in expected_assets.items()
        ),
        "local_checkpoints_exist": all(path.is_file() for path in checkpoint_paths),
        "local_checkpoint_hashes_match": all(
            path.is_file() and sha256_file(path) == expected
            for path, expected in expected_checkpoints.items()
        ),
        "privacy_assets_are_not_git_tracked": all(
            not is_git_tracked(path) for path in asset_paths + checkpoint_paths
        ),
        "registration_passed": registration.get("passed") is True,
        "shared_atrium_median_under_5cm": registration_metrics["shared_atrium_world_residual_median_m"] < 0.05,
        "shared_atrium_p95_under_20cm": registration_metrics["shared_atrium_world_residual_p95_m"] < 0.20,
        "gravity_residual_under_3deg": registration_metrics["gravity_alignment_residual_deg"] < 3.0,
        "registration_not_claimed_as_certified": "not a certified survey" in registration["layer_contract"]["registration_classification"],
        "composite_build_passed": build.get("passed") is True and all(build["checks"].values()),
        "composite_scene_hash_current": build["scene_sha256"] == sha256_file(scene),
        "composite_native_render_passed": composite_render.get("passed") is True,
        "composite_render_uses_current_scene": composite_render["stage_sha256"] == sha256_file(scene),
        "live_player_smoke_completed": live_smoke["status"] == "completed_requested_loops" and live_smoke["frames_presented"] > 0,
        "live_player_uses_current_scene": live_smoke["scene_sha256"] == sha256_file(scene),
        "live_player_uses_principal_route_segments": live_smoke["segments"] == [6, 7, 8, 9],
        "live_player_is_honest_recorded_pose_replay": live_smoke["recorded_pose_selection_without_interpolation"] is True and live_smoke["presentation_player_executes_policy_live"] is False,
        "live_player_preserves_collision_world": live_smoke["navigation_collision_geometry_changed"] is False,
        "main_native_render_passed": main_render.get("passed") is True,
        "principal_native_render_passed": principal_render.get("passed") is True,
        "main_render_uses_trained_asset": main_render["stage_sha256"] == components["main_administration"]["asset"]["sha256"],
        "principal_render_uses_trained_asset": principal_render["stage_sha256"] == components["principal_office"]["asset"]["sha256"],
        "nurec_stays_in_native_basis": layer["nurec_asset_transform"] == "identity_native_training_basis",
        "legacy_render_geometry_hidden": build["checks"]["legacy_render_geometry_hidden"] is True,
        "navigation_collision_unchanged": layer["navigation_collision_geometry_changed"] is False,
        "gaussians_are_visual_only": layer["gaussians_visual_only"] is True,
        "gaussians_not_used_for_lidar_or_collision": layer["raw_gaussians_used_for_lidar_or_collision"] is False,
        "scene_uses_relative_local_asset_references": "@../tmp/phase7l_nurec_runs/" in scene_text,
        "principal_is_captured": disclosure["principal_office_captured"] is True,
        "vp_locked_assumption_disclosed": disclosure["vice_principal_interior_assumed_because_locked"] is True,
        "visual_replay_not_claimed_live": disclosure["visual_replay_is_live_policy_execution"] is False,
        "physical_release_false": disclosure["physical_release"] is False,
        "privacy_review_required": disclosure["privacy_review_required_before_external_media_distribution"] is True,
    }
    passed_count = sum(checks.values())
    report = {
        "report_type": "phase7l_nurec_gaussian_twin_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if passed_count == len(checks) else "rejected",
        "passed": passed_count == len(checks),
        "checks": checks,
        "checks_passed": passed_count,
        "checks_total": len(checks),
        "profile": "config/phase7l_nurec_gaussian_twin.yaml",
        "profile_sha256": sha256_file(PROFILE),
        "scene": profile["composition"]["scene"],
        "scene_sha256": sha256_file(scene),
        "registration_summary": {
            "shared_atrium_median_residual_m": registration_metrics[
                "shared_atrium_world_residual_median_m"
            ],
            "shared_atrium_p95_residual_m": registration_metrics[
                "shared_atrium_world_residual_p95_m"
            ],
            "gravity_residual_deg": registration_metrics[
                "gravity_alignment_residual_deg"
            ],
            "certified_survey_control": False,
        },
        "privacy": {
            "raw_capture_committed": False,
            "nurec_assets_committed": False,
            "checkpoints_committed": False,
            "external_media_requires_review": True,
        },
        "claim_boundary": {
            "gaussian_visual_layer_only": True,
            "visual_replay_is_live_policy_execution": False,
            "physical_release": False,
        },
    }
    output = ROOT / outputs["acceptance"]
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Phase 7L validation: {passed_count}/{len(checks)} checks passed")
    print(f"wrote {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
