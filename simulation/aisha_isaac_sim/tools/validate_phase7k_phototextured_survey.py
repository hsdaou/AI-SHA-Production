#!/usr/bin/env python3
"""Validate the Phase 7K hybrid phototextured photogrammetric survey release."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from aisha_common import PACKAGE_ROOT, sha256_file, write_json


PROFILE = PACKAGE_ROOT / "config/phase7k_phototextured_photogrammetric_survey.yaml"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def media_info(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open media: {path}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return result


def main() -> int:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    outputs = profile["outputs"]
    material = load_json(PACKAGE_ROOT / outputs["material_report"])
    privacy = load_json(PACKAGE_ROOT / outputs["privacy_manifest"])
    dense = load_json(PACKAGE_ROOT / outputs["dense_usd_build"])
    build = load_json(PACKAGE_ROOT / outputs["survey_build"])
    static = load_json(PACKAGE_ROOT / outputs["static_render_report"])
    replay = load_json(PACKAGE_ROOT / outputs["replay_validation"])
    render = load_json(PACKAGE_ROOT / outputs["render_report"])
    video_path = PACKAGE_ROOT / outputs["video"]
    contact_path = PACKAGE_ROOT / outputs["contact_sheet"]
    acceptance_path = PACKAGE_ROOT / outputs["acceptance"]
    video = media_info(video_path)
    layer = profile["layer_contract"]
    contract = profile["render_contract"]
    disclosure = profile["presentation_disclosures"]
    survey_scene = PACKAGE_ROOT / layer["survey_review_scene"]
    presentation_scene = PACKAGE_ROOT / layer["presentation_scene"]
    visual = PACKAGE_ROOT / layer["metric_visual_layer"]
    atrium_dense = PACKAGE_ROOT / layer["atrium_dense_layer"]
    principal_dense = PACKAGE_ROOT / layer["principal_dense_layer"]

    texture_maps = [
        PACKAGE_ROOT / value["path"]
        for asset in material["assets"].values()
        for value in asset["maps"].values()
    ]
    screened_atlases = [
        PACKAGE_ROOT / atlas["output"]
        for cluster in privacy["clusters"].values()
        for atlas in cluster["presentation_atlases"].values()
    ]
    ocr_results = [
        atlas["readable_ocr_text_after_screen"]
        for cluster in privacy["clusters"].values()
        for atlas in cluster["presentation_atlases"].values()
    ]
    static_paths = [Path(shot["path"]) for shot in static["shots"]]

    checks = {
        "profile_ready": profile["status"] == "hybrid_phototextured_survey_ready",
        "phase_is_7k": profile["phase"] == "PHASE7K-PHOTOTEXTURED-PHOTOGRAMMETRIC-SURVEY",
        "material_generation_passed": material.get("passed") is True,
        "seven_capture_material_sets": len(material["assets"]) == 7,
        "twenty_one_pbr_maps_exist": len(texture_maps) == 21 and all(path.is_file() for path in texture_maps),
        "surface_crops_only": "surface-only" in material["privacy_scope"],
        "source_stills_not_committed": material["source_stills_committed"] is False,
        "privacy_manifest_passed": privacy.get("passed") is True,
        "four_screened_atlases_exist": len(screened_atlases) == 4 and all(path.is_file() for path in screened_atlases),
        "screened_atlases_are_ocr_negative": not any(ocr_results),
        "original_atlases_not_committed": all(not cluster["source_atlases_committed"] for cluster in privacy["clusters"].values()),
        "alicevision_engine_recorded": "AliceVision" in privacy["engine"],
        "dense_usd_build_passed": dense.get("passed") is True,
        "two_dense_clusters": len(dense["outputs"]) == 2,
        "dense_vertex_total": dense["total_vertices"] == 311750,
        "dense_face_total": dense["total_faces"] == 625295,
        "clusters_not_false_welded": dense["clusters_kept_separate"] is True,
        "dense_clusters_metric": all(math.isclose(item["metric_scale"], 1.0) for item in dense["outputs"].values()),
        "dense_clusters_have_no_collision": all(item["collision_enabled"] is False for item in dense["outputs"].values()),
        "dense_assets_exist": atrium_dense.is_file() and principal_dense.is_file(),
        "survey_build_passed": build.get("passed") is True,
        "metric_visual_exists": visual.is_file(),
        "survey_review_scene_exists": survey_scene.is_file(),
        "presentation_scene_exists": presentation_scene.is_file(),
        "survey_hash_current": build["composite_scene"]["sha256"] == sha256_file(survey_scene),
        "presentation_hash_current": build["presentation_scene"]["sha256"] == sha256_file(presentation_scene),
        "normal_maps_connected": build["visual_layer"]["normal_maps_connected"] == 5,
        "seven_material_sets_recorded": build["capture_materials"]["asset_sets"] == 7,
        "survey_clusters_visible": build["composite_scene"]["clusters_visible"] is True,
        "presentation_clusters_hidden": build["presentation_scene"]["dense_clusters_present_but_hidden"] is True,
        "visual_collision_separation_preserved": build["composite_scene"]["visual_collision_layers_separated"] is True,
        "navigation_collision_unchanged": build["frozen_safety_contract"]["navigation_collision_geometry_changed"] is False,
        "raw_dense_not_collision": build["frozen_safety_contract"]["raw_dense_mesh_used_for_collision"] is False,
        "static_render_passed": static.get("passed") is True and len(static["shots"]) == 7,
        "static_render_path_traced": static["renderer"] == "PathTracing",
        "raw_evidence_views_present": static["raw_dense_evidence_views"] == 2,
        "clean_views_present": static["clean_presentation_views"] == 5,
        "all_static_views_exist": all(path.is_file() for path in static_paths),
        "replay_validation_passed": replay.get("passed") is True,
        "render_uses_presentation_scene": render["scene_sha256"] == sha256_file(presentation_scene),
        "render_uses_phase7k_profile": render["presentation_profile_sha256"] == sha256_file(PROFILE),
        "source_motion_was_live_policy": render["claim_boundary"]["source_motion_was_live_policy_execution"] is True,
        "presentation_is_honest_replay": render["claim_boundary"]["visual_replay_is_live_policy_execution"] is False,
        "path_tracing_contract_matches": render["renderer"] == contract["renderer"] and render["path_tracing_spp"] == contract["path_tracing_spp"],
        "expected_frames_rendered": render["frame_count"] == contract["expected_frames"],
        "video_is_full_hd": [video["width"], video["height"]] == contract["resolution"],
        "video_frame_rate_matches": math.isclose(video["fps"], contract["fps"], abs_tol=0.01),
        "video_frame_count_matches": video["frames"] == contract["expected_frames"],
        "contact_sheet_exists": contact_path.is_file(),
        "hybrid_claim_supported": "Hybrid metric phototextured survey" in disclosure["supported_claim"],
        "complete_mesh_not_claimed": disclosure["complete_monolithic_photogrammetric_mesh"] is False,
        "certified_control_not_claimed": disclosure["registration_is_certified_survey_control"] is False,
        "vp_locked_assumption_disclosed": disclosure["vice_principal_interior_assumed_because_locked"] is True,
        "physical_release_false": disclosure["physical_release"] is False,
    }
    passed = sum(checks.values())
    report = {
        "report_type": "administration_nav2_phase7k_phototextured_survey_acceptance",
        "status": "accepted" if passed == len(checks) else "rejected",
        "passed": passed == len(checks),
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "profile": str(PROFILE.resolve()),
        "profile_sha256": sha256_file(PROFILE),
        "survey_scene": str(survey_scene.resolve()),
        "survey_scene_sha256": sha256_file(survey_scene),
        "presentation_scene": str(presentation_scene.resolve()),
        "presentation_scene_sha256": sha256_file(presentation_scene),
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "video_info": video,
        "contact_sheet": str(contact_path.resolve()),
        "dense_geometry": {"vertices": dense["total_vertices"], "faces": dense["total_faces"], "clusters": 2},
        "claim_boundary": {
            "supported": disclosure["supported_claim"],
            "complete_monolithic_photogrammetric_mesh": False,
            "registration_is_certified_survey_control": False,
            "visual_replay_is_live_policy_execution": False,
            "physical_release": False,
        },
    }
    write_json(acceptance_path, report)
    print(f"Phase 7K validation: {passed}/{len(checks)} checks passed")
    print(f"wrote {acceptance_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
