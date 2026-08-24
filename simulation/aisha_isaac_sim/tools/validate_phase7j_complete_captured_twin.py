#!/usr/bin/env python3
"""Validate the complete captured-administration Phase 7J presentation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from aisha_common import PACKAGE_ROOT, sha256_file, write_json


PROFILE = PACKAGE_ROOT / "config/phase7j_complete_captured_administration_twin.yaml"
BUILD = PACKAGE_ROOT / "results/phase7j_complete_captured_administration_build.json"
STATIC = PACKAGE_ROOT / "results/phase7j_complete_twin_static_render.json"
ROUTE_AUDIT = PACKAGE_ROOT / "results/administration_nav2_phase7i_measured_route_constraint_audit.json"


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
    build = load_json(BUILD)
    static = load_json(STATIC)
    route_audit = load_json(ROUTE_AUDIT)
    outputs = profile["outputs"]
    visual = PACKAGE_ROOT / profile["layer_contract"]["full_capture_visual_layer"]
    scene = PACKAGE_ROOT / profile["layer_contract"]["composite_scene"]
    replay_path = PACKAGE_ROOT / outputs["replay_validation"]
    render_path = PACKAGE_ROOT / outputs["render_report"]
    video_path = PACKAGE_ROOT / outputs["video"]
    contact_path = PACKAGE_ROOT / outputs["contact_sheet"]
    acceptance_path = PACKAGE_ROOT / outputs["acceptance"]
    replay = load_json(replay_path)
    render = load_json(render_path)
    video = media_info(video_path)
    contract = profile["render_contract"]
    layer = profile["layer_contract"]
    disclosure = profile["presentation_disclosures"]
    primary = profile["registration"]["primary"]
    principal = profile["registration"]["principal_supplement"]
    category_counts = build["visual_layer"]["category_counts"]
    static_paths = [Path(shot["path"]) for shot in static["shots"]]

    checks = {
        "profile_ready": profile["status"] == "complete_captured_area_semantic_twin_ready",
        "phase_is_7j": profile["phase"] == "PHASE7J-COMPLETE-CAPTURED-ADMINISTRATION-TWIN",
        "primary_hash_matches": build["sources"]["primary_roomplan_sha256"] == profile["source_evidence"]["primary_roomplan_sha256"],
        "principal_hash_matches": build["sources"]["principal_supplement_sha256"] == profile["source_evidence"]["principal_supplement_sha256"],
        "raw_capture_not_committed": build["sources"]["raw_capture_committed"] is False,
        "metric_primary_registration": math.isclose(primary["metric_scale"], 1.0),
        "metric_principal_registration": math.isclose(principal["metric_scale"], 1.0),
        "hallway_anchor_used": primary["world_anchor_xy_m"] == [4.7, 0.0] and math.isclose(primary["world_yaw_deg"], -3.56),
        "roomplan_floors_registered_to_world_floor": math.isclose(primary["world_z_offset_m"], 1.3561) and math.isclose(principal["world_z_offset_m"], 1.5663),
        "approved_plan_is_global_authority": "page 2" in profile["source_evidence"]["approved_topology"].lower(),
        "complete_primary_capture_declared": layer["complete_primary_capture_included"] is True,
        "principal_supplement_declared": layer["principal_supplement_registered"] is True,
        "visual_collision_layers_separated": layer["visual_and_collision_layers_separate"] is True and build["composite_scene"]["visual_collision_layers_separated"] is True,
        "plan_authority_floor_and_atrium_step_visible": build["composite_scene"]["visible_plan_authority_floor_with_atrium_step_down"] is True,
        "incomplete_roomplan_floor_replaced_only_in_composite": build["composite_scene"]["presentation_hidden_roomplan_floor_count"] >= 1,
        "raw_scan_not_collision": layer["raw_roomplan_mesh_used_for_collision"] is False,
        "visual_layer_flattened": build["visual_layer"]["flattened"] is True,
        "visual_has_no_external_roomplan_dependency": build["visual_layer"]["external_roomplan_dependencies"] is False,
        "semantic_walls_complete": category_counts.get("Wall") == 81,
        "semantic_floors_from_both_scans_present": category_counts.get("Floor") == 2,
        "uuid_metadata_scrubbed": build["visual_layer"]["uuid_fields_scrubbed"] >= 80,
        "full_capture_furniture_retained": build["composite_scene"]["full_capture_layer_retains_all_captured_furniture"] is True,
        "principal_duplicate_furniture_resolved_in_composite": build["composite_scene"]["presentation_hidden_primary_principal_furniture_duplicate_count"] >= 1,
        "plan_authority_visual_route_reconciliation_recorded": build["composite_scene"]["presentation_hidden_static_visual_route_conflict_count"] >= 1,
        "presentation_movable_conflicts_disclosed": build["composite_scene"]["presentation_hidden_movable_route_conflict_count"] >= 1,
        "visual_stage_exists": visual.is_file(),
        "composite_stage_exists": scene.is_file(),
        "build_visual_hash_current": build["visual_layer"]["sha256"] == sha256_file(visual),
        "build_scene_hash_current": build["composite_scene"]["sha256"] == sha256_file(scene),
        "static_render_passed": static["passed"] is True and len(static["shots"]) == 6,
        "static_render_is_path_traced": static["renderer"] == "PathTracing",
        "all_static_views_exist": all(path.is_file() for path in static_paths),
        "accepted_route_audit_preserved": route_audit.get("passed") is True,
        "replay_validation_passed": replay.get("passed") is True,
        "source_motion_was_live_policy": render["claim_boundary"]["source_motion_was_live_policy_execution"] is True,
        "presentation_is_honest_replay": render["claim_boundary"]["visual_replay_is_live_policy_execution"] is False,
        "render_uses_phase7j_scene": render["scene_sha256"] == sha256_file(scene),
        "render_uses_phase7j_profile": render["presentation_profile_sha256"] == sha256_file(PROFILE),
        "renderer_is_path_tracing": render["renderer"] == contract["renderer"],
        "render_settings_match": render["path_tracing_spp"] == contract["path_tracing_spp"] and math.isclose(render["exposure_bias"], contract["exposure_bias"], abs_tol=1e-6),
        "expected_frames_rendered": render["frame_count"] == contract["expected_frames"],
        "principal_visit_cutaway_recorded": render["shots"][6]["cutaway"] is True,
        "video_is_full_hd": [video["width"], video["height"]] == contract["resolution"],
        "video_frame_rate_matches": math.isclose(video["fps"], contract["fps"], abs_tol=0.01),
        "video_frame_count_matches": video["frames"] == contract["expected_frames"],
        "contact_sheet_exists": contact_path.is_file(),
        "vp_locked_assumption_disclosed": disclosure["vice_principal_interior_assumed_because_locked"] is True,
        "semantic_not_photogrammetric_claim": "not a complete phototextured" in disclosure["environment"].lower(),
        "physical_release_false": disclosure["physical_release"] is False,
    }
    passed = sum(checks.values())
    report = {
        "report_type": "administration_nav2_phase7j_complete_captured_twin_acceptance",
        "status": "accepted" if passed == len(checks) else "rejected",
        "passed": passed == len(checks),
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "profile": str(PROFILE.resolve()),
        "profile_sha256": sha256_file(PROFILE),
        "scene": str(scene.resolve()),
        "scene_sha256": sha256_file(scene),
        "visual_layer": str(visual.resolve()),
        "visual_layer_sha256": sha256_file(visual),
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "video_info": video,
        "contact_sheet": str(contact_path.resolve()),
        "claim_boundary": {
            "supported": "complete user-captured RoomPlan semantic administration twin with a registered Principal supplement and accepted mission replay",
            "complete_primary_roomplan_area_included": True,
            "complete_phototextured_as_built": False,
            "visual_replay_is_live_policy_execution": False,
            "physical_release": False,
        },
    }
    write_json(acceptance_path, report)
    print(f"Phase 7J validation: {passed}/{len(checks)} checks passed")
    print(f"wrote {acceptance_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
