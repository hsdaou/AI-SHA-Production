#!/usr/bin/env python3
"""Validate the Phase 7I scan-matched atrium presentation package."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from aisha_common import PACKAGE_ROOT, sha256_file, write_json


PROFILE = PACKAGE_ROOT / "config/phase7i_scan_matched_atrium.yaml"
OVERLAY = PACKAGE_ROOT / "config/measured_administration_presentation_2026-08-23.yaml"
BUILD = PACKAGE_ROOT / "results/administration_build_report.json"
SCENE = PACKAGE_ROOT / "scenes/administration.usd"
STILL = PACKAGE_ROOT / "media/screenshots/administration_atrium.png"


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
    overlay = yaml.safe_load(OVERLAY.read_text(encoding="utf-8"))
    build = load_json(BUILD)
    outputs = profile["outputs"]
    replay_path = PACKAGE_ROOT / outputs["replay_validation"]
    render_path = PACKAGE_ROOT / outputs["render_report"]
    video_path = PACKAGE_ROOT / outputs["video"]
    contact_path = PACKAGE_ROOT / outputs["contact_sheet"]
    acceptance_path = PACKAGE_ROOT / outputs["acceptance"]
    replay = load_json(replay_path)
    render = load_json(render_path)
    video = media_info(video_path)
    still_image = cv2.imread(str(STILL), cv2.IMREAD_COLOR)
    if still_image is None:
        raise RuntimeError(f"cannot open path-traced atrium still: {STILL}")
    still_height, still_width = still_image.shape[:2]
    contract = profile["scan_matched_atrium_contract"]
    render_contract = profile["render_contract"]
    atrium = overlay["measured_visual_twin"]["atrium"]
    required = set(contract["required_visual_features"])
    expected_features = {
        "opposing_glazed_walnut_reception_windows",
        "octagonal_polished_terrazzo_hall",
        "double_black_island_inlay",
        "prohibited_0_20m_lowered_island",
        "paired_slim_black_public_benches",
        "three_privacy_safe_display_easels",
        "four_route_cleared_white_columns",
        "walnut_perimeter_office_fronts",
        "rear_glazed_double_door_and_emblems",
        "white_octagonal_upper_band_and_colour_panels",
        "tall_white_potted_greenery",
    }
    expected_resolution = render_contract["resolution"]
    disclosures = profile["presentation_disclosures"]

    checks = {
        "profile_ready": profile["status"] == "scan_matched_atrium_ready",
        "phase_is_7i": profile["phase"] == "PHASE7I-SCAN-MATCHED-ATRIUM",
        "previous_proxy_explicitly_rejected": contract["previous_proxy_rejected"] is True,
        "roomplan_is_metric_envelope_authority": "roomplan" in contract["metric_envelope_authority"].lower(),
        "walkthrough_is_visual_semantics_authority": "walkthrough" in contract["visual_semantics_authority"].lower(),
        "approved_plan_remains_global_authority": "page 2" in contract["global_topology_authority"].lower(),
        "required_visual_feature_set_complete": required == expected_features,
        "route_critical_collision_geometry_unchanged": contract["route_critical_collision_geometry_changed"] is False,
        "central_drop_and_no_go_unchanged": contract["central_drop_and_no_go_unchanged"] is True,
        "door_colliders_unchanged": contract["door_colliders_unchanged"] is True,
        "column_collision_cores_unchanged": contract["column_collision_cores_unchanged"] is True,
        "learned_trajectory_unchanged": contract["learned_trajectory_changed"] is False,
        "raw_capture_media_not_committed": contract["raw_capture_media_committed"] is False,
        "overlay_is_atrium_revision": "atrium_revision" in atrium["appearance_status"],
        "overlay_uses_opposing_reception_windows": "opposing" in atrium["reception_treatment"],
        "overlay_uses_three_displays_and_benches": "three" in atrium["central_display_treatment"] and "benches" in atrium["central_display_treatment"],
        "overlay_preserves_route_collision": atrium["route_critical_collision_geometry_changed"] is False,
        "build_passed_all_checks": build["passed"] is True and all(build["checks"].values()),
        "scene_exists": SCENE.is_file(),
        "replay_validation_passed": replay["passed"] is True,
        "replay_uses_current_scene": replay["scene_sha256"] == sha256_file(SCENE),
        "render_uses_current_scene": render["scene_sha256"] == sha256_file(SCENE),
        "render_uses_current_profile": render["presentation_profile_sha256"] == sha256_file(PROFILE),
        "renderer_is_path_tracing": render["renderer"] == "PathTracing",
        "render_settings_match": render["path_tracing_spp"] == render_contract["path_tracing_spp"] and math.isclose(render["exposure_bias"], render_contract["exposure_bias"], abs_tol=1.0e-6),
        "expected_frames_rendered": render["frame_count"] == render_contract["expected_frames"],
        "video_is_full_hd": [video["width"], video["height"]] == expected_resolution,
        "video_frame_rate_matches": math.isclose(video["fps"], render_contract["fps"], abs_tol=0.01),
        "video_frame_count_matches": video["frames"] == render_contract["expected_frames"],
        "contact_sheet_exists": contact_path.is_file(),
        "path_traced_atrium_gate_is_full_hd": [still_width, still_height] == expected_resolution,
        "vp_assumption_remains_disclosed": disclosures["vice_principal_interior_assumed_because_locked"] is True,
        "physical_release_remains_false": disclosures["physical_release"] is False,
    }
    passed = sum(checks.values())
    report = {
        "report_type": "administration_nav2_phase7i_scan_matched_atrium_acceptance",
        "status": "accepted" if passed == len(checks) else "rejected",
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "profile": str(PROFILE.resolve()),
        "profile_sha256": sha256_file(PROFILE),
        "scene": str(SCENE.resolve()),
        "scene_sha256": sha256_file(SCENE),
        "path_traced_atrium_gate": str(STILL.resolve()),
        "path_traced_atrium_gate_sha256": sha256_file(STILL),
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "video_info": video,
        "contact_sheet": str(contact_path.resolve()),
        "claim_boundary": {
            "supported": "route-scoped scan/video-matched atrium visual twin with replay of an accepted learned mission",
            "complete_whole_building_as_built": False,
            "visual_replay_is_live_policy_execution": False,
            "physical_release": False,
        },
    }
    write_json(acceptance_path, report)
    print(f"Phase 7I validation: {passed}/{len(checks)} checks passed")
    print(f"wrote {acceptance_path}")
    return 0 if report["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
