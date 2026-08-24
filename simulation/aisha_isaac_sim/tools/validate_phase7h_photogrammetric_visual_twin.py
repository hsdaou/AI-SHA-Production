#!/usr/bin/env python3
"""Validate the Phase 7H photogrammetry-informed Omniverse presentation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from aisha_common import PACKAGE_ROOT, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=PACKAGE_ROOT / "config/phase7h_photogrammetric_visual_twin.yaml",
    )
    parser.add_argument(
        "--acceptance",
        type=Path,
        default=PACKAGE_ROOT / "results/administration_nav2_phase7h_photogrammetric_acceptance.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_info(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open encoded presentation video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0 or frames <= 0:
        raise RuntimeError(f"invalid encoded presentation video metadata: {path}")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps,
    }


def main() -> int:
    args = parse_args()
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    outputs = profile["outputs"]
    reconstruction = profile["reconstruction_evidence"]
    hybrid = profile["hybrid_visual_contract"]
    render_contract = profile["render_contract"]
    material_report_path = PACKAGE_ROOT / profile["source_evidence"]["material_report"]
    render_report_path = PACKAGE_ROOT / outputs["render_report"]
    video_path = PACKAGE_ROOT / outputs["video"]
    contact_sheet = PACKAGE_ROOT / outputs["contact_sheet"]
    scene = PACKAGE_ROOT / "scenes/administration.usd"
    build_report_path = PACKAGE_ROOT / "results/administration_build_report.json"
    material_report = load_json(material_report_path)
    render_report = load_json(render_report_path)
    build_report = load_json(build_report_path)
    encoded = video_info(video_path)

    material_paths = [PACKAGE_ROOT / value for value in hybrid["capture_derived_materials"]]
    report_asset_hashes = {
        item["path"]: item["sha256"]
        for family in material_report["assets"].values()
        for item in family.values()
    }
    build_asset_names = {
        Path(item["path"]).name for item in build_report["visual_upgrade"]["texture_assets"]
    }
    expected_size = render_contract["resolution"]
    disclosure = profile["presentation_disclosures"]["environment"].lower()

    checks = {
        "profile_ready": profile["status"] == "hybrid_visual_twin_ready",
        "phase_is_7h": profile["phase"] == "PHASE7H-PHOTOGRAMMETRY-INFORMED-VISUAL-TWIN",
        "roomplan_remains_metric_authority": "roomplan" in hybrid["metric_local_geometry_authority"].lower(),
        "approved_plan_remains_global_authority": "page 2" in hybrid["global_topology_authority"].lower(),
        "raw_dense_mesh_collision_disabled": hybrid["raw_dense_mesh_collision_enabled"] is False,
        "raw_dense_mesh_excluded_from_hero_render": hybrid["raw_dense_mesh_in_hero_render"] is False,
        "corridor_dense_evidence_present": reconstruction["atrium_corridor_cluster"]["dense_points"] >= 250000,
        "principal_dense_evidence_present": reconstruction["principal_office_cluster"]["dense_points"] >= 200000,
        "principal_textured_mesh_evidence_present": reconstruction["principal_office_cluster"]["textured_faces"] >= 250000,
        "capture_media_not_committed": reconstruction["source_media_committed"] is False,
        "material_report_passed": material_report["status"] == "passed",
        "material_report_is_surface_only": "surface-only" in material_report["source"]["privacy_scope"],
        "all_photo_materials_exist": all(path.is_file() for path in material_paths),
        "all_photo_material_hashes_match": all(
            report_asset_hashes.get(str(path.relative_to(PACKAGE_ROOT))) == sha256_file(path)
            for path in material_paths
        ),
        "build_passed_all_checks": all(build_report["checks"].values()),
        "build_contains_all_photo_materials": all(path.name in build_asset_names for path in material_paths),
        "scene_exists": scene.is_file(),
        "render_uses_current_scene": render_report["scene_sha256"] == sha256_file(scene),
        "render_uses_phase7h_profile": render_report["presentation_profile_sha256"] == sha256_file(args.profile),
        "renderer_is_path_tracing": render_report["renderer"] == "PathTracing",
        "path_tracing_spp_matches": render_report["path_tracing_spp"] == render_contract["path_tracing_spp"],
        "exposure_bias_matches": math.isclose(render_report["exposure_bias"], render_contract["exposure_bias"], abs_tol=1.0e-6),
        "expected_frames_rendered": render_report["frame_count"] == render_contract["expected_frames"],
        "video_full_hd": [encoded["width"], encoded["height"]] == expected_size,
        "video_frame_rate": math.isclose(encoded["fps"], render_contract["fps"], abs_tol=0.01),
        "video_frame_count": encoded["frames"] == render_contract["expected_frames"],
        "contact_sheet_exists": contact_sheet.is_file(),
        "hybrid_claim_is_disclosed": "hybrid" in disclosure and "not a complete" in disclosure,
        "vice_principal_assumption_disclosed": profile["presentation_disclosures"]["vice_principal_interior_assumed_because_locked"] is True,
        "physical_release_false": profile["presentation_disclosures"]["physical_release"] is False,
    }
    passed = sum(checks.values())
    report = {
        "report_type": "administration_nav2_phase7h_photogrammetric_visual_twin_acceptance",
        "status": "accepted" if passed == len(checks) else "rejected",
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "profile": str(args.profile.resolve()),
        "profile_sha256": sha256_file(args.profile),
        "scene": str(scene.resolve()),
        "scene_sha256": sha256_file(scene),
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "video_info": encoded,
        "contact_sheet": str(contact_sheet.resolve()),
        "claim_boundary": {
            "supported": "capture-derived materials and dense-reconstruction evidence in a plan/RoomPlan-authoritative hybrid Omniverse twin",
            "complete_whole_building_photogrammetric_as_built": False,
            "raw_dense_mesh_is_navigation_geometry": False,
            "physical_release": False,
        },
    }
    write_json(args.acceptance, report)
    print(f"Phase 7H validation: {passed}/{len(checks)} checks passed")
    print(f"wrote {args.acceptance}")
    return 0 if report["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
