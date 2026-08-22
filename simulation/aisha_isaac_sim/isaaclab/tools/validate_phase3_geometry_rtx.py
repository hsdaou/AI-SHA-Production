#!/usr/bin/env python3
"""Validate plan-dimension anchors and RTX PBR material refinement in USD."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade


ROOT = Path(__file__).resolve().parents[2]
PLAN = Path("/home/robot-wst/Downloads/DownloadBuildingRequestApprovedPlan.pdf")
SCENE = ROOT / "scenes" / "administration.usd"
REFINEMENT = ROOT / "config" / "geometry_rtx_refinement.yaml"
ASSUMPTIONS = ROOT / "config" / "administration_assumptions.yaml"
BUILD_REPORT = ROOT / "results" / "administration_build_report.json"
OUTPUT = ROOT / "results" / "phase3_geometry_rtx_refinement_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cube_scale(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f"missing cube {path}")
    for operation in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if operation.GetOpType() == UsdGeom.XformOp.TypeScale:
            value = operation.Get()
            return float(value[0]), float(value[1]), float(value[2])
    raise ValueError(f"cube has no scale operation: {path}")


def material_has_normal(stage: Usd.Stage, name: str, filename: str) -> bool:
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"/World/Looks/{name}/Shader"))
    normal = shader.GetInput("normal")
    if not normal or not normal.HasConnectedSource():
        return False
    texture = stage.GetPrimAtPath(f"/World/Looks/{name}/Normal")
    file_input = UsdShade.Shader(texture).GetInput("file") if texture else None
    value = file_input.Get() if file_input else None
    return value is not None and str(value.path).endswith(filename)


def main() -> int:
    refinement = yaml.safe_load(REFINEMENT.read_text(encoding="utf-8"))
    assumptions = yaml.safe_load(ASSUMPTIONS.read_text(encoding="utf-8"))
    build = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(SCENE))
    if stage is None:
        raise RuntimeError(f"could not open {SCENE}")

    printed = refinement["printed_dimensions"]
    atrium_points = UsdGeom.Mesh(
        stage.GetPrimAtPath("/World/Architecture/Floors/Atrium")
    ).GetPointsAttr().Get()
    atrium_diameter = max(
        math.dist((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        for a in atrium_points
        for b in atrium_points
    )
    east_hall_scale = cube_scale(stage, "/World/Architecture/Floors/EastHallway")
    principal_scale = cube_scale(stage, "/World/Architecture/Floors/Principal")
    tolerance = float(refinement["scene_mapping"]["dimension_tolerance_m"])
    root_data = stage.GetRootLayer().customLayerData
    normal_files = {
        "TerrazzoFinish": "terrazzo_normal.png",
        "WalnutFinish": "walnut_normal.png",
        "OakFinish": "oak_normal.png",
        "MottledGreyFinish": "mottled_grey_normal.png",
    }
    checks = {
        "source_pdf_present": PLAN.is_file(),
        "source_pdf_hash_locked": PLAN.is_file() and sha256(PLAN) == refinement["source"]["sha256"],
        "source_is_page_2_block_a": refinement["source"]["page"] == 2
        and refinement["source"]["block"] == "A",
        "stage_reopens": stage is not None,
        "stage_uses_metres": math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, abs_tol=1.0e-9),
        "atrium_matches_printed_12_75_m": math.isclose(
            atrium_diameter,
            float(printed["atrium_diagonal_m"]["value"]),
            abs_tol=tolerance,
        ),
        "hallway_matches_printed_2_80_m": math.isclose(
            east_hall_scale[1],
            float(printed["administration_hallway_clear_width_m"]["value"]),
            abs_tol=tolerance,
        ),
        "principal_frontage_matches_printed_4_73_m": math.isclose(
            principal_scale[0],
            float(printed["principal_diagonal_frontage_m"]["value"]),
            abs_tol=tolerance,
        ),
        "door_widths_remain_disclosed_assumptions": all(
            "assumption" in door["width_status"] for door in assumptions["doors"].values()
        ),
        "thresholds_remain_disclosed_assumptions": all(
            "assumption" in door["threshold_status"] for door in assumptions["doors"].values()
        ),
        "geometry_refinement_tagged_in_usd": root_data.get("aisha:geometryRefinement")
        == refinement["revision"],
        "rtx_material_version_tagged_in_usd": root_data.get("aisha:rtxMaterialRefinement")
        == "administration_rtx_pbr_v2",
        "all_four_materials_have_normal_maps": all(
            material_has_normal(stage, material, filename)
            for material, filename in normal_files.items()
        ),
        "all_twelve_pbr_textures_exist": all(
            (ROOT / "textures" / "administration" / f"{family}_{map_name}.png").is_file()
            for family in ("terrazzo", "walnut", "oak", "mottled_grey")
            for map_name in ("albedo", "roughness", "normal")
        ),
        "build_report_records_visual_only_impact": build["visual_upgrade"]["collision_geometry_changed"]
        is False,
        "build_report_records_rtx_v2": build["visual_upgrade"]["rtx_material_version"]
        == "administration_rtx_pbr_v2",
        "physical_release_stays_blocked": build["physical_route_released"] is False,
    }
    report = {
        "report_type": "phase3_plan_geometry_rtx_refinement_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": str(SCENE),
        "scene_sha256": sha256(SCENE),
        "source_pdf": str(PLAN),
        "source_pdf_sha256": sha256(PLAN) if PLAN.is_file() else None,
        "refinement_config": str(REFINEMENT),
        "refinement_config_sha256": sha256(REFINEMENT),
        "measured_scene_anchors": {
            "atrium_diagonal_m": atrium_diameter,
            "east_hallway_clear_width_m": east_hall_scale[1],
            "principal_diagonal_frontage_m": principal_scale[0],
        },
        "rtx_materials": {
            "renderer": refinement["rtx_material_profile"]["renderer"],
            "maps": refinement["rtx_material_profile"]["maps"],
            "normal_map_materials": list(normal_files),
        },
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "passed": all(checks.values()),
        "unresolved_site_measurements": refinement["unresolved_site_measurements"],
        "claim_boundary": refinement["rtx_material_profile"]["claim_boundary"],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PHASE3_GEOMETRY_RTX_REPORT={OUTPUT}")
    print(f"PHASE3_GEOMETRY_RTX_CHECKS={report['checks_passed']}/{report['checks_total']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
