#!/usr/bin/env python3
"""Build the Phase 7K hybrid metric, phototextured photogrammetric survey."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("/home/robot-wst/Downloads/Project-2608231545.usdz"),
    )
    parser.add_argument(
        "--principal-supplement",
        type=Path,
        default=Path("/home/robot-wst/Downloads/Project-2608231553.usdz"),
    )
    parser.add_argument(
        "--mission",
        type=Path,
        default=root / "results/administration_nav2_phase7e_static_fusion_mission.json",
    )
    parser.add_argument(
        "--visual-output",
        type=Path,
        default=root / "scenes/phase7k_phototextured_metric_visual.usdc",
    )
    parser.add_argument(
        "--scene-output",
        type=Path,
        default=root / "scenes/phase7k_phototextured_photogrammetric_survey.usda",
    )
    parser.add_argument(
        "--presentation-output",
        type=Path,
        default=root / "scenes/phase7k_phototextured_presentation.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results/phase7k_phototextured_survey_build.json",
    )
    return parser.parse_args()


ARGS = parse_args()

# The Phase 7J builder owns the already-accepted metric registration, crop,
# route reconciliation and collision separation.  Import it with a clean argv
# so Phase 7K reuses those proven routines without duplicating safety logic.
sys.argv = [sys.argv[0]]
import build_phase7j_complete_twin as base

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_TEXTURES = {
    "Wall": ("office_wall_albedo.png", "office_wall_roughness.png", "office_wall_normal.png"),
    "Door": ("grey_door_albedo.png", "grey_door_roughness.png", "grey_door_normal.png"),
    "Floor": (
        "atrium_terrazzo_albedo.png",
        "atrium_terrazzo_roughness.png",
        "atrium_terrazzo_normal.png",
    ),
    "Table": (
        "principal_walnut_albedo.png",
        "principal_walnut_roughness.png",
        "principal_walnut_normal.png",
    ),
    "Storage": (
        "principal_walnut_albedo.png",
        "principal_walnut_roughness.png",
        "principal_walnut_normal.png",
    ),
}

# Native Z-up cluster centre -> provisional world XY registration.  The scale
# remains metric.  No false inter-cluster weld is authored.
CLUSTER_REGISTRATIONS = {
    "AtriumCorridorCluster": {
        "asset": "phase7k_atrium_corridor_photogrammetry.usdc",
        "native_centre_xy_m": (-0.374771, 1.556924),
        "world_centre_xy_m": (1.8, -0.4),
        "world_yaw_deg": -3.56,
        "coverage": "atrium/east-corridor captured cluster",
    },
    "PrincipalOfficeCluster": {
        "asset": "phase7k_principal_office_photogrammetry.usdc",
        "native_centre_xy_m": (0.079262, 1.391665),
        "world_centre_xy_m": (7.6, -8.3),
        "world_yaw_deg": -146.0,
        "coverage": "Principal office captured cluster",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_usda_eof(path: Path) -> None:
    """Keep generated ASCII USD layers clean for source-control checks."""
    path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")


def cluster_matrix(
    native_centre_xy: tuple[float, float],
    world_centre_xy: tuple[float, float],
    yaw_deg: float,
) -> Gf.Matrix4d:
    cx, cy = native_centre_xy
    wx, wy = world_centre_xy
    yaw = math.radians(yaw_deg)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return Gf.Matrix4d(
        cosine,
        sine,
        0.0,
        0.0,
        -sine,
        cosine,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        wx - cosine * cx + sine * cy,
        wy - sine * cx - cosine * cy,
        0.012,
        1.0,
    )


def add_normal_maps(stage: Usd.Stage) -> int:
    connected = 0
    for category, (_, _, normal_file) in CAPTURE_TEXTURES.items():
        path = f"/CapturedAdministration/Materials/{category}"
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/PreviewSurface"))
        reader = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/UVReader"))
        if not shader.GetPrim().IsValid() or not reader.GetPrim().IsValid():
            continue
        normal = UsdShade.Shader.Define(stage, f"{path}/NormalTexture")
        normal.CreateIdAttr("UsdUVTexture")
        normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"../textures/phase7k_capture/{normal_file}")
        )
        normal.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        normal.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        normal.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        normal.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal.ConnectableAPI(), "rgb"
        )
        connected += 1
    return connected


def create_photo_material(
    stage: Usd.Stage,
    path: str,
    albedo: str,
    roughness: str,
    normal: str,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/UVReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for role, filename, output_type, colour_space in (
        ("Albedo", albedo, Sdf.ValueTypeNames.Float3, "sRGB"),
        ("Roughness", roughness, Sdf.ValueTypeNames.Float, "raw"),
        ("Normal", normal, Sdf.ValueTypeNames.Float3, "raw"),
    ):
        texture = UsdShade.Shader.Define(stage, f"{path}/{role}Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"../textures/phase7k_capture/{filename}")
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colour_space)
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
        if role == "Albedo":
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                texture.ConnectableAPI(), "rgb"
            )
        elif role == "Roughness":
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                texture.ConnectableAPI(), "r"
            )
        else:
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                texture.ConnectableAPI(), "rgb"
            )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def add_textured_floor_patch(
    stage: Usd.Stage,
    path: str,
    centre_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_deg: float,
    material: UsdShade.Material,
) -> None:
    half_x, half_y = 0.5 * size_xy[0], 0.5 * size_xy[1]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-half_x, -half_y, 0.0),
            Gf.Vec3f(half_x, -half_y, 0.0),
            Gf.Vec3f(half_x, half_y, 0.0),
            Gf.Vec3f(-half_x, half_y, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set([Gf.Vec2f(0, 0), Gf.Vec2f(4, 0), Gf.Vec2f(4, 5), Gf.Vec2f(0, 5)])
    xformable = UsdGeom.Xformable(mesh)
    xformable.AddTranslateOp().Set(Gf.Vec3d(centre_xy[0], centre_xy[1], 0.018))
    xformable.AddRotateZOp().Set(yaw_deg)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    mesh.GetPrim().SetCustomDataByKey("aisha:role", "photo_material_overlay_non_collision")


def upgrade_visual_layer(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"could not reopen {path}")
    root = stage.GetPrimAtPath("/CapturedAdministration")
    root.SetCustomDataByKey("aisha:phase", "PHASE7K")
    root.SetCustomDataByKey(
        "aisha:claim", "metric_roomplan_visual_with_real_capture_derived_pbr"
    )
    connected = add_normal_maps(stage)
    stage.GetRootLayer().Save()
    return {"normal_maps_connected": connected, "sha256": sha256(path)}


def upgrade_composite(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"could not reopen {path}")
    world = stage.GetPrimAtPath("/World")
    world.SetCustomDataByKey("aisha:phase", "PHASE7K")
    world.SetCustomDataByKey(
        "aisha:visual_layer", "phototextured_metric_twin_plus_dense_survey_clusters"
    )
    world.SetCustomDataByKey(
        "aisha:collision_layer", "unchanged_frozen_phase7i_route_critical_geometry"
    )
    survey = UsdGeom.Xform.Define(stage, "/World/PhotogrammetricSurvey")
    survey.GetPrim().SetCustomDataByKey(
        "aisha:role", "visible_non_collision_dense_photogrammetry_evidence"
    )
    survey.GetPrim().SetCustomDataByKey("aisha:false_welded", False)
    registrations = {}
    for name, registration in CLUSTER_REGISTRATIONS.items():
        prim = UsdGeom.Xform.Define(stage, f"/World/PhotogrammetricSurvey/{name}")
        prim.GetPrim().GetReferences().AddReference(
            f"./{registration['asset']}", Sdf.Path("/Survey")
        )
        matrix = cluster_matrix(
            registration["native_centre_xy_m"],
            registration["world_centre_xy_m"],
            registration["world_yaw_deg"],
        )
        prim.AddTransformOp(opSuffix="provisionalMetricRegistration").Set(matrix)
        prim.GetPrim().SetCustomDataByKey("aisha:coverage", registration["coverage"])
        prim.GetPrim().SetCustomDataByKey("aisha:collision_enabled", False)
        prim.GetPrim().SetCustomDataByKey(
            "aisha:registration_quality", "provisional_visual_alignment_not_survey_control"
        )
        registrations[name] = {
            "asset": registration["asset"],
            "native_centre_xy_m": list(registration["native_centre_xy_m"]),
            "world_centre_xy_m": list(registration["world_centre_xy_m"]),
            "world_yaw_deg": registration["world_yaw_deg"],
            "metric_scale": 1.0,
            "collision_enabled": False,
        }

    looks = UsdGeom.Scope.Define(stage, "/World/Phase7KPhotoMaterials")
    looks.GetPrim().SetCustomDataByKey("aisha:source", "privacy-safe supplied still crop")
    floor_material = create_photo_material(
        stage,
        "/World/Phase7KPhotoMaterials/PrincipalGreyOak",
        "principal_grey_oak_albedo.png",
        "principal_grey_oak_roughness.png",
        "principal_grey_oak_normal.png",
    )
    add_textured_floor_patch(
        stage,
        "/World/PhotoSurfaceOverlays/PrincipalGreyOakFloor",
        (7.6, -8.3),
        (3.75, 4.90),
        -146.0,
        floor_material,
    )
    stage.GetRootLayer().Save()
    normalize_usda_eof(path)
    return {
        "cluster_registrations": registrations,
        "cluster_count": len(registrations),
        "clusters_visible": True,
        "clusters_false_welded": False,
        "principal_photo_floor_overlay": True,
        "sha256": sha256(path),
    }


def build_presentation_layer(survey_scene: Path, presentation_scene: Path) -> dict:
    """Author a non-destructive presentation override over the survey scene.

    The genuine dense clusters remain available in ``survey_scene``.  They are
    intentionally hidden only in this presentation layer because incomplete
    capture surfaces create floating fragments that obscure the robot route.
    """
    presentation_scene.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(presentation_scene))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    world.GetPrim().GetReferences().AddReference(
        f"./{survey_scene.name}", Sdf.Path("/World")
    )
    world.GetPrim().SetCustomDataByKey("aisha:phase", "PHASE7K")
    world.GetPrim().SetCustomDataByKey(
        "aisha:presentation_mode",
        "clean_phototextured_metric_twin_with_raw_dense_evidence_hidden",
    )
    dense = stage.OverridePrim("/World/PhotogrammetricSurvey")
    UsdGeom.Imageable(dense).MakeInvisible()
    dense.SetCustomDataByKey(
        "aisha:hidden_reason",
        "incomplete_dense_capture_fragments_obscure_navigation_presentation",
    )
    stage.GetRootLayer().Save()
    normalize_usda_eof(presentation_scene)
    return {
        "path": str(presentation_scene.relative_to(PACKAGE_ROOT)),
        "sha256": sha256(presentation_scene),
        "survey_scene_referenced": str(survey_scene.relative_to(PACKAGE_ROOT)),
        "dense_clusters_present_but_hidden": True,
        "navigation_collision_geometry_changed": False,
    }


def main() -> int:
    required = (
        ARGS.primary,
        ARGS.principal_supplement,
        ARGS.mission,
        PACKAGE_ROOT / "scenes/administration.usd",
        PACKAGE_ROOT / "scenes/phase7k_atrium_corridor_photogrammetry.usdc",
        PACKAGE_ROOT / "scenes/phase7k_principal_office_photogrammetry.usdc",
        PACKAGE_ROOT / "results/phase7k_capture_materials_report.json",
        PACKAGE_ROOT / "results/phase7k_photogrammetry_asset_manifest.json",
        PACKAGE_ROOT / "results/phase7k_photogrammetry_usd_build.json",
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    # Reuse the accepted Phase 7J builder with real supplied-surface textures.
    base.CATEGORY_TEXTURES = {
        category: (
            f"../phase7k_capture/{textures[0]}",
            f"../phase7k_capture/{textures[1]}",
        )
        for category, textures in CAPTURE_TEXTURES.items()
    }
    visual = ARGS.visual_output.resolve()
    scene = ARGS.scene_output.resolve()
    presentation = ARGS.presentation_output.resolve()
    visual_result = base.build_visual_layer(
        ARGS.primary.resolve(),
        ARGS.principal_supplement.resolve(),
        visual,
    )
    visual_upgrade = upgrade_visual_layer(visual)
    composite_result = base.build_composite_scene(visual, ARGS.mission.resolve(), scene)
    composite_upgrade = upgrade_composite(scene)
    presentation_result = build_presentation_layer(scene, presentation)
    material_report = json.loads(
        (PACKAGE_ROOT / "results/phase7k_capture_materials_report.json").read_text(
            encoding="utf-8"
        )
    )
    photogrammetry_manifest = json.loads(
        (PACKAGE_ROOT / "results/phase7k_photogrammetry_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    photogrammetry_build = json.loads(
        (PACKAGE_ROOT / "results/phase7k_photogrammetry_usd_build.json").read_text(
            encoding="utf-8"
        )
    )
    report = {
        "report_type": "phase7k_phototextured_photogrammetric_survey_build",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "passed": True,
        "visual_layer": {
            **visual_result,
            **visual_upgrade,
            "path": str(visual.relative_to(PACKAGE_ROOT)),
        },
        "composite_scene": {
            **composite_result,
            **composite_upgrade,
            "path": str(scene.relative_to(PACKAGE_ROOT)),
        },
        "presentation_scene": presentation_result,
        "capture_materials": {
            "report_sha256": sha256(
                PACKAGE_ROOT / "results/phase7k_capture_materials_report.json"
            ),
            "asset_sets": len(material_report["assets"]),
            "source_stills_committed": False,
        },
        "photogrammetry": {
            "manifest_sha256": sha256(
                PACKAGE_ROOT / "results/phase7k_photogrammetry_asset_manifest.json"
            ),
            "usd_build_sha256": sha256(
                PACKAGE_ROOT / "results/phase7k_photogrammetry_usd_build.json"
            ),
            "total_vertices": photogrammetry_build["total_vertices"],
            "total_faces": photogrammetry_build["total_faces"],
            "clusters_kept_separate": True,
            "privacy_screened_atlases": photogrammetry_manifest["passed"],
        },
        "frozen_safety_contract": {
            "navigation_collision_geometry_changed": False,
            "raw_dense_mesh_used_for_collision": False,
            "source_motion_was_accepted_live_policy": True,
            "presentation_motion_is_recorded_pose_replay": True,
        },
        "claim_boundary": {
            "supported": (
                "hybrid metric phototextured survey with genuine dense captured clusters "
                "and real image-derived PBR"
            ),
            "complete_monolithic_photogrammetric_mesh": False,
            "cluster_registration_is_survey_control": False,
            "locked_vice_principal_interior_captured": False,
            "physical_release": False,
        },
    }
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        base.APP.close()
