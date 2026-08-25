#!/usr/bin/env python3
"""Build the registered Phase 7L NuRec visual/collision composite scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-asset",
        type=Path,
        default=root / "tmp/phase7l_nurec_runs/administration_full_nurec.usdz",
    )
    parser.add_argument(
        "--principal-asset",
        type=Path,
        default=root / "tmp/phase7l_nurec_runs/principal_full_nurec.usdz",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=root / "results/phase7l_nurec_registration.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "scenes/phase7l_nurec_registered_administration.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results/phase7l_nurec_composite_build.json",
    )
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "multi_gpu": False,
    }
)

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def asset_reference(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.resolve().parent)


def matrix_from_rows(rows: list[list[float]]) -> Gf.Matrix4d:
    return Gf.Matrix4d(*(value for row in rows for value in row))


def hide_legacy_render_geometry(stage: Usd.Stage, root_path: str) -> int:
    """Hide composed Gprims without altering their collision/physics APIs."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"missing frozen-world render root: {root_path}")
    hidden_count = 0
    for source_prim in Usd.PrimRange(root):
        if not source_prim.IsA(UsdGeom.Gprim):
            continue
        hidden = UsdGeom.Imageable(stage.OverridePrim(str(source_prim.GetPath())))
        hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        hidden_count += 1
    root_override = stage.OverridePrim(root_path)
    root_override.SetCustomDataByKey(
        "aisha:role", "hidden_legacy_render_geometry_collision_preserved"
    )
    root_override.SetCustomDataByKey("aisha:hidden_gprim_count", hidden_count)
    return hidden_count


def add_sector_variant(
    stage: Usd.Stage,
    sector: Usd.VariantSet,
    name: str,
    nurec_asset: Path,
    administration_asset: Path,
    output: Path,
    world_transform_rows: list[list[float]],
    role: str,
) -> None:
    """Author one native-coordinate NuRec sector and its aligned metric world.

    NuRec 3D Gaussian assets must remain in their trained coordinate basis for
    correct RTX reconstruction.  The inverse registration is therefore applied
    to the complete frozen navigation world (robot included), rather than to
    the Gaussian volume.  This preserves the visual result while maintaining
    the same metric alignment.
    """
    sector.AddVariant(name)
    sector.SetVariantSelection(name)
    with sector.GetVariantEditContext():
        visual = UsdGeom.Xform.Define(stage, "/World/Presentation/NuRec")
        visual_prim = visual.GetPrim()
        visual_prim.GetReferences().AddReference(asset_reference(nurec_asset, output))
        visual_prim.SetCustomDataByKey("aisha:role", role)
        visual_prim.SetCustomDataByKey("aisha:visual_only", True)
        visual_prim.SetCustomDataByKey("aisha:collision_enabled", False)
        visual_prim.SetCustomDataByKey(
            "aisha:coordinate_strategy", "nurec_native_basis"
        )

        metric_world = UsdGeom.Xform.Define(
            stage, "/World/Presentation/MetricWorld"
        )
        metric_world_prim = metric_world.GetPrim()
        metric_world_prim.GetReferences().AddReference(
            asset_reference(administration_asset, output), "/World"
        )
        native_from_world = matrix_from_rows(world_transform_rows).GetInverse()
        metric_world.AddTransformOp(opSuffix="worldToNuRecNative").Set(native_from_world)
        metric_world_prim.SetCustomDataByKey(
            "aisha:role", "registered_frozen_navigation_world_and_robot"
        )

        robot = stage.OverridePrim("/World/Presentation/MetricWorld/AISHA")
        robot.SetCustomDataByKey(
            "aisha:motion_source", "accepted_phase7e_recorded_policy_pose_replay"
        )
        hidden_total = 0
        for child in ("Architecture", "Appearance", "Furniture"):
            hidden_total += hide_legacy_render_geometry(
                stage, f"/World/Presentation/MetricWorld/{child}"
            )
        metric_world_prim.SetCustomDataByKey(
            "aisha:hidden_legacy_render_gprim_count", hidden_total
        )


def main() -> int:
    main_asset = ARGS.main_asset.expanduser().resolve()
    principal_asset = ARGS.principal_asset.expanduser().resolve()
    registration_path = ARGS.registration.expanduser().resolve()
    output = ARGS.output.expanduser().resolve()
    report_path = ARGS.report.expanduser().resolve()
    administration_asset = output.parent / "administration.usd"
    for path in (main_asset, principal_asset, registration_path, administration_asset):
        if not path.is_file():
            raise FileNotFoundError(path)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if not registration.get("passed"):
        raise RuntimeError("Phase 7L registration has not passed")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(309.0)
    stage.SetTimeCodesPerSecond(24.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey("aisha:phase", "PHASE7L")
    world.GetPrim().SetCustomDataByKey(
        "aisha:visual_layer", "registered_nurec_gaussian_sectors"
    )
    world.GetPrim().SetCustomDataByKey(
        "aisha:collision_layer", "unchanged_frozen_phase7i_route_critical_geometry"
    )
    world.GetPrim().SetCustomDataByKey(
        "aisha:vp_interior", "assumed_locked_not_captured"
    )

    presentation = UsdGeom.Xform.Define(stage, "/World/Presentation")
    presentation.GetPrim().SetCustomDataByKey(
        "aisha:registration_strategy",
        "inverse_transform_frozen_metric_world_into_native_nurec_basis",
    )
    variants = presentation.GetPrim().GetVariantSets().AddVariantSet("visualSector")
    add_sector_variant(
        stage,
        variants,
        "main_administration",
        main_asset,
        administration_asset,
        output,
        registration["world_transforms"]["main_component"]["usd_gf_row_vector_matrix"],
        "atrium_reception_and_captured_administration_corridors",
    )
    add_sector_variant(
        stage,
        variants,
        "principal_office",
        principal_asset,
        administration_asset,
        output,
        registration["world_transforms"]["principal_component"]["usd_gf_row_vector_matrix"],
        "shared_atrium_principal_approach_and_captured_principal_interior",
    )
    variants.SetVariantSelection("principal_office")
    navigation = stage.DefinePrim("/World/NavigationCollisionContract", "Scope")
    navigation.SetCustomDataByKey(
        "aisha:role", "hidden_frozen_phase7i_route_critical_collision_layer"
    )
    navigation.SetCustomDataByKey(
        "aisha:members",
        "/World/Presentation/MetricWorld/Architecture,"
        "/World/Presentation/MetricWorld/Furniture",
    )
    route = stage.DefinePrim("/World/RouteEvidence", "Scope")
    route.GetReferences().AddReference(
        "./phase7j_complete_captured_administration.usda", "/World/RouteEvidence"
    )

    lighting = stage.DefinePrim("/World/Lighting", "Scope")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.GetIntensityAttr().Set(420.0)
    dome.GetColorAttr().Set(Gf.Vec3f(0.82, 0.88, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/RobotKey")
    key.GetIntensityAttr().Set(1800.0)
    key.GetAngleAttr().Set(4.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 24.0, 22.0))
    lighting.SetCustomDataByKey("aisha:purpose", "robot_illumination_only")

    stage.GetRootLayer().Save()
    # USD's ASCII writer leaves an extra blank line; normalize it so the
    # generated presentation shell remains clean under git diff --check.
    output.write_text(output.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    composed = Usd.Stage.Open(str(output))
    visible_legacy_gprims: list[str] = []
    for child in ("Architecture", "Appearance", "Furniture"):
        legacy_root = composed.GetPrimAtPath(
            f"/World/Presentation/MetricWorld/{child}"
        )
        if not legacy_root.IsValid():
            continue
        for prim in Usd.PrimRange(legacy_root):
            if (
                prim.IsA(UsdGeom.Gprim)
                and UsdGeom.Imageable(prim).ComputeVisibility()
                != UsdGeom.Tokens.invisible
            ):
                visible_legacy_gprims.append(str(prim.GetPath()))
    checks = {
        "stage_opens": composed is not None,
        "active_nurec_composes": bool(
            composed
            and composed.GetPrimAtPath("/World/Presentation/NuRec/gauss").IsValid()
        ),
        "robot_composes": bool(
            composed
            and composed.GetPrimAtPath(
                "/World/Presentation/MetricWorld/AISHA"
            ).IsValid()
        ),
        "collision_architecture_composes": bool(
            composed
            and composed.GetPrimAtPath(
                "/World/Presentation/MetricWorld/Architecture"
            ).IsValid()
        ),
        "collision_furniture_composes": bool(
            composed
            and composed.GetPrimAtPath(
                "/World/Presentation/MetricWorld/Furniture"
            ).IsValid()
        ),
        "legacy_render_geometry_hidden": not visible_legacy_gprims,
        "visual_sector_default_is_principal": variants.GetVariantSelection()
        == "principal_office",
    }
    passed = all(checks.values())
    report = {
        "report_type": "phase7l_nurec_composite_build",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scene": portable_path(output),
        "scene_sha256": sha256_file(output),
        "registration": portable_path(registration_path),
        "assets": {
            "main": {
                "path": portable_path(main_asset),
                "sha256": sha256_file(main_asset),
                "size_bytes": main_asset.stat().st_size,
                "privacy_sensitive_local_only": True,
            },
            "principal": {
                "path": portable_path(principal_asset),
                "sha256": sha256_file(principal_asset),
                "size_bytes": principal_asset.stat().st_size,
                "privacy_sensitive_local_only": True,
            },
        },
        "visual_sector_variants": [
            "main_administration",
            "principal_office",
        ],
        "default_visual_sector": "principal_office",
        "checks": checks,
        "visible_legacy_gprims": visible_legacy_gprims,
        "layer_contract": {
            "gaussians_visual_only": True,
            "navigation_collision_geometry_changed": False,
            "frozen_collision_hidden_but_active": True,
            "raw_gaussians_used_for_lidar_or_collision": False,
            "nurec_asset_transform": "identity_native_training_basis",
            "metric_world_transform": "inverse_of_registered_nurec_to_world_sim3",
            "robot_motion": "accepted recorded policy pose replay, not live policy execution",
            "vice_principal_interior_assumed_locked_not_captured": True,
            "physical_release": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        APP.close()
    raise SystemExit(exit_code)
