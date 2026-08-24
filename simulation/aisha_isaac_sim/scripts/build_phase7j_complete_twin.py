#!/usr/bin/env python3
"""Build a privacy-safe, complete RoomPlan administration twin for Omniverse.

The generated visual layer contains the complete primary RoomPlan capture and
the registered Principal supplement.  It is flattened so the presentation does
not depend on files in Downloads.  The composite stage keeps the proven Phase
7I navigation collision layer separate from the capture-derived visuals and
references the existing AI-SHA presentation robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


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
        default=root / "scenes/phase7j_complete_captured_administration_visual.usdc",
    )
    parser.add_argument(
        "--scene-output",
        type=Path,
        default=root / "scenes/phase7j_complete_captured_administration.usda",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results/phase7j_complete_captured_administration_build.json",
    )
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# The complete-primary registration uses the centreline of the 2.8-3.0 m east
# administration hallway, not the old route-scoped unidentified4 section
# origin.  The paired 3.56-degree wall chains run from raw X ~= 8.8 m to the
# conference/VP end and match the approved page-2 hallway at world Y = 0.
PRIMARY_ANCHOR_XZ_M = (8.8, 11.7)
PRIMARY_WORLD_ANCHOR_XY_M = (4.7, 0.0)
PRIMARY_WORLD_YAW_DEG = -3.56
PRIMARY_WORLD_Z_OFFSET_M = 1.3561
PRINCIPAL_NATIVE_ANCHOR_XZ_M = (-6.0, 1.1)
PRINCIPAL_WORLD_ANCHOR_XY_M = (6.978, -7.628)
PRINCIPAL_WORLD_YAW_DEG = -146.0
PRINCIPAL_WORLD_Z_OFFSET_M = 1.5663
PRINCIPAL_CROP_WORLD_XY_M = (3.8, 12.3, -12.8, -3.7)

CATEGORY_MATERIALS = {
    "Wall": ((0.76, 0.72, 0.66), 0.72, 0.0, 1.0),
    "Door": ((0.28, 0.075, 0.028), 0.32, 0.0, 1.0),
    "Window": ((0.16, 0.38, 0.47), 0.12, 0.02, 0.34),
    "Floor": ((0.43, 0.45, 0.46), 0.26, 0.02, 1.0),
    "Chair": ((0.025, 0.030, 0.035), 0.43, 0.02, 1.0),
    "Table": ((0.43, 0.17, 0.055), 0.30, 0.01, 1.0),
    "Storage": ((0.33, 0.095, 0.028), 0.35, 0.01, 1.0),
    "Television": ((0.008, 0.010, 0.014), 0.16, 0.08, 1.0),
}
CATEGORY_TEXTURES = {
    "Door": ("walnut_albedo.png", "walnut_roughness.png"),
    "Floor": ("terrazzo_albedo.png", "terrazzo_roughness.png"),
    "Table": ("oak_albedo.png", "oak_roughness.png"),
    "Storage": ("walnut_albedo.png", "walnut_roughness.png"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_prim_path(path: Path) -> Sdf.Path:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"RoomPlan source has no valid default prim: {path}")
    return stage.GetDefaultPrim().GetPath()


def roomplan_matrix(
    native_anchor_xz: tuple[float, float],
    world_anchor_xy: tuple[float, float],
    yaw_deg: float,
    world_z_offset_m: float,
) -> Gf.Matrix4d:
    """Map RoomPlan Y-up native metres into the Z-up administration frame."""
    ax, az = native_anchor_xz
    wx, wy = world_anchor_xy
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return Gf.Matrix4d(
        c,
        s,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        s,
        -c,
        0.0,
        0.0,
        wx - c * ax - s * az,
        wy - s * ax + c * az,
        world_z_offset_m,
        1.0,
    )


def category_name(prim: Usd.Prim) -> str | None:
    value = str(prim.GetCustomDataByKey("Category") or "")
    for category in CATEGORY_MATERIALS:
        if value.startswith(category):
            return category
    return None


def create_material(
    stage: Usd.Stage,
    name: str,
    colour: tuple[float, float, float],
    roughness: float,
    metallic: float,
    opacity: float,
    texture_files: tuple[str, str] | None = None,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"/CapturedAdministration/Materials/{name}")
    shader = UsdShade.Shader.Define(
        stage, f"/CapturedAdministration/Materials/{name}/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    rough = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    if texture_files is None:
        diffuse.Set(Gf.Vec3f(*colour))
        rough.Set(roughness)
    else:
        reader = UsdShade.Shader.Define(
            stage, f"/CapturedAdministration/Materials/{name}/UVReader"
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        albedo = UsdShade.Shader.Define(
            stage, f"/CapturedAdministration/Materials/{name}/AlbedoTexture"
        )
        albedo.CreateIdAttr("UsdUVTexture")
        albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"../textures/administration/{texture_files[0]}")
        )
        albedo.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        albedo.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        albedo.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        diffuse.ConnectToSource(albedo.ConnectableAPI(), "rgb")
        roughness_texture = UsdShade.Shader.Define(
            stage, f"/CapturedAdministration/Materials/{name}/RoughnessTexture"
        )
        roughness_texture.CreateIdAttr("UsdUVTexture")
        roughness_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"../textures/administration/{texture_files[1]}")
        )
        roughness_texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        roughness_texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        roughness_texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
        rough.ConnectToSource(roughness_texture.ConnectableAPI(), "r")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def semantic_prims(stage: Usd.Stage, prefix: str) -> list[Usd.Prim]:
    return [
        prim
        for prim in stage.TraverseAll()
        if str(prim.GetPath()).startswith(prefix) and category_name(prim) is not None
    ]


def crop_registered_scans(stage: Usd.Stage) -> dict[str, int]:
    xmin, xmax, ymin, ymax = PRINCIPAL_CROP_WORLD_XY_M
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    visible = Counter()
    hidden = Counter()
    for label, prefix in (
        ("primary", "/CapturedAdministration/Primary"),
        ("principal_supplement", "/CapturedAdministration/PrincipalSupplement"),
    ):
        for prim in semantic_prims(stage, prefix):
            category = category_name(prim)
            assert category is not None
            centre = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
            inside = xmin <= centre[0] <= xmax and ymin <= centre[1] <= ymax
            keep = (not inside) if label == "primary" else inside
            # RoomPlan composes all assets of each furniture class through one
            # category group.  Preserve the complete primary furniture group;
            # the cropped supplement then adds the denser Principal capture.
            if category in {"Chair", "Table", "Storage", "Television"} and label == "primary":
                keep = True
            # The primary scan owns the single complete administration floor.
            if category == "Floor":
                keep = label == "primary"
            imageable = UsdGeom.Imageable(prim)
            if keep:
                imageable.MakeVisible()
                visible[f"{label}:{category}"] += 1
            else:
                imageable.MakeInvisible()
                hidden[f"{label}:{category}"] += 1
    return {**{f"visible_{k}": v for k, v in visible.items()}, **{f"hidden_{k}": v for k, v in hidden.items()}}


def bind_semantic_materials(stage: Usd.Stage) -> Counter:
    materials = {
        category: create_material(
            stage, category, *values, texture_files=CATEGORY_TEXTURES.get(category)
        )
        for category, values in CATEGORY_MATERIALS.items()
    }
    counts: Counter = Counter()
    for prim in stage.TraverseAll():
        category = category_name(prim)
        if category is None:
            continue
        counts[category] += 1
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            materials[category],
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
    # Multiple RoomPlan component references compose through category groups.
    # Bind child meshes explicitly so Door/Window and individual furniture
    # retain their own finishes, and author planar UVs for the PBR textures.
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        category = next(
            (
                name
                for name in (
                    "Door",
                    "Window",
                    "Floor",
                    "Chair",
                    "Table",
                    "Storage",
                    "Television",
                    "Wall",
                )
                if name in path
            ),
            None,
        )
        if category is None:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = list(mesh.GetPointsAttr().Get() or [])
        if points:
            ranges = [
                max(float(point[axis]) for point in points)
                - min(float(point[axis]) for point in points)
                for axis in range(3)
            ]
            axes = sorted(range(3), key=lambda axis: ranges[axis], reverse=True)[:2]
            scale = 0.42 if category == "Floor" else 0.85
            values = [
                Gf.Vec2f(float(point[axes[0]]) * scale, float(point[axes[1]]) * scale)
                for point in points
            ]
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            primvar.Set(values)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            materials[category],
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
    return counts


def scrub_source_identifiers(stage: Usd.Stage) -> int:
    scrubbed = 0
    for prim in stage.TraverseAll():
        if prim.HasCustomDataKey("UUID"):
            prim.ClearCustomDataByKey("UUID")
            scrubbed += 1
        asset_info = dict(prim.GetAssetInfo())
        if "identifier" in asset_info:
            asset_info.pop("identifier", None)
            prim.SetAssetInfo(asset_info)
    return scrubbed


def build_visual_layer(primary: Path, supplement: Path, output: Path) -> dict:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/CapturedAdministration")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("aisha:phase", "PHASE7J")
    root.GetPrim().SetCustomDataByKey(
        "aisha:claim", "complete_captured_area_semantic_digital_twin"
    )

    primary_prim = UsdGeom.Xform.Define(stage, "/CapturedAdministration/Primary")
    primary_prim.GetPrim().GetReferences().AddReference(
        str(primary.resolve()), root_prim_path(primary)
    )
    primary_prim.AddTransformOp(opSuffix="roomplanRegistration").Set(
        roomplan_matrix(
            PRIMARY_ANCHOR_XZ_M,
            PRIMARY_WORLD_ANCHOR_XY_M,
            PRIMARY_WORLD_YAW_DEG,
            PRIMARY_WORLD_Z_OFFSET_M,
        )
    )

    principal_prim = UsdGeom.Xform.Define(
        stage, "/CapturedAdministration/PrincipalSupplement"
    )
    principal_prim.GetPrim().GetReferences().AddReference(
        str(supplement.resolve()), root_prim_path(supplement)
    )
    principal_prim.AddTransformOp(opSuffix="roomplanRegistration").Set(
        roomplan_matrix(
            PRINCIPAL_NATIVE_ANCHOR_XZ_M,
            PRINCIPAL_WORLD_ANCHOR_XY_M,
            PRINCIPAL_WORLD_YAW_DEG,
            PRINCIPAL_WORLD_Z_OFFSET_M,
        )
    )

    for _ in range(8):
        APP.update()
    crop_counts = crop_registered_scans(stage)
    category_counts = bind_semantic_materials(stage)
    scrubbed = scrub_source_identifiers(stage)

    output.parent.mkdir(parents=True, exist_ok=True)
    flattened = stage.Flatten(addSourceFileComment=False)
    if not flattened.Export(str(output.resolve())):
        raise RuntimeError(f"could not export flattened visual layer: {output}")
    return {
        "category_counts": dict(sorted(category_counts.items())),
        "crop_counts": dict(sorted(crop_counts.items())),
        "uuid_fields_scrubbed": scrubbed,
        "flattened": True,
        "external_roomplan_dependencies": False,
    }


def add_route_curve(stage: Usd.Stage, mission: dict) -> int:
    trace = mission["pose_trace"]
    stride = max(1, len(trace) // 900)
    samples = trace[::stride]
    if samples[-1] is not trace[-1]:
        samples.append(trace[-1])
    curve = UsdGeom.BasisCurves.Define(stage, "/World/RouteEvidence/AcceptedMissionTrace")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateBasisAttr(UsdGeom.Tokens.bspline)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([len(samples)])
    curve.CreatePointsAttr(
        [Gf.Vec3f(float(s["x_m"]), float(s["y_m"]), 0.035) for s in samples]
    )
    curve.CreateWidthsAttr([0.045])
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.72, 0.54)])
    curve.GetPrim().SetCustomDataByKey("aisha:source", "accepted_phase7e_live_policy_mission")
    UsdGeom.Imageable(curve.GetPrim()).MakeInvisible()
    return len(samples)


def hide_movable_route_conflicts(stage: Usd.Stage, mission: dict) -> list[dict]:
    """Create a presentation override without altering the full capture layer.

    RoomPlan furniture is a capture-time observation, not fixed architecture.
    Items whose exact projected semantic hull enters the 0.85 m corner-swept
    presentation envelope are hidden only in the composite.  The wider radial
    envelope includes AI-SHA's rectangular corners during turns.  The flattened
    visual layer retains every captured item for survey inspection.
    """
    route = [
        (float(sample["x_m"]), float(sample["y_m"]))
        for sample in mission["pose_trace"][::3]
    ]
    hidden: list[dict] = []
    clearance_m = 0.85
    for category_prim in semantic_prims(stage, "/World/CapturedAdministration"):
        category = category_name(category_prim)
        if category not in {"Chair", "Table", "Storage", "Television"}:
            continue
        # RoomPlan assigns the category to a class group (for example
        # Storage_grp); each direct child is one independently placed object.
        # Reconcile those instances, never the entire class group.
        candidates = list(category_prim.GetChildren()) or [category_prim]
        for prim in candidates:
            world_points = semantic_world_points(prim)
            if not world_points:
                continue
            z_min = min(float(point[2]) for point in world_points)
            z_max = max(float(point[2]) for point in world_points)
            if z_max < -0.05 or z_min > 1.90:
                continue
            hull = convex_hull_xy(
                [(float(point[0]), float(point[1])) for point in world_points]
            )
            minimum_distance = min(
                point_to_polygon_distance_xy(point, hull) for point in route
            )
            if minimum_distance > clearance_m:
                continue
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(
                {
                    "path": str(prim.GetPath()),
                    "category": category,
                    "minimum_route_centre_distance_m": round(minimum_distance, 4),
                    "corner_swept_presentation_envelope_radius_m": clearance_m,
                    "full_capture_visual_layer_retains_item": True,
                }
            )
    return hidden


def hide_primary_principal_furniture_duplicates(stage: Usd.Stage) -> list[dict]:
    """Let the denser supplement own movable visuals inside the Principal crop.

    Both scans remain untouched in the flattened survey layer.  This visibility
    override is authored only in the presentation composite, where showing the
    primary capture-time furniture and the supplementary capture simultaneously
    would duplicate desks, chairs and storage in the Principal suite.
    """
    xmin, xmax, ymin, ymax = PRINCIPAL_CROP_WORLD_XY_M
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    hidden: list[dict] = []
    movable_tokens = ("Chair", "Table", "Storage", "Television")
    prefix = "/World/CapturedAdministration/Primary"
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if not path.startswith(prefix) or not prim.IsA(UsdGeom.Mesh):
            continue
        category = next((name for name in movable_tokens if name in path), None)
        if category is None:
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        centre = bounds.GetMidpoint()
        if not (xmin <= centre[0] <= xmax and ymin <= centre[1] <= ymax):
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden.append(
            {
                "path": path,
                "category": category,
                "centre_xy_m": [round(float(centre[0]), 4), round(float(centre[1]), 4)],
                "principal_supplement_is_presentation_authority": True,
                "full_capture_visual_layer_retains_item": True,
            }
        )
    return hidden


def convex_hull_xy(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def point_to_segment_distance_xy(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1.0e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq),
    )
    nearest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def point_to_polygon_distance_xy(
    point: tuple[float, float], hull: list[tuple[float, float]]
) -> float:
    if not hull:
        return math.inf
    if len(hull) == 1:
        return math.hypot(point[0] - hull[0][0], point[1] - hull[0][1])
    if len(hull) == 2:
        return point_to_segment_distance_xy(point, hull[0], hull[1])
    inside = False
    previous = hull[-1]
    for current in hull:
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing_x = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    if inside:
        return 0.0
    return min(
        point_to_segment_distance_xy(point, hull[index - 1], hull[index])
        for index in range(len(hull))
    )


def semantic_world_points(prim: Usd.Prim) -> list[Gf.Vec3d]:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points: list[Gf.Vec3d] = []
    for descendant in Usd.PrimRange(prim):
        if not descendant.IsA(UsdGeom.Mesh):
            continue
        transform = cache.GetLocalToWorldTransform(descendant)
        for point in UsdGeom.Mesh(descendant).GetPointsAttr().Get() or []:
            points.append(transform.Transform(Gf.Vec3d(point)))
    return points


def reconcile_static_visuals_with_validated_route(
    stage: Usd.Stage, mission: dict
) -> list[dict]:
    """Suppress visual scan fragments contradicting the plan-authorized corridor."""
    route = [
        (float(sample["x_m"]), float(sample["y_m"]))
        for sample in mission["pose_trace"][::3]
    ]
    clearance_m = 0.42
    hidden: list[dict] = []
    for prim in semantic_prims(stage, "/World/CapturedAdministration"):
        category = category_name(prim)
        if category not in {"Wall", "Door", "Window"}:
            continue
        world_points = semantic_world_points(prim)
        if not world_points:
            continue
        z_min = min(float(point[2]) for point in world_points)
        z_max = max(float(point[2]) for point in world_points)
        if z_max < 0.0 or z_min > 1.90:
            continue
        hull = convex_hull_xy([(float(point[0]), float(point[1])) for point in world_points])
        minimum_distance = min(point_to_polygon_distance_xy(point, hull) for point in route)
        if minimum_distance > clearance_m:
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden.append(
            {
                "path": str(prim.GetPath()),
                "category": category,
                "minimum_route_centre_distance_m": round(minimum_distance, 4),
                "required_robot_corridor_radius_m": clearance_m,
                "approved_plan_and_frozen_collision_layer_are_authority": True,
                "full_capture_visual_layer_retains_component": True,
            }
        )
    return hidden


def build_composite_scene(visual: Path, mission_path: Path, output: Path) -> dict:
    stage = Usd.Stage.CreateNew(str(output.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    captured = stage.DefinePrim("/World/CapturedAdministration", "Xform")
    captured.GetReferences().AddReference(
        f"./{visual.name}", Sdf.Path("/CapturedAdministration")
    )

    plan_floor = stage.DefinePrim("/World/PlanAuthorityFloors", "Xform")
    plan_floor.GetReferences().AddReference(
        "./administration.usd", Sdf.Path("/World/Architecture/Floors")
    )
    plan_floor.SetCustomDataByKey(
        "aisha:role", "visible_plan_authority_floor_with_atrium_step_down"
    )

    robot = stage.DefinePrim("/World/AISHA", "Xform")
    robot.GetReferences().AddReference("./administration.usd", Sdf.Path("/World/AISHA"))

    collision = UsdGeom.Xform.Define(stage, "/World/NavigationCollision")
    collision.GetPrim().SetCustomDataByKey(
        "aisha:role", "hidden_phase7i_route_critical_collision_layer"
    )
    for name, target in (
        ("Architecture", "/World/Architecture"),
        ("Furniture", "/World/Furniture"),
    ):
        prim = stage.DefinePrim(f"/World/NavigationCollision/{name}", "Xform")
        prim.GetReferences().AddReference("./administration.usd", Sdf.Path(target))
        UsdGeom.Imageable(prim).MakeInvisible()
    # Preserve the material targets used by the referenced hidden geometry.
    # This avoids unresolved-reference noise in the operator-facing Isaac Sim
    # console while the navigation layer itself remains invisible.
    looks = stage.DefinePrim("/World/Looks", "Scope")
    looks.GetReferences().AddReference("./administration.usd", Sdf.Path("/World/Looks"))

    mission = json.loads(mission_path.read_text(encoding="utf-8"))
    if mission.get("outcome") != "success" or int(mission.get("waypoints_completed", 0)) != 12:
        raise RuntimeError("Phase 7J requires the accepted successful 12-leg source mission")
    route_points = add_route_curve(stage, mission)
    for _ in range(8):
        APP.update()
    if not any(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(plan_floor)):
        raise RuntimeError("validated plan-authority floor reference did not compose")
    floor_material = UsdShade.Material(
        stage.GetPrimAtPath("/World/CapturedAdministration/Materials/Floor")
    )
    if not floor_material.GetPrim().IsValid():
        raise RuntimeError("captured RoomPlan floor material did not compose")
    UsdShade.MaterialBindingAPI.Apply(plan_floor).Bind(
        floor_material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    hidden_roomplan_floors = []
    floor_elevation_ranges: dict[str, list[float]] = {}
    floor_bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    plan_floor_range = floor_bbox_cache.ComputeWorldBound(plan_floor).ComputeAlignedRange()
    floor_elevation_ranges["plan_authority_z_m"] = [
        round(float(plan_floor_range.GetMin()[2]), 4),
        round(float(plan_floor_range.GetMax()[2]), 4),
    ]
    for prim in semantic_prims(stage, "/World/CapturedAdministration"):
        if category_name(prim) != "Floor":
            continue
        roomplan_range = floor_bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        floor_elevation_ranges[str(prim.GetPath())] = [
            round(float(roomplan_range.GetMin()[2]), 4),
            round(float(roomplan_range.GetMax()[2]), 4),
        ]
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden_roomplan_floors.append(str(prim.GetPath()))
    hidden_static_visual_conflicts = reconcile_static_visuals_with_validated_route(stage, mission)
    hidden_primary_principal_duplicates = hide_primary_principal_furniture_duplicates(stage)
    hidden_movable_conflicts = hide_movable_route_conflicts(stage, mission)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(520.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    key.CreateIntensityAttr(2900.0)
    key.CreateAngleAttr(3.5)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 24.0, 22.0))
    fill = UsdLux.DistantLight.Define(stage, "/World/Lighting/Fill")
    fill.CreateIntensityAttr(1100.0)
    fill.CreateAngleAttr(7.0)
    UsdGeom.Xformable(fill).AddRotateXYZOp().Set(Gf.Vec3f(-65.0, -35.0, -20.0))

    world.GetPrim().SetCustomDataByKey("aisha:phase", "PHASE7J")
    world.GetPrim().SetCustomDataByKey("aisha:visual_layer", "complete_roomplan_capture")
    world.GetPrim().SetCustomDataByKey(
        "aisha:collision_layer", "frozen_phase7i_route_critical_geometry"
    )
    world.GetPrim().SetCustomDataByKey(
        "aisha:vp_interior", "assumed_locked_not_captured"
    )
    stage.GetRootLayer().Save()
    output.write_text(output.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    return {
        "route_trace_points": route_points,
        "source_motion_live_policy": True,
        "presentation_motion_is_recorded_pose_replay": True,
        "visual_collision_layers_separated": True,
        "visible_plan_authority_floor_with_atrium_step_down": True,
        "floor_elevation_ranges": floor_elevation_ranges,
        "presentation_hidden_roomplan_floor_count": len(hidden_roomplan_floors),
        "presentation_hidden_roomplan_floors": hidden_roomplan_floors,
        "presentation_hidden_movable_route_conflicts": hidden_movable_conflicts,
        "presentation_hidden_movable_route_conflict_count": len(hidden_movable_conflicts),
        "presentation_hidden_static_visual_route_conflicts": hidden_static_visual_conflicts,
        "presentation_hidden_static_visual_route_conflict_count": len(hidden_static_visual_conflicts),
        "presentation_hidden_primary_principal_furniture_duplicates": hidden_primary_principal_duplicates,
        "presentation_hidden_primary_principal_furniture_duplicate_count": len(hidden_primary_principal_duplicates),
        "full_capture_layer_retains_all_captured_furniture": True,
        "vice_principal_interior_assumed": True,
    }


def main() -> int:
    primary = ARGS.primary.resolve()
    supplement = ARGS.principal_supplement.resolve()
    mission = ARGS.mission.resolve()
    for required in (primary, supplement, mission, PACKAGE_ROOT / "scenes/administration.usd"):
        if not required.is_file():
            raise FileNotFoundError(required)

    visual_output = ARGS.visual_output.resolve()
    scene_output = ARGS.scene_output.resolve()
    scene_output.parent.mkdir(parents=True, exist_ok=True)
    visual_result = build_visual_layer(primary, supplement, visual_output)
    composite_result = build_composite_scene(visual_output, mission, scene_output)
    report = {
        "report_type": "phase7j_complete_captured_administration_build",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_captured_area_semantic_twin_built",
        "passed": True,
        "sources": {
            "primary_roomplan_sha256": sha256_file(primary),
            "principal_supplement_sha256": sha256_file(supplement),
            "raw_capture_committed": False,
        },
        "registration": {
            "primary": {
                "native_anchor_xz_m": list(PRIMARY_ANCHOR_XZ_M),
                "world_anchor_xy_m": list(PRIMARY_WORLD_ANCHOR_XY_M),
                "world_yaw_deg": PRIMARY_WORLD_YAW_DEG,
                "world_z_offset_m": PRIMARY_WORLD_Z_OFFSET_M,
                "metric_scale": 1.0,
            },
            "principal_supplement": {
                "native_anchor_xz_m": list(PRINCIPAL_NATIVE_ANCHOR_XZ_M),
                "world_anchor_xy_m": list(PRINCIPAL_WORLD_ANCHOR_XY_M),
                "world_yaw_deg": PRINCIPAL_WORLD_YAW_DEG,
                "world_z_offset_m": PRINCIPAL_WORLD_Z_OFFSET_M,
                "metric_scale": 1.0,
                "crop_world_xy_m": list(PRINCIPAL_CROP_WORLD_XY_M),
            },
            "global_topology_authority": "approved A1 page 2 Block A plan",
        },
        "visual_layer": {
            **visual_result,
            "path": str(visual_output),
            "sha256": sha256_file(visual_output),
            "privacy_safe_semantic_geometry_only": True,
        },
        "composite_scene": {
            **composite_result,
            "path": str(scene_output),
            "sha256": sha256_file(scene_output),
        },
        "claim_boundary": {
            "complete_area_user_captured_by_primary_roomplan_included": True,
            "photoreal_texture_capture_complete": False,
            "locked_vice_principal_interior_captured": False,
            "collision_geometry_is_raw_scan": False,
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
        APP.close()
