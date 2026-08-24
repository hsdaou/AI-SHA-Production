#!/usr/bin/env python3
"""Build or gate the plan-derived Block A administration presentation scene."""

from __future__ import annotations

import argparse
import copy
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from aisha_common import CONFIG_DIR, PACKAGE_ROOT, RESULTS_DIR, SCENES_DIR, USD_DIR, ensure_output_dirs, load_yaml, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--payload", choices=("empty", "loaded"), default="loaded")
    parser.add_argument("--plan", type=Path, help="approved ground-floor plan PDF containing page 2 Block A")
    parser.add_argument("--door-survey", type=Path, help="YAML/JSON with both clear widths and threshold heights")
    parser.add_argument(
        "--measured-geometry",
        type=Path,
        help="validated YAML overlay written by tools/prepare_measured_administration.py",
    )
    parser.add_argument(
        "--presentation-assumptions",
        action="store_true",
        help="accept disclosed presentation-only door/threshold and height assumptions",
    )
    return parser.parse_args()


def deep_merge(base: dict, overlay: dict) -> dict:
    """Return a recursive merge without mutating either input mapping."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_measured_overlay(path: Path) -> dict:
    overlay = load_yaml(path)
    if overlay.get("overlay_type") != "measured_administration_geometry":
        raise ValueError("measured geometry file has the wrong overlay_type")
    status = overlay.get("status")
    if status not in {"measured_site_candidate", "measured_site_presentation_candidate"}:
        raise ValueError("measured geometry file is not an accepted candidate type")
    if status == "measured_site_candidate":
        if overlay.get("candidate_route_geometry_valid") is not True:
            raise ValueError("measured geometry failed one or more route-clearance gates")
    else:
        if overlay.get("candidate_simulation_route_geometry_valid") is not True:
            raise ValueError("measured presentation geometry failed its simulation clearance gate")
        if overlay.get("candidate_route_geometry_valid") is not False:
            raise ValueError("presentation geometry must not claim the production route gate")
        profile = overlay.get("presentation_clearance_profile", {})
        if profile.get("simulation_only") is not True:
            raise ValueError("tight-door presentation profile must be explicitly simulation-only")
    if overlay.get("physical_release") is not False:
        raise ValueError("measured geometry must preserve physical_release: false")
    return overlay


def strict_gate(args: argparse.Namespace) -> int:
    ensure_output_dirs()
    blockers = []
    if args.plan is None or not args.plan.is_file():
        blockers.append("approved ground-floor plan page 2 is missing")
    if args.door_survey is None or not args.door_survey.is_file():
        blockers.append("Principal/Vice-Principal clear-width and threshold survey is missing")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "inputs_present",
        "blockers": blockers,
        "plan": str(args.plan.resolve()) if args.plan and args.plan.exists() else None,
        "door_survey": str(args.door_survey.resolve()) if args.door_survey and args.door_survey.exists() else None,
        "walkthrough_policy": "appearance_only_never_scale_source",
        "presentation_override_available": "rerun with --presentation-assumptions",
    }
    output = RESULTS_DIR / "administration_build_gate.json"
    write_json(output, report)
    if blockers:
        print("administration scene not built in strict mode:")
        for blocker in blockers:
            print(f"  - {blocker}")
        print(f"wrote {output}")
        return 2
    print("Inputs are present; run the presentation builder or supply surveyed doors for physical validation.")
    print(f"wrote {output}")
    return 0


def build_presentation(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})
    try:
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        ensure_output_dirs()
        config_path = CONFIG_DIR / "administration_assumptions.yaml"
        config = load_yaml(config_path)
        measured_overlay = None
        measured_overlay_hash = None
        if args.measured_geometry is not None:
            if not args.measured_geometry.is_file():
                raise FileNotFoundError(f"missing measured geometry overlay {args.measured_geometry}")
            measured_overlay = load_measured_overlay(args.measured_geometry)
            measured_overlay_hash = sha256_file(args.measured_geometry)
            config = deep_merge(config, measured_overlay)
        visual_twin = config.get("measured_visual_twin", {})
        visual_twin_enabled = bool(visual_twin.get("enabled", False))
        principal_twin = visual_twin.get("principal", {}) if visual_twin_enabled else {}
        refinement_path = CONFIG_DIR / "geometry_rtx_refinement.yaml"
        refinement = load_yaml(refinement_path)
        physics = load_yaml(CONFIG_DIR / "physics_materials.yaml")
        sensors = load_yaml(CONFIG_DIR / "sensors.yaml")
        expected_plan_hash = str(config["provenance"]["plan_source"]["sha256"])
        supplied_plan_hash = sha256_file(args.plan) if args.plan and args.plan.is_file() else None
        if str(refinement["source"]["sha256"]) != expected_plan_hash:
            raise ValueError("geometry/RTX refinement source does not match the reviewed plan")
        if supplied_plan_hash is not None and supplied_plan_hash != expected_plan_hash:
            raise ValueError(
                "the supplied plan does not match the reviewed page-2 source: "
                f"expected {expected_plan_hash}, received {supplied_plan_hash}"
            )

        asset = USD_DIR / ("aisha_loaded.usd" if args.payload == "loaded" else "aisha_empty.usd")
        if not asset.exists():
            raise FileNotFoundError(f"missing {asset}; run scripts/import_urdf.py first")

        path = SCENES_DIR / "administration.usd"
        if path.exists():
            path.unlink()
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(float(physics["physics"]["gravity_mps2"]))
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        physx_scene.CreateEnableCCDAttr(bool(physics["physics"]["enable_ccd"]))
        physx_scene.CreateEnableStabilizationAttr(bool(physics["physics"]["enable_stabilization"]))
        physx_scene.CreateEnableGPUDynamicsAttr(bool(physics["physics"]["gpu_dynamics"]))
        physx_scene.CreateBroadphaseTypeAttr(str(physics["physics"]["broadphase"]))
        physx_scene.CreateSolverTypeAttr(str(physics["physics"]["solver"]))

        def physics_material(name: str, values: dict[str, object]) -> UsdShade.Material:
            material = UsdShade.Material.Define(stage, f"/World/Looks/Physics_{name}")
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(float(values["static_friction"]))
            api.CreateDynamicFrictionAttr(float(values["dynamic_friction"]))
            api.CreateRestitutionAttr(float(values["restitution"]))
            physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            physx_api.CreateFrictionCombineModeAttr(str(values["friction_combine_mode"]))
            physx_api.CreateRestitutionCombineModeAttr(str(values["restitution_combine_mode"]))
            return material

        def visual_material(
            name: str,
            color: tuple[float, float, float],
            *,
            roughness: float = 0.5,
            metallic: float = 0.0,
            opacity: float = 1.0,
            emissive: tuple[float, float, float] | None = None,
        ) -> UsdShade.Material:
            material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
            if opacity < 1.0:
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            if emissive is not None:
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material

        def textured_material(
            name: str,
            albedo_path: Path,
            *,
            roughness_path: Path | None = None,
            normal_path: Path | None = None,
            roughness: float = 0.5,
            metallic: float = 0.0,
        ) -> UsdShade.Material:
            """Create a portable USD Preview Surface texture network."""
            if not albedo_path.is_file():
                raise FileNotFoundError(
                    f"missing {albedo_path}; run tools/generate_administration_textures.py"
                )
            if roughness_path is not None and not roughness_path.is_file():
                raise FileNotFoundError(
                    f"missing {roughness_path}; run tools/generate_administration_textures.py"
                )
            if normal_path is not None and not normal_path.is_file():
                raise FileNotFoundError(
                    f"missing {normal_path}; run tools/generate_administration_textures.py"
                )
            material_path = f"/World/Looks/{name}"
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))

            st_reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarST")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            def texture_node(node_name: str, texture_path: Path, color_space: str) -> UsdShade.Shader:
                texture = UsdShade.Shader.Define(stage, material_path + "/" + node_name)
                texture.CreateIdAttr("UsdUVTexture")
                relative_path = os.path.relpath(texture_path, path.parent)
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative_path))
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
                texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    st_reader.ConnectableAPI(), "result"
                )
                texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
                return texture

            albedo = texture_node("Albedo", albedo_path, "sRGB")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                albedo.ConnectableAPI(), "rgb"
            )
            if roughness_path is not None:
                roughness_texture = texture_node("Roughness", roughness_path, "raw")
                shader.GetInput("roughness").ConnectToSource(
                    roughness_texture.ConnectableAPI(), "r"
                )
            if normal_path is not None:
                normal_texture = texture_node("Normal", normal_path, "raw")
                shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                    normal_texture.ConnectableAPI(), "rgb"
                )
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material

        tile_physics = physics_material("polished_tile", physics["materials"]["polished_tile"])
        drive_material = physics_material("drive_wheel", physics["materials"]["drive_wheel"])
        castor_material = physics_material("castor_low_friction", physics["materials"]["castor_low_friction"])

        texture_dir = PACKAGE_ROOT / "textures" / "administration"
        texture_paths = tuple(
            texture_dir / filename
            for filename in (
                "terrazzo_albedo.png",
                "terrazzo_roughness.png",
                "terrazzo_normal.png",
                "walnut_albedo.png",
                "walnut_roughness.png",
                "walnut_normal.png",
                "oak_albedo.png",
                "oak_roughness.png",
                "oak_normal.png",
                "grey_oak_albedo.png",
                "grey_oak_roughness.png",
                "grey_oak_normal.png",
                "mottled_grey_albedo.png",
                "mottled_grey_roughness.png",
                "mottled_grey_normal.png",
            )
        )
        warm_white = visual_material("WarmWhite", (0.82, 0.81, 0.77), roughness=0.68)
        light_grey = visual_material("LightGrey", (0.53, 0.56, 0.58), roughness=0.58)
        dark_grey = visual_material("DarkGrey", (0.11, 0.13, 0.15), roughness=0.48)
        black = visual_material("Black", (0.018, 0.021, 0.024), roughness=0.33)
        terrazzo = visual_material("PolishedTerrazzo", (0.53, 0.55, 0.54), roughness=0.17)
        timber = visual_material("WarmTimber", (0.15, 0.047, 0.018), roughness=0.31)
        timber_light = visual_material("LightTimber", (0.29, 0.105, 0.033), roughness=0.35)
        oak = visual_material("LightOakFloor", (0.58, 0.49, 0.36), roughness=0.43)
        grey_oak = visual_material("GreyOakFloor", (0.53, 0.52, 0.49), roughness=0.46)
        green = visual_material("SchoolGreen", (0.055, 0.255, 0.14), roughness=0.46)
        green_accent = visual_material("SchoolGreenAccent", (0.035, 0.39, 0.21), roughness=0.32)
        cabinet_mint = visual_material("CabinetMint", (0.28, 0.49, 0.39), roughness=0.38)
        leaf_green = visual_material("PlantGreen", (0.035, 0.20, 0.07), roughness=0.72)
        leaf_light = visual_material("PlantLightGreen", (0.08, 0.31, 0.12), roughness=0.68)
        glass = visual_material("FrostedGlass", (0.52, 0.64, 0.68), roughness=0.23, opacity=0.34)
        metal = visual_material("BrushedMetal", (0.38, 0.41, 0.43), roughness=0.20, metallic=0.72)
        bronze = visual_material("WarmBronze", (0.26, 0.16, 0.075), roughness=0.26, metallic=0.60)
        paper = visual_material("Paper", (0.88, 0.87, 0.82), roughness=0.78)
        aisha_white = visual_material("AISHAWhite", (0.68, 0.73, 0.74), roughness=0.20, metallic=0.08)
        aisha_green = visual_material("AISHAGreen", (0.03, 0.38, 0.24), roughness=0.30, metallic=0.05)
        aisha_black = visual_material("AISHABlack", (0.015, 0.022, 0.025), roughness=0.20, metallic=0.20)
        aisha_led = visual_material(
            "AISHALed", (0.01, 0.24, 0.12), roughness=0.16, emissive=(0.0, 1.35, 0.55)
        )
        terrazzo_finish = textured_material(
            "TerrazzoFinish",
            texture_dir / "terrazzo_albedo.png",
            roughness_path=texture_dir / "terrazzo_roughness.png",
            normal_path=texture_dir / "terrazzo_normal.png",
            roughness=0.17,
        )
        walnut_finish = textured_material(
            "WalnutFinish",
            texture_dir / "walnut_albedo.png",
            roughness_path=texture_dir / "walnut_roughness.png",
            normal_path=texture_dir / "walnut_normal.png",
            roughness=0.31,
        )
        oak_finish = textured_material(
            "OakFinish",
            texture_dir / "oak_albedo.png",
            roughness_path=texture_dir / "oak_roughness.png",
            normal_path=texture_dir / "oak_normal.png",
            roughness=0.43,
        )
        grey_oak_finish = textured_material(
            "GreyOakFinish",
            texture_dir / "grey_oak_albedo.png",
            roughness_path=texture_dir / "grey_oak_roughness.png",
            normal_path=texture_dir / "grey_oak_normal.png",
            roughness=0.46,
        )
        mottled_grey_finish = textured_material(
            "MottledGreyFinish",
            texture_dir / "mottled_grey_albedo.png",
            roughness_path=texture_dir / "mottled_grey_roughness.png",
            normal_path=texture_dir / "mottled_grey_normal.png",
            roughness=0.57,
        )
        light_panel = visual_material(
            "LightPanel",
            (0.92, 0.96, 1.00),
            roughness=0.12,
            emissive=(5.0, 5.2, 5.5),
        )

        def bind_physics(prim: Usd.Prim, material: UsdShade.Material) -> None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

        def bind_visual(prim: Usd.Prim, material: UsdShade.Material) -> None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

        def box(
            prim_path: str,
            size_xyz: tuple[float, float, float],
            centre_xyz: tuple[float, float, float],
            material: UsdShade.Material,
            *,
            collision: bool = True,
            physics_binding: UsdShade.Material | None = None,
            rotate_z_deg: float = 0.0,
        ) -> Usd.Prim:
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*centre_xyz))
            if rotate_z_deg:
                xform.AddRotateZOp().Set(float(rotate_z_deg))
            xform.AddScaleOp().Set(Gf.Vec3d(*size_xyz))
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            bind_visual(cube.GetPrim(), material)
            if physics_binding is not None:
                bind_physics(cube.GetPrim(), physics_binding)
            return cube.GetPrim()

        def sphere(
            prim_path: str,
            radius: float,
            centre_xyz: tuple[float, float, float],
            material: UsdShade.Material,
            *,
            collision: bool = True,
        ) -> Usd.Prim:
            shape = UsdGeom.Sphere.Define(stage, prim_path)
            shape.CreateRadiusAttr(radius)
            xform = UsdGeom.Xformable(shape.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*centre_xyz))
            bind_visual(shape.GetPrim(), material)
            if collision:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            return shape.GetPrim()

        def ellipsoid(
            prim_path: str,
            size_xyz: tuple[float, float, float],
            centre_xyz: tuple[float, float, float],
            material: UsdShade.Material,
            *,
            collision: bool = False,
            rotate_xyz_deg: tuple[float, float, float] | None = None,
            rotate_z_deg: float = 0.0,
        ) -> Usd.Prim:
            shape = UsdGeom.Sphere.Define(stage, prim_path)
            shape.CreateRadiusAttr(0.5)
            xform = UsdGeom.Xformable(shape.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*centre_xyz))
            if rotate_xyz_deg is not None:
                xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz_deg))
            elif rotate_z_deg:
                xform.AddRotateZOp().Set(float(rotate_z_deg))
            xform.AddScaleOp().Set(Gf.Vec3d(*size_xyz))
            bind_visual(shape.GetPrim(), material)
            if collision:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            return shape.GetPrim()

        def cylinder(
            prim_path: str,
            radius: float,
            height: float,
            centre_xyz: tuple[float, float, float],
            material: UsdShade.Material,
            *,
            collision: bool = True,
        ) -> Usd.Prim:
            shape = UsdGeom.Cylinder.Define(stage, prim_path)
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(height)
            xform = UsdGeom.Xformable(shape.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*centre_xyz))
            if collision:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            bind_visual(shape.GetPrim(), material)
            return shape.GetPrim()

        def polygon_floor(
            name: str,
            points_xy: list[tuple[float, float]],
            material: UsdShade.Material,
            *,
            z: float = 0.002,
        ) -> Usd.Prim:
            prim_path = name if name.startswith("/") else f"/World/Architecture/Floors/{name}"
            mesh = UsdGeom.Mesh.Define(stage, prim_path)
            mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x, y in points_xy])
            mesh.CreateFaceVertexCountsAttr([len(points_xy)])
            mesh.CreateFaceVertexIndicesAttr(list(range(len(points_xy))))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            bind_visual(mesh.GetPrim(), material)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            bind_physics(mesh.GetPrim(), tile_physics)
            return mesh.GetPrim()

        def textured_polygon_surface(
            name: str,
            points_xy: list[tuple[float, float]],
            material: UsdShade.Material,
            *,
            z: float = 0.006,
            metres_per_tile: float = 2.0,
        ) -> Usd.Prim:
            """Add a render-only textured finish above an existing collision floor."""
            mesh = UsdGeom.Mesh.Define(stage, f"/World/Appearance/SurfaceFinishes/{name}")
            mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x, y in points_xy])
            mesh.CreateFaceVertexCountsAttr([len(points_xy)])
            mesh.CreateFaceVertexIndicesAttr(list(range(len(points_xy))))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
            )
            st.Set([(x / metres_per_tile, y / metres_per_tile) for x, y in points_xy])
            bind_visual(mesh.GetPrim(), material)
            mesh.GetPrim().SetCustomDataByKey("aisha:collision", "none_visual_finish")
            return mesh.GetPrim()

        def textured_rect_surface(
            name: str,
            size_xy: tuple[float, float],
            centre_xy: tuple[float, float],
            material: UsdShade.Material,
            *,
            z: float = 0.006,
            rotate_z_deg: float = 0.0,
            metres_per_tile: float = 2.0,
        ) -> Usd.Prim:
            sx, sy = size_xy
            mesh = UsdGeom.Mesh.Define(stage, f"/World/Appearance/SurfaceFinishes/{name}")
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(-sx / 2.0, -sy / 2.0, 0.0),
                    Gf.Vec3f(sx / 2.0, -sy / 2.0, 0.0),
                    Gf.Vec3f(sx / 2.0, sy / 2.0, 0.0),
                    Gf.Vec3f(-sx / 2.0, sy / 2.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
            )
            st.Set(
                [
                    (0.0, 0.0),
                    (sx / metres_per_tile, 0.0),
                    (sx / metres_per_tile, sy / metres_per_tile),
                    (0.0, sy / metres_per_tile),
                ]
            )
            xform = UsdGeom.Xformable(mesh.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(centre_xy[0], centre_xy[1], z))
            if rotate_z_deg:
                xform.AddRotateZOp().Set(float(rotate_z_deg))
            bind_visual(mesh.GetPrim(), material)
            mesh.GetPrim().SetCustomDataByKey("aisha:collision", "none_visual_finish")
            return mesh.GetPrim()

        def textured_wall_surface(
            name: str,
            start_xy: tuple[float, float],
            end_xy: tuple[float, float],
            material: UsdShade.Material,
            *,
            normal_offset: float = 0.091,
            height: float = 2.72,
            metres_per_tile: float = 2.2,
        ) -> Usd.Prim:
            dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
            length = math.hypot(dx, dy)
            nx, ny = -dy / length, dx / length
            start = (start_xy[0] + nx * normal_offset, start_xy[1] + ny * normal_offset)
            end = (end_xy[0] + nx * normal_offset, end_xy[1] + ny * normal_offset)
            mesh = UsdGeom.Mesh.Define(stage, f"/World/Appearance/WallFinishes/{name}")
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(start[0], start[1], 0.0),
                    Gf.Vec3f(end[0], end[1], 0.0),
                    Gf.Vec3f(end[0], end[1], height),
                    Gf.Vec3f(start[0], start[1], height),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
            )
            st.Set(
                [
                    (0.0, 0.0),
                    (length / metres_per_tile, 0.0),
                    (length / metres_per_tile, height / metres_per_tile),
                    (0.0, height / metres_per_tile),
                ]
            )
            bind_visual(mesh.GetPrim(), material)
            mesh.GetPrim().SetCustomDataByKey("aisha:collision", "none_visual_finish")
            return mesh.GetPrim()

        wall_height = float(config["plan_geometry"]["wall_height_m"]["value"])
        wall_thickness = float(config["plan_geometry"]["wall_thickness_m"]["value"])

        def wall_segment(
            name: str,
            start_xy: tuple[float, float],
            end_xy: tuple[float, float],
            material: UsdShade.Material = warm_white,
        ) -> None:
            dx = end_xy[0] - start_xy[0]
            dy = end_xy[1] - start_xy[1]
            length = math.hypot(dx, dy)
            if length <= 1e-4:
                return
            centre = ((start_xy[0] + end_xy[0]) / 2.0, (start_xy[1] + end_xy[1]) / 2.0, wall_height / 2.0)
            box(
                f"/World/Architecture/Walls/{name}",
                (length, wall_thickness, wall_height),
                centre,
                material,
                rotate_z_deg=math.degrees(math.atan2(dy, dx)),
            )

        def split_wall(
            name: str,
            start_xy: tuple[float, float],
            end_xy: tuple[float, float],
            opening_xy: tuple[float, float],
            opening_width: float,
            material: UsdShade.Material = warm_white,
        ) -> None:
            dx = end_xy[0] - start_xy[0]
            dy = end_xy[1] - start_xy[1]
            length = math.hypot(dx, dy)
            tx, ty = dx / length, dy / length
            opening_t = (opening_xy[0] - start_xy[0]) * tx + (opening_xy[1] - start_xy[1]) * ty
            before = max(0.0, opening_t - opening_width / 2.0)
            after = min(length, opening_t + opening_width / 2.0)
            wall_segment(name + "_A", start_xy, (start_xy[0] + tx * before, start_xy[1] + ty * before), material)
            wall_segment(name + "_B", (start_xy[0] + tx * after, start_xy[1] + ty * after), end_xy, material)

        def local_to_world(
            centre_xy: tuple[float, float], local_xy: tuple[float, float], rotate_z_deg: float
        ) -> tuple[float, float]:
            angle = math.radians(rotate_z_deg)
            return (
                centre_xy[0] + local_xy[0] * math.cos(angle) - local_xy[1] * math.sin(angle),
                centre_xy[1] + local_xy[0] * math.sin(angle) + local_xy[1] * math.cos(angle),
            )

        def doorway(
            name: str,
            values: dict[str, object],
            *,
            hinge_left: bool,
            opens_outward: bool = False,
        ) -> dict[str, object]:
            centre_x, centre_y = (float(v) for v in values["centre_xy_m"])
            angle_deg = float(values["wall_rotation_deg"])
            angle = math.radians(angle_deg)
            tangent = (math.cos(angle), math.sin(angle))
            normal = (-math.sin(angle), math.cos(angle))
            width = float(values["clear_width_m"])
            height = float(values.get("clear_height_m", 2.25))
            frame_depth = float(values.get("frame_depth_m", 0.19))
            post = 0.075
            for side_name, side_sign in (("Left", -1.0), ("Right", 1.0)):
                offset = side_sign * (width / 2.0 + post / 2.0)
                box(
                    f"/World/Architecture/Doors/{name}/Frame{side_name}",
                    (post, frame_depth, height),
                    (centre_x + tangent[0] * offset, centre_y + tangent[1] * offset, height / 2.0),
                    metal,
                    rotate_z_deg=angle_deg,
                )
            box(
                f"/World/Architecture/Doors/{name}/Lintel",
                (width + 2.0 * post, frame_depth, 0.18),
                (centre_x, centre_y, height + 0.09),
                metal,
                rotate_z_deg=angle_deg,
            )
            hinge_sign = -1.0 if hinge_left else 1.0
            hinge = (
                centre_x + tangent[0] * hinge_sign * width / 2.0,
                centre_y + tangent[1] * hinge_sign * width / 2.0,
            )
            swing_sign = 1.0 if opens_outward else -1.0
            leaf_centre = (
                hinge[0] + normal[0] * swing_sign * width / 2.0,
                hinge[1] + normal[1] * swing_sign * width / 2.0,
            )
            open_leaf = box(
                f"/World/Architecture/Doors/{name}/OpenLeaf",
                (width, 0.045, max(0.10, height - 0.07)),
                (leaf_centre[0], leaf_centre[1], max(0.10, height - 0.07) / 2.0),
                light_grey,
                collision=False,
                rotate_z_deg=angle_deg + 90.0,
            )
            # The supplied 0.85 m value is a clear-opening measurement.  Do
            # not subtract the presentation leaf thickness a second time;
            # the filmed route assumes each door is secured fully open.
            open_leaf.SetCustomDataByKey(
                "aisha:collision", "none_visual_only_secured_fully_open"
            )
            leaf_angle = angle_deg + 90.0
            leaf_tangent = (math.cos(math.radians(leaf_angle)), math.sin(math.radians(leaf_angle)))
            leaf_start = (
                leaf_centre[0] - leaf_tangent[0] * width / 2.0,
                leaf_centre[1] - leaf_tangent[1] * width / 2.0,
            )
            leaf_end = (
                leaf_centre[0] + leaf_tangent[0] * width / 2.0,
                leaf_centre[1] + leaf_tangent[1] * width / 2.0,
            )
            textured_wall_surface(
                f"{name}DoorFinish",
                leaf_start,
                leaf_end,
                mottled_grey_finish,
                normal_offset=0.024,
                height=max(0.10, height - 0.07),
                metres_per_tile=1.2,
            )
            free_edge = (
                hinge[0] + normal[0] * swing_sign * (width - 0.12),
                hinge[1] + normal[1] * swing_sign * (width - 0.12),
            )
            box(
                f"/World/Architecture/Doors/{name}/Handle",
                (0.035, 0.15, 0.030),
                (*free_edge, 1.02),
                metal,
                collision=False,
                rotate_z_deg=leaf_angle,
            )
            threshold_m = float(values["threshold_height_mm"]) / 1000.0
            if threshold_m > 0.0:
                threshold = box(
                    f"/World/Architecture/Doors/{name}/Threshold",
                    (width, 0.12, threshold_m),
                    (centre_x, centre_y, threshold_m / 2.0),
                    metal,
                    physics_binding=tile_physics,
                    rotate_z_deg=angle_deg,
                )
                threshold.SetCustomDataByKey("aisha:status", str(values["threshold_status"]))
                threshold.SetCustomDataByKey("aisha:heightMm", int(values["threshold_height_mm"]))
            else:
                door_prim = stage.GetPrimAtPath(f"/World/Architecture/Doors/{name}")
                door_prim.SetCustomDataByKey("aisha:thresholdStatus", str(values["threshold_status"]))
                door_prim.SetCustomDataByKey("aisha:thresholdHeightMm", 0)
            sign_normal = (normal[0] * 0.10, normal[1] * 0.10)
            box(
                f"/World/Architecture/Doors/{name}/Plaque",
                (0.55, 0.035, 0.20),
                (centre_x + tangent[0] * (width / 2.0 + 0.42) + sign_normal[0],
                 centre_y + tangent[1] * (width / 2.0 + 0.42) + sign_normal[1],
                 1.55),
                green,
                collision=False,
                rotate_z_deg=angle_deg,
            )
            return {
                "clear_width_m": width,
                "clear_height_m": height,
                "frame_depth_m": frame_depth,
                "threshold_height_mm": int(values["threshold_height_mm"]),
                "width_status": values["width_status"],
                "threshold_status": values["threshold_status"],
                "centre_xy_m": [centre_x, centre_y],
                "wall_rotation_deg": angle_deg,
                "swing": values.get(
                    "swing_from_hallway",
                    "outward_for_camera_and_route_clearance" if opens_outward else "inward",
                ),
                "open_leaf_collision": "visual_only_secured_fully_open",
                "hinge": values.get(
                    "hinge_side_from_hallway", "left" if hinge_left else "right"
                ),
            }

        def slatted_wall(name: str, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
            wall_segment(name + "_Backing", start_xy, end_xy, timber)
            textured_wall_surface(name + "_WalnutFinish", start_xy, end_xy, walnut_finish)
            dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
            length = math.hypot(dx, dy)
            tx, ty = dx / length, dy / length
            nx, ny = -ty, tx
            count = max(1, int(length / 0.12))
            angle = math.degrees(math.atan2(dy, dx))
            for index in range(count + 1):
                t = min(length, index * length / count)
                box(
                    f"/World/Appearance/TimberSlats/{name}_{index:03d}",
                    (0.035, 0.055, 2.72),
                    (start_xy[0] + tx * t + nx * 0.10, start_xy[1] + ty * t + ny * 0.10, 1.36),
                    timber_light,
                    collision=False,
                    rotate_z_deg=angle,
                )

        def glazed_partition(
            name: str,
            start_xy: tuple[float, float],
            end_xy: tuple[float, float],
            *,
            blinds: bool = False,
        ) -> None:
            """Grey-framed full-height glazing used in the captured office suite."""
            dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
            length = math.hypot(dx, dy)
            if length <= 1.0e-4:
                return
            yaw = math.degrees(math.atan2(dy, dx))
            centre = ((start_xy[0] + end_xy[0]) / 2.0, (start_xy[1] + end_xy[1]) / 2.0)
            box(
                f"/World/Architecture/Glass/{name}/Pane",
                (length, 0.055, 2.42),
                (*centre, 1.42),
                glass,
                rotate_z_deg=yaw,
            )
            for label, point in (("Start", start_xy), ("End", end_xy)):
                box(
                    f"/World/Architecture/Glass/{name}/Frame{label}",
                    (0.055, 0.085, 2.72),
                    (*point, 1.36),
                    metal,
                    collision=False,
                    rotate_z_deg=yaw,
                )
            for z in (0.22, 1.18, 2.62):
                box(
                    f"/World/Architecture/Glass/{name}/Rail_{int(z * 100):03d}",
                    (length, 0.075, 0.045),
                    (*centre, z),
                    metal,
                    collision=False,
                    rotate_z_deg=yaw,
                )
            mullions = max(1, int(length / 1.05))
            tangent = (dx / length, dy / length)
            for index in range(1, mullions + 1):
                offset = -length / 2.0 + length * index / (mullions + 1)
                point = (centre[0] + tangent[0] * offset, centre[1] + tangent[1] * offset)
                box(
                    f"/World/Architecture/Glass/{name}/Mullion_{index:02d}",
                    (0.040, 0.080, 2.42),
                    (*point, 1.42),
                    metal,
                    collision=False,
                    rotate_z_deg=yaw,
                )
            if blinds:
                for index in range(18):
                    z = 0.62 + index * 0.105
                    box(
                        f"/World/Architecture/Glass/{name}/Blind_{index:02d}",
                        (max(0.10, length - 0.10), 0.018, 0.045),
                        (*centre, z),
                        paper,
                        collision=False,
                        rotate_z_deg=yaw,
                    )

        def registered_wall(name: str, values: dict[str, object]) -> None:
            start = tuple(float(value) for value in values["start_xy_m"])
            end = tuple(float(value) for value in values["end_xy_m"])
            style = str(values.get("style", "warm_white"))
            if style == "grey_frame_glass":
                glazed_partition(name, start, end)
            elif style == "white_blinds":
                glazed_partition(name, start, end, blinds=True)
            elif style in {"walnut_slatted", "walnut_and_warm_white"}:
                slatted_wall(name, start, end)
            elif style == "walnut_tv_wall":
                wall_segment(name, start, end, timber)
                textured_wall_surface(name + "_WalnutFinish", start, end, walnut_finish)
            else:
                wall_segment(name, start, end, warm_white)

        def registered_table(name: str, values: dict[str, object]) -> None:
            centre = tuple(float(value) for value in values["centre_xy_m"])
            sx, sy, height = (float(value) for value in values["size_xyz_m"])
            yaw = float(values["yaw_deg"])
            top_thickness = 0.085
            box(
                f"/World/Furniture/PrincipalMeasured/{name}/Top",
                (sx, sy, top_thickness),
                (*centre, height - top_thickness / 2.0),
                timber_light,
                rotate_z_deg=yaw,
            )
            textured_rect_surface(
                f"PrincipalMeasured_{name}_TopFinish",
                (max(0.10, sx - 0.04), max(0.10, sy - 0.04)),
                centre,
                walnut_finish,
                z=height + 0.004,
                rotate_z_deg=yaw,
                metres_per_tile=1.5,
            )
            for index, (local_x, local_y) in enumerate(
                ((-sx * 0.38, -sy * 0.36), (-sx * 0.38, sy * 0.36), (sx * 0.38, -sy * 0.36), (sx * 0.38, sy * 0.36))
            ):
                point = local_to_world(centre, (local_x, local_y), yaw)
                box(
                    f"/World/Furniture/PrincipalMeasured/{name}/Leg_{index:02d}",
                    (0.055, 0.055, max(0.20, height - top_thickness)),
                    (*point, max(0.20, height - top_thickness) / 2.0),
                    metal,
                    rotate_z_deg=yaw,
                )

        def registered_storage(name: str, values: dict[str, object]) -> None:
            centre = tuple(float(value) for value in values["centre_xy_m"])
            sx, sy, height = (float(value) for value in values["size_xyz_m"])
            yaw = float(values["yaw_deg"])
            role = str(values.get("role", "storage"))
            material = cabinet_mint if role == "tv_award_cabinet" else timber
            box(
                f"/World/Furniture/PrincipalMeasured/{name}/Body",
                (sx, max(0.12, sy), height),
                (*centre, height / 2.0),
                material,
                rotate_z_deg=yaw,
            )
            door_count = max(1, int(sx / 0.42))
            for index in range(door_count):
                local_x = -sx / 2.0 + sx * (index + 0.5) / door_count
                face_xy = local_to_world(centre, (local_x, -max(0.12, sy) / 2.0 - 0.012), yaw)
                box(
                    f"/World/Furniture/PrincipalMeasured/{name}/Door_{index:02d}",
                    (max(0.12, sx / door_count - 0.025), 0.018, max(0.15, height - 0.10)),
                    (*face_xy, height / 2.0),
                    cabinet_mint if role == "tv_award_cabinet" else timber_light,
                    collision=False,
                    rotate_z_deg=yaw,
                )

        def sofa(
            name: str,
            centre_xy: tuple[float, float],
            yaw_deg: float,
            *,
            floor_z: float = 0.0,
        ) -> None:
            box(f"/World/Furniture/{name}/Seat", (1.70, 0.72, 0.18), (*centre_xy, floor_z + 0.42), black, rotate_z_deg=yaw_deg)
            back_xy = local_to_world(centre_xy, (0.0, 0.31), yaw_deg)
            box(f"/World/Furniture/{name}/Back", (1.70, 0.16, 0.68), (*back_xy, floor_z + 0.70), black, rotate_z_deg=yaw_deg)
            for side, local_x in (("L", -0.79), ("R", 0.79)):
                arm_xy = local_to_world(centre_xy, (local_x, 0.0), yaw_deg)
                box(f"/World/Furniture/{name}/Arm{side}", (0.14, 0.70, 0.42), (*arm_xy, floor_z + 0.48), black, rotate_z_deg=yaw_deg)

        def desk(name: str, centre_xy: tuple[float, float], yaw_deg: float = 0.0) -> None:
            box(f"/World/Furniture/{name}/Top", (2.00, 0.82, 0.09), (*centre_xy, 0.76), timber_light, rotate_z_deg=yaw_deg)
            textured_rect_surface(
                f"{name}_DesktopFinish",
                (1.96, 0.78),
                centre_xy,
                walnut_finish,
                z=0.807,
                rotate_z_deg=yaw_deg,
                metres_per_tile=1.7,
            )
            for pedestal_index, local_y in enumerate((-0.31, 0.31)):
                position = local_to_world(centre_xy, (-0.78, local_y), yaw_deg)
                box(f"/World/Furniture/{name}/Pedestal_{pedestal_index}", (0.34, 0.26, 0.68), (*position, 0.36), timber, rotate_z_deg=yaw_deg)
                drawer_face = local_to_world(position, (0.18, 0.0), yaw_deg)
                for drawer_index, z in enumerate((0.24, 0.44, 0.62)):
                    box(
                        f"/World/Furniture/{name}/Drawer_{pedestal_index}_{drawer_index}",
                        (0.018, 0.22, 0.13),
                        (*drawer_face, z),
                        timber_light,
                        collision=False,
                        rotate_z_deg=yaw_deg,
                    )
                    handle = local_to_world(drawer_face, (0.018, 0.0), yaw_deg)
                    box(
                        f"/World/Furniture/{name}/Handle_{pedestal_index}_{drawer_index}",
                        (0.014, 0.10, 0.014),
                        (*handle, z),
                        metal,
                        collision=False,
                        rotate_z_deg=yaw_deg,
                    )

            # Walkthrough-style workstation detail. These objects are visual-only
            # so the previously verified furniture collision envelope is intact.
            monitor_xy = local_to_world(centre_xy, (0.30, 0.0), yaw_deg)
            box(f"/World/Furniture/{name}/Monitor", (0.045, 0.62, 0.37), (*monitor_xy, 1.12), black, collision=False, rotate_z_deg=yaw_deg)
            monitor_trim_xy = local_to_world(monitor_xy, (0.026, 0.0), yaw_deg)
            box(f"/World/Furniture/{name}/MonitorScreen", (0.008, 0.55, 0.30), (*monitor_trim_xy, 1.12), dark_grey, collision=False, rotate_z_deg=yaw_deg)
            box(f"/World/Furniture/{name}/MonitorStem", (0.035, 0.055, 0.26), (*monitor_xy, 0.91), metal, collision=False, rotate_z_deg=yaw_deg)
            keyboard_xy = local_to_world(centre_xy, (-0.12, 0.0), yaw_deg)
            box(f"/World/Furniture/{name}/Keyboard", (0.26, 0.48, 0.018), (*keyboard_xy, 0.823), black, collision=False, rotate_z_deg=yaw_deg)
            paper_xy = local_to_world(centre_xy, (0.05, 0.31), yaw_deg)
            for page_index in range(4):
                box(f"/World/Furniture/{name}/Paper_{page_index}", (0.30, 0.20, 0.003), (*paper_xy, 0.820 + page_index * 0.004), paper, collision=False, rotate_z_deg=yaw_deg + page_index * 1.3)

        def chair(name: str, centre_xy: tuple[float, float], yaw_deg: float = 0.0) -> None:
            seat_collision = box(f"/World/Furniture/{name}/SeatCollision", (0.50, 0.50, 0.10), (*centre_xy, 0.48), black, rotate_z_deg=yaw_deg)
            UsdGeom.Imageable(seat_collision).MakeInvisible()
            ellipsoid(f"/World/Furniture/{name}/Seat", (0.54, 0.52, 0.13), (*centre_xy, 0.49), black, rotate_z_deg=yaw_deg)
            back_xy = local_to_world(centre_xy, (-0.24, 0.0), yaw_deg)
            back_collision = box(f"/World/Furniture/{name}/BackCollision", (0.08, 0.50, 0.72), (*back_xy, 0.78), black, rotate_z_deg=yaw_deg)
            UsdGeom.Imageable(back_collision).MakeInvisible()
            ellipsoid(f"/World/Furniture/{name}/Back", (0.12, 0.52, 0.74), (*back_xy, 0.78), black, rotate_z_deg=yaw_deg)
            for x_index, sx in enumerate((-0.18, 0.18)):
                for y_index, sy in enumerate((-0.18, 0.18)):
                    leg_xy = local_to_world(centre_xy, (sx, sy), yaw_deg)
                    box(f"/World/Furniture/{name}/Leg_{x_index}_{y_index}", (0.035, 0.035, 0.44), (*leg_xy, 0.22), metal, rotate_z_deg=yaw_deg)
            for side_index, local_y in enumerate((-0.27, 0.27)):
                arm_xy = local_to_world(centre_xy, (-0.03, local_y), yaw_deg)
                box(f"/World/Furniture/{name}/Arm_{side_index}", (0.38, 0.035, 0.035), (*arm_xy, 0.72), black, collision=False, rotate_z_deg=yaw_deg)

        def cantilever_chair(name: str, centre_xy: tuple[float, float], yaw_deg: float = 0.0) -> None:
            """Walkthrough-style black meeting chair with a brushed-metal sled base."""
            seat_collision = box(f"/World/Furniture/{name}/SeatCollision", (0.54, 0.52, 0.09), (*centre_xy, 0.48), black, rotate_z_deg=yaw_deg)
            UsdGeom.Imageable(seat_collision).MakeInvisible()
            ellipsoid(f"/World/Furniture/{name}/Seat", (0.58, 0.54, 0.12), (*centre_xy, 0.49), black, rotate_z_deg=yaw_deg)
            back_xy = local_to_world(centre_xy, (-0.25, 0.0), yaw_deg)
            back_collision = box(f"/World/Furniture/{name}/BackCollision", (0.075, 0.54, 0.62), (*back_xy, 0.76), black, rotate_z_deg=yaw_deg)
            UsdGeom.Imageable(back_collision).MakeInvisible()
            ellipsoid(f"/World/Furniture/{name}/Back", (0.11, 0.56, 0.64), (*back_xy, 0.76), black, rotate_z_deg=yaw_deg)
            for side_index, local_y in enumerate((-0.22, 0.22)):
                rail_xy = local_to_world(centre_xy, (0.02, local_y), yaw_deg)
                box(f"/World/Furniture/{name}/BaseRail_{side_index}", (0.62, 0.025, 0.025), (*rail_xy, 0.035), metal, rotate_z_deg=yaw_deg)
                support_xy = local_to_world(centre_xy, (-0.24, local_y), yaw_deg)
                box(f"/World/Furniture/{name}/BackSupport_{side_index}", (0.025, 0.025, 0.82), (*support_xy, 0.42), metal, rotate_z_deg=yaw_deg)

        def round_meeting_table(name: str, centre_xy: tuple[float, float], radius: float) -> None:
            cylinder(f"/World/Furniture/{name}/Top", radius, 0.085, (*centre_xy, 0.755), timber_light)
            cylinder(f"/World/Furniture/{name}/Pedestal", 0.14, 0.69, (*centre_xy, 0.365), black)
            cylinder(f"/World/Furniture/{name}/Foot", 0.42, 0.045, (*centre_xy, 0.035), metal)

        def plant(name: str, centre_xy: tuple[float, float]) -> None:
            cylinder(f"/World/Furniture/{name}/Pot", 0.24, 0.42, (*centre_xy, 0.21), light_grey)
            cylinder(f"/World/Furniture/{name}/PotRim", 0.255, 0.055, (*centre_xy, 0.425), bronze, collision=False)
            cylinder(f"/World/Furniture/{name}/Stem", 0.035, 0.70, (*centre_xy, 0.72), timber, collision=False)
            for index, (dx, dy, dz) in enumerate(((0.0, 0.0, 1.18), (0.24, 0.0, 1.10), (-0.22, 0.05, 1.08), (0.0, 0.22, 1.12), (0.05, -0.22, 1.06))):
                leaf_collision = sphere(f"/World/Furniture/{name}/LeafCollision_{index}", 0.24, (centre_xy[0] + dx, centre_xy[1] + dy, dz), leaf_green)
                UsdGeom.Imageable(leaf_collision).MakeInvisible()
                ellipsoid(
                    f"/World/Furniture/{name}/Leaf_{index}",
                    (0.18, 0.38, 0.08),
                    (centre_xy[0] + dx, centre_xy[1] + dy, dz),
                    leaf_green if index % 2 else leaf_light,
                    rotate_xyz_deg=(20.0 + index * 7.0, -25.0 + index * 11.0, index * 67.0),
                )

        def framed_panel(
            name: str,
            centre_xyz: tuple[float, float, float],
            *,
            size_xz: tuple[float, float] = (0.95, 1.30),
            rotate_z_deg: float = 0.0,
            inset_material: UsdShade.Material = green,
        ) -> None:
            """Walkthrough-inspired abstract display panel; no copied artwork."""
            width, height = size_xz
            box(f"/World/Appearance/WallDisplays/{name}/Backing", (width, 0.030, height), centre_xyz, black, collision=False, rotate_z_deg=rotate_z_deg)
            inset_xy = local_to_world((centre_xyz[0], centre_xyz[1]), (0.0, 0.018), rotate_z_deg)
            box(
                f"/World/Appearance/WallDisplays/{name}/Inset",
                (width - 0.10, 0.012, height - 0.10),
                (*inset_xy, centre_xyz[2]),
                inset_material,
                collision=False,
                rotate_z_deg=rotate_z_deg,
            )
            for stripe_index, stripe_z in enumerate((-0.26, 0.0, 0.26)):
                box(
                    f"/World/Appearance/WallDisplays/{name}/Stripe_{stripe_index}",
                    (width - 0.22, 0.010, 0.035),
                    (*inset_xy, centre_xyz[2] + stripe_z),
                    paper if stripe_index != 1 else bronze,
                    collision=False,
                    rotate_z_deg=rotate_z_deg,
                )

        def ceiling_grid(
            name: str,
            centre_xy: tuple[float, float],
            size_xy: tuple[float, float],
            *,
            rotate_z_deg: float = 0.0,
            spacing: float = 0.60,
        ) -> None:
            sx, sy = size_xy
            x_count = max(1, int(sx / spacing))
            y_count = max(1, int(sy / spacing))
            for index in range(x_count + 1):
                local_x = -sx / 2.0 + sx * index / x_count
                point = local_to_world(centre_xy, (local_x, 0.0), rotate_z_deg)
                box(
                    f"/World/Architecture/Ceilings/Grid/{name}_X_{index:02d}",
                    (0.014, sy - 0.06, 0.010),
                    (*point, wall_height - 0.014),
                    light_grey,
                    collision=False,
                    rotate_z_deg=rotate_z_deg,
                )
            for index in range(y_count + 1):
                local_y = -sy / 2.0 + sy * index / y_count
                point = local_to_world(centre_xy, (0.0, local_y), rotate_z_deg)
                box(
                    f"/World/Architecture/Ceilings/Grid/{name}_Y_{index:02d}",
                    (sx - 0.06, 0.014, 0.010),
                    (*point, wall_height - 0.014),
                    light_grey,
                    collision=False,
                    rotate_z_deg=rotate_z_deg,
                )

        def ceiling_sensor(name: str, centre_xy: tuple[float, float]) -> None:
            cylinder(f"/World/Architecture/Ceilings/Devices/{name}/Mount", 0.11, 0.035, (*centre_xy, wall_height - 0.030), warm_white, collision=False)
            ellipsoid(f"/World/Architecture/Ceilings/Devices/{name}/Dome", (0.16, 0.16, 0.10), (*centre_xy, wall_height - 0.075), warm_white)
            lens_xy = (centre_xy[0] + 0.045, centre_xy[1])
            sphere(f"/World/Architecture/Ceilings/Devices/{name}/Lens", 0.030, (*lens_xy, wall_height - 0.100), black, collision=False)

        def ceiling_vent(
            name: str,
            centre_xy: tuple[float, float],
            *,
            rotate_z_deg: float = 0.0,
        ) -> None:
            box(f"/World/Architecture/Ceilings/Vents/{name}/Recess", (0.72, 0.50, 0.015), (*centre_xy, wall_height - 0.027), dark_grey, collision=False, rotate_z_deg=rotate_z_deg)
            for index in range(8):
                local_y = -0.205 + index * 0.0585
                point = local_to_world(centre_xy, (0.0, local_y), rotate_z_deg)
                box(f"/World/Architecture/Ceilings/Vents/{name}/Louver_{index:02d}", (0.62, 0.014, 0.018), (*point, wall_height - 0.038), metal, collision=False, rotate_z_deg=rotate_z_deg)

        # Floors and support slab. The site capture notes that the central
        # polygon is 0.20 m below the surrounding atrium ring and is not robot
        # accessible. The support slab is therefore lowered to the drop level;
        # the surrounding floor meshes remain the robot contact surface.
        atrium_config = config["plan_geometry"]["atrium"]
        central_polygon = atrium_config["central_polygon"]
        central_step_down = float(central_polygon["step_down_m"])
        box(
            "/World/Architecture/SupportSlab",
            (48.0, 32.0, 0.12),
            (7.0, -4.0, -central_step_down - 0.06),
            dark_grey,
            physics_binding=tile_physics,
        )
        radius = float(config["known_dimensions"]["atrium_diagonal_m"]["value"]) / 2.0
        vertices = [
            (radius * math.cos(math.radians(22.5 + 45.0 * index)), radius * math.sin(math.radians(22.5 + 45.0 * index)))
            for index in range(8)
        ]
        central_radius = float(central_polygon["outer_vertex_radius_m"])
        central_orientation = float(central_polygon["orientation_deg"])
        central_centre = tuple(float(value) for value in central_polygon["centre_xy_m"])
        central_vertices = [
            (
                central_centre[0]
                + central_radius
                * math.cos(math.radians(central_orientation + 45.0 * index)),
                central_centre[1]
                + central_radius
                * math.sin(math.radians(central_orientation + 45.0 * index)),
            )
            for index in range(8)
        ]
        for index in range(8):
            polygon_floor(
                f"Atrium/WalkableRing_{index:02d}",
                [
                    vertices[index],
                    vertices[(index + 1) % 8],
                    central_vertices[(index + 1) % 8],
                    central_vertices[index],
                ],
                terrazzo,
            )
        central_floor = polygon_floor(
            "/World/Architecture/RestrictedAreas/CentralAtriumDrop/LowerFloor",
            central_vertices,
            terrazzo,
            z=-central_step_down + 0.002,
        )
        central_floor.SetCustomDataByKey("aisha:robotAccess", "prohibited")
        central_floor.SetCustomDataByKey("aisha:stepDownM", central_step_down)

        # The low riser is visible and dimensionally truthful. A separate
        # invisible collision/raycast proxy makes the mapped no-go edge visible
        # to the simulated crown LiDAR, which cannot sense a downward step. This
        # proxy is presentation/training infrastructure, not a physical-safety
        # claim about the real robot.
        for index in range(8):
            start = central_vertices[index]
            end = central_vertices[(index + 1) % 8]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            centre_xy = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            yaw_deg = math.degrees(math.atan2(dy, dx))
            box(
                f"/World/Architecture/RestrictedAreas/CentralAtriumDrop/Riser_{index:02d}",
                (length, 0.025, central_step_down),
                (*centre_xy, -central_step_down / 2.0),
                dark_grey,
                collision=False,
                rotate_z_deg=yaw_deg,
            )
            barrier = box(
                f"/World/Architecture/RestrictedAreas/CentralAtriumDrop/NavigationBarrier_{index:02d}",
                (length, 0.04, 1.25),
                (*centre_xy, 0.625),
                dark_grey,
                rotate_z_deg=yaw_deg,
            )
            barrier.SetCustomDataByKey("aisha:proxyPurpose", "central_drop_no_go_lidar_and_collision")
            UsdGeom.Imageable(barrier).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        hallway = config["plan_geometry"]["east_hallway"]
        hall_x_min, hall_x_max = (float(value) for value in hallway["x_range_m"])
        hall_y_min, hall_y_max = (float(value) for value in hallway["y_range_m"])
        hall_size = (hall_x_max - hall_x_min, hall_y_max - hall_y_min)
        hall_centre = (
            (hall_x_min + hall_x_max) / 2.0,
            (hall_y_min + hall_y_max) / 2.0,
        )
        vp_values = config["doors"]["vice_principal"]
        vp_door_x, vp_door_y = (float(value) for value in vp_values["centre_xy_m"])
        vice_access_size = (2.40, abs(vp_door_y - hall_y_min))
        vice_access_centre = (vp_door_x, (vp_door_y + hall_y_min) / 2.0)
        box("/World/Architecture/Floors/EastHallway", (*hall_size, 0.055), (*hall_centre, -0.027), terrazzo, physics_binding=tile_physics)
        box("/World/Architecture/Floors/ViceAccess", (*vice_access_size, 0.055), (*vice_access_centre, -0.027), terrazzo, physics_binding=tile_physics)
        cluster = config["plan_geometry"]["south_east_cluster"]
        vice_room = cluster["vice_principal"]
        principal_room = cluster["principal"]
        vice_size = tuple(float(value) for value in vice_room["size_xy_m"])
        vice_centre = tuple(float(value) for value in vice_room["centre_xy_m"])
        vice_rotation = float(vice_room["rotation_deg"])
        principal_size = tuple(float(value) for value in principal_room["size_xy_m"])
        principal_centre = tuple(float(value) for value in principal_room["centre_xy_m"])
        principal_rotation = float(principal_room["rotation_deg"])
        principal_floor_polygon = [
            tuple(float(value) for value in point)
            for point in principal_twin.get("floor_polygon_xy_m", [])
        ]
        box("/World/Architecture/Floors/VicePrincipal", (*vice_size, 0.055), (*vice_centre, -0.027), oak, physics_binding=tile_physics, rotate_z_deg=vice_rotation)
        principal_passage_width = float(
            config["known_dimensions"].get(
                "principal_passage_clear_width_m", {"value": 2.60}
            )["value"]
        )
        box("/World/Architecture/Floors/PrincipalAccess", (5.80, principal_passage_width, 0.055), (5.45, -5.45, -0.027), terrazzo, physics_binding=tile_physics, rotate_z_deg=-45.0)
        if visual_twin_enabled and principal_floor_polygon:
            polygon_floor(
                "/World/Architecture/Floors/PrincipalMeasured",
                principal_floor_polygon,
                grey_oak,
                z=-0.027,
            )
        else:
            box("/World/Architecture/Floors/Principal", (*principal_size, 0.055), (*principal_centre, -0.027), oak, physics_binding=tile_physics, rotate_z_deg=principal_rotation)

        # Render-only PBR finish meshes sit just above the already validated
        # collision floors. They add the dense terrazzo and light timber visible
        # in the walkthrough without changing wheel contact or route clearance.
        for index in range(8):
            textured_polygon_surface(
                f"AtriumTerrazzoRing_{index:02d}",
                [
                    vertices[index],
                    vertices[(index + 1) % 8],
                    central_vertices[(index + 1) % 8],
                    central_vertices[index],
                ],
                terrazzo_finish,
                metres_per_tile=2.2,
            )
        textured_polygon_surface(
            "CentralAtriumLowerTerrazzo",
            central_vertices,
            terrazzo_finish,
            z=-central_step_down + 0.006,
            metres_per_tile=2.2,
        )
        textured_rect_surface("EastHallTerrazzo", hall_size, hall_centre, terrazzo_finish, metres_per_tile=2.2)
        textured_rect_surface("ViceAccessTerrazzo", vice_access_size, vice_access_centre, terrazzo_finish, metres_per_tile=2.2)
        textured_rect_surface("PrincipalAccessTerrazzo", (5.80, principal_passage_width), (5.45, -5.45), terrazzo_finish, rotate_z_deg=-45.0, metres_per_tile=2.2)
        textured_rect_surface("ViceOfficeOak", vice_size, vice_centre, oak_finish, rotate_z_deg=vice_rotation, metres_per_tile=2.6)
        if visual_twin_enabled and principal_floor_polygon:
            textured_polygon_surface(
                "PrincipalOfficeGreyOakMeasured",
                principal_floor_polygon,
                grey_oak_finish,
                metres_per_tile=2.25,
            )
        else:
            textured_rect_surface("PrincipalOfficeOak", principal_size, principal_centre, oak_finish, rotate_z_deg=principal_rotation, metres_per_tile=2.6)

        for room_name in ("office_manager", "meeting_room_1", "meeting_room_2"):
            room = cluster[room_name]
            box(
                f"/World/Architecture/Floors/Dressing/{room_name}",
                (float(room["size_xy_m"][0]), float(room["size_xy_m"][1]), 0.045),
                (float(room["centre_xy_m"][0]), float(room["centre_xy_m"][1]), -0.022),
                light_grey,
                physics_binding=tile_physics,
            )

        # Atrium and east-hallway walls. The east face and south-east diagonal
        # retain the plan's circulation openings.
        for index in range(8):
            start = vertices[index]
            end = vertices[(index + 1) % 8]
            if index == 6:  # south-east diagonal opening toward Principal cluster
                continue
            if index == 7:  # east face split around the 2.80 m hallway
                wall_segment("Atrium_East_South", start, (start[0], hall_y_min), warm_white)
                wall_segment("Atrium_East_North", (start[0], hall_y_max), end, warm_white)
                continue
            material = timber if index in (0, 1) else warm_white
            wall_segment(f"Atrium_{index:02d}", start, end, material)

        access_half_width = vice_access_size[0] / 2.0
        wall_segment("EastHall_North", (vertices[0][0], hall_y_max), (hall_x_max, hall_y_max), warm_white)
        slatted_wall("EastHall_South_West", (vertices[7][0], hall_y_min), (vp_door_x - access_half_width, hall_y_min))
        slatted_wall("EastHall_South_East", (vp_door_x + access_half_width, hall_y_min), (hall_x_max, hall_y_min))
        split_wall("EastHall_End", (hall_x_max, hall_y_min), (hall_x_max, hall_y_max), (hall_x_max, hall_centre[1]), 1.80, light_grey)

        # Frosted-glass double doors at the east side entrance.
        for y in (-0.48, 0.48):
            box("/World/Architecture/Glass/EastDoor_" + ("S" if y < 0 else "N"), (0.055, 0.88, 2.30), (hall_x_max, y, 1.15), glass, rotate_z_deg=90.0)
            box("/World/Architecture/Glass/EastDoorBand_" + ("S" if y < 0 else "N"), (0.060, 0.88, 0.12), (hall_x_max - 0.03, y, 1.20), warm_white, collision=False, rotate_z_deg=90.0)

        # Vice-Principal access and room, east of the angled Principal office as
        # shown on page 2.
        wall_segment("ViceAccess_West", (vp_door_x - access_half_width, vp_door_y), (vp_door_x - access_half_width, hall_y_min), timber)
        wall_segment("ViceAccess_East", (vp_door_x + access_half_width, vp_door_y), (vp_door_x + access_half_width, hall_y_min), warm_white)
        vp_half = float(vp_values["clear_width_m"]) / 2.0
        if vice_room.get("status") == "site_scan_plus_manual_measurement_candidate":
            vice_corners = [
                local_to_world(vice_centre, (-vice_size[0] / 2.0, -vice_size[1] / 2.0), vice_rotation),
                local_to_world(vice_centre, (vice_size[0] / 2.0, -vice_size[1] / 2.0), vice_rotation),
                local_to_world(vice_centre, (vice_size[0] / 2.0, vice_size[1] / 2.0), vice_rotation),
                local_to_world(vice_centre, (-vice_size[0] / 2.0, vice_size[1] / 2.0), vice_rotation),
            ]
            split_wall(
                "Vice_North",
                vice_corners[3],
                vice_corners[2],
                (vp_door_x, vp_door_y),
                2.0 * vp_half,
                warm_white,
            )
            slatted_wall("Vice_South", vice_corners[0], vice_corners[1])
            wall_segment("Vice_West", vice_corners[0], vice_corners[3], warm_white)
            wall_segment("Vice_East", vice_corners[1], vice_corners[2], warm_white)
        else:
            wall_segment("Vice_North_West", (13.95, vp_door_y), (vp_door_x - vp_half, vp_door_y), warm_white)
            wall_segment("Vice_North_East", (vp_door_x + vp_half, vp_door_y), (20.25, vp_door_y), warm_white)
            slatted_wall("Vice_South_West", (13.95, -8.05), (16.10, -8.05))
            box("/World/Architecture/Walls/ViceWindowSill", (4.15, wall_thickness, 0.66), (18.175, -8.05, 0.33), warm_white)
            box("/World/Architecture/Walls/ViceWindowHead", (4.15, wall_thickness, 0.64), (18.175, -8.05, 2.68), warm_white)
            wall_segment("Vice_West", (13.95, -8.05), (13.95, -5.05), warm_white)
            wall_segment("Vice_East_Lower", (20.25, -8.05), (20.25, -6.30), warm_white)
            wall_segment("Vice_East_Upper", (20.25, -5.80), (20.25, -5.05), warm_white)
            box("/World/Architecture/Glass/ViceWindow", (0.055, 0.50, 1.55), (20.25, -6.05, 1.48), glass, rotate_z_deg=90.0)

        # Diagonal reception/secretary passage leading to the Principal office.
        corridor_start = (3.45, -3.45)
        corridor_end = (7.00, -7.00)
        tangent = (math.sqrt(0.5), -math.sqrt(0.5))
        normal = (math.sqrt(0.5), math.sqrt(0.5))
        for side_name, side_sign, material in (("North", 1.0, warm_white), ("South", -1.0, timber)):
            offset = (
                normal[0] * principal_passage_width / 2.0 * side_sign,
                normal[1] * principal_passage_width / 2.0 * side_sign,
            )
            wall_segment(
                f"PrincipalAccess_{side_name}",
                (corridor_start[0] + offset[0], corridor_start[1] + offset[1]),
                (corridor_end[0] + offset[0], corridor_end[1] + offset[1]),
                material,
            )

        principal_values = config["doors"]["principal"]
        if visual_twin_enabled:
            principal_door_centre = tuple(
                float(value) for value in principal_values["centre_xy_m"]
            )
        else:
            principal_corners = [
                local_to_world(principal_centre, (-principal_size[0] / 2.0, -principal_size[1] / 2.0), principal_rotation),
                local_to_world(principal_centre, (principal_size[0] / 2.0, -principal_size[1] / 2.0), principal_rotation),
                local_to_world(principal_centre, (principal_size[0] / 2.0, principal_size[1] / 2.0), principal_rotation),
                local_to_world(principal_centre, (-principal_size[0] / 2.0, principal_size[1] / 2.0), principal_rotation),
            ]
            derived_principal_door_centre = local_to_world(
                principal_centre, (-principal_size[0] / 2.0, 0.0), principal_rotation
            )
            if principal_values.get("width_status") == "manual_site_measurement":
                principal_door_centre = tuple(
                    float(value) for value in principal_values["centre_xy_m"]
                )
            else:
                principal_door_centre = derived_principal_door_centre
                principal_values["centre_xy_m"] = [
                    round(principal_door_centre[0], 3),
                    round(principal_door_centre[1], 3),
                ]
                principal_values["wall_rotation_deg"] = principal_rotation + 90.0

        if visual_twin_enabled:
            for values in principal_twin.get("walls", []):
                registered_wall(f"PrincipalMeasured_{values['id']}", values)
            for values in principal_twin.get("approach_partitions", []):
                registered_wall(f"PrincipalApproachMeasured_{values['id']}", values)
        else:
            split_wall(
                "Principal_West",
                principal_corners[0],
                principal_corners[3],
                principal_door_centre,
                float(config["doors"]["principal"]["clear_width_m"]),
                warm_white,
            )
            slatted_wall("Principal_South", principal_corners[0], principal_corners[1])
            wall_segment("Principal_East", principal_corners[1], principal_corners[2], warm_white)
            wall_segment("Principal_North", principal_corners[2], principal_corners[3], warm_white)

        doors = {
            "vice_principal": doorway(
                "VicePrincipal",
                vp_values,
                hinge_left=vp_values.get("hinge_side_from_hallway", "left") == "left",
                opens_outward=vp_values.get("swing_from_hallway", "outward") == "outward",
            ),
            "principal": doorway(
                "Principal",
                principal_values,
                hinge_left=principal_values.get("hinge_side_from_hallway", "left") == "left",
                opens_outward=principal_values.get("swing_from_hallway", "outward") == "outward",
            ),
        }

        # Walkthrough-derived furniture and finishes.
        box("/World/Furniture/Reception/Base", (4.20, 0.78, 1.08), (-1.10, 3.45, 0.54), timber)
        box("/World/Furniture/Reception/Counter", (4.35, 0.92, 0.09), (-1.10, 3.45, 1.10), timber_light)
        textured_rect_surface("ReceptionCounterFinish", (4.26, 0.84), (-1.10, 3.45), walnut_finish, z=1.147, metres_per_tile=2.1)
        for index in range(40):
            x = -3.05 + index * 0.10
            box(f"/World/Furniture/Reception/Slat_{index:02d}", (0.035, 0.055, 0.86), (x, 3.01, 0.50), timber_light, collision=False)
        for index, x in enumerate((-2.35, -1.10, 0.15)):
            box(f"/World/Furniture/Reception/Glass_{index:02d}", (1.08, 0.025, 0.74), (x, 3.43, 1.52), glass, collision=False)
            for side, offset in (("Left", -0.56), ("Right", 0.56)):
                box(f"/World/Furniture/Reception/GlassFrame_{index:02d}_{side}", (0.025, 0.040, 0.80), (x + offset, 3.43, 1.52), metal, collision=False)
        box("/World/Furniture/Reception/Monitor", (0.055, 0.52, 0.34), (-0.35, 3.18, 1.37), black, collision=False)
        box("/World/Furniture/Reception/MonitorStand", (0.08, 0.08, 0.22), (-0.35, 3.18, 1.18), metal, collision=False)
        if visual_twin_enabled:
            # The capture shows an angled, wraparound reception rather than a
            # single loose desk. This west return and upper fascia reproduce the
            # first 45 seconds of the walkthrough without copying visible signs.
            box("/World/Furniture/Reception/WestReturnBase", (3.35, 0.78, 1.08), (-4.15, 1.55, 0.54), timber, rotate_z_deg=90.0)
            box("/World/Furniture/Reception/WestReturnCounter", (3.48, 0.92, 0.09), (-4.15, 1.55, 1.10), timber_light, rotate_z_deg=90.0)
            textured_rect_surface("ReceptionWestCounterFinish", (3.40, 0.84), (-4.15, 1.55), walnut_finish, z=1.147, rotate_z_deg=90.0, metres_per_tile=2.1)
            for index in range(31):
                y = 0.05 + index * 0.10
                box(f"/World/Furniture/Reception/WestSlat_{index:02d}", (0.055, 0.035, 0.86), (-4.59, y, 0.50), timber_light, collision=False)
            for index in range(42):
                x = -3.20 + index * 0.10
                box(f"/World/Appearance/ReceptionFascia/North_{index:02d}", (0.035, 0.10, 0.78), (x, 3.02, 2.47), timber_light, collision=False)
            for index in range(30):
                y = 0.15 + index * 0.10
                box(f"/World/Appearance/ReceptionFascia/West_{index:02d}", (0.10, 0.035, 0.78), (-4.58, y, 2.47), timber_light, collision=False)
            for index, y in enumerate((0.45, 1.55, 2.65)):
                box(f"/World/Furniture/Reception/WestGlass_{index:02d}", (0.025, 0.96, 0.72), (-4.16, y, 1.51), glass, collision=False)
                box(f"/World/Furniture/Reception/WestGlassFrame_{index:02d}", (0.04, 0.035, 0.78), (-4.16, y - 0.50, 1.51), metal, collision=False)
        box("/World/Furniture/AtriumBench/Seat", (2.60, 0.70, 0.16), (-1.00, -3.35, 0.46), black)
        box("/World/Furniture/AtriumBench/Back", (2.60, 0.12, 0.70), (-1.00, -3.68, 0.78), black)
        for index, x in enumerate((-2.05, 0.05)):
            box(f"/World/Furniture/AtriumBench/Leg_{index}", (0.10, 0.58, 0.38), (x, -3.35, 0.20), metal, collision=False)
        if visual_twin_enabled:
            sofa("CentralAtriumSofaNorth", (-0.62, 0.68), 180.0, floor_z=-central_step_down)
            sofa("CentralAtriumSofaSouth", (0.62, -0.68), 0.0, floor_z=-central_step_down)
            box("/World/Furniture/CentralAtriumCoffeeTable", (0.95, 0.52, 0.08), (0.0, 0.0, -central_step_down + 0.39), timber_light)
            for index, (x, y, yaw) in enumerate(((-1.48, 0.05, -18.0), (1.42, 0.12, 16.0))):
                box(f"/World/Furniture/CentralAtriumDisplay/Easel_{index:02d}/Board", (0.64, 0.045, 0.88), (x, y, -central_step_down + 1.12), paper, collision=False, rotate_z_deg=yaw)
                for leg_index, local_x in enumerate((-0.22, 0.22)):
                    leg_xy = local_to_world((x, y), (local_x, 0.10), yaw)
                    box(f"/World/Furniture/CentralAtriumDisplay/Easel_{index:02d}/Leg_{leg_index}", (0.035, 0.035, 1.18), (*leg_xy, -central_step_down + 0.59), timber_light, collision=False, rotate_z_deg=yaw)
        plant("AtriumPlant", (2.20, 3.45))
        plant("EastHallPlant", (20.70, 0.70))

        # Abstract displays establish the rhythm seen along the real timber
        # corridor while avoiding reproduction of the walkthrough's posters.
        for index, (x, inset) in enumerate(((8.0, green), (10.6, paper), (13.2, green_accent))):
            framed_panel(f"EastHall_{index:02d}", (x, -1.255, 1.58), inset_material=inset)
        framed_panel("AtriumNorth", (1.65, 4.835, 1.55), size_xz=(1.20, 1.45), inset_material=paper)

        # The VP interior was locked during capture. This room is therefore a
        # disclosed plan-envelope and adjacent-material presentation assumption,
        # not a claim that its furniture was observed in the walkthrough.
        vice_table_centre = (15.45, -6.62)
        round_meeting_table("ViceMeetingTable", vice_table_centre, 0.76)
        for index, angle_deg in enumerate((70.0, 135.0, 180.0, 225.0, 290.0)):
            angle = math.radians(angle_deg)
            seat_xy = (
                vice_table_centre[0] + 1.12 * math.cos(angle),
                vice_table_centre[1] + 1.12 * math.sin(angle),
            )
            cantilever_chair(f"ViceMeetingChair_{index:02d}", seat_xy, angle_deg + 180.0)
        box("/World/Furniture/ViceCabinet", (1.70, 0.38, 0.82), (19.00, -7.72, 0.41), timber)
        for index, x in enumerate((18.34, 18.78, 19.22, 19.66)):
            box(f"/World/Furniture/ViceCabinetDoor_{index:02d}", (0.38, 0.025, 0.68), (x, -7.505, 0.43), timber_light, collision=False)
            box(f"/World/Furniture/ViceCabinetHandle_{index:02d}", (0.08, 0.020, 0.018), (x, -7.486, 0.49), metal, collision=False)
        plant("VicePlant", (19.70, -7.35))

        if visual_twin_enabled:
            # Reconstruct the captured Principal suite from the registered
            # RoomPlan semantic footprints. The older invented rectangular
            # furniture layout and emblem are intentionally not authored.
            meeting_values = None
            for values in principal_twin.get("tables", []):
                if values.get("role") == "meeting_table":
                    meeting_values = values
                    centre = tuple(float(value) for value in values["centre_xy_m"])
                    sx, sy, _ = (float(value) for value in values["size_xyz_m"])
                    round_meeting_table("PrincipalMeasuredMeetingTable", centre, min(0.76, max(sx, sy) * 0.43))
                else:
                    registered_table(str(values["id"]), values)
            for values in principal_twin.get("chairs", []):
                centre = tuple(float(value) for value in values["centre_xy_m"])
                yaw = float(values["yaw_deg"])
                if values.get("role") == "executive":
                    chair("PrincipalMeasuredExecutiveChair", centre, yaw)
                else:
                    cantilever_chair(f"PrincipalMeasured{values['id']}", centre, yaw)
            for values in principal_twin.get("storage", []):
                registered_storage(str(values["id"]), values)

            # The round-table chairs were occluded in the RoomPlan mesh but are
            # plainly visible in the walkthrough and still images.
            if meeting_values is not None:
                meeting_centre = tuple(float(value) for value in meeting_values["centre_xy_m"])
                for index, angle_deg in enumerate((150.0, 225.0, 315.0)):
                    angle = math.radians(angle_deg)
                    seat_xy = (
                        meeting_centre[0] + 1.02 * math.cos(angle),
                        meeting_centre[1] + 1.02 * math.sin(angle),
                    )
                    cantilever_chair(f"PrincipalMeasuredMeetingChair_{index:02d}", seat_xy, angle_deg + 180.0)

            main_desk = next(
                values for values in principal_twin["tables"] if values.get("role") == "executive_desk_main"
            )
            desk_centre = tuple(float(value) for value in main_desk["centre_xy_m"])
            desk_yaw = float(main_desk["yaw_deg"])
            monitor_xy = local_to_world(desk_centre, (0.28, 0.03), desk_yaw)
            box("/World/Furniture/PrincipalMeasured/ExecutiveMonitor", (0.055, 0.62, 0.37), (*monitor_xy, 1.10), black, collision=False, rotate_z_deg=desk_yaw)
            box("/World/Furniture/PrincipalMeasured/ExecutiveMonitorStand", (0.06, 0.06, 0.24), (*monitor_xy, 0.92), metal, collision=False, rotate_z_deg=desk_yaw)

            tv_storage = next(
                values for values in principal_twin["storage"] if values.get("role") == "tv_award_cabinet"
            )
            tv_centre = tuple(float(value) for value in tv_storage["centre_xy_m"])
            tv_yaw = float(tv_storage["yaw_deg"])
            tv_face = local_to_world(tv_centre, (0.0, -0.31), tv_yaw)
            box("/World/Furniture/PrincipalMeasured/Television", (1.48, 0.055, 0.82), (*tv_face, 1.72), black, collision=False, rotate_z_deg=tv_yaw)
            box("/World/Furniture/PrincipalMeasured/TelevisionInset", (1.38, 0.018, 0.72), (*local_to_world(tv_face, (0.0, -0.038), tv_yaw), 1.72), dark_grey, collision=False, rotate_z_deg=tv_yaw)
            for index in range(4):
                local_x = -0.68 + index * 0.45
                award_xy = local_to_world(tv_centre, (local_x, 0.0), tv_yaw)
                cylinder(f"/World/Furniture/PrincipalMeasured/Award_{index:02d}/Base", 0.07, 0.07, (*award_xy, 1.06), bronze, collision=False)
                ellipsoid(f"/World/Furniture/PrincipalMeasured/Award_{index:02d}/Top", (0.13, 0.07, 0.20), (*award_xy, 1.22), bronze, collision=False, rotate_xyz_deg=(0.0, 0.0, index * 21.0))
            plant("PrincipalMeasuredPlant", (10.35, -7.15))
            framed_panel("PrincipalMeasuredAbstractPortrait", (10.86, -7.25, 1.72), size_xz=(0.62, 0.82), rotate_z_deg=90.0, inset_material=paper)
        else:
            desk_local = local_to_world(principal_centre, (0.95, -0.95), principal_rotation)
            desk("PrincipalDesk", desk_local, principal_rotation)
            principal_chair = local_to_world(principal_centre, (1.20, -1.15), principal_rotation)
            chair("PrincipalDeskChair", principal_chair, principal_rotation + 180.0)
            for name, local in (("PrincipalVisitorLeft", (0.40, 1.20)), ("PrincipalVisitorRight", (0.40, -1.20))):
                chair(name, local_to_world(principal_centre, local, principal_rotation), principal_rotation)
            principal_cabinet = local_to_world(principal_centre, (1.90, 0.80), principal_rotation)
            box("/World/Furniture/PrincipalCabinet", (0.40, 1.60, 0.86), (*principal_cabinet, 0.43), timber, rotate_z_deg=principal_rotation)
            plant("PrincipalPlant", local_to_world(principal_centre, (1.65, 1.30), principal_rotation))

        # White columns are walkthrough-derived visual anchors rather than
        # surveyed plan geometry. Their disclosed positions are kept in config
        # so the learned-trace swept-clearance validator can gate the render.
        column_config = config["appearance"]["atrium_columns"]
        column_radius = float(column_config["radius_m"])
        column_height = float(column_config["height_m"])
        for index, position in enumerate(column_config["positions_xy_m"]):
            x, y = (float(value) for value in position)
            cylinder(
                f"/World/Architecture/Columns/Atrium_{index:02d}",
                column_radius,
                column_height,
                (x, y, column_height / 2.0),
                warm_white,
            )
        for index, x in enumerate((16.55, 17.35, 18.15, 18.95, 19.75)):
            box(f"/World/Architecture/Glass/ViceExterior_{index:02d}", (0.72, 0.045, 1.62), (x, -8.02, 1.48), glass, collision=False)
            box(f"/World/Architecture/Glass/ViceMullion_{index:02d}", (0.035, 0.07, 1.78), (x - 0.38, -7.99, 1.48), metal, collision=False)

        # Dark geometric inlays are physical-space cues; aggregate itself is now
        # represented by the high-density procedural texture above.
        for index, (start, end) in enumerate((((7.0, 0.45), (10.5, 0.45)), ((10.5, 0.45), (11.6, -0.35)), ((11.6, -0.35), (14.2, -0.35)))):
            dx, dy = end[0] - start[0], end[1] - start[1]
            box(
                f"/World/Appearance/FloorInlay_{index}",
                (math.hypot(dx, dy), 0.035, 0.008),
                ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, 0.011),
                black,
                collision=False,
                rotate_z_deg=math.degrees(math.atan2(dy, dx)),
            )

        # Suspended ceilings and LED panels. The renderers hide ceilings only for
        # the architectural overview; all cinematic shots retain them.
        ceiling_z = wall_height + 0.04
        box("/World/Architecture/Ceilings/Atrium", (11.20, 11.20, 0.08), (0.0, 0.0, ceiling_z), warm_white, collision=False)
        box("/World/Architecture/Ceilings/EastHall", (*hall_size, 0.08), (*hall_centre, ceiling_z), warm_white, collision=False)
        box("/World/Architecture/Ceilings/ViceAccess", (*vice_access_size, 0.08), (*vice_access_centre, ceiling_z), warm_white, collision=False)
        box("/World/Architecture/Ceilings/Vice", (*vice_size, 0.08), (*vice_centre, ceiling_z), warm_white, collision=False, rotate_z_deg=vice_rotation)
        box("/World/Architecture/Ceilings/PrincipalAccess", (5.80, principal_passage_width, 0.08), (5.45, -5.45, ceiling_z), warm_white, collision=False, rotate_z_deg=-45.0)
        if visual_twin_enabled:
            ceiling_bounds = principal_twin["ceiling_bounds"]
            measured_ceiling_centre = tuple(float(value) for value in ceiling_bounds["centre_xy_m"])
            measured_ceiling_size = tuple(float(value) for value in ceiling_bounds["size_xy_m"])
            box("/World/Architecture/Ceilings/PrincipalMeasured", (*measured_ceiling_size, 0.08), (*measured_ceiling_centre, ceiling_z), warm_white, collision=False)
        else:
            measured_ceiling_centre = principal_centre
            measured_ceiling_size = principal_size
            box("/World/Architecture/Ceilings/Principal", (*principal_size, 0.08), (*principal_centre, ceiling_z), warm_white, collision=False, rotate_z_deg=principal_rotation)

        light_positions = [
            (-3.0, -2.5), (-3.0, 0.0), (-3.0, 2.5),
            (0.0, -2.5), (0.0, 0.0), (0.0, 2.5),
            (3.0, -2.5), (3.0, 0.0), (3.0, 2.5),
            (7.0, 0.0), (10.0, 0.0), (13.0, 0.0), (16.0, 0.0), (19.0, 0.0),
            (17.1, -3.1), (15.4, -6.55), (18.7, -6.55),
            (5.4, -5.4), (8.0, -8.6), (9.7, -9.9),
        ]
        for index, (x, y) in enumerate(light_positions):
            box(f"/World/Lighting/Panels/Frame_{index:02d}", (0.92, 0.62, 0.018), (x, y, wall_height - 0.013), metal, collision=False)
            box(f"/World/Lighting/Panels/Panel_{index:02d}", (0.84, 0.54, 0.018), (x, y, wall_height - 0.026), light_panel, collision=False)
            light = UsdLux.RectLight.Define(stage, f"/World/Lighting/Fixtures/Light_{index:02d}")
            light.CreateIntensityAttr(28000.0 if visual_twin_enabled else 11000.0)
            light.CreateColorAttr(Gf.Vec3f(0.96, 0.98, 1.0))
            light.CreateWidthAttr(0.84)
            light.CreateHeightAttr(0.54)
            light_xform = UsdGeom.Xformable(light.GetPrim())
            light_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, wall_height - 0.045))

        # Dropped-ceiling grid and vents reproduce the office scale cues visible
        # in the walkthrough without claiming surveyed dimensions.
        ceiling_grid("Atrium", (0.0, 0.0), (10.80, 10.80))
        ceiling_grid("EastHall", hall_centre, (hall_size[0] - 0.11, hall_size[1] - 0.08))
        ceiling_grid("ViceAccess", vice_access_centre, (vice_access_size[0] - 0.08, vice_access_size[1] - 0.08))
        ceiling_grid("Vice", vice_centre, (vice_size[0] - 0.08, vice_size[1] - 0.08), rotate_z_deg=vice_rotation)
        ceiling_grid("PrincipalAccess", (5.45, -5.45), (5.72, principal_passage_width - 0.08), rotate_z_deg=-45.0)
        if visual_twin_enabled:
            ceiling_grid("PrincipalMeasured", measured_ceiling_centre, (measured_ceiling_size[0] - 0.08, measured_ceiling_size[1] - 0.08))
        else:
            ceiling_grid("Principal", principal_centre, (principal_size[0] - 0.08, principal_size[1] - 0.08), rotate_z_deg=principal_rotation)
        ceiling_vent("Vice", (18.05, -6.02))
        if visual_twin_enabled:
            principal_vent = (8.75, -9.15)
            ceiling_vent("PrincipalMeasured", principal_vent)
        else:
            principal_vent = local_to_world(principal_centre, (-0.65, 0.70), principal_rotation)
            ceiling_vent("Principal", principal_vent, rotate_z_deg=principal_rotation)
        ceiling_vent("EastHall", (12.25, 0.55))
        for index, location in enumerate(((3.7, 1.2), (10.1, -0.6), (17.1, -3.7), (18.6, -6.5), (8.55, -9.25))):
            ceiling_sensor(f"Camera_{index:02d}", location)
        for index, location in enumerate(((-1.8, 0.8), (7.9, 0.65), (16.2, -6.0), (7.6, -8.7))):
            cylinder(f"/World/Architecture/Ceilings/Devices/Smoke_{index:02d}", 0.095, 0.035, (*location, wall_height - 0.035), warm_white, collision=False)
            cylinder(f"/World/Architecture/Ceilings/Devices/SmokeRing_{index:02d}", 0.057, 0.010, (*location, wall_height - 0.058), dark_grey, collision=False)

        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Ambient")
        dome.CreateIntensityAttr(720.0 if visual_twin_enabled else 280.0)
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.87, 0.93))
        sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
        sun.CreateIntensityAttr(700.0)
        sun.CreateAngleAttr(1.2)
        sun_xform = UsdGeom.Xformable(sun.GetPrim())
        sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(48.0, -25.0, 30.0))

        # Hidden route authoring aid and plan-aligned goal metadata.
        route = config["route"]["waypoints"]
        points = [Gf.Vec3f(float(item["x_m"]), float(item["y_m"]), 0.025) for item in route]
        curve = UsdGeom.BasisCurves.Define(stage, "/World/Presentation/Route")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr(points)
        curve.CreateWidthsAttr([0.075])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.55, 0.95)])
        UsdGeom.Imageable(curve.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        for item in route:
            if item["action"] != "start_and_end" and not str(item["action"]).startswith("presentation_stop"):
                continue
            marker = UsdGeom.Cylinder.Define(stage, f"/World/Presentation/Goals/{item['id']}")
            marker.CreateAxisAttr(UsdGeom.Tokens.z)
            marker.CreateRadiusAttr(0.25)
            marker.CreateHeightAttr(0.015)
            marker.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.85, 0.35)])
            marker_xform = UsdGeom.Xformable(marker.GetPrim())
            marker_xform.AddTranslateOp().Set(Gf.Vec3d(float(item["x_m"]), float(item["y_m"]), 0.0075))
            marker.GetPrim().SetCustomDataByKey("aisha:action", str(item["action"]))
            marker.GetPrim().SetCustomDataByKey("aisha:poseStatus", "plan_derived_topology_presentation_offset")
            UsdGeom.Imageable(marker.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        robot = UsdGeom.Xform.Define(stage, "/World/AISHA")
        robot.GetPrim().GetReferences().AddReference(f"../usd/{asset.name}")
        robot_xform = UsdGeom.Xformable(robot.GetPrim())
        robot_xform.AddTranslateOp(opSuffix="route").Set(Gf.Vec3d(0.0, 0.0, 0.0))
        robot_xform.AddRotateZOp(opSuffix="route").Set(0.0)
        robot.GetPrim().SetCustomDataByKey("aisha:payloadVariant", args.payload)
        geometry_status = (
            str(measured_overlay["status"])
            if measured_overlay is not None
            else "plan_derived_route_scoped"
        )
        robot.GetPrim().SetCustomDataByKey("aisha:routeGeometryStatus", geometry_status)

        # A presentation shell gives the imported engineering URDF the intended
        # finished-product read while keeping the Rev D collision and sensor
        # frames unchanged underneath it.
        ellipsoid("/World/AISHA/PresentationShell/Body", (0.82, 0.59, 0.34), (0.02, 0.0, 0.40), aisha_white)
        ellipsoid("/World/AISHA/PresentationShell/LowerBumper", (0.88, 0.64, 0.16), (0.03, 0.0, 0.27), aisha_black)
        box("/World/AISHA/PresentationShell/Tray", (0.78, 0.57, 0.045), (-0.03, 0.0, 0.565), aisha_green, collision=False)
        box("/World/AISHA/PresentationShell/TrayPad", (0.68, 0.47, 0.012), (-0.05, 0.0, 0.594), aisha_black, collision=False)
        box("/World/AISHA/PresentationShell/Mast", (0.09, 0.11, 0.43), (0.40, 0.0, 0.76), aisha_white, collision=False)
        ellipsoid("/World/AISHA/PresentationShell/Head", (0.33, 0.40, 0.27), (0.50, 0.0, 0.96), aisha_white)
        box("/World/AISHA/PresentationShell/Face", (0.022, 0.275, 0.145), (0.691, 0.0, 0.97), aisha_black, collision=False)
        box("/World/AISHA/PresentationShell/FaceStatus", (0.026, 0.060, 0.012), (0.705, -0.080, 0.920), aisha_led, collision=False)
        ellipsoid("/World/AISHA/PresentationShell/CameraAperture", (0.018, 0.040, 0.040), (0.706, 0.068, 0.985), metal)
        cylinder("/World/AISHA/PresentationShell/LidarCollar", 0.075, 0.025, (0.50, 0.0, 1.115), metal, collision=False)
        cylinder("/World/AISHA/PresentationShell/Lidar", 0.055, 0.070, (0.50, 0.0, 1.160), aisha_black, collision=False)
        box("/World/AISHA/PresentationShell/LidarWindow", (0.038, 0.070, 0.018), (0.552, 0.0, 1.165), aisha_led, collision=False)
        for side, y in (("Left", 0.301), ("Right", -0.301)):
            ellipsoid(f"/World/AISHA/PresentationShell/{side}WheelCover", (0.38, 0.055, 0.23), (0.01, y, 0.31), aisha_black)
            box(f"/World/AISHA/PresentationShell/{side}Accent", (0.42, 0.018, 0.055), (0.03, y * 1.018, 0.43), aisha_led, collision=False)

        sensor_bindings = []
        contact_bindings = {"drive_wheel": [], "castor_low_friction": []}
        for prim in stage.TraverseAll():
            name = prim.GetName()
            if name in ("left_wheel_link", "right_wheel_link"):
                bind_physics(prim, drive_material)
                contact_bindings["drive_wheel"].append(str(prim.GetPath()))
            elif name in ("castor_fl_link", "castor_fr_link", "castor_rl_link", "castor_rr_link"):
                bind_physics(prim, castor_material)
                contact_bindings["castor_low_friction"].append(str(prim.GetPath()))
            if name == sensors["frames"]["crown_lidar"]["prim_name"]:
                prim.SetCustomDataByKey("aisha:sensorModel", sensors["frames"]["crown_lidar"]["model"])
                prim.SetCustomDataByKey("aisha:rosTopic", sensors["frames"]["crown_lidar"]["ros_topic"])
                sensor_bindings.append({"prim": str(prim.GetPath()), "model": sensors["frames"]["crown_lidar"]["model"]})
            elif name == sensors["frames"]["front_camera"]["body_prim_name"]:
                prim.SetCustomDataByKey("aisha:sensorModel", sensors["frames"]["front_camera"]["model"])
                prim.SetCustomDataByKey("aisha:alignedDepth", True)
                sensor_bindings.append({"prim": str(prim.GetPath()), "model": sensors["frames"]["front_camera"]["model"]})
            elif name == sensors["frames"]["imu"]["prim_name"]:
                prim.SetCustomDataByKey("aisha:sensorModel", sensors["frames"]["imu"]["model"])
                prim.SetCustomDataByKey("aisha:publishRateHz", float(sensors["frames"]["imu"]["publish_rate_hz"]))
                sensor_bindings.append({"prim": str(prim.GetPath()), "model": sensors["frames"]["imu"]["model"]})

        minimum = float(config["presentation_release"]["minimum_demo_door_clear_width_m"])
        robot_width = float(config["presentation_release"]["robot_transit_width_m"])
        planning_padding = float(
            config.get("presentation_clearance_profile", {}).get(
                "footprint_padding_per_side_m", 0.0
            )
        )
        for values in doors.values():
            values["minimum_required_m"] = minimum
            values["nominal_physical_side_clearance_m"] = (
                float(values["clear_width_m"]) - robot_width
            ) / 2.0
            values["nominal_padded_side_clearance_m"] = (
                float(values["clear_width_m"]) - robot_width - 2.0 * planning_padding
            ) / 2.0
            values["presentation_width_gate_passed"] = float(values["clear_width_m"]) >= minimum
            values["high_fidelity_threshold_validation"] = "blocked"

        stage.GetRootLayer().customLayerData = {
            "aisha:scenePurpose": "walkthrough_matched_environment_for_verified_learned_trajectory_replay",
            "aisha:a1Page2Status": "approved_page_2_reviewed",
            "aisha:planSha256": expected_plan_hash,
            "aisha:geometryStatus": geometry_status,
            "aisha:appearanceStatus": "walkthrough_video_derived_procedural_pbr_not_dimensioned",
            "aisha:visualUpgrade": "administration_walkthrough_procedural_pbr_v1",
            "aisha:geometryRefinement": refinement["revision"],
            "aisha:rtxMaterialRefinement": "administration_rtx_pbr_v2",
            "aisha:visualUpgradeCollisionImpact": "none_visual_only",
            "aisha:physicalRelease": False,
            "aisha:productionRepositoryCommit": config["provenance"]["production_repository"]["commit"],
        }
        stage.GetRootLayer().Save()

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                f"{geometry_status}_scene_built"
                if measured_overlay is not None
                else "walkthrough_matched_plan_derived_presentation_scene_built"
            ),
            "scene": str(path),
            "payload": args.payload,
            "robot_asset": str(asset),
            "approved_plan_page_2": {
                "status": "reviewed_and_traced",
                "authoritative_geometry_used": True,
                "source_filename": config["provenance"]["plan_source"]["filename"],
                "page": 2,
                "expected_sha256": expected_plan_hash,
                "supplied_plan": str(args.plan.resolve()) if args.plan and args.plan.exists() else None,
                "supplied_plan_sha256": supplied_plan_hash,
            },
            "door_survey": {
                "status": (
                    str(
                        measured_overlay.get(
                            "measurement_scope",
                            "measured_overlay_manual_measurements_used",
                        )
                    )
                    if measured_overlay is not None
                    else "not_supplied_assumptions_used"
                ),
                "survey_argument": str(args.door_survey.resolve()) if args.door_survey and args.door_survey.exists() else None,
            },
            "measured_geometry": {
                "status": geometry_status,
                "overlay": str(args.measured_geometry.resolve()) if measured_overlay is not None else None,
                "overlay_sha256": measured_overlay_hash,
                "source_capture_hashes": (
                    measured_overlay.get("source_capture_hashes", [])
                    if measured_overlay is not None
                    else []
                ),
                "registration_status": (
                    measured_overlay.get("registration_status")
                    if measured_overlay is not None
                    else None
                ),
                "presentation_clearance_profile": (
                    measured_overlay.get("presentation_clearance_profile")
                    if measured_overlay is not None
                    else None
                ),
                "physical_release": False,
            },
            "config_file": str(config_path),
            "config_sha256": sha256_file(config_path),
            "known_dimensions": config["known_dimensions"],
            "geometry_rtx_refinement": refinement,
            "geometry_rtx_refinement_config": str(refinement_path),
            "geometry_rtx_refinement_config_sha256": sha256_file(refinement_path),
            "plan_geometry": config["plan_geometry"],
            "appearance": config["appearance"],
            "capture_limitations": (
                measured_overlay.get("capture_limitations", {})
                if measured_overlay is not None
                else {}
            ),
            "measured_visual_twin": visual_twin if visual_twin_enabled else None,
            "visual_upgrade": {
                "version": (
                    "administration_roomplan_registered_walkthrough_matched_v2"
                    if visual_twin_enabled
                    else "administration_walkthrough_procedural_pbr_v1"
                ),
                "rtx_material_version": "administration_rtx_pbr_v3" if visual_twin_enabled else "administration_rtx_pbr_v2",
                "texture_maps": ["albedo", "perceptual_roughness", "tangent_space_normal"],
                "reference_policy": (
                    "approved page-2 global topology; independently registered RoomPlan Principal geometry and atrium scale; walkthrough materials and occluded visual detail"
                    if visual_twin_enabled
                    else "walkthrough appearance only; approved plan remains the geometry authority"
                ),
                "collision_geometry_changed": visual_twin_enabled,
                "revalidation_required": visual_twin_enabled,
                "texture_assets": [
                    {
                        "path": str(texture_path.resolve()),
                        "sha256": sha256_file(texture_path),
                    }
                    for texture_path in texture_paths
                    if texture_path.is_file()
                ],
            },
            "doors": doors,
            "route": route,
            "central_atrium_drop": {
                "step_down_m": central_step_down,
                "robot_access": central_polygon["robot_access"],
                "radius_m": central_radius,
                "radius_status": central_polygon["radius_status"],
                "navigation_boundary": central_polygon["simulation_boundary"],
                "vice_principal_interior_capture": (
                    measured_overlay.get("capture_limitations", {})
                    .get("vice_principal_office_interior", {})
                    .get("status")
                    if measured_overlay is not None
                    else "not_recorded_in_base_assumptions"
                ),
            },
            "sensor_contracts": sensor_bindings,
            "contact_material_bindings": contact_bindings,
            "checks": {
                "plan_hash_matches_reviewed_source": supplied_plan_hash in (None, expected_plan_hash),
                "all_presentation_door_widths_pass": all(item["presentation_width_gate_passed"] for item in doors.values()),
                "hallway_supports_pivot_circle": float(config["known_dimensions"]["hallway_clear_width_m"]["value"])
                >= float(config["presentation_release"]["pivot_clear_circle_m"]),
                "vice_principal_is_east_of_principal": float(cluster["vice_principal"]["centre_xy_m"][0])
                > float(cluster["principal"]["centre_xy_m"][0]),
                "atrium_columns_declared_for_trace_clearance": len(column_config["positions_xy_m"]) == 4
                and float(column_config["minimum_trace_centre_clearance_m"])
                >= math.hypot(
                    float(config["presentation_release"]["robot_transit_width_m"]) / 2.0,
                    float(config["presentation_release"]["robot_transit_length_m"]) / 2.0,
                )
                + column_radius,
                "central_atrium_drop_is_mapped_no_go": central_step_down == 0.20
                and central_polygon["robot_access"] == "prohibited"
                and central_polygon["simulation_boundary"]
                == "mapped_no_go_with_invisible_lidar_collision_proxy",
                "all_visual_texture_assets_present": all(texture_path.is_file() for texture_path in texture_paths),
                "registered_geometry_has_metric_scale": (
                    not visual_twin_enabled
                    or float(visual_twin["registration"]["principal"]["metric_scale"]) == 1.0
                ),
                "registered_principal_entrance_matches_door": (
                    not visual_twin_enabled
                    or all(
                        abs(float(left) - float(right)) <= 0.002
                        for left, right in zip(
                            visual_twin["registration"]["principal"]["world_anchor_xy_m"],
                            principal_values["centre_xy_m"],
                        )
                    )
                ),
                "registered_principal_shell_and_floor_present": (
                    not visual_twin_enabled
                    or (len(principal_twin.get("walls", [])) >= 9 and len(principal_floor_polygon) >= 8)
                ),
                "scene_reopens": Usd.Stage.Open(str(path)) is not None,
            },
            "presentation_ready": True,
            "physical_route_released": False,
            "blocked_for_physical_release": [
                (
                    "measured-site candidate has not completed threshold contact and stopping-distance validation"
                    if measured_overlay is not None
                    else "both door clear widths and thresholds remain unmeasured"
                ),
                "wall thickness, occluded decorative detail, locked VP furniture and some traced PDF offsets remain presentation assumptions",
                "threshold contact requires the articulated compliant carrier and measured caster/spring properties",
                "sensors are not a protective safety system",
            ],
        }
        report["passed"] = all(report["checks"].values())
        output = RESULTS_DIR / "administration_build_report.json"
        gate_output = RESULTS_DIR / "administration_build_gate.json"
        write_json(output, report)
        write_json(
            gate_output,
            {
                "timestamp_utc": report["timestamp_utc"],
                "status": "plan_geometry_accepted_presentation_assumptions_disclosed",
                "scene": str(path),
                "presentation_ready": report["presentation_ready"],
                "physical_route_released": report["physical_route_released"],
                "authoritative_blockers": report["blocked_for_physical_release"],
            },
        )
        print(f"built plan-derived Block A presentation scene {path}")
        print(f"wrote {output}")
        return 0 if report["passed"] else 1
    finally:
        app.close()


def main() -> int:
    args = parse_args()
    if not args.presentation_assumptions and args.measured_geometry is None:
        return strict_gate(args)
    return build_presentation(args)


if __name__ == "__main__":
    raise SystemExit(main())
