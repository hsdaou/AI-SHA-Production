#!/usr/bin/env python3
"""Build or gate the plan-derived Block A administration presentation scene."""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from aisha_common import CONFIG_DIR, RESULTS_DIR, SCENES_DIR, USD_DIR, ensure_output_dirs, load_yaml, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--payload", choices=("empty", "loaded"), default="loaded")
    parser.add_argument("--plan", type=Path, help="approved ground-floor plan PDF containing page 2 Block A")
    parser.add_argument("--door-survey", type=Path, help="YAML/JSON with both clear widths and threshold heights")
    parser.add_argument(
        "--presentation-assumptions",
        action="store_true",
        help="accept disclosed presentation-only door/threshold and height assumptions",
    )
    return parser.parse_args()


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
        physics = load_yaml(CONFIG_DIR / "physics_materials.yaml")
        sensors = load_yaml(CONFIG_DIR / "sensors.yaml")
        expected_plan_hash = str(config["provenance"]["plan_source"]["sha256"])
        supplied_plan_hash = sha256_file(args.plan) if args.plan and args.plan.is_file() else None
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

        tile_physics = physics_material("polished_tile", physics["materials"]["polished_tile"])
        drive_material = physics_material("drive_wheel", physics["materials"]["drive_wheel"])
        castor_material = physics_material("castor_low_friction", physics["materials"]["castor_low_friction"])

        warm_white = visual_material("WarmWhite", (0.90, 0.89, 0.85), roughness=0.62)
        light_grey = visual_material("LightGrey", (0.64, 0.67, 0.69), roughness=0.55)
        dark_grey = visual_material("DarkGrey", (0.11, 0.13, 0.15), roughness=0.48)
        black = visual_material("Black", (0.025, 0.028, 0.032), roughness=0.40)
        terrazzo = visual_material("PolishedTerrazzo", (0.56, 0.58, 0.59), roughness=0.18)
        terrazzo_dark = visual_material("TerrazzoDarkChip", (0.18, 0.20, 0.21), roughness=0.30)
        terrazzo_light = visual_material("TerrazzoLightChip", (0.82, 0.82, 0.79), roughness=0.28)
        timber = visual_material("WarmTimber", (0.34, 0.14, 0.045), roughness=0.34)
        timber_light = visual_material("LightTimber", (0.53, 0.29, 0.11), roughness=0.38)
        oak = visual_material("LightOakFloor", (0.67, 0.53, 0.35), roughness=0.55)
        green = visual_material("SchoolGreen", (0.13, 0.31, 0.18), roughness=0.48)
        leaf_green = visual_material("PlantGreen", (0.05, 0.27, 0.08), roughness=0.70)
        glass = visual_material("FrostedGlass", (0.66, 0.77, 0.80), roughness=0.16, opacity=0.28)
        metal = visual_material("BrushedMetal", (0.42, 0.45, 0.48), roughness=0.24, metallic=0.65)
        aisha_white = visual_material("AISHAWhite", (0.82, 0.86, 0.86), roughness=0.24)
        aisha_green = visual_material("AISHAGreen", (0.03, 0.38, 0.24), roughness=0.30, metallic=0.05)
        aisha_black = visual_material("AISHABlack", (0.015, 0.022, 0.025), roughness=0.20, metallic=0.20)
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

        def polygon_floor(name: str, points_xy: list[tuple[float, float]], material: UsdShade.Material) -> None:
            mesh = UsdGeom.Mesh.Define(stage, f"/World/Architecture/Floors/{name}")
            mesh.CreatePointsAttr([Gf.Vec3f(x, y, 0.002) for x, y in points_xy])
            mesh.CreateFaceVertexCountsAttr([len(points_xy)])
            mesh.CreateFaceVertexIndicesAttr(list(range(len(points_xy))))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            bind_visual(mesh.GetPrim(), material)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            bind_physics(mesh.GetPrim(), tile_physics)

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

        def doorway(name: str, values: dict[str, object], *, hinge_left: bool) -> dict[str, object]:
            centre_x, centre_y = (float(v) for v in values["centre_xy_m"])
            angle_deg = float(values["wall_rotation_deg"])
            angle = math.radians(angle_deg)
            tangent = (math.cos(angle), math.sin(angle))
            normal = (-math.sin(angle), math.cos(angle))
            width = float(values["clear_width_m"])
            height = 2.25
            post = 0.075
            for side_name, side_sign in (("Left", -1.0), ("Right", 1.0)):
                offset = side_sign * (width / 2.0 + post / 2.0)
                box(
                    f"/World/Architecture/Doors/{name}/Frame{side_name}",
                    (post, 0.19, height),
                    (centre_x + tangent[0] * offset, centre_y + tangent[1] * offset, height / 2.0),
                    timber_light,
                    rotate_z_deg=angle_deg,
                )
            box(
                f"/World/Architecture/Doors/{name}/Lintel",
                (width + 2.0 * post, 0.19, 0.18),
                (centre_x, centre_y, height + 0.09),
                timber_light,
                rotate_z_deg=angle_deg,
            )
            hinge_sign = -1.0 if hinge_left else 1.0
            hinge = (
                centre_x + tangent[0] * hinge_sign * width / 2.0,
                centre_y + tangent[1] * hinge_sign * width / 2.0,
            )
            leaf_centre = (hinge[0] - normal[0] * width / 2.0, hinge[1] - normal[1] * width / 2.0)
            box(
                f"/World/Architecture/Doors/{name}/OpenLeaf",
                (width, 0.045, 2.18),
                (leaf_centre[0], leaf_centre[1], 1.09),
                timber_light,
                rotate_z_deg=angle_deg + 90.0,
            )
            threshold_m = float(values["threshold_height_mm"]) / 1000.0
            threshold = box(
                f"/World/Architecture/Doors/{name}/Threshold",
                (width, 0.12, threshold_m),
                (centre_x, centre_y, threshold_m / 2.0),
                metal,
                physics_binding=tile_physics,
                rotate_z_deg=angle_deg,
            )
            threshold.SetCustomDataByKey("aisha:status", "presentation_assumption_not_measured")
            threshold.SetCustomDataByKey("aisha:heightMm", int(values["threshold_height_mm"]))
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
                "threshold_height_mm": int(values["threshold_height_mm"]),
                "width_status": values["width_status"],
                "threshold_status": values["threshold_status"],
                "centre_xy_m": [centre_x, centre_y],
                "wall_rotation_deg": angle_deg,
            }

        def slatted_wall(name: str, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
            wall_segment(name + "_Backing", start_xy, end_xy, timber)
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

        def desk(name: str, centre_xy: tuple[float, float], yaw_deg: float = 0.0) -> None:
            box(f"/World/Furniture/{name}/Top", (2.00, 0.82, 0.09), (*centre_xy, 0.76), timber_light, rotate_z_deg=yaw_deg)
            for pedestal_index, local_y in enumerate((-0.31, 0.31)):
                position = local_to_world(centre_xy, (-0.78, local_y), yaw_deg)
                box(f"/World/Furniture/{name}/Pedestal_{pedestal_index}", (0.34, 0.26, 0.68), (*position, 0.36), timber, rotate_z_deg=yaw_deg)

        def chair(name: str, centre_xy: tuple[float, float], yaw_deg: float = 0.0) -> None:
            box(f"/World/Furniture/{name}/Seat", (0.50, 0.50, 0.10), (*centre_xy, 0.48), black, rotate_z_deg=yaw_deg)
            back_xy = local_to_world(centre_xy, (-0.24, 0.0), yaw_deg)
            box(f"/World/Furniture/{name}/Back", (0.08, 0.50, 0.72), (*back_xy, 0.78), black, rotate_z_deg=yaw_deg)
            for x_index, sx in enumerate((-0.18, 0.18)):
                for y_index, sy in enumerate((-0.18, 0.18)):
                    leg_xy = local_to_world(centre_xy, (sx, sy), yaw_deg)
                    box(f"/World/Furniture/{name}/Leg_{x_index}_{y_index}", (0.035, 0.035, 0.44), (*leg_xy, 0.22), metal, rotate_z_deg=yaw_deg)

        def plant(name: str, centre_xy: tuple[float, float]) -> None:
            cylinder(f"/World/Furniture/{name}/Pot", 0.24, 0.42, (*centre_xy, 0.21), light_grey)
            cylinder(f"/World/Furniture/{name}/Stem", 0.035, 0.70, (*centre_xy, 0.72), timber, collision=False)
            for index, (dx, dy, dz) in enumerate(((0.0, 0.0, 1.18), (0.24, 0.0, 1.10), (-0.22, 0.05, 1.08), (0.0, 0.22, 1.12), (0.05, -0.22, 1.06))):
                sphere(f"/World/Furniture/{name}/Leaf_{index}", 0.24, (centre_xy[0] + dx, centre_xy[1] + dy, dz), leaf_green)

        # Floors and support slab.
        box("/World/Architecture/SupportSlab", (48.0, 32.0, 0.12), (7.0, -4.0, -0.065), dark_grey, physics_binding=tile_physics)
        radius = float(config["known_dimensions"]["atrium_diagonal_m"]["value"]) / 2.0
        vertices = [
            (radius * math.cos(math.radians(22.5 + 45.0 * index)), radius * math.sin(math.radians(22.5 + 45.0 * index)))
            for index in range(8)
        ]
        polygon_floor("Atrium", vertices, terrazzo)
        box("/World/Architecture/Floors/EastHallway", (16.11, 2.80, 0.055), (13.945, 0.0, -0.027), terrazzo, physics_binding=tile_physics)
        box("/World/Architecture/Floors/ViceAccess", (2.40, 3.65, 0.055), (17.10, -3.225, -0.027), terrazzo, physics_binding=tile_physics)
        box("/World/Architecture/Floors/VicePrincipal", (6.30, 3.00, 0.055), (17.10, -6.55, -0.027), oak, physics_binding=tile_physics)
        box("/World/Architecture/Floors/PrincipalAccess", (5.80, 2.60, 0.055), (5.45, -5.45, -0.027), terrazzo, physics_binding=tile_physics, rotate_z_deg=-45.0)
        box("/World/Architecture/Floors/Principal", (4.75, 3.60, 0.055), (8.65, -9.30, -0.027), oak, physics_binding=tile_physics, rotate_z_deg=-45.0)

        cluster = config["plan_geometry"]["south_east_cluster"]
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
                wall_segment("Atrium_East_South", start, (start[0], -1.40), warm_white)
                wall_segment("Atrium_East_North", (start[0], 1.40), end, warm_white)
                continue
            material = timber if index in (0, 1) else warm_white
            wall_segment(f"Atrium_{index:02d}", start, end, material)

        wall_segment("EastHall_North", (vertices[0][0], 1.40), (22.00, 1.40), warm_white)
        slatted_wall("EastHall_South_West", (vertices[7][0], -1.40), (15.90, -1.40))
        slatted_wall("EastHall_South_East", (18.30, -1.40), (22.00, -1.40))
        split_wall("EastHall_End", (22.00, -1.40), (22.00, 1.40), (22.00, 0.0), 1.80, light_grey)

        # Frosted-glass double doors at the east side entrance.
        for y in (-0.48, 0.48):
            box("/World/Architecture/Glass/EastDoor_" + ("S" if y < 0 else "N"), (0.055, 0.88, 2.30), (22.00, y, 1.15), glass, rotate_z_deg=90.0)
            box("/World/Architecture/Glass/EastDoorBand_" + ("S" if y < 0 else "N"), (0.060, 0.88, 0.12), (21.97, y, 1.20), warm_white, collision=False, rotate_z_deg=90.0)

        # Vice-Principal access and room, east of the angled Principal office as
        # shown on page 2.
        wall_segment("ViceAccess_West", (15.90, -5.05), (15.90, -1.40), timber)
        wall_segment("ViceAccess_East", (18.30, -5.05), (18.30, -1.40), warm_white)
        vp_values = config["doors"]["vice_principal"]
        vp_half = float(vp_values["clear_width_m"]) / 2.0
        wall_segment("Vice_North_West", (13.95, -5.05), (17.10 - vp_half, -5.05), warm_white)
        wall_segment("Vice_North_East", (17.10 + vp_half, -5.05), (20.25, -5.05), warm_white)
        slatted_wall("Vice_South", (13.95, -8.05), (20.25, -8.05))
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
            offset = (normal[0] * 1.30 * side_sign, normal[1] * 1.30 * side_sign)
            wall_segment(
                f"PrincipalAccess_{side_name}",
                (corridor_start[0] + offset[0], corridor_start[1] + offset[1]),
                (corridor_end[0] + offset[0], corridor_end[1] + offset[1]),
                material,
            )

        principal_centre = (8.65, -9.30)
        principal_size = (4.75, 3.60)
        principal_rotation = -45.0
        principal_corners = [
            local_to_world(principal_centre, (-principal_size[0] / 2.0, -principal_size[1] / 2.0), principal_rotation),
            local_to_world(principal_centre, (principal_size[0] / 2.0, -principal_size[1] / 2.0), principal_rotation),
            local_to_world(principal_centre, (principal_size[0] / 2.0, principal_size[1] / 2.0), principal_rotation),
            local_to_world(principal_centre, (-principal_size[0] / 2.0, principal_size[1] / 2.0), principal_rotation),
        ]
        principal_door_centre = local_to_world(principal_centre, (-principal_size[0] / 2.0, 0.0), principal_rotation)
        config["doors"]["principal"]["centre_xy_m"] = [round(principal_door_centre[0], 3), round(principal_door_centre[1], 3)]
        config["doors"]["principal"]["wall_rotation_deg"] = 45.0
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
            "vice_principal": doorway("VicePrincipal", vp_values, hinge_left=True),
            "principal": doorway("Principal", config["doors"]["principal"], hinge_left=False),
        }

        # Walkthrough-derived furniture and finishes.
        box("/World/Furniture/Reception/Base", (4.20, 0.78, 1.08), (-1.10, 3.45, 0.54), timber)
        box("/World/Furniture/Reception/Counter", (4.35, 0.92, 0.09), (-1.10, 3.45, 1.10), timber_light)
        for index in range(40):
            x = -3.05 + index * 0.10
            box(f"/World/Furniture/Reception/Slat_{index:02d}", (0.035, 0.055, 0.86), (x, 3.01, 0.50), timber_light, collision=False)
        box("/World/Furniture/AtriumBench/Seat", (2.60, 0.70, 0.16), (-1.00, -3.35, 0.46), black)
        box("/World/Furniture/AtriumBench/Back", (2.60, 0.12, 0.70), (-1.00, -3.68, 0.78), black)
        plant("AtriumPlant", (2.20, 3.45))
        plant("EastHallPlant", (20.70, 0.70))

        desk("ViceDesk", (17.10, -7.35), 0.0)
        chair("ViceDeskChair", (17.10, -7.82), 90.0)
        chair("ViceVisitorLeft", (15.75, -6.65), -90.0)
        chair("ViceVisitorRight", (18.45, -6.65), -90.0)
        box("/World/Furniture/ViceCabinet", (2.00, 0.38, 0.82), (14.30, -7.72, 0.41), timber)
        plant("VicePlant", (19.55, -7.45))

        desk_local = local_to_world(principal_centre, (0.55, -0.55), principal_rotation)
        desk("PrincipalDesk", desk_local, principal_rotation)
        principal_chair = local_to_world(principal_centre, (1.20, -1.15), principal_rotation)
        chair("PrincipalDeskChair", principal_chair, principal_rotation + 180.0)
        for name, local in (("PrincipalVisitorLeft", (-0.40, 1.15)), ("PrincipalVisitorRight", (-0.40, -1.15))):
            chair(name, local_to_world(principal_centre, local, principal_rotation), principal_rotation)
        principal_cabinet = local_to_world(principal_centre, (1.90, 0.80), principal_rotation)
        box("/World/Furniture/PrincipalCabinet", (0.40, 1.60, 0.86), (*principal_cabinet, 0.43), timber, rotate_z_deg=principal_rotation)
        plant("PrincipalPlant", local_to_world(principal_centre, (-1.45, 1.15), principal_rotation))

        # Terrazzo aggregate and geometric inlay cues from the walkthrough.
        rng = random.Random(20260820)
        floor_regions = [(-5.0, 5.0, -5.0, 5.0), (5.8, 21.8, -1.28, 1.28), (15.98, 18.22, -4.95, -1.50)]
        for index in range(280):
            xmin, xmax, ymin, ymax = floor_regions[index % len(floor_regions)]
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            size = rng.uniform(0.016, 0.052)
            chip_material = terrazzo_dark if index % 3 else terrazzo_light
            box(f"/World/Appearance/TerrazzoChip_{index:03d}", (size, size * rng.uniform(0.5, 1.6), 0.003), (x, y, 0.006), chip_material, collision=False, rotate_z_deg=rng.uniform(0.0, 180.0))
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
        box("/World/Architecture/Ceilings/Atrium", (11.20, 11.20, 0.08), (0.0, 0.0, 3.04), warm_white, collision=False)
        box("/World/Architecture/Ceilings/EastHall", (16.11, 2.80, 0.08), (13.945, 0.0, 3.04), warm_white, collision=False)
        box("/World/Architecture/Ceilings/ViceAccess", (2.40, 3.65, 0.08), (17.10, -3.225, 3.04), warm_white, collision=False)
        box("/World/Architecture/Ceilings/Vice", (6.30, 3.00, 0.08), (17.10, -6.55, 3.04), warm_white, collision=False)
        box("/World/Architecture/Ceilings/PrincipalAccess", (5.80, 2.60, 0.08), (5.45, -5.45, 3.04), warm_white, collision=False, rotate_z_deg=-45.0)
        box("/World/Architecture/Ceilings/Principal", (4.75, 3.60, 0.08), (8.65, -9.30, 3.04), warm_white, collision=False, rotate_z_deg=-45.0)

        light_positions = [
            (-3.0, 0.0), (0.0, 0.0), (3.0, 0.0),
            (7.0, 0.0), (10.0, 0.0), (13.0, 0.0), (16.0, 0.0), (19.0, 0.0),
            (17.1, -3.1), (15.4, -6.55), (18.7, -6.55),
            (5.4, -5.4), (8.0, -8.6), (9.7, -9.9),
        ]
        for index, (x, y) in enumerate(light_positions):
            box(f"/World/Lighting/Panels/Panel_{index:02d}", (0.85, 0.55, 0.025), (x, y, 2.985), light_panel, collision=False)
            light = UsdLux.RectLight.Define(stage, f"/World/Lighting/Fixtures/Light_{index:02d}")
            light.CreateIntensityAttr(16000.0)
            light.CreateColorAttr(Gf.Vec3f(0.93, 0.96, 1.0))
            light.CreateWidthAttr(0.85)
            light.CreateHeightAttr(0.55)
            light_xform = UsdGeom.Xformable(light.GetPrim())
            light_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 2.96))

        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Ambient")
        dome.CreateIntensityAttr(1800.0)
        dome.CreateColorAttr(Gf.Vec3f(0.78, 0.84, 0.92))
        sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
        sun.CreateIntensityAttr(1800.0)
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
            if item["action"] not in ("start_and_end", "presentation_stop"):
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
        robot.GetPrim().SetCustomDataByKey("aisha:routeGeometryStatus", "plan_derived_route_scoped")

        # A presentation shell gives the imported engineering URDF the intended
        # finished-product read while keeping the Rev D collision and sensor
        # frames unchanged underneath it.
        box("/World/AISHA/PresentationShell/Body", (0.82, 0.59, 0.34), (0.02, 0.0, 0.39), aisha_white, collision=False)
        box("/World/AISHA/PresentationShell/LowerBumper", (0.86, 0.63, 0.10), (0.03, 0.0, 0.245), aisha_black, collision=False)
        box("/World/AISHA/PresentationShell/Tray", (0.78, 0.57, 0.055), (-0.03, 0.0, 0.555), aisha_green, collision=False)
        box("/World/AISHA/PresentationShell/Mast", (0.11, 0.13, 0.46), (0.42, 0.0, 0.72), aisha_white, collision=False)
        sphere("/World/AISHA/PresentationShell/Head", 0.232, (0.50, 0.0, 0.925), aisha_white, collision=False)
        box("/World/AISHA/PresentationShell/Face", (0.025, 0.265, 0.135), (0.728, 0.0, 0.94), aisha_black, collision=False)
        box("/World/AISHA/PresentationShell/FaceAccent", (0.028, 0.15, 0.020), (0.742, 0.0, 0.91), aisha_green, collision=False)
        for side, y in (("Left", 0.301), ("Right", -0.301)):
            box(f"/World/AISHA/PresentationShell/{side}Accent", (0.42, 0.018, 0.08), (0.03, y, 0.42), aisha_green, collision=False)

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
        for values in doors.values():
            values["minimum_required_m"] = minimum
            values["nominal_side_clearance_m"] = (float(values["clear_width_m"]) - robot_width) / 2.0
            values["presentation_width_gate_passed"] = float(values["clear_width_m"]) >= minimum
            values["high_fidelity_threshold_validation"] = "blocked"

        stage.GetRootLayer().customLayerData = {
            "aisha:scenePurpose": "plan_derived_scripted_cinematic",
            "aisha:a1Page2Status": "approved_page_2_reviewed",
            "aisha:planSha256": expected_plan_hash,
            "aisha:geometryStatus": "plan_derived_route_scoped",
            "aisha:appearanceStatus": "walkthrough_video_derived_not_dimensioned",
            "aisha:physicalRelease": False,
            "aisha:productionRepositoryCommit": config["provenance"]["production_repository"]["commit"],
        }
        stage.GetRootLayer().Save()

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "plan_derived_presentation_scene_built",
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
                "status": "not_supplied_assumptions_used",
                "survey_argument": str(args.door_survey.resolve()) if args.door_survey and args.door_survey.exists() else None,
            },
            "config_file": str(config_path),
            "config_sha256": sha256_file(config_path),
            "known_dimensions": config["known_dimensions"],
            "plan_geometry": config["plan_geometry"],
            "appearance": config["appearance"],
            "doors": doors,
            "route": route,
            "sensor_contracts": sensor_bindings,
            "contact_material_bindings": contact_bindings,
            "checks": {
                "plan_hash_matches_reviewed_source": supplied_plan_hash in (None, expected_plan_hash),
                "all_presentation_door_widths_pass": all(item["presentation_width_gate_passed"] for item in doors.values()),
                "hallway_supports_pivot_circle": float(config["known_dimensions"]["hallway_clear_width_m"]["value"])
                >= float(config["presentation_release"]["pivot_clear_circle_m"]),
                "vice_principal_is_east_of_principal": float(cluster["vice_principal"]["centre_xy_m"][0])
                > float(cluster["principal"]["centre_xy_m"][0]),
                "scene_reopens": Usd.Stage.Open(str(path)) is not None,
            },
            "presentation_ready": True,
            "physical_route_released": False,
            "blocked_for_physical_release": [
                "both door clear widths and thresholds remain unmeasured",
                "ceiling, wall thickness, office furniture offsets and some traced PDF offsets are presentation assumptions",
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
    if not args.presentation_assumptions:
        return strict_gate(args)
    return build_presentation(args)


if __name__ == "__main__":
    raise SystemExit(main())
