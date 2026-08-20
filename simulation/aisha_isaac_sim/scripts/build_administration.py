#!/usr/bin/env python3
"""Build or gate the Block A administration scene.

Strict mode preserves the authoritative A1-plan/site-survey gate. The explicit
``--presentation-assumptions`` mode builds a route-scoped, visibly disclosed
proxy so a presentation can proceed without misrepresenting assumed geometry as
surveyed fact.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from aisha_common import CONFIG_DIR, RESULTS_DIR, SCENES_DIR, USD_DIR, ensure_output_dirs, load_yaml, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--payload", choices=("empty", "loaded"), default="loaded")
    parser.add_argument("--plan", type=Path, help="approved A1 building plan, page 2 PDF/DWG")
    parser.add_argument("--door-survey", type=Path, help="YAML/JSON with both clear widths and threshold heights")
    parser.add_argument(
        "--presentation-assumptions",
        action="store_true",
        help="build the disclosed route proxy from config/administration_assumptions.yaml",
    )
    return parser.parse_args()


def strict_gate(args: argparse.Namespace) -> int:
    ensure_output_dirs()
    blockers = []
    if args.plan is None or not args.plan.is_file():
        blockers.append("approved A1 building plan page 2 is missing")
    if args.door_survey is None or not args.door_survey.is_file():
        blockers.append("Principal/Vice-Principal clear-width and threshold survey is missing")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "inputs_present_not_yet_traced",
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
    print("Inputs are present. Authoritative plan tracing is not implemented by this proxy builder.")
    print(f"wrote {output}")
    return 3


def build_presentation(args: argparse.Namespace) -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})
    try:
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        ensure_output_dirs()
        assumptions_path = CONFIG_DIR / "administration_assumptions.yaml"
        assumptions = load_yaml(assumptions_path)
        physics = load_yaml(CONFIG_DIR / "physics_materials.yaml")
        sensors = load_yaml(CONFIG_DIR / "sensors.yaml")
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
            material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(float(values["static_friction"]))
            api.CreateDynamicFrictionAttr(float(values["dynamic_friction"]))
            api.CreateRestitutionAttr(float(values["restitution"]))
            physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            physx_api.CreateFrictionCombineModeAttr(str(values["friction_combine_mode"]))
            physx_api.CreateRestitutionCombineModeAttr(str(values["restitution_combine_mode"]))
            return material

        tile_material = physics_material("polished_tile", physics["materials"]["polished_tile"])
        drive_material = physics_material("drive_wheel", physics["materials"]["drive_wheel"])
        castor_material = physics_material("castor_low_friction", physics["materials"]["castor_low_friction"])

        def bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

        def box(
            prim_path: str,
            size_xyz: tuple[float, float, float],
            centre_xyz: tuple[float, float, float],
            color: tuple[float, float, float],
            *,
            collision: bool = True,
            material: UsdShade.Material | None = None,
            rotate_z_deg: float = 0.0,
        ) -> Usd.Prim:
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*centre_xyz))
            if rotate_z_deg:
                xform.AddRotateZOp().Set(rotate_z_deg)
            xform.AddScaleOp().Set(Gf.Vec3d(*size_xyz))
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if material is not None:
                bind(cube.GetPrim(), material)
            return cube.GetPrim()

        def wall_segment(name: str, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
            wall_height = float(assumptions["assumed_geometry"]["wall_height_m"]["value"])
            thickness = float(assumptions["assumed_geometry"]["wall_thickness_m"]["value"])
            dx = end_xy[0] - start_xy[0]
            dy = end_xy[1] - start_xy[1]
            length = math.hypot(dx, dy)
            centre = ((start_xy[0] + end_xy[0]) / 2.0, (start_xy[1] + end_xy[1]) / 2.0, wall_height / 2.0)
            box(
                f"/World/Architecture/Walls/{name}",
                (length, thickness, wall_height),
                centre,
                (0.82, 0.79, 0.72),
                rotate_z_deg=math.degrees(math.atan2(dy, dx)),
            )

        # Route floors. The octagonal outline retains the supplied 12.75 m
        # diagonal; the attached office cluster is a presentation assumption.
        box(
            "/World/Architecture/Floors/Atrium",
            (12.75, 12.75, 0.10),
            (0.0, 0.0, -0.05),
            (0.67, 0.68, 0.66),
            material=tile_material,
        )
        box(
            "/World/Architecture/Floors/Corridor",
            (12.11, 2.80, 0.10),
            (11.945, 0.0, -0.05),
            (0.67, 0.68, 0.66),
            material=tile_material,
        )
        box(
            "/World/Architecture/Floors/VicePrincipal",
            (4.30, 4.50, 0.10),
            (9.95, -3.65, -0.05),
            (0.76, 0.72, 0.66),
            material=tile_material,
        )
        box(
            "/World/Architecture/Floors/Principal",
            (4.60, 4.50, 0.10),
            (14.60, -3.65, -0.05),
            (0.76, 0.72, 0.66),
            material=tile_material,
        )

        # Regular-octagon atrium wall, with a 2.80 m opening on the east face.
        radius = float(assumptions["known_dimensions"]["atrium_diagonal_m"]["value"]) / 2.0
        vertices = [
            (radius * math.cos(math.radians(22.5 + 45.0 * index)), radius * math.sin(math.radians(22.5 + 45.0 * index)))
            for index in range(8)
        ]
        for index in range(7):
            wall_segment(f"Atrium_{index:02d}", vertices[index], vertices[index + 1])
        east_x = vertices[0][0]
        east_extent = abs(vertices[0][1])
        wall_segment("Atrium_East_North", (east_x, 1.40), (east_x, east_extent))
        wall_segment("Atrium_East_South", (east_x, -east_extent), (east_x, -1.40))

        # Corridor walls and assumed office envelopes.
        wall_segment("Corridor_North", (east_x, 1.40), (18.0, 1.40))
        wall_segment("Corridor_East", (18.0, -1.40), (18.0, 1.40))
        vp = assumptions["doors"]["vice_principal"]
        principal = assumptions["doors"]["principal"]
        door_intervals = []
        for values in (vp, principal):
            centre_x = float(values["centre_xy_m"][0])
            half_width = float(values["clear_width_m"]) / 2.0
            door_intervals.append((centre_x - half_width, centre_x + half_width))
        south_wall_starts = [east_x, door_intervals[0][1], door_intervals[1][1]]
        south_wall_ends = [door_intervals[0][0], door_intervals[1][0], 18.0]
        for index, (start_x, end_x) in enumerate(zip(south_wall_starts, south_wall_ends)):
            wall_segment(f"Corridor_South_{index:02d}", (start_x, -1.40), (end_x, -1.40))
        wall_segment("Vice_West", (7.80, -5.90), (7.80, -1.40))
        wall_segment("Vice_East", (12.10, -5.90), (12.10, -1.40))
        wall_segment("Vice_South", (7.80, -5.90), (12.10, -5.90))
        wall_segment("Principal_West", (12.30, -5.90), (12.30, -1.40))
        wall_segment("Principal_East", (16.90, -5.90), (16.90, -1.40))
        wall_segment("Principal_South", (12.30, -5.90), (16.90, -5.90))

        def doorway(name: str, values: dict[str, object], hinge: str) -> dict[str, object]:
            centre_x, centre_y = (float(v) for v in values["centre_xy_m"])
            width = float(values["clear_width_m"])
            height = 2.40
            post = 0.08
            left_inner = centre_x - width / 2.0
            right_inner = centre_x + width / 2.0
            box(
                f"/World/Architecture/Doors/{name}/FrameLeft",
                (post, 0.18, height),
                (left_inner - post / 2.0, centre_y, height / 2.0),
                (0.28, 0.15, 0.08),
            )
            box(
                f"/World/Architecture/Doors/{name}/FrameRight",
                (post, 0.18, height),
                (right_inner + post / 2.0, centre_y, height / 2.0),
                (0.28, 0.15, 0.08),
            )
            box(
                f"/World/Architecture/Doors/{name}/Lintel",
                (width + 2.0 * post, 0.18, 0.60),
                (centre_x, centre_y, height + 0.30),
                (0.28, 0.15, 0.08),
            )
            hinge_x = left_inner - 0.025 if hinge == "left" else right_inner + 0.025
            box(
                f"/World/Architecture/Doors/{name}/OpenLeaf",
                (0.05, width, 2.25),
                (hinge_x, centre_y - width / 2.0, 1.125),
                (0.42, 0.22, 0.10),
            )
            threshold_m = float(values["threshold_height_mm"]) / 1000.0
            threshold = box(
                f"/World/Architecture/Doors/{name}/Threshold",
                (width, 0.10, threshold_m),
                (centre_x, centre_y, threshold_m / 2.0),
                (0.44, 0.45, 0.47),
                material=tile_material,
            )
            threshold.SetCustomDataByKey("aisha:status", "presentation_assumption_not_measured")
            threshold.SetCustomDataByKey("aisha:heightMm", int(values["threshold_height_mm"]))
            return {
                "clear_width_m": width,
                "threshold_height_mm": int(values["threshold_height_mm"]),
                "width_status": values["width_status"],
                "threshold_status": values["threshold_status"],
            }

        doors = {
            "vice_principal": doorway("VicePrincipal", vp, "left"),
            "principal": doorway("Principal", principal, "right"),
        }

        # Appearance cues from the walkthrough: timber reception fronts, glazed
        # partitions, dark seating, light office floors, and office furniture.
        box("/World/Furniture/ReceptionCounter", (4.20, 0.70, 1.10), (-0.80, 3.35, 0.55), (0.36, 0.18, 0.08))
        box("/World/Furniture/AtriumBench", (2.40, 0.65, 0.75), (-0.40, -3.10, 0.375), (0.08, 0.09, 0.10))
        box("/World/Furniture/ViceDesk", (2.10, 0.85, 0.78), (9.95, -5.15, 0.39), (0.34, 0.18, 0.09))
        box("/World/Furniture/PrincipalDesk", (2.20, 0.90, 0.78), (14.60, -5.10, 0.39), (0.28, 0.15, 0.08))
        for room_name, x in (("Vice", 9.95), ("Principal", 14.60)):
            box(f"/World/Furniture/{room_name}VisitorChairLeft", (0.55, 0.55, 0.85), (x - 0.65, -4.10, 0.425), (0.08, 0.09, 0.10))
            box(f"/World/Furniture/{room_name}VisitorChairRight", (0.55, 0.55, 0.85), (x + 0.65, -4.10, 0.425), (0.08, 0.09, 0.10))

        # The route line and markers are visual aids, excluded from collision.
        route = assumptions["route"]["waypoints"]
        points = [Gf.Vec3f(float(item["x_m"]), float(item["y_m"]), 0.025) for item in route]
        curve = UsdGeom.BasisCurves.Define(stage, "/World/Presentation/Route")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr(points)
        curve.CreateWidthsAttr([0.075])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.55, 0.95)])
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
            marker.GetPrim().SetCustomDataByKey("aisha:poseStatus", "presentation_assumption")

        robot = UsdGeom.Xform.Define(stage, "/World/AISHA")
        robot.GetPrim().GetReferences().AddReference(f"../usd/{asset.name}")
        robot.GetPrim().SetCustomDataByKey("aisha:payloadVariant", args.payload)
        robot.GetPrim().SetCustomDataByKey("aisha:routeGeometryStatus", "presentation_assumption_not_surveyed")

        sensor_bindings = []
        contact_bindings = {"drive_wheel": [], "castor_low_friction": []}
        for prim in stage.TraverseAll():
            name = prim.GetName()
            if name in ("left_wheel_link", "right_wheel_link"):
                bind(prim, drive_material)
                contact_bindings["drive_wheel"].append(str(prim.GetPath()))
            elif name in ("castor_fl_link", "castor_fr_link", "castor_rl_link", "castor_rr_link"):
                bind(prim, castor_material)
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

        sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
        sun.CreateIntensityAttr(650.0)
        sun.CreateAngleAttr(0.5)
        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Ambient")
        dome.CreateIntensityAttr(350.0)
        dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
        for index, x in enumerate((-3.0, 1.0, 5.0, 9.0, 13.0, 17.0)):
            light = UsdLux.RectLight.Define(stage, f"/World/Lighting/Ceiling_{index:02d}")
            light.CreateIntensityAttr(900.0)
            light.CreateWidthAttr(1.2)
            light.CreateHeightAttr(0.6)
            light_xform = UsdGeom.Xformable(light.GetPrim())
            light_xform.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 2.85))

        camera = UsdGeom.Camera.Define(stage, "/World/Presentation/HeroCamera")
        camera.CreateFocalLengthAttr(28.0)
        camera.CreateFocusDistanceAttr(16.0)
        camera.CreateFStopAttr(5.6)
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.AddTranslateOp().Set(Gf.Vec3d(-9.5, -12.5, 10.0))
        camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(58.0, 0.0, -38.0))

        minimum = float(assumptions["presentation_release"]["minimum_demo_door_clear_width_m"])
        robot_width = float(assumptions["presentation_release"]["robot_transit_width_m"])
        for values in doors.values():
            values["minimum_required_m"] = minimum
            values["nominal_side_clearance_m"] = (float(values["clear_width_m"]) - robot_width) / 2.0
            values["presentation_width_gate_passed"] = float(values["clear_width_m"]) >= minimum
            values["high_fidelity_threshold_validation"] = "blocked"

        stage.GetRootLayer().customLayerData = {
            "aisha:scenePurpose": "scripted_presentation_proxy",
            "aisha:a1Page2Status": "unavailable_not_confirmed",
            "aisha:geometryStatus": "presentation_assumption_not_surveyed",
            "aisha:physicalRelease": False,
            "aisha:productionRepositoryCommit": assumptions["provenance"]["production_repository"]["commit"],
        }
        stage.GetRootLayer().Save()

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "presentation_proxy_built",
            "scene": str(path),
            "payload": args.payload,
            "robot_asset": str(asset),
            "a1_page_2": {
                "status": "unavailable_not_confirmed",
                "authoritative_geometry_used": False,
                "plan_argument": str(args.plan.resolve()) if args.plan and args.plan.exists() else None,
                "plan_sha256": sha256_file(args.plan) if args.plan and args.plan.exists() else None,
            },
            "door_survey": {
                "status": "not_supplied_assumptions_used",
                "survey_argument": str(args.door_survey.resolve()) if args.door_survey and args.door_survey.exists() else None,
            },
            "assumptions_file": str(assumptions_path),
            "assumptions_sha256": sha256_file(assumptions_path),
            "known_dimensions": assumptions["known_dimensions"],
            "doors": doors,
            "route": route,
            "sensor_contracts": sensor_bindings,
            "contact_material_bindings": contact_bindings,
            "checks": {
                "all_presentation_door_widths_pass": all(item["presentation_width_gate_passed"] for item in doors.values()),
                "hallway_supports_pivot_circle": float(assumptions["known_dimensions"]["hallway_clear_width_m"]["value"])
                >= float(assumptions["presentation_release"]["pivot_clear_circle_m"]),
                "scene_reopens": Usd.Stage.Open(str(path)) is not None,
            },
            "presentation_ready": True,
            "physical_route_released": False,
            "blocked_for_physical_release": [
                "approved A1 page-2 plan remains unavailable",
                "both door clear widths and thresholds remain unmeasured",
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
                "status": "presentation_assumptions_accepted",
                "scene": str(path),
                "presentation_ready": report["presentation_ready"],
                "physical_route_released": report["physical_route_released"],
                "authoritative_blockers": report["blocked_for_physical_release"],
            },
        )
        print(f"built disclosed presentation proxy {path}")
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
