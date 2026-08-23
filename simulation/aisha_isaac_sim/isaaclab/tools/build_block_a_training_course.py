#!/usr/bin/env python3
"""Build the lightweight, plan-aligned Block A collision course used by Isaac Lab."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml
from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PACKAGE_ROOT / "config" / "administration_assumptions.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "usd" / "block_a_training_course.usda"
DEFAULT_REPORT = PACKAGE_ROOT / "results" / "block_a_training_course_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deep_merge(base: dict, overlay: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--measured-geometry",
        type=Path,
        help="simulation-only measured presentation overlay for doors and no-go geometry",
    )
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    measured_overlay = None
    if args.measured_geometry is not None:
        overlay_path = args.measured_geometry.expanduser().resolve()
        measured_overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
        if measured_overlay.get("status") != "measured_site_presentation_candidate":
            raise ValueError("training overlay must be a measured-site presentation candidate")
        if measured_overlay.get("candidate_simulation_route_geometry_valid") is not True:
            raise ValueError("training overlay failed its simulation route gate")
        if measured_overlay.get("physical_release") is not False:
            raise ValueError("training overlay must preserve physical_release: false")
        config = deep_merge(config, measured_overlay)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    course = UsdGeom.Xform.Define(stage, "/Course")
    stage.SetDefaultPrim(course.GetPrim())

    def material(name: str, color: tuple[float, float, float], roughness: float) -> UsdShade.Material:
        value = UsdShade.Material.Define(stage, f"/Course/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/Course/Looks/{name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        value.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return value

    wall_material = material("Wall", (0.80, 0.82, 0.84), 0.72)
    accent_material = material("Accent", (0.18, 0.38, 0.27), 0.55)
    floor_material = material("Floor", (0.32, 0.34, 0.36), 0.40)
    furniture_material = material("Furniture", (0.28, 0.14, 0.06), 0.55)
    threshold_material = material("Threshold", (0.36, 0.39, 0.42), 0.30)

    authored_colliders: list[str] = []

    def box(
        name: str,
        size: tuple[float, float, float],
        centre: tuple[float, float, float],
        visual: UsdShade.Material,
        yaw_deg: float = 0.0,
    ) -> Usd.Prim:
        cube = UsdGeom.Cube.Define(stage, f"/Course/Geometry/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        if yaw_deg:
            xform.AddRotateZOp().Set(float(yaw_deg))
        xform.AddScaleOp().Set(Gf.Vec3d(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        # Keep solver contact skin from consuming the 41 mm/side physical
        # clearance of the measured 0.85 m stress-test apertures. The loaded
        # robot uses a 10 mm contact offset; a 5 mm course offset leaves a
        # conservative geometric margin without enlarging the doorway.
        physx_collision.CreateContactOffsetAttr().Set(0.005)
        physx_collision.CreateRestOffsetAttr().Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(visual)
        authored_colliders.append(str(cube.GetPath()))
        return cube.GetPrim()

    wall_height = float(config["plan_geometry"]["wall_height_m"]["value"])
    wall_thickness = float(config["plan_geometry"]["wall_thickness_m"]["value"])

    def wall(name: str, start: tuple[float, float], end: tuple[float, float]) -> None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-4:
            return
        box(
            "Wall_" + name,
            (length, wall_thickness, wall_height),
            ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, wall_height / 2.0),
            wall_material,
            math.degrees(math.atan2(dy, dx)),
        )

    def split_wall(
        name: str,
        start: tuple[float, float],
        end: tuple[float, float],
        opening: tuple[float, float],
        width: float,
    ) -> None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        tx, ty = dx / length, dy / length
        opening_t = (opening[0] - start[0]) * tx + (opening[1] - start[1]) * ty
        before = max(0.0, opening_t - width / 2.0)
        after = min(length, opening_t + width / 2.0)
        wall(name + "_A", start, (start[0] + tx * before, start[1] + ty * before))
        wall(name + "_B", (start[0] + tx * after, start[1] + ty * after), end)

    def local_to_world(
        centre: tuple[float, float], point: tuple[float, float], yaw_deg: float
    ) -> tuple[float, float]:
        angle = math.radians(yaw_deg)
        return (
            centre[0] + point[0] * math.cos(angle) - point[1] * math.sin(angle),
            centre[1] + point[0] * math.sin(angle) + point[1] * math.cos(angle),
        )

    # One support slab keeps all physical route segments on the same surface.
    box("Floor", (48.0, 32.0, 0.10), (7.0, -4.0, -0.05), floor_material)

    radius = float(config["known_dimensions"]["atrium_diagonal_m"]["value"]) / 2.0
    vertices = [
        (radius * math.cos(math.radians(22.5 + 45.0 * index)), radius * math.sin(math.radians(22.5 + 45.0 * index)))
        for index in range(8)
    ]
    central_polygon = config["plan_geometry"]["atrium"]["central_polygon"]
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
        start, end = central_vertices[index], central_vertices[(index + 1) % 8]
        dx, dy = end[0] - start[0], end[1] - start[1]
        barrier = box(
            f"CentralDropNavigationBarrier_{index:02d}",
            (math.hypot(dx, dy), 0.04, 1.25),
            ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, 0.625),
            accent_material,
            math.degrees(math.atan2(dy, dx)),
        )
        barrier.SetCustomDataByKey("aisha:proxyPurpose", "central_drop_no_go_lidar_and_collision")
        UsdGeom.Imageable(barrier).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    for index in range(8):
        start, end = vertices[index], vertices[(index + 1) % 8]
        if index == 6:  # route opening to Principal suite
            continue
        if index == 7:  # 2.80 m east hallway opening
            wall("AtriumEastSouth", start, (start[0], -1.40))
            wall("AtriumEastNorth", (start[0], 1.40), end)
        else:
            wall(f"Atrium{index:02d}", start, end)

    wall("EastHallNorth", (vertices[0][0], 1.40), (22.00, 1.40))
    wall("EastHallSouthWest", (vertices[7][0], -1.40), (15.90, -1.40))
    wall("EastHallSouthEast", (18.30, -1.40), (22.00, -1.40))
    split_wall("EastHallEnd", (22.00, -1.40), (22.00, 1.40), (22.00, 0.0), 1.80)

    vp_door = config["doors"]["vice_principal"]
    vp_width = float(vp_door["clear_width_m"])
    training_padding = float(
        config.get("presentation_clearance_profile", {}).get(
            "footprint_padding_per_side_m", 0.0
        )
    )
    # Author the measured physical aperture.  Footprint padding belongs to the
    # route-clearance acceptance test, not the wall collider: shrinking the
    # world by the padding and then driving the full-size robot through it
    # double-counts the margin (and makes PhysX contact offsets dominate this
    # unusually tight presentation stress case).
    vp_training_width = vp_width
    vp_padded_centre_band = vp_width - (
        float(config["presentation_clearance_profile"]["physical_body_width_m"])
        + 2.0 * training_padding
    )
    if vp_padded_centre_band <= 0.0:
        raise ValueError("Vice-Principal aperture cannot contain the padded robot footprint")
    wall("ViceAccessWest", (15.90, -5.05), (15.90, -1.40))
    wall("ViceAccessEast", (18.30, -5.05), (18.30, -1.40))
    wall(
        "ViceNorthWest",
        (13.95, -5.05),
        (17.10 - vp_training_width / 2.0, -5.05),
    )
    wall(
        "ViceNorthEast",
        (17.10 + vp_training_width / 2.0, -5.05),
        (20.25, -5.05),
    )
    wall("ViceSouth", (13.95, -8.05), (20.25, -8.05))
    wall("ViceWest", (13.95, -8.05), (13.95, -5.05))
    wall("ViceEast", (20.25, -8.05), (20.25, -5.05))

    corridor_start, corridor_end = (3.45, -3.45), (7.00, -7.00)
    normal = (math.sqrt(0.5), math.sqrt(0.5))
    for side_name, side_sign in (("North", 1.0), ("South", -1.0)):
        offset = (normal[0] * 1.30 * side_sign, normal[1] * 1.30 * side_sign)
        wall(
            "PrincipalAccess" + side_name,
            (corridor_start[0] + offset[0], corridor_start[1] + offset[1]),
            (corridor_end[0] + offset[0], corridor_end[1] + offset[1]),
        )

    principal_centre, principal_size, principal_yaw = (8.65, -9.30), (4.75, 3.60), -45.0
    corners = [
        local_to_world(principal_centre, (-principal_size[0] / 2.0, -principal_size[1] / 2.0), principal_yaw),
        local_to_world(principal_centre, (principal_size[0] / 2.0, -principal_size[1] / 2.0), principal_yaw),
        local_to_world(principal_centre, (principal_size[0] / 2.0, principal_size[1] / 2.0), principal_yaw),
        local_to_world(principal_centre, (-principal_size[0] / 2.0, principal_size[1] / 2.0), principal_yaw),
    ]
    principal_door = config["doors"]["principal"]
    principal_door_centre = local_to_world(principal_centre, (-principal_size[0] / 2.0, 0.0), principal_yaw)
    principal_training_width = float(principal_door["clear_width_m"])
    principal_padded_centre_band = principal_training_width - (
        float(config["presentation_clearance_profile"]["physical_body_width_m"])
        + 2.0 * training_padding
    )
    if principal_padded_centre_band <= 0.0:
        raise ValueError("Principal aperture cannot contain the padded robot footprint")
    split_wall(
        "PrincipalWest",
        corners[0],
        corners[3],
        principal_door_centre,
        principal_training_width,
    )
    wall("PrincipalSouth", corners[0], corners[1])
    wall("PrincipalEast", corners[1], corners[2])
    wall("PrincipalNorth", corners[2], corners[3])

    # Major furniture remains in the collision/sensor course; decorative slats,
    # lighting and ceilings are intentionally omitted from parallel training.
    box("Reception", (4.20, 0.78, 1.08), (-1.10, 3.45, 0.54), furniture_material)
    box("AtriumBench", (2.60, 0.70, 0.70), (-1.00, -3.50, 0.46), furniture_material)
    # Match the presentation scene's side-positioned round meeting table with
    # a conservative square proxy. The previous centreline desk was not in the
    # walkthrough-derived scene and made the declared in-room pivot impossible.
    box("ViceMeetingTableProxy", (1.52, 1.52, 0.76), (15.45, -6.62, 0.38), furniture_material)
    principal_desk = local_to_world(principal_centre, (0.95, -0.95), principal_yaw)
    box("PrincipalDesk", (2.00, 0.82, 0.82), (*principal_desk, 0.41), furniture_material, principal_yaw)

    for door_name, door, yaw in (
        ("Vice", vp_door, 0.0),
        ("Principal", principal_door, 45.0),
    ):
        centre = tuple(float(value) for value in door["centre_xy_m"])
        threshold_m = float(door["threshold_height_mm"]) / 1000.0
        if threshold_m <= 0.0:
            continue
        box(
            door_name + "Threshold",
            (float(door["clear_width_m"]), 0.12, threshold_m),
            (centre[0], centre[1], threshold_m / 2.0),
            threshold_material,
            yaw,
        )

    course.GetPrim().SetCustomDataByKey("aisha:sourcePlanPage", 2)
    geometry_status = (
        "measured_site_presentation_candidate_training_proxy"
        if measured_overlay is not None
        else "plan_derived_training_proxy"
    )
    door_width_status = (
        "user_reported_administration_minimum_conservative_destination_assumption"
        if measured_overlay is not None
        else "presentation_assumption_not_measured"
    )
    course.GetPrim().SetCustomDataByKey("aisha:geometryStatus", geometry_status)
    course.GetPrim().SetCustomDataByKey("aisha:doorWidthsStatus", door_width_status)
    course.GetPrim().SetCustomDataByKey("aisha:centralAtriumStepDownM", float(central_polygon["step_down_m"]))
    course.GetPrim().SetCustomDataByKey("aisha:centralAtriumRobotAccess", "prohibited")
    stage.GetRootLayer().customLayerData = {
        "aisha:purpose": "isaac_lab_parallel_sensor_training",
        "aisha:physicalRelease": False,
        "aisha:planSha256": config["provenance"]["plan_source"]["sha256"],
    }
    stage.GetRootLayer().Save()

    report = {
        "report_type": "block_a_training_course_build",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "output_sha256": _sha256(output),
        "source_config": str(CONFIG),
        "source_config_sha256": _sha256(CONFIG),
        "source_overlay": (
            str(args.measured_geometry.expanduser().resolve())
            if measured_overlay is not None
            else None
        ),
        "source_overlay_sha256": (
            _sha256(args.measured_geometry.expanduser().resolve())
            if measured_overlay is not None
            else None
        ),
        "source_plan_page": 2,
        "source_plan_sha256": config["provenance"]["plan_source"]["sha256"],
        "authored_static_colliders": len(authored_colliders),
        "course_reopens": Usd.Stage.Open(str(output)) is not None,
        "geometry_status": geometry_status,
        "door_width_status": door_width_status,
        "door_clear_widths_m": {
            name: float(values["clear_width_m"])
            for name, values in config["doors"].items()
        },
        "effective_training_apertures_m": {
            "vice_principal": vp_training_width,
            "principal": principal_training_width,
        },
        "padded_route_acceptance_centre_bands_m": {
            "vice_principal": vp_padded_centre_band,
            "principal": principal_padded_centre_band,
        },
        "padding_enforcement": "trajectory_acceptance_not_collision_geometry",
        "footprint_padding_per_side_m": training_padding,
        "central_atrium_drop": {
            "step_down_m": float(central_polygon["step_down_m"]),
            "robot_access": central_polygon["robot_access"],
            "lidar_collision_proxy_segments": 8,
        },
        "physical_release": False,
        "course_contact_model": {
            "contact_offset_m": 0.005,
            "rest_offset_m": 0.0,
            "purpose": "avoid solver-inflated measured doorway geometry",
        },
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"COURSE={output}")
    print(f"COURSE_REPORT={report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
