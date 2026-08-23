#!/usr/bin/env python3
"""Rasterize the composed administration USD collision geometry for Nav2.

The resulting map is deliberately marked provisional.  It is generated from
the current plan-derived presentation scene and must be replaced by the
measured-site map before any physical-release claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from pxr import Gf, Usd, UsdGeom, UsdPhysics


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PACKAGE_ROOT / "usd" / "administration_live_environment.usda"
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "administration_assumptions.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "maps" / "administration_provisional"
DEFAULT_REPORT = PACKAGE_ROOT / "results" / "administration_provisional_map_report.json"

WALKABLE_FLOOR_GROUPS = {
    "Atrium",
    "EastHallway",
    "ViceAccess",
    "VicePrincipal",
    "PrincipalAccess",
    "Principal",
}


def walkable_floor_group(path: str, name: str) -> str | None:
    if path.startswith("/World/Architecture/Floors/Atrium/WalkableRing_"):
        return "Atrium"
    if "/Architecture/Floors/" in path and name in WALKABLE_FLOOR_GROUPS:
        return name
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transformed_points(prim: Usd.Prim, samples: int = 48) -> list[tuple[float, float, float]]:
    """Return a conservative world-space outline and vertical extent."""

    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    type_name = prim.GetTypeName()
    local: list[Gf.Vec3d]
    if type_name == "Cube":
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
        half = size / 2.0
        local = [
            Gf.Vec3d(x, y, z)
            for x in (-half, half)
            for y in (-half, half)
            for z in (-half, half)
        ]
    elif type_name == "Cylinder":
        cylinder = UsdGeom.Cylinder(prim)
        radius = float(cylinder.GetRadiusAttr().Get() or 1.0)
        half_height = float(cylinder.GetHeightAttr().Get() or 2.0) / 2.0
        local = [
            Gf.Vec3d(
                radius * math.cos(2.0 * math.pi * index / samples),
                radius * math.sin(2.0 * math.pi * index / samples),
                z,
            )
            for z in (-half_height, half_height)
            for index in range(samples)
        ]
    elif type_name == "Sphere":
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
        local = [
            Gf.Vec3d(
                radius * math.cos(2.0 * math.pi * index / samples),
                radius * math.sin(2.0 * math.pi * index / samples),
                z,
            )
            for z in (-radius, 0.0, radius)
            for index in range(samples)
        ]
    elif type_name == "Mesh":
        local = [Gf.Vec3d(point) for point in (UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])]
    else:
        bound = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeWorldBound(prim).ComputeAlignedBox()
        return [
            (x, y, z)
            for x in (bound.GetMin()[0], bound.GetMax()[0])
            for y in (bound.GetMin()[1], bound.GetMax()[1])
            for z in (bound.GetMin()[2], bound.GetMax()[2])
        ]
    return [tuple(transform.Transform(point)) for point in local]


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(origin, first, second) -> float:
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


def world_to_pixel(
    point: tuple[float, float], origin_x: float, max_y: float, resolution: float
) -> tuple[int, int]:
    return (
        int(round((point[0] - origin_x) / resolution)),
        int(round((max_y - point[1]) / resolution)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--map-name", default="administration_provisional")
    parser.add_argument(
        "--source-overlay",
        type=Path,
        help="measured/presentation overlay used to build the supplied scene",
    )
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--obstacle-min-z", type=float, default=0.10)
    parser.add_argument("--obstacle-max-z", type=float, default=1.25)
    args = parser.parse_args()
    if args.resolution <= 0.0 or args.margin < 0.0:
        parser.error("resolution must be positive and margin must be non-negative")

    scene = args.scene.resolve()
    config_path = args.config.resolve()
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {scene}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_overlay = None
    if args.source_overlay is not None:
        source_overlay = yaml.safe_load(args.source_overlay.read_text(encoding="utf-8"))
        if source_overlay.get("physical_release") is not False:
            raise ValueError("source overlay must preserve physical_release: false")

    floors: list[dict] = []
    obstacles: list[dict] = []
    ignored_collision_prims: list[str] = []
    collision_count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_count += 1
        collision_enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if collision_enabled is False:
            ignored_collision_prims.append(str(prim.GetPath()))
            continue
        points = transformed_points(prim)
        if not points:
            ignored_collision_prims.append(str(prim.GetPath()))
            continue
        path = str(prim.GetPath())
        name = prim.GetName()
        polygon = convex_hull([(float(point[0]), float(point[1])) for point in points])
        floor_group = walkable_floor_group(path, name)
        if floor_group is not None:
            floors.append({"path": path, "polygon": polygon, "group": floor_group})
            continue
        if "/Architecture/Floors/" in path or path.endswith("/SupportSlab"):
            ignored_collision_prims.append(path)
            continue
        minimum_z = min(float(point[2]) for point in points)
        maximum_z = max(float(point[2]) for point in points)
        if maximum_z < args.obstacle_min_z or minimum_z > args.obstacle_max_z:
            ignored_collision_prims.append(path)
            continue
        if len(polygon) >= 3:
            obstacles.append(
                {
                    "path": path,
                    "polygon": polygon,
                    "minimum_z_m": minimum_z,
                    "maximum_z_m": maximum_z,
                }
            )

    found_floor_groups = {item["group"] for item in floors}
    if found_floor_groups != WALKABLE_FLOOR_GROUPS:
        found = sorted(found_floor_groups)
        raise RuntimeError(f"walkable-floor set mismatch: found {found}")

    floor_points = [point for item in floors for point in item["polygon"]]
    min_x = math.floor((min(point[0] for point in floor_points) - args.margin) / args.resolution) * args.resolution
    min_y = math.floor((min(point[1] for point in floor_points) - args.margin) / args.resolution) * args.resolution
    max_x = math.ceil((max(point[0] for point in floor_points) + args.margin) / args.resolution) * args.resolution
    max_y = math.ceil((max(point[1] for point in floor_points) + args.margin) / args.resolution) * args.resolution
    width = int(round((max_x - min_x) / args.resolution)) + 1
    height = int(round((max_y - min_y) / args.resolution)) + 1

    occupancy = Image.new("L", (width, height), 205)
    draw = ImageDraw.Draw(occupancy)
    for floor in floors:
        draw.polygon(
            [world_to_pixel(point, min_x, max_y, args.resolution) for point in floor["polygon"]],
            fill=254,
        )
    for obstacle in obstacles:
        draw.polygon(
            [world_to_pixel(point, min_x, max_y, args.resolution) for point in obstacle["polygon"]],
            fill=0,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.output_dir / f"{args.map_name}.pgm"
    yaml_path = args.output_dir / f"{args.map_name}.yaml"
    preview_path = args.output_dir / f"{args.map_name}_preview.png"
    occupancy.save(map_path)
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "image": map_path.name,
                "mode": "trinary",
                "resolution": args.resolution,
                "origin": [round(min_x, 6), round(min_y, 6), 0.0],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.196,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    route = config["route"]["waypoints"]
    waypoint_checks = []
    all_waypoints_free = True
    for waypoint in route:
        pixel = world_to_pixel(
            (float(waypoint["x_m"]), float(waypoint["y_m"])),
            min_x,
            max_y,
            args.resolution,
        )
        in_bounds = 0 <= pixel[0] < width and 0 <= pixel[1] < height
        value = occupancy.getpixel(pixel) if in_bounds else None
        free = value == 254
        all_waypoints_free &= free
        waypoint_checks.append(
            {
                "id": waypoint["id"],
                "xy_m": [float(waypoint["x_m"]), float(waypoint["y_m"])],
                "pixel": list(pixel),
                "occupancy_value": value,
                "free": free,
            }
        )

    central_polygon = config["plan_geometry"]["atrium"]["central_polygon"]
    central_pixel = world_to_pixel(
        tuple(float(value) for value in central_polygon["centre_xy_m"]),
        min_x,
        max_y,
        args.resolution,
    )
    central_value = occupancy.getpixel(central_pixel)
    central_polygon_is_no_go = central_value != 254

    preview = occupancy.convert("RGB")
    preview_draw = ImageDraw.Draw(preview)
    route_pixels = [tuple(item["pixel"]) for item in waypoint_checks]
    preview_draw.line(route_pixels, fill=(15, 155, 93), width=3)
    for item in waypoint_checks:
        x, y = item["pixel"]
        colour = (0, 210, 125) if item["free"] else (255, 70, 70)
        preview_draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
    preview.save(preview_path)

    report = {
        "report_type": (
            "administration_measured_presentation_occupancy_map"
            if source_overlay is not None
            else "administration_provisional_occupancy_map"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "measured_site_presentation_candidate_map_generated"
            if source_overlay is not None
            else "provisional_plan_derived_map_generated"
        ),
        "source": {
            "scene": str(scene),
            "scene_sha256": sha256(scene),
            "configuration": str(config_path),
            "configuration_sha256": sha256(config_path),
            "geometry_status": (
                source_overlay.get("status")
                if source_overlay is not None
                else config["provenance"]["geometry_status"]
            ),
            "overlay": (
                str(args.source_overlay.resolve()) if args.source_overlay is not None else None
            ),
            "overlay_sha256": (
                sha256(args.source_overlay) if args.source_overlay is not None else None
            ),
        },
        "map": {
            "yaml": str(yaml_path.resolve()),
            "image": str(map_path.resolve()),
            "preview": str(preview_path.resolve()),
            "resolution_m_per_pixel": args.resolution,
            "origin_xy_m": [min_x, min_y],
            "bounds_xy_m": [min_x, min_y, max_x, max_y],
            "dimensions_pixels": [width, height],
            "walkable_floor_primitives": [item["path"] for item in floors],
            "walkable_floor_groups": sorted(found_floor_groups),
            "obstacle_primitives": len(obstacles),
            "collision_primitives_inspected": collision_count,
            "collision_primitives_ignored": len(ignored_collision_prims),
        },
        "route": {
            "waypoints": waypoint_checks,
            "all_waypoint_centres_free": all_waypoints_free,
        },
        "central_atrium_drop": {
            "centre_pixel": list(central_pixel),
            "occupancy_value": central_value,
            "step_down_m": float(central_polygon["step_down_m"]),
            "robot_access": central_polygon["robot_access"],
            "mapped_as_no_go": central_polygon_is_no_go,
        },
        "checks": {
            "all_expected_walkable_floor_groups_present": found_floor_groups
            == WALKABLE_FLOOR_GROUPS,
            "obstacles_rasterized": bool(obstacles),
            "all_route_waypoint_centres_free": all_waypoints_free,
            "central_atrium_drop_is_not_free_space": central_polygon_is_no_go,
            "map_is_explicitly_non_physical": True,
            "physical_release_disabled": True,
        },
        "physical_release": False,
        "replacement_gate": (
            "Complete native-scan-to-plan registration, identify both destination doors, measure "
            "thresholds, and rerun the full mission, contact, and stopping validation before any "
            "physical-release claim."
        ),
    }
    report["passed"] = all(report["checks"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PROVISIONAL_MAP passed={report['passed']} size={width}x{height} "
        f"obstacles={len(obstacles)} map={yaml_path} report={args.report}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
