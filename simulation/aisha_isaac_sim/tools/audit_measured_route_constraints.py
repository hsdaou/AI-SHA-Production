#!/usr/bin/env python3
"""Audit a recorded administration trace against measured presentation constraints.

This is a deterministic geometry gate. It checks the simulation-only padded
robot footprint against the central atrium no-go polygon and against the clear
apertures of both office doors. It does not certify physical operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "administration_assumptions.yaml"
DEFAULT_OVERLAY = (
    PACKAGE_ROOT / "config" / "measured_administration_presentation_2026-08-23.yaml"
)
DEFAULT_TRACE = PACKAGE_ROOT / "results" / "phase3n_administration_final_omniverse_report.json"
DEFAULT_REPORT = PACKAGE_ROOT / "results" / "measured_route_constraint_audit.json"
DOOR_ROUTE_SEGMENTS = {
    "vice_principal": {3, 4},
    "principal": {8, 9},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shortest_angle_delta(start: float, end: float) -> float:
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_trace(trace: list[dict], spacing_m: float, yaw_step_rad: float) -> list[dict]:
    if not trace:
        return []
    dense: list[dict] = []
    for first, second in zip(trace, trace[1:]):
        if int(second["segment_id"]) < int(first["segment_id"]):
            continue
        dx = float(second["x_m"]) - float(first["x_m"])
        dy = float(second["y_m"]) - float(first["y_m"])
        dyaw = shortest_angle_delta(float(first["yaw_rad"]), float(second["yaw_rad"]))
        parts = max(1, math.ceil(math.hypot(dx, dy) / spacing_m), math.ceil(abs(dyaw) / yaw_step_rad))
        for part in range(parts):
            alpha = part / parts
            dense.append(
                {
                    "step": int(first["step"]),
                    "segment_id": int(first["segment_id"]),
                    "x_m": float(first["x_m"]) + dx * alpha,
                    "y_m": float(first["y_m"]) + dy * alpha,
                    "yaw_rad": float(first["yaw_rad"]) + dyaw * alpha,
                }
            )
    dense.append(
        {
            "step": int(trace[-1]["step"]),
            "segment_id": int(trace[-1]["segment_id"]),
            "x_m": float(trace[-1]["x_m"]),
            "y_m": float(trace[-1]["y_m"]),
            "yaw_rad": float(trace[-1]["yaw_rad"]),
        }
    )
    return dense


def robot_polygon(sample: dict, length_m: float, width_m: float) -> list[tuple[float, float]]:
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    yaw = float(sample["yaw_rad"])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    result = []
    for local_x, local_y in (
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    ):
        result.append(
            (
                float(sample["x_m"]) + local_x * cosine - local_y * sine,
                float(sample["y_m"]) + local_x * sine + local_y * cosine,
            )
        )
    return result


def polygon_axes(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    axes = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        edge_x, edge_y = second[0] - first[0], second[1] - first[1]
        length = math.hypot(edge_x, edge_y)
        axes.append((-edge_y / length, edge_x / length))
    return axes


def polygons_intersect(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> bool:
    for axis in polygon_axes(first) + polygon_axes(second):
        first_projection = [point[0] * axis[0] + point[1] * axis[1] for point in first]
        second_projection = [point[0] * axis[0] + point[1] * axis[1] for point in second]
        if max(first_projection) < min(second_projection) or max(second_projection) < min(first_projection):
            return False
    return True


def regular_polygon(centre: tuple[float, float], radius: float, orientation_deg: float) -> list[tuple[float, float]]:
    return [
        (
            centre[0] + radius * math.cos(math.radians(orientation_deg + 45.0 * index)),
            centre[1] + radius * math.sin(math.radians(orientation_deg + 45.0 * index)),
        )
        for index in range(8)
    ]


def door_aperture_clearance(
    footprint: list[tuple[float, float]], door: dict
) -> float | None:
    centre_x, centre_y = (float(value) for value in door["centre_xy_m"])
    angle = math.radians(float(door["wall_rotation_deg"]))
    tangent = (math.cos(angle), math.sin(angle))
    normal = (-math.sin(angle), math.cos(angle))
    tangent_values = []
    normal_values = []
    for x, y in footprint:
        relative = (x - centre_x, y - centre_y)
        tangent_values.append(relative[0] * tangent[0] + relative[1] * tangent[1])
        normal_values.append(relative[0] * normal[0] + relative[1] * normal[1])
    frame_depth = float(door.get("frame_depth_m", 0.0))
    if max(normal_values) < -frame_depth / 2.0 or min(normal_values) > frame_depth / 2.0:
        return None
    half_clear_width = float(door["clear_width_m"]) / 2.0
    return min(half_clear_width - max(tangent_values), min(tangent_values) + half_clear_width)


def audit(config: dict, overlay: dict, trace_report: dict) -> dict:
    profile = overlay["presentation_clearance_profile"]
    padding = float(profile["footprint_padding_per_side_m"])
    robot_width = float(profile["physical_body_width_m"]) + 2.0 * padding
    robot_length = float(config["presentation_release"]["robot_transit_length_m"]) + 2.0 * padding
    completed_step = int(trace_report["completed_steps"])
    raw_trace = [
        sample
        for sample in trace_report.get("pose_trace", [])
        if int(sample["step"]) < completed_step
    ]
    dense_trace = interpolate_trace(raw_trace, spacing_m=0.01, yaw_step_rad=math.radians(1.0))

    polygon = config["plan_geometry"]["atrium"]["central_polygon"]
    no_go_polygon = regular_polygon(
        tuple(float(value) for value in polygon["centre_xy_m"]),
        float(polygon["outer_vertex_radius_m"]),
        float(polygon["orientation_deg"]),
    )

    no_go_violations = []
    door_results = {
        name: {"encounters": 0, "minimum_padded_clearance_m": None, "violations": []}
        for name in overlay["doors"]
    }
    for sample in dense_trace:
        footprint = robot_polygon(sample, robot_length, robot_width)
        if polygons_intersect(footprint, no_go_polygon):
            if len(no_go_violations) < 50:
                no_go_violations.append(
                    {
                        "step": sample["step"],
                        "segment_id": sample["segment_id"],
                        "xy_m": [round(sample["x_m"], 4), round(sample["y_m"], 4)],
                    }
                )
        for name, overlay_door in overlay["doors"].items():
            if int(sample["segment_id"]) not in DOOR_ROUTE_SEGMENTS[name]:
                continue
            door = {**config["doors"][name], **overlay_door}
            clearance = door_aperture_clearance(footprint, door)
            if clearance is None:
                continue
            result = door_results[name]
            result["encounters"] += 1
            prior = result["minimum_padded_clearance_m"]
            result["minimum_padded_clearance_m"] = clearance if prior is None else min(prior, clearance)
            if clearance < -1.0e-6 and len(result["violations"]) < 50:
                result["violations"].append(
                    {
                        "step": sample["step"],
                        "segment_id": sample["segment_id"],
                        "xy_m": [round(sample["x_m"], 4), round(sample["y_m"], 4)],
                        "yaw_deg": round(math.degrees(sample["yaw_rad"]), 3),
                        "padded_clearance_m": round(clearance, 5),
                    }
                )

    for result in door_results.values():
        if result["minimum_padded_clearance_m"] is not None:
            result["minimum_padded_clearance_m"] = round(
                result["minimum_padded_clearance_m"], 5
            )
        result["passed"] = result["encounters"] > 0 and not result["violations"]

    checks = {
        "trace_completed_all_twelve_segments": trace_report.get("outcome") == "success"
        and trace_report.get("waypoints_completed") == 12,
        "trace_contains_world_frame_samples": bool(raw_trace),
        "central_atrium_no_go_respected": not no_go_violations,
        "vice_principal_85cm_aperture_respected": door_results["vice_principal"]["passed"],
        "principal_85cm_aperture_respected": door_results["principal"]["passed"],
        "physical_release_disabled": overlay.get("physical_release") is False,
    }
    passed = all(checks.values())
    return {
        "status": "measured_route_constraints_passed" if passed else "former_trace_invalidated_retraining_required",
        "passed": passed,
        "checks": checks,
        "sampling": {
            "source_records": len(raw_trace),
            "interpolated_records": len(dense_trace),
            "maximum_translation_spacing_m": 0.01,
            "maximum_yaw_spacing_deg": 1.0,
        },
        "padded_footprint": {
            "length_m": robot_length,
            "width_m": robot_width,
            "padding_per_side_m": padding,
        },
        "central_atrium_no_go": {
            "step_down_m": float(polygon["step_down_m"]),
            "radius_m": float(polygon["outer_vertex_radius_m"]),
            "violation_count_capped": len(no_go_violations),
            "first_violations": no_go_violations,
        },
        "doors": door_results,
        "physical_release": False,
        "claim_boundary": (
            "Deterministic simulation-geometry audit only. Passing does not establish real-world "
            "threshold, localization, stopping-distance, or physical safety performance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    overlay = yaml.safe_load(args.overlay.read_text(encoding="utf-8"))
    trace_report = json.loads(args.trace.read_text(encoding="utf-8"))
    result = audit(config, overlay, trace_report)
    result.update(
        {
            "report_type": "measured_administration_route_constraint_audit",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "config": str(args.config.resolve()),
                "config_sha256": sha256(args.config),
                "overlay": str(args.overlay.resolve()),
                "overlay_sha256": sha256(args.overlay),
                "trace": str(args.trace.resolve()),
                "trace_sha256": sha256(args.trace),
            },
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "AISHA_MEASURED_ROUTE_AUDIT "
        f"passed={result['passed']} status={result['status']} report={args.report}"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
