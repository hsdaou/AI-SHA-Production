#!/usr/bin/env python3
"""Validate measured-site capture inputs and prepare a scene-geometry overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ALLOWED_SCAN_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf", ".obj", ".ply"}
PHYSICAL_TRANSIT_WIDTH_M = 0.768
NAV2_FOOTPRINT_PADDING_M = 0.08
PADDED_TRANSIT_WIDTH_M = PHYSICAL_TRANSIT_WIDTH_M + 2.0 * NAV2_FOOTPRINT_PADDING_M
PIVOT_CLEAR_DIAMETER_M = 1.640


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(finite_number(item) for item in value)


def inspect_obj(path: Path) -> dict[str, Any]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    vertices = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            point = [float(fields[index]) for index in range(1, 4)]
            for axis, coordinate in enumerate(point):
                minimum[axis] = min(minimum[axis], coordinate)
                maximum[axis] = max(maximum[axis], coordinate)
            vertices += 1
    if vertices == 0:
        raise ValueError("OBJ contains no vertices")
    return {"format": "obj", "vertices": vertices, "bounds_native": [minimum, maximum]}


def inspect_ply(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(65536)
    marker = header.find(b"end_header")
    if not header.startswith(b"ply\n") or marker < 0:
        raise ValueError("invalid PLY header")
    text = header[: marker + len(b"end_header")].decode("ascii", errors="replace")
    vertex_count = 0
    encoding = None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ["format"] and len(fields) >= 2:
            encoding = fields[1]
        if fields[:2] == ["element", "vertex"] and len(fields) == 3:
            vertex_count = int(fields[2])
    if vertex_count <= 0 or encoding is None:
        raise ValueError("PLY has no declared vertices or encoding")
    return {"format": "ply", "vertices": vertex_count, "encoding": encoding}


def inspect_gltf_json(data: dict[str, Any], format_name: str) -> dict[str, Any]:
    asset = data.get("asset", {})
    if not str(asset.get("version", "")).startswith("2"):
        raise ValueError("only glTF 2.x captures are supported")
    accessors = data.get("accessors", [])
    position_bounds = []
    for accessor in accessors:
        if accessor.get("type") == "VEC3" and "min" in accessor and "max" in accessor:
            if finite_vector(accessor["min"], 3) and finite_vector(accessor["max"], 3):
                position_bounds.append([accessor["min"], accessor["max"]])
    return {
        "format": format_name,
        "gltf_version": str(asset.get("version")),
        "meshes": len(data.get("meshes", [])),
        "nodes": len(data.get("nodes", [])),
        "declared_accessor_bounds": position_bounds,
    }


def inspect_glb(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("truncated GLB header")
        magic, version, total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or total_length != path.stat().st_size:
            raise ValueError("invalid GLB 2.0 header")
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            raise ValueError("GLB has no JSON chunk")
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:
            raise ValueError("GLB first chunk is not JSON")
        data = json.loads(stream.read(chunk_length).rstrip(b"\x00 \t\r\n"))
    return inspect_gltf_json(data, "glb")


def inspect_usdz(path: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise ValueError("USDZ is not a valid ZIP container")
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        usd_members = [name for name in members if Path(name).suffix.lower() in {".usd", ".usda", ".usdc"}]
    if not usd_members:
        raise ValueError("USDZ contains no USD layer")
    return {"format": "usdz", "members": len(members), "usd_layers": usd_members}


def inspect_scan(path: Path) -> dict[str, Any]:
    extension = path.suffix.lower()
    if extension not in ALLOWED_SCAN_EXTENSIONS:
        raise ValueError(f"unsupported scan extension {extension}")
    if path.stat().st_size <= 0:
        raise ValueError("scan file is empty")
    if extension == ".obj":
        details = inspect_obj(path)
    elif extension == ".ply":
        details = inspect_ply(path)
    elif extension == ".gltf":
        details = inspect_gltf_json(json.loads(path.read_text(encoding="utf-8")), "gltf")
    elif extension == ".glb":
        details = inspect_glb(path)
    elif extension == ".usdz":
        details = inspect_usdz(path)
    elif extension == ".usda":
        if not path.read_bytes()[:16].lstrip().startswith(b"#usda"):
            raise ValueError("USDA file has no #usda header")
        details = {"format": "usda"}
    else:
        header = path.read_bytes()[:16]
        if not (header.startswith(b"PXR-USDC") or header.lstrip().startswith(b"#usda")):
            raise ValueError("USD/USDC file header is not recognized")
        details = {"format": extension.removeprefix(".")}
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **details,
    }


def resolve_optional(scan_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = scan_root / path
    return path.resolve()


def require_measurement(measurements: dict[str, Any], dotted: str) -> Any:
    value: Any = measurements
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def measurement_checks(measurements: dict[str, Any]) -> dict[str, bool]:
    scalar_fields = (
        "east_hallway_clear_width_m",
        "principal_passage_clear_width_m",
        "ceiling_height_m",
        "vice_principal_door.clear_width_m",
        "vice_principal_door.clear_height_m",
        "vice_principal_door.frame_depth_m",
        "vice_principal_door.threshold_hallway_mm",
        "vice_principal_door.threshold_office_mm",
        "principal_door.clear_width_m",
        "principal_door.clear_height_m",
        "principal_door.frame_depth_m",
        "principal_door.threshold_hallway_mm",
        "principal_door.threshold_office_mm",
    )
    checks = {}
    for field in scalar_fields:
        value = require_measurement(measurements, field)
        valid = finite_number(value)
        if valid and "threshold_" in field:
            valid = float(value) >= 0.0
        elif valid:
            valid = float(value) > 0.0
        checks[field.replace(".", "_")] = valid
    checks["vice_principal_turn_zone_size_m"] = finite_vector(
        measurements.get("vice_principal_turn_zone_size_m"), 2
    ) and all(float(value) > 0.0 for value in measurements["vice_principal_turn_zone_size_m"])
    checks["principal_turn_zone_size_m"] = finite_vector(
        measurements.get("principal_turn_zone_size_m"), 2
    ) and all(float(value) > 0.0 for value in measurements["principal_turn_zone_size_m"])
    for door_name in ("vice_principal_door", "principal_door"):
        door = measurements.get(door_name, {})
        checks[f"{door_name}_threshold_profile"] = bool(door.get("threshold_profile"))
        checks[f"{door_name}_hinge_side"] = door.get("hinge_side_from_hallway") in {"left", "right"}
        checks[f"{door_name}_swing"] = door.get("swing_from_hallway") in {"inward", "outward"}
        checks[f"{door_name}_centre_xy_m"] = finite_vector(door.get("centre_xy_m"), 2)
        checks[f"{door_name}_wall_rotation_deg"] = finite_number(door.get("wall_rotation_deg"))
    return checks


def transform_checks(transform: Any) -> dict[str, bool]:
    if not isinstance(transform, dict):
        return {"scale_xyz": False, "rotate_xyz_deg": False, "translate_xyz_m": False}
    return {
        "scale_xyz": finite_vector(transform.get("scale_xyz"), 3)
        and all(float(value) > 0.0 for value in transform["scale_xyz"]),
        "rotate_xyz_deg": finite_vector(transform.get("rotate_xyz_deg"), 3),
        "translate_xyz_m": finite_vector(transform.get("translate_xyz_m"), 3),
    }


def build_overlay(manifest: dict[str, Any], scan_records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = manifest["manual_measurements"]
    vp = measured["vice_principal_door"]
    principal = measured["principal_door"]
    columns = measured.get("atrium_columns", {})
    optional = manifest.get("optional_measured_geometry", {})
    overlay: dict[str, Any] = {
        "overlay_type": "measured_administration_geometry",
        "schema_version": 1,
        "status": "measured_site_candidate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_capture_hashes": [record["sha256"] for record in scan_records],
        "scan_assets": scan_records,
        "known_dimensions": {
            "hallway_clear_width_m": {
                "value": float(measured["east_hallway_clear_width_m"]),
                "status": "manual_site_measurement",
            },
            "principal_passage_clear_width_m": {
                "value": float(measured["principal_passage_clear_width_m"]),
                "status": "manual_site_measurement",
            },
        },
        "plan_geometry": {
            "ceiling_height_m": {
                "value": float(measured["ceiling_height_m"]),
                "status": "manual_site_measurement",
            },
            "wall_height_m": {
                "value": float(measured["ceiling_height_m"]),
                "status": "manual_site_measurement_to_suspended_ceiling",
            },
            "east_hallway": {
                "y_range_m": [
                    -float(measured["east_hallway_clear_width_m"]) / 2.0,
                    float(measured["east_hallway_clear_width_m"]) / 2.0,
                ]
            },
        },
        "doors": {},
        "measurement_evidence": {
            "principal_passage_clear_width_m": float(measured["principal_passage_clear_width_m"]),
            "vice_principal_turn_zone_size_m": [float(v) for v in measured["vice_principal_turn_zone_size_m"]],
            "principal_turn_zone_size_m": [float(v) for v in measured["principal_turn_zone_size_m"]],
            "physical_release": False,
        },
    }
    for output_name, source in (("vice_principal", vp), ("principal", principal)):
        threshold = max(float(source["threshold_hallway_mm"]), float(source["threshold_office_mm"]))
        overlay["doors"][output_name] = {
            "centre_xy_m": [float(value) for value in source["centre_xy_m"]],
            "wall_rotation_deg": float(source["wall_rotation_deg"]),
            "clear_width_m": float(source["clear_width_m"]),
            "clear_height_m": float(source["clear_height_m"]),
            "frame_depth_m": float(source["frame_depth_m"]),
            "width_status": "manual_site_measurement",
            "threshold_height_mm": threshold,
            "threshold_status": "manual_site_measurement",
            "threshold_profile": str(source["threshold_profile"]),
            "hinge_side_from_hallway": str(source["hinge_side_from_hallway"]),
            "swing_from_hallway": str(source["swing_from_hallway"]),
        }
    if finite_number(columns.get("radius_m")) and all(
        finite_vector(point, 2) for point in columns.get("positions_xy_m", [])
    ):
        overlay.setdefault("appearance", {})["atrium_columns"] = {
            "status": "manual_site_measurement",
            "radius_m": float(columns["radius_m"]),
            "height_m": float(measured["ceiling_height_m"]),
            "positions_xy_m": [[float(v) for v in point] for point in columns["positions_xy_m"]],
        }
    room_mapping = {
        "vice_principal_room": "vice_principal",
        "principal_room": "principal",
    }
    for source_name, target_name in room_mapping.items():
        room = optional.get(source_name, {})
        if (
            finite_vector(room.get("centre_xy_m"), 2)
            and finite_vector(room.get("size_xy_m"), 2)
            and all(float(value) > 0.0 for value in room["size_xy_m"])
            and finite_number(room.get("rotation_deg"))
        ):
            overlay["plan_geometry"].setdefault("south_east_cluster", {})[target_name] = {
                "centre_xy_m": [float(v) for v in room["centre_xy_m"]],
                "size_xy_m": [float(v) for v in room["size_xy_m"]],
                "rotation_deg": float(room["rotation_deg"]),
                "status": "site_scan_plus_manual_measurement_candidate",
            }
    waypoints = optional.get("route_waypoints", [])
    if waypoints:
        overlay["route"] = {"status": "measured_site_candidate", "waypoints": waypoints}
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    scan_root = args.scan_root.expanduser().resolve()
    schema_checks = {
        "schema_version_1": manifest.get("schema_version") == 1,
        "coordinate_contract_present": isinstance(manifest.get("coordinate_contract"), dict),
        "scan_contract_present": isinstance(manifest.get("scans"), dict),
        "manual_measurement_contract_present": isinstance(manifest.get("manual_measurements"), dict),
        "safety_boundary_explicit": manifest.get("safety_boundary", {}).get(
            "generated_overlay_never_releases_physical_operation"
        )
        is True,
    }
    coordinate = manifest.get("coordinate_contract", {})
    coordinate_checks = {
        "source_units_declared": coordinate.get("source_units") in {"metre", "millimetre"},
        "source_up_axis_declared": coordinate.get("source_up_axis") in {"X", "Y", "Z"},
        "target_is_metre_z_up": coordinate.get("target_units") == "metre"
        and coordinate.get("target_up_axis") == "Z",
        "target_origin_declared": bool(coordinate.get("target_origin")),
    }
    capture = manifest.get("capture", {})
    capture_checks = {
        "capture_date_recorded": bool(capture.get("date")),
        "operator_recorded": bool(capture.get("operator")),
        "device_model_recorded": bool(capture.get("device_model")),
        "application_recorded": bool(capture.get("application")),
        "application_version_recorded": bool(capture.get("application_version")),
        "privacy_reviewed": capture.get("privacy_reviewed") is True,
        "people_and_sensitive_documents_excluded": capture.get(
            "people_and_sensitive_documents_excluded"
        )
        is True,
        "source_metadata_verified_in_export": coordinate.get(
            "source_metadata_verified_in_export"
        )
        is True,
    }

    scan_records: list[dict[str, Any]] = []
    scan_errors: list[str] = []
    complete = manifest.get("scans", {}).get("complete_structure", {})
    complete_path = resolve_optional(scan_root, complete.get("file"))
    complete_valid = False
    if complete_path is not None:
        try:
            record = inspect_scan(complete_path)
            record["id"] = "complete_structure"
            record["transform"] = complete.get("transform")
            transform_valid = all(transform_checks(complete.get("transform")).values())
            if not transform_valid:
                raise ValueError("complete_structure has an incomplete source-to-stage transform")
            scan_records.append(record)
            complete_valid = True
        except (OSError, ValueError, json.JSONDecodeError) as error:
            scan_errors.append(f"complete_structure: {error}")

    required_sections = []
    supplied_required_sections = []
    for section in manifest.get("scans", {}).get("sections", []):
        if section.get("required"):
            required_sections.append(str(section.get("id")))
        path = resolve_optional(scan_root, section.get("file"))
        if path is None:
            continue
        try:
            record = inspect_scan(path)
            record["id"] = str(section.get("id"))
            record["transform"] = section.get("transform")
            transform_valid = all(transform_checks(section.get("transform")).values())
            if not transform_valid:
                raise ValueError(f"{section.get('id')} has an incomplete source-to-stage transform")
            scan_records.append(record)
            if section.get("required"):
                supplied_required_sections.append(str(section.get("id")))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            scan_errors.append(f"{section.get('id')}: {error}")
    section_set_complete = set(supplied_required_sections) == set(required_sections)
    scans_complete = (complete_valid or section_set_complete) and not scan_errors

    measure_checks = measurement_checks(manifest.get("manual_measurements", {}))
    measurements_complete = bool(measure_checks) and all(measure_checks.values())
    intake_complete = (
        all(schema_checks.values())
        and all(coordinate_checks.values())
        and all(capture_checks.values())
        and scans_complete
        and measurements_complete
    )

    route_checks: dict[str, bool] = {}
    threshold_validation_required = None
    if measurements_complete:
        measured = manifest["manual_measurements"]
        route_checks = {
            "vice_principal_door_accepts_padded_footprint": float(
                measured["vice_principal_door"]["clear_width_m"]
            )
            >= PADDED_TRANSIT_WIDTH_M,
            "principal_door_accepts_padded_footprint": float(
                measured["principal_door"]["clear_width_m"]
            )
            >= PADDED_TRANSIT_WIDTH_M,
            "east_hallway_supports_pivot": float(measured["east_hallway_clear_width_m"])
            >= PIVOT_CLEAR_DIAMETER_M,
            "vice_principal_turn_zone_supports_pivot": min(
                float(v) for v in measured["vice_principal_turn_zone_size_m"]
            )
            >= PIVOT_CLEAR_DIAMETER_M,
            "principal_turn_zone_supports_pivot": min(
                float(v) for v in measured["principal_turn_zone_size_m"]
            )
            >= PIVOT_CLEAR_DIAMETER_M,
        }
        threshold_validation_required = any(
            float(measured[door][side]) > 0.0
            for door in ("vice_principal_door", "principal_door")
            for side in ("threshold_hallway_mm", "threshold_office_mm")
        )
    candidate_geometry_valid = intake_complete and bool(route_checks) and all(route_checks.values())

    overlay_written = False
    if intake_complete and args.overlay_output is not None:
        overlay = build_overlay(manifest, scan_records)
        overlay["candidate_route_geometry_checks"] = route_checks
        overlay["candidate_route_geometry_valid"] = candidate_geometry_valid
        overlay["threshold_contact_validation_required"] = threshold_validation_required
        overlay["physical_release"] = False
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        args.overlay_output.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
        overlay_written = True

    report = {
        "report_type": "measured_administration_capture_intake",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "candidate_geometry_ready"
            if candidate_geometry_valid
            else (
                "capture_complete_route_geometry_rejected"
                if intake_complete
                else "awaiting_capture_or_measurements"
            )
        ),
        "passed": candidate_geometry_valid,
        "pipeline_preflight_passed": all(schema_checks.values()) and all(coordinate_checks.values()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "scan_root": str(scan_root),
        "schema_checks": schema_checks,
        "coordinate_checks": coordinate_checks,
        "capture_gate": {
            "checks": capture_checks,
            "checks_passed": sum(capture_checks.values()),
            "checks_total": len(capture_checks),
            "passed": all(capture_checks.values()),
        },
        "scan_gate": {
            "complete_structure_valid": complete_valid,
            "required_sections": required_sections,
            "supplied_required_sections": supplied_required_sections,
            "section_set_complete": section_set_complete,
            "passed": scans_complete,
            "errors": scan_errors,
            "artifacts": scan_records,
        },
        "measurement_gate": {
            "checks": measure_checks,
            "checks_passed": sum(measure_checks.values()),
            "checks_total": len(measure_checks),
            "passed": measurements_complete,
        },
        "route_geometry_gate": {
            "physical_transit_width_m": PHYSICAL_TRANSIT_WIDTH_M,
            "nav2_footprint_padding_m": NAV2_FOOTPRINT_PADDING_M,
            "padded_transit_width_m": PADDED_TRANSIT_WIDTH_M,
            "pivot_clear_diameter_m": PIVOT_CLEAR_DIAMETER_M,
            "checks": route_checks,
            "candidate_geometry_valid": candidate_geometry_valid,
            "threshold_contact_validation_required": threshold_validation_required,
            "physical_release": False,
        },
        "overlay": {
            "requested": args.overlay_output is not None,
            "written": overlay_written,
            "path": str(args.overlay_output.resolve()) if overlay_written else None,
            "sha256": sha256_file(args.overlay_output) if overlay_written else None,
        },
        "claim_boundary": (
            "A complete intake creates a measured-geometry candidate for simulation. "
            "It does not validate threshold contact, stopping distance, protective "
            "coverage, sim-to-real performance, or physical deployment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if candidate_geometry_valid or args.allow_incomplete:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
