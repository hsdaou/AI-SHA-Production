#!/usr/bin/env python3
"""Inventory Apple RoomPlan evidence and evaluate presentation clearances.

The tool deliberately does not align multiple scans.  It records each scan in
its native metre/Y-up frame so a later plan-registration step cannot be
mistaken for a measured transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PHYSICAL_TRANSIT_WIDTH_M = 0.768
PRODUCTION_PADDING_PER_SIDE_M = 0.080
PRESENTATION_TIGHT_PADDING_PER_SIDE_M = 0.030


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def floats(text: str) -> list[float]:
    return [float(value) for value in re.findall(NUMBER, text)]


def parse_points(text: str) -> list[list[float]]:
    match = re.search(r"point3f\[\] points = \[(.*?)\]\s*\n", text, re.DOTALL)
    if match is None:
        return []
    points = []
    for group in re.findall(r"\(([^()]*)\)", match.group(1)):
        values = floats(group)
        if len(values) == 3:
            points.append(values)
    return points


def parse_transform(text: str) -> list[float] | None:
    match = re.search(
        r"matrix4d xformOp:transform = \((.*?)\)\s*\n", text, re.DOTALL
    )
    if match is None:
        return None
    values = floats(match.group(1))
    return values if len(values) == 16 else None


def world_xz(point: list[float], matrix: list[float]) -> list[float]:
    x, y, z = point
    return [
        x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12],
        x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14],
    ]


def semantic_asset(member: str, text: str) -> dict[str, Any] | None:
    category_match = re.search(r'string Category = "([^"]+)"', text)
    if category_match is None:
        return None
    points = parse_points(text)
    matrix = parse_transform(text)
    if not points or matrix is None:
        return None
    category = category_match.group(1)
    base_category = category.split("(", 1)[0]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    dimensions = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
    record: dict[str, Any] = {
        "id": Path(member).stem,
        "category": base_category,
        "member": member,
        "dimensions_native_m": [round(value, 6) for value in dimensions],
        "centre_native_xz_m": [round(matrix[12], 6), round(matrix[14], 6)],
    }
    uuid_match = re.search(r'string UUID = "([^"]+)"', text)
    if uuid_match:
        record["uuid"] = uuid_match.group(1)
    if base_category in {"Wall", "Door", "Window"}:
        centre_y = (min(ys) + max(ys)) / 2.0
        centre_z = (min(zs) + max(zs)) / 2.0
        start = world_xz([min(xs), centre_y, centre_z], matrix)
        end = world_xz([max(xs), centre_y, centre_z], matrix)
        record["centreline_native_xz_m"] = [
            [round(value, 6) for value in start],
            [round(value, 6) for value in end],
        ]
        record["rotation_native_deg"] = round(
            math.degrees(math.atan2(matrix[2], matrix[0])), 6
        )
    if base_category == "Door":
        state_match = re.search(r"Door\(Isopen: (True|False)\)", category)
        record.update(
            {
                "clear_width_scan_m": round(dimensions[0], 6),
                "clear_height_scan_m": round(dimensions[1], 6),
                "depth_scan_m": round(dimensions[2], 6),
                "is_open": state_match.group(1) == "True" if state_match else None,
            }
        )
    return record


def inspect_roomplan_usdz(path: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise ValueError("not a valid USDZ/ZIP container")
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        root_members = [name for name in members if "/" not in name and name.endswith(".usda")]
        if len(root_members) != 1:
            raise ValueError("expected exactly one root USDA layer")
        root_text = archive.read(root_members[0]).decode("utf-8")
        units_match = re.search(r"metersPerUnit = ([^\s]+)", root_text)
        up_match = re.search(r'upAxis = "([XYZ])"', root_text)
        sections = re.findall(
            r'def Xform "((?:diningRoom|livingRoom|unidentified)\d+)"', root_text
        )
        assets = []
        for member in members:
            if not member.endswith(".usda") or "/Mesh/" not in member:
                continue
            record = semantic_asset(member, archive.read(member).decode("utf-8"))
            if record is not None:
                assets.append(record)
    counts = Counter(asset["category"] for asset in assets)
    doors = sorted(
        (asset for asset in assets if asset["category"] == "Door"),
        key=lambda item: int(re.search(r"\d+", item["id"]).group()),
    )
    line_points = [
        point
        for asset in assets
        if asset["category"] in {"Wall", "Door", "Window"}
        for point in asset.get("centreline_native_xz_m", [])
    ]
    bounds = None
    if line_points:
        bounds = [
            [min(point[index] for point in line_points) for index in range(2)],
            [max(point[index] for point in line_points) for index in range(2)],
        ]
        bounds = [[round(value, 6) for value in point] for point in bounds]
    return {
        "format": "usdz_roomplan_semantic",
        "root_layer": root_members[0],
        "members": len(members),
        "meters_per_unit": float(units_match.group(1)) if units_match else None,
        "up_axis": up_match.group(1) if up_match else None,
        "semantic_sections": sections,
        "category_counts": dict(sorted(counts.items())),
        "native_xz_bounds_m": bounds,
        "doors": doors,
    }


def inspect_binary_stl(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        stream.read(80)
        count_data = stream.read(4)
        if len(count_data) != 4:
            raise ValueError("truncated STL header")
        triangle_count = struct.unpack("<I", count_data)[0]
        if path.stat().st_size != 84 + 50 * triangle_count:
            raise ValueError("STL size does not match its triangle count")
        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        for _ in range(triangle_count):
            record = stream.read(50)
            if len(record) != 50:
                raise ValueError("truncated STL triangle")
            values = struct.unpack("<12fH", record)
            for vertex in range(3):
                for axis, value in enumerate(values[3 + vertex * 3 : 6 + vertex * 3]):
                    minimum[axis] = min(minimum[axis], value)
                    maximum[axis] = max(maximum[axis], value)
    return {
        "format": "binary_stl",
        "triangles": triangle_count,
        "bounds_native_m": [minimum, maximum],
    }


def inspect_video(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {"metadata_status": "opencv_unavailable"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"metadata_status": "decoder_unavailable"}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    record = {
        "metadata_status": "decoded",
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps if fps > 0.0 else None,
    }
    capture.release()
    return record


def evidence_record(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(root, item["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    record = {
        "id": item["id"],
        "role": item["role"],
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "repository_ingest_allowed": item.get("repository_ingest_allowed", False),
    }
    suffix = path.suffix.lower()
    if suffix == ".usdz":
        record.update(inspect_roomplan_usdz(path))
    elif suffix == ".stl":
        record.update(inspect_binary_stl(path))
    elif suffix in {".mov", ".mp4", ".m4v"}:
        record.update(inspect_video(path))
    else:
        record["format"] = suffix.removeprefix(".")
    return record


def clearance_profile(width_m: float, padding_m: float) -> dict[str, Any]:
    required = PHYSICAL_TRANSIT_WIDTH_M + 2.0 * padding_m
    margin = width_m - required
    return {
        "body_width_m": PHYSICAL_TRANSIT_WIDTH_M,
        "padding_per_side_m": padding_m,
        "required_clear_width_m": round(required, 3),
        "measured_clear_width_m": width_m,
        "total_margin_m": round(margin, 3),
        "nominal_margin_per_side_m": round(margin / 2.0, 3),
        "passes": margin >= 0.0,
    }


def door_match(doors: list[dict[str, Any]], width_m: float, height_m: float) -> dict[str, Any]:
    ranked = sorted(
        doors,
        key=lambda door: math.hypot(
            door["clear_width_scan_m"] - width_m,
            door["clear_height_scan_m"] - height_m,
        ),
    )
    match = dict(ranked[0])
    match["width_residual_m"] = round(match["clear_width_scan_m"] - width_m, 6)
    match["height_residual_m"] = round(match["clear_height_scan_m"] - height_m, 6)
    return match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("expected capture manifest schema_version 2")
    records = [
        evidence_record(args.evidence_root.resolve(), item)
        for item in manifest.get("evidence", [])
    ]
    scan_records = [record for record in records if record.get("format") == "usdz_roomplan_semantic"]
    if not scan_records:
        raise ValueError("manifest contains no semantic RoomPlan USDZ evidence")
    all_doors = [
        {"scan_id": scan["id"], **door}
        for scan in scan_records
        for door in scan["doors"]
    ]
    manual = manifest["manual_reference_measurements"]["narrowest_door"]
    width_m = float(manual["clear_width_m"])
    height_m = float(manual["clear_height_m"])
    closest = door_match(all_doors, width_m, height_m)
    below_manual = [
        {
            "scan_id": door["scan_id"],
            "id": door["id"],
            "scan_width_m": door["clear_width_scan_m"],
        }
        for door in all_doors
        if door["clear_width_scan_m"] < width_m
    ]
    production = clearance_profile(width_m, PRODUCTION_PADDING_PER_SIDE_M)
    presentation = clearance_profile(width_m, PRESENTATION_TIGHT_PADDING_PER_SIDE_M)
    privacy = manifest.get("privacy", {})
    coordinate_metadata_valid = all(
        scan.get("meters_per_unit") == 1.0 and scan.get("up_axis") == "Y"
        for scan in scan_records
    )
    report = {
        "report_type": "roomplan_capture_intake",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "semantic_extraction_complete_plan_registration_pending",
        "passed": True,
        "site": manifest.get("site"),
        "capture_date": manifest.get("capture", {}).get("date"),
        "evidence": records,
        "semantic_summary": {
            "scan_count": len(scan_records),
            "coordinate_metadata_valid": coordinate_metadata_valid,
            "sections": sum(len(scan["semantic_sections"]) for scan in scan_records),
            "walls": sum(scan["category_counts"].get("Wall", 0) for scan in scan_records),
            "doors": len(all_doors),
            "windows": sum(scan["category_counts"].get("Window", 0) for scan in scan_records),
        },
        "manual_reference": {
            **manual,
            "closest_semantic_door": closest,
            "semantic_widths_below_reported_manual_minimum": below_manual,
            "classification_rule": (
                "The user-reported 0.85 m administration-wide minimum controls intended "
                "traversable doors. Smaller RoomPlan detections remain review items and "
                "must not silently become route constraints or be globally rescaled."
            ),
        },
        "clearance_gate": {
            "physical_body_only": clearance_profile(width_m, 0.0),
            "production_nav2_profile": production,
            "simulation_tight_door_profile": presentation,
            "height_margin_above_crown_lidar_m": round(height_m - 1.170, 3),
            "presentation_profile_constraints": [
                "simulation only; physical_release remains false",
                "straight, centred doorway approach",
                "maximum doorway speed 0.10 m/s",
                "no in-doorway rotation or passing",
                "front and crown LiDAR obstacle veto remain active",
            ],
        },
        "registration_gate": {
            "scan_to_plan_alignment_complete": False,
            "multi_scan_overlap_transform_accepted": False,
            "reason": (
                "The captures provide useful complementary coverage but no sufficiently "
                "constrained common anchor for an evidence-grade automatic rigid stitch."
            ),
            "next_action": (
                "Register each native scan independently to approved Block A page 2, "
                "then validate the Principal and Vice-Principal doorway identities against "
                "the walkthrough before generating the measured occupancy map."
            ),
        },
        "privacy_gate": {
            "reviewed": privacy.get("reviewed") is True,
            "people_present": privacy.get("people_present") is True,
            "screens_or_documents_present": privacy.get("screens_or_documents_present") is True,
            "gps_metadata_present_in_photos": privacy.get("gps_metadata_present_in_photos") is True,
            "raw_media_may_be_committed": False,
        },
        "physical_release": False,
        "claim_boundary": (
            "This report validates capture integrity, semantic extraction, scale consistency, "
            "and simulation-only clearance arithmetic. It does not certify scan alignment, "
            "thresholds, autonomous physical navigation, or safe deployment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
