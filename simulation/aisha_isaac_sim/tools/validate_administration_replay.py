#!/usr/bin/env python3
"""Validate the evidence chain for the administration learned-trajectory replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TRACE_PATH = RESULTS / "isaaclab_learned_route_playback_report.json"
BUILD_PATH = RESULTS / "administration_build_report.json"
CONFIG_PATH = ROOT / "config" / "administration_assumptions.yaml"
SCENE_PATH = ROOT / "scenes" / "administration.usd"
OUTPUT_PATH = RESULTS / "administration_learned_replay_validation.json"
EXPECTED_SEGMENTS = set(range(12))
RENDER_SHOT_SEGMENTS = ((0,), (1, 2), (3, 4), (5, 6), (7, 8, 9), (10, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-report", type=Path, default=TRACE_PATH)
    parser.add_argument("--build-report", type=Path, default=BUILD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Return 2D distance while conservatively interpolating trace samples."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def main() -> int:
    args = parse_args()
    trajectory_path = args.trajectory_report.resolve()
    build_path = args.build_report.resolve()
    output_path = args.output.resolve()
    source = load_json(trajectory_path)
    build = load_json(build_path)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    trace = source.get("pose_trace")
    if not isinstance(trace, list):
        trace = []

    segment_ids = {int(item["segment_id"]) for item in trace if isinstance(item, dict) and "segment_id" in item}
    mode_counts = Counter(str(item.get("control_mode")) for item in trace if isinstance(item, dict))
    steps = [int(item["step"]) for item in trace if isinstance(item, dict) and "step" in item]
    times = [float(item["elapsed_s"]) for item in trace if isinstance(item, dict) and "elapsed_s" in item]
    finite_poses = all(
        isinstance(item, dict)
        and all(math.isfinite(float(item[key])) for key in ("x_m", "y_m", "yaw_rad"))
        for item in trace
    )
    checkpoint = Path(str(source.get("checkpoint", "")))
    control_steps = source.get("control_steps", {})
    if not isinstance(control_steps, dict):
        control_steps = {}
    expected_modes = {str(mode) for mode, count in control_steps.items() if int(count) > 0}
    completed_steps = int(source.get("completed_steps", 0))
    door_values = list(build.get("doors", {}).values())
    rendered_segment_sequence = [segment for shot in RENDER_SHOT_SEGMENTS for segment in shot]
    trace_xy = [
        (float(item["x_m"]), float(item["y_m"]))
        for item in trace
        if isinstance(item, dict) and "x_m" in item and "y_m" in item
    ]
    column_config = config["appearance"]["atrium_columns"]
    column_positions = [tuple(float(value) for value in item) for item in column_config["positions_xy_m"]]
    minimum_column_distance = float(column_config["minimum_trace_centre_clearance_m"])
    column_radius = float(column_config["radius_m"])
    robot_width = float(config["presentation_release"]["robot_transit_width_m"])
    robot_length = float(config["presentation_release"]["robot_transit_length_m"])
    conservative_robot_radius = math.hypot(robot_width / 2.0, robot_length / 2.0)
    column_clearances = []
    for column_index, column in enumerate(column_positions):
        segment_distances = [
            point_to_segment_distance(column, start, end)
            for start, end in zip(trace_xy, trace_xy[1:])
        ]
        centre_distance = min(segment_distances, default=math.inf)
        closest_segment = segment_distances.index(centre_distance) if segment_distances else None
        column_clearances.append(
            {
                "column_index": column_index,
                "position_xy_m": list(column),
                "minimum_trace_centre_distance_m": round(centre_distance, 6),
                "minimum_surface_clearance_m": round(
                    centre_distance - conservative_robot_radius - column_radius, 6
                ),
                "closest_trace_segment_index": closest_segment,
                "passed": centre_distance >= minimum_column_distance,
            }
        )

    checks = {
        "scene_file_exists": SCENE_PATH.is_file(),
        "scene_reopened_during_build": bool(build.get("checks", {}).get("scene_reopens")),
        "scene_build_passed": bool(build.get("passed")),
        "scene_presentation_ready": bool(build.get("presentation_ready")),
        "scene_built_from_current_presentation_config": build.get("config_sha256") == sha256(CONFIG_PATH),
        "scene_build_records_current_column_layout": build.get("appearance", {}).get("atrium_columns")
        == column_config,
        "physical_release_remains_false": build.get("physical_route_released") is False,
        "both_door_width_gates_pass": len(door_values) == 2
        and all(bool(item.get("presentation_width_gate_passed")) for item in door_values),
        "both_thresholds_assumed_flush": len(door_values) == 2
        and all(int(item.get("threshold_height_mm", -1)) == 0 for item in door_values),
        "trajectory_outcome_success": source.get("outcome") == "success",
        "trajectory_waypoints_completed": int(source.get("waypoints_completed", 0)) == 12,
        "trajectory_completed_steps_positive": completed_steps > 0,
        "trajectory_control_step_sum_matches": sum(int(value) for value in control_steps.values())
        == completed_steps,
        "trajectory_has_expected_segments": segment_ids == EXPECTED_SEGMENTS,
        "trajectory_has_only_disclosed_control_modes": set(mode_counts) == expected_modes,
        "policy_only_trace_has_no_supervisor_or_dwell": source.get("route_control") != "policy-only"
        or (
            int(control_steps.get("physics_supervisor_turn", 0)) == 0
            and int(control_steps.get("presentation_dwell", 0)) == 0
            and set(mode_counts) == {"learned_sensor_policy"}
        ),
        "trajectory_trace_nonempty": len(trace) > 0,
        "trajectory_trace_steps_monotonic": steps == sorted(steps) and len(steps) == len(trace),
        "trajectory_trace_time_monotonic": times == sorted(times) and len(times) == len(trace),
        "trajectory_trace_poses_finite": finite_poses,
        "checkpoint_exists": checkpoint.is_file(),
        "render_shots_cover_each_segment_once": rendered_segment_sequence == list(range(12)),
        "all_atrium_columns_clear_conservative_robot_sweep": bool(column_clearances)
        and all(item["passed"] for item in column_clearances),
    }
    passed = all(checks.values())
    report = {
        "status": "passed" if passed else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
        "scene": str(SCENE_PATH.resolve()),
        "scene_sha256": sha256(SCENE_PATH) if SCENE_PATH.is_file() else None,
        "scene_build_report": str(build_path),
        "scene_build_report_sha256": sha256(build_path),
        "trajectory_report": str(trajectory_path),
        "trajectory_report_sha256": sha256(trajectory_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "trajectory": {
            "seed": source.get("seed"),
            "waypoints_completed": source.get("waypoints_completed"),
            "completed_steps": source.get("completed_steps"),
            "duration_s": source.get("duration_s"),
            "pose_trace_records": len(trace),
            "segment_ids": sorted(segment_ids),
            "control_mode_records": dict(mode_counts),
            "control_steps": source.get("control_steps"),
            "route_control": source.get("route_control"),
        },
        "render_shot_segments": [list(shot) for shot in RENDER_SHOT_SEGMENTS],
        "atrium_column_clearance": {
            "method": "minimum 2D distance to every interpolated learned-trace segment",
            "robot_envelope_width_m": robot_width,
            "robot_envelope_length_m": robot_length,
            "conservative_robot_radius_m": round(conservative_robot_radius, 6),
            "column_radius_m": column_radius,
            "required_trace_centre_distance_m": minimum_column_distance,
            "columns": column_clearances,
        },
        "motion_contract": "renderer selects recorded wheel-physics pose samples; it does not generate a second route",
        "visual_claim_boundary": "walkthrough-matched presentation environment; visual replay is not live policy execution",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['status']}: wrote {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
