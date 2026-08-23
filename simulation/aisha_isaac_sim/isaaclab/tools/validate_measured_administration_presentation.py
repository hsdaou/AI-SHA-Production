#!/usr/bin/env python3
"""Validate the measured-door live administration presentation evidence."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_REPORT = ROOT / "results" / "measured_administration_final_cinematic_policy_only_seed10643.json"
ASSET_REPORT = ROOT / "results" / "administration_live_assets_report.json"
BUILD_REPORT = ROOT / "results" / "administration_build_report.json"
PRESENTATION_REPORT = ROOT / "results" / "measured_administration_live_policy_presentation_video_report.json"
RAW_VIDEO = ROOT / "outputs" / "measured_administration_final_speedgate_cinematic_seed10643" / "aisha-block-a-learned-route-step-0.mp4"
PRESENTATION_VIDEO = ROOT / "media" / "videos" / "AI-SHA_Measured_Administration_Live_Policy_3x.mp4"
CONTACT_SHEET = ROOT / "media" / "AI-SHA_Measured_Administration_Live_Policy_3x_contact_sheet.jpg"
OUTPUT = ROOT / "results" / "measured_administration_final_presentation_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "fps": fps,
        "frame_count": frames,
        "resolution": [width, height],
        "duration_s": frames / fps,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def uniform_occlusion_intervals(path: Path) -> list[list[float]]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    hits: list[float] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % 6 == 0:
            sample = cv2.resize(frame, (160, 90))
            grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            if float(np.std(sample)) < 18.0 and float((cv2.Canny(grey, 50, 120) > 0).mean()) < 0.005:
                hits.append(frame_index / fps)
        frame_index += 1
    capture.release()
    intervals: list[list[float]] = []
    for timestamp in hits:
        if not intervals or timestamp - intervals[-1][1] > 0.25:
            intervals.append([timestamp, timestamp])
        else:
            intervals[-1][1] = timestamp
    return [[round(start, 3), round(end, 3)] for start, end in intervals]


def doorway_speed_gate(trace: list[dict[str, object]]) -> dict[str, object]:
    diagonal = math.pi / 4.0
    doors = {
        3: ((17.1, -5.05), (1.0, 0.0), (0.0, 1.0), "vice_principal_entry"),
        4: ((17.1, -5.05), (1.0, 0.0), (0.0, 1.0), "vice_principal_exit"),
        8: ((6.978, -7.628), (math.cos(diagonal), math.sin(diagonal)), (-math.sin(diagonal), math.cos(diagonal)), "principal_entry"),
        9: ((6.978, -7.628), (math.cos(diagonal), math.sin(diagonal)), (-math.sin(diagonal), math.cos(diagonal)), "principal_exit"),
    }
    samples: list[dict[str, object]] = []
    maximum = 0.0
    maximum_sample: dict[str, object] | None = None
    for pose in trace:
        segment = int(pose["segment_id"])
        if segment not in doors:
            continue
        centre, tangent, normal, label = doors[segment]
        relative = (float(pose["x_m"]) - centre[0], float(pose["y_m"]) - centre[1])
        tangent_distance = abs(relative[0] * tangent[0] + relative[1] * tangent[1])
        normal_distance = abs(relative[0] * normal[0] + relative[1] * normal[1])
        if tangent_distance <= 1.10 and normal_distance <= 0.90:
            speed = abs(float(pose["linear_velocity_mps"]))
            sample = {
                "segment": label,
                "step": int(pose["step"]),
                "speed_mps": speed,
                "normal_distance_m": normal_distance,
                "tangent_distance_m": tangent_distance,
            }
            samples.append(sample)
            if speed > maximum:
                maximum = speed
                maximum_sample = sample
    return {
        "sample_count": len(samples),
        "maximum_abs_speed_mps": maximum,
        "limit_mps": 0.10,
        "maximum_sample": maximum_sample,
        "passed": bool(samples) and maximum <= 0.10,
    }


def main() -> int:
    required = (RUN_REPORT, ASSET_REPORT, BUILD_REPORT, PRESENTATION_REPORT, RAW_VIDEO, PRESENTATION_VIDEO, CONTACT_SHEET)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation inputs: {missing}")

    run = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    assets = json.loads(ASSET_REPORT.read_text(encoding="utf-8"))
    build = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    presentation = json.loads(PRESENTATION_REPORT.read_text(encoding="utf-8"))
    checkpoint = Path(run["checkpoint"])
    raw_video = video_metadata(RAW_VIDEO)
    presentation_video = video_metadata(PRESENTATION_VIDEO)
    occlusions = uniform_occlusion_intervals(RAW_VIDEO)
    non_initial_occlusions = [interval for interval in occlusions if interval[1] > 0.20]
    trace = run.get("pose_trace", [])
    speed_gate = doorway_speed_gate(trace)
    body_circumscribed_radius_m = math.hypot(0.725, 0.384)
    minimum_polygon_centre_distance_m = min(
        math.hypot(float(pose["x_m"]), float(pose["y_m"])) for pose in trace
    )
    polygon_clearance_m = minimum_polygon_centre_distance_m - 2.30 - body_circumscribed_radius_m
    doors = build.get("doors", {})
    events = run.get("waypoint_events", [])
    controls = run.get("control_steps", {})
    checks = {
        "measured_live_task": run.get("task") == "Isaac-AISHA-Administration-Live-MeasuredTightDoor-Direct-v0",
        "route_success": run.get("outcome") == "success",
        "all_12_segments_completed": run.get("waypoints_completed") == 12
        and [event.get("segment_id") for event in events] == list(range(12)),
        "zero_collisions": run.get("termination_details", {}).get("static_collision") is False
        and run.get("termination_details", {}).get("dynamic_obstacle_collision") is False,
        "no_turn_or_dwell_supervisor": controls.get("physics_supervisor_turn") == 0
        and controls.get("presentation_dwell") == 0,
        "policy_plus_doorway_safety_disclosed": run.get("policy_architecture")
        == "ppo_route_policy_plus_deterministic_mapped_doorway_safety",
        "root_transform_animation_disabled": run.get("root_transform_animation") is False,
        "checkpoint_hash_matches": checkpoint.is_file()
        and sha256_file(checkpoint) == run.get("checkpoint_sha256"),
        "build_and_asset_gates_pass": build.get("passed") is True and assets.get("passed") is True,
        "live_asset_scene_current": assets.get("source_scene_sha256")
        == sha256_file(ROOT / "scenes" / "administration.usd"),
        "reported_door_dimensions_applied": doors.get("vice_principal", {}).get("clear_width_m") == 0.85
        and doors.get("principal", {}).get("clear_width_m") == 0.90
        and all(door.get("clear_height_m") == 2.12 for door in doors.values()),
        "secured_open_leaves_are_visual_only": all(
            door.get("open_leaf_collision") == "visual_only_secured_fully_open" for door in doors.values()
        ),
        "vp_locked_capture_limitation_disclosed": build.get("capture_limitations", {})
        .get("vice_principal_office_interior", {})
        .get("status")
        == "not_captured_locked_during_site_visit",
        "central_drop_is_020m_mapped_no_go": build.get("central_atrium_drop", {}).get("step_down_m") == 0.20
        and build.get("central_atrium_drop", {}).get("robot_access") == "prohibited",
        "robot_footprint_stays_outside_polygon": polygon_clearance_m > 0.0,
        "doorway_speed_limit_passed": speed_gate["passed"],
        "cinematic_camera_has_six_shots": run.get("camera", {}).get("mode") == "cinematic"
        and len(run.get("camera", {}).get("shot_events", [])) == 6,
        "raw_video_matches_run": raw_video["resolution"] == [1280, 720]
        and abs(float(raw_video["duration_s"]) - float(run["duration_s"])) < 0.2,
        "no_noninitial_uniform_camera_occlusion": not non_initial_occlusions,
        "presentation_video_hash_matches": presentation.get("output_video_sha256")
        == presentation_video["sha256"],
        "presentation_motion_unchanged": presentation.get("motion_changed") is False,
        "presentation_disclosures_present": presentation.get("measured_doorway_safety_layer") is True
        and presentation.get("camera_mode") == "cinematic",
        "pose_trace_finite": bool(trace)
        and all(
            all(math.isfinite(float(pose[key])) for key in ("x_m", "y_m", "yaw_rad", "linear_velocity_mps"))
            for pose in trace
        ),
    }
    report = {
        "report_type": "measured_administration_final_presentation_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "route": {
            "outcome": run.get("outcome"),
            "waypoints_completed": run.get("waypoints_completed"),
            "completed_steps": run.get("completed_steps"),
            "duration_s": run.get("duration_s"),
            "control_steps": controls,
        },
        "doorway_speed_gate": speed_gate,
        "central_polygon_no_go_gate": {
            "step_down_m": 0.20,
            "outer_radius_m": 2.30,
            "body_circumscribed_radius_m": body_circumscribed_radius_m,
            "minimum_robot_centre_distance_m": minimum_polygon_centre_distance_m,
            "minimum_footprint_clearance_m": polygon_clearance_m,
            "passed": polygon_clearance_m > 0.0,
        },
        "camera": {
            "mode": run.get("camera", {}).get("mode"),
            "shot_events": run.get("camera", {}).get("shot_events", []),
            "uniform_occlusion_intervals_s": occlusions,
            "non_initial_uniform_occlusion_intervals_s": non_initial_occlusions,
        },
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "raw_video": raw_video,
        "presentation_video": presentation_video,
        "contact_sheet": {"path": str(CONTACT_SHEET), "sha256": sha256_file(CONTACT_SHEET)},
        "claim_boundary": run.get("claim_boundary"),
        "physical_release": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(OUTPUT), "checks": checks}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
