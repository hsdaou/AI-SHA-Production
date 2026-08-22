#!/usr/bin/env python3
"""Validate live-policy administration evidence, provenance, route, and video."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_REPORT = ROOT / "results" / "administration_live_policy_video_report.json"
ASSET_REPORT = ROOT / "results" / "administration_live_assets_report.json"
BUILD_REPORT = ROOT / "results" / "administration_build_report.json"
PRESENTATION_REPORT = ROOT / "results" / "administration_live_policy_presentation_video_report.json"
RAW_VIDEO = ROOT / "media" / "videos" / "administration_live_policy" / "aisha-block-a-learned-route-step-0.mp4"
PRESENTATION_VIDEO = ROOT / "media" / "videos" / "AI-SHA_Administration_Live_Policy_3x.mp4"
OUTPUT = ROOT / "results" / "administration_live_policy_validation.json"


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
    frame_index = 0
    hits: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % 3 == 0:
            sample = cv2.resize(frame, (160, 90))
            standard_deviation = float(np.std(sample))
            grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            edge_fraction = float((cv2.Canny(grey, 50, 120) > 0).mean())
            if standard_deviation < 18.0 and edge_fraction < 0.005:
                hits.append(frame_index / fps)
        frame_index += 1
    capture.release()

    intervals: list[list[float]] = []
    for timestamp in hits:
        if not intervals or timestamp - intervals[-1][1] > 0.15:
            intervals.append([timestamp, timestamp])
        else:
            intervals[-1][1] = timestamp
    return [[round(start, 3), round(end, 3)] for start, end in intervals]


def main() -> int:
    required = (
        RUN_REPORT,
        ASSET_REPORT,
        BUILD_REPORT,
        PRESENTATION_REPORT,
        RAW_VIDEO,
        PRESENTATION_VIDEO,
    )
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
    occlusion_intervals = uniform_occlusion_intervals(RAW_VIDEO)
    non_initial_occlusions = [value for value in occlusion_intervals if value[1] > 0.2]

    events = run.get("waypoint_events", [])
    controls = run.get("control_steps", {})
    live_stage_checks = run.get("live_stage_checks", {})
    doors = build.get("doors", {})
    checks = {
        "live_administration_task": run.get("task") == "Isaac-AISHA-Administration-Live-Direct-v0",
        "live_execution_mode": run.get("execution_mode")
        == "checkpoint_policy_live_in_walkthrough_matched_administration_scene",
        "route_success": run.get("outcome") == "success",
        "all_route_segments_completed": run.get("waypoints_completed") == 12
        and run.get("route_segment_count") == 12,
        "route_segment_ids_complete": [event.get("segment_id") for event in events] == list(range(12)),
        "control_accounting_complete": sum(int(value) for value in controls.values())
        == run.get("completed_steps"),
        "learned_policy_is_majority_control": controls.get("learned_sensor_policy", 0)
        > controls.get("physics_supervisor_turn", 0) + controls.get("presentation_dwell", 0),
        "root_transform_animation_disabled": run.get("root_transform_animation") is False,
        "physics_and_policy_rates_declared": run.get("physics_rate_hz") == 120.0
        and run.get("policy_rate_hz") == 30.0,
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_hash_matches": checkpoint.is_file()
        and sha256_file(checkpoint) == run.get("checkpoint_sha256"),
        "live_asset_build_passed": assets.get("passed") is True,
        "live_asset_source_scene_current": assets.get("source_scene_sha256")
        == sha256_file(ROOT / "scenes" / "administration.usd"),
        "presentation_robot_current": assets.get("presentation_robot_usd_sha256")
        == sha256_file(ROOT / "usd" / "aisha_loaded_presentation.usda"),
        "replay_robot_excluded": live_stage_checks.get("excluded_replay_robot", {}).get("active")
        is False,
        "live_robot_and_shell_present": live_stage_checks.get("live_robot_base", {}).get("active")
        is True
        and live_stage_checks.get("live_shell_body", {}).get("active") is True,
        "presentation_doors_disclosed_and_clear": len(doors) == 2
        and all(
            door.get("clear_width_m") == 1.4
            and door.get("threshold_height_mm") == 0
            and door.get("hinge_assumption") == "left_jamb"
            and "outward" in door.get("swing_assumption", "")
            for door in doors.values()
        ),
        "raw_video_complete": raw_video["resolution"] == [1280, 720]
        and abs(float(raw_video["duration_s"]) - float(run["duration_s"])) < 0.2,
        "no_noninitial_uniform_camera_occlusion": not non_initial_occlusions,
        "presentation_video_passed": presentation.get("passed") is True,
        "presentation_video_hash_matches": presentation.get("output_video_sha256")
        == presentation_video["sha256"],
        "presentation_motion_unchanged": presentation.get("motion_changed") is False,
    }
    finite_trace = all(
        all(math.isfinite(float(pose[key])) for key in ("x_m", "y_m", "yaw_rad"))
        for pose in run.get("pose_trace", [])
    )
    checks["pose_trace_finite"] = finite_trace and bool(run.get("pose_trace"))

    report = {
        "report_type": "administration_live_policy_validation",
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
        "camera": {
            "configuration": run.get("camera"),
            "tracking": run.get("camera_tracking"),
            "uniform_occlusion_intervals_s": occlusion_intervals,
            "non_initial_uniform_occlusion_intervals_s": non_initial_occlusions,
        },
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "raw_video": raw_video,
        "presentation_video": presentation_video,
        "claim_boundary": run.get("claim_boundary"),
        "physical_release": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(OUTPUT), "checks": checks}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
