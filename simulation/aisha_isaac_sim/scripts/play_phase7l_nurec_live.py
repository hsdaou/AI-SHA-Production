#!/usr/bin/env python3
"""Play accepted Principal-route poses through the registered NuRec camera path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scene",
        type=Path,
        default=ROOT / "scenes/phase7l_nurec_registered_administration.usda",
    )
    parser.add_argument(
        "--trajectory-report",
        type=Path,
        default=ROOT
        / "results/administration_nav2_phase7e_static_fusion_mission.json",
    )
    parser.add_argument(
        "--camera",
        default="/World/Presentation/NuRec/gauss/Cameras/camera_0",
    )
    parser.add_argument(
        "--robot", default="/World/Presentation/MetricWorld/AISHA"
    )
    parser.add_argument("--segments", default="6,7,8,9")
    parser.add_argument("--camera-start", type=int, default=0)
    parser.add_argument("--camera-end", type=int, default=105)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="number of loops; zero repeats until the GUI is closed",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "tmp/phase7l_nurec_live_session.json",
    )
    return parser.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp


APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "multi_gpu": False,
    }
)

import carb
import omni.timeline
import omni.usd
from pxr import Gf, Usd


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_recorded_poses(samples: list[dict], count: int) -> list[dict]:
    if not samples:
        raise ValueError("the selected Principal-route segments contain no poses")
    if count <= 1:
        return [samples[0]]
    return [
        samples[round(index * (len(samples) - 1) / (count - 1))]
        for index in range(count)
    ]


def wait_for_stage(scene: Path, max_updates: int = 600) -> Usd.Stage:
    context = omni.usd.get_context()
    expected = scene.resolve()
    last_identifier: Path | None = None
    last_status = ("", 0, 0)
    for _ in range(max_updates):
        APP.update()
        stage = context.get_stage()
        last_status = context.get_stage_loading_status()
        if stage is None:
            continue
        root_layer = stage.GetRootLayer()
        real_path = root_layer.realPath or root_layer.identifier
        if real_path:
            last_identifier = Path(real_path).resolve()
        if last_status[2] == 0 and last_identifier == expected:
            for _ in range(16):
                APP.update()
            return context.get_stage()
    raise RuntimeError(
        "Isaac Sim did not finish opening the Phase 7L stage: "
        f"expected={expected}, current={last_identifier}, status={last_status}"
    )


def set_robot_pose(robot: Usd.Prim, sample: dict) -> None:
    robot.GetAttribute("xformOp:translate:route").Set(
        Gf.Vec3d(float(sample["x_m"]), float(sample["y_m"]), 0.0)
    )
    robot.GetAttribute("xformOp:rotateZ:route").Set(
        math.degrees(float(sample["yaw_rad"]))
    )


def main() -> int:
    scene = ARGS.scene.expanduser().resolve()
    trajectory = ARGS.trajectory_report.expanduser().resolve()
    if not scene.is_file() or not trajectory.is_file():
        raise FileNotFoundError(scene if not scene.is_file() else trajectory)
    mission = json.loads(trajectory.read_text(encoding="utf-8"))
    if mission.get("outcome") != "success" or mission.get("waypoints_completed") != 12:
        raise RuntimeError("the accepted successful 12-leg Phase 7E mission is required")
    segment_ids = {
        int(value.strip()) for value in ARGS.segments.split(",") if value.strip()
    }
    source = [
        sample
        for sample in mission.get("pose_trace", [])
        if int(sample["segment_id"]) in segment_ids
    ]
    frame_count = ARGS.camera_end - ARGS.camera_start + 1
    if frame_count <= 1:
        raise ValueError("camera end must be greater than camera start")
    selected = select_recorded_poses(source, frame_count)

    settings = carb.settings.get_settings()
    settings.set_bool("/renderer/multiGpu/enabled", False)
    settings.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
    context = omni.usd.get_context()
    if not context.open_stage(str(scene)):
        raise RuntimeError(f"could not open {scene}")
    stage = wait_for_stage(scene)
    camera = stage.GetPrimAtPath(ARGS.camera)
    robot = stage.GetPrimAtPath(ARGS.robot)
    if not camera.IsValid() or not robot.IsValid():
        raise RuntimeError(f"missing camera={ARGS.camera} or robot={ARGS.robot}")
    translate = robot.GetAttribute("xformOp:translate:route")
    rotate = robot.GetAttribute("xformOp:rotateZ:route")
    if not translate.IsValid() or not rotate.IsValid():
        raise RuntimeError("registered robot is missing route transform attributes")

    # Pose playback is an in-memory presentation overlay; never mutate the
    # hash-locked composite root layer.
    stage.SetEditTarget(stage.GetSessionLayer())
    viewport = None
    if not ARGS.headless:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        for _ in range(120):
            APP.update()
            viewport = get_viewport_from_window_name("Viewport")
            if viewport is not None:
                break
        if viewport is None:
            raise RuntimeError("Isaac Sim GUI viewport did not become available")
        viewport.set_active_camera(ARGS.camera)

    timeline = omni.timeline.get_timeline_interface()
    time_codes_per_second = float(stage.GetTimeCodesPerSecond() or 24.0)
    loops_completed = 0
    frames_presented = 0
    interrupted = False
    try:
        while APP.is_running() and (
            ARGS.repeat_count == 0 or loops_completed < ARGS.repeat_count
        ):
            for offset, sample in enumerate(selected):
                if not APP.is_running():
                    break
                started = time.perf_counter()
                camera_code = ARGS.camera_start + offset
                timeline.set_current_time(camera_code / time_codes_per_second)
                set_robot_pose(robot, sample)
                if viewport is not None:
                    viewport.set_active_camera(ARGS.camera)
                APP.update()
                frames_presented += 1
                if not ARGS.headless:
                    time.sleep(
                        max(0.0, 1.0 / ARGS.fps - (time.perf_counter() - started))
                    )
            loops_completed += 1
    except KeyboardInterrupt:
        interrupted = True

    report = {
        "report_type": "phase7l_nurec_live_presentation_session",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "operator_interrupted" if interrupted else "completed_requested_loops",
        "scene": portable_path(scene),
        "scene_sha256": sha256_file(scene),
        "trajectory_report": portable_path(trajectory),
        "trajectory_report_sha256": sha256_file(trajectory),
        "camera": ARGS.camera,
        "robot": ARGS.robot,
        "segments": sorted(segment_ids),
        "camera_time_codes": [ARGS.camera_start, ARGS.camera_end],
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting with NuRec RTX",
        "single_gpu": True,
        "loops_completed": loops_completed,
        "frames_presented": frames_presented,
        "recorded_pose_selection_without_interpolation": True,
        "source_motion_was_live_nav2_and_learned_safety": True,
        "presentation_player_executes_policy_live": False,
        "navigation_collision_geometry_changed": False,
        "physical_release": False,
    }
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        APP.close()
    raise SystemExit(exit_code)
