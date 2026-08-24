#!/usr/bin/env python3
"""Play the verified Phase 7E pose trace in the Isaac Sim GUI for presentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-shot", type=float, default=3.0)
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="number of route loops; zero repeats until the GUI is closed",
    )
    parser.add_argument(
        "--trajectory-report",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_fusion_mission.json",
    )
    parser.add_argument(
        "--camera-profile",
        type=Path,
        default=ROOT / "config/phase7f_operator_presentation.yaml",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "tmp/phase7g_live_omniverse_session.json"
    )
    return parser.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp


APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

import omni.usd
from pxr import Gf, Sdf, UsdGeom


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source() -> tuple[dict, list[dict], list[dict]]:
    mission = json.loads(ARGS.trajectory_report.read_text(encoding="utf-8"))
    profile = yaml.safe_load(ARGS.camera_profile.read_text(encoding="utf-8"))
    if mission.get("outcome") != "success" or mission.get("waypoints_completed") != 12:
        raise RuntimeError("accepted successful 12-leg source mission is required")
    trace = mission.get("pose_trace", [])
    if not trace:
        raise RuntimeError("source mission has no pose trace")
    return mission, trace, profile["shots"]


def select_recorded_poses(samples: list[dict], count: int) -> list[dict]:
    """Select recorded poses without interpolation or invented motion."""
    if not samples:
        raise ValueError("cannot select from an empty trace")
    if count <= 1:
        return [samples[0]]
    return [samples[round(index * (len(samples) - 1) / (count - 1))] for index in range(count)]


def set_robot_pose(stage, sample: dict) -> None:
    prim = stage.GetPrimAtPath("/World/AISHA")
    prim.GetAttribute("xformOp:translate:route").Set(
        Gf.Vec3d(float(sample["x_m"]), float(sample["y_m"]), 0.0)
    )
    prim.GetAttribute("xformOp:rotateZ:route").Set(math.degrees(float(sample["yaw_rad"])))


def wait_for_stage_ready(scene: Path, max_updates: int = 600):
    """Wait for Kit's asynchronous GUI stage transition to finish.

    ``open_stage`` can return before the GUI has installed the new stage.  During
    that short transition ``SimulationApp.is_running()`` reports false because
    its USD context temporarily has no stage.  Entering the presentation loop
    at that point therefore looks like a crash: the loop is skipped and the
    script's normal ``finally`` block closes Isaac Sim.
    """
    context = omni.usd.get_context()
    expected = scene.resolve()
    last_status = ("", 0, 0)
    last_identifier = None
    for _ in range(max_updates):
        APP.update()
        stage = context.get_stage()
        last_status = context.get_stage_loading_status()
        if stage is not None:
            root_layer = stage.GetRootLayer()
            real_path = root_layer.realPath or root_layer.identifier
            if real_path:
                last_identifier = Path(real_path).resolve()
            if last_status[2] == 0 and last_identifier == expected:
                # Give the renderer/viewport a few frames to consume the fully
                # loaded stage before defining and selecting our camera.
                for _ in range(8):
                    APP.update()
                return context.get_stage()
    raise RuntimeError(
        "Isaac Sim did not finish opening the administration stage "
        f"after {max_updates} updates: expected={expected}, "
        f"current={last_identifier}, loading_status={last_status}"
    )


def main() -> int:
    scene = ROOT / "scenes/administration.usd"
    mission, trace, shots = load_source()
    if not omni.usd.get_context().open_stage(str(scene.resolve())):
        raise RuntimeError(f"could not open {scene}")
    stage = wait_for_stage_ready(scene)
    if not APP.is_running():
        raise RuntimeError("Isaac Sim stopped after the administration stage finished loading")
    print(f"PHASE7G LIVE stage_ready={scene.resolve()}")
    camera_path = "/World/Phase7GPresentationCamera"
    camera_prim = stage.DefinePrim(camera_path, "Camera")
    camera = UsdGeom.Camera(camera_prim)
    camera_prim.CreateAttribute(
        "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d
    ).Set(Gf.Vec3d(0.0, 0.0, -1.0))
    viewport = None
    camera_state = None
    if not ARGS.headless:
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        for _ in range(120):
            APP.update()
            viewport = get_viewport_from_window_name("Viewport")
            if viewport is not None:
                break
        if viewport is None:
            raise RuntimeError("Isaac Sim GUI viewport did not become available")
        viewport.set_active_camera(camera_path)
        camera_state = ViewportCameraState(camera_path, viewport)

    loops_completed = 0
    frames_presented = 0
    segments_presented: list[int] = []
    interrupted = False
    try:
        while APP.is_running() and (ARGS.repeat_count == 0 or loops_completed < ARGS.repeat_count):
            for shot_index, shot in enumerate(shots, start=1):
                segment_ids = {int(value) for value in shot["segments"]}
                source = [sample for sample in trace if int(sample["segment_id"]) in segment_ids]
                fraction = shot.get("source_fraction", [0.0, 1.0])
                first = int(math.floor(float(fraction[0]) * (len(source) - 1)))
                last = int(math.ceil(float(fraction[1]) * (len(source) - 1)))
                source = source[first : last + 1]
                samples = select_recorded_poses(
                    source, max(2, round(ARGS.seconds_per_shot * ARGS.fps))
                )
                camera.GetFocalLengthAttr().Set(float(shot["focal_length_mm"]))
                if camera_state is not None:
                    camera_state.set_position_world(Gf.Vec3d(*map(float, shot["camera"])), True)
                    camera_state.set_target_world(Gf.Vec3d(*map(float, shot["look_at"])), True)
                    viewport.set_active_camera(camera_path)
                print(
                    f"PHASE7G LIVE shot={shot_index}/{len(shots)} "
                    f"title={shot['title']} segments={sorted(segment_ids)}"
                )
                for sample in samples:
                    if not APP.is_running():
                        break
                    started = time.perf_counter()
                    set_robot_pose(stage, sample)
                    APP.update()
                    frames_presented += 1
                    segments_presented.append(int(sample["segment_id"]))
                    if not ARGS.headless:
                        time.sleep(max(0.0, 1.0 / ARGS.fps - (time.perf_counter() - started)))
            loops_completed += 1
    except KeyboardInterrupt:
        interrupted = True

    report = {
        "report_type": "phase7g_live_omniverse_presentation_session",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "operator_interrupted" if interrupted else "completed_requested_loops",
        "scene": str(scene.resolve()),
        "scene_sha256": sha256(scene),
        "trajectory_report": str(ARGS.trajectory_report.resolve()),
        "trajectory_report_sha256": sha256(ARGS.trajectory_report),
        "camera_profile": str(ARGS.camera_profile.resolve()),
        "camera_profile_sha256": sha256(ARGS.camera_profile),
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "loops_completed": loops_completed,
        "frames_presented": frames_presented,
        "segment_frame_counts": dict(Counter(segments_presented)),
        "recorded_pose_selection_without_interpolation": True,
        "source_motion_was_live_nav2_and_learned_safety": True,
        "presentation_player_executes_policy_live": False,
        "physical_release": False,
    }
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"AISHA_PHASE7G_LIVE_SESSION report={ARGS.report.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:  # Keep the real error visible before Kit disables logging on close.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        APP.close()
    raise SystemExit(exit_code)
