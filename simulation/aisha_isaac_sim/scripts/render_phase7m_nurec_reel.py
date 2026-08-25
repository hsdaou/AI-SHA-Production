#!/usr/bin/env python3
"""Render recorded AI-SHA poses inside the registered Phase 7L NuRec scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--camera", default="/World/Presentation/NuRec/gauss/Cameras/camera_0"
    )
    parser.add_argument("--robot", default="/World/Presentation/MetricWorld/AISHA")
    parser.add_argument("--segments", default="6,7,8,9")
    parser.add_argument(
        "--shot-plan",
        type=Path,
        help="optional YAML shot plan with fixed or moving NuRec cameras",
    )
    parser.add_argument("--camera-start", type=int, default=0)
    parser.add_argument("--camera-end", type=int, default=105)
    parser.add_argument(
        "--camera-codes",
        default="",
        help="optional comma-separated subset; pose synchronization still uses the full range",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "tmp/phase7m_nurec_reel_frames"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/phase7m_nurec_reel_render.json",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        "multi_gpu": False,
    }
)

import carb
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import yaml
from PIL import Image
from pxr import Gf, Usd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def select_recorded_pose(samples: list[dict], offset: int, count: int) -> dict:
    if not samples:
        raise ValueError("the selected Principal-route segments contain no poses")
    if count <= 1:
        return samples[0]
    source_index = round(offset * (len(samples) - 1) / (count - 1))
    return samples[source_index]


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


def parse_camera_codes(start: int, end: int, values: str) -> list[int]:
    if end < start:
        raise ValueError("--camera-end must not be less than --camera-start")
    if not values.strip():
        return list(range(start, end + 1))
    codes = [int(value.strip()) for value in values.split(",") if value.strip()]
    if not codes:
        raise ValueError("--camera-codes did not contain any integers")
    outside = [code for code in codes if code < start or code > end]
    if outside:
        raise ValueError(f"camera codes outside {start}..{end}: {outside}")
    return codes


def fraction_index(fraction: float, count: int) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"pose fraction must be in [0, 1], got {fraction}")
    return round(fraction * (count - 1))


def build_shot_instructions(mission: dict, shot_plan: Path) -> tuple[list[dict], dict]:
    plan = yaml.safe_load(shot_plan.read_text(encoding="utf-8"))
    instructions: list[dict] = []
    for shot in plan.get("shots", []):
        shot_id = str(shot["id"])
        segment_ids = {int(value) for value in shot["segments"]}
        source = [
            sample
            for sample in mission.get("pose_trace", [])
            if int(sample["segment_id"]) in segment_ids
        ]
        if not source:
            raise ValueError(f"shot {shot_id} has no source poses")
        frame_count = int(shot["frame_count"])
        if frame_count < 1:
            raise ValueError(f"shot {shot_id} frame_count must be positive")
        start_index = fraction_index(
            float(shot.get("pose_fraction_start", 0.0)), len(source)
        )
        end_index = fraction_index(
            float(shot.get("pose_fraction_end", 1.0)), len(source)
        )
        camera_range = shot.get("camera_time_codes")
        if camera_range is None:
            camera_start = camera_end = int(shot["camera_time_code"])
        else:
            camera_start, camera_end = (int(value) for value in camera_range)
        for offset in range(frame_count):
            alpha = 0.0 if frame_count == 1 else offset / (frame_count - 1)
            source_index = round(start_index + alpha * (end_index - start_index))
            camera_code = round(camera_start + alpha * (camera_end - camera_start))
            instructions.append(
                {
                    "shot_id": shot_id,
                    "camera_time_code": camera_code,
                    "sample": source[source_index],
                }
            )
    if not instructions:
        raise ValueError("shot plan must contain at least one rendered frame")
    return instructions, plan


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
    full_count = ARGS.camera_end - ARGS.camera_start + 1
    shot_plan_data = None
    if ARGS.shot_plan is not None:
        shot_plan = ARGS.shot_plan.expanduser().resolve()
        if not shot_plan.is_file():
            raise FileNotFoundError(shot_plan)
        instructions, shot_plan_data = build_shot_instructions(mission, shot_plan)
    else:
        camera_codes = parse_camera_codes(
            ARGS.camera_start, ARGS.camera_end, ARGS.camera_codes
        )
        instructions = []
        for camera_code in camera_codes:
            offset = camera_code - ARGS.camera_start
            instructions.append(
                {
                    "shot_id": "continuous_source_camera",
                    "camera_time_code": camera_code,
                    "sample": select_recorded_pose(source, offset, full_count),
                }
            )

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
    if not robot.GetAttribute("xformOp:translate:route").IsValid() or not robot.GetAttribute(
        "xformOp:rotateZ:route"
    ).IsValid():
        raise RuntimeError("registered robot is missing route transform attributes")

    # All pose edits are ephemeral. The registered, hash-locked composite is
    # never mutated by presentation rendering.
    stage.SetEditTarget(stage.GetSessionLayer())
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    product = rep.create.render_product(ARGS.camera, (ARGS.width, ARGS.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(product)
    timeline = omni.timeline.get_timeline_interface()
    time_codes_per_second = float(stage.GetTimeCodesPerSecond() or 24.0)
    rendered: list[dict[str, object]] = []
    try:
        for output_index, instruction in enumerate(instructions):
            camera_code = int(instruction["camera_time_code"])
            sample = instruction["sample"]
            timeline.set_current_time(camera_code / time_codes_per_second)
            set_robot_pose(robot, sample)
            for _ in range(max(ARGS.warmup, 1)):
                APP.update()
            rep.orchestrator.step(delta_time=0.0)
            rgba = np.asarray(rgb.get_data())
            if rgba.size == 0:
                raise RuntimeError(f"renderer returned no RGB data at {camera_code}")
            output = ARGS.output_dir / f"frame_{output_index:04d}.png"
            Image.fromarray(rgba).convert("RGB").save(output)
            rendered.append(
                {
                    "output_index": output_index,
                    "shot_id": instruction["shot_id"],
                    "camera_time_code": camera_code,
                    "path": portable_path(output),
                    "sha256": sha256_file(output),
                    "pose": {
                        "step": int(sample["step"]),
                        "segment_id": int(sample["segment_id"]),
                        "x_m": float(sample["x_m"]),
                        "y_m": float(sample["y_m"]),
                        "yaw_rad": float(sample["yaw_rad"]),
                        "linear_velocity_mps": float(sample["linear_velocity_mps"]),
                    },
                    "mean_rgb": [
                        float(value) for value in rgba[..., :3].mean(axis=(0, 1))
                    ],
                    "std_rgb": [
                        float(value) for value in rgba[..., :3].std(axis=(0, 1))
                    ],
                }
            )
            print(
                f"PHASE7M: rendered {output_index + 1}/{len(instructions)} "
                f"camera={camera_code} segment={sample['segment_id']} -> {output}",
                flush=True,
            )
    finally:
        rgb.detach()
        product.destroy()

    non_blank = all(max(item["std_rgb"]) > 5.0 for item in rendered)
    rendered_codes = [int(item["camera_time_code"]) for item in instructions]
    report = {
        "report_type": "phase7m_nurec_recorded_pose_reel_render",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rendered" if non_blank else "blank_render_detected",
        "passed": bool(non_blank and len(rendered) == len(instructions)),
        "scene": portable_path(scene),
        "scene_sha256": sha256_file(scene),
        "trajectory_report": portable_path(trajectory),
        "trajectory_report_sha256": sha256_file(trajectory),
        "camera": ARGS.camera,
        "robot": ARGS.robot,
        "segments": sorted(segment_ids),
        "source_pose_count": len(source),
        "camera_time_range": [ARGS.camera_start, ARGS.camera_end],
        "camera_codes_rendered": rendered_codes,
        "shot_plan": portable_path(ARGS.shot_plan) if ARGS.shot_plan else None,
        "shot_plan_sha256": sha256_file(ARGS.shot_plan) if ARGS.shot_plan else None,
        "shot_plan_data": shot_plan_data,
        "resolution": [ARGS.width, ARGS.height],
        "renderer": "RaytracedLighting with NuRec RTX",
        "single_gpu": True,
        "recorded_pose_selection_without_interpolation": True,
        "source_motion_was_live_nav2_and_learned_safety": True,
        "presentation_renderer_executes_policy_live": False,
        "navigation_collision_geometry_changed": False,
        "physical_release": False,
        "frames": rendered,
    }
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE7M: wrote {ARGS.report}", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        APP.close()
    raise SystemExit(exit_code)
