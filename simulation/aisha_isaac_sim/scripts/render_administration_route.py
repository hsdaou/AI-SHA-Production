#!/usr/bin/env python3
"""Replay a verified Isaac Lab trajectory in the administration Omniverse scene."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-shot", type=float, default=4.0)
    parser.add_argument(
        "--renderer",
        choices=("RaytracedLighting", "PathTracing"),
        default="PathTracing",
    )
    parser.add_argument("--path-tracing-spp", type=int, default=16)
    parser.add_argument("--trajectory-report", type=Path)
    parser.add_argument("--frame-directory", type=Path)
    parser.add_argument("--render-report", type=Path)
    parser.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="start Isaac Sim even when nvidia-smi cannot confirm an NVIDIA render device",
    )
    return parser.parse_args()


ARGS = parse_args()
SCRIPT_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_RESULTS_DIR = SCRIPT_PACKAGE_ROOT / "results"


def gpu_preflight() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        result = None
        detail = str(exc)
    else:
        detail = (result.stdout or result.stderr).strip()
    if result is not None and result.returncode == 0 and result.stdout.strip():
        return
    report_path = ARGS.render_report or SCRIPT_RESULTS_DIR / "administration_learned_replay_render_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "blocked_gpu_driver_unavailable",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "renderer": ARGS.renderer,
                "preflight_command": "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
                "detail": detail,
                "scene_build_unaffected": True,
                "required_action": "restore/load the NVIDIA driver, verify nvidia-smi, then rerun this command",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    raise SystemExit(
        "RTX render blocked: nvidia-smi cannot confirm an NVIDIA driver/device. "
        f"Details were written to {report_path}"
    )


if not ARGS.skip_gpu_preflight:
    gpu_preflight()

from isaacsim import SimulationApp


APP = SimulationApp({"headless": ARGS.headless, "renderer": ARGS.renderer})

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, UsdGeom

from aisha_common import PACKAGE_ROOT, RESULTS_DIR, SCENES_DIR, ensure_output_dirs, sha256_file, write_json


if ARGS.renderer == "PathTracing":
    settings = carb.settings.get_settings()
    settings.set_int("/rtx/pathtracing/spp", max(1, ARGS.path_tracing_spp))
    settings.set_int("/rtx/pathtracing/totalSpp", max(1, ARGS.path_tracing_spp))


SHOTS = (
    {
        "title": "Departure - central atrium",
        "camera": (-4.35, 0.0, 1.72),
        "look_at": (2.0, 0.0, 0.72),
        "focal_length": 16.0,
        "cutaway": False,
        "segments": (0,),
    },
    {
        "title": "East administration hall - Vice-Principal approach",
        "camera": (6.0, 1.08, 1.75),
        "look_at": (15.0, -0.40, 0.72),
        "focal_length": 16.0,
        "cutaway": False,
        "segments": (1, 2),
    },
    {
        "title": "Visit 1 - Vice-Principal office",
        "camera": (19.55, -7.55, 1.86),
        "look_at": (16.75, -6.20, 0.70),
        "focal_length": 16.0,
        "cutaway": False,
        "segments": (3, 4),
    },
    {
        "title": "Return through the hall - Principal suite turn",
        "camera": (3.0, 1.00, 1.85),
        "look_at": (9.0, -0.45, 0.68),
        "focal_length": 16.0,
        "cutaway": False,
        "segments": (5, 6),
    },
    {
        "title": "Visit 2 - Principal office",
        "camera": (10.85, -9.78, 1.82),
        "look_at": (8.25, -9.00, 0.72),
        "focal_length": 15.0,
        "cutaway": False,
        "segments": (7, 8, 9),
    },
    {
        "title": "Mission complete - return to the atrium",
        "camera": (-4.35, 0.0, 1.72),
        "look_at": (2.50, -2.0, 0.72),
        "focal_length": 16.0,
        "cutaway": False,
        "segments": (10, 11),
    },
)


def load_verified_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"missing learned playback report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("outcome") != "success":
        raise ValueError(f"trajectory outcome must be success, got {report.get('outcome')!r}")
    if int(report.get("waypoints_completed", 0)) != 12:
        raise ValueError("trajectory must contain the verified 12-waypoint mission")
    trace = report.get("pose_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("trajectory report has no pose_trace")
    required = {"step", "elapsed_s", "x_m", "y_m", "yaw_rad", "segment_id", "control_mode"}
    for index, sample in enumerate(trace):
        if not isinstance(sample, dict) or not required.issubset(sample):
            raise ValueError(f"invalid pose_trace record {index}")
    return report, trace


def resample_trace(samples: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    """Select recorded poses without interpolating or inventing motion."""
    if not samples:
        raise ValueError("cannot resample an empty trace")
    if frame_count <= 1:
        return [samples[0]]
    indices = np.linspace(0, len(samples) - 1, frame_count)
    return [samples[int(round(index))] for index in indices]


def set_robot_pose(stage, sample: dict[str, Any]) -> None:
    prim = stage.GetPrimAtPath("/World/AISHA")
    prim.GetAttribute("xformOp:translate:route").Set(
        Gf.Vec3d(float(sample["x_m"]), float(sample["y_m"]), 0.0)
    )
    prim.GetAttribute("xformOp:rotateZ:route").Set(math.degrees(float(sample["yaw_rad"])))


def set_cutaway(stage, enabled: bool) -> None:
    prefixes = (
        "/World/Architecture/Ceilings",
        "/World/Architecture/Walls",
        "/World/Lighting/Panels",
        "/World/Lighting/Fixtures",
        "/World/Appearance/TimberSlats",
        "/World/Appearance/WallFinishes",
        "/World/Appearance/WallDisplays",
    )
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if not any(path.startswith(prefix) for prefix in prefixes):
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        imageable = UsdGeom.Imageable(prim)
        if enabled:
            imageable.MakeInvisible()
        else:
            imageable.MakeVisible()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def mode_label(mode: str) -> str:
    return {
        "learned_sensor_policy": "learned sensor policy",
        "physics_supervisor_turn": "physical turn supervisor",
        "presentation_dwell": "office dwell",
    }.get(mode, mode.replace("_", " "))


def add_overlay(image: Image.Image, title: str, shot_index: int, sample: dict[str, Any]) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    box_width = min(canvas.width - 52, 890)
    draw.rounded_rectangle((26, 24, 26 + box_width, 118), radius=14, fill=(8, 18, 25, 196))
    draw.text((48, 36), title, font=font(26), fill=(255, 255, 255, 255))
    evidence = (
        f"Verified learned trajectory replay | shot {shot_index}/{len(SHOTS)} | "
        f"segment {int(sample['segment_id']) + 1}/12 | {mode_label(str(sample['control_mode']))}"
    )
    draw.text((49, 75), evidence, font=font(16), fill=(142, 222, 199, 255))
    draw.rectangle((0, canvas.height - 42, canvas.width, canvas.height), fill=(8, 18, 25, 184))
    disclosure = "Recorded wheel-physics pose trace replayed in Omniverse • visual replay, not live policy execution"
    draw.text((26, canvas.height - 32), disclosure, font=font(15), fill=(232, 235, 237, 255))
    return canvas.convert("RGB")


def main() -> int:
    ensure_output_dirs()
    scene = SCENES_DIR / "administration.usd"
    trajectory_path = ARGS.trajectory_report or RESULTS_DIR / "isaaclab_learned_route_playback_report.json"
    frame_dir = ARGS.frame_directory or PACKAGE_ROOT / "media" / "learned_route_replay_frames"
    render_report_path = ARGS.render_report or RESULTS_DIR / "administration_learned_replay_render_report.json"
    source_report, trace = load_verified_trace(trajectory_path.resolve())

    if not scene.exists():
        raise FileNotFoundError(f"missing {scene}; run build_administration.py first")
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError(f"could not open {scene}")
    stage = omni.usd.get_context().get_stage()
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()

    frames_per_shot = max(12, round(ARGS.fps * ARGS.seconds_per_shot))
    frame_number = 0
    rendered_shots: list[dict[str, Any]] = []
    for shot_index, shot in enumerate(SHOTS, start=1):
        segment_ids = set(shot["segments"])
        source_samples = [sample for sample in trace if int(sample["segment_id"]) in segment_ids]
        if not source_samples:
            raise ValueError(f"no learned trace records for shot segments {sorted(segment_ids)}")
        samples = resample_trace(source_samples, frames_per_shot)
        set_cutaway(stage, bool(shot["cutaway"]))
        camera = rep.create.camera(
            position=shot["camera"],
            look_at=shot["look_at"],
            focal_length=shot["focal_length"],
        )
        render_product = rep.create.render_product(camera, (ARGS.width, ARGS.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(render_product)
        for _ in range(8):
            APP.update()
        for sample in samples:
            set_robot_pose(stage, sample)
            rep.orchestrator.step(delta_time=1.0 / ARGS.fps)
            rgba = np.asarray(rgb.get_data())
            if rgba.size == 0:
                raise RuntimeError(f"renderer returned no RGB data for shot {shot_index}")
            image = add_overlay(Image.fromarray(rgba).convert("RGB"), str(shot["title"]), shot_index, sample)
            image.save(frame_dir / f"frame_{frame_number:05d}.png", compress_level=2)
            frame_number += 1
        rgb.detach()
        render_product.destroy()
        rendered_shots.append(
            {
                "title": shot["title"],
                "segment_ids": list(shot["segments"]),
                "source_trace_records": len(source_samples),
                "rendered_frames": len(samples),
                "first_source_step": int(source_samples[0]["step"]),
                "last_source_step": int(source_samples[-1]["step"]),
                "first_elapsed_s": float(source_samples[0]["elapsed_s"]),
                "last_elapsed_s": float(source_samples[-1]["elapsed_s"]),
            }
        )
        print(f"rendered learned-trace shot {shot_index}/{len(SHOTS)}: {shot['title']}")

    set_cutaway(stage, False)

    report = {
        "status": "learned_trajectory_replay_frames_rendered",
        "scene": str(scene.resolve()),
        "scene_sha256": sha256_file(scene),
        "frame_directory": str(frame_dir.resolve()),
        "frame_count": frame_number,
        "fps": ARGS.fps,
        "renderer": ARGS.renderer,
        "path_tracing_spp": ARGS.path_tracing_spp if ARGS.renderer == "PathTracing" else None,
        "resolution": [ARGS.width, ARGS.height],
        "duration_s": frame_number / ARGS.fps,
        "shots": rendered_shots,
        "trajectory_report": str(trajectory_path.resolve()),
        "trajectory_report_sha256": sha256_file(trajectory_path.resolve()),
        "trajectory_outcome": source_report["outcome"],
        "trajectory_seed": source_report.get("seed"),
        "checkpoint": source_report.get("checkpoint"),
        "completed_steps": source_report.get("completed_steps"),
        "waypoints_completed": source_report.get("waypoints_completed"),
        "source_trace_record_count": len(trace),
        "source_trace_control_modes": dict(Counter(str(sample["control_mode"]) for sample in trace)),
        "motion": "recorded pose samples from the successful live Nav2/Isaac wheel-physics mission; no scripted route interpolation",
        "disclosure": "visual replay of a verified learned run in the presentation environment; not live policy execution",
        "claim_boundary": {
            "supported": (
                "PathTracing visual replay of recorded poses from the accepted "
                "live Nav2/Isaac wheel-physics mission"
            ),
            "visual_replay_is_live_policy_execution": False,
            "source_motion_was_live_policy_execution": True,
            "physical_localization_credit": False,
            "physical_release": False,
        },
    }
    write_json(render_report_path, report)
    print(f"wrote {frame_number} learned-replay frames to {frame_dir}")
    print(f"wrote evidence report to {render_report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
