#!/usr/bin/env python3
"""Render a four-shot Isaac Sim cinematic of AI-SHA visiting both offices."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-shot", type=float, default=3.0)
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, UsdGeom

from aisha_common import PACKAGE_ROOT, RESULTS_DIR, SCENES_DIR, ensure_output_dirs, write_json


SHOTS = (
    {
        "title": "Block A - departing the central atrium",
        "camera": (13.0, -12.0, 8.0),
        "look_at": (6.5, -0.5, 0.25),
        "focal_length": 30.0,
        "cutaway": True,
        "poses": [(0.0, 0.0, 0.0), (5.2, 0.0, 0.0), (10.5, 0.0, 0.0)],
    },
    {
        "title": "Stop 1 - Vice-Principal office (east cluster)",
        "camera": (17.10, -2.75, 1.52),
        "look_at": (17.10, -6.05, 0.64),
        "focal_length": 30.0,
        "cutaway": False,
        "poses": [(13.0, 0.0, 0.0), (17.1, 0.0, -90.0), (17.1, -3.8, -90.0), (17.1, -6.25, -90.0)],
    },
    {
        "title": "Plan-derived transfer to the angled Principal suite",
        "camera": (20.0, -15.0, 9.0),
        "look_at": (10.0, -3.5, 0.25),
        "focal_length": 31.0,
        "cutaway": True,
        "poses": [(17.1, -3.8, 90.0), (17.1, 0.0, 180.0), (10.2, 0.0, 180.0), (5.1, -3.8, -45.0), (6.45, -6.95, -45.0)],
    },
    {
        "title": "Stop 2 - Principal office (south-east angled room)",
        "camera": (5.35, -6.05, 1.52),
        "look_at": (7.95, -8.55, 0.64),
        "focal_length": 30.0,
        "cutaway": False,
        "poses": [(6.45, -6.95, -45.0), (6.97, -7.62, -45.0), (8.65, -9.10, -45.0)],
    },
)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def shortest_yaw(start: float, end: float, value: float) -> float:
    delta = (end - start + 180.0) % 360.0 - 180.0
    return start + delta * value


def sample_poses(poses: list[tuple[float, float, float]], frame_count: int) -> list[tuple[float, float, float]]:
    lengths = []
    for first, second in zip(poses, poses[1:]):
        lengths.append(max(0.35, math.hypot(second[0] - first[0], second[1] - first[1])))
    total = sum(lengths)
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    sampled = []
    dwell_start = int(frame_count * 0.84)
    movement_frames = max(2, dwell_start)
    for frame in range(frame_count):
        if frame >= movement_frames:
            sampled.append(poses[-1])
            continue
        progress = smoothstep(frame / max(1, movement_frames - 1)) * total
        segment = min(len(lengths) - 1, next((index for index, end in enumerate(cumulative[1:]) if progress <= end), len(lengths) - 1))
        local = (progress - cumulative[segment]) / lengths[segment]
        first, second = poses[segment], poses[segment + 1]
        sampled.append(
            (
                first[0] + (second[0] - first[0]) * local,
                first[1] + (second[1] - first[1]) * local,
                shortest_yaw(first[2], second[2], local),
            )
        )
    return sampled


def set_robot_pose(stage, pose: tuple[float, float, float]) -> None:
    prim = stage.GetPrimAtPath("/World/AISHA")
    prim.GetAttribute("xformOp:translate:route").Set(Gf.Vec3d(pose[0], pose[1], 0.0))
    prim.GetAttribute("xformOp:rotateZ:route").Set(float(pose[2]))


def set_cutaway(stage, enabled: bool) -> None:
    prefixes = ("/World/Architecture/Ceilings", "/World/Architecture/Walls", "/World/Appearance/TimberSlats")
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


def add_overlay(image: Image.Image, title: str, shot_index: int) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((26, 24, 770, 108), radius=14, fill=(8, 18, 25, 190))
    draw.text((48, 36), title, font=font(27), fill=(255, 255, 255, 255))
    draw.text((49, 72), f"AI-SHA scripted Isaac Sim presentation | shot {shot_index}/4", font=font(17), fill=(142, 222, 199, 255))
    draw.rectangle((0, canvas.height - 36, canvas.width, canvas.height), fill=(8, 18, 25, 175))
    draw.text((26, canvas.height - 29), "Door clearances and thresholds are presentation assumptions - physical route not released", font=font(15), fill=(232, 235, 237, 255))
    return canvas.convert("RGB")


def main() -> int:
    ensure_output_dirs()
    scene = SCENES_DIR / "administration.usd"
    if not scene.exists():
        raise FileNotFoundError(f"missing {scene}; run build_administration.py first")
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError(f"could not open {scene}")
    stage = omni.usd.get_context().get_stage()
    frame_dir = PACKAGE_ROOT / "media" / "route_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()

    frames_per_shot = max(12, round(ARGS.fps * ARGS.seconds_per_shot))
    frame_number = 0
    for shot_index, shot in enumerate(SHOTS, start=1):
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
        poses = sample_poses(list(shot["poses"]), frames_per_shot)
        for pose in poses:
            set_robot_pose(stage, pose)
            rep.orchestrator.step(delta_time=1.0 / ARGS.fps)
            rgba = np.asarray(rgb.get_data())
            if rgba.size == 0:
                raise RuntimeError(f"renderer returned no RGB data for shot {shot_index}")
            image = add_overlay(Image.fromarray(rgba).convert("RGB"), str(shot["title"]), shot_index)
            image.save(frame_dir / f"frame_{frame_number:05d}.png", compress_level=2)
            frame_number += 1
        rgb.detach()
        render_product.destroy()
        print(f"rendered shot {shot_index}/{len(SHOTS)}: {shot['title']}")

    set_cutaway(stage, False)

    report = {
        "status": "route_frames_rendered",
        "scene": str(scene),
        "frame_directory": str(frame_dir),
        "frame_count": frame_number,
        "fps": ARGS.fps,
        "resolution": [ARGS.width, ARGS.height],
        "duration_s": frame_number / ARGS.fps,
        "shots": [shot["title"] for shot in SHOTS],
        "motion": "scripted plan-aligned presentation transform, not Nav2 or physical validation",
    }
    write_json(RESULTS_DIR / "administration_render_report.json", report)
    print(f"wrote {frame_number} frames to {frame_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
