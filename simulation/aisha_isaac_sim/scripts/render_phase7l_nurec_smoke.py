#!/usr/bin/env python3
"""Render authored camera keyframes from the Phase 7L NuRec administration asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "media/screenshots/phase7l_nurec_smoke",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results/phase7l_nurec_isaac_render.json",
    )
    parser.add_argument("--frames", default="0,60,120,180,240,309")
    parser.add_argument("--camera", default="/World/gauss/Cameras/camera_0")
    parser.add_argument("--width", type=int, default=540)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
        # NuRec currently supports one render device per process.  Set this at
        # launch as well as in Carb settings so Kit never starts in multi-GPU
        # mode before the stage is opened.
        "multi_gpu": False,
    }
)

import carb
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def main() -> int:
    print("PHASE7L: main entered", flush=True)
    stage_path = ARGS.stage.expanduser().resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    frame_indices = [int(value.strip()) for value in ARGS.frames.split(",") if value.strip()]
    if not frame_indices:
        raise ValueError("--frames must contain at least one integer time code")

    settings = carb.settings.get_settings()
    settings.set_bool("/renderer/multiGpu/enabled", False)
    settings.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)

    context = omni.usd.get_context()
    print(f"PHASE7L: opening {stage_path}", flush=True)
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"could not open NuRec stage: {stage_path}")
    print("PHASE7L: stage opened; warming renderer", flush=True)
    for _ in range(max(ARGS.warmup, 12)):
        APP.update()

    stage = context.get_stage()
    print("PHASE7L: renderer warm-up complete", flush=True)
    if stage is None:
        raise RuntimeError("NuRec stage did not finish loading")
    camera_prim = stage.GetPrimAtPath(ARGS.camera)
    if not camera_prim.IsValid():
        raise RuntimeError(f"authored camera does not exist: {ARGS.camera}")

    time_codes_per_second = float(stage.GetTimeCodesPerSecond() or 24.0)
    start = float(stage.GetStartTimeCode())
    end = float(stage.GetEndTimeCode())
    for frame in frame_indices:
        if not start <= frame <= end:
            raise ValueError(f"frame {frame} is outside the authored range {start}..{end}")

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"PHASE7L: creating render product for {ARGS.camera}", flush=True)
    product = rep.create.render_product(ARGS.camera, (ARGS.width, ARGS.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(product)
    timeline = omni.timeline.get_timeline_interface()
    rendered: list[dict[str, object]] = []
    try:
        for frame in frame_indices:
            print(f"PHASE7L: rendering time code {frame}", flush=True)
            timeline.set_current_time(frame / time_codes_per_second)
            for _ in range(max(ARGS.warmup, 1)):
                APP.update()
            rep.orchestrator.step(delta_time=0.0)
            print("PHASE7L: render step complete", flush=True)
            rgba = np.asarray(rgb.get_data())
            print(
                f"PHASE7L: RGB shape={rgba.shape} size={rgba.size} dtype={rgba.dtype}",
                flush=True,
            )
            if rgba.size == 0:
                raise RuntimeError(f"renderer returned no RGB data at frame {frame}")
            output = ARGS.output_dir / f"nurec_camera_frame_{frame:04d}.png"
            Image.fromarray(rgba).convert("RGB").save(output)
            rendered.append(
                {
                    "time_code": frame,
                    "path": portable_path(output),
                    "sha256": sha256_file(output),
                    "mean_rgb": [float(value) for value in rgba[..., :3].mean(axis=(0, 1))],
                    "std_rgb": [float(value) for value in rgba[..., :3].std(axis=(0, 1))],
                }
            )
            print(f"wrote {output}")
    finally:
        rgb.detach()
        product.destroy()

    non_blank = all(max(item["std_rgb"]) > 5.0 for item in rendered)
    report = {
        "report_type": "phase7l_nurec_isaac_render",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rendered" if non_blank else "blank_render_detected",
        "passed": bool(non_blank and len(rendered) == len(frame_indices)),
        "stage": portable_path(stage_path),
        "stage_sha256": sha256_file(stage_path),
        "camera": ARGS.camera,
        "authored_time_range": [start, end],
        "time_codes_per_second": time_codes_per_second,
        "renderer": "RaytracedLighting",
        "single_gpu": True,
        "gaussian_tonemapping_enabled": True,
        "resolution": [ARGS.width, ARGS.height],
        "frames": rendered,
    }
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ARGS.report}")
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
