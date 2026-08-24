#!/usr/bin/env python3
"""Render survey and human-height QA views of the complete Phase 7J twin."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=root / "scenes/phase7j_complete_captured_administration.usda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "media/screenshots/phase7j_complete_twin",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results/phase7j_complete_twin_static_render.json",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--renderer", choices=("RaytracedLighting", "PathTracing"), default="RaytracedLighting"
    )
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": ARGS.renderer})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import UsdGeom


SHOTS = (
    {
        "name": "complete_capture_overview.png",
        "title": "Complete captured administration - RoomPlan survey overview",
        "position": (6.0, -5.5, 39.0),
        "look_at": (6.0, -5.5, 0.0),
        "focal_length": 18.0,
    },
    {
        "name": "atrium_and_reception.png",
        "title": "Administration atrium and reception capture",
        "position": (-6.2, 5.0, 4.7),
        "look_at": (0.4, -0.8, 0.75),
        "focal_length": 14.0,
    },
    {
        "name": "east_administration_wing.png",
        "title": "Captured east administration office wing",
        "position": (6.5, 3.8, 4.0),
        "look_at": (15.0, -7.4, 0.70),
        "focal_length": 14.0,
    },
    {
        "name": "captured_office_cluster.png",
        "title": "Captured offices, furniture and doorway topology",
        "position": (22.0, -5.0, 4.4),
        "look_at": (15.0, -10.0, 0.70),
        "focal_length": 15.0,
    },
    {
        "name": "principal_registered_supplement.png",
        "title": "Registered Principal-office supplement",
        "position": (12.8, -4.4, 3.4),
        "look_at": (7.8, -8.9, 0.72),
        "focal_length": 14.0,
    },
    {
        "name": "accepted_route_context.png",
        "title": "AI-SHA accepted office mission in captured-area context",
        "position": (-4.8, 2.8, 6.3),
        "look_at": (8.6, -6.0, 0.55),
        "focal_length": 17.0,
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def overlay(image: Image.Image, title: str) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((24, 22, min(canvas.width - 24, 990), 111), 14, fill=(7, 17, 24, 206))
    draw.text((47, 35), title, font=font(25), fill=(255, 255, 255, 255))
    draw.text(
        (48, 76),
        "PHASE 7J | COMPLETE PRIMARY ROOMPLAN + REGISTERED PRINCIPAL SUPPLEMENT",
        font=font(15),
        fill=(136, 224, 195, 255),
    )
    draw.rectangle((0, canvas.height - 42, canvas.width, canvas.height), fill=(7, 17, 24, 190))
    draw.text(
        (24, canvas.height - 31),
        "Semantic captured-area twin • VP interior assumed (locked) • hidden validated navigation collision layer",
        font=font(14),
        fill=(236, 238, 240, 255),
    )
    return canvas.convert("RGB")


def main() -> int:
    scene = ARGS.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)
    context = omni.usd.get_context()
    if not context.open_stage(str(scene)):
        raise RuntimeError(f"could not open {scene}")
    for _ in range(30):
        APP.update()
    if context.get_stage() is None:
        raise RuntimeError("scene did not finish loading")

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for shot in SHOTS:
        route_prim = context.get_stage().GetPrimAtPath(
            "/World/RouteEvidence/AcceptedMissionTrace"
        )
        if route_prim.IsValid():
            route_imageable = UsdGeom.Imageable(route_prim)
            if shot["name"] == "accepted_route_context.png":
                route_imageable.MakeVisible()
            else:
                route_imageable.MakeInvisible()
        camera = rep.create.camera(
            position=shot["position"],
            look_at=shot["look_at"],
            focal_length=shot["focal_length"],
        )
        product = rep.create.render_product(camera, (ARGS.width, ARGS.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        for _ in range(12):
            APP.update()
        rep.orchestrator.step(delta_time=0.0)
        rgba = np.asarray(rgb.get_data())
        if rgba.size == 0:
            raise RuntimeError(f"renderer returned no RGB data for {shot['name']}")
        output = ARGS.output_dir / shot["name"]
        overlay(Image.fromarray(rgba).convert("RGB"), shot["title"]).save(
            output, quality=95
        )
        rendered.append(
            {
                "name": shot["name"],
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
                "camera": list(shot["position"]),
                "look_at": list(shot["look_at"]),
            }
        )
        rgb.detach()
        product.destroy()
        print(f"wrote {output}")

    report = {
        "report_type": "phase7j_complete_twin_static_render",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rendered",
        "passed": len(rendered) == len(SHOTS),
        "scene": str(scene),
        "scene_sha256": sha256_file(scene),
        "renderer": ARGS.renderer,
        "resolution": [ARGS.width, ARGS.height],
        "shots": rendered,
    }
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
