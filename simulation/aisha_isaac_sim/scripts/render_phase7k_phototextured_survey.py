#!/usr/bin/env python3
"""Render raw-evidence and clean phototextured QA views of the Phase 7K survey."""

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
        default=root / "scenes/phase7k_phototextured_photogrammetric_survey.usda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "media/screenshots/phase7k_phototextured_survey",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results/phase7k_phototextured_survey_static_render.json",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--renderer", choices=("RaytracedLighting", "PathTracing"), default="PathTracing"
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
        "name": "raw_atrium_corridor_cluster.png",
        "title": "Raw dense capture evidence - atrium and corridor cluster",
        "position": (-2.8, 5.1, 2.8),
        "look_at": (1.8, -0.4, 0.75),
        "focal_length": 18.0,
        "mode": "raw_atrium",
    },
    {
        "name": "raw_principal_office_cluster.png",
        "title": "Raw dense capture evidence - Principal office cluster",
        "position": (12.2, -3.7, 3.0),
        "look_at": (7.6, -8.3, 1.05),
        "focal_length": 18.0,
        "mode": "raw_principal",
    },
    {
        "name": "phototextured_atrium_reception.png",
        "title": "Phototextured metric twin - atrium and reception",
        "position": (-6.2, 5.0, 4.7),
        "look_at": (0.4, -0.8, 0.75),
        "focal_length": 14.0,
        "mode": "clean",
    },
    {
        "name": "phototextured_east_hall.png",
        "title": "Phototextured east administration hallway",
        "position": (6.0, 1.14, 1.80),
        "look_at": (15.1, -0.15, 0.70),
        "focal_length": 15.0,
        "mode": "clean",
    },
    {
        "name": "phototextured_principal_office.png",
        "title": "Principal office - registered geometry and captured finishes",
        "position": (10.0, -5.0, 4.0),
        "look_at": (7.2, -7.8, 0.55),
        "focal_length": 18.0,
        "mode": "clean",
    },
    {
        "name": "phototextured_complete_overview.png",
        "title": "Hybrid metric phototextured administration survey",
        "position": (6.0, -5.5, 39.0),
        "look_at": (6.0, -5.5, 0.0),
        "focal_length": 18.0,
        "mode": "clean",
    },
    {
        "name": "accepted_route_context.png",
        "title": "Accepted AI-SHA mission over the frozen navigation layer",
        "position": (-4.8, 2.8, 6.3),
        "look_at": (8.6, -6.0, 0.55),
        "focal_length": 17.0,
        "mode": "route",
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def overlay(image: Image.Image, title: str, mode: str) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((24, 22, min(canvas.width - 24, 1125), 111), 14, fill=(7, 17, 24, 210))
    draw.text((47, 35), title, font=font(25), fill=(255, 255, 255, 255))
    label = (
        "PHASE 7K | GENUINE PRIVACY-SCREENED DENSE PHOTOGRAMMETRY EVIDENCE"
        if mode.startswith("raw_")
        else "PHASE 7K | CAPTURE-DERIVED PBR + METRIC ROOMPLAN/LIDAR GEOMETRY"
    )
    draw.text((48, 76), label, font=font(15), fill=(136, 224, 195, 255))
    draw.rectangle((0, canvas.height - 42, canvas.width, canvas.height), fill=(7, 17, 24, 194))
    footer = (
        "Incomplete captured cluster • provisional visual registration • excluded from collision"
        if mode.startswith("raw_")
        else "Frozen validated navigation collision • VP interior assumed (locked) • presentation visual layer"
    )
    draw.text((24, canvas.height - 31), footer, font=font(14), fill=(236, 238, 240, 255))
    return canvas.convert("RGB")


def set_visible(stage, path: str, visible: bool) -> None:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


def configure_mode(stage, mode: str) -> None:
    raw = mode.startswith("raw_")
    for path in (
        "/World/CapturedAdministration",
        "/World/PlanAuthorityFloors",
        "/World/PhotoSurfaceOverlays",
        "/World/Architecture",
        "/World/Furniture",
        "/World/AISHA",
    ):
        set_visible(stage, path, not raw)
    set_visible(stage, "/World/PhotogrammetricSurvey", raw)
    set_visible(
        stage,
        "/World/PhotogrammetricSurvey/AtriumCorridorCluster",
        mode == "raw_atrium",
    )
    set_visible(
        stage,
        "/World/PhotogrammetricSurvey/PrincipalOfficeCluster",
        mode == "raw_principal",
    )
    set_visible(stage, "/World/RouteEvidence/AcceptedMissionTrace", mode == "route")


def main() -> int:
    scene = ARGS.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)
    context = omni.usd.get_context()
    if not context.open_stage(str(scene)):
        raise RuntimeError(f"could not open {scene}")
    for _ in range(30):
        APP.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("scene did not finish loading")

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for shot in SHOTS:
        configure_mode(stage, shot["mode"])
        camera = rep.create.camera(
            position=shot["position"],
            look_at=shot["look_at"],
            focal_length=shot["focal_length"],
        )
        product = rep.create.render_product(camera, (ARGS.width, ARGS.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        for _ in range(16):
            APP.update()
        rep.orchestrator.step(delta_time=0.0)
        rgba = np.asarray(rgb.get_data())
        if rgba.size == 0:
            raise RuntimeError(f"renderer returned no RGB data for {shot['name']}")
        output = ARGS.output_dir / shot["name"]
        overlay(Image.fromarray(rgba).convert("RGB"), shot["title"], shot["mode"]).save(
            output, quality=95
        )
        rendered.append(
            {
                "name": shot["name"],
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
                "camera": list(shot["position"]),
                "look_at": list(shot["look_at"]),
                "mode": shot["mode"],
            }
        )
        rgb.detach()
        product.destroy()
        print(f"wrote {output}")

    report = {
        "report_type": "phase7k_phototextured_survey_static_render",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rendered",
        "passed": len(rendered) == len(SHOTS),
        "scene": str(scene),
        "scene_sha256": sha256_file(scene),
        "renderer": ARGS.renderer,
        "resolution": [ARGS.width, ARGS.height],
        "raw_dense_evidence_views": sum(shot["mode"].startswith("raw_") for shot in SHOTS),
        "clean_presentation_views": sum(not shot["mode"].startswith("raw_") for shot in SHOTS),
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
