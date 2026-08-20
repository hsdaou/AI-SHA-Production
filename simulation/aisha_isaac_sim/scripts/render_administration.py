#!/usr/bin/env python3
"""Render a deterministic overview of the disclosed administration proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image

from aisha_common import PACKAGE_ROOT, SCENES_DIR, ensure_output_dirs


def main() -> int:
    ensure_output_dirs()
    scene = SCENES_DIR / "administration.usd"
    if not scene.exists():
        raise FileNotFoundError(f"missing {scene}; run build_administration.py first")
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError(f"could not open {scene}")
    for _ in range(12):
        APP.update()

    camera = rep.create.camera(
        position=(4.5, -24.0, 20.0),
        look_at=(6.5, -0.4, 0.25),
        focal_length=26.0,
    )
    render_product = rep.create.render_product(camera, (ARGS.width, ARGS.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(render_product)
    rep.orchestrator.step(delta_time=0.0)
    rgba = np.asarray(rgb.get_data())
    if rgba.size == 0:
        raise RuntimeError("renderer returned no RGB data")
    output = PACKAGE_ROOT / "media" / "screenshots" / "administration_overview.png"
    Image.fromarray(rgba).convert("RGB").save(output)
    rgb.detach()
    render_product.destroy()
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
