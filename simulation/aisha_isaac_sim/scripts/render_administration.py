#!/usr/bin/env python3
"""Render plan-derived Block A overview and two office-visit stills."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

from aisha_common import PACKAGE_ROOT, SCENES_DIR, ensure_output_dirs


SHOTS = (
    {
        "name": "administration_overview.png",
        "position": (25.0, -29.0, 18.0),
        "look_at": (7.0, -3.0, 0.1),
        "focal_length": 30.0,
        "robot": (5.0, 0.0, 0.0),
        "hide_ceilings": True,
    },
    {
        "name": "administration_vice_principal.png",
        "position": (17.10, -2.75, 1.52),
        "look_at": (17.10, -6.05, 0.62),
        "focal_length": 30.0,
        "robot": (17.10, -5.70, -90.0),
        "hide_ceilings": False,
    },
    {
        "name": "administration_principal.png",
        "position": (5.35, -6.05, 1.52),
        "look_at": (7.95, -8.55, 0.62),
        "focal_length": 30.0,
        "robot": (7.80, -8.30, -45.0),
        "hide_ceilings": False,
    },
)


def set_robot_pose(stage, x: float, y: float, yaw_deg: float) -> None:
    prim = stage.GetPrimAtPath("/World/AISHA")
    if not prim.IsValid():
        raise RuntimeError("scene has no /World/AISHA prim")
    prim.GetAttribute("xformOp:translate:route").Set(Gf.Vec3d(x, y, 0.0))
    prim.GetAttribute("xformOp:rotateZ:route").Set(float(yaw_deg))


def set_group_visibility(stage, prefixes: tuple[str, ...], visible: bool) -> None:
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if not any(path.startswith(prefix) for prefix in prefixes):
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def main() -> int:
    ensure_output_dirs()
    scene = SCENES_DIR / "administration.usd"
    if not scene.exists():
        raise FileNotFoundError(f"missing {scene}; run build_administration.py first")
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError(f"could not open {scene}")
    stage = omni.usd.get_context().get_stage()
    output_dir = PACKAGE_ROOT / "media" / "screenshots"

    for shot in SHOTS:
        set_robot_pose(stage, *shot["robot"])
        set_group_visibility(
            stage,
            ("/World/Architecture/Ceilings", "/World/Architecture/Walls", "/World/Appearance/TimberSlats"),
            not bool(shot["hide_ceilings"]),
        )
        for _ in range(10):
            APP.update()
        camera = rep.create.camera(
            position=shot["position"],
            look_at=shot["look_at"],
            focal_length=shot["focal_length"],
        )
        render_product = rep.create.render_product(camera, (ARGS.width, ARGS.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(render_product)
        rep.orchestrator.step(delta_time=0.0)
        rgba = np.asarray(rgb.get_data())
        if rgba.size == 0:
            raise RuntimeError(f"renderer returned no RGB data for {shot['name']}")
        output = output_dir / shot["name"]
        Image.fromarray(rgba).convert("RGB").save(output, quality=95)
        rgb.detach()
        render_product.destroy()
        print(f"wrote {output}")

    set_group_visibility(
        stage,
        ("/World/Architecture/Ceilings", "/World/Architecture/Walls", "/World/Appearance/TimberSlats"),
        True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
