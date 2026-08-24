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
    parser.add_argument(
        "--renderer",
        choices=("RaytracedLighting", "PathTracing"),
        default="PathTracing",
    )
    parser.add_argument("--path-tracing-spp", type=int, default=64)
    parser.add_argument(
        "--shot",
        action="append",
        default=[],
        help="Render only the named PNG shot; repeat to select more than one.",
    )
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": ARGS.renderer})

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom

from aisha_common import PACKAGE_ROOT, SCENES_DIR, ensure_output_dirs


if ARGS.renderer == "PathTracing":
    settings = carb.settings.get_settings()
    settings.set_int("/rtx/pathtracing/spp", max(1, ARGS.path_tracing_spp))
    settings.set_int("/rtx/pathtracing/totalSpp", max(1, ARGS.path_tracing_spp))
    settings.set_float("/rtx/post/tonemap/exposureBias", 0.65)


SHOTS = (
    {
        "name": "administration_overview.png",
        "position": (22.0, -25.0, 15.0),
        "look_at": (8.0, -3.5, 0.25),
        "focal_length": 34.0,
        "robot": (5.0, 0.0, 0.0),
        "hide_ceilings": True,
    },
    {
        "name": "administration_atrium.png",
        "position": (4.20, -4.00, 1.72),
        "look_at": (-1.20, 1.10, 0.82),
        "focal_length": 18.0,
        "robot": (3.40, 0.00, 180.0),
        "hide_ceilings": False,
    },
    {
        "name": "administration_east_hall.png",
        "position": (6.15, 0.72, 1.55),
        "look_at": (14.40, -0.05, 0.82),
        "focal_length": 27.0,
        "robot": (14.30, 0.0, 0.0),
        "hide_ceilings": False,
    },
    {
        "name": "administration_principal_approach.png",
        "position": (2.60, -2.70, 1.82),
        "look_at": (6.80, -7.45, 0.78),
        "focal_length": 16.0,
        "robot": (6.42, -7.18, -45.0),
        "hide_ceilings": False,
    },
    {
        "name": "administration_vice_principal.png",
        "position": (19.55, -7.55, 1.86),
        "look_at": (16.75, -6.20, 0.70),
        "focal_length": 16.0,
        "robot": (17.10, -6.40, -90.0),
        "hide_ceilings": False,
    },
    {
        "name": "administration_principal.png",
        "position": (10.70, -6.10, 2.05),
        "look_at": (9.00, -9.80, 0.82),
        "focal_length": 12.0,
        "robot": (8.01, -8.66, -45.0),
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

    selected_shots = set(ARGS.shot)
    for shot in SHOTS:
        if selected_shots and shot["name"] not in selected_shots:
            continue
        set_robot_pose(stage, *shot["robot"])
        set_group_visibility(
            stage,
            (
                "/World/Architecture/Ceilings",
                "/World/Architecture/Walls",
                "/World/Appearance/TimberSlats",
                "/World/Appearance/WallFinishes",
                "/World/Appearance/WallDisplays",
            ),
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
        (
            "/World/Architecture/Ceilings",
            "/World/Architecture/Walls",
            "/World/Appearance/TimberSlats",
            "/World/Appearance/WallFinishes",
            "/World/Appearance/WallDisplays",
        ),
        True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
