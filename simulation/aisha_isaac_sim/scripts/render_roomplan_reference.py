#!/usr/bin/env python3
"""Render privacy-safe semantic RoomPlan reference views for scene registration.

The source USDZ remains outside the repository.  The generated views use flat
category colours and contain no camera imagery, documents, portraits or GPS
metadata; they are intended for geometry-registration QA only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
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
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


CATEGORY_COLOURS = {
    "Wall": (0.70, 0.70, 0.72),
    "Door": (0.10, 0.68, 0.62),
    "Window": (0.18, 0.52, 0.82),
    "Floor": (0.50, 0.52, 0.54),
    "Chair": (0.035, 0.045, 0.055),
    "Table": (0.30, 0.10, 0.045),
    "Storage": (0.52, 0.21, 0.07),
    "Television": (0.015, 0.018, 0.025),
}

SHOTS = (
    {
        "name": "roomplan_overview.png",
        "position": (6.8, -8.0, 42.0),
        "look_at": (6.8, -8.0, 0.0),
        "focal_length": 26.0,
    },
)


def material(stage, name: str, colour: tuple[float, float, float]) -> UsdShade.Material:
    result = UsdShade.Material.Define(stage, f"/ReferenceMaterials/{name}")
    shader = UsdShade.Shader.Define(stage, f"/ReferenceMaterials/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.62)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def category_name(value: object) -> str | None:
    text = str(value or "")
    for name in CATEGORY_COLOURS:
        if text.startswith(name):
            return name
    return None


def main() -> int:
    source = ARGS.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not omni.usd.get_context().open_stage(str(source)):
        raise RuntimeError(f"could not open RoomPlan source: {source}")
    stage = omni.usd.get_context().get_stage()
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError("RoomPlan source has no default prim")

    # Apple RoomPlan is Y-up. Rotate its root into the Z-up Isaac convention
    # without scaling or altering any semantic dimensions.
    UsdGeom.Xformable(root).AddRotateXOp(opSuffix="roomplanToIsaac").Set(90.0)
    materials = {
        name: material(stage, name, colour) for name, colour in CATEGORY_COLOURS.items()
    }
    category_counts = {name: 0 for name in CATEGORY_COLOURS}
    for prim in stage.TraverseAll():
        category = category_name(prim.GetCustomDataByKey("Category"))
        if category is None:
            continue
        category_counts[category] += 1
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            materials[category],
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    dome = UsdLux.DomeLight.Define(stage, "/ReferenceLighting/Dome")
    dome.CreateIntensityAttr(650.0)
    key = UsdLux.DistantLight.Define(stage, "/ReferenceLighting/Key")
    key.CreateIntensityAttr(2400.0)
    key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 28.0, 18.0))

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        APP.update()
    for shot in SHOTS:
        camera = rep.create.camera(
            position=shot["position"],
            look_at=shot["look_at"],
            focal_length=shot["focal_length"],
        )
        product = rep.create.render_product(camera, (ARGS.width, ARGS.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        rep.orchestrator.step(delta_time=0.0)
        rgba = np.asarray(rgb.get_data())
        if rgba.size == 0:
            raise RuntimeError(f"renderer returned no RGB data for {shot['name']}")
        output = ARGS.output_dir / shot["name"]
        Image.fromarray(rgba).convert("RGB").save(output, quality=95)
        rgb.detach()
        product.destroy()
        print(f"wrote {output}")
    print(f"category_counts={category_counts}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
