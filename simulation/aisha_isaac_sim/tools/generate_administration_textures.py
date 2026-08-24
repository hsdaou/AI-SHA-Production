#!/usr/bin/env python3
"""Generate deterministic, seamless-enough presentation textures for Block A.

The textures are deliberately procedural and project-owned.  They reproduce
the material families visible in the supplied walkthrough without copying any
photographs, logos, artwork, or personally identifying content from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_ROOT / "textures" / "administration"
SIZE = 1024
SEED = 20260822


def _save_rgb(name: str, array: np.ndarray) -> Path:
    output = OUTPUT_DIR / name
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB").save(
        output, optimize=True
    )
    return output


def _save_grey(name: str, array: np.ndarray) -> Path:
    output = OUTPUT_DIR / name
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "L").save(
        output, optimize=True
    )
    return output


def _save_normal(name: str, height: np.ndarray, strength: float) -> Path:
    """Convert a seamless scalar height signal into an OpenGL tangent normal."""
    height = np.asarray(height, dtype=np.float32)
    height = (height - height.mean()) / max(float(height.std()), 1.0e-6)
    gradient_y, gradient_x = np.gradient(height)
    normal = np.stack(
        (-gradient_x * strength, -gradient_y * strength, np.ones_like(height)),
        axis=-1,
    )
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(1.0e-6)
    encoded = (normal * 0.5 + 0.5) * 255.0
    return _save_rgb(name, encoded)


def _wrapped_ellipse(
    draw: ImageDraw.ImageDraw,
    centre: tuple[int, int],
    radius: tuple[int, int],
    fill: tuple[int, int, int],
) -> None:
    """Draw an aggregate chip across tile boundaries when necessary."""
    cx, cy = centre
    rx, ry = radius
    for ox in (-SIZE, 0, SIZE):
        for oy in (-SIZE, 0, SIZE):
            x, y = cx + ox, cy + oy
            if x + rx < 0 or x - rx >= SIZE or y + ry < 0 or y - ry >= SIZE:
                continue
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=fill)


def terrazzo(rng: np.random.Generator) -> tuple[Path, Path, Path]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    base = np.array([157.0, 160.0, 158.0])
    broad = 2.6 * np.sin(2.0 * np.pi * xx / SIZE) + 1.8 * np.cos(2.0 * np.pi * yy / SIZE)
    fine = rng.normal(0.0, 2.2, (SIZE, SIZE))
    image = np.clip(base[None, None, :] + (broad + fine)[..., None], 0, 255).astype(np.uint8)
    texture = Image.fromarray(image, "RGB")
    draw = ImageDraw.Draw(texture)
    palette = (
        (224, 224, 215),
        (195, 192, 181),
        (102, 108, 106),
        (66, 70, 69),
        (141, 126, 109),
        (121, 139, 129),
    )
    for _ in range(1850):
        centre = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        radius = (int(rng.integers(1, 6)), int(rng.integers(1, 5)))
        _wrapped_ellipse(draw, centre, radius, palette[int(rng.integers(0, len(palette)))])
    for _ in range(230):
        centre = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        radius = (int(rng.integers(6, 18)), int(rng.integers(4, 14)))
        _wrapped_ellipse(draw, centre, radius, palette[int(rng.integers(0, len(palette)))])
    texture = texture.filter(ImageFilter.GaussianBlur(radius=0.25))
    albedo = OUTPUT_DIR / "terrazzo_albedo.png"
    texture.save(albedo, optimize=True)

    roughness = np.full((SIZE, SIZE), 43.0) + rng.normal(0.0, 3.0, (SIZE, SIZE))
    roughness += 3.0 * np.sin(2.0 * np.pi * xx / SIZE)
    return (
        albedo,
        _save_grey("terrazzo_roughness.png", roughness),
        _save_normal("terrazzo_normal.png", np.asarray(texture.convert("L")), 0.12),
    )


def walnut(rng: np.random.Generator) -> tuple[Path, Path, Path]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    phase = 0.55 * np.sin(2.0 * np.pi * yy / SIZE) + 0.19 * np.sin(6.0 * np.pi * yy / SIZE)
    grain = (
        12.0 * np.sin(2.0 * np.pi * (xx / 92.0 + phase))
        + 6.0 * np.sin(2.0 * np.pi * (xx / 31.0 - phase * 0.4))
        + 3.0 * np.sin(2.0 * np.pi * xx / 9.0)
    )
    grain += rng.normal(0.0, 1.8, (SIZE, SIZE))
    base = np.array([70.0, 31.0, 17.0])
    tint = np.array([1.00, 0.58, 0.34])
    image = base[None, None, :] + grain[..., None] * tint[None, None, :]
    albedo = _save_rgb("walnut_albedo.png", image)
    roughness = 76.0 + 7.0 * np.sin(2.0 * np.pi * xx / 92.0) + rng.normal(0.0, 2.0, (SIZE, SIZE))
    return (
        albedo,
        _save_grey("walnut_roughness.png", roughness),
        _save_normal("walnut_normal.png", grain, 0.20),
    )


def oak(rng: np.random.Generator) -> tuple[Path, Path, Path]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    plank_height = 128
    plank = yy // plank_height
    plank_tint = np.array([7.0, -3.0, 4.0, -7.0, 2.0, 6.0, -4.0, 0.0])[plank]
    waviness = 0.30 * np.sin(2.0 * np.pi * yy / SIZE) + 0.10 * np.sin(8.0 * np.pi * yy / SIZE)
    grain = 7.0 * np.sin(2.0 * np.pi * (xx / 118.0 + waviness))
    grain += 3.0 * np.sin(2.0 * np.pi * xx / 31.0)
    grain += rng.normal(0.0, 1.3, (SIZE, SIZE))
    base = np.array([181.0, 158.0, 123.0])
    image = base[None, None, :] + (grain + plank_tint)[..., None] * np.array([1.0, 0.78, 0.50])
    seam = (yy % plank_height <= 2) | (yy % plank_height >= plank_height - 2)
    image[seam] *= 0.68
    albedo = _save_rgb("oak_albedo.png", image)
    roughness = 103.0 + 5.0 * np.sin(2.0 * np.pi * xx / 118.0) + rng.normal(0.0, 2.0, (SIZE, SIZE))
    roughness[seam] = 125.0
    height = grain + plank_tint
    height[seam] -= 12.0
    return (
        albedo,
        _save_grey("oak_roughness.png", roughness),
        _save_normal("oak_normal.png", height, 0.18),
    )


def grey_oak(rng: np.random.Generator) -> tuple[Path, Path, Path]:
    """Grey wood-plank finish observed throughout the Principal suite."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    plank_height = 128
    plank = yy // plank_height
    plank_tint = np.array([3.0, -5.0, 2.0, -3.0, 4.0, 0.0, -4.0, 3.0])[plank]
    waviness = 0.27 * np.sin(2.0 * np.pi * yy / SIZE) + 0.08 * np.sin(
        10.0 * np.pi * yy / SIZE
    )
    grain = 6.0 * np.sin(2.0 * np.pi * (xx / 126.0 + waviness))
    grain += 2.5 * np.sin(2.0 * np.pi * xx / 37.0)
    grain += rng.normal(0.0, 1.15, (SIZE, SIZE))
    base = np.array([166.0, 164.0, 158.0])
    image = base[None, None, :] + (grain + plank_tint)[..., None] * np.array(
        [0.83, 0.78, 0.68]
    )
    seam = (yy % plank_height <= 2) | (yy % plank_height >= plank_height - 2)
    image[seam] *= 0.70
    albedo = _save_rgb("grey_oak_albedo.png", image)
    roughness = 111.0 + 5.0 * np.sin(2.0 * np.pi * xx / 126.0)
    roughness += rng.normal(0.0, 1.8, (SIZE, SIZE))
    roughness[seam] = 132.0
    height = grain + plank_tint
    height[seam] -= 11.0
    return (
        albedo,
        _save_grey("grey_oak_roughness.png", roughness),
        _save_normal("grey_oak_normal.png", height, 0.16),
    )


def mottled_paint(rng: np.random.Generator) -> tuple[Path, Path, Path]:
    small = rng.normal(0.0, 1.0, (128, 128))
    broad = np.asarray(
        Image.fromarray(((small - small.min()) / np.ptp(small) * 255.0).astype(np.uint8), "L")
        .resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        .filter(ImageFilter.GaussianBlur(radius=9.0)),
        dtype=np.float32,
    )
    broad = (broad - broad.mean()) / max(1.0, broad.std())
    fine = rng.normal(0.0, 1.6, (SIZE, SIZE))
    value = broad * 4.0 + fine
    image = np.array([165.0, 169.0, 170.0])[None, None, :] + value[..., None]
    albedo = _save_rgb("mottled_grey_albedo.png", image)
    return (
        albedo,
        _save_grey("mottled_grey_roughness.png", 145.0 + broad * 5.0),
        _save_normal("mottled_grey_normal.png", broad + fine * 0.25, 0.05),
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    outputs = [
        *terrazzo(rng),
        *walnut(rng),
        *oak(rng),
        *mottled_paint(rng),
        *grey_oak(rng),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
