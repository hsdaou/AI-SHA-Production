#!/usr/bin/env python3
"""Generate auditable, privacy-safe PBR materials from the supplied survey stills.

Only close surface crops are retained.  The source stills remain outside the
repository and the recipes deliberately avoid names, paperwork, portraits,
screens and complete-room photographs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "tmp/phase7k_stills"
DEFAULT_OUTPUT = ROOT / "textures/phase7k_capture"


@dataclass(frozen=True)
class Recipe:
    source: str
    points: tuple[tuple[float, float], ...]
    roughness: int
    normal_strength: float
    note: str


# Normalized, privacy-reviewed quadrilaterals in the 3024 x 4032 review JPEGs.
# The points are ordered top-left, top-right, bottom-right, bottom-left.
RECIPES = {
    "atrium_terrazzo": Recipe(
        "IMG_6702.jpg",
        ((0.05, 0.64), (0.45, 0.64), (0.45, 0.70), (0.05, 0.70)),
        112,
        1.55,
        "speckled atrium floor away from furniture and wall graphics",
    ),
    "hall_terrazzo": Recipe(
        "IMG_6704.jpg",
        ((0.16, 0.78), (0.62, 0.78), (0.64, 0.89), (0.13, 0.89)),
        118,
        1.45,
        "office-hall terrazzo below all door signs",
    ),
    "grey_door": Recipe(
        "IMG_6705.jpg",
        ((0.31, 0.48), (0.45, 0.48), (0.45, 0.73), (0.31, 0.73)),
        132,
        1.20,
        "grey wood door surface excluding the name plate and hardware",
    ),
    "office_wall": Recipe(
        "IMG_6704.jpg",
        ((0.65, 0.13), (0.78, 0.13), (0.78, 0.26), (0.65, 0.26)),
        188,
        0.52,
        "plain off-white partition infill without signage",
    ),
    "principal_grey_oak": Recipe(
        "IMG_6707.jpg",
        ((0.40, 0.82), (0.90, 0.78), (0.98, 0.995), (0.30, 0.995)),
        126,
        1.85,
        "Principal-office grey plank floor",
    ),
    "principal_walnut": Recipe(
        "IMG_6708.jpg",
        ((0.05, 0.15), (0.18, 0.17), (0.18, 0.23), (0.05, 0.21)),
        94,
        1.35,
        "Principal-office walnut cabinet surface",
    ),
    "principal_green": Recipe(
        "IMG_6708.jpg",
        ((0.13, 0.62), (0.19, 0.62), (0.19, 0.69), (0.13, 0.69)),
        142,
        0.45,
        "Principal-office muted-green cabinet fronts without awards",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/phase7k_capture_materials_report.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rectify(image: np.ndarray, points: tuple[tuple[float, float], ...]) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32([(x * width, y * height) for x, y in points])
    destination = np.float32([[0, 0], [767, 0], [767, 767], [0, 767]])
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (768, 768),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def suppress_lighting(image: np.ndarray, strength: float = 0.60) -> np.ndarray:
    linear = image.astype(np.float32) / 255.0
    illumination = cv2.GaussianBlur(linear, (0, 0), 54.0)
    mean = illumination.mean(axis=(0, 1), keepdims=True)
    corrected = linear * (mean / np.maximum(illumination, 0.075)) ** strength
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def seamless_tile(image: np.ndarray, size: int = 1024) -> np.ndarray:
    """Make opposite edges agree while preserving the middle of the crop."""
    base = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
    result = base.astype(np.float32)
    blend = size // 9
    for index in range(blend):
        weight = index / max(1, blend - 1)
        average = 0.5 * (
            base[:, index].astype(np.float32) + base[:, -1 - index].astype(np.float32)
        )
        result[:, index] = average * (1.0 - weight) + result[:, index] * weight
        result[:, -1 - index] = average * (1.0 - weight) + result[:, -1 - index] * weight
    horizontal = np.clip(result, 0, 255).astype(np.uint8)
    for index in range(blend):
        weight = index / max(1, blend - 1)
        average = 0.5 * (
            horizontal[index].astype(np.float32)
            + horizontal[-1 - index].astype(np.float32)
        )
        result[index] = average * (1.0 - weight) + result[index] * weight
        result[-1 - index] = average * (1.0 - weight) + result[-1 - index] * weight
    return np.clip(result, 0, 255).astype(np.uint8)


def pbr_maps(
    albedo: np.ndarray, *, roughness: int, normal_strength: float
) -> tuple[np.ndarray, np.ndarray]:
    grey = cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grey = cv2.GaussianBlur(grey, (0, 0), 1.1)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3) * normal_strength
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3) * normal_strength
    normal = np.dstack((-gx, gy, np.ones_like(grey)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1.0e-6)
    normal = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    local = cv2.GaussianBlur(grey, (0, 0), 6.0)
    variation = np.clip((grey - local) * 90.0, -22.0, 22.0)
    rough = np.clip(roughness - variation, 28, 242).astype(np.uint8)
    return rough, normal


def write_asset(output_dir: Path, name: str, crop: np.ndarray, recipe: Recipe) -> dict:
    albedo = seamless_tile(suppress_lighting(crop))
    roughness, normal = pbr_maps(
        albedo,
        roughness=recipe.roughness,
        normal_strength=recipe.normal_strength,
    )
    paths = {
        "albedo": output_dir / f"{name}_albedo.png",
        "roughness": output_dir / f"{name}_roughness.png",
        "normal": output_dir / f"{name}_normal.png",
    }
    for key, image in (("albedo", albedo), ("roughness", roughness), ("normal", normal)):
        if not cv2.imwrite(str(paths[key]), image):
            raise RuntimeError(f"could not write {paths[key]}")
    return {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "dimensions_px": [1024, 1024],
        }
        for key, path in paths.items()
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    for recipe in RECIPES.values():
        if recipe.source in images:
            continue
        path = args.source_dir / recipe.source
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read survey review JPEG: {path}")
        if min(image.shape[:2]) < 2400:
            raise ValueError(f"survey still is unexpectedly small: {path} {image.shape[:2]}")
        images[recipe.source] = image
        sources[recipe.source] = {
            "sha256": sha256(path),
            "dimensions_px": [int(image.shape[1]), int(image.shape[0])],
            "path_committed": False,
        }

    assets = {}
    for name, recipe in RECIPES.items():
        crop = rectify(images[recipe.source], recipe.points)
        assets[name] = {
            "source": recipe.source,
            "normalized_rectification_quad": [list(point) for point in recipe.points],
            "privacy_note": recipe.note,
            "maps": write_asset(args.output_dir, name, crop, recipe),
        }

    report = {
        "report_type": "phase7k_privacy_safe_capture_material_generation",
        "status": "passed",
        "passed": True,
        "sources": sources,
        "source_stills_committed": False,
        "privacy_scope": (
            "surface-only crops; no names, paperwork, portraits, screens, people, GPS "
            "or complete-room photographs retained"
        ),
        "assets": assets,
        "collision_geometry_changed": False,
        "physical_release": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
