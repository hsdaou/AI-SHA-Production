#!/usr/bin/env python3
"""Generate privacy-safe PBR material crops from the Principal office stills.

The source photographs remain outside the repository.  Only rectified, mirrored
surface crops are written: no people, documents, portraits or complete room
images are retained in the generated assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "tmp/phase7h_reconstruction/principal/images/principal_still_6707.jpg"
DEFAULT_DETAIL_SOURCE = ROOT / "tmp/phase7h_reconstruction/principal/images/principal_still_6708.jpg"
DEFAULT_OUTPUT = ROOT / "textures/administration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--principal-still", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--principal-detail-still", type=Path, default=DEFAULT_DETAIL_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/phase7h_photo_materials_report.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seamless_tile(image: np.ndarray, size: int = 1024) -> np.ndarray:
    """Cross-fade opposing borders without introducing a central mirror seam."""
    base = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
    out = base.astype(np.float32)
    blend = max(48, size // 10)
    for index in range(blend):
        weight = index / max(1, blend - 1)
        average = (base[:, index].astype(np.float32) + base[:, -1 - index].astype(np.float32)) * 0.5
        out[:, index] = average * (1.0 - weight) + out[:, index] * weight
        out[:, -1 - index] = average * (1.0 - weight) + out[:, -1 - index] * weight
    horizontal = np.clip(out, 0, 255).astype(np.uint8)
    for index in range(blend):
        weight = index / max(1, blend - 1)
        average = (
            horizontal[index].astype(np.float32)
            + horizontal[-1 - index].astype(np.float32)
        ) * 0.5
        out[index] = average * (1.0 - weight) + out[index] * weight
        out[-1 - index] = average * (1.0 - weight) + out[-1 - index] * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def rectify(image: np.ndarray, source_points: list[list[float]]) -> np.ndarray:
    src = np.float32(source_points)
    dst = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image,
        matrix,
        (512, 512),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def suppress_lighting(image: np.ndarray, strength: float = 0.62) -> np.ndarray:
    linear = image.astype(np.float32) / 255.0
    illumination = cv2.GaussianBlur(linear, (0, 0), 42.0)
    mean = illumination.mean(axis=(0, 1), keepdims=True)
    corrected = linear * (mean / np.maximum(illumination, 0.08)) ** strength
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def pbr_maps(albedo: np.ndarray, *, roughness: int, normal_strength: float) -> tuple[np.ndarray, np.ndarray]:
    grey = cv2.cvtColor(albedo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grey = cv2.GaussianBlur(grey, (0, 0), 1.15)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3) * normal_strength
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3) * normal_strength
    normal = np.dstack((-gx, gy, np.ones_like(grey)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1.0e-6)
    normal = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    local = cv2.GaussianBlur(grey, (0, 0), 6.0)
    variation = np.clip((grey - local) * 95.0, -24.0, 24.0)
    rough = np.clip(roughness - variation, 24, 245).astype(np.uint8)
    return rough, normal


def write_set(output_dir: Path, prefix: str, image: np.ndarray, roughness: int, normal_strength: float) -> dict[str, object]:
    albedo = seamless_tile(suppress_lighting(image))
    rough, normal = pbr_maps(albedo, roughness=roughness, normal_strength=normal_strength)
    paths = {
        "albedo": output_dir / f"{prefix}_albedo.png",
        "roughness": output_dir / f"{prefix}_roughness.png",
        "normal": output_dir / f"{prefix}_normal.png",
    }
    cv2.imwrite(str(paths["albedo"]), albedo)
    cv2.imwrite(str(paths["roughness"]), rough)
    cv2.imwrite(str(paths["normal"]), normal)
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for name, path in paths.items()
    }


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.principal_still), cv2.IMREAD_COLOR)
    detail = cv2.imread(str(args.principal_detail_still), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read Principal still: {args.principal_still}")
    if detail is None:
        raise FileNotFoundError(
            f"cannot read Principal detail still: {args.principal_detail_still}"
        )
    height, width = image.shape[:2]
    detail_height, detail_width = detail.shape[:2]
    if width < 1200 or height < 1800:
        raise ValueError(f"expected portrait still of at least 1200x1800, got {width}x{height}")
    if detail_width < 1200 or detail_height < 1800:
        raise ValueError(
            "expected portrait detail still of at least 1200x1800, "
            f"got {detail_width}x{detail_height}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Privacy-reviewed surface-only quadrilaterals in IMG_6707.  The first is
    # unobstructed grey plank flooring; the second is the right-hand walnut
    # storage face below any photographs, paperwork or awards.
    floor = rectify(
        image,
        [[0.40 * width, 0.82 * height], [0.90 * width, 0.78 * height],
         [0.98 * width, 0.995 * height], [0.30 * width, 0.995 * height]],
    )
    walnut = rectify(
        detail,
        [[0.05 * detail_width, 0.15 * detail_height],
         [0.18 * detail_width, 0.17 * detail_height],
         [0.18 * detail_width, 0.23 * detail_height],
         [0.05 * detail_width, 0.21 * detail_height]],
    )

    assets = {
        "principal_grey_oak": write_set(
            args.output_dir, "principal_grey_oak", floor, roughness=126, normal_strength=1.85
        ),
        "principal_walnut": write_set(
            args.output_dir, "principal_walnut", walnut, roughness=94, normal_strength=1.35
        ),
    }
    report = {
        "report_type": "phase7h_privacy_safe_photo_material_generation",
        "status": "passed",
        "source": {
            "path_committed": False,
            "sha256": [
                sha256(args.principal_still),
                sha256(args.principal_detail_still),
            ],
            "dimensions_px": [width, height],
            "detail_dimensions_px": [detail_width, detail_height],
            "privacy_scope": "surface-only floor and walnut crops; no people, portraits, documents or complete-room imagery",
        },
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
