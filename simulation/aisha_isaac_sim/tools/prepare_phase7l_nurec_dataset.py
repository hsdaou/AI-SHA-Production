#!/usr/bin/env python3
"""Prepare and audit the connected COLMAP component used by Phase 7L NuRec."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "tmp/phase7k_photogrammetry_2fps",
    )
    parser.add_argument("--component", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp/phase7l_nurec_dataset",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/phase7l_nurec_dataset_preflight.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


def registered_image_names(images_txt: Path) -> list[str]:
    names: list[str] = []
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if (
            len(fields) >= 10
            and fields[0].isdigit()
            and fields[9].lower().endswith((".jpg", ".jpeg", ".png"))
        ):
            names.append(fields[9])
    return names


def header_value(path: Path, prefix: str) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    images = source / "images/video"
    sparse = source / f"sparse/{args.component}"
    sparse_txt = source / f"sparse/{args.component}_txt"
    required = (
        images,
        sparse / "cameras.bin",
        sparse / "images.bin",
        sparse / "points3D.bin",
        sparse_txt / "cameras.txt",
        sparse_txt / "images.txt",
        sparse_txt / "points3D.txt",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    names = registered_image_names(sparse_txt / "images.txt")
    missing = [name for name in names if not (images / name).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} registered COLMAP images are missing")
    frame_ids = sorted(int(Path(name).stem.split("_")[-1]) for name in names)

    output = args.output.resolve()
    (output / "sparse").mkdir(parents=True, exist_ok=True)
    replace_symlink(output / "images", images)
    replace_symlink(output / "sparse/0", sparse)
    downsample_dir = output / "images_2"
    downsample_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names, start=1):
        destination = downsample_dir / name
        if destination.is_file():
            existing = cv2.imread(str(destination), cv2.IMREAD_COLOR)
            if existing is not None and existing.shape[:2] == (960, 540):
                continue
        image = cv2.imread(str(images / name), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (1920, 1080):
            raise RuntimeError(f"unexpected source image geometry: {images / name}")
        resized = cv2.resize(image, (540, 960), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"could not write {destination}")
        if index % 50 == 0:
            print(f"prepared {index}/{len(names)} downsampled training images")

    cameras_lines = [
        line
        for line in (sparse_txt / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    report = {
        "report_type": "phase7l_nurec_colmap_dataset_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "passed": True,
        "source": {
            "pipeline": "COLMAP 2 fps walkthrough reconstruction",
            "component": args.component,
            "registered_images": len(names),
            "frame_id_range": [frame_ids[0], frame_ids[-1]],
            "mean_observations_per_image": header_value(
                sparse_txt / "images.txt", "# Number of images:"
            ),
            "points3d_header": header_value(
                sparse_txt / "points3D.txt", "# Number of points:"
            ),
            "cameras": cameras_lines,
            "cameras_bin_sha256": sha256(sparse / "cameras.bin"),
            "images_bin_sha256": sha256(sparse / "images.bin"),
            "points3d_bin_sha256": sha256(sparse / "points3D.bin"),
            "raw_images_committed": False,
            "dataset_symlinks_committed": False,
        },
        "dataset": {
            "path": portable_path(output),
            "image_link_target": portable_path(images),
            "sparse_link_target": portable_path(sparse),
            "all_registered_images_available": True,
            "recommended_downsample_factor": 2,
            "downsampled_training_images": len(names),
            "downsampled_resolution_px": [540, 960],
            "downsampled_images_committed": False,
        },
        "training_contract": {
            "engine": "NVIDIA 3DGRUT",
            "target_export": "NuRec USDZ for Isaac Sim 5.1",
            "visual_layer_only": True,
            "navigation_collision_geometry_changed": False,
            "physical_release": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
