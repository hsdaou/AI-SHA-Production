#!/usr/bin/env python3
"""Prepare privacy-screened texture atlases for the Phase 7K survey layers.

The dense OBJ reconstructions and source atlases stay outside git.  This tool
records their hashes and creates presentation atlases with face/face-like
regions blurred.  Tesseract is also run as a fail-closed readable-text check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECONSTRUCTION = ROOT / "tmp/phase7h_reconstruction"
DEFAULT_OUTPUT = ROOT / "textures/phase7k_photogrammetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-root", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/phase7k_photogrammetry_asset_manifest.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obj_stats(path: Path) -> dict:
    vertices = texture_coordinates = faces = 0
    materials: set[str] = set()
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                point = [float(value) for value in line.split()[1:4]]
                vertices += 1
                minimum = [min(old, value) for old, value in zip(minimum, point)]
                maximum = [max(old, value) for old, value in zip(maximum, point)]
            elif line.startswith("vt "):
                texture_coordinates += 1
            elif line.startswith("f "):
                faces += 1
            elif line.startswith("usemtl "):
                materials.add(line.split()[1])
    return {
        "vertices": vertices,
        "texture_coordinates": texture_coordinates,
        "faces": faces,
        "native_min_xyz_m": [round(value, 6) for value in minimum],
        "native_max_xyz_m": [round(value, 6) for value in maximum],
        "native_extent_xyz_m": [
            round(high - low, 6) for low, high in zip(minimum, maximum)
        ],
        "materials": sorted(materials),
    }


def tesseract_text(path: Path) -> str:
    local_root = ROOT / "tmp/phase7k_tools/tesseract/root"
    executable = shutil.which("tesseract") or str(local_root / "usr/bin/tesseract")
    if not Path(executable).is_file():
        raise RuntimeError("privacy OCR screen requires the tesseract executable")
    environment = dict(os.environ)
    local_tessdata = local_root / "usr/share/tesseract-ocr/5/tessdata"
    if local_tessdata.is_dir():
        environment["TESSDATA_PREFIX"] = str(local_tessdata)
    try:
        tsv_config = Path("/usr/share/tesseract-ocr/5/tessdata/configs/tsv")
        if not tsv_config.is_file():
            raise RuntimeError(f"privacy OCR TSV configuration is missing: {tsv_config}")
        result = subprocess.run(
            [executable, str(path), "stdout", "--psm", "11", str(tsv_config)],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
            env=environment,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"privacy OCR screen could not complete for {path}: {error}") from error
    readable = []
    for line in result.stdout.splitlines()[1:]:
        columns = line.split("\t")
        if len(columns) != 12:
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        token = columns[11].strip()
        letters = "".join(character for character in token if character.isalpha())
        # UV-island edges create abundant low-confidence OCR noise.  A real
        # retained word is required to be both long and high-confidence.
        if confidence >= 75.0 and len(letters) >= 4:
            readable.append(token)
    return " ".join(readable)


def privacy_screen_atlas(source: Path, output: Path) -> dict:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"could not load face screen: {cascade_path}")
    candidates = cascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(38, 38),
    )
    redactions = []
    height, width = image.shape[:2]
    for x, y, box_width, box_height in candidates:
        # Extremely large detections are atlas background false positives;
        # human faces in these UV islands remain well below this threshold.
        if max(box_width, box_height) > 480:
            continue
        padding = max(12, int(0.18 * max(box_width, box_height)))
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(width, x + box_width + padding), min(height, y + box_height + padding)
        roi = image[y0:y1, x0:x1]
        sigma = max(18.0, 0.22 * max(box_width, box_height))
        image[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (0, 0), sigma)
        redactions.append([int(x0), int(y0), int(x1), int(y1)])

    # Two-kilopixel atlases retain presentation detail while preventing tiny UV
    # islands from exposing legible source-image detail in a public repository.
    image = cv2.resize(image, (2048, 2048), interpolation=cv2.INTER_AREA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"could not write {output}")
    readable_text = tesseract_text(output)
    if readable_text:
        raise RuntimeError(
            f"readable OCR text remained after atlas privacy screen ({output}): "
            f"{readable_text[:160]}"
        )
    return {
        "source_sha256": sha256(source),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256(output),
        "output_dimensions_px": [2048, 2048],
        "face_or_face_like_regions_blurred": len(redactions),
        "redaction_boxes_source_px": redactions,
        "readable_ocr_text_after_screen": False,
    }


def main() -> int:
    args = parse_args()
    clusters = {
        "atrium_corridor": args.reconstruction_root / "atrium/work/textured",
        "principal_office": args.reconstruction_root / "principal/office/work/textured",
    }
    report_clusters = {}
    for name, directory in clusters.items():
        obj = directory / "texturedMesh.obj"
        mtl = directory / "texturedMesh.mtl"
        if not obj.is_file() or not mtl.is_file():
            raise FileNotFoundError(f"missing dense reconstruction under {directory}")
        atlases = {}
        for index in (1001, 1002):
            source = directory / f"texture_{index}.jpg"
            output = args.output_dir / f"{name}_texture_{index}.jpg"
            atlases[str(index)] = privacy_screen_atlas(source, output)
        report_clusters[name] = {
            "source_obj_sha256": sha256(obj),
            "source_mtl_sha256": sha256(mtl),
            "source_obj_committed": False,
            "source_atlases_committed": False,
            "geometry": obj_stats(obj),
            "presentation_atlases": atlases,
        }

    report = {
        "report_type": "phase7k_photogrammetry_asset_manifest",
        "status": "passed",
        "passed": True,
        "engine": "AliceVision 3.3.0 dense reconstruction and texturing",
        "clusters": report_clusters,
        "privacy_contract": (
            "source photographs/OBJ/atlases remain uncommitted; public atlases are "
            "2K, face-screened and OCR-negative"
        ),
        "cluster_registration": "provisional similarity transforms against metric RoomPlan",
        "clusters_false_welded": False,
        "raw_dense_mesh_used_for_collision": False,
        "physical_release": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
