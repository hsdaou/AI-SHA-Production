#!/usr/bin/env python3
"""Build a two-frame-per-shot visual QA sheet from a route render report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-report", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.render_report.read_text(encoding="utf-8"))
    shots = report.get("shots", [])
    if not isinstance(shots, list) or not shots:
        raise ValueError("render report has no shots")
    frames = sorted(args.frames_dir.glob("frame_*.png"))
    expected_frames = int(report.get("frame_count", 0))
    if not frames or len(frames) != expected_frames:
        raise ValueError(f"found {len(frames)} frames; render report declares {expected_frames}")

    representative_indices: list[int] = []
    offset = 0
    for shot in shots:
        count = int(shot.get("rendered_frames", 0))
        if count < 2:
            raise ValueError("each shot must contain at least two rendered frames")
        representative_indices.extend((offset + count // 3, offset + (2 * count) // 3))
        offset += count
    if offset != len(frames):
        raise ValueError(f"shot frame sum {offset} does not match {len(frames)} files")

    first = Image.open(frames[representative_indices[0]]).convert("RGB")
    aspect = first.height / first.width
    cell_width = args.cell_width
    cell_height = round(cell_width * aspect)
    gap = 6
    columns = max(1, args.columns)
    rows = math.ceil(len(representative_indices) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_width + (columns + 1) * gap, rows * cell_height + (rows + 1) * gap),
        (12, 18, 22),
    )
    draw = ImageDraw.Draw(sheet)
    for tile, frame_index in enumerate(representative_indices):
        image = Image.open(frames[frame_index]).convert("RGB")
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = gap + (tile % columns) * (cell_width + gap)
        y = gap + (tile // columns) * (cell_height + gap)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + image.width - 1, y + image.height - 1), outline=(98, 188, 168), width=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92, optimize=True, progressive=True)
    print(f"wrote {args.output} ({len(representative_indices)} representative frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
