#!/usr/bin/env python3
"""Encode Isaac Sim route frames into the presentation MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "media" / "videos" / "administration_route.mp4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = sorted((ROOT / "media" / "route_frames").glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError("no route frames; run scripts/render_administration_route.py first")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"could not read {frames[0]}")
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer")
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"invalid route frame {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(args.output))
    encoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if encoded_count != len(frames):
        raise RuntimeError(f"encoded {encoded_count} frames; expected {len(frames)}")
    report_path = ROOT / "results" / "administration_render_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(
        {
            "status": "route_video_encoded",
            "video": str(args.output),
            "encoded_frame_count": encoded_count,
            "encoded_fps": encoded_fps,
            "encoded_duration_s": encoded_count / encoded_fps,
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({encoded_count} frames at {encoded_fps:.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
