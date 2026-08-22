#!/usr/bin/env python3
"""Encode Isaac Sim route frames into the presentation MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
VALIDATION_PATH = ROOT / "results" / "administration_learned_replay_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_ffmpeg() -> Path | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable)
    bundled_root = Path.home() / "isaacsim" / "kit" / "python" / "lib"
    candidates = sorted(bundled_root.glob("python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"))
    return candidates[-1] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "media" / "learned_route_replay_frames")
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "media" / "videos" / "administration_learned_trajectory_replay.mp4",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "administration_learned_replay_render_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = sorted(args.frames_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError("no route frames; run scripts/render_administration_route.py first")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"could not read {frames[0]}")
    height, width = first.shape[:2]
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f"invalid route frame {frame_path}")
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required; neither PATH nor the Isaac Sim bundle contains it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-i",
            str(args.frames_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        check=True,
    )
    capture = cv2.VideoCapture(str(args.output))
    encoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if encoded_count != len(frames):
        raise RuntimeError(f"encoded {encoded_count} frames; expected {len(frames)}")
    validation_path = args.validation.resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise RuntimeError(f"administration replay validation has not passed: {validation_path}")
    report_path = args.report
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(
        {
            "status": "route_video_encoded",
            "video": str(args.output),
            "encoded_frame_count": encoded_count,
            "encoded_fps": encoded_fps,
            "encoded_duration_s": encoded_count / encoded_fps,
            "encoder": "ffmpeg/libx264",
            "encoder_preset": args.preset,
            "encoder_crf": args.crf,
            "video_sha256": sha256_file(args.output),
            "video_size_bytes": args.output.stat().st_size,
            "clearance_validation_report": str(validation_path),
            "clearance_validation_report_sha256": sha256_file(validation_path),
            "atrium_column_clearance": validation["atrium_column_clearance"],
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({encoded_count} frames at {encoded_fps:.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
