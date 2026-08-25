#!/usr/bin/env python3
"""Encode the Phase 7M NuRec frames as a labelled presentation reel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
NAVY = (33, 23, 7)
TEAL = (207, 229, 39)
OFF_WHITE = (250, 248, 244)
MUTED = (216, 199, 174)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "config/phase7m_nurec_presentation_reel.yaml",
    )
    parser.add_argument(
        "--render-report",
        type=Path,
        default=ROOT / "results/phase7m_nurec_reel_render.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "media/videos/AI-SHA_Phase7M_NuRec_Principal_Visit_Presentation.mp4",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/phase7m_nurec_reel_encode.json",
    )
    parser.add_argument("--intro-seconds", type=float, default=2.5)
    parser.add_argument("--outro-seconds", type=float, default=3.0)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--ffmpeg", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def resolve_ffmpeg(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["AISHA_FFMPEG"]) if "AISHA_FFMPEG" in os.environ else None,
        Path(value) if (value := shutil.which("ffmpeg")) else None,
        Path(
            "/home/robot-wst/isaacsim/kit/python/lib/python3.11/"
            "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
        ),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "ffmpeg was not found; pass --ffmpeg or set AISHA_FFMPEG"
    )


def alpha_box(
    frame: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, first, second, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = OFF_WHITE,
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fade_factor(index: int, count: int, fade_frames: int) -> float:
    if fade_frames <= 0:
        return 1.0
    entrance = min(1.0, (index + 1) / fade_frames)
    exit_factor = min(1.0, (count - index) / fade_frames)
    return max(0.05, min(entrance, exit_factor))


def main() -> int:
    args = parse_args()
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    profile_path = args.profile.expanduser().resolve()
    render_report_path = args.render_report.expanduser().resolve()
    if not profile_path.is_file() or not render_report_path.is_file():
        raise FileNotFoundError(
            profile_path if not profile_path.is_file() else render_report_path
        )
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    render = json.loads(render_report_path.read_text(encoding="utf-8"))
    if not render.get("passed"):
        raise RuntimeError("a passing Phase 7M render report is required")
    width, height = (int(value) for value in profile["render"]["resolution_px"])
    fps = int(profile["render"]["fps"])
    if render["resolution"] != [width, height]:
        raise RuntimeError("profile and render report resolutions differ")
    frames_dir = ROOT / profile["render"]["output_frames"]
    frame_paths = [frames_dir / f"frame_{index:04d}.png" for index in range(len(render["frames"]))]
    missing = [path for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} rendered frame(s), first={missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_main_seconds = len(frame_paths) / fps
    shot_titles: list[str] = []
    cursor = 0
    for shot in profile["shots"]:
        count = int(shot["frame_count"])
        shot_titles.extend([str(shot["title"])] * count)
        cursor += count
    if cursor != len(frame_paths):
        raise RuntimeError("shot frame counts do not equal the rendered frame count")
    intro_frames = round(args.intro_seconds * fps)
    outro_frames = round(args.outro_seconds * fps)
    sx, sy = width / 1920.0, height / 1080.0

    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg raw-video pipe did not open")
    try:
        for index in range(intro_frames):
            frame = np.full((height, width, 3), NAVY, dtype=np.uint8)
            cv2.rectangle(
                frame,
                (round(120 * sx), round(206 * sy)),
                (round(136 * sx), round(636 * sy)),
                TEAL,
                thickness=-1,
            )
            label(frame, "AI-SHA", (round(190 * sx), round(320 * sy)), 3.7 * sy, OFF_WHITE, 8)
            label(
                frame,
                "ADMINISTRATION DIGITAL TWIN",
                (round(198 * sx), round(415 * sy)),
                1.55 * sy,
                TEAL,
                4,
            )
            label(
                frame,
                "NVIDIA ISAAC SIM + NUREC",
                (round(202 * sx), round(485 * sy)),
                1.05 * sy,
                MUTED,
                2,
            )
            label(
                frame,
                "Captured atrium-to-Principal route | recorded navigation replay",
                (round(202 * sx), round(555 * sy)),
                0.78 * sy,
                MUTED,
                2,
            )
            frame = np.clip(
                frame.astype(np.float32)
                * fade_factor(index, intro_frames, round(0.45 * fps)),
                0,
                255,
            ).astype(np.uint8)
            process.stdin.write(frame.tobytes())

        for index, (path, shot_title) in enumerate(zip(frame_paths, shot_titles)):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"could not read full-resolution frame: {path}")
            alpha_box(frame, (0, 0), (width, round(76 * sy)), NAVY, 0.72)
            label(
                frame,
                "AI-SHA  |  ADMINISTRATION DIGITAL TWIN",
                (round(36 * sx), round(51 * sy)),
                0.78 * sy,
                OFF_WHITE,
                2,
            )
            label(
                frame,
                "NVIDIA ISAAC SIM + NUREC",
                (round(1420 * sx), round(50 * sy)),
                0.72 * sy,
                TEAL,
                2,
            )
            alpha_box(
                frame,
                (0, height - round(62 * sy)),
                (width, height),
                NAVY,
                0.62,
            )
            label(
                frame,
                "RECORDED-POSE PRESENTATION REPLAY  |  NOT PHYSICAL RELEASE",
                (round(1035 * sx), height - round(22 * sy)),
                0.52 * sy,
                MUTED,
                1,
            )
            alpha_box(
                frame,
                (round(32 * sx), height - round(156 * sy)),
                (round(900 * sx), height - round(86 * sy)),
                NAVY,
                0.76,
            )
            label(
                frame,
                shot_title,
                (round(58 * sx), height - round(109 * sy)),
                0.75 * sy,
                OFF_WHITE,
                2,
            )
            factor = min(
                fade_factor(index, len(frame_paths), round(0.30 * fps)), 1.0
            )
            if factor < 1.0:
                frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(
                    np.uint8
                )
            process.stdin.write(frame.tobytes())

        for index in range(outro_frames):
            frame = np.full((height, width, 3), NAVY, dtype=np.uint8)
            label(
                frame,
                "WHAT THIS REEL SHOWS",
                (round(150 * sx), round(270 * sy)),
                1.85 * sy,
                TEAL,
                4,
            )
            label(
                frame,
                "Accepted simulated navigation poses replayed inside the captured",
                (round(154 * sx), round(380 * sy)),
                0.87 * sy,
                OFF_WHITE,
                2,
            )
            label(
                frame,
                "Principal-office NuRec twin",
                (round(154 * sx), round(430 * sy)),
                0.87 * sy,
                OFF_WHITE,
                2,
            )
            label(
                frame,
                "Provisional visual registration | frozen collision geometry remains separate",
                (round(154 * sx), round(525 * sy)),
                0.72 * sy,
                MUTED,
                2,
            )
            label(
                frame,
                "Not live policy execution during rendering | Not a physical safety release",
                (round(154 * sx), round(590 * sy)),
                0.72 * sy,
                MUTED,
                2,
            )
            label(frame, "AI-SHA", (round(154 * sx), round(760 * sy)), 2.15 * sy, OFF_WHITE, 5)
            frame = np.clip(
                frame.astype(np.float32)
                * fade_factor(index, outro_frames, round(0.45 * fps)),
                0,
                255,
            ).astype(np.uint8)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")
    capture = cv2.VideoCapture(str(args.output))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open encoded video: {args.output}")
    encoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    encoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first_ok, first_frame = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(encoded_frames - 1, 0))
    last_ok, last_frame = capture.read()
    capture.release()
    duration = encoded_frames / encoded_fps if encoded_fps > 0 else 0.0
    media = {
        "width": encoded_width,
        "height": encoded_height,
        "fps": encoded_fps,
        "frame_count": encoded_frames,
        "duration_s": duration,
        "first_frame_nonblank": bool(first_ok and first_frame.std() > 2.0),
        "last_frame_nonblank": bool(last_ok and last_frame.std() > 2.0),
    }
    passed = bool(
        encoded_width == width
        and encoded_height == height
        and abs(encoded_fps - fps) < 0.01
        and media["first_frame_nonblank"]
        and media["last_frame_nonblank"]
        and duration
        >= args.intro_seconds + total_main_seconds + args.outro_seconds - 0.1
    )
    report = {
        "report_type": "phase7m_nurec_presentation_reel_encode",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "encoded" if passed else "failed_validation",
        "passed": passed,
        "profile": portable_path(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "render_report": portable_path(render_report_path),
        "render_report_sha256": sha256_file(render_report_path),
        "source_frame_count": len(frame_paths),
        "intro_seconds": args.intro_seconds,
        "main_seconds": total_main_seconds,
        "outro_seconds": args.outro_seconds,
        "output": portable_path(args.output),
        "output_sha256": sha256_file(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "ffmpeg": portable_path(ffmpeg),
        "media_probe": media,
        "recorded_pose_presentation_replay": True,
        "presentation_retimed": True,
        "source_policy_executed_live_during_render": False,
        "physical_release": False,
        "external_distribution_requires_user_privacy_review": True,
        "video_committed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
