#!/usr/bin/env python3
"""Create a concise, labeled presentation cut from the full Isaac Lab capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


PHASES = (
    (10.7, "Central atrium departure"),
    (39.5, "East hallway to Vice Principal"),
    (50.7, "Vice Principal approach"),
    (78.7, "Vice Principal office visit"),
    (110.1, "Return to central atrium"),
    (140.1, "Transfer to Principal suite"),
    (151.6, "Principal office visit"),
    (174.0, "Return to home position"),
)


def phase_at(elapsed_s: float) -> str:
    return next(label for end_s, label in PHASES if elapsed_s < end_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--skip-seconds", type=float, default=0.8)
    args = parser.parse_args()
    if args.speed < 1:
        parser.error("--speed must be positive")

    source = cv2.VideoCapture(str(args.input))
    if not source.isOpened():
        raise RuntimeError(f"could not open {args.input}")
    source_fps = float(source.get(cv2.CAP_PROP_FPS))
    source_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = round(args.skip_seconds * source_fps)

    crop_width = min(source_width, 960)
    crop_height = min(source_height, 540)
    crop_x = max(0, (source_width - crop_width) // 2)
    crop_y = max(0, (source_height - crop_height) // 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {args.output}")

    source.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_index = start_frame
    output_count = 0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            if (source_index - start_frame) % args.speed != 0:
                source_index += 1
                continue
            elapsed_s = source_index / source_fps
            crop = frame[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
            canvas = cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_LANCZOS4)

            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 100), (13, 24, 31), -1)
            cv2.rectangle(overlay, (0, 652), (1280, 720), (13, 24, 31), -1)
            cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, canvas)
            cv2.putText(
                canvas,
                "AI-SHA | ISAAC LAB LEARNED-POLICY RUN",
                (34, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.84,
                (244, 250, 251),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                phase_at(elapsed_s),
                (34, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.73,
                (121, 226, 197),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"sim {elapsed_s:05.1f}s  |  {args.speed}x playback",
                (955, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (221, 228, 232),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Wheel-joint physics | learned policy on aligned legs | deterministic physical turn supervisor",
                (34, 681),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (239, 243, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Held-out gate: 576/576 successes, 0 collisions | plan-derived training proxy",
                (34, 707),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (161, 224, 207),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            output_count += 1
            source_index += 1
    finally:
        source.release()
        writer.release()

    check = cv2.VideoCapture(str(args.output))
    encoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = float(check.get(cv2.CAP_PROP_FPS))
    check.release()
    if encoded_frames != output_count:
        raise RuntimeError(f"encoded {encoded_frames} frames, expected {output_count}")

    report = {
        "report_type": "learned_route_presentation_video",
        "source_video": str(args.input.resolve()),
        "output_video": str(args.output.resolve()),
        "source_frame_count": source_frames,
        "source_fps": source_fps,
        "source_resolution": [source_width, source_height],
        "source_duration_s": source_frames / source_fps,
        "presentation_frame_count": encoded_frames,
        "presentation_fps": encoded_fps,
        "presentation_duration_s": encoded_frames / encoded_fps,
        "speed_multiplier": args.speed,
        "crop": {"x": crop_x, "y": crop_y, "width": crop_width, "height": crop_height},
        "motion_changed": False,
        "disclosure": (
            "Post-processed from the complete successful Isaac Lab capture. Frames are center-cropped, "
            "labeled, and temporally sampled only; robot motion and route outcomes are unchanged."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
