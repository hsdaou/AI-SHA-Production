#!/usr/bin/env python3
"""Assemble the accepted office mission and dynamic-safety films into one reel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


INTRO_FRAMES = 90
TRANSITION_FRAMES = 60
END_FRAMES = 120
EXPECTED_CHECKPOINT_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_ffmpeg() -> Path:
    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable)
    candidates = sorted(
        (Path.home() / "isaacsim" / "kit" / "python" / "lib").glob(
            "python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        )
    )
    if not candidates:
        raise RuntimeError("ffmpeg is required")
    return candidates[-1]


def inspect_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return metadata


def read_reference_frame(path: Path, last: bool = False) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    if last:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read a reference frame from {path}")
    return frame


def add_centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(24, (frame.shape[1] - size[0]) // 2)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_card(
    background: np.ndarray,
    eyebrow: str,
    title: str,
    lines: list[str],
    footer: str,
) -> np.ndarray:
    frame = cv2.GaussianBlur(background, (0, 0), 18.0)
    shade = np.full_like(frame, (6, 15, 20))
    frame = cv2.addWeighted(frame, 0.30, shade, 0.70, 0.0)
    cv2.rectangle(frame, (0, 0), (1280, 9), (111, 214, 166), -1)
    add_centered_text(frame, eyebrow, 180, 0.55, (164, 205, 215), 1)
    add_centered_text(frame, title, 270, 1.05, (247, 250, 250), 2)
    cv2.line(frame, (450, 305), (830, 305), (111, 214, 166), 2, cv2.LINE_AA)
    y = 365
    for line in lines:
        add_centered_text(frame, line, y, 0.58, (218, 231, 233), 1)
        y += 48
    add_centered_text(frame, footer, 662, 0.42, (147, 174, 180), 1)
    return frame


def write_repeated(writer: cv2.VideoWriter, frame: np.ndarray, count: int) -> None:
    for _ in range(count):
        writer.write(frame)


def copy_video_frames(
    writer: cv2.VideoWriter, source_path: Path, expected_frames: int
) -> int:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source_path}")
    copied = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            copied += 1
    finally:
        capture.release()
    if copied != expected_frames:
        raise RuntimeError(
            f"copied {copied} frames from {source_path}; expected {expected_frames}"
        )
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-video", type=Path, required=True)
    parser.add_argument("--mission-report", type=Path, required=True)
    parser.add_argument("--safety-video", type=Path, required=True)
    parser.add_argument("--safety-video-report", type=Path, required=True)
    parser.add_argument("--safety-run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    mission_report = json.loads(args.mission_report.read_text(encoding="utf-8"))
    safety_video_report = json.loads(
        args.safety_video_report.read_text(encoding="utf-8")
    )
    safety_run_report = json.loads(
        args.safety_run_report.read_text(encoding="utf-8")
    )
    if mission_report.get("report_type") != "administration_live_policy_presentation_video":
        raise RuntimeError("mission report has the wrong type")
    if not mission_report.get("passed") or mission_report.get("motion_changed"):
        raise RuntimeError("accepted unchanged-motion mission evidence is required")
    if safety_video_report.get("report_type") != "phase4a_dynamic_safety_presentation_video":
        raise RuntimeError("safety video report has the wrong type")
    if not safety_video_report.get("passed") or not all(
        safety_video_report.get("checks", {}).values()
    ):
        raise RuntimeError("accepted safety video evidence is required")
    if safety_run_report.get("report_type") != "phase4a_live_dynamic_safety_showcase":
        raise RuntimeError("safety run report has the wrong type")
    if not safety_run_report.get("passed") or not all(
        safety_run_report.get("checks", {}).values()
    ):
        raise RuntimeError("accepted Phase 4A run evidence is required")

    mission_meta = inspect_video(args.mission_video)
    safety_meta = inspect_video(args.safety_video)
    if mission_meta["width"] != 1280 or mission_meta["height"] != 720:
        raise RuntimeError("mission video must be 1280x720")
    if safety_meta["width"] != 1280 or safety_meta["height"] != 720:
        raise RuntimeError("safety video must be 1280x720")
    if not 29.0 <= float(mission_meta["fps"]) <= 31.0:
        raise RuntimeError("mission video must be 30 fps")
    if abs(float(mission_meta["fps"]) - float(safety_meta["fps"])) > 0.01:
        raise RuntimeError("source frame rates do not match")
    if mission_report.get("output_video_sha256") != sha256_file(args.mission_video):
        raise RuntimeError("mission video hash does not match its report")
    if safety_video_report.get("output_video_sha256") != sha256_file(
        args.safety_video
    ):
        raise RuntimeError("safety video hash does not match its report")
    if safety_video_report.get("source_run_report_sha256") != sha256_file(
        args.safety_run_report
    ):
        raise RuntimeError("safety run report hash is not linked")
    if mission_report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("mission does not use the accepted Phase 3N checkpoint")
    if (
        safety_run_report.get("checkpoint", {}).get("sha256")
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("safety run does not use the accepted Phase 3N checkpoint")

    fps = float(mission_meta["fps"])
    mission_frames = int(mission_meta["frames"])
    safety_frames = int(safety_meta["frames"])
    total_frames = (
        INTRO_FRAMES
        + mission_frames
        + TRANSITION_FRAMES
        + safety_frames
        + END_FRAMES
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    working = args.output.with_name(args.output.stem + ".working.mp4")
    writer = cv2.VideoWriter(
        str(working), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {working}")

    intro = make_card(
        read_reference_frame(args.mission_video),
        "AI-SHA | ISAAC SIM + ISAAC LAB",
        "ADMINISTRATION MISSION",
        [
            "Live learned-policy office route",
            "Principal + Vice-Principal visits | dynamic safety insert",
        ],
        "Simulation evidence | plan/walkthrough-derived scene | not physical release",
    )
    transition = make_card(
        read_reference_frame(args.safety_video),
        "ACCEPTED PHASE 3N CHECKPOINT",
        "DYNAMIC SAFETY DEMONSTRATION",
        [
            "Deterministic pedestrian crossing",
            "Learned 360 DEG brake authority + frozen protective-stop stack",
        ],
        "Presentation proxy | learned and protective states labeled separately",
    )
    ending = make_card(
        read_reference_frame(args.safety_video, last=True),
        "VERIFIED ISAAC SIM EVIDENCE",
        "MISSION COMPLETE",
        [
            "12/12 office-route waypoints | zero collisions",
            "Dynamic encounter: stop - wait - resume | zero contacts",
            "Next: measured site | Nav2/sim-to-real | hardware safety",
        ],
        "Presentation-grade simulation evidence | no physical safety claim",
    )

    try:
        write_repeated(writer, intro, INTRO_FRAMES)
        copied_mission = copy_video_frames(writer, args.mission_video, mission_frames)
        write_repeated(writer, transition, TRANSITION_FRAMES)
        copied_safety = copy_video_frames(writer, args.safety_video, safety_frames)
        write_repeated(writer, ending, END_FRAMES)
    finally:
        writer.release()

    ffmpeg = find_ffmpeg()
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(working),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        check=True,
    )
    working.unlink()

    final_meta = inspect_video(args.output)
    timeline = [
        {"section": "intro_card", "start_frame": 0, "end_frame": 89, "frames": 90},
        {
            "section": "complete_office_mission",
            "start_frame": 90,
            "end_frame": 89 + mission_frames,
            "frames": mission_frames,
        },
        {
            "section": "dynamic_safety_transition_card",
            "start_frame": 90 + mission_frames,
            "end_frame": 149 + mission_frames,
            "frames": 60,
        },
        {
            "section": "dynamic_safety_encounter",
            "start_frame": 150 + mission_frames,
            "end_frame": 149 + mission_frames + safety_frames,
            "frames": safety_frames,
        },
        {
            "section": "evidence_and_next_gates_card",
            "start_frame": 150 + mission_frames + safety_frames,
            "end_frame": total_frames - 1,
            "frames": 120,
        },
    ]
    checks = {
        "mission_report_passed": mission_report["passed"] is True,
        "mission_motion_unchanged": mission_report["motion_changed"] is False,
        "safety_video_report_passed": safety_video_report["passed"] is True,
        "safety_run_report_passed": safety_run_report["passed"] is True,
        "same_accepted_checkpoint": (
            mission_report["checkpoint_sha256"]
            == safety_run_report["checkpoint"]["sha256"]
            == EXPECTED_CHECKPOINT_SHA256
        ),
        "mission_source_hash_linked": mission_report["output_video_sha256"]
        == sha256_file(args.mission_video),
        "safety_source_hash_linked": safety_video_report["output_video_sha256"]
        == sha256_file(args.safety_video),
        "all_mission_frames_included": copied_mission == mission_frames,
        "all_safety_frames_included": copied_safety == safety_frames,
        "output_frame_count_exact": int(final_meta["frames"]) == total_frames,
        "presentation_duration_60_to_90_seconds": 60.0
        <= total_frames / fps
        <= 90.0,
        "presentation_video_created": args.output.is_file()
        and args.output.stat().st_size > 0,
    }
    report = {
        "report_type": "final_omniverse_administration_presentation_reel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "output_video": str(args.output.resolve()),
        "output_video_sha256": sha256_file(args.output),
        "resolution": [int(final_meta["width"]), int(final_meta["height"])],
        "fps": float(final_meta["fps"]),
        "frames": int(final_meta["frames"]),
        "duration_s": int(final_meta["frames"]) / float(final_meta["fps"]),
        "audio_track": False,
        "timeline": timeline,
        "sources": {
            "complete_office_mission": {
                "video": str(args.mission_video.resolve()),
                "video_sha256": sha256_file(args.mission_video),
                "report": str(args.mission_report.resolve()),
                "report_sha256": sha256_file(args.mission_report),
                "frames_included": copied_mission,
                "motion_changed_by_assembly": False,
            },
            "dynamic_safety_encounter": {
                "video": str(args.safety_video.resolve()),
                "video_sha256": sha256_file(args.safety_video),
                "video_report": str(args.safety_video_report.resolve()),
                "video_report_sha256": sha256_file(args.safety_video_report),
                "run_report": str(args.safety_run_report.resolve()),
                "run_report_sha256": sha256_file(args.safety_run_report),
                "frames_included": copied_safety,
                "motion_changed_by_assembly": False,
            },
        },
        "motion_changed_by_assembly": False,
        "assembly_disclosure": (
            "Every decoded frame from both accepted presentation sources is included "
            "once and in order. Title cards were added between sources. Final H.264 "
            "encoding changes pixel compression but does not retime or alter motion "
            "inside either accepted source clip."
        ),
        "claim_boundary": (
            "Presentation-grade Isaac Sim and Isaac Lab evidence in a plan- and "
            "walkthrough-derived administration scene. Door dimensions and scene "
            "geometry remain disclosed presentation assumptions. This is not an "
            "as-built survey, physical human-safety claim, Nav2 release, sim-to-real "
            "validation, or permission to deploy on hardware."
        ),
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
