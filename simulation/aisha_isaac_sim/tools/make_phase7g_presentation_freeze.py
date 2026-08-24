#!/usr/bin/env python3
"""Assemble the Full HD Phase 7G presentation freeze from accepted sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "config/phase7g_presentation_freeze.yaml"
    )
    parser.add_argument(
        "--mission-video",
        type=Path,
        default=ROOT / "media/videos/AI-SHA_Phase7F_Operator_Omniverse_Presentation.mp4",
    )
    parser.add_argument(
        "--mission-acceptance",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7f_operator_presentation_acceptance.json",
    )
    parser.add_argument(
        "--dynamic-video",
        type=Path,
        default=ROOT / "media/videos/AI-SHA_Phase4A_Administration_Dynamic_Safety_Showcase.mp4",
    )
    parser.add_argument(
        "--dynamic-video-report",
        type=Path,
        default=ROOT / "results/phase4a_dynamic_safety_presentation_video_report.json",
    )
    parser.add_argument(
        "--dynamic-run-report",
        type=Path,
        default=ROOT / "results/phase4a_administration_dynamic_showcase_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "media/videos/AI-SHA_Phase7G_Omniverse_Presentation_Freeze.mp4",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=ROOT / "media/AI-SHA_Phase7G_Omniverse_Presentation_Freeze_contact_sheet.jpg",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7g_presentation_freeze_report.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    result["duration_s"] = result["frames"] / result["fps"]
    return result


def find_ffmpeg() -> Path:
    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable)
    candidates = sorted(
        (Path.home() / "isaacsim/kit/python/lib").glob(
            "python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        )
    )
    if not candidates:
        raise RuntimeError("ffmpeg is required")
    return candidates[-1]


def add_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def add_centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    add_text(frame, text, ((frame.shape[1] - size[0]) // 2, y), scale, color, thickness)


def reference_frame(path: Path, fraction: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(count - 1, round(fraction * (count - 1)))))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read reference frame from {path}")
    return frame


def title_card(
    background: np.ndarray,
    eyebrow: str,
    title: str,
    lines: list[str],
    footer: str,
) -> np.ndarray:
    background = cv2.resize(background, (1920, 1080), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(background, (0, 0), 24.0)
    shade = np.full_like(blurred, (7, 17, 22))
    frame = cv2.addWeighted(blurred, 0.24, shade, 0.76, 0.0)
    cv2.rectangle(frame, (0, 0), (1920, 12), (146, 221, 191), -1)
    add_centered_text(frame, eyebrow, 255, 0.76, (176, 213, 220), 1)
    add_centered_text(frame, title, 395, 1.52, (248, 250, 250), 3)
    cv2.line(frame, (670, 445), (1250, 445), (146, 221, 191), 3, cv2.LINE_AA)
    y = 540
    for line in lines:
        add_centered_text(frame, line, y, 0.78, (222, 234, 236), 2)
        y += 68
    add_centered_text(frame, footer, 975, 0.54, (153, 181, 187), 1)
    return frame


def repeated(writer: cv2.VideoWriter, frame: np.ndarray, count: int) -> None:
    for _ in range(count):
        writer.write(frame)


def copy_full_hd_source(writer: cv2.VideoWriter, path: Path, expected_frames: int) -> int:
    capture = cv2.VideoCapture(str(path))
    copied = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1::-1] != (1920, 1080):
                raise RuntimeError("wide mission source must remain 1920x1080")
            writer.write(frame)
            copied += 1
    finally:
        capture.release()
    if copied != expected_frames:
        raise RuntimeError(f"copied {copied} mission frames; expected {expected_frames}")
    return copied


def dynamic_layout(frame: np.ndarray, metrics: dict, source_frame: int, source_total: int) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (8, 18, 24), dtype=np.uint8)
    x, y, width, height = (40, 135, 1440, 810)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    canvas[y : y + height, x : x + width] = resized
    cv2.rectangle(canvas, (x - 2, y - 2), (x + width + 2, y + height + 2), (146, 221, 191), 2)
    add_text(canvas, "DYNAMIC OBSTACLE SAFETY", (40, 75), 1.02, (244, 248, 248), 2)
    add_text(canvas, "Accepted deterministic pedestrian encounter", (40, 108), 0.58, (150, 204, 190), 1)
    panel_x = 1510
    cv2.rectangle(canvas, (panel_x, 135), (1880, 945), (12, 29, 37), -1)
    cv2.rectangle(canvas, (panel_x, 135), (1880, 945), (74, 122, 116), 2)
    add_text(canvas, "LIVE SOURCE EVIDENCE", (1540, 195), 0.54, (146, 221, 191), 1)
    add_text(canvas, "LEARNED 360 SAFETY", (1540, 252), 0.66, (246, 248, 248), 2)
    add_text(canvas, "+ PROTECTIVE STOP", (1540, 290), 0.58, (246, 248, 248), 1)
    cv2.line(canvas, (1540, 326), (1850, 326), (74, 122, 116), 1)
    add_text(canvas, "STOP", (1540, 390), 0.72, (116, 190, 255), 2)
    add_text(canvas, "WAIT", (1540, 445), 0.72, (116, 190, 255), 2)
    add_text(canvas, "RESUME", (1540, 500), 0.72, (146, 221, 191), 2)
    add_text(canvas, "Measured evidence", (1540, 585), 0.52, (184, 205, 210), 1)
    add_text(
        canvas,
        f"full stop  {float(metrics['protective_stop_duration_s']):.2f} s",
        (1540, 635),
        0.52,
        (238, 242, 243),
        1,
    )
    add_text(
        canvas,
        f"resume     {float(metrics['maximum_resumed_velocity_mps']):.2f} m/s",
        (1540, 680),
        0.52,
        (238, 242, 243),
        1,
    )
    add_text(
        canvas,
        f"contacts   {int(metrics['collisions'])}",
        (1540, 725),
        0.52,
        (238, 242, 243),
        1,
    )
    progress = 0.0 if source_total <= 1 else source_frame / (source_total - 1)
    cv2.rectangle(canvas, (1540, 805), (1850, 823), (35, 59, 65), -1)
    cv2.rectangle(canvas, (1540, 805), (1540 + round(310 * progress), 823), (146, 221, 191), -1)
    add_text(canvas, "SIMULATION EVIDENCE INSERT", (1540, 875), 0.42, (142, 168, 174), 1)
    add_text(canvas, "No physical safety claim", (1540, 910), 0.42, (142, 168, 174), 1)
    add_text(
        canvas,
        "Accepted Phase 4A source | frame-rate normalized only; encounter duration preserved",
        (40, 1018),
        0.49,
        (167, 190, 195),
        1,
    )
    return canvas


def copy_dynamic_resampled(
    writer: cv2.VideoWriter,
    path: Path,
    target_fps: float,
    expected_frames: int,
    metrics: dict,
) -> tuple[int, list[int]]:
    capture = cv2.VideoCapture(str(path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    desired = [
        min(source_frames - 1, round(index * source_fps / target_fps))
        for index in range(expected_frames)
    ]
    source_index = -1
    frame = None
    written = 0
    try:
        for desired_index in desired:
            while source_index < desired_index:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("dynamic source ended during frame-rate normalization")
                source_index += 1
            if frame is None:
                raise RuntimeError("dynamic source returned no frames")
            writer.write(dynamic_layout(frame, metrics, desired_index, source_frames))
            written += 1
    finally:
        capture.release()
    return written, desired


def make_contact_sheet(video: Path, output: Path, frame_count: int) -> None:
    capture = cv2.VideoCapture(str(video))
    sample_indices = np.linspace(0, frame_count - 1, 16).round().astype(int).tolist()
    thumbnails = []
    for index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not sample final video frame {index}")
        thumbnails.append(cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA))
    capture.release()
    sheet = np.vstack([np.hstack(thumbnails[row : row + 4]) for row in range(0, 16, 4)])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"could not write {output}")


def main() -> int:
    args = parse_args()
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    mission_acceptance = load_json(args.mission_acceptance)
    dynamic_video_report = load_json(args.dynamic_video_report)
    dynamic_run_report = load_json(args.dynamic_run_report)
    contract = profile["freeze_contract"]
    metrics = profile["dynamic_insert"]["metrics"]
    mission_meta = inspect_video(args.mission_video)
    dynamic_meta = inspect_video(args.dynamic_video)

    if not mission_acceptance.get("passed") or mission_acceptance.get("checks_passed") != 19:
        raise RuntimeError("accepted 19/19 Phase 7F mission evidence is required")
    if mission_acceptance.get("video", {}).get("sha256") != sha256(args.mission_video):
        raise RuntimeError("Phase 7F video hash does not match its acceptance")
    if not dynamic_video_report.get("passed") or not all(
        dynamic_video_report.get("checks", {}).values()
    ):
        raise RuntimeError("accepted dynamic-safety video evidence is required")
    if dynamic_video_report.get("output_video_sha256") != sha256(args.dynamic_video):
        raise RuntimeError("dynamic-safety video hash does not match its report")
    if not dynamic_run_report.get("passed") or not all(dynamic_run_report.get("checks", {}).values()):
        raise RuntimeError("accepted dynamic-safety run evidence is required")
    if [mission_meta["width"], mission_meta["height"]] != contract["resolution"]:
        raise RuntimeError("Phase 7F mission resolution does not match the freeze contract")
    if not math.isclose(float(mission_meta["fps"]), float(contract["fps"]), abs_tol=0.01):
        raise RuntimeError("Phase 7F mission frame rate does not match the freeze contract")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    working = args.output.with_name(args.output.stem + ".working.mp4")
    writer = cv2.VideoWriter(
        str(working),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(contract["fps"]),
        tuple(contract["resolution"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {working}")

    intro = title_card(
        reference_frame(args.mission_video, 0.15),
        "AI-SHA | NVIDIA ISAAC SIM + ISAAC LAB",
        "ADMINISTRATION MISSION",
        [
            "Verified 12-leg office route",
            "Vice-Principal + Principal visits | wide RTX presentation camera",
        ],
        "Simulation evidence | plan, capture and walkthrough-informed presentation twin",
    )
    transition = title_card(
        reference_frame(args.dynamic_video, 0.38),
        "ACCEPTED LEARNED-SAFETY CHECKPOINT",
        "DYNAMIC OBSTACLE RESPONSE",
        [
            "Pedestrian detected in the administration route",
            "Learned brake authority + controlled stop, wait and resume",
        ],
        "Accepted simulation evidence | no physical safety or deployment claim",
    )
    ending = title_card(
        reference_frame(args.mission_video, 0.95),
        "PRESENTATION FREEZE | VERIFIED OMNIVERSE EVIDENCE",
        "MISSION COMPLETE",
        [
            "12/12 route legs | both offices visited | zero route contacts",
            "Dynamic encounter | 2.23 s stop | resumed safely | zero contacts",
        ],
        "Next physical gate: received-unit identity, read-only telemetry and measured calibration",
    )

    try:
        repeated(writer, intro, int(contract["intro_frames"]))
        mission_frames = copy_full_hd_source(
            writer, args.mission_video, int(contract["expected_wide_mission_frames"])
        )
        repeated(writer, transition, int(contract["transition_frames"]))
        dynamic_frames, selected_dynamic_indices = copy_dynamic_resampled(
            writer,
            args.dynamic_video,
            float(contract["fps"]),
            int(contract["expected_dynamic_safety_frames"]),
            metrics,
        )
        repeated(writer, ending, int(contract["ending_frames"]))
    finally:
        writer.release()

    subprocess.run(
        [
            str(find_ffmpeg()),
            "-y",
            "-v",
            "error",
            "-i",
            str(working),
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        check=True,
    )
    working.unlink()
    output_meta = inspect_video(args.output)
    make_contact_sheet(args.output, args.contact_sheet, int(output_meta["frames"]))

    section_counts = [
        int(contract["intro_frames"]),
        mission_frames,
        int(contract["transition_frames"]),
        dynamic_frames,
        int(contract["ending_frames"]),
    ]
    section_names = [
        "intro_card",
        "wide_path_traced_office_mission",
        "dynamic_safety_transition_card",
        "framed_dynamic_safety_evidence",
        "mission_complete_card",
    ]
    timeline = []
    first = 0
    for name, count in zip(section_names, section_counts):
        timeline.append(
            {"section": name, "start_frame": first, "end_frame": first + count - 1, "frames": count}
        )
        first += count

    source_duration_error = abs(
        dynamic_frames / float(contract["fps"]) - float(dynamic_meta["duration_s"])
    )
    checks = {
        "phase7f_mission_source_accepted": mission_acceptance.get("passed") is True,
        "dynamic_safety_sources_accepted": dynamic_video_report.get("passed") is True
        and dynamic_run_report.get("passed") is True,
        "mission_source_hash_linked": mission_acceptance["video"]["sha256"]
        == sha256(args.mission_video),
        "dynamic_source_hash_linked": dynamic_video_report["output_video_sha256"]
        == sha256(args.dynamic_video),
        "all_wide_mission_frames_retained": mission_frames
        == int(contract["expected_wide_mission_frames"]),
        "dynamic_duration_preserved_within_one_target_frame": source_duration_error
        <= 1.0 / float(contract["fps"]),
        "full_hd_24fps_output": [output_meta["width"], output_meta["height"]]
        == contract["resolution"]
        and math.isclose(float(output_meta["fps"]), float(contract["fps"]), abs_tol=0.01),
        "expected_frame_count": output_meta["frames"] == contract["expected_total_frames"],
        "contact_sheet_created": args.contact_sheet.is_file(),
        "no_physical_release": profile["presentation_disclosures"]["physical_release"] is False,
    }
    report = {
        "report_type": "administration_nav2_phase7g_presentation_freeze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "status": "presentation_freeze_built" if all(checks.values()) else "freeze_build_failed",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "output": {
            "video": str(args.output.resolve()),
            "video_sha256": sha256(args.output),
            "video_size_bytes": args.output.stat().st_size,
            "contact_sheet": str(args.contact_sheet.resolve()),
            "contact_sheet_sha256": sha256(args.contact_sheet),
            **output_meta,
        },
        "timeline": timeline,
        "sources": {
            "wide_mission": {
                "video_sha256": sha256(args.mission_video),
                "acceptance_sha256": sha256(args.mission_acceptance),
                "metadata": mission_meta,
                "frames_retained_once_in_order": mission_frames,
            },
            "dynamic_safety": {
                "video_sha256": sha256(args.dynamic_video),
                "video_report_sha256": sha256(args.dynamic_video_report),
                "run_report_sha256": sha256(args.dynamic_run_report),
                "metadata": dynamic_meta,
                "target_frames": dynamic_frames,
                "first_selected_source_frame": selected_dynamic_indices[0],
                "last_selected_source_frame": selected_dynamic_indices[-1],
                "duration_error_s": source_duration_error,
                "frame_rate_normalized_only": True,
            },
        },
        "assembly": {
            "mission_motion_changed": False,
            "dynamic_motion_retimed": False,
            "dynamic_frames_resampled_for_target_rate": True,
            "dynamic_evidence_window_scaled_and_framed": True,
            "audio_track": False,
        },
        "claim_boundary": profile["presentation_disclosures"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PHASE7G_FREEZE_BUILT passed={report['passed']} "
        f"checks={report['checks_passed']}/{report['checks_total']} "
        f"frames={output_meta['frames']} duration={output_meta['duration_s']:.3f}s "
        f"video={args.output.resolve()}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
