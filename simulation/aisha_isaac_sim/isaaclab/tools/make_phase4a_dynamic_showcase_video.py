#!/usr/bin/env python3
"""Create a synchronized presentation cut from a passing Phase 4A live run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_report.read_text(encoding="utf-8"))
    if run.get("report_type") != "phase4a_live_dynamic_safety_showcase":
        raise RuntimeError("input report is not a Phase 4A showcase")
    if not run.get("passed") or run.get("outcome") != "success":
        raise RuntimeError("Phase 4A source run did not pass")
    if not all(run.get("checks", {}).values()):
        raise RuntimeError("Phase 4A source run contains a failed check")
    policy_contract = run.get("policy_contract", {})
    if policy_contract.get("physics_supervisor") is not False:
        raise RuntimeError("source run does not exclude a physics supervisor")
    if policy_contract.get("root_transform_animation") is not False:
        raise RuntimeError("source run does not exclude root-transform animation")
    trace = run.get("trace", [])
    if not trace:
        raise RuntimeError("source run has no synchronized trace")

    source = cv2.VideoCapture(str(args.input))
    if not source.isOpened():
        raise RuntimeError(f"could not open {args.input}")
    fps = float(source.get(cv2.CAP_PROP_FPS))
    source_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (1280, 720):
        raise RuntimeError(f"expected 1280x720 source, got {width}x{height}")
    if not 29.0 <= fps <= 31.0:
        raise RuntimeError(f"expected a 30 fps source, got {fps}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    working = args.output.with_name(args.output.stem + ".working.mp4")
    writer = cv2.VideoWriter(
        str(working),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {working}")

    trace_index = 0
    latest_authority_step = -10_000
    output_frames = 0
    encounter_authority_frames = 0
    protective_stop_frames = 0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            simulation_step = output_frames + 1
            while (
                trace_index + 1 < len(trace)
                and int(trace[trace_index + 1]["step"]) <= simulation_step
            ):
                trace_index += 1
            telemetry = trace[trace_index]
            if bool(telemetry["safety_authority_active"]):
                latest_authority_step = simulation_step
            learned_active = simulation_step - latest_authority_step <= 5
            protective_stop = bool(telemetry["protective_stop_latched"])
            progress = float(telemetry["pedestrian_crossing_progress"])
            triggered = bool(telemetry["pedestrian_triggered"])
            speed = float(telemetry["linear_velocity_mps"])
            clearance = float(telemetry["minimum_360_ring_clearance_m"])
            brake = float(telemetry["learned_brake_fraction"])
            if learned_active and triggered and progress < 1.0:
                encounter_authority_frames += 1
            if protective_stop and triggered and progress < 1.0:
                protective_stop_frames += 1

            top = frame.copy()
            cv2.rectangle(top, (0, 0), (1280, 94), (8, 18, 24), -1)
            cv2.rectangle(top, (0, 626), (1280, 720), (8, 18, 24), -1)
            cv2.addWeighted(top, 0.84, frame, 0.16, 0.0, frame)

            cv2.putText(
                frame,
                "AI-SHA | ISAAC SIM + ISAAC LAB | LIVE CHECKPOINT",
                (30, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.76,
                (244, 249, 250),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Principal office approach | deterministic pedestrian crossing",
                (30, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (164, 205, 215),
                1,
                cv2.LINE_AA,
            )

            if protective_stop:
                status = "PROTECTIVE STOP | PERSON CROSSING"
                status_color = (48, 178, 255)
            elif progress >= 1.0 and speed >= 0.20:
                status = "PATH CLEAR | RESUMING MISSION"
                status_color = (105, 225, 170)
            elif triggered:
                status = "DYNAMIC OBSTACLE DETECTED"
                status_color = (74, 202, 242)
            else:
                status = "LIVE POLICY APPROACH"
                status_color = (155, 206, 218)
            (status_width, _), _ = cv2.getTextSize(
                status, cv2.FONT_HERSHEY_SIMPLEX, 0.57, 2
            )
            cv2.putText(
                frame,
                status,
                (1248 - status_width, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                status_color,
                2,
                cv2.LINE_AA,
            )
            learned_label = (
                "LEARNED 360 DEG BRAKE: ACTIVE"
                if learned_active
                else "LEARNED 360 DEG BRAKE: MONITORING"
            )
            learned_color = (105, 225, 170) if learned_active else (160, 182, 189)
            (learned_width, _), _ = cv2.getTextSize(
                learned_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            cv2.putText(
                frame,
                learned_label,
                (1248 - learned_width, 71),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                learned_color,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"speed {speed:+.2f} m/s   |   360 clearance {clearance:.2f} m   |   "
                    f"learned brake {brake * 100:04.1f}%   |   crossing {progress * 100:05.1f}%"
                ),
                (30, 661),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (223, 236, 238),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                (
                    "Wheel physics | frozen Phase 3M navigation + learned Phase 3N brake | "
                    "no scripted robot motion | zero contacts"
                ),
                (30, 699),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (150, 224, 202),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            output_frames += 1
    finally:
        source.release()
        writer.release()

    if output_frames != source_frames:
        raise RuntimeError(
            f"encoded {output_frames} frames from a {source_frames}-frame source"
        )
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

    checks = {
        "source_run_passed": run["passed"] is True,
        "source_run_success": run["outcome"] == "success",
        "source_checks_all_passed": all(run["checks"].values()),
        "checkpoint_policy_only_robot_motion": (
            policy_contract["physics_supervisor"] is False
            and policy_contract["root_transform_animation"] is False
        ),
        "learned_authority_visible_in_overlay": encounter_authority_frames > 0,
        "protective_stop_visible_in_overlay": protective_stop_frames > 0,
        "all_source_frames_encoded": output_frames == source_frames,
        "presentation_video_created": args.output.is_file()
        and args.output.stat().st_size > 0,
    }
    report = {
        "report_type": "phase4a_dynamic_safety_presentation_video",
        "passed": all(checks.values()),
        "source_video": str(args.input.resolve()),
        "source_video_sha256": sha256_file(args.input),
        "source_run_report": str(args.run_report.resolve()),
        "source_run_report_sha256": sha256_file(args.run_report),
        "output_video": str(args.output.resolve()),
        "output_video_sha256": sha256_file(args.output),
        "resolution": [width, height],
        "fps": fps,
        "frames": output_frames,
        "duration_s": output_frames / fps,
        "encounter_learned_authority_overlay_frames": encounter_authority_frames,
        "protective_stop_overlay_frames": protective_stop_frames,
        "checks": checks,
        "overlay_disclosure": (
            "The learned Phase 3N actor may reduce translation only when the 360-degree "
            "clearance gate grants authority. The separate frozen Phase 3M protective-stop "
            "state is labeled independently; the video does not attribute the entire stop "
            "to the outer actor."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
