#!/usr/bin/env python3
"""Create a labeled presentation cut from a successful live administration run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2


PHASE_LABELS = {
    0: "Central atrium departure",
    1: "Transit to Vice Principal wing",
    2: "Vice Principal office approach",
    3: "Vice Principal entry | interior appearance assumed (locked)",
    4: "Vice Principal departure | interior appearance assumed (locked)",
    5: "Return through east hallway",
    6: "Principal suite turn",
    7: "Principal office approach",
    8: "Principal office entry",
    9: "Principal office departure",
    10: "Return to central atrium",
    11: "Mission return home",
}


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


def phase_at(elapsed_s: float, waypoint_events: list[dict[str, object]]) -> str:
    for event in waypoint_events:
        if elapsed_s <= float(event["elapsed_s"]):
            return PHASE_LABELS.get(int(event["segment_id"]), "Administration mission")
    return PHASE_LABELS[11]


def completed_waypoints_at(elapsed_s: float, waypoint_events: list[dict[str, object]]) -> int:
    return sum(float(event["elapsed_s"]) <= elapsed_s for event in waypoint_events)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--build-report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "results"
        / "administration_build_report.json",
    )
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--skip-seconds", type=float, default=0.4)
    parser.add_argument(
        "--camera-mode",
        choices=("follow", "cinematic"),
        default=None,
        help="Override camera metadata when the evidence report was replayed headlessly.",
    )
    args = parser.parse_args()
    if args.speed < 1:
        parser.error("--speed must be positive")

    run = json.loads(args.run_report.read_text(encoding="utf-8"))
    build = json.loads(args.build_report.read_text(encoding="utf-8"))
    if run.get("outcome") != "success" or run.get("waypoints_completed") != 12:
        raise RuntimeError("live run report is not a successful 12-segment route")
    if run.get("root_transform_animation") is not False:
        raise RuntimeError("live run does not explicitly disable root-transform animation")
    waypoint_events = run.get("waypoint_events", [])
    if len(waypoint_events) != 12:
        raise RuntimeError("live run does not contain 12 timestamped waypoint events")
    controls = run.get("control_steps", {})
    policy_only = (
        run.get("route_control") == "policy-only"
        and int(controls.get("physics_supervisor_turn", -1)) == 0
        and int(controls.get("presentation_dwell", -1)) == 0
    )
    learned_skill_ensemble = (
        run.get("policy_architecture") == "route_planner_selected_learned_skill_ensemble"
    )
    phase3n_dynamic_safety = run.get("policy_architecture") == (
        "frozen_phase3m_recovery_stack_plus_outer_recurrent_360_degree_brake_layer"
    )
    measured_door_safety = run.get("policy_architecture") == (
        "ppo_route_policy_plus_deterministic_mapped_doorway_safety"
    )
    control_label = (
        "Wheel physics | 360-degree rays | frozen Phase 3M + learned brake safety | no supervisor"
        if policy_only and phase3n_dynamic_safety
        else "Wheel physics | PPO route policy + mapped doorway safety | no scripted trajectory"
        if policy_only and measured_door_safety
        else
        "Wheel physics | LD19-style rays | PPO base + imitation specialist | no supervisor"
        if policy_only and learned_skill_ensemble
        else "Wheel physics | LD19-style rays | PPO policy-only control | no turn supervisor"
        if policy_only
        else "Wheel physics | LD19-style rays | PPO checkpoint | physical turn supervisor"
    )
    trace = run.get("pose_trace", [])
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("live run has no pose trace for synchronized telemetry")

    source = cv2.VideoCapture(str(args.input))
    if not source.isOpened():
        raise RuntimeError(f"could not open {args.input}")
    source_fps = float(source.get(cv2.CAP_PROP_FPS))
    source_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (1280, 720):
        raise RuntimeError(f"expected 1280x720 live capture, got {width}x{height}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    working = args.output.with_name(args.output.stem + ".working.mp4")
    writer = cv2.VideoWriter(
        str(working),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {working}")

    start_frame = round(args.skip_seconds * source_fps)
    source.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_index = start_frame
    output_frames = 0
    trace_index = 0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            if (source_index - start_frame) % args.speed:
                source_index += 1
                continue

            elapsed_s = source_index / source_fps
            while (
                trace_index + 1 < len(trace)
                and float(trace[trace_index + 1]["elapsed_s"]) <= elapsed_s
            ):
                trace_index += 1
            telemetry = trace[trace_index]
            minimum_range = float(telemetry.get("minimum_lidar_range_m", 0.0))
            range_label = f"{minimum_range:.2f} m" if math.isfinite(minimum_range) else "n/a"
            linear_velocity = float(telemetry.get("linear_velocity_mps", 0.0))
            yaw_rate = float(telemetry.get("yaw_rate_rad_s", 0.0))
            action = telemetry.get("policy_action", [0.0, 0.0])
            applied_command = telemetry.get(
                "applied_frozen_stack_command", action
            )
            if len(action) == 1 and len(applied_command) >= 2:
                action_label = (
                    f"safety {float(action[0]):+.2f} | stack "
                    f"[{float(applied_command[0]):+.2f}, "
                    f"{float(applied_command[1]):+.2f}]"
                )
            else:
                action_label = (
                    f"action [{float(action[0]):+.2f}, "
                    f"{float(action[1]):+.2f}]"
                )
            waypoint_count = completed_waypoints_at(elapsed_s, waypoint_events)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 96), (10, 22, 28), -1)
            cv2.rectangle(overlay, (0, 646), (1280, 720), (10, 22, 28), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0.0, frame)
            cv2.putText(
                frame,
                "AI-SHA | ISAAC SIM + ISAAC LAB | LIVE POLICY",
                (32, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (246, 250, 251),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                phase_at(elapsed_s, waypoint_events),
                (32, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (113, 226, 190),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"sim {elapsed_s:05.1f}s | {args.speed}x playback",
                (1000, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (222, 229, 232),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"v {linear_velocity:+.2f} m/s | yaw {yaw_rate:+.2f} rad/s | {action_label}",
                (700, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (205, 231, 224),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                control_label,
                (32, 676),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (242, 246, 247),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                (
                    f"waypoints {waypoint_count}/12 | ray {range_label} | "
                    "VP 0.85 m | Principal 0.90 m | 0.20 m polygon NO-GO | 0 collisions"
                    if measured_door_safety
                    else f"waypoints {waypoint_count}/12 | nearest LD19-style ray "
                    f"{range_label} | 0 collisions"
                ),
                (32, 704),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (151, 224, 202),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            output_frames += 1
            source_index += 1
    finally:
        source.release()
        writer.release()

    ffmpeg = find_ffmpeg()
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
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

    check = cv2.VideoCapture(str(args.output))
    encoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = float(check.get(cv2.CAP_PROP_FPS))
    check.release()
    if encoded_frames != output_frames:
        raise RuntimeError(f"encoded {encoded_frames} frames, expected {output_frames}")

    report = {
        "report_type": "administration_live_policy_presentation_video",
        "source_video": str(args.input.resolve()),
        "source_video_sha256": sha256_file(args.input),
        "source_run_report": str(args.run_report.resolve()),
        "source_run_report_sha256": sha256_file(args.run_report),
        "checkpoint": run["checkpoint"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "policy_architecture": run.get("policy_architecture"),
        "phase3n_dynamic_safety_overlay": phase3n_dynamic_safety,
        "measured_doorway_safety_layer": measured_door_safety,
        "source_build_report": str(args.build_report.resolve()),
        "source_build_report_sha256": sha256_file(args.build_report),
        "segment_policy_checkpoints": run.get("segment_policy_checkpoints", {}),
        "output_video": str(args.output.resolve()),
        "output_video_sha256": sha256_file(args.output),
        "source_frame_count": source_frames,
        "source_duration_s": source_frames / source_fps,
        "presentation_frame_count": encoded_frames,
        "presentation_duration_s": encoded_frames / encoded_fps,
        "speed_multiplier": args.speed,
        "motion_changed": False,
        "policy_only_control": policy_only,
        "telemetry_overlay": True,
        "telemetry_source_trace_records": len(trace),
        "camera_mode": args.camera_mode or run.get("camera", {}).get("mode"),
        "passed": True,
        "disclosure": (
            "Temporally sampled and labeled from the complete successful live-policy capture; "
            "robot motion is unchanged. The VP interior appearance is assumed because it was locked; "
            "the 0.85 m VP door uses the reported administration minimum, the 0.90 m Principal door is a "
            "presentation assumption, and the 0.20 m central drop is a mapped no-go. This is not evidence "
            "of physical deployment readiness."
        ),
        "geometry_disclosure": {
            "doors": build.get("doors", {}),
            "central_atrium_drop": build.get("central_atrium_drop", {}),
            "capture_limitations": build.get("capture_limitations", {}),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
