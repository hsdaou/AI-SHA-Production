#!/usr/bin/env python3
"""Record the scoped local privacy review for the Phase 7M presentation reel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


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
        "--encode-report",
        type=Path,
        default=ROOT / "results/phase7m_nurec_reel_encode.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase7m_nurec_reel_privacy_review.json",
    )
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


def is_git_tracked(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.parent.parent)
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=ROOT.parent.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    args = parse_args()
    profile_path = args.profile.expanduser().resolve()
    render_path = args.render_report.expanduser().resolve()
    encode_path = args.encode_report.expanduser().resolve()
    for path in (profile_path, render_path, encode_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    render = json.loads(render_path.read_text(encoding="utf-8"))
    encode = json.loads(encode_path.read_text(encoding="utf-8"))
    selected_codes = sorted(set(int(value) for value in render["camera_codes_rendered"]))
    expected_codes = sorted(int(value) for value in profile["privacy"]["selected_camera_codes"])
    video = ROOT / encode["output"]
    frames = [ROOT / item["path"] for item in render["frames"]]
    local_assets_untracked = bool(
        video.is_file()
        and not is_git_tracked(video)
        and all(path.is_file() and not is_git_tracked(path) for path in frames)
    )
    local_preview_passed = bool(
        render.get("passed")
        and encode.get("passed")
        and selected_codes == expected_codes
        and profile["privacy"]["excludes_close_certificate_and_portrait_views"]
        and profile["privacy"]["excludes_live_people"]
        and local_assets_untracked
    )
    report = {
        "report_type": "phase7m_nurec_reel_privacy_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_local_preview_only" if local_preview_passed else "failed",
        "passed_local_preview": local_preview_passed,
        "profile": portable_path(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "render_report": portable_path(render_path),
        "render_report_sha256": sha256_file(render_path),
        "encode_report": portable_path(encode_path),
        "encode_report_sha256": sha256_file(encode_path),
        "selected_camera_codes": selected_codes,
        "reviewed_visual_scope": [
            "widescreen source-camera contact sheet",
            "robot-overlay contact sheets",
            "final Full HD render contact sheet",
            "one-frame-per-second encoded-video contact sheet",
        ],
        "observations": {
            "live_people_observed": False,
            "close_certificate_views_included": False,
            "close_portrait_views_included": False,
            "raw_capture_included": False,
            "student_biometric_content_observed": False,
            "selected_signage_or_documents_readable_as_sensitive": False,
            "robot_upright_and_floor_aligned": True,
            "robot_does_not_fill_most_of_frame": True,
        },
        "local_media_and_frames_untracked": local_assets_untracked,
        "coding_agent_visual_selection_review_completed": True,
        "authorized_human_privacy_review_completed": False,
        "external_distribution_approved": False,
        "external_distribution_requires_user_review": True,
        "safe_use": "local presentation preview on the authorized project machine",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if local_preview_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
