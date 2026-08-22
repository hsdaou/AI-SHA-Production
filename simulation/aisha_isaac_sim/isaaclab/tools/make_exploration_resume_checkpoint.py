#!/usr/bin/env python3
"""Reset only PPO action exploration in a hash-locked RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--linear-std", type=float, default=2.0)
    parser.add_argument("--angular-std", type=float, default=0.6)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    source_sha256 = sha256_file(source)
    if source_sha256 != args.expected_source_sha256:
        raise ValueError(
            f"source checkpoint hash mismatch: {source_sha256} != {args.expected_source_sha256}"
        )
    if args.linear_std <= 0.0 or args.angular_std <= 0.0:
        raise ValueError("exploration standard deviations must be positive")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    old_std = state["std"].detach().cpu().tolist()
    new_std = state["std"].new_tensor([args.linear_std, args.angular_std])
    state["std"] = new_std
    if checkpoint.get("infos") is None:
        checkpoint["infos"] = {}
    checkpoint["infos"]["aisha_exploration_reset"] = {
        "source_sha256": source_sha256,
        "old_std": old_std,
        "new_std": new_std.detach().cpu().tolist(),
        "network_weights_changed": False,
        "normalizers_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    report = {
        "report_type": "phase2b_exploration_resume_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": source_sha256},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "old_action_std": old_std,
        "new_action_std": new_std.detach().cpu().tolist(),
        "network_weights_changed": False,
        "normalizers_changed": False,
        "purpose": (
            "restore PPO exploration around a saturated inherited forward mean so live-scene "
            "collision outcomes can teach a near-zero-speed office pivot"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
