#!/usr/bin/env python3
"""Create an auditable Phase 2 initializer with restored exploration variance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--linear-std", type=float, default=0.6)
    parser.add_argument("--angular-std", type=float, default=1.0)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    actual_source_sha = sha256(source)
    if actual_source_sha != args.expected_sha256:
        raise SystemExit(
            f"source hash mismatch: expected {args.expected_sha256}, got {actual_source_sha}"
        )
    if min(args.linear_std, args.angular_std) <= 0.0:
        raise SystemExit("exploration standard deviations must be positive")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    std = checkpoint["model_state_dict"]["std"]
    checkpoint["model_state_dict"]["std"] = torch.tensor(
        [args.linear_std, args.angular_std], dtype=std.dtype
    )
    # Preserve the learned actor, critic, and observation normalization while
    # removing stale Adam moments from the earlier low-variance curriculum.
    checkpoint["optimizer_state_dict"]["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["phase2_exploration_reset"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": actual_source_sha,
        "linear_std": args.linear_std,
        "angular_std": args.angular_std,
        "optimizer_moments_reset": True,
        "actor_critic_weights_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": actual_source_sha,
                "output": str(output),
                "output_sha256": sha256(output),
                "linear_std": args.linear_std,
                "angular_std": args.angular_std,
                "optimizer_moments_reset": True,
                "actor_critic_weights_changed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
