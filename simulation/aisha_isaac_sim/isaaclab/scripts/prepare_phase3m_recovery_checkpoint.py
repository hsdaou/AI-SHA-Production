#!/usr/bin/env python3
"""Prepare a hash-verified Phase 3M resume checkpoint from Phase 3L model 200."""

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
    parser.add_argument("--brake-std", type=float, default=0.10)
    parser.add_argument("--steering-std", type=float, default=0.45)
    parser.add_argument("--brake-bias-shift", type=float, default=0.0)
    parser.add_argument("--steering-bias-shift", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve() if args.report else None
    actual_source_sha256 = sha256(source)
    if actual_source_sha256 != args.expected_sha256:
        raise SystemExit(
            "source hash mismatch: "
            f"expected {args.expected_sha256}, got {actual_source_sha256}"
        )
    if min(args.brake_std, args.steering_std, args.learning_rate) <= 0.0:
        raise SystemExit("exploration deviations and learning rate must be positive")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_model_state = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in checkpoint["model_state_dict"].items()
    }
    std = checkpoint["model_state_dict"]["std"]
    if not isinstance(std, torch.Tensor) or std.numel() != 2:
        raise SystemExit(f"expected two-action exploration std tensor, got {std!r}")
    checkpoint["model_state_dict"]["std"] = torch.tensor(
        [args.brake_std, args.steering_std], dtype=std.dtype, device=std.device
    ).reshape_as(std)
    final_actor_bias_key = "actor.4.bias"
    final_actor_bias = checkpoint["model_state_dict"].get(final_actor_bias_key)
    if not isinstance(final_actor_bias, torch.Tensor) or final_actor_bias.numel() != 2:
        raise SystemExit(
            f"expected two-action final actor bias at {final_actor_bias_key}"
        )
    final_actor_bias = final_actor_bias.clone()
    final_actor_bias[0] += args.brake_bias_shift
    final_actor_bias[1] += args.steering_bias_shift
    checkpoint["model_state_dict"][final_actor_bias_key] = final_actor_bias

    optimizer_state = checkpoint["optimizer_state_dict"]
    optimizer_state["state"] = {}
    for group in optimizer_state["param_groups"]:
        group["lr"] = args.learning_rate
        if "initial_lr" in group:
            group["initial_lr"] = args.learning_rate
    checkpoint["iter"] = 0
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["phase3m_targeted_recovery_reset"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": actual_source_sha256,
        "brake_std": args.brake_std,
        "steering_std": args.steering_std,
        "learning_rate": args.learning_rate,
        "brake_bias_shift": args.brake_bias_shift,
        "steering_bias_shift": args.steering_bias_shift,
        "optimizer_moments_reset": True,
        "iteration_reset": True,
        "changed_model_keys": [final_actor_bias_key]
        if args.brake_bias_shift != 0.0 or args.steering_bias_shift != 0.0
        else [],
        "hidden_actor_critic_recurrent_and_normalizers_changed": False,
    }

    intentionally_changed_keys = {"std", final_actor_bias_key}
    unchanged_keys = [
        key for key in source_model_state if key not in intentionally_changed_keys
    ]
    unchanged = all(
        torch.equal(source_model_state[key], checkpoint["model_state_dict"][key])
        if isinstance(source_model_state[key], torch.Tensor)
        else source_model_state[key] == checkpoint["model_state_dict"][key]
        for key in unchanged_keys
    )
    if not unchanged:
        raise SystemExit("actor, critic, recurrent memory, or normalizer state changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    output_sha256 = sha256(output)
    checksum_path = output.parent / "checkpoint.sha256"
    checksum_path.write_text(f"{output_sha256}  {output.name}\n", encoding="utf-8")
    report = {
        "report_type": "phase3m_targeted_recovery_ppo_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source),
        "source_sha256": actual_source_sha256,
        "output_checkpoint": str(output),
        "output_sha256": output_sha256,
        "checksum_file": str(checksum_path),
        "brake_std": args.brake_std,
        "steering_std": args.steering_std,
        "brake_bias_shift": args.brake_bias_shift,
        "steering_bias_shift": args.steering_bias_shift,
        "learning_rate": args.learning_rate,
        "optimizer_moments_reset": True,
        "iteration_reset": True,
        "changed_model_keys": [final_actor_bias_key]
        if args.brake_bias_shift != 0.0 or args.steering_bias_shift != 0.0
        else [],
        "hidden_actor_critic_recurrent_and_normalizers_changed": False,
        "claim_boundary": (
            "Checkpoint initialization evidence only. Optional final actor-head bias "
            "shifts are an auditable exploration warm start; they do not establish "
            "trained-policy performance or safety acceptance."
        ),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
