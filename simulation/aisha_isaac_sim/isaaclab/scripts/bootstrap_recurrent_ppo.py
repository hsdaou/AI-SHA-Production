#!/usr/bin/env python3
"""Convert a recurrent distillation student into a fresh recurrent PPO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticRecurrent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_prefix(
    target: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    target_prefix: str,
    source_prefix: str,
) -> list[str]:
    copied: list[str] = []
    for target_key, target_value in target.items():
        if not target_key.startswith(target_prefix):
            continue
        source_key = source_prefix + target_key.removeprefix(target_prefix)
        if source_key in source and source[source_key].shape == target_value.shape:
            target[target_key] = source[source_key].detach().clone()
            copied.append(target_key)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distilled-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.distilled_checkpoint.expanduser().resolve()
    if not source_path.is_file():
        parser.error(f"distilled checkpoint does not exist: {source_path}")
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = source_checkpoint["model_state_dict"]
    if not any(key.startswith("memory_s.") for key in source_state):
        raise ValueError("checkpoint does not contain a recurrent distillation student")

    observations = TensorDict(
        {"policy": torch.zeros((1, 46), dtype=torch.float32)},
        batch_size=[1],
    )
    policy = ActorCriticRecurrent(
        observations,
        {"policy": ["policy"], "critic": ["policy"]},
        num_actions=2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=0.20,
        noise_std_type="scalar",
        rnn_type="gru",
        rnn_hidden_dim=46,
        rnn_num_layers=1,
    )
    target_state = policy.state_dict()
    copied: list[str] = []
    copied += _copy_prefix(target_state, source_state, "memory_a.", "memory_s.")
    copied += _copy_prefix(target_state, source_state, "actor.", "student.")
    copied += _copy_prefix(
        target_state,
        source_state,
        "actor_obs_normalizer.",
        "student_obs_normalizer.",
    )
    # Give the critic the same temporal encoder and observation statistics, but
    # leave its value head fresh. Hidden critic MLP layers are copied only when
    # their shapes match the student's feature layers.
    copied += _copy_prefix(target_state, source_state, "memory_c.", "memory_s.")
    copied += _copy_prefix(
        target_state,
        source_state,
        "critic_obs_normalizer.",
        "student_obs_normalizer.",
    )
    for layer in ("0", "2", "4"):
        copied += _copy_prefix(target_state, source_state, f"critic.{layer}.", f"student.{layer}.")
    if "std" in source_state and source_state["std"].shape == target_state["std"].shape:
        target_state["std"] = source_state["std"].detach().clone().clamp_min(0.05)
        copied.append("std")
    policy.load_state_dict(target_state, strict=True)

    # RSL-RL resumes PPO only from a complete checkpoint. An empty Adam state
    # with matching parameter groups deliberately starts a new optimizer while
    # retaining the distilled actor and temporal encoder.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
    output_path = args.output_checkpoint.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": 0,
            "infos": {
                "bootstrap": "phase3_recurrent_distillation_student_to_ppo",
                "source_checkpoint": str(source_path),
            },
        },
        output_path,
    )
    output_sha256 = _sha256(output_path)
    (output_path.parent / "checkpoint.sha256").write_text(
        f"{output_sha256}  {output_path.name}\n",
        encoding="utf-8",
    )

    report = {
        "report_type": "phase3_recurrent_ppo_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_distillation_checkpoint": str(source_path),
        "source_distillation_sha256": _sha256(source_path),
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": output_sha256,
        "policy": {
            "class": "ActorCriticRecurrent",
            "rnn_type": "gru",
            "rnn_hidden_dim": 46,
            "rnn_num_layers": 1,
            "observation_count": 46,
            "action_count": 2,
        },
        "copied_tensor_keys": sorted(copied),
        "copied_tensor_count": len(copied),
        "critic_value_head_initialized_fresh": True,
        "optimizer_initialized_fresh": True,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE3_RECURRENT_BOOTSTRAP_READY checkpoint={output_path} sha256={output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
