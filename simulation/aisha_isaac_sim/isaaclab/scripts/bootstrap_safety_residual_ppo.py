#!/usr/bin/env python3
"""Create a zero-output recurrent PPO checkpoint for bounded safety training."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

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
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=0.12,
        noise_std_type="scalar",
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
    )
    final_actor_layer = policy.actor[-1]
    if not isinstance(final_actor_layer, torch.nn.Linear) or final_actor_layer.out_features != 2:
        raise TypeError("unexpected recurrent residual actor output layer")
    torch.nn.init.zeros_(final_actor_layer.weight)
    torch.nn.init.zeros_(final_actor_layer.bias)

    # Deterministic inference must be an exact no-op before PPO starts. The
    # stochastic 0.12 standard deviation still provides bounded exploration.
    policy.eval()
    with torch.inference_mode():
        deterministic_output = policy.act_inference(observations)
    if not torch.equal(deterministic_output, torch.zeros_like(deterministic_output)):
        raise RuntimeError("zero-output safety residual bootstrap invariant failed")
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=5.0e-5)
    output_path = args.output_checkpoint.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": 0,
            "infos": {
                "bootstrap": "phase3_zero_output_bounded_safety_residual",
                "deterministic_zero_output": True,
            },
        },
        output_path,
    )
    output_sha256 = _sha256(output_path)
    (output_path.parent / "checkpoint.sha256").write_text(
        f"{output_sha256}  {output_path.name}\n", encoding="utf-8"
    )

    report = {
        "report_type": "phase3_safety_residual_ppo_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": output_sha256,
        "policy": {
            "class": "ActorCriticRecurrent",
            "rnn_type": "gru",
            "rnn_hidden_dim": 64,
            "rnn_num_layers": 1,
            "observation_count": 46,
            "action_count": 2,
            "actor_hidden_dims": [128, 64],
            "deterministic_initial_action": [0.0, 0.0],
            "stochastic_initial_std": 0.12,
        },
        "action_semantics": {
            "action_0_negative": "brake_fraction; positive values are a no-op",
            "action_1_negative": "bounded steering attenuation; positive values are a no-op",
            "can_increase_speed": False,
            "can_reverse": False,
            "can_flip_steering_sign": False,
        },
        "optimizer_initialized_fresh": True,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE3_SAFETY_RESIDUAL_BOOTSTRAP_READY checkpoint={output_path} sha256={output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
