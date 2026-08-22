#!/usr/bin/env python3
"""Behavior-clone a safe pivot-then-drive maneuver in the live administration scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--expected-checkpoint-sha256", required=True)
parser.add_argument("--output-checkpoint", type=Path, required=True)
parser.add_argument("--output-report", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--num-envs", type=int, default=12)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--anchor-weight", type=float, default=1.0e-3)
parser.add_argument("--seed", type=int, default=7099)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import aisha_isaaclab.tasks  # noqa: E402,F401


TASK = "Isaac-AISHA-Administration-Live-Rehearsal-Direct-v0"
PRINCIPAL_TURN_SEGMENT_ID = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_actions(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Continuous pivot-then-drive controller used only to label demonstrations."""

    policy_obs = observations["policy"]
    distance = policy_obs[:, 2] * 12.0
    heading_error = torch.atan2(policy_obs[:, 3], policy_obs[:, 4])
    # Suppress translation during the large initial turn, then increase speed
    # continuously as alignment improves. This avoids a discrete mode boundary.
    alignment = torch.clamp(torch.cos(heading_error), min=0.0).pow(4)
    speed_mps = torch.minimum(torch.full_like(distance, 0.34), distance * 0.45) * alignment
    linear_action = (speed_mps / 0.25 - 1.0).clamp(-1.0, 1.0)
    angular_action = (2.5 * heading_error).clamp(-1.0, 1.0)
    return torch.stack((linear_action, angular_action), dim=-1)


def main() -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    output_checkpoint = args.output_checkpoint.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.num_envs < 12:
        raise ValueError("balanced retention collection requires at least 12 environments")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint hash mismatch: {checkpoint_sha256} != {args.expected_checkpoint_sha256}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    env_cfg.balanced_segment_assignment = True
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = "cuda:0"
    raw_env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    source_infos = runner.load(str(checkpoint), load_optimizer=True)
    policy_nn = runner.alg.policy
    policy_nn.eval()

    observations = env.get_observations()
    expert_obs: list[torch.Tensor] = []
    expert_targets: list[torch.Tensor] = []
    retention_obs: list[torch.Tensor] = []
    retention_targets: list[torch.Tensor] = []
    outcomes = {"success": 0, "collision": 0, "time_out": 0}
    completed = 0
    collection_step = 0
    while completed < args.episodes and simulation_app.is_running():
        with torch.inference_mode():
            teacher_actions = policy_nn.act_inference(observations)
            labels = expert_actions(observations)
            segment_ids = env.unwrapped._segment_ids.clone()  # noqa: SLF001
            expert_mask = segment_ids == PRINCIPAL_TURN_SEGMENT_ID
            if not torch.any(expert_mask):
                raise RuntimeError("balanced assignment did not allocate a Principal-turn environment")
            actions = teacher_actions.clone()
            actions[expert_mask] = labels[expert_mask]
            expert_obs.append(observations["policy"][expert_mask].detach().clone())
            expert_targets.append(labels[expert_mask].detach().clone())

            # Retain an equal number of original-policy samples per step so the
            # new maneuver cannot dominate or be diluted by the other 11 legs.
            retention_ids = torch.nonzero(~expert_mask, as_tuple=False).flatten()
            retain_count = int(expert_mask.sum().item())
            offset = (collection_step * retain_count) % int(retention_ids.numel())
            rolled_ids = torch.roll(retention_ids, shifts=-offset)
            selected_ids = rolled_ids[:retain_count]
            retention_obs.append(observations["policy"][selected_ids].detach().clone())
            retention_targets.append(teacher_actions[selected_ids].detach().clone())
            observations, _, dones, extras = env.step(actions)
            collection_step += 1
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue
        episode_outcomes = extras["episode_outcomes"]
        for env_id in done_ids.tolist():
            if completed >= args.episodes:
                break
            segment_id = int(episode_outcomes["segment_id"][env_id].item())
            if segment_id != PRINCIPAL_TURN_SEGMENT_ID:
                continue
            outcome = "time_out"
            if bool(episode_outcomes["success"][env_id]):
                outcome = "success"
            elif bool(episode_outcomes["collision"][env_id]):
                outcome = "collision"
            outcomes[outcome] += 1
            completed += 1
        policy_nn.reset(dones)

    if outcomes["success"] != args.episodes:
        raise RuntimeError(f"expert demonstration gate failed: {outcomes}")

    raw_expert_obs = torch.cat(expert_obs, dim=0)
    raw_retention_obs = torch.cat(retention_obs, dim=0)
    raw_obs = torch.cat((raw_expert_obs, raw_retention_obs), dim=0)
    targets = torch.cat((torch.cat(expert_targets, dim=0), torch.cat(retention_targets, dim=0)), dim=0)
    with torch.inference_mode():
        normalized_obs = policy_nn.actor_obs_normalizer(raw_obs).detach()

    actor = policy_nn.actor
    anchors = {name: value.detach().clone() for name, value in actor.named_parameters()}
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=normalized_obs.device).manual_seed(args.seed + 1)
    initial_loss = None
    final_loss = None
    actor.train()
    for _ in range(args.epochs):
        permutation = torch.randperm(normalized_obs.shape[0], generator=generator, device=normalized_obs.device)
        for start in range(0, normalized_obs.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            prediction = actor(normalized_obs[indices])
            imitation_loss = torch.mean(torch.square(prediction - targets[indices]))
            anchor_loss = torch.zeros((), device=normalized_obs.device)
            for name, parameter in actor.named_parameters():
                anchor_loss += torch.mean(torch.square(parameter - anchors[name]))
            loss = imitation_loss + args.anchor_weight * anchor_loss
            if initial_loss is None:
                initial_loss = float(imitation_loss.detach().item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            final_loss = float(imitation_loss.detach().item())
    actor.eval()

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_infos = dict(source_infos or {})
    bootstrap_infos["aisha_principal_turn_behavior_cloning"] = {
        "source_sha256": checkpoint_sha256,
        "expert": "continuous pivot-then-drive labels in live administration physics",
        "demonstration_episodes": args.episodes,
        "demonstration_successes": outcomes["success"],
        "network_weights_changed": True,
        "normalizers_changed": False,
    }
    runner.save(str(output_checkpoint), infos=bootstrap_infos)
    output_sha256 = sha256_file(output_checkpoint)

    report = {
        "report_type": "phase2_principal_turn_behavior_cloning_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "seed": args.seed,
        "source_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "output_checkpoint": {"path": str(output_checkpoint), "sha256": output_sha256},
        "demonstrations": {
            "episodes": args.episodes,
            "parallel_environments": args.num_envs,
            "expert_samples": int(raw_expert_obs.shape[0]),
            "original_policy_retention_samples": int(raw_retention_obs.shape[0]),
            "total_samples": int(raw_obs.shape[0]),
            **outcomes,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "anchor_weight": args.anchor_weight,
            "initial_imitation_mse": initial_loss,
            "final_imitation_mse": final_loss,
        },
        "claim_boundary": (
            "The expert controller labels Principal-turn training demonstrations only; the original policy "
            "labels equal-volume retention data on all other route legs. Acceptance requires a separate "
            "deterministic policy-only evaluation with no expert, supervisor, dwell, or root animation."
        ),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
