#!/usr/bin/env python3
"""Behavior-clone measured-door pivots while retaining the frozen route actor."""

from __future__ import annotations

import argparse
import hashlib
import json
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
parser.add_argument("--episodes-per-segment", type=int, default=4)
parser.add_argument("--num-envs", type=int, default=12)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--anchor-weight", type=float, default=1.0e-3)
parser.add_argument(
    "--retention-ratio",
    type=float,
    default=1.0,
    help="Maximum original-policy samples retained per specialist sample and step.",
)
parser.add_argument("--collection-episode-length-s", type=float, default=70.0)
parser.add_argument("--progress-interval", type=int, default=250)
parser.add_argument("--segments", default="3,4,8,9")
parser.add_argument("--start-lateral-jitter-m", type=float)
parser.add_argument("--start-yaw-jitter-deg", type=float)
parser.add_argument("--goal-jitter-m", type=float)
parser.add_argument("--seed", type=int, default=10635)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import aisha_isaaclab.tasks  # noqa: E402,F401


TASK = "Isaac-AISHA-BlockA-MeasuredTightDoor-SensorNav-Direct-v0"
SPECIALIST_SEGMENTS = tuple(int(value) for value in args.segments.split(",") if value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_tree(value):
    """Detach checkpoint tensors before serialization to avoid CUDA writer stalls."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def expert_actions(observations: dict[str, torch.Tensor], task_env) -> torch.Tensor:
    """Privileged route-centre demonstrations; the deployed actor remains sensor-only."""

    policy_obs = observations["policy"]
    distance = policy_obs[:, 2] * 12.0
    heading_error = torch.atan2(policy_obs[:, 3], policy_obs[:, 4])
    local_xy = task_env._local_xy()  # noqa: SLF001
    quaternion = task_env._robot.data.root_quat_w  # noqa: SLF001
    yaw = torch.atan2(
        2.0
        * (
            quaternion[:, 0] * quaternion[:, 3]
            + quaternion[:, 1] * quaternion[:, 2]
        ),
        1.0
        - 2.0
        * (quaternion[:, 2].square() + quaternion[:, 3].square()),
    )
    alignment_threshold_deg = torch.full_like(heading_error, 0.5)
    staging_active = torch.zeros_like(heading_error, dtype=torch.bool)
    for door_index, segment_pair in enumerate(((3, 4), (8, 9))):
        centre = task_env._tight_door_centres[door_index]  # noqa: SLF001
        normal = task_env._tight_door_normals[door_index]  # noqa: SLF001
        for segment_id, direction_sign in zip(segment_pair, (-1.0, 1.0), strict=True):
            mask = task_env._segment_ids == segment_id  # noqa: SLF001
            # First converge on a short centreline waypoint on the approach
            # side, then target the matching point beyond the frame. A single
            # far goal under-corrects the 2--3 cm lateral translation produced
            # by a heavy in-place castor pivot.
            approach_stage = centre - direction_sign * normal
            crossing_stage = centre + direction_sign * normal
            signed_normal = torch.sum((local_xy - centre) * normal, dim=1)
            approach_depth = -direction_sign * signed_normal
            goal_side_depth = direction_sign * signed_normal
            use_approach_stage = mask & (approach_depth > 1.05)
            stage = torch.where(
                use_approach_stage.unsqueeze(1),
                approach_stage.unsqueeze(0),
                crossing_stage.unsqueeze(0),
            )
            delta = stage - local_xy
            stage_distance = torch.linalg.norm(delta, dim=1)
            stage_heading = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
            stage_heading = torch.atan2(torch.sin(stage_heading), torch.cos(stage_heading))
            use_crossing_stage = mask & ~use_approach_stage & (goal_side_depth < 0.75)
            use_stage = use_approach_stage | use_crossing_stage
            staging_active |= use_stage
            distance = torch.where(use_stage, stage_distance, distance)
            heading_error = torch.where(use_stage, stage_heading, heading_error)
            normal_distance = torch.abs(torch.sum((local_xy - centre) * normal, dim=1))
            inside_straight_through_zone = mask & (
                normal_distance
                <= task_env.cfg.tight_door_no_rotation_normal_half_extent_m
            )
            alignment_threshold_deg = torch.where(
                inside_straight_through_zone,
                # Rotation is intentionally suppressed here by the task. The
                # expert must keep creeping through after its pre-frame
                # centering manoeuvre instead of asking for an impossible
                # final correction and deadlocking in the aperture.
                torch.full_like(alignment_threshold_deg, 180.0),
                alignment_threshold_deg,
            )
    alignment = torch.clamp(torch.cos(heading_error), min=0.0).pow(6)
    speed_mps = torch.minimum(torch.full_like(distance, 0.30), distance * 0.35) * alignment
    speed_mps = torch.where(
        staging_active,
        torch.maximum(speed_mps, torch.full_like(speed_mps, 0.15)),
        speed_mps,
    )
    precise_alignment = torch.abs(heading_error) <= torch.deg2rad(alignment_threshold_deg)
    speed_mps = torch.where(precise_alignment, speed_mps, torch.zeros_like(speed_mps))
    yaw_rate = policy_obs[:, 6]
    yaw_settling = precise_alignment & (
        torch.abs(yaw_rate)
        > task_env.cfg.tight_door_alignment_hold_maximum_yaw_rate_rad_s
    )
    speed_mps = torch.where(yaw_settling, torch.zeros_like(speed_mps), speed_mps)
    # The task maps action -1/+1 to 0.0/0.5 m/s.
    linear_action = (4.0 * speed_mps - 1.0).clamp(-1.0, 1.0)
    angular_action = (3.0 * heading_error).clamp(-1.0, 1.0)
    # The 171 kg imported chassis needs a finite breakaway request for a
    # stationary pivot; proportional steering alone stalls around 4--7 deg.
    needs_alignment = torch.abs(heading_error) > torch.deg2rad(alignment_threshold_deg)
    breakaway_action = torch.sign(heading_error) * 0.55
    angular_action = torch.where(
        needs_alignment
        & (torch.abs(angular_action) < torch.abs(breakaway_action)),
        breakaway_action,
        angular_action,
    )
    angular_action = torch.where(yaw_settling, torch.zeros_like(angular_action), angular_action)
    return torch.stack((linear_action, angular_action), dim=-1)


def main() -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    output_checkpoint = args.output_checkpoint.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    if not SPECIALIST_SEGMENTS:
        raise ValueError("--segments must select at least one segment")
    if any(segment_id not in (3, 4, 8, 9) for segment_id in SPECIALIST_SEGMENTS):
        raise ValueError("--segments may only select measured-door segments 3,4,8,9")
    if args.num_envs < len(SPECIALIST_SEGMENTS):
        raise ValueError("--num-envs must cover every selected specialist segment")
    if args.episodes_per_segment < 1:
        raise ValueError("--episodes-per-segment must be positive")
    if args.retention_ratio < 0.0:
        raise ValueError("--retention-ratio must be non-negative")
    source_sha256 = sha256_file(checkpoint)
    if source_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint hash mismatch: {source_sha256} != {args.expected_checkpoint_sha256}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    env_cfg.balanced_segment_assignment = True
    if args.num_envs < 12:
        env_cfg.balanced_segment_ids = SPECIALIST_SEGMENTS
    env_cfg.episode_length_s = args.collection_episode_length_s
    if args.start_lateral_jitter_m is not None:
        env_cfg.start_lateral_jitter_m = args.start_lateral_jitter_m
    if args.start_yaw_jitter_deg is not None:
        env_cfg.start_yaw_jitter_rad = torch.deg2rad(
            torch.tensor(args.start_yaw_jitter_deg)
        ).item()
    if args.goal_jitter_m is not None:
        env_cfg.goal_jitter_m = args.goal_jitter_m
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = "cuda:0"
    raw_env = gym.make(TASK, cfg=env_cfg)
    # Demonstration collection must inspect the terminal state itself. Isaac
    # Lab otherwise auto-resets inside ``step`` before this script can retain
    # the final observation, and replicated USD property resets dominate the
    # small measured-door data collection. The production task is unchanged;
    # this override exists only on this short-lived collection environment.
    def no_auto_reset_dones() -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(raw_env.unwrapped.num_envs, dtype=torch.bool, device="cuda:0")
        return zeros, zeros

    raw_env.unwrapped._get_dones = no_auto_reset_dones  # noqa: SLF001
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
    outcomes = {
        segment_id: {"success": 0, "collision": 0, "time_out": 0}
        for segment_id in SPECIALIST_SEGMENTS
    }
    parked = torch.zeros(env.num_envs, dtype=torch.bool, device="cuda:0")
    collection_step = 0
    while (
        any(
            values["success"] < args.episodes_per_segment
            for values in outcomes.values()
        )
        and simulation_app.is_running()
    ):
        with torch.no_grad():
            teacher_actions = policy_nn.act_inference(observations)
            labels = expert_actions(observations, env.unwrapped)
            segment_ids = env.unwrapped._segment_ids.clone()  # noqa: SLF001
            specialist_mask = torch.zeros_like(segment_ids, dtype=torch.bool)
            for segment_id in SPECIALIST_SEGMENTS:
                if outcomes[segment_id]["success"] < args.episodes_per_segment:
                    specialist_mask |= segment_ids == segment_id
            specialist_mask &= ~parked
            if not torch.any(specialist_mask):
                break
            actions = teacher_actions.clone()
            actions[specialist_mask] = labels[specialist_mask]
            actions[parked, 0] = -1.0
            actions[parked, 1] = 0.0
            expert_obs.append(observations["policy"][specialist_mask].detach().clone())
            expert_targets.append(labels[specialist_mask].detach().clone())

            retention_ids = torch.nonzero(~specialist_mask & ~parked, as_tuple=False).flatten()
            retain_count = min(
                int(round(specialist_mask.sum().item() * args.retention_ratio)),
                int(retention_ids.numel()),
            )
            offset = (collection_step * max(retain_count, 1)) % max(int(retention_ids.numel()), 1)
            selected_ids = torch.roll(retention_ids, shifts=-offset)[:retain_count]
            if retain_count > 0:
                retention_obs.append(observations["policy"][selected_ids].detach().clone())
                retention_targets.append(teacher_actions[selected_ids].detach().clone())
            observations, _, _, _ = env.step(actions)
            collection_step += 1
            if args.progress_interval > 0 and collection_step % args.progress_interval == 0:
                local_xy = env.unwrapped._local_xy()  # noqa: SLF001
                _, _, distance, heading_error = env.unwrapped._goal_geometry()  # noqa: SLF001
                specialist_rows = []
                for segment_id in SPECIALIST_SEGMENTS:
                    matches = torch.nonzero(segment_ids == segment_id, as_tuple=False).flatten()
                    if matches.numel() == 0:
                        continue
                    env_id = int(matches[0].item())
                    specialist_rows.append(
                        {
                            "segment": segment_id,
                            "env_id": env_id,
                            "xy_m": [round(float(value), 3) for value in local_xy[env_id].tolist()],
                            "distance_m": round(float(distance[env_id].item()), 3),
                            "heading_error_deg": round(
                                float(torch.rad2deg(heading_error[env_id]).item()), 2
                            ),
                            "action": [
                                round(float(value), 3) for value in actions[env_id].tolist()
                            ],
                        }
                    )
                print(
                    json.dumps(
                        {
                            "collection_step": collection_step,
                            "outcomes": outcomes,
                            "specialists": specialist_rows,
                        }
                    ),
                    flush=True,
                )
        collision, success, invalid = env.unwrapped._termination_masks()  # noqa: SLF001
        terminal_failure = collision | invalid
        for env_id in torch.nonzero(~parked, as_tuple=False).flatten().tolist():
            segment_id = int(env.unwrapped._segment_ids[env_id].item())  # noqa: SLF001
            if segment_id not in outcomes:
                if bool((terminal_failure | success)[env_id]):
                    parked[env_id] = True
                continue
            if outcomes[segment_id]["success"] >= args.episodes_per_segment:
                parked[env_id] = True
                continue
            if bool(terminal_failure[env_id]):
                outcomes[segment_id]["collision"] += 1
                raise RuntimeError(
                    f"expert demonstration gate failed for segment {segment_id}: {outcomes[segment_id]}"
                )
            if bool(success[env_id]):
                outcomes[segment_id]["success"] += 1
                parked[env_id] = True
        if collection_step * env.unwrapped.step_dt >= args.collection_episode_length_s:
            for segment_id, values in outcomes.items():
                if values["success"] < args.episodes_per_segment:
                    values["time_out"] += 1
            raise RuntimeError(f"expert demonstration collection timed out: {outcomes}")

    if any(
        values["success"] != args.episodes_per_segment for values in outcomes.values()
    ):
        raise RuntimeError(f"expert demonstration quotas not met: {outcomes}")
    print(json.dumps({"collection_complete": outcomes, "steps": collection_step}), flush=True)

    raw_expert_obs = torch.cat(expert_obs, dim=0)
    raw_retention_obs = (
        torch.cat(retention_obs, dim=0)
        if retention_obs
        else raw_expert_obs.new_empty((0, raw_expert_obs.shape[1]))
    )
    raw_obs = torch.cat((raw_expert_obs, raw_retention_obs), dim=0)
    raw_retention_targets = (
        torch.cat(retention_targets, dim=0)
        if retention_targets
        else raw_expert_obs.new_empty((0, 2))
    )
    targets = torch.cat((torch.cat(expert_targets, dim=0), raw_retention_targets), dim=0)
    with torch.no_grad():
        normalized_obs = policy_nn.actor_obs_normalizer(raw_obs).detach()
    print(
        json.dumps(
            {
                "dataset_ready": int(normalized_obs.shape[0]),
                "expert_samples": int(raw_expert_obs.shape[0]),
                "retention_samples": int(raw_retention_obs.shape[0]),
            }
        ),
        flush=True,
    )

    actor = policy_nn.actor
    anchors = {name: value.detach().clone() for name, value in actor.named_parameters()}
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=normalized_obs.device).manual_seed(args.seed + 1)
    initial_loss = None
    final_loss = None
    actor.train()
    for _ in range(args.epochs):
        permutation = torch.randperm(
            normalized_obs.shape[0], generator=generator, device=normalized_obs.device
        )
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
    print(json.dumps({"optimization_complete": final_loss}), flush=True)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_infos = dict(source_infos or {})
    bootstrap_infos["aisha_measured_tight_door_behavior_cloning"] = {
        "source_sha256": source_sha256,
        "specialist_segments": list(SPECIALIST_SEGMENTS),
        "episodes_per_segment": args.episodes_per_segment,
        "network_weights_changed": True,
        "normalizers_changed": False,
    }
    torch.save(
        {
            "model_state_dict": cpu_tree(policy_nn.state_dict()),
            "optimizer_state_dict": cpu_tree(runner.alg.optimizer.state_dict()),
            "iter": runner.current_learning_iteration,
            "infos": bootstrap_infos,
        },
        output_checkpoint,
    )
    print(json.dumps({"checkpoint_saved": str(output_checkpoint)}), flush=True)
    output_sha256 = sha256_file(output_checkpoint)

    report = {
        "report_type": "measured_tight_door_behavior_cloning_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "seed": args.seed,
        "source_checkpoint": {"path": str(checkpoint), "sha256": source_sha256},
        "output_checkpoint": {"path": str(output_checkpoint), "sha256": output_sha256},
        "specialist_segments": list(SPECIALIST_SEGMENTS),
        "demonstrations": {
            "episodes_per_segment": args.episodes_per_segment,
            "collection_episode_length_s": args.collection_episode_length_s,
            "collection_auto_reset_disabled": True,
            "expert_samples": int(raw_expert_obs.shape[0]),
            "original_policy_retention_samples": int(raw_retention_obs.shape[0]),
            "outcomes": outcomes,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "anchor_weight": args.anchor_weight,
            "retention_ratio": args.retention_ratio,
            "initial_imitation_mse": initial_loss,
            "final_imitation_mse": final_loss,
        },
        "physical_release": False,
        "claim_boundary": (
            "The privileged door-centre expert labels training demonstrations only and is absent "
            "from deterministic sensor-policy evaluation. Passing supports the measured-site "
            "presentation candidate, not physical release."
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
