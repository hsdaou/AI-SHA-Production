#!/usr/bin/env python3
"""Run deterministic, held-out evaluation of an AI-SHA RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL .pt checkpoint to evaluate.")
parser.add_argument("--output", type=Path, required=True, help="Destination JSON report.")
parser.add_argument("--task", default="Isaac-AISHA-OfficeNav-Direct-v0")
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1001, help="Evaluation seed, distinct from training seed 42.")
parser.add_argument(
    "--require-acceptance",
    action="store_true",
    help="Return a non-zero exit status when the task's declared acceptance gate does not pass.",
)
parser.add_argument(
    "--fixed-segment-id",
    type=int,
    default=None,
    help="Evaluate only one route segment when the task exposes fixed_segment_id.",
)
parser.add_argument(
    "--episodes-per-segment",
    type=int,
    default=None,
    help="Require an equal held-out episode quota for every route segment.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.episodes < 1 or args.num_envs < 1:
    parser.error("--episodes and --num_envs must be positive")
if args.episodes_per_segment is not None and args.episodes_per_segment < 1:
    parser.error("--episodes-per-segment must be positive")
if args.episodes_per_segment is not None and args.fixed_segment_id is not None:
    parser.error("--episodes-per-segment and --fixed-segment-id are mutually exclusive")
checkpoint = args.checkpoint.expanduser().resolve()
if not checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {checkpoint}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aisha_isaaclab.tasks  # noqa: E402,F401

try:  # Imported only to give sensor-policy reports human-readable route labels.
    from aisha_isaaclab.tasks.office_nav.block_a_sensor_env import ROUTE_SEGMENTS  # noqa: E402
except ImportError:
    ROUTE_SEGMENTS = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    device = args.device or "cuda:0"
    is_phase3_task = "Phase3-" in args.task
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    phase3_curriculum_strength = None
    if is_phase3_task:
        if not hasattr(env_cfg, "curriculum_minimum_strength"):
            raise ValueError(f"Phase 3 task {args.task} does not expose curriculum strength")
        # Evaluation is a robustness gate, not a training warm-up. A newly
        # constructed environment has common_step_counter == 0, so leaving the
        # training schedule untouched would silently disable its moving people
        # and most domain randomization.
        env_cfg.curriculum_minimum_strength = 1.0
        phase3_curriculum_strength = 1.0
    target_episodes = args.episodes
    if args.fixed_segment_id is not None:
        if not hasattr(env_cfg, "fixed_segment_id"):
            raise ValueError(f"task {args.task} does not support --fixed-segment-id")
        env_cfg.fixed_segment_id = args.fixed_segment_id
    if args.episodes_per_segment is not None:
        if not ROUTE_SEGMENTS or not hasattr(env_cfg, "balanced_segment_assignment"):
            raise ValueError(f"task {args.task} does not support balanced segment evaluation")
        if args.num_envs < len(ROUTE_SEGMENTS):
            raise ValueError(
                f"balanced evaluation requires at least {len(ROUTE_SEGMENTS)} parallel environments"
            )
        env_cfg.balanced_segment_assignment = True
        target_episodes = args.episodes_per_segment * len(ROUTE_SEGMENTS)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    training_seed = agent_cfg.seed
    saved_agent_cfg = checkpoint.parent / "params" / "agent.yaml"
    if saved_agent_cfg.is_file():
        training_seed = int(yaml.safe_load(saved_agent_cfg.read_text(encoding="utf-8"))["seed"])
    agent_cfg.seed = args.seed
    agent_cfg.device = device

    raw_env = gym.make(args.task, cfg=env_cfg)
    print(f"EVALUATION_ENV_READY task={args.task} envs={args.num_envs}", flush=True)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint))
    print(f"EVALUATION_POLICY_LOADED checkpoint={checkpoint.name}", flush=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    observations = env.get_observations()
    running_returns = torch.zeros(args.num_envs, device=env.device)
    running_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    completed_distances: list[float] = []
    completed_heading_errors: list[float] = []
    completed_minimum_ranges: list[float] = []
    counts = {"success": 0, "collision": 0, "time_out": 0}
    collision_classes = {"dynamic_obstacle": 0, "static": 0}
    segment_counts: dict[int, dict[str, int]] = {}
    first_step_actions: torch.Tensor | None = None
    diagnostic_action_sum = torch.zeros(2, device=env.device)
    diagnostic_action_samples = 0
    diagnostic_steps_remaining = 30
    diagnostic_step = 0
    diagnostic_snapshots: list[dict[str, object]] = []
    diagnostic_snapshot_steps = {0, 15, 30, 60, 90, 120, 180, 300, 600, 1200, 2000}
    progress_interval = max(1, min(64, target_episodes))
    next_progress_report = progress_interval

    started = time.perf_counter()
    while len(completed_returns) < target_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(observations)
            if diagnostic_step in diagnostic_snapshot_steps:
                diagnostic_snapshots.append(
                    {
                        "step": diagnostic_step,
                        "mean_action": actions.mean(dim=0).tolist(),
                    }
                )
            if first_step_actions is None:
                first_step_actions = actions.detach().clone()
            if diagnostic_steps_remaining > 0:
                diagnostic_action_sum += actions.sum(dim=0)
                diagnostic_action_samples += actions.shape[0]
                diagnostic_steps_remaining -= 1
            diagnostic_step += 1
            observations, rewards, dones, extras = env.step(actions)
            running_returns += rewards
            running_lengths += 1
            policy_nn.reset(dones)

        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue
        outcomes = extras["episode_outcomes"]

        for env_id in done_ids.tolist():
            segment_id = None
            if "segment_id" in outcomes:
                segment_id = int(outcomes["segment_id"][env_id].item())
            if args.episodes_per_segment is not None:
                if segment_id is None:
                    raise RuntimeError("balanced evaluation requires segment_id outcomes")
                already_completed = segment_counts.get(segment_id, {}).get("episodes", 0)
                if already_completed >= args.episodes_per_segment:
                    continue
            elif len(completed_returns) >= target_episodes:
                break

            outcome = "time_out"
            # Success takes precedence in the extremely rare case that an
            # episode reaches the goal on its final allowed step.
            if bool(outcomes["success"][env_id]):
                outcome = "success"
            elif bool(outcomes["collision"][env_id]):
                outcome = "collision"
            counts[outcome] += 1
            if outcome == "collision":
                if bool(outcomes.get("dynamic_obstacle_collision", torch.zeros_like(outcomes["collision"]))[env_id]):
                    collision_classes["dynamic_obstacle"] += 1
                else:
                    collision_classes["static"] += 1
            completed_returns.append(float(running_returns[env_id].item()))
            completed_lengths.append(int(running_lengths[env_id].item()))
            completed_distances.append(float(outcomes["final_distance_m"][env_id].item()))
            if "final_heading_error_rad" in outcomes:
                completed_heading_errors.append(
                    float(outcomes["final_heading_error_rad"][env_id].item())
                )
            if "minimum_lidar_range_m" in outcomes:
                completed_minimum_ranges.append(float(outcomes["minimum_lidar_range_m"][env_id].item()))
            if segment_id is not None:
                segment = segment_counts.setdefault(
                    segment_id, {"episodes": 0, "success": 0, "collision": 0, "time_out": 0}
                )
                segment["episodes"] += 1
                segment[outcome] += 1

        running_returns[done_ids] = 0.0
        running_lengths[done_ids] = 0
        if len(completed_returns) >= next_progress_report:
            print(
                f"EVALUATION_PROGRESS completed={len(completed_returns)}/{target_episodes}",
                flush=True,
            )
            next_progress_report += progress_interval

    elapsed = time.perf_counter() - started
    print("EVALUATION_ROLLOUT_COMPLETE", flush=True)

    if len(completed_returns) != target_episodes:
        raise RuntimeError(f"simulator stopped after {len(completed_returns)} of {target_episodes} episodes")

    training_contract = yaml.safe_load(
        (PROJECT_ROOT.parent / "config" / "training.yaml").read_text(encoding="utf-8")
    )
    print("EVALUATION_CONTRACT_LOADED", flush=True)
    is_sensor_task = "SensorNav" in args.task
    is_phase2_turn_task = "Phase2-Turn" in args.task
    is_phase2_route_task = "Phase2-EndToEnd" in args.task
    is_phase2_task = is_phase2_turn_task or is_phase2_route_task
    task_contract = (
        training_contract["phase3_curriculum"]
        if is_phase3_task
        else training_contract["phase2_curriculum"]
        if is_phase2_task
        else training_contract["sensor_curriculum"]
        if is_sensor_task
        else training_contract["task"]
    )
    mean_return = sum(completed_returns) / target_episodes
    mean_steps = sum(completed_lengths) / target_episodes
    per_segment = []
    for segment_id, segment in sorted(segment_counts.items()):
        label = (
            f"{ROUTE_SEGMENTS[segment_id][0]} -> {ROUTE_SEGMENTS[segment_id][1]}"
            if segment_id < len(ROUTE_SEGMENTS)
            else f"segment_{segment_id}"
        )
        per_segment.append(
            {
                "segment_id": segment_id,
                "label": label,
                **segment,
                "success_rate": segment["success"] / segment["episodes"],
                "collision_rate": segment["collision"] / segment["episodes"],
                "time_out_rate": segment["time_out"] / segment["episodes"],
            }
        )
    acceptance_gate = None
    gate_contract = (
        task_contract.get("acceptance_gate")
        if is_phase3_task
        else task_contract.get("full_route_acceptance_gate")
        if is_phase2_route_task
        else task_contract.get("turn_acceptance_gate")
        if is_phase2_turn_task
        else task_contract.get("held_out_acceptance_gate")
    )
    if is_phase3_task and gate_contract:
        rates_by_segment = {item["segment_id"]: item["success_rate"] for item in per_segment}
        enabled_segments = task_contract["dynamic_obstacles"]["enabled_route_segments"]
        protocol_satisfied = (
            args.episodes_per_segment is not None
            and args.episodes_per_segment >= gate_contract["randomized_episodes_per_segment"]
            and len(rates_by_segment) == len(ROUTE_SEGMENTS)
        )
        observed = {
            "overall_success_rate": counts["success"] / target_episodes,
            "dynamic_obstacle_collision_rate": collision_classes["dynamic_obstacle"]
            / target_episodes,
            "static_collision_rate": collision_classes["static"] / target_episodes,
            "minimum_enabled_segment_success_rate": min(
                (rates_by_segment.get(segment_id, 0.0) for segment_id in enabled_segments),
                default=0.0,
            ),
        }
        passed = (
            protocol_satisfied
            and observed["overall_success_rate"] >= gate_contract["overall_success_rate_min"]
            and observed["dynamic_obstacle_collision_rate"]
            <= gate_contract["dynamic_obstacle_collision_rate_max"]
            and observed["static_collision_rate"] <= gate_contract["static_collision_rate_max"]
            and observed["minimum_enabled_segment_success_rate"]
            >= gate_contract["every_enabled_segment_success_rate_min"]
        )
        acceptance_gate = {
            "scope": "phase3_randomized_segment_subgate_only",
            "protocol_satisfied": protocol_satisfied,
            "passed": passed,
            "full_phase3_acceptance": False,
            "pending_separate_gates": [
                "Phase 2 static-route regression",
                "12 live-administration dynamic scenarios",
            ],
            "requirements": gate_contract,
            "observed": observed,
        }
    elif is_phase2_route_task and gate_contract:
        protocol_satisfied = (
            args.episodes_per_segment is None
            and target_episodes >= gate_contract["full_route_episodes"]
        )
        observed = {
            "full_route_success_rate": counts["success"] / target_episodes,
            "collision_rate": counts["collision"] / target_episodes,
            "time_out_rate": counts["time_out"] / target_episodes,
        }
        passed = (
            protocol_satisfied
            and observed["full_route_success_rate"] >= gate_contract["full_route_success_rate_min"]
            and observed["collision_rate"] <= gate_contract["collision_rate_max"]
            and observed["time_out_rate"] <= gate_contract["time_out_rate_max"]
        )
        acceptance_gate = {
            "protocol_satisfied": protocol_satisfied,
            "passed": passed,
            "requirements": gate_contract,
            "observed": observed,
        }
    elif is_sensor_task and gate_contract:
        rates_by_segment = {item["segment_id"]: item["success_rate"] for item in per_segment}
        office_segment_ids = gate_contract["office_threshold_segment_ids"]
        protocol_satisfied = (
            args.episodes_per_segment is not None
            and args.episodes_per_segment >= gate_contract["episodes_per_segment"]
            and len(rates_by_segment) == len(ROUTE_SEGMENTS)
        )
        observed = {
            "overall_success_rate": counts["success"] / target_episodes,
            "overall_collision_rate": counts["collision"] / target_episodes,
            "minimum_segment_success_rate": min(rates_by_segment.values(), default=0.0),
            "minimum_office_threshold_success_rate": min(
                (rates_by_segment.get(segment_id, 0.0) for segment_id in office_segment_ids),
                default=0.0,
            ),
        }
        passed = (
            protocol_satisfied
            and observed["overall_success_rate"] >= gate_contract["overall_success_rate_min"]
            and observed["overall_collision_rate"] <= gate_contract["overall_collision_rate_max"]
            and observed["minimum_segment_success_rate"]
            >= gate_contract["every_segment_success_rate_min"]
            and observed["minimum_office_threshold_success_rate"]
            >= gate_contract["office_threshold_success_rate_min"]
        )
        acceptance_gate = {
            "protocol_satisfied": protocol_satisfied,
            "passed": passed,
            "requirements": gate_contract,
            "observed": observed,
        }
    report = {
        "report_type": (
            "held_out_phase3_dynamic_randomized_evaluation"
            if is_phase3_task
            else "held_out_end_to_end_route_evaluation"
            if is_phase2_route_task
            else "held_out_phase2_turn_evaluation"
            if is_phase2_turn_task
            else "held_out_sensor_evaluation"
            if is_sensor_task
            else "held_out_foundation_evaluation"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "training_contract_revision": training_contract["revision"],
        "checkpoint": {
            "path": _relative_or_absolute(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "protocol": {
            "policy": "deterministic_inference",
            "seed": args.seed,
            "training_seed": training_seed,
            "episodes": target_episodes,
            "episodes_per_segment": args.episodes_per_segment,
            "parallel_environments": args.num_envs,
            "physics_rate_hz": task_contract["physics_rate_hz"],
            "policy_rate_hz": task_contract["policy_rate_hz"],
            "randomization": (
                "full-strength Phase 3 domain randomization and dynamic-obstacle distribution; unseen sequence"
                if is_phase3_task
                else "full chained route; training-declared start, goal and observation randomization; unseen sequence"
                if is_phase2_route_task
                else
                "equal quota per route segment; training-declared pose and goal jitter; unseen random sequence"
                if args.episodes_per_segment is not None
                else "same declared distribution as training; unseen random sequence"
            ),
            "phase3_curriculum_strength": phase3_curriculum_strength,
            "fixed_segment_id": args.fixed_segment_id,
        },
        "results": {
            "success_count": counts["success"],
            "success_rate": counts["success"] / target_episodes,
            "collision_count": counts["collision"],
            "collision_rate": counts["collision"] / target_episodes,
            "dynamic_obstacle_collision_count": collision_classes["dynamic_obstacle"],
            "static_collision_count": collision_classes["static"],
            "time_out_count": counts["time_out"],
            "time_out_rate": counts["time_out"] / target_episodes,
            "mean_episode_return": mean_return,
            "mean_episode_steps": mean_steps,
            "mean_episode_duration_s": mean_steps / task_contract["policy_rate_hz"],
            "mean_final_goal_distance_m": sum(completed_distances) / target_episodes,
            "mean_abs_final_heading_error_deg": (
                sum(abs(value) for value in completed_heading_errors)
                / len(completed_heading_errors)
                * 180.0
                / torch.pi
                if completed_heading_errors
                else None
            ),
            "mean_minimum_lidar_range_m": (
                sum(completed_minimum_ranges) / len(completed_minimum_ranges)
                if completed_minimum_ranges
                else None
            ),
            "wall_clock_evaluation_s": elapsed,
            "per_segment": per_segment,
        },
        "action_diagnostics": {
            "dimensions": ["normalized_linear_command", "normalized_angular_command"],
            "mapping": (
                f"linear -1 maps to {env_cfg.linear_velocity_range_mps[0]} m/s; "
                f"linear +1 maps to {env_cfg.linear_velocity_range_mps[1]} m/s; "
                f"angular ±1 maps to ±{env_cfg.angular_velocity_max_rad_s} rad/s"
            ),
            "first_step_mean": first_step_actions.mean(dim=0).tolist(),
            "first_step_min": first_step_actions.amin(dim=0).tolist(),
            "first_step_max": first_step_actions.amax(dim=0).tolist(),
            "first_30_step_mean": (
                diagnostic_action_sum / max(diagnostic_action_samples, 1)
            ).tolist(),
            "timeline": diagnostic_snapshots,
        },
        "acceptance_gate": acceptance_gate,
        "claim_boundary": {
            "supported": (
                "Phase 3 randomized Block A segments with moving ray-visible person proxies"
                if is_phase3_task
                else "policy-only chained Block A route with no turn or dwell action override"
                if is_phase2_route_task
                else "Phase 2 arbitrary-heading turn-acquisition curriculum"
                if is_phase2_turn_task
                else
                "Isaac Lab plan-derived Block A policy with downsampled LD19-style ray observations"
                if is_sensor_task
                else "state-observation Isaac Lab doorway-foundation policy evaluation"
            ),
            "not_supported": [
                "RTX-rendered LiDAR or depth-camera policy input",
                "Nav2 integration",
                "photoreal administration-office fidelity",
                "sim-to-real transfer",
                "physical robot release",
            ],
            "doorway_width": training_contract["geometry"]["doorway_width_status"],
        },
    }
    print("EVALUATION_REPORT_COMPOSED", flush=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("EVALUATION_REPORT_WRITTEN", flush=True)
    print("EVALUATION_RESULT=" + json.dumps(report["results"], sort_keys=True))
    print(f"EVALUATION_REPORT={output}")
    # Write the evidence before closing Kit. In Isaac Sim 5.1 a
    # MultiMeshRayCaster teardown can terminate the process during env.close().
    env.close()
    if args.require_acceptance and acceptance_gate is not None and not acceptance_gate["passed"]:
        print("EVALUATION_ACCEPTANCE_GATE=failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
