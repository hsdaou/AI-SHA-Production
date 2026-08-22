#!/usr/bin/env python3
"""Film and measure the learned Phase 3N brake layer during one office crossing."""

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
BUNDLE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TASK = "Isaac-AISHA-Administration-Live-Phase4A-DynamicSafety-Showcase-Direct-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-report", type=Path, required=True)
parser.add_argument("--video-folder", type=Path, default=None)
parser.add_argument("--max-steps", type=int, default=1800)
parser.add_argument("--seed", type=int, default=10401)
parser.add_argument("--trace-interval", type=int, default=1)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(2.10, -1.90, 2.00))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(5.05, -5.25, 0.72))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_steps < 1 or args.trace_interval < 1:
    parser.error("--max-steps and --trace-interval must be positive")
if args.video_folder is not None:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pxr import UsdPhysics  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import aisha_isaaclab.tasks  # noqa: E402,F401
from aisha_isaaclab.tasks.office_nav.administration_dynamic_safety_env import (  # noqa: E402
    PHASE4A_PEDESTRIAN_USD,
)
from aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env import (  # noqa: E402
    PHASE3M_FROZEN_RECOVERY_CHECKPOINT,
    PHASE3_FROZEN_ROUTE_CHECKPOINT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bool(value: torch.Tensor) -> bool:
    return bool(value[0].item())


def tensor_float(value: torch.Tensor) -> float:
    return float(value[0].item())


def main() -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_report = args.output_report.expanduser().resolve()
    device = args.device or "cuda:0"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1, use_fabric=True)
    env_cfg.seed = args.seed
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    env_cfg.viewer.eye = tuple(args.camera_eye)
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    env_cfg.viewer.resolution = (1280, 720)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = device

    raw_env = gym.make(
        TASK,
        cfg=env_cfg,
        render_mode="rgb_array" if args.video_folder is not None else None,
    )
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage_paths = {
        "live_robot_base": "/World/envs/env_0/Robot/base_link",
        "pedestrian_root": "/World/envs/env_0/DynamicObstacle_0",
        "pedestrian_torso": "/World/envs/env_0/DynamicObstacle_0/Torso",
        "pedestrian_head": "/World/envs/env_0/DynamicObstacle_0/Head",
        "pedestrian_collision_torso": "/World/envs/env_0/DynamicObstacle_0/Torso",
        "excluded_replay_robot": "/World/envs/env_0/Administration/AISHA",
    }
    stage_checks = {}
    for name, prim_path in stage_paths.items():
        prim = stage.GetPrimAtPath(prim_path)
        stage_checks[name] = {
            "path": prim_path,
            "valid": bool(prim),
            "active": bool(prim and prim.IsActive()),
            "type_name": prim.GetTypeName() if prim else None,
            "children": [child.GetName() for child in prim.GetChildren()] if prim else [],
            "collision_api": bool(prim and prim.HasAPI(UsdPhysics.CollisionAPI)),
        }

    video_folder = None
    if args.video_folder is not None:
        video_folder = args.video_folder.expanduser().resolve()
        video_folder.mkdir(parents=True, exist_ok=True)
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=args.max_steps,
            name_prefix="aisha-phase4a-dynamic-showcase",
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    origin = env.unwrapped.scene.env_origins[0]
    env.unwrapped.sim.set_camera_view(
        eye=tuple(float(origin[index].item()) + args.camera_eye[index] for index in range(3)),
        target=tuple(
            float(origin[index].item()) + args.camera_lookat[index] for index in range(3)
        ),
    )
    observations = env.get_observations()
    trace: list[dict[str, object]] = []
    trigger_step = None
    crossing_complete_step = None
    authority_steps: list[int] = []
    outcome = "max_steps"
    termination = None

    for step in range(args.max_steps):
        with torch.inference_mode():
            policy_action = policy(observations)
            observations, _, dones, extras = env.step(policy_action)
            policy_nn.reset(dones)

        showcase = env.unwrapped.showcase_state()
        triggered = tensor_bool(showcase["triggered"])
        crossing_progress = tensor_float(showcase["crossing_progress"])
        if triggered and trigger_step is None:
            trigger_step = step + 1
        if crossing_progress >= 0.999 and crossing_complete_step is None:
            crossing_complete_step = step + 1
        authority = tensor_bool(env.unwrapped._safety_authority_active)
        if authority:
            authority_steps.append(step + 1)

        if (step + 1) % args.trace_interval == 0:
            robot_xy = env.unwrapped._local_xy()[0]
            person_xy = showcase["person_position_xy_m"][0]
            exact_ranges = env.unwrapped._lidar_ranges()[0]
            ring_clearance = torch.amin(
                exact_ranges - env.unwrapped._lidar_envelope_ranges
            )
            trace.append(
                {
                    "step": step + 1,
                    "elapsed_s": round((step + 1) / 30.0, 5),
                    "robot_xy_m": [round(float(value.item()), 5) for value in robot_xy],
                    "pedestrian_xy_m": [
                        round(float(value.item()), 5) for value in person_xy
                    ],
                    "robot_pedestrian_centre_distance_m": round(
                        float(torch.linalg.norm(person_xy - robot_xy).item()), 5
                    ),
                    "pedestrian_triggered": triggered,
                    "pedestrian_crossing_progress": round(crossing_progress, 5),
                    "linear_velocity_mps": round(
                        float(env.unwrapped._robot.data.root_lin_vel_b[0, 0].item()), 5
                    ),
                    "yaw_rate_rad_s": round(
                        float(env.unwrapped._robot.data.root_ang_vel_b[0, 2].item()), 5
                    ),
                    "minimum_360_ring_clearance_m": round(float(ring_clearance.item()), 5),
                    "safety_action": round(float(policy_action[0, 0].item()), 5),
                    "safety_authority_active": authority,
                    "learned_brake_fraction": round(
                        tensor_float(env.unwrapped._safety_brake_fraction), 5
                    ),
                    "protective_stop_latched": tensor_bool(
                        env.unwrapped._protective_stop_latched
                    ),
                    "frozen_stack_command": [
                        round(float(value.item()), 5)
                        for value in env.unwrapped._frozen_stack_actions[0]
                    ],
                    "applied_command": [
                        round(float(value.item()), 5)
                        for value in env.unwrapped._actions[0]
                    ],
                }
            )

        if bool(dones[0].item()):
            episode = extras["episode_outcomes"]
            if bool(episode["success"][0].item()):
                outcome = "success"
            elif bool(episode["collision"][0].item()):
                outcome = "collision"
            else:
                outcome = "time_out"
            termination = {
                "segment_id": int(episode["segment_id"][0].item()),
                "final_goal_distance_m": float(episode["final_distance_m"][0].item()),
                "collision": bool(episode["collision"][0].item()),
                "dynamic_obstacle_collision": bool(
                    episode["dynamic_obstacle_collision"][0].item()
                ),
                "static_collision": bool(episode["static_collision"][0].item()),
                "safety_steps": int(episode["safety_steps"][0].item()),
                "safety_authority_steps": int(
                    episode["safety_authority_steps"][0].item()
                ),
                "safety_brake_fraction_sum": float(
                    episode["safety_brake_fraction_sum"][0].item()
                ),
                "minimum_360_ring_clearance_m": float(
                    episode["minimum_ring_clearance_m"][0].item()
                ),
            }
            completed_steps = step + 1
            break
    else:
        completed_steps = args.max_steps

    encounter_trace = [
        row
        for row in trace
        if trigger_step is not None
        and crossing_complete_step is not None
        and trigger_step <= int(row["step"]) <= crossing_complete_step
    ]
    encounter_authority_trace = [
        row for row in encounter_trace if row["safety_authority_active"]
    ]
    encounter_authority_steps = [
        int(row["step"]) for row in encounter_authority_trace
    ]
    first_authority = (
        encounter_authority_steps[0] if encounter_authority_steps else None
    )
    last_authority = (
        encounter_authority_steps[-1] if encounter_authority_steps else None
    )
    pre_crossing_velocity = [
        float(row["linear_velocity_mps"])
        for row in trace
        if trigger_step is not None and trigger_step - 45 <= int(row["step"]) < trigger_step
    ]
    during_velocity = [
        float(row["linear_velocity_mps"]) for row in encounter_trace
    ]
    resumed_velocity = [
        float(row["linear_velocity_mps"])
        for row in trace
        if crossing_complete_step is not None
        and int(row["step"]) >= crossing_complete_step
    ]
    maximum_brake = max(
        (
            float(row["learned_brake_fraction"])
            for row in encounter_authority_trace
        ),
        default=0.0,
    )
    full_stop_steps = sum(
        abs(float(row["linear_velocity_mps"])) <= 0.03
        and bool(row["protective_stop_latched"])
        for row in encounter_trace
    )
    metrics = {
        "trigger_step": trigger_step,
        "crossing_complete_step": crossing_complete_step,
        "total_safety_authority_steps_observed": len(authority_steps),
        "encounter_safety_authority_steps": len(encounter_authority_steps),
        "first_encounter_safety_authority_step": first_authority,
        "last_encounter_safety_authority_step": last_authority,
        "maximum_learned_brake_fraction": maximum_brake,
        "mean_pre_crossing_velocity_mps": (
            sum(pre_crossing_velocity) / len(pre_crossing_velocity)
            if pre_crossing_velocity
            else None
        ),
        "minimum_velocity_during_authority_mps": (
            min(during_velocity) if during_velocity else None
        ),
        "maximum_resumed_velocity_mps": (
            max(resumed_velocity) if resumed_velocity else None
        ),
        "protective_full_stop_steps_during_crossing": full_stop_steps,
        "protective_full_stop_duration_s": full_stop_steps / 30.0,
        "minimum_robot_pedestrian_centre_distance_m": min(
            (float(row["robot_pedestrian_centre_distance_m"]) for row in trace),
            default=math.inf,
        ),
        "minimum_360_ring_clearance_m": min(
            (float(row["minimum_360_ring_clearance_m"]) for row in trace),
            default=math.inf,
        ),
    }
    brake_then_resume = (
        len(encounter_authority_steps) > 0
        and maximum_brake >= 0.02
        and full_stop_steps >= 15
        and metrics["maximum_resumed_velocity_mps"] is not None
        and metrics["maximum_resumed_velocity_mps"] >= 0.25
    )
    checks = {
        "checkpoint_is_accepted_phase3n": checkpoint.name
        == "aisha_phase3n_dynamic_safety_model_50.pt",
        "single_fixed_segment_7": env_cfg.fixed_segment_id == 7
        and not env_cfg.route_chain_mode,
        "pedestrian_visuals_loaded": all(
            stage_checks[name]["valid"] and stage_checks[name]["active"]
            for name in ("pedestrian_root", "pedestrian_torso", "pedestrian_head")
        ),
        "pedestrian_collision_torso_loaded": (
            stage_checks["pedestrian_collision_torso"]["valid"]
            and stage_checks["pedestrian_collision_torso"]["collision_api"]
        ),
        "crossing_triggered": trigger_step is not None,
        "crossing_completed": crossing_complete_step is not None,
        "learned_safety_authority_observed_during_crossing": (
            len(encounter_authority_steps) > 0
        ),
        "learned_brake_then_resume_observed": brake_then_resume,
        "route_segment_completed": outcome == "success",
        "zero_dynamic_collision": bool(
            termination and not termination["dynamic_obstacle_collision"]
        ),
        "zero_static_collision": bool(termination and not termination["static_collision"]),
        "no_emergency_guard": not env_cfg.safety_emergency_guard_enabled,
        "no_supervisor_or_root_animation": True,
    }
    passed = all(checks.values())
    report = {
        "report_type": "phase4a_live_dynamic_safety_showcase",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "seed": args.seed,
        "outcome": outcome,
        "passed": passed,
        "completed_steps": completed_steps,
        "duration_s": completed_steps / 30.0,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "frozen_stack": {
            "phase3m_recovery": {
                "path": str(PHASE3M_FROZEN_RECOVERY_CHECKPOINT.resolve()),
                "sha256": sha256_file(PHASE3M_FROZEN_RECOVERY_CHECKPOINT),
            },
            "route_actor": {
                "path": str(PHASE3_FROZEN_ROUTE_CHECKPOINT.resolve()),
                "sha256": sha256_file(PHASE3_FROZEN_ROUTE_CHECKPOINT),
            },
        },
        "scenario": {
            "route_segment_id": 7,
            "route_leg": "principal_turn_to_principal_approach",
            "pedestrian_asset": {
                "path": str(PHASE4A_PEDESTRIAN_USD.resolve()),
                "sha256": sha256_file(PHASE4A_PEDESTRIAN_USD),
                "disclosure": "stylized kinematic pedestrian proxy with a torso collision envelope",
            },
            "route_fraction": env_cfg.showcase_route_fraction,
            "crossing_half_span_m": env_cfg.showcase_crossing_half_span_m,
            "crossing_speed_mps": env_cfg.showcase_crossing_speed_mps,
            "trigger_distance_m": env_cfg.showcase_trigger_distance_m,
            "deterministic_presentation_parameters": True,
            "formal_phase3n_evaluation_contract_changed": False,
        },
        "camera": {
            "mode": "fixed_human_height_wide",
            "eye": list(args.camera_eye),
            "lookat": list(args.camera_lookat),
            "resolution": [1280, 720],
        },
        "video_folder": str(video_folder) if video_folder is not None else None,
        "stage_checks": stage_checks,
        "metrics": metrics,
        "checks": checks,
        "termination": termination,
        "trace_interval_steps": args.trace_interval,
        "trace": trace,
        "policy_contract": {
            "policy_input": "checkpoint-compatible 36-bin 360-degree LD19-style range observation",
            "outer_actor_authority": "translation reduction only while clearance is closing",
            "steering_source": "hash-locked Phase 3M stack",
            "pedestrian_state_exposed_to_policy": False,
            "emergency_guard_enabled": False,
            "physics_supervisor": False,
            "root_transform_animation": False,
        },
        "claim_boundary": (
            "Isaac Sim/Isaac Lab checkpoint inference in the walkthrough-matched administration "
            "scene. The crossing is a deterministic presentation scenario with a stylized human "
            "proxy; it is not an as-built survey, human-behaviour model, or physical safety release."
        ),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "PHASE4A_SHOWCASE_RESULT="
        + json.dumps(
            {
                "outcome": outcome,
                "passed": passed,
                "completed_steps": completed_steps,
                "metrics": metrics,
                "failed_checks": [name for name, value in checks.items() if not value],
                "output_report": str(output_report),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
