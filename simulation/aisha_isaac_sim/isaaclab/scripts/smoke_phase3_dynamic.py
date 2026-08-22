#!/usr/bin/env python3
"""Run a bounded Isaac Lab smoke gate for the Phase 3 dynamic curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--output-report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
import aisha_isaaclab.tasks  # noqa: E402,F401


TASK_ID = "Isaac-AISHA-BlockA-Phase3-DynamicDR-SensorNav-Direct-v0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    output = args.output_report or PACKAGE_ROOT / "results" / "phase3_dynamic_dr_smoke_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs, use_fabric=True)
    # Exercise the maximum declared perturbation envelope; normal PPO training
    # begins at zero strength and ramps only after its retention warm-up.
    cfg.curriculum_minimum_strength = 1.0
    env = gym.make(TASK_ID, cfg=cfg)
    unwrapped = env.unwrapped
    try:
        observations, _ = env.reset()
        initial_obstacle_positions = torch.stack(
            [obstacle.data.root_pos_w.clone() for obstacle in unwrapped._dynamic_obstacles]
        )
        minimum_range = float("inf")
        finite_observations = True
        reset_events = 0
        for step in range(args.steps):
            # Normalized forward -0.20 maps to 0.20 m/s; the alternating small
            # yaw command exercises delayed/mismatched wheel actuation.
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            actions[:, 0] = -0.20
            actions[:, 1] = 0.12 if (step // 30) % 2 == 0 else -0.12
            observations, _, terminated, truncated, _ = env.step(actions)
            policy_observation = observations["policy"]
            finite_observations &= bool(torch.isfinite(policy_observation).all().item())
            minimum_range = min(minimum_range, float(torch.amin(unwrapped._lidar_ranges()).item()))
            reset_events += int(torch.count_nonzero(terminated | truncated).item())

        final_obstacle_positions = torch.stack(
            [obstacle.data.root_pos_w.clone() for obstacle in unwrapped._dynamic_obstacles]
        )
        displacement = torch.linalg.norm(
            final_obstacle_positions[..., :2] - initial_obstacle_positions[..., :2], dim=-1
        )
        active = unwrapped._obstacle_active
        active_displacement = displacement[active]
        dynamic_motion_max = float(torch.amax(active_displacement).item()) if active_displacement.numel() else 0.0
        observation_shape = list(observations["policy"].shape)
        checks = {
            "task_registered": env.spec.id == TASK_ID,
            "observation_contract_46": observation_shape[-1] == 46,
            "observation_values_finite": finite_observations,
            "dynamic_obstacles_active": bool(torch.any(active).item()),
            "dynamic_obstacles_moved": dynamic_motion_max > 0.05,
            "dynamic_obstacles_in_lidar_contract": any(
                target.track_mesh_transforms
                and "DynamicObstacle" in target.prim_expr
                for target in cfg.scene.crown_lidar.mesh_prim_paths
            ),
            "action_latency_randomized": int(torch.amax(unwrapped._action_latency_steps).item()) > 0,
            "motor_strength_randomized": not bool(
                torch.allclose(unwrapped._motor_strength, torch.ones_like(unwrapped._motor_strength))
            ),
            "mass_randomized": not bool(
                torch.allclose(unwrapped._mass_scale, torch.ones_like(unwrapped._mass_scale))
            ),
            "friction_randomized": not bool(
                torch.allclose(unwrapped._static_friction, torch.ones_like(unwrapped._static_friction))
            ),
            "uncorrupted_collision_ranges_available": minimum_range >= cfg.lidar_min_range_m,
        }
        source = PROJECT_ROOT / "aisha_isaaclab" / "tasks" / "office_nav" / "phase3_dynamic_dr_env.py"
        report = {
            "report_type": "phase3_dynamic_obstacle_domain_randomization_smoke",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": TASK_ID,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "policy_rate_hz": round(1.0 / unwrapped.step_dt),
            "observation_shape": observation_shape,
            "active_obstacles_per_env": active.sum(dim=0).tolist(),
            "maximum_active_obstacle_displacement_m": dynamic_motion_max,
            "minimum_true_lidar_range_m": minimum_range,
            "reset_events_during_smoke": reset_events,
            "sampled_randomization": {
                "curriculum_strength": unwrapped._curriculum_strength(),
                "action_latency_steps": unwrapped._action_latency_steps.tolist(),
                "motor_strength_scale": unwrapped._motor_strength.tolist(),
                "wheel_radius_scale": unwrapped._wheel_radius_scale.tolist(),
                "wheel_track_scale": unwrapped._wheel_track_scale.tolist(),
                "base_mass_scale": unwrapped._mass_scale.tolist(),
                "static_friction": unwrapped._static_friction.tolist(),
                "dynamic_friction": unwrapped._dynamic_friction.tolist(),
                "lidar_bias_m": unwrapped._lidar_episode_bias.tolist(),
                "lidar_scale": unwrapped._lidar_episode_scale.tolist(),
            },
            "phase3_environment_source": str(source),
            "phase3_environment_sha256": sha256(source),
            "checks": checks,
            "passed": all(checks.values()),
            "claim_boundary": (
                "Bounded simulation smoke evidence only; not trained-policy acceptance, "
                "dynamic-person safety validation, or physical release."
            ),
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PHASE3_SMOKE_REPORT={output.resolve()}")
        print(f"PHASE3_SMOKE_PASSED={report['passed']}")
        return 0 if report["passed"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:  # Isaac shutdown can otherwise hide the originating traceback.
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
