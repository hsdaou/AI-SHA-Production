#!/usr/bin/env python3
"""Verify the frozen Phase 3M boundary and outer 360-degree safety authority."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument(
    "--task",
    default="Isaac-AISHA-BlockA-Phase3-DynamicSafety-SensorNav-Direct-v0",
)
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


def main() -> int:
    output = args.output_report or (
        PACKAGE_ROOT / "results" / "phase3n_dynamic_safety_smoke_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=True
    )
    env = gym.make(args.task, cfg=cfg)
    unwrapped = env.unwrapped
    try:
        observations, _ = env.reset()
        frozen = torch.zeros((args.num_envs, 2), device=unwrapped.device)
        frozen[:, 0] = 0.60
        frozen[:, 1] = torch.where(
            torch.arange(args.num_envs, device=unwrapped.device) % 2 == 0,
            torch.full((args.num_envs,), 0.75, device=unwrapped.device),
            torch.full((args.num_envs,), -0.75, device=unwrapped.device),
        )
        zeros = torch.zeros((args.num_envs, 1), device=unwrapped.device)
        zero_output, zero_brake, zero_attenuation = (
            unwrapped._compose_dynamic_safety(frozen, zeros)
        )
        positive = torch.ones_like(zeros)
        positive_output, _, _ = unwrapped._compose_dynamic_safety(frozen, positive)
        full = -torch.ones_like(zeros)
        full_output, full_brake, full_attenuation = (
            unwrapped._compose_dynamic_safety(frozen, full)
        )

        hidden_before = unwrapped._frozen_recovery_hidden.clone()
        inner_actions = unwrapped._frozen_phase3m_actions()
        hidden_after = unwrapped._frozen_recovery_hidden.clone()
        env.reset()
        hidden_reset = unwrapped._frozen_recovery_hidden.clone()

        finite_observations = True
        authority_invariant = True
        inner_actions_finite = bool(torch.isfinite(inner_actions).all().item())
        for step in range(args.steps):
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            actions[:, 0] = -0.70 if (step // 20) % 2 == 0 else 0.50
            observations, _, _, _, _ = env.step(actions)
            finite_observations &= bool(
                torch.isfinite(observations["policy"]).all().item()
            )
            base = unwrapped._frozen_stack_actions
            applied = unwrapped._actions
            scope = unwrapped._safety_authority_active
            if torch.any(scope):
                base_forward = ((base[scope, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
                applied_forward = ((applied[scope, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
                authority_invariant &= bool(
                    torch.all(applied_forward <= base_forward + 1.0e-6).item()
                )
                authority_invariant &= bool(
                    torch.all(
                        torch.abs(applied[scope, 1])
                        <= torch.abs(base[scope, 1]) + 1.0e-6
                    ).item()
                )
                authority_invariant &= bool(
                    torch.all(applied[scope, 1] * base[scope, 1] >= -1.0e-6).item()
                )

        dynamic_ids = tuple(cfg.safety_dynamic_segment_ids)
        dynamic_weight = sum(cfg.segment_sampling_weights[index] for index in dynamic_ids)
        checks = {
            "task_registered": env.spec.id == args.task,
            "observation_contract_46": observations["policy"].shape[-1] == 46,
            "brake_only_action_contract_1": env.action_space.shape[-1] == 1,
            "observation_values_finite": finite_observations,
            "lidar_is_full_360_degrees": (
                cfg.lidar_training_bins == 36
                and unwrapped._safety_ray_angles.numel() == 36
                and bool(torch.all(unwrapped._safety_front_ray_mask == (
                    torch.abs(unwrapped._safety_ray_angles)
                    <= cfg.safety_emergency_front_half_angle_rad
                )).item())
            ),
            "frozen_route_checkpoint_hash_matches": (
                unwrapped._frozen_route_checkpoint_actual_sha256
                == cfg.frozen_route_checkpoint_sha256
            ),
            "frozen_recovery_checkpoint_hash_matches": (
                unwrapped._frozen_recovery_checkpoint_actual_sha256
                == cfg.frozen_recovery_checkpoint_sha256
            ),
            "frozen_route_actor_has_no_trainable_parameters": not any(
                parameter.requires_grad
                for parameter in unwrapped._frozen_route_actor.parameters()
            ),
            "frozen_recovery_memory_has_no_trainable_parameters": not any(
                parameter.requires_grad
                for parameter in unwrapped._frozen_recovery_memory.parameters()
            ),
            "frozen_recovery_actor_has_no_trainable_parameters": not any(
                parameter.requires_grad
                for parameter in unwrapped._frozen_recovery_actor.parameters()
            ),
            "frozen_recovery_inference_is_finite": inner_actions_finite,
            "frozen_recovery_hidden_state_advances": bool(
                torch.any(hidden_after != hidden_before).item()
            ),
            "frozen_recovery_hidden_state_resets": bool(
                torch.equal(hidden_reset, torch.zeros_like(hidden_reset))
            ),
            "zero_safety_output_preserves_frozen_stack": bool(
                torch.equal(zero_output, frozen)
            ),
            "zero_safety_output_has_no_brake": bool(
                torch.equal(zero_brake, torch.zeros_like(zero_brake))
            ),
            "zero_safety_output_has_no_attenuation": bool(
                torch.equal(zero_attenuation, torch.zeros_like(zero_attenuation))
            ),
            "positive_safety_output_is_no_op": bool(
                torch.equal(positive_output, frozen)
            ),
            "full_brake_removes_forward_motion": bool(
                torch.allclose(full_output[:, 0], -torch.ones_like(full_output[:, 0]))
            ),
            "full_brake_fraction_is_one": bool(
                torch.equal(full_brake, torch.ones_like(full_brake))
            ),
            "steering_attenuation_is_zero": bool(
                torch.equal(full_attenuation, torch.zeros_like(full_attenuation))
            ),
            "safety_layer_never_adds_speed_or_steering": authority_invariant,
            "safety_authority_matches_pedestrian_segments": (
                dynamic_ids == tuple(cfg.dynamic_obstacle_segment_ids)
            ),
            "dynamic_segments_receive_training_majority": (
                dynamic_weight / sum(cfg.segment_sampling_weights) > 0.80
            ),
            "all_static_segments_rehearsed": all(
                weight > 0.0 for weight in cfg.segment_sampling_weights
            ),
            "office_recovery_supervisor_remains_enabled": (
                cfg.recovery_supervisor_enabled is True
            ),
            "duplicate_outer_emergency_guard_is_disabled": (
                cfg.safety_emergency_guard_enabled is False
            ),
            "rated_motor_torque_contract_preserved": (
                cfg.rated_motor_effort_limit_nm == 6.0
            ),
            "peak_motor_torque_contract_preserved": (
                cfg.peak_motor_effort_limit_nm == 18.0
                and cfg.peak_motor_time_limit_s == 3.0
            ),
            "full_domain_randomization_strength": (
                unwrapped._curriculum_strength() == 1.0
            ),
        }
        passed = all(checks.values())
        report = {
            "report_type": "phase3n_dynamic_safety_smoke",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "checks": checks,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "passed": passed,
            "contract": unwrapped.extras.get("dynamic_safety", {}),
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"PHASE3N_DYNAMIC_SAFETY_SMOKE passed={passed} "
            f"checks={sum(checks.values())}/{len(checks)} report={output}"
        )
        return 0 if passed else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
