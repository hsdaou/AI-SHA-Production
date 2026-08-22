#!/usr/bin/env python3
"""Verify the Phase 3 safety-residual action boundary in Isaac Lab."""

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
parser.add_argument("--steps", type=int, default=90)
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


TASK_ID = "Isaac-AISHA-BlockA-Phase3-SafetyResidual-SensorNav-Direct-v0"


def main() -> int:
    output = args.output_report or PACKAGE_ROOT / "results" / "phase3_safety_residual_smoke_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs, use_fabric=True)
    env = gym.make(TASK_ID, cfg=cfg)
    unwrapped = env.unwrapped
    try:
        observations, _ = env.reset()
        base = unwrapped._route_actions()
        zeros = torch.zeros_like(base)
        no_op, no_op_brake, no_op_attenuation = unwrapped._compose_residual_actions(base, zeros)

        full_brake_residual = torch.zeros_like(base)
        full_brake_residual[:, 0] = -1.0
        full_brake, brake_fraction, _ = unwrapped._compose_residual_actions(
            base, full_brake_residual
        )

        maximum_attenuation_residual = torch.zeros_like(base)
        maximum_attenuation_residual[:, 1] = -1.0
        attenuated, _, attenuation = unwrapped._compose_residual_actions(
            base, maximum_attenuation_residual
        )

        positive_residual = torch.ones_like(base)
        positive_no_op, _, _ = unwrapped._compose_residual_actions(base, positive_residual)

        finite_observations = True
        for _ in range(args.steps):
            observations, _, _, _, _ = env.step(torch.zeros(env.action_space.shape, device=unwrapped.device))
            finite_observations &= bool(torch.isfinite(observations["policy"]).all().item())

        base_steering = base[:, 1]
        attenuated_steering = attenuated[:, 1]
        sign_preserved = torch.all(
            (base_steering == 0.0)
            | (torch.sign(base_steering) == torch.sign(attenuated_steering))
        )
        checks = {
            "task_registered": env.spec.id == TASK_ID,
            "observation_contract_46": observations["policy"].shape[-1] == 46,
            "observation_values_finite": finite_observations,
            "frozen_route_checkpoint_hash_matches": (
                unwrapped._frozen_route_checkpoint_actual_sha256
                == cfg.frozen_route_checkpoint_sha256
            ),
            "frozen_route_actor_has_no_trainable_parameters": not any(
                parameter.requires_grad for parameter in unwrapped._frozen_route_actor.parameters()
            ),
            "zero_residual_is_exact_pass_through": bool(torch.equal(no_op, base)),
            "zero_residual_brake_is_zero": bool(torch.equal(no_op_brake, torch.zeros_like(no_op_brake))),
            "zero_residual_attenuation_is_zero": bool(
                torch.equal(no_op_attenuation, torch.zeros_like(no_op_attenuation))
            ),
            "positive_residual_is_no_op": bool(torch.equal(positive_no_op, base)),
            "full_brake_maps_forward_command_to_zero_speed": bool(
                torch.allclose(full_brake[:, 0], -torch.ones_like(full_brake[:, 0]))
            ),
            "full_brake_fraction_is_one": bool(
                torch.equal(brake_fraction, torch.ones_like(brake_fraction))
            ),
            "steering_sign_is_preserved": bool(sign_preserved.item()),
            "steering_magnitude_never_increases": bool(
                torch.all(torch.abs(attenuated_steering) <= torch.abs(base_steering) + 1.0e-7).item()
            ),
            "steering_attenuation_is_bounded": bool(
                torch.allclose(
                    attenuation,
                    torch.full_like(attenuation, cfg.maximum_angular_attenuation),
                )
            ),
            "full_domain_randomization_strength": unwrapped._curriculum_strength() == 1.0,
        }
        report = {
            "report_type": "phase3_bounded_safety_residual_smoke",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": TASK_ID,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "policy_rate_hz": round(1.0 / unwrapped.step_dt),
            "frozen_route_checkpoint": str(unwrapped._frozen_route_checkpoint_path),
            "frozen_route_checkpoint_sha256": unwrapped._frozen_route_checkpoint_actual_sha256,
            "maximum_angular_attenuation": cfg.maximum_angular_attenuation,
            "sample_base_actions": base.detach().cpu().tolist(),
            "checks": checks,
            "passed": all(checks.values()),
            "claim_boundary": (
                "Action-boundary and simulation-runtime evidence only; not a trained-policy "
                "acceptance result, human-safety certification, or physical release."
            ),
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PHASE3_SAFETY_RESIDUAL_SMOKE_REPORT={output.resolve()}")
        print(f"PHASE3_SAFETY_RESIDUAL_SMOKE_PASSED={report['passed']}")
        return 0 if report["passed"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
