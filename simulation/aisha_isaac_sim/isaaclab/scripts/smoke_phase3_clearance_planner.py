#!/usr/bin/env python3
"""Verify the Phase 3L planner and protective-stop boundary in Isaac Lab."""

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


TASK_ID = "Isaac-AISHA-BlockA-Phase3-ClearancePlanner-SensorNav-Direct-v0"


def main() -> int:
    output = (
        args.output_report
        or PACKAGE_ROOT / "results" / "phase3l_clearance_planner_smoke_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=args.num_envs, use_fabric=True)
    env = gym.make(TASK_ID, cfg=cfg)
    unwrapped = env.unwrapped
    try:
        observations, _ = env.reset()
        base = unwrapped._route_actions()
        zeros = torch.zeros_like(base)
        no_request, no_request_brake, no_request_steering = (
            unwrapped._compose_planner_request(base, zeros)
        )
        full_brake_residual = torch.zeros_like(base)
        full_brake_residual[:, 0] = -1.0
        full_brake, brake_fraction, _ = unwrapped._compose_planner_request(
            base, full_brake_residual
        )
        positive_brake_residual = torch.zeros_like(base)
        positive_brake_residual[:, 0] = 1.0
        positive_brake_no_op, _, _ = unwrapped._compose_planner_request(
            base, positive_brake_residual
        )
        signed_steering_residual = torch.zeros_like(base)
        signed_steering_residual[:, 1] = torch.where(
            torch.arange(args.num_envs, device=unwrapped.device) % 2 == 0,
            torch.ones(args.num_envs, device=unwrapped.device),
            -torch.ones(args.num_envs, device=unwrapped.device),
        )
        steering_request, _, steering_delta = unwrapped._compose_planner_request(
            base, signed_steering_residual
        )

        clear_ranges = torch.full(
            (args.num_envs, cfg.lidar_training_bins),
            cfg.lidar_max_range_m,
            device=unwrapped.device,
        )
        protected_clear = unwrapped._apply_protective_stop(no_request, clear_ranges)
        ray_slice = slice(
            cfg.protective_stop_front_ray_start,
            cfg.protective_stop_front_ray_end,
        )
        forced_trigger_ranges = clear_ranges.clone()
        forced_trigger_ranges[:, ray_slice] = (
            unwrapped._lidar_envelope_ranges[ray_slice]
            + cfg.protective_stop_trigger_clearance_m
            - 0.01
        )
        protected_trigger = unwrapped._apply_protective_stop(
            no_request, forced_trigger_ranges
        )
        forced_hysteresis_ranges = clear_ranges.clone()
        forced_hysteresis_ranges[:, ray_slice] = (
            unwrapped._lidar_envelope_ranges[ray_slice]
            + 0.5
            * (
                cfg.protective_stop_trigger_clearance_m
                + cfg.protective_stop_release_clearance_m
            )
        )
        protected_hysteresis = unwrapped._apply_protective_stop(
            no_request, forced_hysteresis_ranges
        )
        protected_release = unwrapped._apply_protective_stop(no_request, clear_ranges)

        finite_observations = True
        unsafe_accepted = False
        maximum_observed_correction = 0.0
        protective_stop_observed = False
        for step in range(args.steps):
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            actions[:, 1] = 1.0 if (step // 20) % 2 == 0 else -1.0
            observations, _, _, _, _ = env.step(actions)
            finite_observations &= bool(torch.isfinite(observations["policy"]).all().item())
            unsafe_accepted |= bool(
                torch.any(
                    unwrapped._planner_request_accepted
                    & (
                        unwrapped._planner_candidate_clearance
                        < cfg.planner_minimum_predicted_clearance_m
                    )
                ).item()
            )
            maximum_observed_correction = max(
                maximum_observed_correction,
                float(torch.amax(torch.abs(unwrapped._applied_steering_request)).item()),
            )
            protective_stop_observed |= bool(
                torch.any(unwrapped._protective_stop_intervened).item()
            )

        maximum_normalized_correction = (
            cfg.maximum_lateral_correction_rad_s / cfg.angular_velocity_max_rad_s
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
                parameter.requires_grad
                for parameter in unwrapped._frozen_route_actor.parameters()
            ),
            "zero_policy_output_preserves_route_request": bool(
                torch.equal(no_request, base)
            ),
            "zero_policy_output_has_no_brake": bool(
                torch.equal(no_request_brake, torch.zeros_like(no_request_brake))
            ),
            "zero_policy_output_has_no_steering_request": bool(
                torch.equal(no_request_steering, torch.zeros_like(no_request_steering))
            ),
            "positive_brake_action_is_no_op": bool(
                torch.equal(positive_brake_no_op, base)
            ),
            "full_brake_maps_to_zero_forward_speed": bool(
                torch.allclose(full_brake[:, 0], -torch.ones_like(full_brake[:, 0]))
            ),
            "full_brake_fraction_is_one": bool(
                torch.equal(brake_fraction, torch.ones_like(brake_fraction))
            ),
            "signed_steering_request_is_bounded": bool(
                torch.all(torch.abs(steering_delta) <= maximum_normalized_correction + 1.0e-7).item()
            ),
            "steering_request_never_changes_forward_component": bool(
                torch.equal(steering_request[:, 0], base[:, 0])
            ),
            "clear_scan_does_not_stop": bool(torch.equal(protected_clear, no_request)),
            "trigger_scan_removes_forward_motion": bool(
                torch.allclose(protected_trigger[:, 0], -torch.ones_like(protected_trigger[:, 0]))
            ),
            "protective_stop_preserves_steering": bool(
                torch.equal(protected_trigger[:, 1], no_request[:, 1])
            ),
            "protective_stop_hysteresis_holds": bool(
                torch.allclose(
                    protected_hysteresis[:, 0],
                    -torch.ones_like(protected_hysteresis[:, 0]),
                )
            ),
            "protective_stop_releases_on_clear_scan": bool(
                torch.equal(protected_release, no_request)
            ),
            "planner_never_accepts_below_clearance_floor": not unsafe_accepted,
            "runtime_steering_request_stays_bounded": (
                maximum_observed_correction <= maximum_normalized_correction + 1.0e-7
            ),
            "full_domain_randomization_strength": unwrapped._curriculum_strength() == 1.0,
        }
        report = {
            "report_type": "phase3l_clearance_planner_runtime_smoke",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": TASK_ID,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "policy_rate_hz": round(1.0 / unwrapped.step_dt),
            "frozen_route_checkpoint": str(unwrapped._frozen_route_checkpoint_path),
            "frozen_route_checkpoint_sha256": (
                unwrapped._frozen_route_checkpoint_actual_sha256
            ),
            "planner": {
                "prediction_horizon_s": cfg.planner_prediction_horizon_s,
                "prediction_samples": cfg.planner_prediction_samples,
                "footprint_margin_m": cfg.planner_footprint_margin_m,
                "minimum_predicted_clearance_m": (
                    cfg.planner_minimum_predicted_clearance_m
                ),
                "maximum_lateral_correction_rad_s": (
                    cfg.maximum_lateral_correction_rad_s
                ),
                "maximum_observed_normalized_correction": maximum_observed_correction,
            },
            "protective_stop": {
                "trigger_clearance_beyond_envelope_m": (
                    cfg.protective_stop_trigger_clearance_m
                ),
                "release_clearance_beyond_envelope_m": (
                    cfg.protective_stop_release_clearance_m
                ),
                "observed_during_environment_rollout": protective_stop_observed,
            },
            "checks": checks,
            "passed": all(checks.values()),
            "claim_boundary": (
                "Simulation runtime and command-boundary evidence only; not a trained-policy "
                "acceptance result, human-safety certification, or physical release."
            ),
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PHASE3L_CLEARANCE_SMOKE_REPORT={output.resolve()}")
        print(f"PHASE3L_CLEARANCE_SMOKE_PASSED={report['passed']}")
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
