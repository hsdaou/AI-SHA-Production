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
parser.add_argument(
    "--task",
    default="Isaac-AISHA-BlockA-Phase3-ClearancePlanner-SensorNav-Direct-v0",
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


TASK_ID = args.task


def main() -> int:
    is_targeted_recovery = "Phase3-TargetedRecovery" in TASK_ID
    is_targeted_recovery_training = "Phase3-TargetedRecoveryTraining" in TASK_ID
    default_report_name = (
        "phase3m_targeted_recovery_smoke_report.json"
        if is_targeted_recovery
        else "phase3l_clearance_planner_smoke_report.json"
    )
    output = args.output_report or PACKAGE_ROOT / "results" / default_report_name
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
        if is_targeted_recovery:
            target_ids = tuple(cfg.targeted_recovery_segment_ids)
            target_weight = sum(cfg.segment_sampling_weights[index] for index in target_ids)
            original_segment_ids = unwrapped._segment_ids.clone()
            original_peak_elapsed = unwrapped._peak_torque_elapsed_s.clone()
            stationary_pivot = torch.zeros_like(base)
            stationary_pivot[:, 0] = -1.0
            stationary_pivot[:, 1] = 1.0
            large_heading_error = torch.full(
                (args.num_envs,), 3.141592653589793, device=unwrapped.device
            )
            unwrapped._segment_ids.fill_(target_ids[0])
            unwrapped._peak_torque_elapsed_s.zero_()
            unwrapped._update_pivot_torque_limits(
                stationary_pivot, large_heading_error
            )
            peak_limits = unwrapped._robot.data.joint_effort_limits[
                :, unwrapped._wheel_ids
            ].clone()
            translating = stationary_pivot.clone()
            translating[:, 0] = 0.0
            unwrapped._update_pivot_torque_limits(translating, large_heading_error)
            translating_limits = unwrapped._robot.data.joint_effort_limits[
                :, unwrapped._wheel_ids
            ].clone()
            unwrapped._peak_torque_elapsed_s.fill_(cfg.peak_motor_time_limit_s)
            unwrapped._update_pivot_torque_limits(
                stationary_pivot, large_heading_error
            )
            exhausted_limits = unwrapped._robot.data.joint_effort_limits[
                :, unwrapped._wheel_ids
            ].clone()
            unwrapped._segment_ids.zero_()
            unwrapped._peak_torque_elapsed_s.zero_()
            unwrapped._update_pivot_torque_limits(
                stationary_pivot, large_heading_error
            )
            non_target_limits = unwrapped._robot.data.joint_effort_limits[
                :, unwrapped._wheel_ids
            ].clone()
            material_properties = (
                unwrapped._robot.root_physx_view.get_material_properties()
            )
            castor_shape_ids = unwrapped._castor_material_shape_ids
            castor_materials = material_properties[:, castor_shape_ids]
            unwrapped._segment_ids.copy_(original_segment_ids)
            unwrapped._peak_torque_elapsed_s.copy_(original_peak_elapsed)
            checks.update(
                {
                    "recovery_supervisor_mode_matches_task": bool(
                        cfg.recovery_supervisor_enabled
                        == (not is_targeted_recovery_training)
                    ),
                    "target_segments_are_4_6_9": target_ids == (4, 6, 9),
                    "all_route_segments_rehearsed": all(
                        weight > 0.0 for weight in cfg.segment_sampling_weights
                    ),
                    "target_segments_receive_majority_of_resets": (
                        target_weight / sum(cfg.segment_sampling_weights) > 0.50
                    ),
                    "targeted_shaping_does_not_bypass_planner": bool(
                        torch.all(
                            unwrapped._planner_applied_clearance
                            >= torch.where(
                                unwrapped._planner_request_accepted,
                                unwrapped._planner_candidate_clearance,
                                unwrapped._planner_baseline_clearance,
                            )
                            - 1.0e-7
                        ).item()
                    ),
                    "stationary_targeted_pivot_uses_declared_peak_torque": bool(
                        torch.all(peak_limits == cfg.peak_motor_effort_limit_nm).item()
                    ),
                    "translation_command_retains_rated_torque": bool(
                        torch.all(
                            translating_limits == cfg.rated_motor_effort_limit_nm
                        ).item()
                    ),
                    "peak_time_limit_is_enforced": bool(
                        torch.all(
                            exhausted_limits == cfg.rated_motor_effort_limit_nm
                        ).item()
                    ),
                    "non_target_segment_retains_rated_torque": bool(
                        torch.all(
                            non_target_limits == cfg.rated_motor_effort_limit_nm
                        ).item()
                    ),
                    "castor_proxy_static_friction_stays_in_low_band": bool(
                        torch.all(
                            (castor_materials[..., 0]
                             >= cfg.castor_static_friction_range[0] - 1.0e-7)
                            & (castor_materials[..., 0]
                               <= cfg.castor_static_friction_range[1] + 1.0e-7)
                        ).item()
                    ),
                    "castor_proxy_dynamic_friction_stays_in_low_band": bool(
                        torch.all(
                            (castor_materials[..., 1]
                             >= cfg.castor_dynamic_friction_range[0] - 1.0e-7)
                            & (castor_materials[..., 1]
                               <= cfg.castor_dynamic_friction_range[1] + 1.0e-7)
                        ).item()
                    ),
                    "office_pivot_supervisor_is_limited_to_segments_4_and_9": (
                        tuple(cfg.office_departure_segment_ids) == (4, 9)
                    ),
                    "office_pivot_supervisor_has_heading_hysteresis": (
                        cfg.pivot_supervisor_release_heading_error_rad
                        < cfg.pivot_supervisor_engage_heading_error_rad
                    ),
                    "office_pivot_rate_is_within_peak_and_task_limits": (
                        cfg.peak_pivot_minimum_angular_command_rad_s
                        <= cfg.pivot_supervisor_angular_command_rad_s
                        <= cfg.angular_velocity_max_rad_s
                    ),
                    "predictive_clearance_guard_is_scoped": (
                        tuple(cfg.predictive_stop_segment_ids) == (6, 10, 11)
                    ),
                    "predictive_clearance_guard_has_hysteresis": (
                        cfg.predictive_stop_trigger_clearance_m
                        < cfg.predictive_stop_release_clearance_m
                    ),
                    "dynamic_crossing_creep_is_bounded": (
                        cfg.linear_velocity_range_mps[0]
                        <= cfg.dynamic_crossing_predictive_creep_linear_velocity_mps
                        <= cfg.predictive_creep_linear_velocity_mps
                        < cfg.linear_velocity_range_mps[1]
                    ),
                }
            )
        report = {
            "report_type": (
                "phase3m_targeted_recovery_runtime_smoke"
                if is_targeted_recovery
                else "phase3l_clearance_planner_runtime_smoke"
            ),
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
            "targeted_peak_torque": (
                {
                    "rated_effort_limit_nm": cfg.rated_motor_effort_limit_nm,
                    "peak_effort_limit_nm": cfg.peak_motor_effort_limit_nm,
                    "peak_time_limit_s": cfg.peak_motor_time_limit_s,
                    "scope": "targeted stationary large-heading pivots only",
                }
                if is_targeted_recovery
                else None
            ),
            "targeted_castor_contact": (
                {
                    "model": "fixed_sphere_low_friction_proxy",
                    "static_friction_range": cfg.castor_static_friction_range,
                    "dynamic_friction_range": cfg.castor_dynamic_friction_range,
                    "material_shape_count": len(unwrapped._castor_material_shape_ids),
                }
                if is_targeted_recovery
                else None
            ),
            "targeted_recovery_supervisor": (
                {
                    "office_departure_segment_ids": cfg.office_departure_segment_ids,
                    "pivot_engage_heading_error_rad": (
                        cfg.pivot_supervisor_engage_heading_error_rad
                    ),
                    "pivot_release_heading_error_rad": (
                        cfg.pivot_supervisor_release_heading_error_rad
                    ),
                    "pivot_angular_command_rad_s": (
                        cfg.pivot_supervisor_angular_command_rad_s
                    ),
                    "predictive_guard_segment_ids": cfg.predictive_stop_segment_ids,
                    "predictive_creep_linear_velocity_mps": (
                        cfg.predictive_creep_linear_velocity_mps
                    ),
                    "dynamic_crossing_creep_linear_velocity_mps": (
                        cfg.dynamic_crossing_predictive_creep_linear_velocity_mps
                    ),
                }
                if is_targeted_recovery
                else None
            ),
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
