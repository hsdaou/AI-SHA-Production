#!/usr/bin/env python3
"""Exercise the measured padded-door task and its simulation-only motion envelope."""

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
parser.add_argument("--steps", type=int, default=1100)
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


TASK = "Isaac-AISHA-BlockA-MeasuredTightDoor-SensorNav-Direct-v0"


def main() -> int:
    output = args.output_report or (
        PACKAGE_ROOT / "results" / "measured_tight_door_runtime_smoke.json"
    )
    cfg = parse_env_cfg(
        TASK, device=args.device, num_envs=args.num_envs, use_fabric=True
    )
    cfg.fixed_segment_id = 3
    cfg.start_lateral_jitter_m = 0.0
    cfg.start_yaw_jitter_rad = 0.0
    cfg.start_linear_velocity_range_mps = (0.0, 0.0)
    cfg.goal_jitter_m = 0.0
    cfg.action_latency_steps_range = (0, 0)
    cfg.motor_strength_scale_range = (1.0, 1.0)
    cfg.wheel_radius_scale_range = (1.0, 1.0)
    cfg.wheel_track_scale_range = (1.0, 1.0)
    cfg.curriculum_minimum_strength = 0.0
    env = gym.make(TASK, cfg=cfg)
    unwrapped = env.unwrapped
    try:
        observations, _ = env.reset()
        finite = True
        slow_steps = 0
        no_rotation_steps = 0
        maximum_speed_in_no_rotation_zone = 0.0
        maximum_speed_at_frame = 0.0
        maximum_applied_yaw_action_in_no_rotation_zone = 0.0
        reset_events = 0
        successful_crossings = 0
        for _ in range(args.steps):
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            actions[:, 0] = 1.0
            # Request steering only after the robot has already made a centred
            # approach.  This proves the frame envelope removes yaw authority
            # without deliberately spoiling the 11 mm/side padded alignment.
            door_relative = unwrapped._local_xy() - unwrapped._tight_door_centres[0]
            door_normal_distance = torch.abs(
                torch.sum(door_relative * unwrapped._tight_door_normals[0], dim=1)
            )
            actions[door_normal_distance <= 0.95, 1] = 0.04
            observations, _, terminated, truncated, _ = env.step(actions)
            finite &= bool(torch.isfinite(observations["policy"]).all().item())
            slow = unwrapped._tight_door_slow_zone_active
            straight = unwrapped._tight_door_no_rotation_active
            slow_steps += int(torch.count_nonzero(slow).item())
            no_rotation_steps += int(torch.count_nonzero(straight).item())
            if torch.any(straight):
                maximum_speed_in_no_rotation_zone = max(
                    maximum_speed_in_no_rotation_zone,
                    float(
                        torch.amax(
                            torch.abs(unwrapped._robot.data.root_lin_vel_b[straight, 0])
                        ).item()
                    ),
                )
                maximum_applied_yaw_action_in_no_rotation_zone = max(
                    maximum_applied_yaw_action_in_no_rotation_zone,
                    float(torch.amax(torch.abs(unwrapped._actions[straight, 1])).item()),
                )
                at_frame = straight & (door_normal_distance <= 0.30)
                if torch.any(at_frame):
                    maximum_speed_at_frame = max(
                        maximum_speed_at_frame,
                        float(
                            torch.amax(
                                torch.abs(unwrapped._robot.data.root_lin_vel_b[at_frame, 0])
                            ).item()
                        ),
                    )
            outcomes = unwrapped.extras.get("episode_outcomes", {})
            if outcomes:
                successful_crossings += int(
                    torch.count_nonzero(terminated & outcomes["success"]).item()
                )
            reset_events += int(torch.count_nonzero(terminated | truncated).item())

        course_report = json.loads(
            (PACKAGE_ROOT / "results" / "block_a_training_course_report.json").read_text(
                encoding="utf-8"
            )
        )
        checks = {
            "task_registered": env.spec.id == TASK,
            "observation_contract_46": observations["policy"].shape[-1] == 46,
            "action_contract_2": env.action_space.shape[-1] == 2,
            "observation_values_finite": finite,
            "measured_course_loaded": course_report.get("geometry_status")
            == "measured_site_presentation_candidate_training_proxy",
            "physical_doors_are_0_85m": all(
                abs(float(value) - 0.85) < 1.0e-9
                for value in course_report["door_clear_widths_m"].values()
            ),
            "physical_collision_apertures_are_0_85m": all(
                abs(float(value) - 0.85) < 1.0e-9
                for value in course_report["effective_training_apertures_m"].values()
            ),
            "padding_enforced_by_trajectory_acceptance": course_report.get(
                "padding_enforcement"
            )
            == "trajectory_acceptance_not_collision_geometry",
            "padded_centre_band_is_0_022m": all(
                abs(float(value) - 0.022) < 1.0e-9
                for value in course_report[
                    "padded_route_acceptance_centre_bands_m"
                ].values()
            ),
            "central_drop_no_go_proxy_present": course_report["central_atrium_drop"]
            ["lidar_collision_proxy_segments"]
            == 8,
            "slow_zone_exercised": slow_steps > 0,
            "no_rotation_zone_exercised": no_rotation_steps > 0,
            "rotation_suppressed_at_frame": maximum_applied_yaw_action_in_no_rotation_zone
            <= 1.0e-6,
            "centred_physics_crossing_completed": successful_crossings > 0,
            "frame_speed_at_or_below_0_10mps_with_tolerance": maximum_speed_at_frame
            <= 0.105,
            "physical_release_disabled": course_report.get("physical_release") is False,
        }
        report = {
            "report_type": "measured_tight_door_runtime_smoke",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": TASK,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "checks": checks,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
            "passed": all(checks.values()),
            "metrics": {
                "slow_zone_env_steps": slow_steps,
                "no_rotation_zone_env_steps": no_rotation_steps,
                "maximum_speed_in_no_rotation_zone_mps": maximum_speed_in_no_rotation_zone,
                "maximum_speed_at_frame_mps": maximum_speed_at_frame,
                "maximum_applied_yaw_action_in_no_rotation_zone": (
                    maximum_applied_yaw_action_in_no_rotation_zone
                ),
                "reset_events": reset_events,
                "successful_crossings": successful_crossings,
                "final_local_xy_m": unwrapped._local_xy().tolist(),
                "final_root_linear_velocity_body_mps": unwrapped._robot.data.root_lin_vel_b[
                    :, :2
                ].tolist(),
                "final_wheel_joint_velocity_rad_s": unwrapped._robot.data.joint_vel[
                    :, unwrapped._wheel_ids
                ].tolist(),
                "final_wheel_target_rad_s": unwrapped._wheel_targets.tolist(),
                "final_applied_action": unwrapped._actions.tolist(),
                "final_minimum_lidar_range_m": torch.amin(
                    unwrapped._lidar_ranges(), dim=1
                ).tolist(),
                "final_lidar_envelope_collision": unwrapped._lidar_envelope_collision().tolist(),
            },
            "physical_release": False,
            "claim_boundary": (
                "Simulation-only task smoke; this is not learned-policy acceptance or a "
                "physical doorway safety release."
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"MEASURED_TIGHT_DOOR_SMOKE passed={report['passed']} "
            f"checks={report['checks_passed']}/{report['checks_total']} report={output}"
        )
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
