#!/usr/bin/env python3
"""Chain the learned Block A segments with a physical turn/dwell supervisor."""

from __future__ import annotations

import argparse
import copy
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    choices=(
        "Isaac-AISHA-BlockA-SensorNav-Direct-v0",
        "Isaac-AISHA-BlockA-Phase2-EndToEnd-SensorNav-Direct-v0",
        "Isaac-AISHA-Administration-Live-Direct-v0",
        "Isaac-AISHA-Administration-Live-Phase3-DynamicSafety-Direct-v0",
        "Isaac-AISHA-Administration-Live-Phase3-DynamicSafety-Presentation-Direct-v0",
    ),
    default="Isaac-AISHA-BlockA-SensorNav-Direct-v0",
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--segment-policy-checkpoint",
    action="append",
    default=[],
    metavar="SEGMENT_ID=CHECKPOINT",
    help=(
        "Select a learned specialist checkpoint for one route segment. May be repeated; "
        "all remaining segments use --checkpoint."
    ),
)
parser.add_argument("--output-report", type=Path, required=True)
parser.add_argument("--video-folder", type=Path, default=None)
parser.add_argument("--max-steps", type=int, default=7200)
parser.add_argument("--dwell-seconds", type=float, default=2.0)
parser.add_argument(
    "--route-control",
    choices=("hybrid", "policy-only"),
    default="hybrid",
    help="Use the Phase 1 turn/dwell supervisor or leave every action to the checkpoint.",
)
parser.add_argument("--debug-interval", type=int, default=300)
parser.add_argument("--trace-interval", type=int, default=3)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(-3.8, 0.0, 2.4))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.45, 0.0, 0.55))
parser.add_argument(
    "--camera-mode",
    choices=("follow", "cinematic"),
    default="follow",
    help="Use the adaptive route-leg follow camera or six fixed administration cameras.",
)
parser.add_argument("--seed", type=int, default=6084)
parser.add_argument(
    "--dynamic-obstacle-probability",
    type=float,
    default=None,
    help=(
        "Optional presentation-scenario override for Phase 3 dynamic tasks. "
        "Use zero for the clean final architecture run; formal dynamic gates "
        "must retain the task default."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.dynamic_obstacle_probability is not None and not (
    0.0 <= args.dynamic_obstacle_probability <= 1.0
):
    parser.error("--dynamic-obstacle-probability must be in [0, 1]")
if args.video_folder is not None:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import aisha_isaaclab.tasks  # noqa: E402,F401
from aisha_isaaclab.tasks.office_nav.block_a_sensor_env import ROUTE_SEGMENTS  # noqa: E402


CINEMATIC_SHOTS = (
    {
        "name": "Departure - central atrium",
        "eye": (-4.35, 0.0, 1.72),
        "target": (2.0, 0.0, 0.72),
        "segments": (0,),
    },
    {
        "name": "East administration hall - Vice-Principal approach",
        "eye": (6.0, 1.08, 1.75),
        "target": (15.0, -0.40, 0.72),
        "segments": (1, 2),
    },
    {
        "name": "Visit 1 - Vice-Principal office",
        "eye": (19.55, -7.55, 1.86),
        "target": (16.75, -6.20, 0.70),
        "segments": (3, 4),
    },
    {
        "name": "Return through the hall - Principal suite turn",
        # Keep the lens inside the open east hallway. The earlier y=2.50
        # position sat behind a wall surface and produced a uniform gray shot.
        "eye": (3.0, 1.00, 1.85),
        "target": (9.0, -0.45, 0.68),
        "segments": (5, 6),
    },
    {
        "name": "Visit 2 - Principal office",
        "eye": (10.85, -9.78, 1.82),
        "target": (8.25, -9.00, 0.72),
        "segments": (7, 8, 9),
    },
    {
        "name": "Mission complete - return to the atrium",
        "eye": (-4.35, 0.0, 1.72),
        "target": (2.50, -2.0, 0.72),
        "segments": (10, 11),
    },
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = args.device or "cuda:0"
    use_fabric = True
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1, use_fabric=use_fabric)
    env_cfg.seed = args.seed
    env_cfg.route_chain_mode = True
    env_cfg.episode_length_s = args.max_steps / 30.0
    env_cfg.start_lateral_jitter_m = 0.0
    env_cfg.start_yaw_jitter_rad = 0.0
    env_cfg.goal_jitter_m = 0.0
    if args.dynamic_obstacle_probability is not None:
        if not hasattr(env_cfg, "dynamic_obstacle_activation_probability"):
            raise ValueError(
                "--dynamic-obstacle-probability requires a Phase 3 dynamic task"
            )
        env_cfg.dynamic_obstacle_activation_probability = (
            args.dynamic_obstacle_probability
        )
    manual_administration_camera = (
        "Administration-Live" in args.task and args.video_folder is not None
    )
    manual_follow_camera = manual_administration_camera and args.camera_mode == "follow"
    manual_cinematic_camera = manual_administration_camera and args.camera_mode == "cinematic"
    env_cfg.viewer.origin_type = "world" if manual_administration_camera else "asset_root"
    env_cfg.viewer.asset_name = None if manual_administration_camera else "robot"
    env_cfg.viewer.eye = tuple(args.camera_eye)
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    env_cfg.viewer.resolution = (1280, 720)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = device

    raw_env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args.video_folder is not None else None,
    )
    live_stage_checks = None
    if "Administration-Live" in args.task:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        stage_paths = {
            "live_shell_body": "/World/envs/env_0/Robot/base_link/presentation_shell_link/Geometry/Body",
            "excluded_replay_robot": "/World/envs/env_0/Administration/AISHA",
            "live_robot_base": "/World/envs/env_0/Robot/base_link",
        }
        live_stage_checks = {}
        for name, prim_path in stage_paths.items():
            prim = stage.GetPrimAtPath(prim_path)
            live_stage_checks[name] = {
                "path": prim_path,
                "valid": bool(prim),
                "active": bool(prim and prim.IsActive()),
            }
        print("LIVE_STAGE_CHECKS=" + json.dumps(live_stage_checks, sort_keys=True), flush=True)
    if args.video_folder is not None:
        video_folder = args.video_folder.expanduser().resolve()
        video_folder.mkdir(parents=True, exist_ok=True)
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=args.max_steps,
            name_prefix="aisha-block-a-learned-route",
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

    segment_policies: dict[int, object] = {}
    segment_policy_networks: dict[int, object] = {}
    segment_policy_checkpoints: dict[int, Path] = {}
    for specification in args.segment_policy_checkpoint:
        try:
            segment_text, checkpoint_text = specification.split("=", 1)
            segment_id = int(segment_text)
        except ValueError as error:
            raise ValueError(
                f"invalid --segment-policy-checkpoint {specification!r}; expected SEGMENT_ID=CHECKPOINT"
            ) from error
        if segment_id < 0 or segment_id >= len(ROUTE_SEGMENTS):
            raise ValueError(f"specialist segment id out of range: {segment_id}")
        specialist_checkpoint = Path(checkpoint_text).expanduser().resolve()
        if not specialist_checkpoint.is_file():
            raise FileNotFoundError(specialist_checkpoint)
        specialist_network = copy.deepcopy(policy_nn)
        specialist_state = torch.load(
            specialist_checkpoint,
            map_location=env.unwrapped.device,
            weights_only=False,
        )["model_state_dict"]
        specialist_network.load_state_dict(specialist_state)
        specialist_network.eval()
        segment_policies[segment_id] = specialist_network.act_inference
        segment_policy_networks[segment_id] = specialist_network
        segment_policy_checkpoints[segment_id] = specialist_checkpoint

    observations = env.get_observations()
    camera_distance_min_m = math.inf
    camera_distance_max_m = 0.0
    latest_lidar_min_m = math.nan
    cinematic_camera_events: list[dict[str, object]] = []
    active_cinematic_shot_index: int | None = None

    def update_manual_follow_camera() -> None:
        nonlocal camera_distance_min_m, camera_distance_max_m, latest_lidar_min_m
        if not manual_follow_camera:
            return
        position = env.unwrapped._robot.data.root_pos_w[0]
        quat = env.unwrapped._robot.data.root_quat_w[0]
        yaw = torch.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2].square() + quat[3].square()),
        )
        cos_yaw = float(torch.cos(yaw).item())
        sin_yaw = float(torch.sin(yaw).item())

        segment_id = int(env.unwrapped._segment_ids[0].item())
        route_delta = (
            env.unwrapped._segment_goals[segment_id]
            - env.unwrapped._segment_starts[segment_id]
        )
        route_yaw = math.atan2(float(route_delta[1].item()), float(route_delta[0].item()))
        cos_route = math.cos(route_yaw)
        sin_route = math.sin(route_yaw)

        eye_offset = tuple(args.camera_eye)
        desired_distance = math.hypot(eye_offset[0], eye_offset[1])
        if desired_distance > 1.0e-6:
            world_eye_angle = route_yaw + math.atan2(eye_offset[1], eye_offset[0])
            eye_angle = math.atan2(
                math.sin(world_eye_angle - float(yaw.item())),
                math.cos(world_eye_angle - float(yaw.item())),
            )
            ray_index = round((math.degrees(eye_angle) + 180.0) / 10.0) % 36
            lidar_ranges = env.unwrapped._lidar_ranges()[0]
            latest_lidar_min_m = float(lidar_ranges.min().item())
            # Anchor the shot to the route leg, not the robot's instantaneous yaw,
            # so a learned pivot does not sweep the camera through a wall. A narrow
            # visibility fan catches a doorway jamb without reacting to side walls.
            root_clearances = []
            for ray_offset in range(-1, 2):
                candidate_index = (ray_index + ray_offset) % 36
                candidate_angle = math.radians(-180.0 + 10.0 * candidate_index)
                environment_range = float(lidar_ranges[candidate_index].item())
                root_clearances.append(
                    environment_range
                    + env.unwrapped.cfg.lidar_x_m * math.cos(candidate_angle)
                )
            root_to_obstacle = min(root_clearances)
            safe_distance = max(0.30, root_to_obstacle - 0.20)
            actual_distance = min(desired_distance, safe_distance)
            scale = actual_distance / desired_distance
            eye_offset = (eye_offset[0] * scale, eye_offset[1] * scale, eye_offset[2])
            camera_distance_min_m = min(camera_distance_min_m, actual_distance)
            camera_distance_max_m = max(camera_distance_max_m, actual_distance)

        def world_point(offset: tuple[float, float, float]) -> tuple[float, float, float]:
            return (
                float(position[0].item()) + cos_route * offset[0] - sin_route * offset[1],
                float(position[1].item()) + sin_route * offset[0] + cos_route * offset[1],
                float(position[2].item()) + offset[2],
            )

        env.unwrapped.sim.set_camera_view(
            eye=world_point(eye_offset),
            target=world_point(tuple(args.camera_lookat)),
        )

    def synchronize_cinematic_sensor_read() -> None:
        """Preserve the passing live run's pre-policy ray-read schedule."""
        nonlocal latest_lidar_min_m
        if not manual_cinematic_camera:
            return
        lidar_ranges = env.unwrapped._lidar_ranges()[0]
        latest_lidar_min_m = float(lidar_ranges.min().item())

    def update_cinematic_camera(step: int) -> None:
        nonlocal active_cinematic_shot_index
        if not manual_cinematic_camera:
            return
        segment_id = int(env.unwrapped._segment_ids[0].item())
        shot_index = next(
            index
            for index, shot in enumerate(CINEMATIC_SHOTS)
            if segment_id in shot["segments"]
        )
        if shot_index == active_cinematic_shot_index:
            return
        shot = CINEMATIC_SHOTS[shot_index]
        origin = env.unwrapped.scene.env_origins[0]

        def world_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
            return tuple(float(origin[index].item()) + point[index] for index in range(3))

        env.unwrapped.sim.set_camera_view(
            eye=world_point(shot["eye"]),
            target=world_point(shot["target"]),
        )
        active_cinematic_shot_index = shot_index
        event = {
            "shot_index": shot_index + 1,
            "name": shot["name"],
            "segment_id": segment_id,
            "step": step,
            "elapsed_s": step / 30.0,
            "eye": list(shot["eye"]),
            "target": list(shot["target"]),
        }
        cinematic_camera_events.append(event)
        print("CINEMATIC_CAMERA=" + json.dumps(event, sort_keys=True), flush=True)

    synchronize_cinematic_sensor_read()
    update_manual_follow_camera()
    update_cinematic_camera(0)
    dwell_steps = max(0, round(args.dwell_seconds * 30.0))
    dwell_remaining = 0
    waypoint_events = []
    pose_trace = []
    learned_policy_steps = 0
    learned_policy_steps_by_source = {"base": 0}
    learned_policy_steps_by_source.update(
        {f"segment_{segment_id}_specialist": 0 for segment_id in segment_policies}
    )
    supervisor_turn_steps = 0
    supervisor_dwell_steps = 0
    final_outcome = "max_steps"
    termination_details = None

    for step in range(args.max_steps):
        synchronize_cinematic_sensor_read()
        update_manual_follow_camera()
        update_cinematic_camera(step)
        with torch.inference_mode():
            active_segment_id = int(env.unwrapped._segment_ids[0].item())
            active_policy = segment_policies.get(active_segment_id, policy)
            active_policy_source = (
                f"segment_{active_segment_id}_specialist"
                if active_segment_id in segment_policies
                else "base"
            )
            actions = active_policy(observations)
            heading_error = env.unwrapped._goal_geometry()[3]
            control_mode = "learned_sensor_policy"
            if args.route_control == "hybrid" and actions.shape[1] != 2:
                raise RuntimeError(
                    "hybrid route overrides require a two-action wheel policy; "
                    "run the Phase 3N safety task with --route-control policy-only"
                )
            if args.route_control == "hybrid" and dwell_remaining > 0:
                actions[0, 0] = -1.0
                actions[0, 1] = 0.0
                dwell_remaining -= 1
                supervisor_dwell_steps += 1
                control_mode = "presentation_dwell"
            elif (
                args.route_control == "hybrid"
                and abs(float(heading_error[0].item())) > math.radians(8.0)
            ):
                actions[0, 0] = -1.0
                # The rigid proxy needs a minimum differential-wheel command to overcome
                # static friction. Keep the physical pivot speed constant, then hand back
                # to the learned policy once the new segment is aligned.
                actions[0, 1] = 0.75 if heading_error[0] > 0.0 else -0.75
                supervisor_turn_steps += 1
                control_mode = "physics_supervisor_turn"
                if args.debug_interval > 0 and step % args.debug_interval == 0:
                    print(
                        "ROUTE_TURN="
                        + json.dumps(
                            {
                                "step": step,
                                "segment_id": int(env.unwrapped._segment_ids[0].item()),
                                "heading_error_deg": math.degrees(float(heading_error[0].item())),
                                "yaw_rate_rad_s": float(
                                    env.unwrapped._robot.data.root_ang_vel_b[0, 2].item()
                                ),
                                "angular_action": float(actions[0, 1].item()),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            else:
                learned_policy_steps += 1
                learned_policy_steps_by_source[active_policy_source] += 1

            observations, _, dones, extras = env.step(actions)
            applied_commands = env.unwrapped._actions.clone()
            policy_nn.reset(dones)
            for specialist_network in segment_policy_networks.values():
                specialist_network.reset(dones)

        if args.debug_interval > 0 and (step + 1) % args.debug_interval == 0:
            _, _, debug_distance, debug_heading = env.unwrapped._goal_geometry()
            debug_state = {
                "step": step + 1,
                "segment_id": int(env.unwrapped._segment_ids[0].item()),
                "goal_distance_m": float(debug_distance[0].item()),
                "heading_error_deg": math.degrees(float(debug_heading[0].item())),
                "linear_velocity_mps": float(
                    env.unwrapped._robot.data.root_lin_vel_b[0, 0].item()
                ),
                "safety_action": [float(value.item()) for value in actions[0]],
                "applied_command": [
                    float(value.item()) for value in applied_commands[0]
                ],
            }
            for name in (
                "_protective_stop_latched",
                "_predictive_stop_latched",
                "_pivot_supervisor_latched",
                "_safety_authority_active",
            ):
                if hasattr(env.unwrapped, name):
                    debug_state[name.removeprefix("_")] = bool(
                        getattr(env.unwrapped, name)[0].item()
                    )
            print("ROUTE_DEBUG=" + json.dumps(debug_state, sort_keys=True), flush=True)

        if args.trace_interval > 0 and (step + 1) % args.trace_interval == 0:
            local_xy = env.unwrapped._local_xy()[0]
            quat = env.unwrapped._robot.data.root_quat_w[0]
            yaw = torch.atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2].square() + quat[3].square()),
            )
            pose_trace.append(
                {
                    "step": step + 1,
                    "elapsed_s": (step + 1) / 30.0,
                    "x_m": round(float(local_xy[0].item()), 5),
                    "y_m": round(float(local_xy[1].item()), 5),
                    "yaw_rad": round(float(yaw.item()), 6),
                    "segment_id": int(env.unwrapped._segment_ids[0].item()),
                    "control_mode": control_mode,
                    "policy_source": active_policy_source,
                    "linear_velocity_mps": round(
                        float(env.unwrapped._robot.data.root_lin_vel_b[0, 0].item()), 5
                    ),
                    "yaw_rate_rad_s": round(
                        float(env.unwrapped._robot.data.root_ang_vel_b[0, 2].item()), 5
                    ),
                    "minimum_lidar_range_m": round(
                        latest_lidar_min_m, 5
                    ),
                    "policy_action": [
                        round(float(value.item()), 5) for value in actions[0]
                    ],
                    "applied_frozen_stack_command": [
                        round(float(value.item()), 5)
                        for value in applied_commands[0]
                    ],
                }
            )

        route_chain = extras.get("route_chain", {})
        if route_chain and bool(route_chain["waypoint_reached"][0].item()):
            reached_segment_id = int(route_chain["reached_segment_id"][0].item())
            start, goal = ROUTE_SEGMENTS[reached_segment_id]
            waypoint_events.append(
                {
                    "segment_id": reached_segment_id,
                    "start": start,
                    "goal": goal,
                    "step": step + 1,
                    "elapsed_s": (step + 1) / 30.0,
                }
            )
            print("ROUTE_WAYPOINT=" + json.dumps(waypoint_events[-1], sort_keys=True), flush=True)
            if args.route_control == "hybrid" and reached_segment_id in (3, 8):
                dwell_remaining = dwell_steps

        if bool(dones[0].item()):
            outcomes = extras["episode_outcomes"]
            if bool(outcomes["success"][0].item()):
                final_outcome = "success"
            elif bool(outcomes["collision"][0].item()):
                final_outcome = "collision"
            else:
                final_outcome = "time_out"
            final_xy = outcomes["position_xy_m"][0]
            minimum_ray_index = int(outcomes["minimum_lidar_ray_index"][0].item())
            lidar_hit_world = env.unwrapped._crown_lidar.data.ray_hits_w[0, minimum_ray_index]
            termination_details = {
                "segment_id": int(outcomes["segment_id"][0].item()),
                "position_xy_m": [float(final_xy[0].item()), float(final_xy[1].item())],
                "final_goal_distance_m": float(outcomes["final_distance_m"][0].item()),
                "minimum_lidar_range_m": float(outcomes["minimum_lidar_range_m"][0].item()),
                "minimum_lidar_ray_index": minimum_ray_index,
                "minimum_lidar_ray_angle_deg": -180.0 + 10.0 * minimum_ray_index,
                "minimum_lidar_hit_world_xyz_m": [float(value.item()) for value in lidar_hit_world],
            }
            for key in (
                "dynamic_obstacle_collision",
                "static_collision",
                "safety_steps",
                "safety_authority_steps",
                "safety_brake_fraction_sum",
                "minimum_ring_clearance_m",
            ):
                if key in outcomes:
                    value = outcomes[key][0]
                    termination_details[key] = (
                        bool(value.item())
                        if value.dtype == torch.bool
                        else float(value.item())
                    )
            completed_steps = step + 1
            break
    else:
        completed_steps = args.max_steps

    report = {
        "report_type": (
            "end_to_end_policy_route_playback"
            if args.route_control == "policy-only"
            else "hybrid_learned_block_a_route_playback"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "execution_mode": (
            "checkpoint_policy_live_in_walkthrough_matched_administration_scene"
            if "Administration-Live" in args.task
            else "checkpoint_policy_live_in_plan_derived_training_course"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": args.seed,
        "outcome": final_outcome,
        "completed_steps": completed_steps,
        "duration_s": completed_steps / 30.0,
        "waypoints_completed": len(waypoint_events),
        "route_segment_count": len(ROUTE_SEGMENTS),
        "camera": {
            "mode": args.camera_mode if manual_administration_camera else "viewer",
            "eye": list(args.camera_eye),
            "lookat": list(args.camera_lookat),
        },
        "camera_tracking": (
            "static_segment_cinematic_cameras"
            if manual_cinematic_camera
            else "manual_route_leg_follow"
            if manual_follow_camera
            else "isaaclab_viewer"
        ),
        "waypoint_events": waypoint_events,
        "pose_trace_interval_steps": args.trace_interval,
        "pose_trace": pose_trace,
        "control_steps": {
            "learned_sensor_policy": learned_policy_steps,
            "learned_sensor_policy_by_source": learned_policy_steps_by_source,
            "physics_supervisor_turn": supervisor_turn_steps,
            "presentation_dwell": supervisor_dwell_steps,
        },
        "route_control": args.route_control,
        "policy_architecture": (
            "frozen_phase3m_recovery_stack_plus_outer_recurrent_360_degree_brake_layer"
            if hasattr(env.unwrapped, "_frozen_recovery_actor")
            else "route_planner_selected_learned_skill_ensemble"
            if segment_policies
            else "single_learned_policy"
        ),
        "segment_policy_checkpoints": {
            str(segment_id): {
                "path": str(specialist_checkpoint),
                "sha256": sha256_file(specialist_checkpoint),
            }
            for segment_id, specialist_checkpoint in segment_policy_checkpoints.items()
        },
        "route_geometry_assumptions": getattr(
            env.unwrapped, "live_route_waypoint_assumptions", {}
        ),
        "termination_details": termination_details,
        "control_disclosure": (
            "The outer recurrent policy may only reduce translation while full-ring LiDAR clearance is closing; "
            "the hash-locked Phase 3M route, clearance, protective-stop, pivot and torque stack supplies every "
            "steering command. There is no turn or dwell action override and no root-transform animation."
            if hasattr(env.unwrapped, "_frozen_recovery_actor")
            else "Every wheel action is emitted by a learned LD19-style policy; the route planner selects a declared "
            "learned specialist on configured segments. There is no turn or dwell action override and no "
            "root-transform animation."
            if args.route_control == "policy-only"
            else "The learned LD19-style policy drives each aligned route segment. A deterministic wheel-command "
            "supervisor stops, dwells, and turns the physical robot between segments; no root-transform animation."
        ),
        "physics_rate_hz": 120.0,
        "policy_rate_hz": 30.0,
        "root_transform_animation": False,
        "route_transition_state": "continuous recurrent state preserved across all route legs",
        "presentation_dynamic_obstacle_probability": (
            args.dynamic_obstacle_probability
        ),
        "fabric_enabled": use_fabric,
        "claim_boundary": (
            "live checkpoint inference with wheel physics and ray sensing in the walkthrough-matched "
            "administration USD; presentation geometry is not an as-built survey, and this is not "
            "Nav2, sim-to-real, or physical release"
            if "Administration-Live" in args.task
            else "plan-derived training proxy; not photoreal, Nav2, sim-to-real, or physical release"
        ),
    }
    if "Administration-Live" in args.task:
        live_asset_paths = {
            "administration_scene": BUNDLE_ROOT / "scenes" / "administration.usd",
            "live_environment": BUNDLE_ROOT / "usd" / "administration_live_environment.usda",
            "presentation_shell": BUNDLE_ROOT / "usd" / "aisha_presentation_shell.usda",
            "presentation_robot": BUNDLE_ROOT / "usd" / "aisha_loaded_presentation.usda",
        }
        report["live_assets"] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in live_asset_paths.items()
        }
        report["live_stage_checks"] = live_stage_checks
        if manual_follow_camera:
            report["camera"]["clearance_adaptation"] = (
                "three_ray_route_leg_visibility_fan_with_0.20_m_buffer"
            )
            report["camera"]["actual_distance_range_m"] = [
                camera_distance_min_m,
                camera_distance_max_m,
            ]
        if manual_cinematic_camera:
            report["camera"]["shot_definitions"] = [
                {
                    "shot_index": index + 1,
                    "name": shot["name"],
                    "segments": list(shot["segments"]),
                    "eye": list(shot["eye"]),
                    "target": list(shot["target"]),
                }
                for index, shot in enumerate(CINEMATIC_SHOTS)
            ]
            report["camera"]["shot_events"] = cinematic_camera_events
    output = args.output_report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "ROUTE_RESULT="
        + json.dumps(
            {
                "outcome": report["outcome"],
                "waypoints_completed": report["waypoints_completed"],
                "route_segment_count": report["route_segment_count"],
                "completed_steps": report["completed_steps"],
                "duration_s": report["duration_s"],
                "control_steps": report["control_steps"],
                "termination_details": report["termination_details"],
                "output_report": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    env.close()
    return 0 if final_outcome == "success" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
