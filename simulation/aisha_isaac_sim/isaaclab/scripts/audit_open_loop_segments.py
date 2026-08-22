#!/usr/bin/env python3
"""Verify that straight, aligned wheel commands can traverse every route segment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--segment-ids", type=int, nargs="*", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import aisha_isaaclab.tasks  # noqa: E402,F401
from aisha_isaaclab.tasks.office_nav.block_a_sensor_env import ROUTE_SEGMENTS  # noqa: E402


def main() -> int:
    task = "Isaac-AISHA-BlockA-SensorNav-Direct-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1, use_fabric=True)
    cfg.seed = 4040
    cfg.start_lateral_jitter_m = 0.0
    cfg.start_yaw_jitter_rad = 0.0
    cfg.goal_jitter_m = 0.0
    cfg.fixed_segment_id = 0
    env = gym.make(task, cfg=cfg)
    device = env.unwrapped.device
    # Normalized forward action 0.4 maps to 0.35 m/s under the presentation
    # segment action contract. Starts are exactly aligned for this feasibility
    # audit, so zero angular command is intentional.
    action = torch.tensor([[0.4, 0.0]], dtype=torch.float32, device=device)
    records = []

    selected_segment_ids = args.segment_ids if args.segment_ids is not None else range(len(ROUTE_SEGMENTS))
    for segment_id in selected_segment_ids:
        start, goal = ROUTE_SEGMENTS[segment_id]
        env.unwrapped.cfg.fixed_segment_id = segment_id
        env.reset()
        result = None
        for step in range(env.unwrapped.max_episode_length + 2):
            _, _, terminated, truncated, extras = env.step(action)
            if bool((terminated | truncated)[0].item()):
                outcomes = extras["episode_outcomes"]
                outcome = "time_out"
                if bool(outcomes["success"][0].item()):
                    outcome = "success"
                elif bool(outcomes["collision"][0].item()):
                    outcome = "collision"
                result = {
                    "segment_id": segment_id,
                    "start": start,
                    "goal": goal,
                    "outcome": outcome,
                    "steps": step + 1,
                    "duration_s": (step + 1) / 30.0,
                    "final_goal_distance_m": float(outcomes["final_distance_m"][0].item()),
                    "minimum_lidar_range_m": float(outcomes["minimum_lidar_range_m"][0].item()),
                }
                break
        if result is None:
            raise RuntimeError(f"segment {segment_id} did not terminate")
        records.append(result)
        print("OPEN_LOOP_SEGMENT=" + json.dumps(result, sort_keys=True), flush=True)

    report = {
        "task": task,
        "controller": "constant 0.35 m/s aligned wheel command; no learned policy",
        "purpose": "physics/geometry feasibility audit only",
        "selected_segment_ids": list(selected_segment_ids),
        "all_segments_traversable": all(record["outcome"] == "success" for record in records),
        "segments": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"OPEN_LOOP_AUDIT={output}", flush=True)
    env.close()
    return 0 if report["all_segments_traversable"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
