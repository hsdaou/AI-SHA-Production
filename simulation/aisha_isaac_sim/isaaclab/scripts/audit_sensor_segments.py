#!/usr/bin/env python3
"""Audit every Block A curriculum segment start against the live Isaac Lab scene."""

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
    cfg.start_lateral_jitter_m = 0.0
    cfg.start_yaw_jitter_rad = 0.0
    cfg.goal_jitter_m = 0.0
    cfg.fixed_segment_id = 0
    env = gym.make(task, cfg=cfg)
    unwrapped = env.unwrapped
    records = []

    for segment_id, (start, goal) in enumerate(ROUTE_SEGMENTS):
        unwrapped.cfg.fixed_segment_id = segment_id
        env.reset()
        for _ in range(2):
            env.step(torch.zeros((1, 2), device=unwrapped.device))
        collision, _, invalid = unwrapped._termination_masks()
        lidar = unwrapped._lidar_ranges()
        records.append(
            {
                "segment_id": segment_id,
                "start": start,
                "goal": goal,
                "start_xy_m": [float(value) for value in unwrapped._local_xy()[0].tolist()],
                "goal_distance_m": float(unwrapped._goal_geometry()[2][0].item()),
                "minimum_lidar_range_m": float(torch.amin(lidar[0]).item()),
                "collision_at_start": bool(collision[0].item()),
                "invalid_at_start": bool(invalid[0].item()),
            }
        )

    env.close()
    report = {
        "task": task,
        "segment_count": len(records),
        "all_starts_clear": all(
            not record["collision_at_start"] and not record["invalid_at_start"] for record in records
        ),
        "segments": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SEGMENT_AUDIT=" + json.dumps(report, sort_keys=True))
    return 0 if report["all_starts_clear"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
