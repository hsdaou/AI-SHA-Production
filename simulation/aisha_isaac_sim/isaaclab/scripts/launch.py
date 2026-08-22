#!/usr/bin/env python3
"""Register AI-SHA tasks and delegate to an installed Isaac Lab launcher."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# This import performs only Gym registration. The environment module itself is
# loaded after the official launcher has started Omniverse/Isaac Sim.
import aisha_isaaclab.tasks  # noqa: E402,F401


LAUNCHERS = {
    "list": ("scripts", "environments", "list_envs.py"),
    "zero": ("scripts", "environments", "zero_agent.py"),
    "random": ("scripts", "environments", "random_agent.py"),
    "train": ("scripts", "reinforcement_learning", "rsl_rl", "train.py"),
    "play": ("scripts", "reinforcement_learning", "rsl_rl", "play.py"),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in LAUNCHERS:
        choices = ", ".join(LAUNCHERS)
        raise SystemExit(f"usage: launch.py <{choices}> [launcher arguments]")
    mode = sys.argv.pop(1)
    isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", "/home/robot-wst/IsaacLab")).resolve()
    script = isaaclab_root.joinpath(*LAUNCHERS[mode])
    if not script.is_file():
        raise FileNotFoundError(f"Isaac Lab launcher not found: {script}")
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
