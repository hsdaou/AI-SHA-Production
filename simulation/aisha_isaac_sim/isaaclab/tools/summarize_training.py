#!/usr/bin/env python3
"""Create auditable JSON and chart evidence from an RSL-RL TensorBoard run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-json", type=Path, required=True)
parser.add_argument("--output-plot", type=Path, required=True)
parser.add_argument("--title", default="AI-SHA Isaac Lab PPO training evidence")
args = parser.parse_args()
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _points(accumulator: EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def _moving_average(values: list[float], window: int = 10) -> list[float]:
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(sum(values[start : index + 1]) / (index - start + 1))
    return averaged


def _first_at_or_above(steps: list[int], values: list[float], threshold: float) -> int | None:
    return next((step for step, value in zip(steps, values, strict=True) if value >= threshold), None)


def main() -> int:
    run = args.run.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    event_files = sorted(run.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"expected one TensorBoard event file in {run}, found {len(event_files)}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    accumulator = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    reward_steps, mean_reward = _points(accumulator, "Train/mean_reward")
    success_steps, success_rate = _points(accumulator, "Metrics/success_rate")
    collision_steps, collision_rate = _points(accumulator, "Metrics/collision_rate")
    fps_steps, total_fps = _points(accumulator, "Perf/total_fps")

    agent_params = yaml.safe_load((run / "params" / "agent.yaml").read_text(encoding="utf-8"))
    env_text = (run / "params" / "env.yaml").read_text(encoding="utf-8")
    env_match = re.search(r"(?m)^scene:\n  num_envs: ([0-9]+)$", env_text)
    if not env_match:
        raise RuntimeError("could not read scene.num_envs from params/env.yaml")
    num_envs = int(env_match.group(1))
    # RSL-RL preserves the absolute iteration index when a run resumes.  Count
    # event records for the executed budget instead of treating the final
    # absolute index as work performed by this continuation run.
    recorded_step_sets = (reward_steps, success_steps, collision_steps, fps_steps)
    iterations = max(len(steps) for steps in recorded_step_sets)
    first_iteration = min(min(steps) for steps in recorded_step_sets if steps)
    final_iteration = max(max(steps) for steps in recorded_step_sets if steps)
    transitions = iterations * num_envs * int(agent_params["num_steps_per_env"])

    report = {
        "report_type": "isaac_lab_training_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": _relative_or_absolute(run),
        "tensorboard_event_sha256": _sha256(event_files[0]),
        "checkpoint": {"path": _relative_or_absolute(checkpoint), "sha256": _sha256(checkpoint)},
        "protocol": {
            "algorithm": "RSL-RL PPO",
            "training_seed": int(agent_params["seed"]),
            "parallel_environments": num_envs,
            "steps_per_environment_per_iteration": int(agent_params["num_steps_per_env"]),
            "executed_iterations": iterations,
            "first_logged_iteration": first_iteration,
            "final_logged_iteration": final_iteration,
            "simulated_policy_transitions": transitions,
        },
        "training_log_metrics": {
            "final_mean_reward": mean_reward[-1],
            "peak_mean_reward": max(mean_reward),
            "final_training_batch_success_rate": success_rate[-1],
            "peak_training_batch_success_rate": max(success_rate),
            "final_training_batch_collision_rate": collision_rate[-1],
            "first_iteration_at_or_above_90_percent_training_success": _first_at_or_above(
                success_steps, success_rate, 0.90
            ),
            "mean_reported_training_fps": sum(total_fps) / len(total_fps),
        },
        "interpretation": (
            "Training-batch metrics show learning progress but are not held-out evaluation. "
            "Use the corresponding held-out evaluation report for deterministic unseen-seed results."
        ),
    }

    output_json = args.output_json.expanduser().resolve()
    output_plot = args.output_plot.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    figure, (reward_axis, outcome_axis) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    figure.patch.set_facecolor("#f6f8fb")
    for axis in (reward_axis, outcome_axis):
        axis.set_facecolor("#ffffff")
        axis.grid(True, alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)

    reward_axis.plot(reward_steps, mean_reward, color="#2457a7", linewidth=1.8)
    reward_axis.set_ylabel("Mean episode reward")
    reward_axis.set_title(args.title, loc="left", weight="bold")

    outcome_axis.plot(success_steps, success_rate, color="#2e8b57", alpha=0.22, linewidth=1.0)
    outcome_axis.plot(
        success_steps,
        _moving_average(success_rate),
        color="#2e8b57",
        linewidth=2.2,
        label="Training success (10-point moving average)",
    )
    outcome_axis.plot(collision_steps, collision_rate, color="#c44e52", alpha=0.20, linewidth=1.0)
    outcome_axis.plot(
        collision_steps,
        _moving_average(collision_rate),
        color="#c44e52",
        linewidth=2.2,
        label="Training collision (10-point moving average)",
    )
    outcome_axis.set_ylim(-0.03, 1.03)
    outcome_axis.set_ylabel("Episode outcome rate")
    outcome_axis.set_xlabel("PPO iteration")
    outcome_axis.legend(loc="lower right", frameon=False)
    figure.text(
        0.01,
        0.005,
        f"{num_envs} parallel PhysX environments • {transitions:,} transitions • "
        f"seed {int(agent_params['seed'])} • training metrics, not held-out evaluation",
        color="#4d5968",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    figure.savefig(output_plot, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"TRAINING_REPORT={output_json}")
    print(f"TRAINING_PLOT={output_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
