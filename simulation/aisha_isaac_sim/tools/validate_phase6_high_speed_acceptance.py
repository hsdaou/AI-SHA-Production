#!/usr/bin/env python3
"""Build the hash-linked Phase 6 high-speed simulation acceptance record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def combine(reports: list[dict]) -> dict:
    result_keys = (
        "success_count",
        "collision_count",
        "dynamic_obstacle_collision_count",
        "static_collision_count",
        "time_out_count",
    )
    counts = {
        key: sum(int(report["results"][key]) for report in reports)
        for key in result_keys
    }
    episodes = sum(int(report["protocol"]["episodes"]) for report in reports)
    segment_success_rates = [
        float(report["results"]["success_rate"]) for report in reports
    ]
    maximum_speeds = [
        float(report["planner_diagnostics"]["maximum_high_speed_segment_speed_mps"])
        for report in reports
    ]
    return {
        "episodes": episodes,
        **counts,
        "success_rate": counts["success_count"] / episodes,
        "dynamic_obstacle_collision_rate": (
            counts["dynamic_obstacle_collision_count"] / episodes
        ),
        "static_collision_rate": counts["static_collision_count"] / episodes,
        "minimum_segment_success_rate": min(segment_success_rates),
        "minimum_observed_peak_high_speed_mps": min(maximum_speeds),
        "maximum_observed_peak_high_speed_mps": max(maximum_speeds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "isaaclab/checkpoints/aisha_phase6_high_speed_080_model_223.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase6_high_speed_080_acceptance.json",
    )
    args = parser.parse_args()

    config_path = ROOT / "config/phase6_high_speed_curriculum.yaml"
    smoke_path = ROOT / "results/phase6_high_speed_065_smoke_report.json"
    flat_path = ROOT / "results/validation_high_speed_loaded.json"
    nav2_path = ROOT / "results/administration_nav2_measured_integration_gate.json"
    stage1_paths = [
        ROOT / "results/phase6_high_speed_065_model124_segment1_screen_seed10721.json",
        ROOT / "results/phase6_high_speed_065_model124_segment5_screen_seed10722.json",
    ]
    formal_paths = [
        ROOT / "results/phase6_high_speed_080_model223_segment1_formal_seed10741.json",
        ROOT / "results/phase6_high_speed_080_model223_segment5_formal_seed10742.json",
    ]

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    smoke = load_json(smoke_path)
    flat = load_json(flat_path)
    nav2 = load_json(nav2_path)
    stage1_reports = [load_json(path) for path in stage1_paths]
    formal_reports = [load_json(path) for path in formal_paths]
    stage1 = combine(stage1_reports)
    formal = combine(formal_reports)
    checkpoint_hash = sha256_file(args.checkpoint)
    formal_gate = config["formal_gate"]
    screen_gate = config["screen_gate"]
    stop = flat["physics"]["high_speed_stop"]
    straight = flat["physics"]["straight"][0]
    nav2_guard = nav2["live_stack"]["mapped_guard"]

    checks = {
        "checkpoint_hash_matches_both_formal_reports": all(
            report["checkpoint"]["sha256"] == checkpoint_hash
            for report in formal_reports
        ),
        "formal_protocol_is_64_unseen_episodes_per_high_speed_segment": all(
            report["protocol"]["episodes"]
            == formal_gate["episodes_per_high_speed_segment"]
            and report["protocol"]["fixed_segment_id"] == segment_id
            and report["protocol"]["phase3_curriculum_strength"] == 1.0
            and report["protocol"]["explicit_post_construction_reset"] is True
            for segment_id, report in zip(
                formal_gate["high_speed_route_segment_ids"], formal_reports
            )
        ),
        "stage1_screen_success_rate_passed": (
            stage1["success_rate"] >= screen_gate["combined_success_rate_min"]
        ),
        "stage1_screen_collision_rates_passed": (
            stage1["dynamic_obstacle_collision_rate"]
            <= screen_gate["dynamic_obstacle_collision_rate_max"]
            and stage1["static_collision_rate"]
            <= screen_gate["static_collision_rate_max"]
        ),
        "stage1_screen_speed_observed": (
            stage1["minimum_observed_peak_high_speed_mps"]
            >= screen_gate["maximum_high_speed_segment_speed_mps_min"]
        ),
        "formal_success_rate_passed": (
            formal["success_rate"]
            >= formal_gate["combined_high_speed_success_rate_min"]
        ),
        "formal_each_direction_success_rate_passed": (
            formal["minimum_segment_success_rate"]
            >= formal_gate["every_high_speed_segment_success_rate_min"]
        ),
        "formal_dynamic_collision_rate_passed": (
            formal["dynamic_obstacle_collision_rate"]
            <= formal_gate["dynamic_obstacle_collision_rate_max"]
        ),
        "formal_static_collision_rate_passed": (
            formal["static_collision_rate"]
            <= formal_gate["static_collision_rate_max"]
        ),
        "formal_080_speed_observed": (
            formal["minimum_observed_peak_high_speed_mps"]
            >= formal_gate["maximum_high_speed_segment_speed_mps_min_by_tier"][
                "stage_2"
            ]
        ),
        "runtime_contract_smoke_passed": (
            smoke["passed"] is True
            and smoke["checks_passed"] == smoke["checks_total"] == 32
        ),
        "flat_floor_speed_and_controlled_stop_preflight_passed": (
            flat["passed"] is True
            and straight["passed"] is True
            and stop["passed"] is True
            and stop["claim_boundary"].startswith("software-limited flat-floor")
        ),
        "geometry_and_sensor_configuration_frozen": (
            config["geometry"]["urdf_change_allowed"] is False
            and config["geometry"]["usd_collision_change_allowed"] is False
            and config["geometry"]["mass_change_allowed"] is False
            and config["geometry"]["sensor_geometry_change_allowed"] is False
        ),
        "doorway_limit_unchanged": (
            config["speed_envelope"]["maximum_doorway_speed_mps"] == 0.10
            and nav2_guard["maximum_doorway_speed_mps"] == 0.10
            and nav2_guard["maximum_abs_speed_in_doorway_mps"] <= 0.10
        ),
        "existing_measured_nav2_retention_gate_passed": (
            nav2["passed"] is True
            and nav2["checks_passed"] == nav2["checks_total"] == 27
        ),
        "physical_release_remains_false": config["physical_release"] is False,
    }
    passed = all(checks.values())
    report = {
        "report_type": "phase6_high_speed_080_simulation_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "accepted_simulation_hallway_tier_pending_measured_nav2_replay"
            if passed
            else "not_accepted"
        ),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "selected_checkpoint": {
            "path": relative(args.checkpoint),
            "sha256": checkpoint_hash,
            "training_run": (
                "isaaclab/logs/rsl_rl/aisha_block_a_sensor_nav/"
                "2026-08-23_23-48-45_phase6_high_speed_080_seed10701"
            ),
            "iteration": 223,
        },
        "stage1_screen": {
            "reports": [relative(path) for path in stage1_paths],
            **stage1,
        },
        "stage2_formal_gate": {
            "reports": [relative(path) for path in formal_paths],
            **formal,
            "requirements": formal_gate,
        },
        "flat_floor_preflight": {
            "report": relative(flat_path),
            "steady_speed_mps": straight["steady_speed_mps"],
            "stopping_distance_m": stop["stopping_distance_m"],
            "stopped_after_s": stop["stopped_after_s"],
            "physical_stopping_distance_credit": False,
        },
        "measured_nav2_retention": {
            "report": relative(nav2_path),
            "passed": nav2["passed"],
            "legs_completed": nav2["live_stack"]["legs_completed"],
            "maximum_abs_speed_in_doorway_mps": nav2_guard[
                "maximum_abs_speed_in_doorway_mps"
            ],
            "note": (
                "This is the already accepted 0.30 m/s measured-scene retention "
                "gate. The selected Phase 6 checkpoint has not yet been replayed "
                "through the measured Nav2/Omniverse mission."
            ),
        },
        "claim_boundary": {
            "supported": (
                "Isaac Lab full-strength domain-randomized 0.80 m/s target on "
                "declared straight hallway segments 1 and 5"
            ),
            "not_yet_supported": [
                "0.80 m/s measured-administration Nav2 mission replay",
                "final RTX Omniverse presentation run at the new hallway tier",
                "sim-to-real performance",
                "emergency stopping or human protective-field validation",
                "physical robot release",
            ],
            "geometry_changed": False,
            "doorway_speed_limit_mps": 0.10,
            "physical_release": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"PHASE6_HIGH_SPEED_ACCEPTANCE passed={passed} "
        f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
