#!/usr/bin/env python3
"""Validate native Nav2 costmap marking and spatial detour evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK = "Isaac-AISHA-Phase7C-NativeCostmap-Detour-Safety-Direct-v0"
PHASE6_SHA256 = (
    "e49767507925548aa0086c38e764c43037f25734943b2c5712cb58eecb0b6318"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=ROOT / "results/phase7c_native_costmap_detour_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "results/phase7c_native_costmap_detour_bridge.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase7c_native_costmap_detour_integration_gate.json",
    )
    args = parser.parse_args()

    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    phase6 = load_json(
        ROOT / "results/administration_nav2_phase6_high_speed_integration_gate.json"
    )
    phase7a = load_json(
        ROOT / "results/administration_nav2_phase7_dynamic_integration_gate.json"
    )
    phase7b = load_json(
        ROOT
        / "results/administration_nav2_phase7b_blocked_route_integration_gate.json"
    )
    nav2 = yaml.safe_load(
        (ROOT / "config/nav2_phase7c_native_detour_params.yaml").read_text(
            encoding="utf-8"
        )
    )

    baseline = mission.get("baseline", {})
    before = mission.get("costmap_before_activation", {})
    marked = mission.get("native_global_costmap_marking", {})
    detour = mission.get("detour", {})
    execution = mission.get("execution", {})
    blocked = bridge.get("blocked_route", {})
    learned = bridge.get("learned_safety", {})
    events = bridge.get("events", {})
    localization = bridge.get("localization", {})
    outcomes = bridge.get("termination_diagnostics", {}).get(
        "episode_outcomes", {}
    )
    messages = bridge.get("topics", {}).get("message_counts", {})
    local_obstacle = nav2["local_costmap"]["local_costmap"]["ros__parameters"][
        "obstacle_layer"
    ]
    global_obstacle = nav2["global_costmap"]["global_costmap"]["ros__parameters"][
        "obstacle_layer"
    ]

    checks = {
        "phase6_static_baseline_retained": (
            phase6.get("passed") is True
            and phase6.get("checks_passed") == phase6.get("checks_total") == 28
        ),
        "phase7a_crossing_baseline_retained": (
            phase7a.get("passed") is True
            and phase7a.get("checks_passed") == phase7a.get("checks_total") == 24
        ),
        "phase7b_safe_wait_baseline_retained": (
            phase7b.get("passed") is True
            and phase7b.get("checks_passed") == phase7b.get("checks_total") == 26
        ),
        "bridge_passed": bridge.get("passed") is True,
        "native_detour_task_selected": bridge.get("task") == TASK,
        "mission_passed": mission.get("passed") is True,
        "static_baseline_selected_top_branch": (
            baseline.get("planning_status") == "succeeded"
            and baseline.get("branch") == "top"
            and float(baseline.get("mean_island_span_y_m", 0.0)) > 1.50
        ),
        "blocker_absent_from_initial_centre_cell": (
            before.get("available") is True
            and before.get("centre_cost") == 0
        ),
        "blockage_activation_synchronized": (
            mission.get("blockage_active_seen") is True
            and int(events.get("blockage_activation_requests", 0)) > 0
        ),
        "native_global_costmap_centre_became_lethal": (
            mission.get("native_nav2_dynamic_costmap_marking_credit") is True
            and marked.get("centre_cost") >= 253
        ),
        "native_lethal_surface_expanded_across_branch": (
            int(marked.get("lethal_or_inscribed_samples", 0))
            - int(before.get("lethal_or_inscribed_samples", 0))
            >= 18
        ),
        "fresh_nav2_path_selected_bottom_branch": (
            detour.get("planning_status") == "succeeded"
            and detour.get("branch") == "bottom"
            and float(detour.get("mean_island_span_y_m", 0.0)) < -1.50
        ),
        "detour_is_spatially_distinct_and_longer": (
            mission.get("spatial_detour_credit") is True
            and float(detour.get("path_length_m", 0.0))
            >= float(baseline.get("path_length_m", 0.0)) + 0.50
        ),
        "detour_executed_successfully": (
            execution.get("attempted") is True
            and execution.get("succeeded") is True
            and execution.get("status") == "succeeded"
        ),
        "goal_disc_reached": (
            mission.get("final_goal_distance_m") is not None
            and float(mission["final_goal_distance_m"]) <= 0.32
        ),
        "blockage_remained_active_during_execution": (
            mission.get("blockage_active_at_execution_end") is True
            and blocked.get("triggered") is True
            and blocked.get("cleared") is False
            and int(blocked.get("active_steps", 0)) > 0
            and int(events.get("blockage_release_requests", 0)) == 0
        ),
        "episode_never_reset_or_collided": (
            events.get("episode_reset_gate_detected") is False
            and outcomes.get("collision", False) is False
            and outcomes.get("dynamic_obstacle_collision", False) is False
            and outcomes.get("static_collision", False) is False
        ),
        "mission_completion_reached_bridge": (
            events.get("mission_complete_signal_received") is True
        ),
        "accepted_phase6_checkpoint_loaded": (
            learned.get("checkpoint_sha256") == PHASE6_SHA256
            and learned.get("checkpoint_is_accepted_phase6") is True
            and learned.get("base_command_source") == "nav2_cmd_vel"
        ),
        "learned_360_safety_remained_in_loop": (
            bridge.get("learned_360_safety_coupled") is True
            and int(learned.get("authority_steps", 0)) > 0
            and int(learned.get("brake_steps", 0)) > 0
        ),
        "positive_360_clearance_retained": (
            float(bridge.get("minimum_ring_clearance_m", -1.0)) > 0.08
        ),
        "no_privileged_policy_blocker_state": (
            mission.get("scenario_state_exposed_to_policy") is False
            and blocked.get("blocker_state_exposed_to_policy") is False
            and blocked.get("coordination_state_exposed_to_mission_only") is True
        ),
        "native_nav2_observation_claim_is_explicit": (
            blocked.get("planner_observation_source")
            == "native_nav2_obstacle_layer_from_live_isaac_laserscan"
        ),
        "real_alternate_route_topology_is_explicit": (
            blocked.get("route_topology")
            == "two_route_loop_spatial_detour_available"
        ),
        "live_scan_streams_exercised": (
            int(messages.get("scan", 0)) > 100
            and int(messages.get("front_scan", 0)) > 100
            and int(messages.get("blocked_route_active", 0)) > 100
        ),
        "per_source_height_filter_fix_is_explicit": all(
            float(layer[source]["max_obstacle_height"]) >= minimum
            for layer in (local_obstacle, global_obstacle)
            for source, minimum in (("crown_scan", 2.20), ("front_scan", 2.00))
        ),
        "both_costmaps_consume_both_live_scans": all(
            layer.get("observation_sources") == "crown_scan front_scan"
            and layer["crown_scan"].get("marking") is True
            and layer["front_scan"].get("marking") is True
            for layer in (local_obstacle, global_obstacle)
        ),
        "command_speed_stayed_within_phase7c_profile": (
            0.30 <= float(mission.get("maximum_commanded_linear_mps", 0.0)) <= 0.451
        ),
        "localization_and_physical_claim_boundary_retained": (
            localization.get("nav2_global_pose_source")
            == "isaac_ground_truth_odom_with_identity_map_to_odom"
            and localization.get("physical_localization_credit") is False
            and bridge.get("physical_release") is False
            and blocked.get("physical_safety_credit") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "phase7c_native_costmap_detour_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "accepted_native_nav2_dynamic_costmap_spatial_detour"
            if passed
            else "rejected"
        ),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "retention": {
            "phase6": {"passed": phase6.get("passed"), "checks": "28/28"},
            "phase7a": {"passed": phase7a.get("passed"), "checks": "24/24"},
            "phase7b": {"passed": phase7b.get("passed"), "checks": "26/26"},
        },
        "native_costmap": {
            "before_centre_cost": before.get("centre_cost"),
            "blocked_centre_cost": marked.get("centre_cost"),
            "before_lethal_samples": before.get("lethal_or_inscribed_samples"),
            "blocked_lethal_samples": marked.get("lethal_or_inscribed_samples"),
            "source": blocked.get("planner_observation_source"),
        },
        "replanning": {
            "baseline_branch": baseline.get("branch"),
            "baseline_path_length_m": baseline.get("path_length_m"),
            "blocked_route_branch": detour.get("branch"),
            "detour_path_length_m": detour.get("path_length_m"),
            "executed": execution.get("succeeded"),
            "final_goal_distance_m": mission.get("final_goal_distance_m"),
        },
        "learned_safety": {
            "checkpoint_sha256": learned.get("checkpoint_sha256"),
            "authority_steps": learned.get("authority_steps"),
            "brake_steps": learned.get("brake_steps"),
            "minimum_360_clearance_m": bridge.get("minimum_ring_clearance_m"),
        },
        "claim_boundary": (
            "Accepted only as an isolated Isaac Sim/Nav2 integration gate: live "
            "Isaac LaserScan returns marked Nav2's native costmap, changed a top-branch "
            "plan into a bottom-branch detour, and the learned-safety-coupled robot "
            "executed it. The administration hallway remains single-path; this does not "
            "add an alternate route there or provide physical localization, stopping-distance, "
            "sim-to-real, safety-certification, or deployment credit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PHASE7C_GATE passed={passed} "
        f"checks={report['checks_passed']}/{report['checks_total']} "
        f"report={args.output}"
    )
    if not passed:
        for name, value in checks.items():
            if not value:
                print(f"FAILED {name}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
