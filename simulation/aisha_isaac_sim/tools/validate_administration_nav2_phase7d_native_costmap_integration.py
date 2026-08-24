#!/usr/bin/env python3
"""Validate administration-native costmap marking and safe-wait replanning."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7D-"
    "NativeCostmap-SafeWait-Safety-Direct-v0"
)
CONTROL_STACK = "nav2_mapped_doorway_phase7d_native_costmap_safe_wait_safety"
PHASE6_SHA256 = (
    "e49767507925548aa0086c38e764c43037f25734943b2c5712cb58eecb0b6318"
)
PHASE3N_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def obstacle_layers(config: dict) -> list[dict]:
    return [
        config[name][name]["ros__parameters"]["obstacle_layer"]
        for name in ("local_costmap", "global_costmap")
    ]


def without_source_height_limits(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    for layer in obstacle_layers(normalized):
        for source in ("crown_scan", "front_scan"):
            layer[source].pop("max_obstacle_height", None)
            layer[source].pop("inf_is_valid", None)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7d_native_costmap_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7d_native_costmap_bridge.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/administration_nav2_phase7d_native_costmap_integration_gate.json"
        ),
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
    phase7c = load_json(
        ROOT / "results/phase7c_native_costmap_detour_integration_gate.json"
    )
    runtime_profile = yaml.safe_load(
        (ROOT / "config/nav2_administration_native_costmap_params.yaml").read_text(
            encoding="utf-8"
        )
    )
    frozen_profile = yaml.safe_load(
        (ROOT / "config/nav2_sim_tight_door_params.yaml").read_text(
            encoding="utf-8"
        )
    )

    blocked = bridge.get("blocked_route", {})
    replan = mission.get("blocked_route_replanning") or {}
    attempts = replan.get("attempts", [])
    safe_wait = replan.get("safe_wait", {})
    before = replan.get("costmap_before_activation", {})
    marked = replan.get("native_global_costmap_during_blockage", {})
    cleared = replan.get("native_global_costmap_after_clearance", {})
    baseline = replan.get("baseline", {})
    learned = bridge.get("learned_safety", {})
    mapped = bridge.get("mapped_site_safety", {})
    events = bridge.get("events", {})
    localization = bridge.get("localization", {})
    outcomes = bridge.get("termination_diagnostics", {}).get(
        "episode_outcomes", {}
    )
    observed = bridge.get("route_scoped_speed_evidence", {}).get(
        "maximum_observed_linear_mps_by_segment", {}
    )
    messages = bridge.get("topics", {}).get("message_counts", {})
    legs = mission.get("legs", [])
    layers = obstacle_layers(runtime_profile)

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
        "phase7c_native_detour_baseline_retained": (
            phase7c.get("passed") is True
            and phase7c.get("checks_passed") == phase7c.get("checks_total") == 29
        ),
        "bridge_passed": bridge.get("passed") is True,
        "administration_native_costmap_task_selected": bridge.get("task") == TASK,
        "scoped_administration_mission_passed_both_legs": (
            mission.get("passed") is True
            and mission.get("completed_legs") == mission.get("expected_legs") == 2
            and len([leg for leg in legs if leg.get("execution_status") == "succeeded"])
            == 2
        ),
        "phase7d_control_stack_declared": (
            mission.get("control_stack") == CONTROL_STACK
            and mission.get("phase7d_administration_native_costmap_coupled") is True
        ),
        "episode_never_reset": events.get("episode_reset_gate_detected") is False,
        "mission_completion_reached_bridge": (
            events.get("mission_complete_signal_received") is True
        ),
        "ground_truth_localization_boundary_retained": (
            localization.get("nav2_global_pose_source")
            == "isaac_ground_truth_odom_with_identity_map_to_odom"
            and localization.get("physical_localization_credit") is False
        ),
        "accepted_phase6_primary_loaded": (
            learned.get("checkpoint_sha256") == PHASE6_SHA256
            and learned.get("checkpoint_is_accepted_phase6") is True
        ),
        "accepted_phase3n_fallback_loaded": (
            learned.get("fallback_checkpoint_sha256") == PHASE3N_SHA256
            and learned.get("fallback_checkpoint_is_accepted_phase3n") is True
        ),
        "native_laserscan_observation_without_privileged_policy_state": (
            blocked.get("planner_observation_source")
            == "native_nav2_obstacle_layer_from_live_isaac_laserscan"
            and blocked.get("registered_pointcloud_topic") is None
            and blocked.get("blocker_state_exposed_to_policy") is False
            and replan.get("observation_source")
            == "native_nav2_obstacle_layer_from_live_isaac_laserscan"
            and replan.get("scenario_state_exposed_to_policy") is False
        ),
        "mission_authorized_route_scope_disclosed": (
            blocked.get("route_topology")
            == "map_connected_detour_available_but_outside_mission_authorized_east_hallway"
            and replan.get("topology")
            == "mission_authorized_single_path_with_unapproved_map_detour_available"
        ),
        "blockage_triggered_and_cleared_on_segment_1": (
            blocked.get("blockage_segment_id") == 1
            and blocked.get("triggered") is True
            and blocked.get("cleared") is True
            and int(blocked.get("active_steps", 0)) > 0
            and int(blocked.get("release_requests", 0)) > 0
        ),
        "clear_baseline_corridor_was_free": (
            before.get("available") is True
            and int(before.get("lethal_or_inscribed_samples", -1)) == 0
        ),
        "native_costmap_marked_full_width_barrier": (
            replan.get("nav2_dynamic_costmap_marking_credit") is True
            and marked.get("available") is True
            and int(marked.get("lethal_or_inscribed_samples", 0)) >= 20
            and int(marked.get("maximum_cost", 0)) >= 253
        ),
        "native_costmap_cleared_after_physical_removal": (
            replan.get("explicit_global_costmap_clear_succeeded") is True
            and
            cleared.get("available") is True
            and int(cleared.get("lethal_or_inscribed_samples", -1)) == 0
        ),
        "clear_baseline_path_was_direct_and_authorized": (
            baseline.get("planning_status") == "succeeded"
            and int(baseline.get("path_pose_count", 0)) > 0
            and baseline.get("route_authorization", {}).get("authorized") is True
        ),
        "native_costmap_detour_was_rejected_by_route_authorization": (
            replan.get("blocked_plan_rejected") is True
            and replan.get("rejection_authority")
            == "native_costmap_plan_plus_mission_route_authorization"
            and len(attempts) == 2
            and attempts[0].get("phase") == "barrier_active"
            and attempts[0].get("path_pose_count", 0) > 0
            and attempts[0].get("planning_status") == "succeeded"
            and attempts[0].get("route_authorization", {}).get("authorized") is False
            and float(attempts[0].get("path_length_m", 0.0))
            > float(baseline.get("path_length_m", 0.0)) + 10.0
        ),
        "fresh_global_path_computed_after_clearance": (
            replan.get("fresh_path_computed_after_clearance") is True
            and replan.get("planner_attempt_count") == 2
            and attempts[1].get("phase") == "barrier_cleared"
            and attempts[1].get("planning_status") == "succeeded"
            and int(attempts[1].get("path_pose_count", 0)) > 0
            and attempts[1].get("route_authorization", {}).get("authorized") is True
        ),
        "safe_wait_held_robot_stationary": (
            float(safe_wait.get("maximum_absolute_linear_velocity_mps", 1.0)) <= 0.05
            and float(safe_wait.get("displacement_m", 1.0)) <= 0.08
        ),
        "replanned_leg_executed_successfully": next(
            (
                leg.get("execution_status") == "succeeded"
                and leg.get("blocked_route_replanning", {}).get("passed") is True
                for leg in legs
                if leg.get("route_segment_id") == 1
            ),
            False,
        ),
        "high_speed_resumed_after_fresh_plan": float(observed.get("1", 0.0)) >= 0.72,
        "no_collision_with_blocker_or_static_scene": (
            outcomes.get("dynamic_obstacle_collision", False) is False
            and outcomes.get("static_collision", False) is False
        ),
        "central_drop_clearance_retained": (
            float(mapped.get("minimum_polygon_full_footprint_clearance_m", -1.0)) > 0.0
        ),
        "per_source_obstacle_height_fix_applied_to_both_costmaps": all(
            layer.get("observation_sources") == "crown_scan front_scan"
            and float(layer["crown_scan"].get("max_obstacle_height", 0.0)) == 2.20
            and float(layer["front_scan"].get("max_obstacle_height", 0.0)) == 2.00
            and layer["crown_scan"].get("inf_is_valid") is True
            and layer["front_scan"].get("inf_is_valid") is True
            for layer in layers
        ),
        "new_profile_retains_frozen_profile_behavior": (
            without_source_height_limits(runtime_profile) == frozen_profile
        ),
        "both_live_scan_streams_exercised": (
            int(messages.get("scan", 0)) > 100
            and int(messages.get("front_scan", 0)) > 100
            and int(messages.get("blocked_route_active", 0)) > 50
        ),
        "reverse_remains_disabled": (
            all(float(layer_source) >= 0.0 for layer_source in [
                runtime_profile["controller_server"]["ros__parameters"]["FollowPath"]["min_vel_x"],
                runtime_profile["velocity_smoother"]["ros__parameters"]["min_velocity"][0],
            ])
            and bridge.get("command_constraints", {}).get("reverse_allowed") is False
        ),
        "physical_release_remains_false": (
            mission.get("physical_release") is False
            and bridge.get("physical_release") is False
            and blocked.get("physical_safety_credit") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase7d_native_costmap_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "accepted_administration_native_costmap_safe_wait_replanning"
            if passed
            else "not_accepted"
        ),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "retention": {
            "phase6": "28/28",
            "phase7a": "24/24",
            "phase7b": "26/26",
            "phase7c": "29/29",
        },
        "native_costmap": {
            "before": before,
            "blocked": marked,
            "cleared": cleared,
            "source": blocked.get("planner_observation_source"),
        },
        "safe_wait": safe_wait,
        "fresh_planning": {
            "baseline_path_length_m": baseline.get("path_length_m"),
            "blocked_status": attempts[0].get("planning_status") if attempts else None,
            "blocked_path_length_m": attempts[0].get("path_length_m") if attempts else None,
            "blocked_route_authorized": (
                attempts[0].get("route_authorization", {}).get("authorized")
                if attempts
                else None
            ),
            "fresh_status": attempts[1].get("planning_status") if len(attempts) > 1 else None,
            "fresh_path_pose_count": attempts[1].get("path_pose_count") if len(attempts) > 1 else 0,
        },
        "claim_boundary": (
            "Accepted only as a measured-administration presentation simulation gate: "
            "live Isaac LaserScan data marked and cleared Nav2's native costmap, the "
            "planner produced a long map-connected detour outside the authorized east-"
            "hallway mission envelope, AI-SHA rejected it and waited, and a fresh direct "
            "plan was executed after removal. The live gate is intentionally scoped to "
            "the first two administration legs; the accepted Phase 7B 12-leg mission "
            "retains office-doorway and pivot evidence. This provides no physical "
            "localization, stopping-distance, sim-to-real, protective-sensor, safety-"
            "certification, or deployment credit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PHASE7D_GATE passed={passed} "
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
