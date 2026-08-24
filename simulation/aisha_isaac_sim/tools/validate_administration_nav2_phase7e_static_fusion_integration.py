#!/usr/bin/env python3
"""Validate static-map/live-LiDAR fusion across the full administration mission."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from build_administration_static_fused_nav2_profile import fuse_observation_sources


ROOT = Path(__file__).resolve().parents[1]
TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7E-"
    "StaticFusion-FullOffice-Safety-Direct-v0"
)
CONTROL_STACK = "nav2_mapped_doorway_phase7e_static_fusion_full_office_safety"
PHASE6_SHA256 = (
    "e49767507925548aa0086c38e764c43037f25734943b2c5712cb58eecb0b6318"
)
PHASE3N_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)
EXPECTED_WAYPOINTS = [
    "east_atrium_exit",
    "vice_principal_turn",
    "vice_principal_approach",
    "vice_principal",
    "vice_principal_depart",
    "hallway_return",
    "principal_turn",
    "principal_approach",
    "principal",
    "principal_depart",
    "atrium_return",
    "home_return",
]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def obstacle_layers(config: dict) -> list[dict]:
    return [
        config[name][name]["ros__parameters"]["obstacle_layer"]
        for name in ("local_costmap", "global_costmap")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_fusion_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_fusion_bridge.json",
    )
    parser.add_argument(
        "--fusion",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7e_static_scan_fusion.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/administration_nav2_phase7e_static_fusion_integration_gate.json"
        ),
    )
    args = parser.parse_args()

    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    fusion = load_json(args.fusion)
    phase6 = load_json(
        ROOT / "results/administration_nav2_phase6_high_speed_integration_gate.json"
    )
    phase7a = load_json(
        ROOT / "results/administration_nav2_phase7_dynamic_integration_gate.json"
    )
    phase7b = load_json(
        ROOT / "results/administration_nav2_phase7b_blocked_route_integration_gate.json"
    )
    phase7c = load_json(
        ROOT / "results/phase7c_native_costmap_detour_integration_gate.json"
    )
    phase7d = load_json(
        ROOT / "results/administration_nav2_phase7d_native_costmap_integration_gate.json"
    )
    native_profile = yaml.safe_load(
        (ROOT / "config/nav2_administration_native_costmap_params.yaml").read_text(
            encoding="utf-8"
        )
    )
    fused_profile = fuse_observation_sources(copy.deepcopy(native_profile))
    layers = obstacle_layers(fused_profile)
    legs = mission.get("legs", [])
    replan = mission.get("blocked_route_replanning") or {}
    attempts = replan.get("attempts", [])
    safe_wait = replan.get("safe_wait", {})
    before = replan.get("costmap_before_activation", {})
    marked = replan.get("native_global_costmap_during_blockage", {})
    cleared = replan.get("native_global_costmap_after_clearance", {})
    baseline = replan.get("baseline", {})
    blocked = bridge.get("blocked_route", {})
    learned = bridge.get("learned_safety", {})
    mapped = bridge.get("mapped_site_safety", {})
    events = bridge.get("events", {})
    outcomes = bridge.get("termination_diagnostics", {}).get("episode_outcomes", {})
    observed = bridge.get("route_scoped_speed_evidence", {}).get(
        "maximum_observed_linear_mps_by_segment", {}
    )
    messages = bridge.get("topics", {}).get("message_counts", {})
    crown_stats = fusion.get("statistics", {}).get("crown", {})
    front_stats = fusion.get("statistics", {}).get("front", {})
    architecture = fusion.get("architecture", {})

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
        "phase7d_native_costmap_baseline_retained": (
            phase7d.get("passed") is True
            and phase7d.get("checks_passed") == phase7d.get("checks_total") == 32
        ),
        "bridge_passed": bridge.get("passed") is True,
        "phase7e_task_selected": bridge.get("task") == TASK,
        "full_12_leg_mission_passed": (
            mission.get("passed") is True
            and mission.get("completed_legs") == mission.get("expected_legs") == 12
            and [leg.get("waypoint_id") for leg in legs] == EXPECTED_WAYPOINTS
            and all(leg.get("execution_status") == "succeeded" for leg in legs)
        ),
        "phase7e_control_stack_declared": (
            mission.get("control_stack") == CONTROL_STACK
            and mission.get("phase7e_administration_static_fusion_coupled") is True
            and mission.get("phase7d_administration_native_costmap_coupled") is False
        ),
        "no_planning_failure_diagnostics_triggered": all(
            "planning_failure_diagnostics" not in leg for leg in legs
        ),
        "both_office_visits_and_departures_completed": all(
            next(
                (
                    leg.get("execution_status") == "succeeded"
                    for leg in legs
                    if leg.get("waypoint_id") == waypoint
                ),
                False,
            )
            for waypoint in (
                "vice_principal",
                "vice_principal_depart",
                "principal",
                "principal_depart",
            )
        ),
        "both_in_office_pivots_completed": all(
            next(
                (
                    leg.get("post_visit_pivot", {}).get("passed") is True
                    for leg in legs
                    if leg.get("waypoint_id") == waypoint
                ),
                False,
            )
            for waypoint in ("vice_principal", "principal")
        ),
        "both_departure_headings_aligned_in_pivot_zones": all(
            next(
                (
                    leg.get("post_visit_departure_alignment", {}).get("passed")
                    is True
                    and leg.get("post_visit_departure_alignment", {}).get(
                        "rotation_location"
                    )
                    == "mapped_in_office_pivot_clearance_zone"
                    and abs(
                        float(
                            leg.get("post_visit_departure_alignment", {}).get(
                                "final_error_deg", 180.0
                            )
                        )
                    )
                    <= 2.0
                    for leg in legs
                    if leg.get("waypoint_id") == waypoint
                ),
                False,
            )
            for waypoint in ("vice_principal", "principal")
        ),
        "episode_never_reset": events.get("episode_reset_gate_detected") is False,
        "mission_completion_reached_bridge": (
            events.get("mission_complete_signal_received") is True
        ),
        "accepted_phase6_primary_loaded": (
            learned.get("checkpoint_sha256") == PHASE6_SHA256
            and learned.get("checkpoint_is_accepted_phase6") is True
        ),
        "accepted_phase3n_fallback_loaded": (
            learned.get("fallback_checkpoint_sha256") == PHASE3N_SHA256
            and learned.get("fallback_checkpoint_is_accepted_phase3n") is True
        ),
        "native_laserscan_planner_observation_retained": (
            blocked.get("planner_observation_source")
            == "native_nav2_obstacle_layer_from_live_isaac_laserscan"
            and blocked.get("registered_pointcloud_topic") is None
            and blocked.get("blocker_state_exposed_to_policy") is False
        ),
        "raw_scans_are_clearing_only": all(
            layer.get("crown_clear", {}).get("topic") == "/scan"
            and layer.get("crown_clear", {}).get("marking") is False
            and layer.get("crown_clear", {}).get("clearing") is True
            and layer.get("front_clear", {}).get("topic") == "/front_scan"
            and layer.get("front_clear", {}).get("marking") is False
            and layer.get("front_clear", {}).get("clearing") is True
            for layer in layers
        ),
        "filtered_scans_are_dynamic_marking_only": all(
            layer.get("observation_sources")
            == "crown_clear front_clear crown_dynamic front_dynamic"
            and layer.get("crown_dynamic", {}).get("topic")
            == "/aisha/static_fused/scan_dynamic"
            and layer.get("crown_dynamic", {}).get("marking") is True
            and layer.get("crown_dynamic", {}).get("clearing") is False
            and layer.get("front_dynamic", {}).get("topic")
            == "/aisha/static_fused/front_scan_dynamic"
            and layer.get("front_dynamic", {}).get("marking") is True
            and layer.get("front_dynamic", {}).get("clearing") is False
            for layer in layers
        ),
        "source_height_and_no_return_limits_retained": all(
            float(layer["crown_clear"].get("max_obstacle_height", 0.0)) == 2.20
            and float(layer["front_clear"].get("max_obstacle_height", 0.0)) == 2.00
            and layer["crown_clear"].get("inf_is_valid") is True
            and layer["front_clear"].get("inf_is_valid") is True
            and float(layer["crown_dynamic"].get("max_obstacle_height", 0.0))
            == 2.20
            and float(layer["front_dynamic"].get("max_obstacle_height", 0.0))
            == 2.00
            for layer in layers
        ),
        "physical_footprint_and_padding_unchanged": all(
            fused_profile[name][name]["ros__parameters"]["footprint"]
            == native_profile[name][name]["ros__parameters"]["footprint"]
            and fused_profile[name][name]["ros__parameters"]["footprint_padding"]
            == native_profile[name][name]["ros__parameters"]["footprint_padding"]
            == 0.03
            for name in ("local_costmap", "global_costmap")
        ),
        "filter_received_map_without_tf_failures": (
            int(fusion.get("statistics", {}).get("map_messages", 0)) >= 1
            and int(crown_stats.get("transform_failures", -1)) == 0
            and int(front_stats.get("transform_failures", -1)) == 0
        ),
        "static_returns_were_masked_from_duplicate_marking": (
            int(crown_stats.get("static_returns_masked", 0)) > 1000
            and int(front_stats.get("static_returns_masked", 0)) > 1000
            and architecture.get("masked_returns_remain_represented_by_static_layer")
            is True
        ),
        "mapped_free_returns_were_preserved": (
            int(crown_stats.get("mapped_free_returns_preserved", 0)) > 0
            and int(front_stats.get("mapped_free_returns_preserved", 0)) > 0
            and architecture.get("mapped_free_obstacle_returns_preserved") is True
        ),
        "filter_published_both_scan_streams": (
            int(crown_stats.get("published_messages", 0)) > 100
            and int(front_stats.get("published_messages", 0)) > 100
        ),
        "clear_baseline_corridor_was_free": (
            before.get("available") is True
            and int(before.get("lethal_or_inscribed_samples", -1)) == 0
        ),
        "native_costmap_still_marked_full_width_dynamic_barrier": (
            replan.get("nav2_dynamic_costmap_marking_credit") is True
            and marked.get("available") is True
            and int(marked.get("lethal_or_inscribed_samples", 0)) >= 20
            and int(marked.get("maximum_cost", 0)) >= 253
        ),
        "native_costmap_cleared_after_physical_removal": (
            replan.get("explicit_global_costmap_clear_succeeded") is True
            and cleared.get("available") is True
            and int(cleared.get("lethal_or_inscribed_samples", -1)) == 0
        ),
        "clear_baseline_path_was_direct_and_authorized": (
            baseline.get("planning_status") == "succeeded"
            and baseline.get("route_authorization", {}).get("authorized") is True
        ),
        "blocked_detour_was_rejected_by_route_authorization": (
            replan.get("blocked_plan_rejected") is True
            and len(attempts) == 2
            and attempts[0].get("phase") == "barrier_active"
            and attempts[0].get("route_authorization", {}).get("authorized") is False
            and float(attempts[0].get("path_length_m", 0.0))
            > float(baseline.get("path_length_m", 0.0)) + 10.0
        ),
        "fresh_direct_path_computed_after_clearance": (
            replan.get("fresh_path_computed_after_clearance") is True
            and attempts[1].get("phase") == "barrier_cleared"
            and attempts[1].get("route_authorization", {}).get("authorized") is True
        ),
        "safe_wait_held_robot_stationary": (
            float(safe_wait.get("maximum_absolute_linear_velocity_mps", 1.0)) <= 0.05
            and float(safe_wait.get("displacement_m", 1.0)) <= 0.08
        ),
        "high_speed_hallway_tiers_retained": (
            float(observed.get("1", 0.0)) >= 0.72
            and float(observed.get("5", 0.0)) >= 0.72
        ),
        "mapped_doorway_limits_retained": (
            mapped.get("doorway_entries", {}).get("vice_principal") == 2
            and mapped.get("doorway_entries", {}).get("principal") == 2
            and float(mapped.get("maximum_abs_speed_in_doorway_mps", 1.0)) <= 0.10
            and float(mapped.get("maximum_abs_tangent_offset_in_doorway_m", 1.0))
            <= 0.03
        ),
        "central_drop_clearance_retained": (
            float(mapped.get("minimum_polygon_full_footprint_clearance_m", -1.0))
            > 0.0
        ),
        "no_collision_with_blocker_or_static_scene": (
            outcomes.get("dynamic_obstacle_collision", False) is False
            and outcomes.get("static_collision", False) is False
        ),
        "both_live_scan_streams_exercised": (
            int(messages.get("scan", 0)) > 100
            and int(messages.get("front_scan", 0)) > 100
            and int(messages.get("blocked_route_active", 0)) > 50
        ),
        "reverse_remains_disabled": (
            bridge.get("command_constraints", {}).get("reverse_allowed") is False
            and bridge.get("events", {}).get("rejected_reverse_commands") == 0
        ),
        "physical_release_remains_false": (
            mission.get("physical_release") is False
            and bridge.get("physical_release") is False
            and fusion.get("physical_release") is False
            and blocked.get("physical_safety_credit") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase7e_static_fusion_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "accepted_full_office_native_costmap_static_scan_fusion"
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
            "phase7d": "32/32",
        },
        "mission": {
            "completed_legs": mission.get("completed_legs"),
            "expected_legs": mission.get("expected_legs"),
            "waypoints": [leg.get("waypoint_id") for leg in legs],
        },
        "fusion": {
            "architecture": architecture,
            "statistics": fusion.get("statistics"),
        },
        "native_costmap": {
            "before": before,
            "blocked": marked,
            "cleared": cleared,
        },
        "safe_wait": safe_wait,
        "fresh_planning": {
            "baseline_path_length_m": baseline.get("path_length_m"),
            "blocked_path_length_m": (
                attempts[0].get("path_length_m") if attempts else None
            ),
            "fresh_path_length_m": (
                attempts[1].get("path_length_m") if len(attempts) > 1 else None
            ),
        },
        "claim_boundary": (
            "Accepted only as a measured-administration presentation-simulation "
            "gate. Known static LiDAR endpoints are represented by the static map "
            "instead of being inflated twice; raw rays still clear, mapped-free "
            "returns still mark native Nav2 obstacles, and the unchanged padded "
            "AI-SHA footprint completed all 12 legs. This provides no physical "
            "localization, stopping-distance, sim-to-real, certification, or "
            "deployment credit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_PHASE7E_GATE passed={passed} "
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
