#!/usr/bin/env python3
"""Validate Phase 7B temporary blocked-route safe-wait replanning evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7B-"
    "BlockedRoute-Replanning-Safety-Direct-v0"
)
CONTROL_STACK = "nav2_mapped_doorway_phase7b_blocked_route_replanning_safety"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=(
            ROOT / "results/administration_nav2_phase7b_blocked_route_mission.json"
        ),
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=(
            ROOT / "results/administration_nav2_phase7b_blocked_route_bridge.json"
        ),
    )
    parser.add_argument(
        "--static-retention",
        type=Path,
        default=(
            ROOT
            / "results/administration_nav2_phase6_high_speed_integration_gate.json"
        ),
    )
    parser.add_argument(
        "--crossing-retention",
        type=Path,
        default=(
            ROOT / "results/administration_nav2_phase7_dynamic_integration_gate.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/administration_nav2_phase7b_blocked_route_integration_gate.json"
        ),
    )
    args = parser.parse_args()

    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    static = load_json(args.static_retention)
    crossing = load_json(args.crossing_retention)
    blocked = bridge.get("blocked_route", {})
    replan = mission.get("blocked_route_replanning") or {}
    attempts = replan.get("attempts", [])
    safe_wait = replan.get("safe_wait", {})
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
    legs = mission.get("legs", [])

    checks = {
        "phase6_static_baseline_retained": (
            static.get("passed") is True
            and static.get("checks_passed") == static.get("checks_total") == 28
        ),
        "phase7a_crossing_baseline_retained": (
            crossing.get("passed") is True
            and crossing.get("checks_passed") == crossing.get("checks_total") == 24
        ),
        "bridge_passed": bridge.get("passed") is True,
        "blocked_route_task_selected": bridge.get("task") == TASK,
        "mission_passed_all_12_legs": (
            mission.get("passed") is True
            and mission.get("completed_legs") == mission.get("expected_legs") == 12
            and len(
                [leg for leg in legs if leg.get("execution_status") == "succeeded"]
            )
            == 12
        ),
        "blocked_route_control_stack_declared": (
            mission.get("control_stack") == CONTROL_STACK
            and mission.get("phase7b_blocked_route_replanning_coupled") is True
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
        "blocker_sensed_without_privileged_policy_state": (
            blocked.get("enabled") is True
            and blocked.get("planner_observation_source")
            == "registered_front_lidar_supervisory_path_validator"
            and blocked.get("registered_pointcloud_topic")
            == "/aisha/phase7b/front_points"
            and blocked.get("blocker_state_exposed_to_policy") is False
            and replan.get("observation_source")
            == "registered_front_lidar_supervisory_path_validator"
            and replan.get("nav2_dynamic_costmap_marking_credit") is False
        ),
        "single_path_safe_wait_scope_disclosed": (
            blocked.get("route_topology")
            == "single_path_safe_wait_required_no_detour_available"
            and replan.get("topology")
            == "single_path_safe_wait_no_detour_available"
        ),
        "blockage_triggered_on_segment_1": (
            blocked.get("blockage_segment_id") == 1
            and blocked.get("triggered") is True
            and int(blocked.get("active_steps", 0)) > 0
        ),
        "blockage_was_cleared_after_release": (
            blocked.get("cleared") is True
            and int(blocked.get("release_requests", 0)) > 0
            and int(blocked.get("clear_step", 0))
            > int(blocked.get("trigger_step", 0))
        ),
        "mission_observed_active_and_clear_states": (
            replan.get("blockage_active_seen") is True
            and replan.get("blockage_cleared_seen") is True
        ),
        "active_blocked_plan_was_rejected": (
            replan.get("blocked_plan_rejected") is True
            and replan.get("rejection_authority")
            == "registered_lidar_supervisory_path_validator"
            and len(attempts) == 2
            and attempts[0].get("phase") == "barrier_active"
            and attempts[0].get("path_pose_count") == 0
            and attempts[0].get("nav2_candidate_planning_status") == "succeeded"
            and attempts[0].get("planning_status")
            == "rejected_by_registered_lidar_path_validator"
            and attempts[0]
            .get("registered_lidar_validation", {})
            .get("candidate_rejected")
            is True
        ),
        "fresh_global_path_computed_after_clearance": (
            replan.get("fresh_path_computed_after_clearance") is True
            and replan.get("planner_attempt_count") == 2
            and attempts[1].get("phase") == "barrier_cleared"
            and attempts[1].get("planning_status") == "succeeded"
            and int(attempts[1].get("path_pose_count", 0)) > 0
            and attempts[1]
            .get("registered_lidar_validation", {})
            .get("candidate_rejected")
            is False
        ),
        "safe_wait_held_robot_stationary": (
            float(safe_wait.get("maximum_absolute_linear_velocity_mps", 1.0))
            <= 0.05
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
        "high_speed_resumed_after_replan": float(observed.get("1", 0.0)) >= 0.72,
        "no_collision_with_blocker_or_static_scene": (
            outcomes.get("dynamic_obstacle_collision", False) is False
            and outcomes.get("static_collision", False) is False
        ),
        "doorway_limit_retained": (
            float(mapped.get("maximum_abs_speed_in_doorway_mps", 1.0)) <= 0.10
        ),
        "central_drop_clearance_retained": (
            float(mapped.get("minimum_polygon_full_footprint_clearance_m", -1.0))
            > 0.0
        ),
        "both_office_pivots_retained": all(
            next(
                (
                    leg.get("post_visit_pivot", {}).get("passed") is True
                    for leg in legs
                    if leg.get("waypoint_id") == office
                ),
                False,
            )
            for office in ("vice_principal", "principal")
        ),
        "physical_release_remains_false": (
            mission.get("physical_release") is False
            and bridge.get("physical_release") is False
            and blocked.get("physical_safety_credit") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase7b_blocked_route_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "accepted_single_path_safe_wait_replanning" if passed else "not_accepted"
        ),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "mission": {
            "report": str(args.mission.resolve()),
            "legs_completed": mission.get("completed_legs"),
            "elapsed_wall_s": mission.get("elapsed_wall_s"),
            "replanning": replan,
        },
        "bridge": str(args.bridge.resolve()),
        "static_retention": str(args.static_retention.resolve()),
        "crossing_retention": str(args.crossing_retention.resolve()),
        "blocked_route": blocked,
        "mapped_guard": mapped,
        "learned_safety": learned,
        "claim_boundary": {
            "supported": (
                "One temporary full-width single-path hallway blockage sensed "
                "through registered LiDAR path validation, rejection of the "
                "unsafe Nav2 candidate, safe wait, fresh validated global path "
                "after clearance and full mission completion"
            ),
            "spatial_detour_supported": False,
            "reason_no_detour": (
                "The route-scoped east office hallway has no mapped alternate corridor"
            ),
            "persistent_blockage_navigation_supported": False,
            "physical_localization_credit": False,
            "physical_safety_credit": False,
            "physical_release": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_NAV2_PHASE7B_BLOCKED_ROUTE passed={passed} "
        f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
