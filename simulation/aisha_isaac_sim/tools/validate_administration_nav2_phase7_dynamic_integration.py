#!/usr/bin/env python3
"""Validate Phase 7 live Nav2 stop-wait-resume dynamic-crossing evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7-"
    "DynamicCrossing-Safety-Direct-v0"
)
CONTROL_STACK = "nav2_mapped_doorway_phase7_dynamic_crossing_safety"
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
        default=ROOT / "results/administration_nav2_phase7_dynamic_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=ROOT / "results/administration_nav2_phase7_dynamic_bridge.json",
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
        "--output",
        type=Path,
        default=(
            ROOT / "results/administration_nav2_phase7_dynamic_integration_gate.json"
        ),
    )
    args = parser.parse_args()

    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    static = load_json(args.static_retention)
    dynamic = bridge.get("dynamic_obstacle", {})
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
    requested = bridge.get("route_scoped_speed_evidence", {}).get(
        "maximum_requested_linear_mps_by_segment", {}
    )
    legs = mission.get("legs", [])

    checks = {
        "phase6_static_baseline_retained": (
            static.get("passed") is True
            and static.get("checks_passed") == static.get("checks_total") == 28
        ),
        "bridge_passed": bridge.get("passed") is True,
        "dynamic_task_selected": bridge.get("task") == TASK,
        "mission_passed_all_12_legs": (
            mission.get("passed") is True
            and mission.get("completed_legs") == mission.get("expected_legs") == 12
            and len([leg for leg in legs if leg.get("execution_status") == "succeeded"])
            == 12
        ),
        "dynamic_control_stack_declared": (
            mission.get("control_stack") == CONTROL_STACK
            and mission.get("phase7_dynamic_crossing_safety_coupled") is True
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
        "pedestrian_proxy_is_sensed_without_privileged_policy_state": (
            dynamic.get("enabled") is True
            and dynamic.get("stylized_proxy_not_human_model") is True
            and dynamic.get("pedestrian_state_exposed_to_policy") is False
        ),
        "crossing_triggered_and_completed_on_segment_1": (
            dynamic.get("crossing_segment_id") == 1
            and dynamic.get("triggered") is True
            and dynamic.get("crossing_completed") is True
            and int(dynamic.get("crossing_steps", 0)) > 0
        ),
        "high_speed_approach_was_exercised": (
            float(requested.get("1", 0.0)) >= 0.799
            and float(dynamic.get("maximum_pre_trigger_forward_speed_mps", 0.0))
            >= 0.72
        ),
        "front_scan_detected_crossing": (
            0.12
            <= float(dynamic.get("minimum_central_front_range_m", 0.0))
            <= 1.50
        ),
        "front_protective_latch_engaged": (
            int(dynamic.get("front_stop_latched_steps", 0)) > 0
        ),
        "learned_safety_had_encounter_authority": (
            int(dynamic.get("learned_authority_steps_during_encounter", 0)) > 0
            and int(dynamic.get("sensor_scoped_phase3n_handoff_steps", 0)) > 0
            and int(dynamic.get("learned_brake_steps_during_encounter", 0)) > 0
        ),
        "controlled_stop_observed": (
            dynamic.get("controlled_stop_observed") is True
            and float(dynamic.get("minimum_encounter_forward_speed_mps", 1.0))
            <= 0.05
        ),
        "post_crossing_recovery_observed": (
            dynamic.get("post_crossing_recovery_observed") is True
            and float(
                dynamic.get("maximum_post_crossing_recovery_speed_mps", 0.0)
            )
            >= 0.30
        ),
        "pedestrian_contact_envelope_not_entered": (
            float(
                dynamic.get("minimum_robot_pedestrian_centre_distance_m", 0.0)
            )
            >= 1.00
            and outcomes.get("dynamic_obstacle_collision", False) is False
        ),
        "return_high_speed_leg_retained": float(observed.get("5", 0.0)) >= 0.72,
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
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase7_dynamic_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "accepted_simulation_stop_wait_resume_dynamic_crossing"
            if passed
            else "not_accepted"
        ),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "mission": {
            "report": str(args.mission.resolve()),
            "legs_completed": mission.get("completed_legs"),
            "elapsed_wall_s": mission.get("elapsed_wall_s"),
        },
        "bridge": str(args.bridge.resolve()),
        "static_retention": str(args.static_retention.resolve()),
        "dynamic_obstacle": dynamic,
        "mapped_guard": mapped,
        "learned_safety": learned,
        "claim_boundary": {
            "supported": (
                "One deterministic sensed pedestrian crossing with live Nav2, "
                "accepted learned safety authority, a separate protective stop, "
                "wait and resume "
                "inside the measured-presentation Isaac scene"
            ),
            "blocked_route_replanning_supported": False,
            "human_behavior_or_biomechanics_model": False,
            "physical_localization_credit": False,
            "physical_safety_credit": False,
            "physical_release": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_NAV2_PHASE7_DYNAMIC passed={passed} "
        f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
