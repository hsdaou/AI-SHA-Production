#!/usr/bin/env python3
"""Validate the paired Nav2/Phase 3N run against the frozen-policy evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_PHASE3N_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=PACKAGE_ROOT / "results" / "administration_nav2_phase3n_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=PACKAGE_ROOT / "results" / "administration_nav2_phase3n_bridge.json",
    )
    parser.add_argument(
        "--phase3n-acceptance",
        type=Path,
        default=PACKAGE_ROOT / "results" / "phase3n_dynamic_safety_acceptance.json",
    )
    parser.add_argument(
        "--frozen-route-report",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results"
            / "phase3n_administration_final_omniverse_report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results"
            / "administration_nav2_phase3n_integration_gate.json"
        ),
    )
    args = parser.parse_args()

    mission = load(args.mission)
    bridge = load(args.bridge)
    acceptance = load(args.phase3n_acceptance)
    frozen_route = load(args.frozen_route_report)

    successful_legs = [
        leg for leg in mission.get("legs", []) if leg.get("execution_status") == "succeeded"
    ]
    office_pivots = {
        leg["waypoint_id"]: leg.get("post_visit_pivot", {})
        for leg in mission.get("legs", [])
        if leg.get("waypoint_id") in {"vice_principal", "principal"}
    }
    bridge_events = bridge.get("events", {})
    learned_safety = bridge.get("learned_safety", {})
    topic_counts = bridge.get("topics", {}).get("message_counts", {})

    checks = {
        "nav2_mission_passed": bool(mission.get("passed")),
        "all_12_nav2_legs_succeeded": (
            mission.get("expected_legs") == 12
            and mission.get("completed_legs") == 12
            and len(successful_legs) == 12
        ),
        "both_office_pivots_succeeded": (
            set(office_pivots) == {"vice_principal", "principal"}
            and all(bool(item.get("passed")) for item in office_pivots.values())
        ),
        "paired_bridge_passed": bool(bridge.get("passed")),
        "mission_completion_reached_bridge": bool(
            bridge_events.get("mission_complete_signal_received")
        ),
        "bridge_episode_never_reset": not bool(
            bridge_events.get("episode_reset_gate_detected")
        ),
        "all_ros_channels_exercised": bool(topic_counts)
        and all(int(count) > 0 for count in topic_counts.values()),
        "accepted_phase3n_checkpoint_loaded": (
            learned_safety.get("enabled") is True
            and learned_safety.get("checkpoint_is_accepted_phase3n") is True
            and learned_safety.get("checkpoint_sha256") == ACCEPTED_PHASE3N_SHA256
        ),
        "learned_safety_received_nav2_base_commands": (
            learned_safety.get("base_command_source") == "nav2_cmd_vel"
        ),
        "formal_phase3n_dynamic_acceptance_passed": bool(
            acceptance.get("decision", {}).get(
                "full_phase3_simulation_acceptance_passed"
            )
        ),
        "formal_phase3n_checkpoint_matches": (
            acceptance.get("architecture", {}).get(
                "selected_phase3n_checkpoint_sha256"
            )
            == ACCEPTED_PHASE3N_SHA256
        ),
        "frozen_phase3m_stack_completed_administration_route": (
            frozen_route.get("outcome") == "success"
            and frozen_route.get("route_control") == "policy-only"
            and frozen_route.get("waypoints_completed") == 12
            and frozen_route.get("route_segment_count") == 12
            and frozen_route.get("checkpoint_sha256") == ACCEPTED_PHASE3N_SHA256
        ),
        "no_root_transform_animation_in_frozen_route": (
            frozen_route.get("root_transform_animation") is False
        ),
        "physical_release_remains_false": (
            bridge.get("physical_release") is False
            and mission.get("physical_release") is False
            and acceptance.get("decision", {}).get("physical_robot_release") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase3n_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
        "live_nav2_phase3n_run": {
            "mission_report": str(args.mission),
            "bridge_report": str(args.bridge),
            "legs_completed": mission.get("completed_legs"),
            "elapsed_wall_s": mission.get("elapsed_wall_s"),
            "bridge_steps": bridge.get("steps_completed"),
            "safety_authority_steps": learned_safety.get("authority_steps"),
            "safety_brake_steps": learned_safety.get("brake_steps"),
            "maximum_brake_fraction": learned_safety.get(
                "maximum_brake_fraction"
            ),
            "minimum_360_clearance_m": learned_safety.get(
                "minimum_360_clearance_m"
            ),
        },
        "frozen_learned_navigation_evidence": {
            "report": str(args.frozen_route_report),
            "architecture": frozen_route.get("policy_architecture"),
            "waypoints_completed": frozen_route.get("waypoints_completed"),
            "control_disclosure": frozen_route.get("control_disclosure"),
        },
        "dynamic_training_evidence": {
            "report": str(args.phase3n_acceptance),
            "training_transitions": acceptance.get("training", {}).get(
                "simulated_policy_transitions"
            ),
            "randomized_segment_gate": acceptance.get(
                "randomized_segment_gate"
            ),
            "static_regression_gate": acceptance.get("static_regression_gate"),
            "live_administration_dynamic_gate": acceptance.get(
                "live_administration_dynamic_gate"
            ),
        },
        "architecture_boundary": {
            "verified_live_path": (
                "Nav2 global/local planning -> Phase 3N learned 360-degree "
                "brake arbitration -> articulated wheel physics"
            ),
            "verified_frozen_policy_path": (
                "frozen learned route actor -> frozen Phase 3M recovery, clearance, "
                "protective-stop and pivot stack -> Phase 3N -> articulated wheel physics"
            ),
            "not_claimed": (
                "Nav2 global paths and the frozen Phase 3M local navigator have not been "
                "combined into one controller path; running two independent local planners "
                "in series is intentionally avoided until an explicit arbitration design is "
                "selected."
            ),
        },
        "map_status": "provisional_plan_and_walkthrough_derived_not_measured",
        "physical_release": False,
        "remaining_gates": [
            "as-built iPhone LiDAR/RoomPlan capture and critical manual measurements",
            "measured geometry and occupancy-map rebuild",
            "measured-scene regression of this exact integration gate",
            "hardware emergency-stop and supervised sim-to-real commissioning",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_NAV2_PHASE3N_INTEGRATION passed={passed} "
        f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
