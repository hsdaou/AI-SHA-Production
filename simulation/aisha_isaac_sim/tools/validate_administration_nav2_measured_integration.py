#!/usr/bin/env python3
"""Validate the measured-presentation Nav2, mapped guard, and learned safety run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_PHASE3N_SHA256 = (
    "11016d3e79a23f966597922ec165e73d0de24a509bfebcfdd53761d7a7f0343b"
)
ACCEPTED_MEASURED_ROUTE_SHA256 = (
    "6bf032350d36539d6e18651d8c3344c17951ff33dfade17a7910a694933d2d5f"
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=PACKAGE_ROOT / "results" / "administration_nav2_measured_mission.json",
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=PACKAGE_ROOT / "results" / "administration_nav2_measured_bridge.json",
    )
    parser.add_argument(
        "--map-report",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results"
            / "administration_measured_presentation_1cm_map_report.json"
        ),
    )
    parser.add_argument(
        "--build-report",
        type=Path,
        default=PACKAGE_ROOT / "results" / "administration_build_report.json",
    )
    parser.add_argument(
        "--phase3n-acceptance",
        type=Path,
        default=PACKAGE_ROOT / "results" / "phase3n_dynamic_safety_acceptance.json",
    )
    parser.add_argument(
        "--measured-route-acceptance",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results"
            / "measured_tight_door_model2350_extended_handoff_acceptance_seed10640.json"
        ),
    )
    parser.add_argument(
        "--measured-presentation-validation",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results"
            / "measured_administration_final_presentation_validation.json"
        ),
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "config"
            / "measured_administration_presentation_2026-08-23.yaml"
        ),
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=PACKAGE_ROOT / "scenes" / "administration.usd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results"
            / "administration_nav2_measured_integration_gate.json"
        ),
    )
    args = parser.parse_args()

    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    map_report = load_json(args.map_report)
    build = load_json(args.build_report)
    phase3n = load_json(args.phase3n_acceptance)
    measured_route = load_json(args.measured_route_acceptance)
    measured_presentation = load_json(args.measured_presentation_validation)
    overlay = yaml.safe_load(args.overlay.read_text(encoding="utf-8"))

    successful_legs = [
        leg for leg in mission.get("legs", []) if leg.get("execution_status") == "succeeded"
    ]
    pivots = {
        leg.get("waypoint_id"): leg.get("post_visit_pivot", {})
        for leg in mission.get("legs", [])
        if leg.get("waypoint_id") in {"vice_principal", "principal"}
    }
    events = bridge.get("events", {})
    learned = bridge.get("learned_safety", {})
    mapped = bridge.get("mapped_site_safety", {})
    drop_routing = bridge.get("central_drop_safety_routing", {})
    termination_envelope = bridge.get("measured_nav2_termination_envelope", {})
    topics = bridge.get("topics", {}).get("message_counts", {})
    door_entries = mapped.get("doorway_entries", {})
    formal_route_checkpoint = measured_route.get("checkpoint", {}).get("sha256")

    checks = {
        "measured_nav2_mission_passed": mission.get("passed") is True,
        "measured_site_profile_selected": (
            mission.get("site_profile") == "measured_presentation"
            and mission.get("map_status") == "measured_site_presentation_candidate"
        ),
        "all_12_legs_succeeded": (
            mission.get("expected_legs") == 12
            and mission.get("completed_legs") == 12
            and len(successful_legs) == 12
        ),
        "both_office_pivots_succeeded": (
            set(pivots) == {"vice_principal", "principal"}
            and all(item.get("passed") is True for item in pivots.values())
        ),
        "paired_bridge_passed": bridge.get("passed") is True,
        "mission_completion_reached_bridge": events.get("mission_complete_signal_received") is True,
        "bridge_episode_never_reset": events.get("episode_reset_gate_detected") is False,
        "all_ros_channels_exercised": bool(topics)
        and all(int(count) > 0 for count in topics.values()),
        "accepted_learned_360_checkpoint_loaded": (
            learned.get("enabled") is True
            and learned.get("checkpoint_is_accepted_phase3n") is True
            and learned.get("checkpoint_sha256") == ACCEPTED_PHASE3N_SHA256
        ),
        "formal_dynamic_safety_gate_passed": phase3n.get("decision", {}).get(
            "full_phase3_simulation_acceptance_passed"
        )
        is True,
        "mapped_site_guard_enabled": mapped.get("enabled") is True,
        "central_drop_safety_authority_is_explicit": (
            drop_routing.get("learned_crown_scan_excludes_navigation_barrier") is True
            and drop_routing.get("occupancy_map_keeps_navigation_barrier") is True
            and drop_routing.get("physics_collider_kept") is True
            and drop_routing.get("mapped_full_footprint_guard_required") is True
        ),
        "measured_nav2_numerical_envelope_disclosed": (
            termination_envelope.get("lidar_collision_margin_m") == 0.015
            and termination_envelope.get("nav2_footprint_padding_m") == 0.03
            and termination_envelope.get("physical_geometry_changed") is False
            and termination_envelope.get("physical_safety_credit") is False
        ),
        "mapped_overlay_hash_matches": mapped.get("overlay_sha256")
        == sha256_file(args.overlay),
        "both_measured_doorways_exercised": (
            int(door_entries.get("vice_principal", 0)) > 0
            and int(door_entries.get("principal", 0)) > 0
        ),
        "mapped_doorway_alignment_exercised": int(
            mapped.get("doorway_alignment_steps", 0)
        )
        > 0,
        "doorway_speed_gate_passed": float(
            mapped.get("maximum_abs_speed_in_doorway_mps", float("inf"))
        )
        <= float(mapped.get("maximum_doorway_speed_mps", 0.0)) + 1.0e-6,
        "central_polygon_full_footprint_clearance_positive": float(
            mapped.get("minimum_polygon_full_footprint_clearance_m", -1.0)
        )
        > 0.0,
        "measured_map_passed": map_report.get("passed") is True,
        "measured_map_is_current_scene": map_report.get("source", {}).get(
            "scene_sha256"
        )
        == sha256_file(args.scene),
        "measured_map_resolution_is_1cm": map_report.get("map", {}).get(
            "resolution_m_per_pixel"
        )
        == 0.01,
        "central_drop_mapped_no_go": (
            map_report.get("central_atrium_drop", {}).get("mapped_as_no_go") is True
            and overlay["plan_geometry"]["atrium"]["central_polygon"]["step_down_m"]
            == 0.20
        ),
        "reported_door_geometry_applied": (
            build.get("doors", {}).get("vice_principal", {}).get("clear_width_m")
            == 0.85
            and build.get("doors", {}).get("principal", {}).get("clear_width_m")
            == 0.90
        ),
        "vp_locked_assumption_disclosed": build.get("capture_limitations", {})
        .get("vice_principal_office_interior", {})
        .get("status")
        == "not_captured_locked_during_site_visit",
        "separate_measured_route_policy_accepted": (
            measured_route.get("acceptance_gate", {}).get("passed") is True
            and formal_route_checkpoint == ACCEPTED_MEASURED_ROUTE_SHA256
        ),
        "measured_omniverse_presentation_still_valid": measured_presentation.get(
            "passed"
        )
        is True,
        "physical_release_remains_false": (
            bridge.get("physical_release") is False
            and mission.get("physical_release") is False
            and overlay.get("physical_release") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_measured_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "live_stack": {
            "architecture": (
                "Nav2 global planner + DWB local controller -> deterministic mapped "
                "doorway/polygon guard -> accepted learned Phase 3N 360 brake -> "
                "articulated wheel physics"
            ),
            "mission_report": str(args.mission.resolve()),
            "bridge_report": str(args.bridge.resolve()),
            "legs_completed": mission.get("completed_legs"),
            "elapsed_wall_s": mission.get("elapsed_wall_s"),
            "bridge_steps": bridge.get("steps_completed"),
            "learned_safety_authority_steps": learned.get("authority_steps"),
            "learned_safety_brake_steps": learned.get("brake_steps"),
            "mapped_guard": mapped,
        },
        "separate_learned_route_policy": {
            "architecture_boundary": (
                "The accepted measured route PPO is retained as independent learned-route "
                "evidence. It is not placed in series with DWB because both are local motion "
                "authorities."
            ),
            "checkpoint_sha256": formal_route_checkpoint,
            "formal_success_rate": measured_route.get("results", {}).get("success_rate"),
            "formal_collision_count": measured_route.get("results", {}).get(
                "collision_count"
            ),
        },
        "geometry_boundary": {
            "status": overlay.get("status"),
            "vice_principal_door_m": [0.85, 2.12],
            "principal_door_m": [0.90, 2.12],
            "vice_principal_interior": "locked_not_captured_appearance_assumed",
            "central_polygon_step_down_m": 0.20,
            "native_scan_registration_complete": False,
        },
        "phase5_progress": {
            "measured_static_nav2_gate_percent": 55 if passed else 40,
            "remaining": [
                "blocked-route dynamic replanning gate",
                "multi-goal office directory beyond the two presentation destinations",
                "native RoomPlan section registration and visual mesh refinement",
                "operator-facing Nav2/LiDAR/costmap presentation capture",
            ],
        },
        "physical_release": False,
        "claim_boundary": (
            "Passing proves a measured-presentation static Nav2 mission with mapped simulation "
            "guards and a learned 360 brake layer. It is not an as-built digital twin, a "
            "blocked-route/replanning acceptance, sim-to-real evidence, or physical approval."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AISHA_NAV2_MEASURED_INTEGRATION passed={passed} "
        f"checks={report['checks_passed']}/{report['checks_total']} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
