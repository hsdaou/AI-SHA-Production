#!/usr/bin/env python3
"""Validate the Phase 6 policy in the measured administration Nav2 mission."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_PHASE6_SHA256 = (
    "e49767507925548aa0086c38e764c43037f25734943b2c5712cb58eecb0b6318"
)
PHASE6_TASK = (
    "Isaac-AISHA-Administration-Live-Measured-Nav2-Phase6-"
    "HighSpeed80-DynamicSafety-Direct-v0"
)
HIGH_SPEED_SEGMENTS = {1, 5}


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


def numeric_segment_map(source: dict) -> dict[int, float]:
    return {int(key): float(value) for key, value in source.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results/administration_nav2_phase6_high_speed_mission.json"
        ),
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results/administration_nav2_phase6_high_speed_bridge.json"
        ),
    )
    parser.add_argument(
        "--phase6-acceptance",
        type=Path,
        default=PACKAGE_ROOT / "results/phase6_high_speed_080_acceptance.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "isaaclab/checkpoints/aisha_phase6_high_speed_080_model_223.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results/administration_nav2_phase6_high_speed_integration_gate.json"
        ),
    )
    args = parser.parse_args()

    mission = load_json(args.mission)
    bridge = load_json(args.bridge)
    phase6 = load_json(args.phase6_acceptance)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    events = bridge.get("events", {})
    learned = bridge.get("learned_safety", {})
    mapped = bridge.get("mapped_site_safety", {})
    localization = bridge.get("localization", {})
    speed = bridge.get("route_scoped_speed_evidence", {})
    requested = numeric_segment_map(
        speed.get("maximum_requested_linear_mps_by_segment", {})
    )
    guarded = numeric_segment_map(
        speed.get("maximum_guarded_linear_mps_by_segment", {})
    )
    observed = numeric_segment_map(
        speed.get("maximum_observed_linear_mps_by_segment", {})
    )
    legs = mission.get("legs", [])
    successful_legs = [
        leg for leg in legs if leg.get("execution_status") == "succeeded"
    ]
    segment_ids = {int(leg.get("route_segment_id", -1)) for leg in legs}
    non_high_requested = [
        value for segment_id, value in requested.items()
        if segment_id not in HIGH_SPEED_SEGMENTS
    ]
    high_requested = [requested.get(segment_id, 0.0) for segment_id in HIGH_SPEED_SEGMENTS]
    high_guarded = [guarded.get(segment_id, 0.0) for segment_id in HIGH_SPEED_SEGMENTS]
    high_observed = [observed.get(segment_id, 0.0) for segment_id in HIGH_SPEED_SEGMENTS]

    checks = {
        "phase6_formal_acceptance_passed": (
            phase6.get("passed") is True
            and int(phase6.get("checks_total", 0)) > 0
            and phase6.get("checks_passed") == phase6.get("checks_total")
        ),
        "packaged_phase6_checkpoint_hash_matches": (
            checkpoint_sha256 == ACCEPTED_PHASE6_SHA256
            and phase6.get("selected_checkpoint", {}).get("sha256")
            == ACCEPTED_PHASE6_SHA256
        ),
        "paired_bridge_passed": bridge.get("passed") is True,
        "phase6_measured_task_selected": bridge.get("task") == PHASE6_TASK,
        "ground_truth_localization_is_explicit_and_simulation_only": (
            localization.get("nav2_global_pose_source")
            == "isaac_ground_truth_odom_with_identity_map_to_odom"
            and localization.get("bridge_publishes_map_to_odom") is True
            and localization.get("physical_localization_credit") is False
        ),
        "accepted_phase6_checkpoint_loaded": (
            learned.get("enabled") is True
            and learned.get("checkpoint_is_accepted_phase6") is True
            and learned.get("accepted_checkpoint_profile")
            == "phase6_high_speed_080"
            and learned.get("checkpoint_sha256") == ACCEPTED_PHASE6_SHA256
        ),
        "accepted_phase3n_fallback_loaded_for_non_high_speed_legs": (
            learned.get("fallback_checkpoint_is_accepted_phase3n") is True
            and learned.get("policy_selection")
            == "phase6_on_declared_high_speed_segments_phase3n_elsewhere"
            and int(learned.get("primary_policy_steps", 0)) > 0
            and int(learned.get("fallback_policy_steps", 0)) > 0
        ),
        "mission_passed": mission.get("passed") is True,
        "all_12_legs_succeeded": (
            mission.get("expected_legs") == 12
            and mission.get("completed_legs") == 12
            and len(successful_legs) == 12
        ),
        "all_route_segments_explicitly_handed_off": segment_ids == set(range(12)),
        "phase6_control_stack_declared": (
            mission.get("control_stack")
            == "nav2_mapped_doorway_phase6_high_speed_safety"
            and mission.get("phase6_high_speed_safety_coupled") is True
        ),
        "mission_completion_reached_bridge": (
            events.get("mission_complete_signal_received") is True
        ),
        "bridge_episode_never_reset": (
            events.get("episode_reset_gate_detected") is False
        ),
        "segment_messages_received_without_invalid_ids": (
            int(events.get("route_segment_messages", 0)) >= 12
            and int(events.get("invalid_route_segment_messages", -1)) == 0
        ),
        "route_speed_contract_declared": (
            bridge.get("command_constraints", {}).get(
                "high_speed_route_segment_ids"
            )
            == [1, 5]
            and bridge.get("command_constraints", {}).get("maximum_forward_mps")
            == 0.80
            and bridge.get("command_constraints", {}).get(
                "non_high_speed_navigation_maximum_mps"
            )
            == 0.30
            and bridge.get("command_constraints", {}).get(
                "route_scoped_phase3n_thresholds_enabled"
            )
            is True
        ),
        "both_high_speed_legs_requested_target_tier": (
            min(high_requested, default=0.0) >= 0.72
        ),
        "both_high_speed_legs_survived_guard_at_target_tier": (
            min(high_guarded, default=0.0) >= 0.72
        ),
        "both_high_speed_legs_physically_reached_target_tier": (
            min(high_observed, default=0.0) >= 0.72
        ),
        "non_high_speed_requests_remained_at_accepted_replay_speed": (
            bool(non_high_requested)
            and max(non_high_requested) <= 0.300001
        ),
        "both_office_pivots_succeeded": all(
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
        "both_pre_door_alignments_succeeded_in_open_corridor": all(
            next(
                (
                    leg.get("pre_door_alignment", {}).get("passed") is True
                    and leg.get("pre_door_alignment", {}).get("rotation_location")
                    == "open_approach_corridor_before_doorway"
                    for leg in legs
                    if leg.get("waypoint_id") == approach
                ),
                False,
            )
            for approach in ("vice_principal_approach", "principal_approach")
        ),
        "both_pre_door_stages_tightly_centred": all(
            next(
                (
                    leg.get("pre_door_stage_convergence", {}).get("passed") is True
                    and float(
                        leg.get("pre_door_stage_convergence", {}).get(
                            "final_distance_m", float("inf")
                        )
                    )
                    <= 0.015001
                    for leg in legs
                    if leg.get("waypoint_id") == approach
                ),
                False,
            )
            for approach in ("vice_principal_approach", "principal_approach")
        ),
        "both_measured_doorways_exercised": (
            int(mapped.get("doorway_entries", {}).get("vice_principal", 0)) > 0
            and int(mapped.get("doorway_entries", {}).get("principal", 0)) > 0
        ),
        "doorway_speed_limit_preserved": (
            float(mapped.get("maximum_doorway_speed_mps", 0.0)) == 0.10
            and float(
                mapped.get("maximum_abs_speed_in_doorway_mps", float("inf"))
            )
            <= 0.100001
        ),
        "measured_scene_alignment_breakaway_is_disclosed": (
            float(mapped.get("breakaway_angular_rad_s", 0.0)) == 0.42
        ),
        "central_drop_full_footprint_clearance_positive": (
            float(
                mapped.get("minimum_polygon_full_footprint_clearance_m", -1.0)
            )
            > 0.0
        ),
        "learned_safety_was_live": (
            int(learned.get("authority_steps", 0)) > 0
            and int(learned.get("brake_steps", 0)) > 0
        ),
        "physical_release_remains_false": (
            bridge.get("physical_release") is False
            and mission.get("physical_release") is False
            and phase6.get("claim_boundary", {}).get("physical_release") is False
        ),
    }
    passed = all(checks.values())
    report = {
        "report_type": "administration_nav2_phase6_high_speed_integration_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "accepted_measured_nav2_replay"
            if passed
            else "not_accepted"
        ),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "selected_checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
        },
        "mission": {
            "report": str(args.mission.resolve()),
            "legs_completed": mission.get("completed_legs"),
            "elapsed_wall_s": mission.get("elapsed_wall_s"),
            "office_pivots": {
                leg.get("waypoint_id"): leg.get("post_visit_pivot")
                for leg in legs
                if leg.get("waypoint_id") in {"vice_principal", "principal"}
            },
        },
        "speed_evidence": {
            "target_high_speed_mps": 0.80,
            "acceptance_floor_mps": 0.72,
            "high_speed_segment_ids": sorted(HIGH_SPEED_SEGMENTS),
            "maximum_requested_linear_mps_by_segment": requested,
            "maximum_guarded_linear_mps_by_segment": guarded,
            "maximum_observed_linear_mps_by_segment": observed,
            "minimum_high_speed_requested_mps": min(high_requested, default=0.0),
            "minimum_high_speed_guarded_mps": min(high_guarded, default=0.0),
            "minimum_high_speed_observed_mps": min(high_observed, default=0.0),
        },
        "mapped_guard": mapped,
        "learned_safety": learned,
        "claim_boundary": {
            "supported": (
                "Isaac Sim measured-administration Nav2 replay with the accepted "
                "Phase 6 actor and route-scoped 0.80 m/s straight-hallway tier"
            ),
            "not_yet_supported": [
                "final RTX Omniverse presentation video by this integration gate",
                "sim-to-real performance",
                "physical stopping distance or human protective fields",
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
        f"AISHA_NAV2_PHASE6_INTEGRATION passed={passed} "
        f"checks={sum(checks.values())}/{len(checks)} report={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
