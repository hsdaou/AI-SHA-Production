#!/usr/bin/env python3
"""Validate Phase 8B protocol, differential odometry and fail-safe boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
DRIVER_ROOT = REPO / "src/aisha_rev_d_driver"
sys.path.insert(0, str(DRIVER_ROOT))

from aisha_rev_d_driver.modbus import (  # noqa: E402
    REG_CONTROL_WORD,
    REG_POSITION_LEFT_HIGH,
    REG_TARGET_VELOCITY_LEFT,
    REG_TARGET_VELOCITY_RIGHT,
    append_crc,
    assert_read_only_request,
    build_read_holding_registers,
    build_write_single_register,
    decode_telemetry_registers,
    parse_read_holding_registers,
)
from aisha_rev_d_driver.odometry import (  # noqa: E402
    DifferentialGeometry,
    EncoderOdometry,
    signed_int32_delta,
)
from aisha_rev_d_driver.safety import DryRunCommandGate  # noqa: E402
from aisha_rev_d_driver.transport import ReplayTransport  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "config/phase8b_rev_d_differential_adapter.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase8b_rev_d_differential_adapter_preflight.json",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    profile = load_yaml(args.profile)
    drive = load_yaml(ROOT / "config/aisha_drive.yaml")
    replay_config_path = DRIVER_ROOT / "config/phase8b_replay.yaml"
    read_only_config_path = DRIVER_ROOT / "config/phase8b_rs485_read_only.yaml"
    replay_path = DRIVER_ROOT / "config/phase8b_encoder_replay.jsonl"
    node_path = DRIVER_ROOT / "aisha_rev_d_driver/node.py"
    transport_path = DRIVER_ROOT / "aisha_rev_d_driver/transport.py"
    launch_path = DRIVER_ROOT / "launch/phase8b_replay.launch.py"
    package_path = DRIVER_ROOT / "package.xml"
    probe_path = ROOT / "tools/probe_phase8b_rs485_read_only.py"
    replay_config = load_yaml(replay_config_path)["aisha_rev_d_encoder_adapter"]["ros__parameters"]
    read_only_config = load_yaml(read_only_config_path)["aisha_rev_d_encoder_adapter"]["ros__parameters"]
    node_source = node_path.read_text(encoding="utf-8")
    transport_source = transport_path.read_text(encoding="utf-8")
    launch_source = launch_path.read_text(encoding="utf-8")
    package_source = package_path.read_text(encoding="utf-8")
    probe_source = probe_path.read_text(encoding="utf-8")
    hardware = profile["hardware_contract"]
    protocol = profile["supplier_protocol"]
    authorization = profile["authorization_boundary"]
    adapter = profile["adapter_contract"]
    dry_run = profile["dry_run_safety_contract"]

    geometry = DifferentialGeometry(
        hardware["wheel_radius_design_m"],
        hardware["wheel_track_design_m"],
        hardware["encoder_counts_per_rev_candidate"],
    )
    left_rpm, right_rpm = geometry.body_to_wheel_rpm(
        dry_run["equivalent_straight_speed_design_mps"], 0.0
    )
    pivot_geometry = DifferentialGeometry(0.1, 0.72, 1000)
    pivot_odom = EncoderOdometry(pivot_geometry)
    pivot_odom.update(0.0, 0, 0)
    pivot = pivot_odom.update(1.0, -900, 900)

    registers = [0xFFFF, 0xFFFE, 0x0001, 0x0002, 0xFFCE, 0x0032]
    response = append_crc(
        bytes([1, 3, 12]) + b"".join(value.to_bytes(2, "big") for value in registers)
    )
    decoded = decode_telemetry_registers(
        parse_read_holding_registers(response, expected_unit=1, expected_count=6)
    )
    read_guard_rejects_write = False
    try:
        assert_read_only_request(build_write_single_register(1, REG_CONTROL_WORD, 8))
    except PermissionError:
        read_guard_rejects_write = True

    replay_samples = list(ReplayTransport(replay_path).samples())
    replay_odom = EncoderOdometry(geometry)
    replay_estimate = None
    for sample in replay_samples:
        estimate = replay_odom.update(sample.stamp_s, sample.left_count, sample.right_count)
        if estimate is not None:
            replay_estimate = estimate

    gate = DryRunCommandGate(
        geometry,
        max_wheel_rpm=dry_run["maximum_wheel_speed_rpm"],
        timeout_s=dry_run["command_timeout_s"],
        reverse_enabled=dry_run["reverse_enabled"],
    )
    clamped = gate.accept(0.0, linear_mps=0.8, angular_rad_s=0.0)
    timed_out = gate.sample(dry_run["command_timeout_s"])
    reverse_rejected = False
    lateral_rejected = False
    try:
        gate.accept(1.0, linear_mps=-0.01, angular_rad_s=0.0)
    except ValueError:
        reverse_rejected = True
    try:
        gate.accept(1.0, linear_mps=0.0, angular_rad_s=0.0, lateral_mps=0.01)
    except ValueError:
        lateral_rejected = True

    register_addresses = {
        name: item["address"] for name, item in protocol["registers"].items()
    }
    blocker_ids = {item["id"] for item in profile["hard_blockers"]}
    checks = {
        "rev_d_geometry_matches_design_contract": (
            hardware["robot_revision"] == drive["model"]["revision"] == "D"
            and hardware["drive_architecture"] == drive["model"]["architecture"]
            and math.isclose(hardware["wheel_radius_design_m"], drive["geometry"]["wheel_radius_nominal_m"])
            and math.isclose(hardware["wheel_track_design_m"], drive["geometry"]["wheel_track_design_m"])
        ),
        "candidate_encoder_scale_is_not_claimed_verified": (
            hardware["encoder_lines_per_rev"] == 4096
            and hardware["encoder_counts_per_rev_candidate"] == 16384
            and hardware["encoder_counts_per_rev_verified"] is False
        ),
        "manual_to_expected_hardware_version_mismatch_is_explicit": (
            "V4.2" in hardware["ordered_hardware_label_expected"]
            and hardware["supplied_rs485_manual_family"] == "ZLAC8015D V4 Series"
            and hardware["manual_to_hardware_exact_match_verified"] is False
        ),
        "supplier_register_addresses_are_exact": register_addresses == {
            "control_mode": 0x200D,
            "control_word": 0x200E,
            "target_velocity_left": 0x2088,
            "target_velocity_right": 0x2089,
            "status_word": 0x20A2,
            "fault_left": 0x20A5,
            "fault_right": 0x20A6,
            "position_left_high": 0x20A7,
            "position_left_low": 0x20A8,
            "position_right_high": 0x20A9,
            "position_right_low": 0x20AA,
            "actual_velocity_left": 0x20AB,
            "actual_velocity_right": 0x20AC,
        },
        "supplier_position_word_order_and_signed_range_retained": (
            protocol["position_word_order"] == "high_16_then_low_16"
            and protocol["position_range"] == "signed_int32"
            and decoded.left_count == -2
            and decoded.right_count == 0x00010002
        ),
        "supplier_velocity_scale_and_sign_decode": (
            math.isclose(decoded.left_rpm, -5.0) and math.isclose(decoded.right_rpm, 5.0)
        ),
        "supplier_enable_frame_matches_manual": (
            build_write_single_register(1, REG_CONTROL_WORD, 8).hex(" ").upper()
            == "01 06 20 0E 00 08 E2 0F"
        ),
        "supplier_left_positive_velocity_frame_matches_manual": (
            build_write_single_register(1, REG_TARGET_VELOCITY_LEFT, 100).hex(" ").upper()
            == "01 06 20 88 00 64 03 CB"
        ),
        "supplier_right_negative_velocity_frame_matches_manual": (
            build_write_single_register(1, REG_TARGET_VELOCITY_RIGHT, -100).hex(" ").upper()
            == "01 06 20 89 FF 9C 12 79"
        ),
        "supplier_left_encoder_read_frame_matches_manual": (
            build_read_holding_registers(1, REG_POSITION_LEFT_HIGH, 2).hex(" ").upper()
            == "01 03 20 A7 00 02 7E 28"
        ),
        "live_transport_allows_only_function_03": (
            protocol["live_transport_allowed_functions"] == [3]
            and "assert_read_only_request(request)" in transport_source
            and read_guard_rejects_write
        ),
        "straight_5rpm_conversion_is_consistent": (
            math.isclose(left_rpm, 5.0, rel_tol=1e-8)
            and math.isclose(right_rpm, 5.0, rel_tol=1e-8)
        ),
        "positive_pivot_integration_is_consistent": (
            pivot is not None and math.isclose(pivot.pose.yaw_rad, math.pi / 2.0, abs_tol=1e-12)
        ),
        "signed_int32_rollover_is_continuous": (
            signed_int32_delta(-0x80000000, 0x7FFFFFFF) == 1
            and signed_int32_delta(0x7FFFFFFF, -0x80000000) == -1
        ),
        "replay_fixture_is_deterministic": (
            len(replay_samples) == 21
            and replay_samples[0].left_count == 0
            and replay_samples[-1].left_count == replay_samples[-1].right_count == 1365
        ),
        "replay_odometry_is_straight_and_bounded": (
            replay_estimate is not None
            and 0.0522 < replay_estimate.pose.x_m < 0.0525
            and math.isclose(replay_estimate.pose.y_m, 0.0)
            and math.isclose(replay_estimate.pose.yaw_rad, 0.0)
        ),
        "dry_run_clamps_to_wheels_lifted_limit": (
            clamped.clamped
            and math.isclose(abs(clamped.left_rpm), 5.0)
            and math.isclose(abs(clamped.right_rpm), 5.0)
        ),
        "dry_run_timeout_fails_to_zero": (
            timed_out.timed_out and timed_out.left_rpm == timed_out.right_rpm == 0.0
        ),
        "reverse_is_disabled_and_lateral_motion_rejected": (
            reverse_rejected and lateral_rejected and dry_run["reverse_enabled"] is False
        ),
        "ros_node_has_no_command_subscription_or_tf_broadcast": (
            "create_subscription" not in node_source
            and "cmd_vel" not in node_source
            and "TransformBroadcaster" not in node_source
        ),
        "ros_node_suppresses_uncalibrated_physical_odometry": (
            'self._transport_name == "replay" or calibration_verified' in node_source
            and "Physical odometry publication suppressed" in node_source
        ),
        "replay_topics_are_isolated_from_physical_ekf_input": (
            replay_config["transport"] == "replay"
            and replay_config["odom_topic"] == "/phase8b/replay/wheel_odom_raw"
            and replay_config["publish_odom"] is True
            and 'self._odom_topic == "/wheel/odom_raw"' in node_source
            and "Replay odometry must use an isolated topic" in node_source
        ),
        "physical_profile_is_read_only_and_fails_closed": (
            read_only_config["transport"] == "rs485_read_only"
            and read_only_config["publish_odom"] is False
            and all(
                read_only_config[name] is False
                for name in (
                    "encoder_scale_verified",
                    "rolling_radius_verified",
                    "encoder_signs_verified",
                    "hardware_label_verified",
                    "motor_leads_isolated",
                    "external_estop_verified",
                )
            )
        ),
        "replay_launch_starts_only_new_adapter": (
            'package="aisha_rev_d_driver"' in launch_source
            and "mecanum_driver" not in launch_source
            and "cmd_vel" not in launch_source
        ),
        "package_declares_ros_and_serial_dependencies": all(
            token in package_source
            for token in (
                "<depend>ament_index_python</depend>",
                "<depend>nav_msgs</depend>",
                "<depend>rclpy</depend>",
                "<depend>std_msgs</depend>",
                "<exec_depend>python3-serial</exec_depend>",
            )
        ),
        "read_only_probe_blocks_before_serial_open": (
            '"status": "blocked_before_serial_open"' in probe_source
            and "REQUIRED_CONFIRMATIONS" in probe_source
            and '"register_writes_possible": False' in probe_source
        ),
        "all_motion_and_release_authority_remains_false": (
            authorization["offline_protocol_and_replay"] is True
            and authorization["rs485_register_reads_after_operator_gate"] is True
            and all(
                value is False
                for name, value in authorization.items()
                if name not in {
                    "offline_protocol_and_replay",
                    "rs485_register_reads_after_operator_gate",
                }
            )
        ),
        "motor_write_transport_is_explicitly_absent": (
            adapter["motor_write_transport_implemented"] is False
            and adapter["subscribes_cmd_vel"] is False
            and dry_run["hardware_output_exists"] is False
        ),
        "wheels_lifted_gate_remains_blocked": (
            profile["wheels_lifted_direction_encoder_gate"]["status"] == "blocked"
            and profile["wheels_lifted_direction_encoder_gate"]["completion_claimed"] is False
            and profile["rs485_read_only_operator_gate"]["gate_passed"] is False
        ),
        "critical_physical_blockers_are_retained": {
            "exact_driver_protocol_match_pending",
            "physical_encoder_scale_pending",
            "loaded_rolling_radius_pending",
            "encoder_direction_pending",
            "live_motor_write_transport_not_implemented",
            "physical_runtime_not_observed",
            "physical_route_release_blockers_retained",
        }.issubset(blocker_ids),
    }

    passed = all(checks.values())
    report = {
        "report_type": "phase8b_rev_d_differential_adapter_offline_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": "accepted_offline_replay_physical_blocked" if passed else "offline_contract_failed",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "replay": {
            "samples": len(replay_samples),
            "distance_m": replay_estimate.pose.x_m if replay_estimate else None,
            "candidate_counts_per_rev": hardware["encoder_counts_per_rev_candidate"],
            "candidate_scale_physical_credit": False,
        },
        "live_transport": {
            "available": "rs485_read_only",
            "allowed_modbus_functions": [3],
            "runtime_observed": False,
            "motor_write_available": False,
        },
        "wheels_lifted_gate_passed": False,
        "floor_motion_authorized": False,
        "physical_release": False,
        "hard_blockers": profile["hard_blockers"],
        "source_hashes": {
            "profile": sha256(args.profile),
            "drive_contract": sha256(ROOT / "config/aisha_drive.yaml"),
            "node": sha256(node_path),
            "transport": sha256(transport_path),
            "replay_fixture": sha256(replay_path),
            "read_only_probe": sha256(probe_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "AISHA_PHASE8B_REV_D_ADAPTER "
        f"passed={passed} checks={report['checks_passed']}/{report['checks_total']} "
        f"motor_write=false wheels_lifted=false report={args.output.resolve()}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
