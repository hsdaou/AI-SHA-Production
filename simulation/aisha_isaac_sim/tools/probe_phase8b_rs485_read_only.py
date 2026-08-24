#!/usr/bin/env python3
"""Collect stationary ZLAC encoder telemetry without writing any register."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
DRIVER_SOURCE = REPO / "src/aisha_rev_d_driver"
sys.path.insert(0, str(DRIVER_SOURCE))

from aisha_rev_d_driver.transport import ReadOnlyRs485Transport  # noqa: E402


REQUIRED_CONFIRMATIONS = (
    "confirm_hardware_label_v4_2",
    "confirm_matching_rs485_manual",
    "confirm_motor_leads_isolated",
    "confirm_wheels_chocked",
    "confirm_external_estop_verified",
    "confirm_operator_present",
    "confirm_independent_spotter_present",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/aisha_zlac")
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase8b_rs485_read_only_probe.json",
    )
    for confirmation in REQUIRED_CONFIRMATIONS:
        parser.add_argument("--" + confirmation.replace("_", "-"), action="store_true")
    return parser.parse_args()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    confirmations = {name: bool(getattr(args, name)) for name in REQUIRED_CONFIRMATIONS}
    missing = [name for name, value in confirmations.items() if not value]
    base_report = {
        "report_type": "phase8b_rs485_read_only_stationary_probe",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "port": args.port,
        "unit": args.unit,
        "baud_rate": args.baud_rate,
        "confirmations": confirmations,
        "outgoing_modbus_functions_allowed": [3],
        "register_writes_possible": False,
        "motor_enable_possible": False,
        "motion_authorized": False,
        "physical_release": False,
    }
    if missing:
        report = {
            **base_report,
            "status": "blocked_before_serial_open",
            "passed": False,
            "missing_confirmations": missing,
            "samples": 0,
        }
        write_report(args.output, report)
        print(
            "AISHA_PHASE8B_RS485_READ_ONLY blocked=true serial_opened=false "
            f"missing={','.join(missing)} report={args.output.resolve()}"
        )
        return 2
    if args.duration_s <= 0.0 or args.rate_hz <= 0.0:
        raise ValueError("duration and rate must be positive")

    records = []
    error = None
    started = time.monotonic()
    try:
        with ReadOnlyRs485Transport(
            args.port,
            unit=args.unit,
            baud_rate=args.baud_rate,
        ) as transport:
            while time.monotonic() - started < args.duration_s:
                cycle_started = time.monotonic()
                records.append(transport.sample())
                remaining = 1.0 / args.rate_hz - (time.monotonic() - cycle_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    except Exception as exc:  # record hardware absence/protocol failure without motion
        error = f"{type(exc).__name__}: {exc}"

    elapsed = max(time.monotonic() - started, 1e-9)
    left_counts = [sample.left_count for sample in records]
    right_counts = [sample.right_count for sample in records]
    sample_rate = len(records) / elapsed
    fault_free = bool(records) and all(
        sample.left_fault == 0 and sample.right_fault == 0 for sample in records
    )
    stationary_span = (
        max(left_counts) - min(left_counts) if left_counts else None,
        max(right_counts) - min(right_counts) if right_counts else None,
    )
    passed = error is None and len(records) >= 2 and fault_free
    report = {
        **base_report,
        "status": "accepted_read_only_observation" if passed else "read_only_observation_failed",
        "passed": passed,
        "error": error,
        "duration_observed_s": elapsed,
        "samples": len(records),
        "sample_rate_hz": sample_rate,
        "fault_free": fault_free,
        "stationary_count_span_left_right": stationary_span,
        "median_actual_rpm_left": statistics.median(
            [sample.left_rpm for sample in records]
        ) if records else None,
        "median_actual_rpm_right": statistics.median(
            [sample.right_rpm for sample in records]
        ) if records else None,
        "wheels_lifted_gate_authorized": False,
    }
    write_report(args.output, report)
    print(
        "AISHA_PHASE8B_RS485_READ_ONLY "
        f"passed={passed} samples={len(records)} writes=false "
        f"report={args.output.resolve()}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
