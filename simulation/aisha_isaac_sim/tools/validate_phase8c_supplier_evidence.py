#!/usr/bin/env python3
"""Validate the sanitized Phase 8C supplier evidence contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_BLOCKERS = {
    "received_driver_label_not_provided",
    "no_stable_usb_rs485_device",
}
SOURCE_ARGUMENTS = {
    "procurement_record": "procurement_record",
    "shipping_record": "shipping_record",
    "rs485_manual": "rs485_manual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "config/phase8c_supplier_documentary_evidence.yaml",
    )
    parser.add_argument("--procurement-record", type=Path)
    parser.add_argument("--shipping-record", type=Path)
    parser.add_argument("--rs485-manual", type=Path)
    parser.add_argument("--require-local-hash-verification", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase8c_supplier_documentary_evidence.json",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source(path: Path | None, expected_hash: str) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "readable": False, "sha256_matches": None}
    readable = path.is_file()
    observed_hash = sha256(path) if readable else None
    return {
        "provided": True,
        "readable": readable,
        "sha256_matches": observed_hash == expected_hash if readable else False,
    }


def main() -> int:
    args = parse_args()
    profile = load_yaml(args.profile)
    evidence = profile.get("documentary_evidence", {})
    gate = profile.get("gate", {})
    privacy = profile.get("privacy_boundary", {})
    attestation = profile.get("supplier_attestation", {})
    serialized = args.profile.read_text(encoding="utf-8")

    contract_checks = {
        "phase_is_8c": profile.get("phase") == "8C",
        "driver_model_is_v4_2": (
            profile.get("expected_hardware", {}).get("driver_model_and_revision")
            == "ZLAC8015D V4.2"
        ),
        "procurement_record_confirms_v4_2": (
            evidence.get("procurement_record", {}).get("confirms_driver_model_and_revision")
            is True
        ),
        "shipping_record_confirms_v4_2": (
            evidence.get("shipping_record", {}).get("confirms_driver_model_and_revision")
            is True
        ),
        "manual_version_registered": (
            evidence.get("rs485_manual", {}).get("version") == "1.06-20251111"
            and evidence.get("rs485_manual", {}).get("pages") == 25
        ),
        "manual_read_only_function_documented": (
            evidence.get("rs485_manual", {}).get(
                "read_holding_register_function_0x03_documented"
            )
            is True
        ),
        "supplier_compatibility_attested": (
            attestation.get("v4_series_manual_compatible_with_zlac8015d_v4_2") is True
        ),
        "encoder_scale_remains_unverified": (
            attestation.get("candidate_encoder_scale_physically_verified") is False
        ),
        "documentary_gate_passed": gate.get("documentary_compatibility_passed") is True,
        "received_label_not_claimed": gate.get("received_driver_label_verified") is False,
        "usb_identity_not_claimed": gate.get("usb_rs485_identity_verified") is False,
        "runtime_not_claimed": gate.get("rs485_read_only_runtime_passed") is False,
        "motion_not_authorized": gate.get("wheel_motion_authorized") is False,
        "physical_release_not_claimed": gate.get("physical_release") is False,
        "remaining_blockers_are_exact": set(gate.get("remaining_blockers", []))
        == EXPECTED_BLOCKERS,
        "private_sources_not_committed": (
            privacy.get("source_documents_committed") is False
            and privacy.get("email_body_committed") is False
            and privacy.get("personal_contact_details_committed") is False
            and privacy.get("local_source_paths_committed") is False
            and privacy.get("sanitized_derivative_only") is True
        ),
        "profile_contains_no_email_address_or_home_path": (
            re.search(r"[\w.+-]+@[\w.-]+", serialized) is None
            and "/home/" not in serialized
        ),
    }

    local_paths = {
        "procurement_record": args.procurement_record,
        "shipping_record": args.shipping_record,
        "rs485_manual": args.rs485_manual,
    }
    local_verification = {
        logical_id: check_source(
            local_paths[argument_name],
            str(evidence[profile_key]["sha256"]),
        )
        for logical_id, profile_key in SOURCE_ARGUMENTS.items()
        for argument_name in (logical_id,)
    }
    local_hashes_passed = all(
        item["readable"] and item["sha256_matches"] for item in local_verification.values()
    )
    contract_passed = all(contract_checks.values())
    contract_checks_passed = sum(contract_checks.values())
    contract_checks_total = len(contract_checks)
    passed = contract_passed and (
        local_hashes_passed if args.require_local_hash_verification else True
    )

    report = {
        "report_type": "phase8c_sanitized_supplier_documentary_evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status": (
            "documentary_evidence_passed_physical_intake_blocked"
            if passed
            else "supplier_evidence_validation_failed"
        ),
        "expected_driver_model_and_revision": profile["expected_hardware"][
            "driver_model_and_revision"
        ],
        "registered_hashes": {
            logical_id: evidence[profile_key]["sha256"]
            for logical_id, profile_key in SOURCE_ARGUMENTS.items()
        },
        "contract_checks": contract_checks,
        "contract_checks_passed": contract_checks_passed,
        "contract_checks_total": contract_checks_total,
        "local_hash_verification": local_verification,
        "local_hash_verification_required": args.require_local_hash_verification,
        "local_hashes_passed": local_hashes_passed,
        "documentary_compatibility_passed": gate["documentary_compatibility_passed"],
        "received_driver_label_verified": gate["received_driver_label_verified"],
        "usb_rs485_identity_verified": gate["usb_rs485_identity_verified"],
        "rs485_read_only_runtime_passed": gate["rs485_read_only_runtime_passed"],
        "wheel_motion_authorized": gate["wheel_motion_authorized"],
        "physical_release": gate["physical_release"],
        "remaining_blockers": gate["remaining_blockers"],
        "private_source_documents_committed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "AISHA_PHASE8C_SUPPLIER_EVIDENCE "
        f"passed={passed} checks={contract_checks_passed}/{contract_checks_total} "
        f"local_hashes_passed={local_hashes_passed} "
        "physical_release=false "
        f"report={args.output.resolve()}"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
