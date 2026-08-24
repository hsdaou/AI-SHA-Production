#!/usr/bin/env python3
"""Passively audit Phase 8B hardware identity without opening a serial port."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "config/phase8b_hardware_attachment_gate.yaml",
    )
    parser.add_argument("--supplier-archive", type=Path)
    parser.add_argument("--driver-label-photo", type=Path)
    parser.add_argument("--confirmed-driver-label", default="")
    parser.add_argument("--matching-rs485-manual", type=Path)
    parser.add_argument("--confirm-manual-matches-label", action="store_true")
    parser.add_argument("--expected-usb-serial", default="")
    parser.add_argument("--serial-by-id-root", type=Path, default=Path("/dev/serial/by-id"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase8b_hardware_attachment_inventory.json",
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


def udev_properties(device: Path) -> dict[str, str]:
    result = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={device}"],
        check=False,
        capture_output=True,
        text=True,
    )
    properties: dict[str, str] = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                properties[key] = value
    return properties


def serial_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    inventory = []
    for stable_path in sorted(root.iterdir()):
        resolved = stable_path.resolve()
        properties = udev_properties(resolved)
        inventory.append(
            {
                "stable_path": str(stable_path),
                "resolved_device": str(resolved),
                "id_vendor": properties.get("ID_VENDOR_ID"),
                "id_model": properties.get("ID_MODEL_ID"),
                "id_serial": properties.get("ID_SERIAL"),
                "id_serial_short": properties.get("ID_SERIAL_SHORT"),
                "usb_driver": properties.get("ID_USB_DRIVER"),
            }
        )
    return inventory


def archive_inventory(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "readable": False, "entries": []}
    if not path.is_file() or not zipfile.is_zipfile(path):
        return {
            "provided": True,
            "path": str(path),
            "readable": False,
            "entries": [],
        }
    nested_archives = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for info in archive.infolist():
            name_lower = info.filename.casefold()
            if not name_lower.endswith(".zip") or "zlac" not in name_lower:
                continue
            with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as nested_stream:
                with archive.open(info) as source:
                    shutil.copyfileobj(source, nested_stream, length=1024 * 1024)
                nested_stream.seek(0)
                if not zipfile.is_zipfile(nested_stream):
                    nested_archives.append(
                        {
                            "entry": info.filename,
                            "readable": False,
                            "entries": [],
                        }
                    )
                    continue
                with zipfile.ZipFile(nested_stream) as nested:
                    nested_names = nested.namelist()
                nested_archives.append(
                    {
                        "entry": info.filename,
                        "readable": True,
                        "total_entries": len(nested_names),
                        "evidence_entries": [
                            name
                            for name in nested_names
                            if name.casefold().endswith(".pdf")
                            and any(
                                token in Path(name).name.casefold()
                                for token in ("manual", "rs485")
                            )
                        ],
                        "v4_0_named_material_present": any(
                            "v4.0" in name.casefold() for name in nested_names
                        ),
                        "v4_2_named_material_present": any(
                            "v4.2" in name.casefold() for name in nested_names
                        ),
                    }
                )
    relevant = [
        name for name in names if any(token in name.casefold() for token in ("zlac", "8015", "rs485"))
    ]
    all_v4_0_findings = ["v4.0" in name.casefold() for name in relevant]
    all_v4_2_findings = ["v4.2" in name.casefold() for name in relevant]
    all_v4_0_findings.extend(
        item.get("v4_0_named_material_present") is True for item in nested_archives
    )
    all_v4_2_findings.extend(
        item.get("v4_2_named_material_present") is True for item in nested_archives
    )
    return {
        "provided": True,
        "path": str(path.resolve()),
        "readable": True,
        "sha256": sha256(path),
        "entries": relevant,
        "nested_archives": nested_archives,
        "v4_0_named_material_present": any(all_v4_0_findings),
        "v4_2_named_material_present": any(all_v4_2_findings),
    }


def optional_file_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "readable": False}
    readable = path.is_file()
    result: dict[str, Any] = {
        "provided": True,
        "path": str(path.resolve()),
        "readable": readable,
    }
    if readable:
        result["sha256"] = sha256(path)
        result["size_bytes"] = path.stat().st_size
    return result


def main() -> int:
    args = parse_args()
    profile = load_yaml(args.profile)
    expected_label = profile["required_identity_evidence"]["exact_received_driver_label"]
    supplier = archive_inventory(args.supplier_archive)
    label_photo = optional_file_evidence(args.driver_label_photo)
    manual = optional_file_evidence(args.matching_rs485_manual)
    serial_devices = serial_inventory(args.serial_by_id_root)
    confirmed_label_matches = (
        bool(args.confirmed_driver_label)
        and args.confirmed_driver_label.strip().casefold() == expected_label.casefold()
    )
    stable_serial_match = False
    if args.expected_usb_serial:
        expected_serial = args.expected_usb_serial.casefold()
        stable_serial_match = any(
            expected_serial
            in " ".join(
                str(device.get(field) or "")
                for field in ("stable_path", "id_serial", "id_serial_short")
            ).casefold()
            for device in serial_devices
        )

    identity_checks = {
        "supplier_archive_readable": supplier.get("readable") is True,
        "v4_0_supplier_material_identified": supplier.get("v4_0_named_material_present") is True,
        "v4_2_specific_supplier_material_found": supplier.get("v4_2_named_material_present") is True,
        "received_driver_label_photo_hashed": label_photo.get("readable") is True,
        "received_driver_label_text_matches_expected": confirmed_label_matches,
        "matching_rs485_manual_hashed": manual.get("readable") is True,
        "operator_confirmed_manual_matches_label": args.confirm_manual_matches_label,
        "stable_usb_rs485_device_found": bool(serial_devices),
        "expected_usb_serial_matches_stable_device": stable_serial_match,
    }
    gate_passed = all(
        identity_checks[name]
        for name in (
            "received_driver_label_photo_hashed",
            "received_driver_label_text_matches_expected",
            "matching_rs485_manual_hashed",
            "operator_confirmed_manual_matches_label",
            "stable_usb_rs485_device_found",
            "expected_usb_serial_matches_stable_device",
        )
    )
    blockers = []
    if not identity_checks["v4_2_specific_supplier_material_found"]:
        blockers.append("v4_2_manual_absent_from_supplier_archive")
    if not identity_checks["received_driver_label_photo_hashed"]:
        blockers.append("received_driver_label_not_provided")
    elif not identity_checks["received_driver_label_text_matches_expected"]:
        blockers.append("received_driver_label_text_not_confirmed_or_mismatched")
    if not identity_checks["matching_rs485_manual_hashed"]:
        blockers.append("exact_matching_manual_not_provided")
    elif not identity_checks["operator_confirmed_manual_matches_label"]:
        blockers.append("manual_to_received_label_match_not_confirmed")
    if not identity_checks["stable_usb_rs485_device_found"]:
        blockers.append("no_stable_usb_rs485_device")
    elif not identity_checks["expected_usb_serial_matches_stable_device"]:
        blockers.append("usb_rs485_serial_identity_not_confirmed")

    report = {
        "report_type": "phase8b_passive_hardware_attachment_inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": gate_passed,
        "status": "ready_for_operator_review" if gate_passed else "blocked_missing_hardware_evidence",
        "expected_driver_label": expected_label,
        "identity_checks": identity_checks,
        "supplier_archive": supplier,
        "driver_label_photo": label_photo,
        "matching_rs485_manual": manual,
        "serial_by_id_root": str(args.serial_by_id_root),
        "serial_devices": serial_devices,
        "blockers": blockers,
        "serial_port_opened": False,
        "modbus_frames_sent": 0,
        "motor_write_available": False,
        "motion_authorized": False,
        "physical_release": False,
        "next_gate": (
            "operator_review_then_guarded_rs485_read_only_probe"
            if gate_passed
            else "provide_exact_label_photo_matching_manual_and_stable_usb_rs485_identity"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "AISHA_PHASE8B_HARDWARE_ATTACHMENT "
        f"passed={gate_passed} serial_opened=false frames_sent=0 "
        f"blockers={','.join(blockers) or 'none'} report={args.output.resolve()}"
    )
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
