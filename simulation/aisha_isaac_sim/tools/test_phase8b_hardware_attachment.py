#!/usr/bin/env python3
"""Contract tests for the passive Phase 8B hardware attachment audit."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
PROFILE = ROOT / "config/phase8b_hardware_attachment_gate.yaml"
REPORT = ROOT / "results/phase8b_hardware_attachment_inventory.json"
AUDITOR = ROOT / "tools/audit_phase8b_hardware_attachment.py"
LOOPBACK = REPO / "src/aisha_rev_d_driver/test/test_serial_loopback.py"


class Phase8BHardwareAttachmentContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_auditor_cannot_open_serial_or_send_modbus(self) -> None:
        source = AUDITOR.read_text(encoding="utf-8")
        self.assertNotIn("import serial", source)
        self.assertNotIn("serial.Serial", source)
        self.assertNotIn("build_read_holding_registers", source)
        self.assertNotIn("build_write_single_register", source)
        self.assertFalse(self.profile["authorization_boundary"]["serial_port_open"])
        self.assertFalse(self.profile["authorization_boundary"]["modbus_read"])
        self.assertFalse(self.profile["authorization_boundary"]["modbus_write"])

    def test_current_inventory_is_honestly_blocked(self) -> None:
        self.assertFalse(self.report["passed"])
        self.assertEqual(self.report["status"], "blocked_missing_hardware_evidence")
        self.assertFalse(self.report["serial_port_opened"])
        self.assertEqual(self.report["modbus_frames_sent"], 0)
        self.assertFalse(self.report["motion_authorized"])
        self.assertFalse(self.report["physical_release"])
        self.assertEqual(
            set(self.report["blockers"]),
            {
                "v4_2_manual_absent_from_supplier_archive",
                "received_driver_label_not_provided",
                "exact_matching_manual_not_provided",
                "no_stable_usb_rs485_device",
            },
        )

    def test_archive_and_local_usb_findings_are_specific(self) -> None:
        archive = self.report["supplier_archive"]
        self.assertTrue(archive["readable"])
        self.assertTrue(archive["v4_0_named_material_present"])
        self.assertFalse(archive["v4_2_named_material_present"])
        self.assertEqual(len(archive["nested_archives"]), 1)
        nested = archive["nested_archives"][0]
        self.assertTrue(nested["readable"])
        self.assertTrue(
            any("RS485 Communication Version 1.06" in name for name in nested["evidence_entries"])
        )
        self.assertFalse(nested["v4_2_named_material_present"])
        self.assertEqual(self.report["serial_devices"], [])

    def test_loopback_test_asserts_read_only_register_sequence(self) -> None:
        source = LOOPBACK.read_text(encoding="utf-8")
        self.assertIn("assert request[1] == 0x03", source)
        self.assertIn("[\n        0x20A7,\n        0x20A2,\n        0x20A5,\n    ]", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
