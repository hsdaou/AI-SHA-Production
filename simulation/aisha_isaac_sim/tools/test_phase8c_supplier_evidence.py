#!/usr/bin/env python3
"""Contract tests for the Phase 8C supplier documentary evidence gate."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "config/phase8c_supplier_documentary_evidence.yaml"
REPORT = ROOT / "results/phase8c_supplier_documentary_evidence.json"
ATTACHMENT_REPORT = ROOT / "results/phase8b_hardware_attachment_inventory.json"


class Phase8CSupplierEvidenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile_text = PROFILE.read_text(encoding="utf-8")
        cls.profile = yaml.safe_load(cls.profile_text)
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.attachment_report = json.loads(ATTACHMENT_REPORT.read_text(encoding="utf-8"))

    def test_sanitized_documentary_gate_passes(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(self.report["documentary_compatibility_passed"])
        self.assertTrue(all(self.report["contract_checks"].values()))
        self.assertEqual(
            self.report["contract_checks_passed"], self.report["contract_checks_total"]
        )
        self.assertEqual(self.report["contract_checks_total"], 17)
        self.assertTrue(self.report["local_hash_verification_required"])
        self.assertTrue(self.report["local_hashes_passed"])

    def test_registered_source_hashes_are_exact(self) -> None:
        self.assertEqual(
            self.report["registered_hashes"],
            {
                "procurement_record": "1ddc55331e7adae275958e9ed203108b28310955699ea5925b5ea375ead203fb",
                "shipping_record": "0623f6e03e7033f3b1c58e8013e4aa69fc6d876ca0ba2dacc640cceb9cb0021e",
                "rs485_manual": "79f6e112d6820d4740ec8b507ccfde8840b71811e1c2bee36d7d82a3b4a31aa7",
            },
        )

    def test_private_source_material_is_not_committed_in_profile(self) -> None:
        self.assertNotIn("/home/", self.profile_text)
        self.assertIsNone(re.search(r"[\w.+-]+@[\w.-]+", self.profile_text))
        privacy = self.profile["privacy_boundary"]
        self.assertFalse(privacy["source_documents_committed"])
        self.assertFalse(privacy["email_body_committed"])
        self.assertFalse(privacy["personal_contact_details_committed"])

    def test_no_physical_runtime_or_release_is_claimed(self) -> None:
        self.assertFalse(self.report["received_driver_label_verified"])
        self.assertFalse(self.report["usb_rs485_identity_verified"])
        self.assertFalse(self.report["rs485_read_only_runtime_passed"])
        self.assertFalse(self.report["wheel_motion_authorized"])
        self.assertFalse(self.report["physical_release"])
        self.assertFalse(
            self.profile["supplier_attestation"]["candidate_encoder_scale_physically_verified"]
        )

    def test_attachment_gate_has_only_physical_identity_blockers(self) -> None:
        self.assertFalse(self.attachment_report["passed"])
        self.assertTrue(
            self.attachment_report["identity_checks"][
                "supplier_attested_v4_2_manual_compatibility"
            ]
        )
        self.assertEqual(
            set(self.attachment_report["blockers"]),
            {
                "received_driver_label_not_provided",
                "no_stable_usb_rs485_device",
            },
        )
        self.assertFalse(self.attachment_report["serial_port_opened"])
        self.assertEqual(self.attachment_report["modbus_frames_sent"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
