#!/usr/bin/env python3
"""Repository-level contract tests for the Phase 8B Rev D adapter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
PROFILE = ROOT / "config/phase8b_rev_d_differential_adapter.yaml"
REPORT = ROOT / "results/phase8b_rev_d_differential_adapter_preflight.json"
TRAINING = ROOT / "config/training.yaml"
DRIVER = REPO / "src/aisha_rev_d_driver"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Phase8BRevDAdapterContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_yaml(PROFILE)

    def test_only_offline_and_guarded_reads_are_authorized(self) -> None:
        authorization = self.profile["authorization_boundary"]
        self.assertTrue(authorization["offline_protocol_and_replay"])
        self.assertTrue(authorization["rs485_register_reads_after_operator_gate"])
        for name, value in authorization.items():
            if name not in {
                "offline_protocol_and_replay",
                "rs485_register_reads_after_operator_gate",
            }:
                self.assertFalse(value, name)

    def test_adapter_has_no_physical_command_path(self) -> None:
        adapter = self.profile["adapter_contract"]
        self.assertFalse(adapter["motor_write_transport_implemented"])
        self.assertFalse(adapter["subscribes_cmd_vel"])
        self.assertFalse(adapter["broadcasts_tf"])
        node = (DRIVER / "aisha_rev_d_driver/node.py").read_text(encoding="utf-8")
        self.assertNotIn("cmd_vel", node)
        self.assertNotIn("create_subscription", node)
        self.assertNotIn("TransformBroadcaster", node)

    def test_physical_profile_defaults_fail_closed(self) -> None:
        config = load_yaml(DRIVER / "config/phase8b_rs485_read_only.yaml")
        params = config["aisha_rev_d_encoder_adapter"]["ros__parameters"]
        self.assertEqual(params["transport"], "rs485_read_only")
        self.assertFalse(params["publish_odom"])
        for name in (
            "encoder_scale_verified",
            "rolling_radius_verified",
            "encoder_signs_verified",
            "hardware_label_verified",
            "motor_leads_isolated",
            "external_estop_verified",
        ):
            self.assertFalse(params[name], name)

    def test_offline_report_passes_without_physical_credit(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks_passed"], report["checks_total"])
        self.assertEqual(report["checks_total"], 30)
        self.assertFalse(report["live_transport"]["runtime_observed"])
        self.assertFalse(report["live_transport"]["motor_write_available"])
        self.assertFalse(report["wheels_lifted_gate_passed"])
        self.assertFalse(report["floor_motion_authorized"])
        self.assertFalse(report["physical_release"])

    def test_release_ledger_matches_phase8b_evidence(self) -> None:
        release = load_yaml(TRAINING)["release"]
        self.assertTrue(release["phase8b_rev_d_adapter_offline_preflight_passed"])
        self.assertEqual(release["phase8b_rev_d_adapter_offline_checks_passed"], "30/30")
        self.assertEqual(release["phase8b_rev_d_adapter_focused_tests_passed"], "12/12")
        self.assertTrue(release["phase8b_rs485_read_only_transport_prepared"])
        self.assertFalse(release["phase8b_rs485_read_only_runtime_passed"])
        self.assertFalse(release["phase8b_wheels_lifted_gate_passed"])
        self.assertFalse(release["physical_robot_release"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
