#!/usr/bin/env python3
"""Regression tests for measured-site intake and scene-overlay plumbing."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SIM_ROOT = Path(__file__).resolve().parents[1]
PREPARER = SIM_ROOT / "tools" / "prepare_measured_administration.py"
TEMPLATE = SIM_ROOT / "config" / "measured_administration_template.yaml"
PRESENTATION_OVERLAY = (
    SIM_ROOT / "config" / "measured_administration_presentation_2026-08-23.yaml"
)
BUILDER = SIM_ROOT / "scripts" / "build_administration.py"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MeasuredAdministrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preparer = import_file("prepare_measured_administration", PREPARER)
        sys.path.insert(0, str(SIM_ROOT / "scripts"))
        try:
            cls.builder = import_file("build_administration", BUILDER)
        finally:
            sys.path.pop(0)

    def complete_manifest(self) -> dict:
        manifest = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
        manifest["status"] = "capture_complete_pending_validation"
        manifest["coordinate_contract"]["source_metadata_verified_in_export"] = True
        manifest["capture"].update(
            {
                "date": "2026-08-23",
                "operator": "test",
                "device_model": "test lidar device",
                "application": "test",
                "application_version": "1",
                "privacy_reviewed": True,
                "people_and_sensitive_documents_excluded": True,
            }
        )
        manifest["scans"]["complete_structure"]["file"] = "administration.obj"
        measurements = manifest["manual_measurements"]
        measurements.update(
            {
                "east_hallway_clear_width_m": 2.80,
                "principal_passage_clear_width_m": 2.60,
                "ceiling_height_m": 3.00,
                "vice_principal_turn_zone_size_m": [2.00, 2.00],
                "principal_turn_zone_size_m": [1.80, 1.80],
            }
        )
        for name, centre, rotation in (
            ("vice_principal_door", [17.10, -5.05], 0.0),
            ("principal_door", [6.978, -7.628], 45.0),
        ):
            measurements[name].update(
                {
                    "clear_width_m": 1.00,
                    "clear_height_m": 2.10,
                    "frame_depth_m": 0.15,
                    "threshold_hallway_mm": 0.0,
                    "threshold_office_mm": 0.0,
                    "threshold_profile": "flush",
                    "hinge_side_from_hallway": "left",
                    "swing_from_hallway": "outward",
                    "centre_xy_m": centre,
                    "wall_rotation_deg": rotation,
                }
            )
        return manifest

    def test_template_preflight_is_complete_but_awaiting_site_data(self) -> None:
        manifest = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
        checks = self.preparer.measurement_checks(manifest["manual_measurements"])
        self.assertEqual(len(checks), 25)
        self.assertFalse(any(checks.values()))
        self.assertIs(manifest["safety_boundary"]["generated_overlay_never_releases_physical_operation"], True)

    def test_complete_intake_hashes_scan_and_writes_gated_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "administration.obj").write_text(
                "v 0 0 0\nv 22 0 0\nv 22 10 3\nv 0 10 3\nf 1 2 3 4\n",
                encoding="utf-8",
            )
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(self.complete_manifest(), sort_keys=False), encoding="utf-8"
            )
            report_path = root / "report.json"
            overlay_path = root / "overlay.yaml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PREPARER),
                    "--manifest",
                    str(manifest_path),
                    "--scan-root",
                    str(root),
                    "--output",
                    str(report_path),
                    "--overlay-output",
                    str(overlay_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertTrue(report["route_geometry_gate"]["candidate_geometry_valid"])
            self.assertEqual(report["scan_gate"]["artifacts"][0]["vertices"], 4)
            self.assertEqual(len(report["scan_gate"]["artifacts"][0]["sha256"]), 64)
            self.assertEqual(overlay["status"], "measured_site_candidate")
            self.assertTrue(overlay["candidate_route_geometry_valid"])
            self.assertIs(overlay["physical_release"], False)
            self.assertEqual(
                overlay["known_dimensions"]["hallway_clear_width_m"]["status"],
                "manual_site_measurement",
            )

    def test_narrow_door_blocks_candidate_geometry(self) -> None:
        manifest = self.complete_manifest()
        manifest["manual_measurements"]["principal_door"]["clear_width_m"] = 0.90
        checks = self.preparer.measurement_checks(manifest["manual_measurements"])
        self.assertTrue(all(checks.values()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "administration.obj").write_text(
                "v 0 0 0\nv 22 0 0\nv 22 10 3\nv 0 10 3\nf 1 2 3 4\n",
                encoding="utf-8",
            )
            manifest_path = root / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
            report_path = root / "report.json"
            overlay_path = root / "overlay.yaml"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PREPARER),
                    "--manifest",
                    str(manifest_path),
                    "--scan-root",
                    str(root),
                    "--output",
                    str(report_path),
                    "--overlay-output",
                    str(overlay_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(result.returncode, 2)
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "capture_complete_route_geometry_rejected")
            self.assertFalse(
                report["route_geometry_gate"]["checks"][
                    "principal_door_accepts_padded_footprint"
                ]
            )

    def test_deep_merge_preserves_unmeasured_base_geometry(self) -> None:
        base = {"plan_geometry": {"wall_thickness_m": 0.15, "ceiling_height_m": 3.0}}
        overlay = {"plan_geometry": {"ceiling_height_m": 2.85}}
        original = copy.deepcopy(base)
        merged = self.builder.deep_merge(base, overlay)
        self.assertEqual(base, original)
        self.assertEqual(merged["plan_geometry"]["wall_thickness_m"], 0.15)
        self.assertEqual(merged["plan_geometry"]["ceiling_height_m"], 2.85)

    def test_tight_door_overlay_is_accepted_only_as_simulation_candidate(self) -> None:
        overlay = self.builder.load_measured_overlay(PRESENTATION_OVERLAY)
        self.assertEqual(overlay["status"], "measured_site_presentation_candidate")
        self.assertFalse(overlay["candidate_route_geometry_valid"])
        self.assertTrue(overlay["candidate_simulation_route_geometry_valid"])
        self.assertTrue(overlay["presentation_clearance_profile"]["simulation_only"])
        self.assertIs(overlay["physical_release"], False)
        self.assertEqual(
            overlay["plan_geometry"]["atrium"]["central_polygon"]["step_down_m"],
            0.20,
        )
        self.assertEqual(
            overlay["capture_limitations"]["vice_principal_office_interior"]["status"],
            "not_captured_locked_during_site_visit",
        )

    def test_tight_door_overlay_cannot_claim_production_route_gate(self) -> None:
        overlay = yaml.safe_load(PRESENTATION_OVERLAY.read_text(encoding="utf-8"))
        overlay["candidate_route_geometry_valid"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not claim the production route gate"):
                self.builder.load_measured_overlay(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
