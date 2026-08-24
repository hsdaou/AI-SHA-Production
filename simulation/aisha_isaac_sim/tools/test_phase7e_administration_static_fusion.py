from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def load_builder():
    path = TOOLS / "build_administration_static_fused_nav2_profile.py"
    spec = importlib.util.spec_from_file_location("phase7e_profile_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def obstacle_layers(config: dict) -> list[dict]:
    return [
        config[name][name]["ros__parameters"]["obstacle_layer"]
        for name in ("local_costmap", "global_costmap")
    ]


def test_phase7e_profile_splits_clearing_and_dynamic_marking_without_other_changes() -> None:
    builder = load_builder()
    base = yaml.safe_load(
        (ROOT / "config/nav2_administration_native_costmap_params.yaml").read_text()
    )
    fused = builder.fuse_observation_sources(copy.deepcopy(base))
    restored = copy.deepcopy(fused)
    for layer in obstacle_layers(restored):
        crown = layer.pop("crown_clear")
        front = layer.pop("front_clear")
        crown["marking"] = True
        front["marking"] = True
        layer.pop("crown_dynamic")
        layer.pop("front_dynamic")
        layer["observation_sources"] = "crown_scan front_scan"
        layer["crown_scan"] = crown
        layer["front_scan"] = front
    assert restored == base


def test_phase7e_fused_sources_retain_heights_and_separate_authority() -> None:
    builder = load_builder()
    base = yaml.safe_load(
        (ROOT / "config/nav2_administration_native_costmap_params.yaml").read_text()
    )
    fused = builder.fuse_observation_sources(copy.deepcopy(base))
    for layer in obstacle_layers(fused):
        assert layer["observation_sources"] == (
            "crown_clear front_clear crown_dynamic front_dynamic"
        )
        assert layer["crown_clear"]["max_obstacle_height"] == 2.20
        assert layer["front_clear"]["max_obstacle_height"] == 2.00
        assert layer["crown_clear"]["inf_is_valid"] is True
        assert layer["front_clear"]["inf_is_valid"] is True
        assert layer["crown_clear"]["marking"] is False
        assert layer["front_clear"]["marking"] is False
        assert layer["crown_dynamic"]["clearing"] is False
        assert layer["front_dynamic"]["clearing"] is False


def test_phase7e_runtime_wires_distinct_full_office_profile() -> None:
    registrations = (
        ROOT / "isaaclab/aisha_isaaclab/tasks/office_nav/__init__.py"
    ).read_text()
    bridge = (TOOLS / "run_administration_nav2_phase7e_static_fusion_bridge.sh").read_text()
    mission = (TOOLS / "run_administration_nav2_phase7e_static_fusion_mission.sh").read_text()
    servers = (TOOLS / "run_administration_nav2_phase7e_static_fusion_servers.sh").read_text()
    assert '"Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7E-"' in registrations
    assert '"StaticFusion-FullOffice-Safety-Direct-v0"' in registrations
    assert "Phase7E-StaticFusion-FullOffice-Safety-Direct-v0" in bridge
    assert "phase7e_static_fusion_full_office_safety" in mission
    assert "stop-after-waypoint" not in mission
    assert "filter_administration_static_scan_returns.py" in servers
    assert "build_administration_static_fused_nav2_profile.py" in servers


def test_phase7e_filter_preserves_static_and_live_obstacle_authorities() -> None:
    source = (TOOLS / "filter_administration_static_scan_returns.py").read_text()
    assert '"raw_scan_role": "native_costmap_clearing_only"' in source
    assert '"filtered_scan_role": "native_costmap_dynamic_marking_only"' in source
    assert '"static_map_role": "mapped_static_obstacle_authority"' in source
    assert 'filtered_ranges[index] = math.nan' in source
    assert '"mapped_free_obstacle_returns_preserved": True' in source


def test_checked_in_phase7e_gate_is_fully_accepted() -> None:
    gate = json.loads(
        (
            ROOT
            / "results/administration_nav2_phase7e_static_fusion_integration_gate.json"
        ).read_text()
    )
    assert gate["passed"] is True
    assert gate["checks_passed"] == gate["checks_total"]
    assert gate["mission"]["completed_legs"] == gate["mission"]["expected_legs"] == 12
