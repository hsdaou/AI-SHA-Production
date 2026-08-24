"""Static and checked-evidence tests for the Phase 7D administration gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def obstacle_layers(config: dict) -> list[dict]:
    return [
        config[name][name]["ros__parameters"]["obstacle_layer"]
        for name in ("local_costmap", "global_costmap")
    ]


def test_phase7d_profile_adds_only_height_and_no_return_clearing_fields() -> None:
    frozen = load_yaml("nav2_sim_tight_door_params.yaml")
    phase7d = load_yaml("nav2_administration_native_costmap_params.yaml")
    stripped = copy.deepcopy(phase7d)
    for layer in obstacle_layers(stripped):
        layer["crown_scan"].pop("max_obstacle_height")
        layer["front_scan"].pop("max_obstacle_height")
        layer["crown_scan"].pop("inf_is_valid")
        layer["front_scan"].pop("inf_is_valid")
    assert stripped == frozen


def test_both_costmaps_accept_the_actual_lidar_mount_heights() -> None:
    phase7d = load_yaml("nav2_administration_native_costmap_params.yaml")
    for layer in obstacle_layers(phase7d):
        assert layer["observation_sources"] == "crown_scan front_scan"
        assert layer["crown_scan"]["max_obstacle_height"] == 2.2
        assert layer["front_scan"]["max_obstacle_height"] == 2.0
        assert layer["crown_scan"]["inf_is_valid"] is True
        assert layer["front_scan"]["inf_is_valid"] is True


def test_reverse_and_lateral_motion_remain_disabled() -> None:
    phase7d = load_yaml("nav2_administration_native_costmap_params.yaml")
    planner = phase7d["controller_server"]["ros__parameters"]["FollowPath"]
    smoother = phase7d["velocity_smoother"]["ros__parameters"]
    assert planner["min_vel_x"] == 0.0
    assert planner["min_vel_y"] == planner["max_vel_y"] == 0.0
    assert smoother["min_velocity"][0] == 0.0


def test_phase7d_runtime_wires_a_distinct_task_profile_and_control_stack() -> None:
    registrations = (
        ROOT
        / "isaaclab/aisha_isaaclab/tasks/office_nav/__init__.py"
    ).read_text(encoding="utf-8")
    servers = (
        ROOT / "tools/run_administration_nav2_phase7d_native_costmap_servers.sh"
    ).read_text(encoding="utf-8")
    mission = (
        ROOT / "tools/run_administration_nav2_phase7d_native_costmap_mission.sh"
    ).read_text(encoding="utf-8")
    assert '"Isaac-AISHA-Administration-Live-Measured-Nav2-Phase7D-"' in registrations
    assert '"NativeCostmap-SafeWait-Safety-Direct-v0"' in registrations
    assert "nav2_administration_native_costmap_params.yaml" in servers
    assert "phase7d_native_costmap_safe_wait_safety" in mission


def test_checked_in_phase7d_gate_is_fully_accepted() -> None:
    report = json.loads(
        (
            ROOT
            / "results/administration_nav2_phase7d_native_costmap_integration_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["checks_passed"] == report["checks_total"]
