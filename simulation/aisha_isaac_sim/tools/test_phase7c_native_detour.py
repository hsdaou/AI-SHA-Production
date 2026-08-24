"""Unit checks for the isolated Phase 7C map and Nav2 profile."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_pgm() -> tuple[int, int, bytes]:
    data = (
        ROOT
        / "maps/phase7c_native_detour_loop/phase7c_native_detour_loop.pgm"
    ).read_bytes()
    magic, dimensions, maximum, pixels = data.split(b"\n", 3)
    assert magic == b"P5"
    width, height = (int(value) for value in dimensions.split())
    assert maximum == b"255"
    return width, height, pixels


def pixel_at_world(x_m: float, y_m: float) -> int:
    width, height, pixels = load_pgm()
    cell_x = int(round(x_m / 0.05))
    cell_y = int(round((y_m + 4.0) / 0.05))
    image_y = height - 1 - cell_y
    return pixels[image_y * width + cell_x]


def test_loop_map_has_two_free_branches_and_a_solid_island() -> None:
    assert pixel_at_world(6.0, 1.90) == 254
    assert pixel_at_world(6.0, -1.90) == 254
    assert pixel_at_world(6.0, 0.0) == 0


def test_dynamic_blocker_is_not_baked_into_static_map() -> None:
    assert pixel_at_world(5.80, 2.10) == 254


def test_nav2_observation_sources_declare_nonzero_height_limits() -> None:
    config = yaml.safe_load(
        (ROOT / "config/nav2_phase7c_native_detour_params.yaml").read_text(
            encoding="utf-8"
        )
    )
    for costmap_name in ("local_costmap", "global_costmap"):
        obstacle = config[costmap_name][costmap_name]["ros__parameters"][
            "obstacle_layer"
        ]
        assert obstacle["front_scan"]["max_obstacle_height"] == 2.0
        assert obstacle["crown_scan"]["max_obstacle_height"] == 2.2


def test_checked_in_phase7c_gate_is_fully_accepted() -> None:
    report = json.loads(
        (
            ROOT / "results/phase7c_native_costmap_detour_integration_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["checks_passed"] == report["checks_total"] == 29
