#!/usr/bin/env python3
"""Derive the administration static-map/live-LiDAR fusion Nav2 profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PACKAGE_ROOT / "config" / "nav2_administration_native_costmap_params.yaml"


def fuse_observation_sources(profile: dict) -> dict:
    """Split raw clearing rays from map-filtered dynamic marking returns."""
    for costmap_name in ("local_costmap", "global_costmap"):
        obstacle_layer = profile[costmap_name][costmap_name]["ros__parameters"][
            "obstacle_layer"
        ]
        crown = dict(obstacle_layer.pop("crown_scan"))
        front = dict(obstacle_layer.pop("front_scan"))
        obstacle_layer["observation_sources"] = (
            "crown_clear front_clear crown_dynamic front_dynamic"
        )
        for source_name, source, filtered_topic in (
            ("crown", crown, "/aisha/static_fused/scan_dynamic"),
            ("front", front, "/aisha/static_fused/front_scan_dynamic"),
        ):
            clear_source = dict(source)
            clear_source["marking"] = False
            clear_source["clearing"] = True
            dynamic_source = dict(source)
            dynamic_source["topic"] = filtered_topic
            dynamic_source["marking"] = True
            dynamic_source["clearing"] = False
            # No-return beams are useful to the raw clearing source only.
            dynamic_source["inf_is_valid"] = False
            obstacle_layer[f"{source_name}_clear"] = clear_source
            obstacle_layer[f"{source_name}_dynamic"] = dynamic_source
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    fused = fuse_observation_sources(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(fused, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
