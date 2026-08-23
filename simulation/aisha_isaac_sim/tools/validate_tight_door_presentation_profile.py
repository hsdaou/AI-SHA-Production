#!/usr/bin/env python3
"""Validate separation and arithmetic of the measured tight-door profile."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SIM_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def close(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, abs_tol=1.0e-9
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production", type=Path, default=SIM_ROOT / "config" / "nav2_sim_params.yaml"
    )
    parser.add_argument(
        "--tight",
        type=Path,
        default=SIM_ROOT / "config" / "nav2_sim_tight_door_params.yaml",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=SIM_ROOT / "config" / "measured_administration_presentation_2026-08-23.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SIM_ROOT / "results" / "tight_door_presentation_profile_validation.json",
    )
    args = parser.parse_args()

    production = load_yaml(args.production)
    tight = load_yaml(args.tight)
    overlay = load_yaml(args.overlay)
    production_local = production["local_costmap"]["local_costmap"]["ros__parameters"]
    production_global = production["global_costmap"]["global_costmap"]["ros__parameters"]
    tight_local = tight["local_costmap"]["local_costmap"]["ros__parameters"]
    tight_global = tight["global_costmap"]["global_costmap"]["ros__parameters"]
    tight_controller = tight["controller_server"]["ros__parameters"]["FollowPath"]
    tight_smoother = tight["velocity_smoother"]["ros__parameters"]
    profile = overlay["presentation_clearance_profile"]
    checks = {
        "production_local_padding_unchanged": close(production_local["footprint_padding"], 0.08),
        "production_global_padding_unchanged": close(production_global["footprint_padding"], 0.08),
        "tight_local_padding_is_0_030_m": close(tight_local["footprint_padding"], 0.03),
        "tight_global_padding_is_0_030_m": close(tight_global["footprint_padding"], 0.03),
        "tight_speed_is_0_10_mps": close(tight_controller["max_vel_x"], 0.10)
        and close(tight_controller["max_speed_xy"], 0.10)
        and close(tight_smoother["max_velocity"][0], 0.10),
        "tight_acceleration_is_0_15_mps2": close(tight_controller["acc_lim_x"], 0.15)
        and close(tight_controller["decel_lim_x"], -0.15)
        and close(tight_smoother["max_accel"][0], 0.15)
        and close(tight_smoother["max_decel"][0], -0.15),
        "padded_width_is_0_828_m": close(profile["padded_transit_width_m"], 0.828),
        "measured_width_is_0_850_m": close(profile["measured_narrowest_clear_width_m"], 0.85),
        "nominal_total_margin_is_0_022_m": close(profile["nominal_total_margin_m"], 0.022),
        "straight_approach_required": profile["straight_centreline_approach_required"] is True,
        "doorway_rotation_forbidden": profile["rotation_in_doorway_forbidden"] is True,
        "simulation_only": profile["simulation_only"] is True,
        "production_route_gate_false": overlay["candidate_route_geometry_valid"] is False,
        "physical_release_false": overlay["physical_release"] is False,
    }
    report = {
        "report_type": "tight_door_presentation_profile_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "production_params": str(args.production.resolve()),
        "tight_door_params": str(args.tight.resolve()),
        "measured_overlay": str(args.overlay.resolve()),
        "claim_boundary": (
            "Passing validates only the separate simulation parameter profile and clearance "
            "arithmetic. It does not prove a collision-free run or physical safe passage."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
