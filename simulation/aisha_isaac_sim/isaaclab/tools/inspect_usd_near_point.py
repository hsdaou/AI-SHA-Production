#!/usr/bin/env python3
"""List composed USD geometry whose world bounds are near an XY point."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd", type=Path)
parser.add_argument("--xy", type=float, nargs=2, required=True)
parser.add_argument("--radius", type=float, default=2.0)
parser.add_argument("--collision-only", action="store_true")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def main() -> int:
    stage = Usd.Stage.Open(str(args.usd.expanduser().resolve()))
    if stage is None:
        raise RuntimeError(f"could not open {args.usd}")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    query_x, query_y = args.xy
    matches: list[tuple[float, str, tuple[float, ...], tuple[float, ...]]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Boundable):
            continue
        if args.collision_only and not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = world_range.GetMin()
        maximum = world_range.GetMax()
        dx = max(float(minimum[0]) - query_x, 0.0, query_x - float(maximum[0]))
        dy = max(float(minimum[1]) - query_y, 0.0, query_y - float(maximum[1]))
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= args.radius:
            matches.append(
                (
                    distance,
                    str(prim.GetPath()),
                    tuple(round(float(value), 4) for value in minimum),
                    tuple(round(float(value), 4) for value in maximum),
                )
            )

    for distance, path, minimum, maximum in sorted(matches):
        print(f"NEAR distance_xy_m={distance:.4f} min={minimum} max={maximum} prim={path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
