#!/usr/bin/env python3
"""Generate the compact two-route Phase 7C occupancy map."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "maps" / "phase7c_native_detour_loop"
RESOLUTION_M = 0.05
ORIGIN_X_M = 0.0
ORIGIN_Y_M = -4.0
WIDTH = 240
HEIGHT = 160


def world_to_cell(x_m: float, y_m: float) -> tuple[int, int]:
    return (
        int(round((x_m - ORIGIN_X_M) / RESOLUTION_M)),
        int(round((y_m - ORIGIN_Y_M) / RESOLUTION_M)),
    )


def fill_world_rectangle(
    pixels: bytearray,
    *,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    value: int = 0,
) -> None:
    cell_x_min, cell_y_min = world_to_cell(x_min_m, y_min_m)
    cell_x_max, cell_y_max = world_to_cell(x_max_m, y_max_m)
    for cell_y in range(max(0, cell_y_min), min(HEIGHT, cell_y_max + 1)):
        image_y = HEIGHT - 1 - cell_y
        row = image_y * WIDTH
        for cell_x in range(max(0, cell_x_min), min(WIDTH, cell_x_max + 1)):
            pixels[row + cell_x] = value


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pixels = bytearray([254] * (WIDTH * HEIGHT))

    # Closed 11.2 m x 6.0 m envelope. The temporary obstacle is deliberately
    # absent: it must enter Nav2 only through the live Isaac LiDAR.
    fill_world_rectangle(pixels, x_min_m=0.0, x_max_m=0.50, y_min_m=-4.0, y_max_m=4.0)
    fill_world_rectangle(pixels, x_min_m=11.50, x_max_m=12.0, y_min_m=-4.0, y_max_m=4.0)
    fill_world_rectangle(pixels, x_min_m=0.0, x_max_m=12.0, y_min_m=-4.0, y_max_m=-3.00)
    fill_world_rectangle(pixels, x_min_m=0.0, x_max_m=12.0, y_min_m=3.00, y_max_m=4.0)

    # A central island produces two genuine, footprint-feasible branches.
    fill_world_rectangle(pixels, x_min_m=4.50, x_max_m=7.50, y_min_m=-1.20, y_max_m=1.20)

    pgm_path = OUTPUT_DIR / "phase7c_native_detour_loop.pgm"
    yaml_path = OUTPUT_DIR / "phase7c_native_detour_loop.yaml"
    pgm_path.write_bytes(
        f"P5\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + bytes(pixels)
    )
    yaml_path.write_text(
        "\n".join(
            (
                "image: phase7c_native_detour_loop.pgm",
                "mode: trinary",
                f"resolution: {RESOLUTION_M}",
                f"origin: [{ORIGIN_X_M}, {ORIGIN_Y_M}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"generated {pgm_path} and {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
