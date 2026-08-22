#!/usr/bin/env python3
"""Shared, simulator-independent helpers for the AI-SHA Isaac workflow."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
RESULTS_DIR = PACKAGE_ROOT / "results"
SCENES_DIR = PACKAGE_ROOT / "scenes"
URDF_DIR = PACKAGE_ROOT / "urdf"
USD_DIR = PACKAGE_ROOT / "usd"


def ensure_output_dirs() -> None:
    for path in (
        RESULTS_DIR,
        SCENES_DIR,
        USD_DIR,
        PACKAGE_ROOT / "media" / "screenshots",
        PACKAGE_ROOT / "media" / "route_frames",
        PACKAGE_ROOT / "media" / "learned_route_replay_frames",
        PACKAGE_ROOT / "media" / "videos",
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def wheel_speeds_rad_s(linear_mps: float, angular_rad_s: float, radius_m: float, track_m: float) -> tuple[float, float]:
    """Return left/right wheel velocity for +X forward, +Y left, +Z up."""
    if radius_m <= 0.0 or track_m <= 0.0:
        raise ValueError("wheel radius and track must be positive")
    left = (linear_mps - angular_rad_s * track_m / 2.0) / radius_m
    right = (linear_mps + angular_rad_s * track_m / 2.0) / radius_m
    return left, right


def clamp_delta(previous: float, requested: float, maximum_rate: float, dt: float) -> float:
    if maximum_rate < 0.0 or dt <= 0.0:
        raise ValueError("maximum_rate must be non-negative and dt must be positive")
    delta = maximum_rate * dt
    return max(previous - delta, min(previous + delta, requested))


class DifferentialDriveLimiter:
    """Deterministic speed/acceleration limiter with a latched command watchdog."""

    def __init__(
        self,
        *,
        wheel_radius_m: float,
        wheel_track_m: float,
        max_linear_mps: float,
        max_angular_rad_s: float,
        max_acceleration_mps2: float,
        max_angular_acceleration_rad_s2: float,
        watchdog_timeout_s: float,
    ) -> None:
        self.radius = wheel_radius_m
        self.track = wheel_track_m
        self.max_linear = max_linear_mps
        self.max_angular = max_angular_rad_s
        self.max_acceleration = max_acceleration_mps2
        self.max_angular_acceleration = max_angular_acceleration_rad_s2
        self.watchdog_timeout = watchdog_timeout_s
        self.linear = 0.0
        self.angular = 0.0
        self.last_command_s: float | None = None
        self.stop_latched = False
        self.stop_reason: str | None = None

    def command(self, linear_mps: float, angular_rad_s: float, now_s: float) -> None:
        if not math.isfinite(linear_mps) or not math.isfinite(angular_rad_s) or not math.isfinite(now_s):
            raise ValueError("commands and time must be finite")
        if self.stop_latched:
            return
        self.requested_linear = max(-self.max_linear, min(self.max_linear, linear_mps))
        self.requested_angular = max(-self.max_angular, min(self.max_angular, angular_rad_s))
        self.last_command_s = now_s

    def update(self, now_s: float, dt: float) -> tuple[float, float]:
        stale = self.last_command_s is None or now_s - self.last_command_s > self.watchdog_timeout
        if stale:
            self.stop_latched = True
            self.stop_reason = "command_watchdog"
        if self.stop_latched:
            self.linear = 0.0
            self.angular = 0.0
        else:
            self.linear = clamp_delta(self.linear, self.requested_linear, self.max_acceleration, dt)
            self.angular = clamp_delta(self.angular, self.requested_angular, self.max_angular_acceleration, dt)
        return wheel_speeds_rad_s(self.linear, self.angular, self.radius, self.track)

    def reset(self, now_s: float) -> None:
        self.linear = 0.0
        self.angular = 0.0
        self.last_command_s = now_s
        self.requested_linear = 0.0
        self.requested_angular = 0.0
        self.stop_latched = False
        self.stop_reason = None
