"""Offline command mapping with a fail-safe timeout and no hardware output."""

from __future__ import annotations

from dataclasses import dataclass

from .odometry import DifferentialGeometry


@dataclass(frozen=True)
class WheelCommand:
    stamp_s: float
    left_rpm: float
    right_rpm: float
    timed_out: bool
    clamped: bool


class DryRunCommandGate:
    """Validate Rev D command math without exposing a motor transport."""

    def __init__(
        self,
        geometry: DifferentialGeometry,
        *,
        max_wheel_rpm: float = 5.0,
        timeout_s: float = 0.20,
        reverse_enabled: bool = False,
    ) -> None:
        if max_wheel_rpm <= 0.0:
            raise ValueError("maximum wheel RPM must be positive")
        if timeout_s <= 0.0:
            raise ValueError("command timeout must be positive")
        self.geometry = geometry
        self.max_wheel_rpm = max_wheel_rpm
        self.timeout_s = timeout_s
        self.reverse_enabled = reverse_enabled
        self._last_stamp_s: float | None = None
        self._last_left_rpm = 0.0
        self._last_right_rpm = 0.0

    def accept(
        self,
        stamp_s: float,
        *,
        linear_mps: float,
        angular_rad_s: float,
        lateral_mps: float = 0.0,
    ) -> WheelCommand:
        if abs(lateral_mps) > 1e-9:
            raise ValueError("Rev D differential drive rejects lateral commands")
        if linear_mps < 0.0 and not self.reverse_enabled:
            raise ValueError("reverse is disabled by the Rev D occupied-area policy")
        left_rpm, right_rpm = self.geometry.body_to_wheel_rpm(linear_mps, angular_rad_s)
        peak = max(abs(left_rpm), abs(right_rpm))
        clamped = peak > self.max_wheel_rpm
        if clamped:
            scale = self.max_wheel_rpm / peak
            left_rpm *= scale
            right_rpm *= scale
        self._last_stamp_s = stamp_s
        self._last_left_rpm = left_rpm
        self._last_right_rpm = right_rpm
        return WheelCommand(stamp_s, left_rpm, right_rpm, False, clamped)

    def sample(self, stamp_s: float) -> WheelCommand:
        timed_out = (
            self._last_stamp_s is None
            or stamp_s - self._last_stamp_s >= self.timeout_s - 1e-12
            or stamp_s < self._last_stamp_s
        )
        if timed_out:
            return WheelCommand(stamp_s, 0.0, 0.0, True, False)
        return WheelCommand(
            stamp_s,
            self._last_left_rpm,
            self._last_right_rpm,
            False,
            False,
        )
