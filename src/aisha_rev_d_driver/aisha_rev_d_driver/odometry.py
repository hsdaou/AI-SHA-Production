"""Pure differential-drive kinematics and encoder odometry for AI-SHA Rev D."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DifferentialGeometry:
    wheel_radius_m: float
    wheel_track_m: float
    encoder_counts_per_rev: int
    left_encoder_sign: int = 1
    right_encoder_sign: int = 1

    def __post_init__(self) -> None:
        if self.wheel_radius_m <= 0.0:
            raise ValueError("wheel radius must be positive")
        if self.wheel_track_m <= 0.0:
            raise ValueError("wheel track must be positive")
        if self.encoder_counts_per_rev <= 0:
            raise ValueError("encoder counts per revolution must be positive")
        if self.left_encoder_sign not in (-1, 1):
            raise ValueError("left encoder sign must be -1 or +1")
        if self.right_encoder_sign not in (-1, 1):
            raise ValueError("right encoder sign must be -1 or +1")

    @property
    def metres_per_count(self) -> float:
        return 2.0 * math.pi * self.wheel_radius_m / self.encoder_counts_per_rev

    def body_to_wheel_rpm(self, linear_mps: float, angular_rad_s: float) -> tuple[float, float]:
        left_mps = linear_mps - 0.5 * angular_rad_s * self.wheel_track_m
        right_mps = linear_mps + 0.5 * angular_rad_s * self.wheel_track_m
        scale = 60.0 / (2.0 * math.pi * self.wheel_radius_m)
        return left_mps * scale, right_mps * scale

    def wheel_rpm_to_body(self, left_rpm: float, right_rpm: float) -> tuple[float, float]:
        scale = 2.0 * math.pi * self.wheel_radius_m / 60.0
        left_mps = left_rpm * scale
        right_mps = right_rpm * scale
        return (
            0.5 * (left_mps + right_mps),
            (right_mps - left_mps) / self.wheel_track_m,
        )


@dataclass(frozen=True)
class Pose2D:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class OdometryEstimate:
    stamp_s: float
    pose: Pose2D
    linear_mps: float
    angular_rad_s: float
    delta_left_counts: int
    delta_right_counts: int


def signed_int32_delta(current: int, previous: int) -> int:
    """Return a rollover-safe signed delta for the driver's int32 position."""

    return ((current - previous + 0x8000_0000) & 0xFFFF_FFFF) - 0x8000_0000


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class EncoderOdometry:
    def __init__(self, geometry: DifferentialGeometry) -> None:
        self.geometry = geometry
        self.pose = Pose2D()
        self._previous_stamp_s: float | None = None
        self._previous_left: int | None = None
        self._previous_right: int | None = None

    def reset(self, pose: Pose2D | None = None) -> None:
        self.pose = pose or Pose2D()
        self._previous_stamp_s = None
        self._previous_left = None
        self._previous_right = None

    def update(self, stamp_s: float, left_count: int, right_count: int) -> OdometryEstimate | None:
        if self._previous_stamp_s is None:
            self._previous_stamp_s = stamp_s
            self._previous_left = left_count
            self._previous_right = right_count
            return None
        dt = stamp_s - self._previous_stamp_s
        if dt <= 0.0:
            raise ValueError("encoder timestamps must increase monotonically")
        assert self._previous_left is not None
        assert self._previous_right is not None
        delta_left_counts = (
            signed_int32_delta(left_count, self._previous_left)
            * self.geometry.left_encoder_sign
        )
        delta_right_counts = (
            signed_int32_delta(right_count, self._previous_right)
            * self.geometry.right_encoder_sign
        )
        distance_left = delta_left_counts * self.geometry.metres_per_count
        distance_right = delta_right_counts * self.geometry.metres_per_count
        distance = 0.5 * (distance_left + distance_right)
        delta_yaw = (distance_right - distance_left) / self.geometry.wheel_track_m
        midpoint_yaw = self.pose.yaw_rad + 0.5 * delta_yaw
        self.pose = Pose2D(
            x_m=self.pose.x_m + distance * math.cos(midpoint_yaw),
            y_m=self.pose.y_m + distance * math.sin(midpoint_yaw),
            yaw_rad=normalize_angle(self.pose.yaw_rad + delta_yaw),
        )
        self._previous_stamp_s = stamp_s
        self._previous_left = left_count
        self._previous_right = right_count
        return OdometryEstimate(
            stamp_s=stamp_s,
            pose=self.pose,
            linear_mps=distance / dt,
            angular_rad_s=delta_yaw / dt,
            delta_left_counts=delta_left_counts,
            delta_right_counts=delta_right_counts,
        )
