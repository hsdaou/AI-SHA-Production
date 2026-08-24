from __future__ import annotations

import math

import pytest

from aisha_rev_d_driver.odometry import (
    DifferentialGeometry,
    EncoderOdometry,
    signed_int32_delta,
)


def test_straight_one_revolution() -> None:
    geometry = DifferentialGeometry(0.1, 0.72, 1000)
    odometry = EncoderOdometry(geometry)
    assert odometry.update(0.0, 0, 0) is None
    estimate = odometry.update(1.0, 1000, 1000)
    assert estimate is not None
    assert estimate.pose.x_m == pytest.approx(2.0 * math.pi * 0.1)
    assert estimate.pose.y_m == pytest.approx(0.0)
    assert estimate.pose.yaw_rad == pytest.approx(0.0)


def test_positive_ninety_degree_pivot() -> None:
    # With r=0.1, track=0.72 and 1000 CPR, +/-900 counts is exactly pi/2.
    geometry = DifferentialGeometry(0.1, 0.72, 1000)
    odometry = EncoderOdometry(geometry)
    odometry.update(0.0, 0, 0)
    estimate = odometry.update(1.0, -900, 900)
    assert estimate is not None
    assert estimate.pose.x_m == pytest.approx(0.0, abs=1e-12)
    assert estimate.pose.y_m == pytest.approx(0.0, abs=1e-12)
    assert estimate.pose.yaw_rad == pytest.approx(math.pi / 2.0)


def test_signed_int32_rollover_is_continuous() -> None:
    assert signed_int32_delta(-0x80000000, 0x7FFFFFFF) == 1
    assert signed_int32_delta(0x7FFFFFFF, -0x80000000) == -1


def test_encoder_signs_are_explicit() -> None:
    geometry = DifferentialGeometry(0.1, 0.72, 1000, left_encoder_sign=-1)
    odometry = EncoderOdometry(geometry)
    odometry.update(0.0, 0, 0)
    estimate = odometry.update(1.0, -100, 100)
    assert estimate is not None
    assert estimate.pose.x_m > 0.0
    assert estimate.pose.yaw_rad == pytest.approx(0.0)
