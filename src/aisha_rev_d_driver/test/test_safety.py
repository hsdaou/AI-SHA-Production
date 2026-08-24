from __future__ import annotations

import pytest

from aisha_rev_d_driver.odometry import DifferentialGeometry
from aisha_rev_d_driver.safety import DryRunCommandGate


def make_gate() -> DryRunCommandGate:
    return DryRunCommandGate(
        DifferentialGeometry(0.1, 0.72, 16384),
        max_wheel_rpm=5.0,
        timeout_s=0.2,
        reverse_enabled=False,
    )


def test_wheels_lifted_dry_run_is_clamped_to_five_rpm() -> None:
    command = make_gate().accept(0.0, linear_mps=0.8, angular_rad_s=0.0)
    assert command.clamped
    assert command.left_rpm == pytest.approx(5.0)
    assert command.right_rpm == pytest.approx(5.0)


def test_timeout_fails_to_zero() -> None:
    gate = make_gate()
    gate.accept(1.0, linear_mps=0.02, angular_rad_s=0.0)
    assert not gate.sample(1.199).timed_out
    stopped = gate.sample(1.2)
    assert stopped.timed_out
    assert stopped.left_rpm == stopped.right_rpm == 0.0


def test_reverse_and_lateral_commands_are_rejected() -> None:
    gate = make_gate()
    with pytest.raises(ValueError, match="reverse"):
        gate.accept(0.0, linear_mps=-0.01, angular_rad_s=0.0)
    with pytest.raises(ValueError, match="lateral"):
        gate.accept(0.0, linear_mps=0.0, angular_rad_s=0.0, lateral_mps=0.01)
