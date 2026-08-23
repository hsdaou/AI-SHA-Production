"""Deterministic mapped-site guards for the measured Nav2 presentation.

This module is deliberately independent of ROS and Isaac APIs so its geometry
and state transitions can be unit tested.  It is a simulation presentation
guard, not a safety-rated controller or a substitute for physical sensing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class Doorway:
    name: str
    centre_x_m: float
    centre_y_m: float
    tangent_x: float
    tangent_y: float
    clear_width_m: float

    @property
    def normal(self) -> tuple[float, float]:
        return -self.tangent_y, self.tangent_x


@dataclass(frozen=True)
class GuardResult:
    linear_mps: float
    angular_rad_s: float
    active_door: str | None
    doorway_slow_active: bool
    doorway_alignment_active: bool
    doorway_alignment_hold: bool
    doorway_overspeed_stop: bool
    polygon_no_go_stop: bool
    yaw_error_rad: float
    tangent_distance_m: float | None
    normal_distance_m: float | None
    polygon_footprint_clearance_m: float


class MappedNav2SafetyGuard:
    """Apply the measured doorway and central-drop presentation invariants."""

    body_circumscribed_radius_m = math.hypot(0.725, 0.384)

    def __init__(
        self,
        doors: list[Doorway],
        *,
        maximum_doorway_speed_mps: float = 0.10,
        target_doorway_speed_mps: float = 0.08,
        overspeed_stop_mps: float = 0.095,
        traction_command_ceiling_mps: float = 0.18,
        traction_gain: float = 1.5,
        slow_zone_normal_half_extent_m: float = 1.05,
        alignment_zone_normal_half_extent_m: float = 1.60,
        no_rotation_normal_half_extent_m: float = 0.80,
        tangent_half_extent_m: float = 1.10,
        alignment_tolerance_rad: float = math.radians(0.75),
        approach_alignment_tolerance_rad: float = math.radians(3.0),
        approach_stage_capture_radius_m: float = 0.018,
        yaw_settle_rate_rad_s: float = 0.05,
        heading_gain: float = 4.0,
        yaw_damping_gain: float = 2.0,
        maximum_angular_rad_s: float = 0.55,
        breakaway_angular_rad_s: float = 0.30,
        polygon_outer_radius_m: float = 2.30,
        polygon_minimum_clearance_m: float = 0.05,
        polygon_prediction_horizon_s: float = 1.0,
    ) -> None:
        if not doors:
            raise ValueError("at least one mapped doorway is required")
        if target_doorway_speed_mps > maximum_doorway_speed_mps:
            raise ValueError("target doorway speed exceeds the declared maximum")
        self.doors = tuple(doors)
        self.maximum_doorway_speed_mps = maximum_doorway_speed_mps
        self.target_doorway_speed_mps = target_doorway_speed_mps
        self.overspeed_stop_mps = overspeed_stop_mps
        self.traction_command_ceiling_mps = traction_command_ceiling_mps
        self.traction_gain = traction_gain
        self.slow_zone_normal_half_extent_m = slow_zone_normal_half_extent_m
        self.alignment_zone_normal_half_extent_m = alignment_zone_normal_half_extent_m
        self.no_rotation_normal_half_extent_m = no_rotation_normal_half_extent_m
        self.tangent_half_extent_m = tangent_half_extent_m
        self.alignment_tolerance_rad = alignment_tolerance_rad
        self.approach_alignment_tolerance_rad = approach_alignment_tolerance_rad
        self.approach_stage_capture_radius_m = approach_stage_capture_radius_m
        self.yaw_settle_rate_rad_s = yaw_settle_rate_rad_s
        self.heading_gain = heading_gain
        self.yaw_damping_gain = yaw_damping_gain
        self.maximum_angular_rad_s = maximum_angular_rad_s
        self.breakaway_angular_rad_s = breakaway_angular_rad_s
        self.polygon_outer_radius_m = polygon_outer_radius_m
        self.polygon_minimum_clearance_m = polygon_minimum_clearance_m
        self.polygon_prediction_horizon_s = polygon_prediction_horizon_s
        self._active_door_index: int | None = None
        self._crossing_sign = 0.0
        self.statistics: dict[str, Any] = {
            "steps": 0,
            "doorway_active_steps": 0,
            "doorway_slow_steps": 0,
            "doorway_alignment_steps": 0,
            "doorway_alignment_hold_steps": 0,
            "doorway_overspeed_stops": 0,
            "doorway_direction_rearms": 0,
            "polygon_no_go_stops": 0,
            "doorway_entries": {door.name: 0 for door in self.doors},
            "maximum_abs_speed_in_doorway_mps": 0.0,
            "maximum_abs_tangent_offset_in_doorway_m": 0.0,
            "minimum_polygon_full_footprint_clearance_m": math.inf,
        }

    @classmethod
    def from_site_configs(
        cls, site: dict[str, Any], measured_overlay: dict[str, Any]
    ) -> "MappedNav2SafetyGuard":
        doors = []
        for name in ("vice_principal", "principal"):
            geometry = site["doors"][name]
            measured = measured_overlay["doors"][name]
            angle = math.radians(float(geometry["wall_rotation_deg"]))
            centre = geometry["centre_xy_m"]
            doors.append(
                Doorway(
                    name=name,
                    centre_x_m=float(centre[0]),
                    centre_y_m=float(centre[1]),
                    tangent_x=math.cos(angle),
                    tangent_y=math.sin(angle),
                    clear_width_m=float(measured["clear_width_m"]),
                )
            )
        profile = measured_overlay["presentation_clearance_profile"]
        polygon = measured_overlay["plan_geometry"]["atrium"]["central_polygon"]
        return cls(
            doors,
            maximum_doorway_speed_mps=float(profile["maximum_doorway_speed_mps"]),
            polygon_outer_radius_m=float(
                site["plan_geometry"]["atrium"]["central_polygon"][
                    "outer_vertex_radius_m"
                ]
            ),
        )

    def _door_coordinates(
        self, door: Doorway, x_m: float, y_m: float
    ) -> tuple[float, float]:
        relative_x = x_m - door.centre_x_m
        relative_y = y_m - door.centre_y_m
        normal_x, normal_y = door.normal
        tangent = relative_x * door.tangent_x + relative_y * door.tangent_y
        normal = relative_x * normal_x + relative_y * normal_y
        return tangent, normal

    def _select_door(self, x_m: float, y_m: float, yaw_rad: float) -> tuple[int | None, float, float]:
        if self._active_door_index is not None:
            active = self.doors[self._active_door_index]
            tangent, normal = self._door_coordinates(active, x_m, y_m)
            if (
                abs(tangent) <= self.tangent_half_extent_m + 0.20
                and abs(normal) <= self.alignment_zone_normal_half_extent_m + 0.20
            ):
                return self._active_door_index, tangent, normal
            self._active_door_index = None
            self._crossing_sign = 0.0

        candidates: list[tuple[float, int, float, float]] = []
        for index, door in enumerate(self.doors):
            tangent, normal = self._door_coordinates(door, x_m, y_m)
            if (
                abs(tangent) <= self.tangent_half_extent_m
                and abs(normal) <= self.alignment_zone_normal_half_extent_m
            ):
                candidates.append((math.hypot(tangent, normal), index, tangent, normal))
        if not candidates:
            return None, math.inf, math.inf
        _, index, tangent, normal = min(candidates)
        self._active_door_index = index
        if abs(normal) > 0.05:
            self._crossing_sign = -math.copysign(1.0, normal)
        else:
            normal_x, normal_y = self.doors[index].normal
            projection = math.cos(yaw_rad) * normal_x + math.sin(yaw_rad) * normal_y
            self._crossing_sign = math.copysign(1.0, projection or 1.0)
        self.statistics["doorway_entries"][self.doors[index].name] += 1
        return index, tangent, normal

    def _polygon_guard(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        linear_mps: float,
    ) -> tuple[float, bool, float]:
        radius = math.hypot(x_m, y_m)
        clearance = radius - self.polygon_outer_radius_m - self.body_circumscribed_radius_m
        self.statistics["minimum_polygon_full_footprint_clearance_m"] = min(
            self.statistics["minimum_polygon_full_footprint_clearance_m"], clearance
        )
        if linear_mps <= 0.0:
            return linear_mps, False, clearance
        predicted_x = x_m + linear_mps * math.cos(yaw_rad) * self.polygon_prediction_horizon_s
        predicted_y = y_m + linear_mps * math.sin(yaw_rad) * self.polygon_prediction_horizon_s
        predicted_clearance = (
            math.hypot(predicted_x, predicted_y)
            - self.polygon_outer_radius_m
            - self.body_circumscribed_radius_m
        )
        radial_heading = (
            (x_m * math.cos(yaw_rad) + y_m * math.sin(yaw_rad)) / max(radius, 1.0e-6)
        )
        stop = radial_heading < 0.0 and predicted_clearance < self.polygon_minimum_clearance_m
        if stop:
            self.statistics["polygon_no_go_stops"] += 1
            return 0.0, True, clearance
        return linear_mps, False, clearance

    def apply(
        self,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        yaw_rate_rad_s: float,
        forward_speed_mps: float,
        requested_linear_mps: float,
        requested_angular_rad_s: float,
    ) -> GuardResult:
        self.statistics["steps"] += 1
        linear = max(0.0, requested_linear_mps)
        angular = max(
            -self.maximum_angular_rad_s,
            min(self.maximum_angular_rad_s, requested_angular_rad_s),
        )
        door_index, tangent, normal = self._select_door(x_m, y_m, yaw_rad)
        slow = alignment = hold = overspeed = False
        yaw_error = 0.0
        active_name = None
        if door_index is not None:
            door = self.doors[door_index]
            active_name = door.name
            normal_x, normal_y = door.normal
            heading_normal_projection = (
                math.cos(yaw_rad) * normal_x + math.sin(yaw_rad) * normal_y
            )
            completed_previous_crossing = normal * self._crossing_sign > 0.20
            facing_return_direction = (
                heading_normal_projection * self._crossing_sign < -0.50
            )
            if linear > 1.0e-3 and completed_previous_crossing and facing_return_direction:
                self._crossing_sign = math.copysign(1.0, heading_normal_projection)
                self.statistics["doorway_direction_rearms"] += 1
                self.statistics["doorway_entries"][door.name] += 1
            normal_distance = abs(normal)
            tangent_distance = abs(tangent)
            in_tangent_zone = tangent_distance <= self.tangent_half_extent_m
            slow = in_tangent_zone and normal_distance <= self.slow_zone_normal_half_extent_m
            no_rotation = in_tangent_zone and normal_distance <= self.no_rotation_normal_half_extent_m
            alignment = in_tangent_zone and (
                no_rotation
                or (
                    normal_distance <= self.alignment_zone_normal_half_extent_m
                    and linear > 1.0e-3
                )
            )
            if alignment:
                direction_x = self._crossing_sign * normal_x
                direction_y = self._crossing_sign * normal_y
                on_approach_side = normal * self._crossing_sign < 0.0
                approach_stage_x = (
                    door.centre_x_m - self._crossing_sign * normal_x
                )
                approach_stage_y = (
                    door.centre_y_m - self._crossing_sign * normal_y
                )
                approach_stage_distance = math.hypot(
                    approach_stage_x - x_m, approach_stage_y - y_m
                )
                coarse_approach = (
                    on_approach_side
                    and normal_distance
                    > 1.0 - self.approach_stage_capture_radius_m
                    and approach_stage_distance > self.approach_stage_capture_radius_m
                )
                if coarse_approach:
                    stage_x = approach_stage_x
                    stage_y = approach_stage_y
                else:
                    stage_x = door.centre_x_m + self._crossing_sign * normal_x
                    stage_y = door.centre_y_m + self._crossing_sign * normal_y
                stage_yaw = math.atan2(stage_y - y_m, stage_x - x_m)
                normal_yaw = math.atan2(direction_y, direction_x)
                desired_yaw = stage_yaw if coarse_approach else normal_yaw
                yaw_error = wrap_angle(desired_yaw - yaw_rad)
                correction = self.heading_gain * yaw_error - self.yaw_damping_gain * yaw_rate_rad_s
                correction = max(-self.maximum_angular_rad_s, min(self.maximum_angular_rad_s, correction))
                if (
                    abs(yaw_error) > self.alignment_tolerance_rad
                    and abs(correction) < self.breakaway_angular_rad_s
                    and abs(yaw_rate_rad_s) < 0.01
                ):
                    correction = math.copysign(self.breakaway_angular_rad_s, yaw_error)
                angular = correction
                active_alignment_tolerance = (
                    self.alignment_tolerance_rad
                    if normal_distance <= self.slow_zone_normal_half_extent_m
                    else self.approach_alignment_tolerance_rad
                )
                heading_hold = abs(yaw_error) > active_alignment_tolerance
                settling_hold = not heading_hold and abs(yaw_rate_rad_s) > self.yaw_settle_rate_rad_s
                hold = heading_hold or settling_hold
                if hold:
                    linear = 0.0
                    if settling_hold:
                        angular = 0.0
                else:
                    # The furnished USD's raw velocity drives need more wheel
                    # command than measured chassis speed at this crawl.  Use
                    # closed-loop compensation from actual body speed; the
                    # overspeed brake and acceptance telemetry stay measured.
                    traction_command = self.target_doorway_speed_mps + self.traction_gain * max(
                        0.0, self.target_doorway_speed_mps - forward_speed_mps
                    )
                    linear = min(
                        linear,
                        traction_command,
                        self.traction_command_ceiling_mps,
                    )
                if forward_speed_mps > self.overspeed_stop_mps:
                    linear = 0.0
                    overspeed = True
            elif slow:
                linear = min(linear, self.maximum_doorway_speed_mps)

            self.statistics["doorway_active_steps"] += 1
            self.statistics["doorway_slow_steps"] += int(slow)
            self.statistics["doorway_alignment_steps"] += int(alignment)
            self.statistics["doorway_alignment_hold_steps"] += int(hold)
            self.statistics["doorway_overspeed_stops"] += int(overspeed)
            if no_rotation:
                self.statistics["maximum_abs_speed_in_doorway_mps"] = max(
                    self.statistics["maximum_abs_speed_in_doorway_mps"],
                    abs(forward_speed_mps),
                )
                self.statistics["maximum_abs_tangent_offset_in_doorway_m"] = max(
                    self.statistics["maximum_abs_tangent_offset_in_doorway_m"],
                    tangent_distance,
                )

        linear, polygon_stop, polygon_clearance = self._polygon_guard(
            x_m, y_m, yaw_rad, linear
        )
        return GuardResult(
            linear_mps=linear,
            angular_rad_s=angular,
            active_door=active_name,
            doorway_slow_active=slow,
            doorway_alignment_active=alignment,
            doorway_alignment_hold=hold,
            doorway_overspeed_stop=overspeed,
            polygon_no_go_stop=polygon_stop,
            yaw_error_rad=yaw_error,
            tangent_distance_m=abs(tangent) if door_index is not None else None,
            normal_distance_m=abs(normal) if door_index is not None else None,
            polygon_footprint_clearance_m=polygon_clearance,
        )

    def report(self) -> dict[str, Any]:
        report = dict(self.statistics)
        minimum = report["minimum_polygon_full_footprint_clearance_m"]
        report["minimum_polygon_full_footprint_clearance_m"] = (
            round(float(minimum), 6) if math.isfinite(minimum) else None
        )
        report.update(
            {
                "enabled": True,
                "architecture": "nav2_cmd_vel_then_mapped_doorway_and_polygon_guard",
                "maximum_doorway_speed_mps": self.maximum_doorway_speed_mps,
                "target_doorway_speed_mps": self.target_doorway_speed_mps,
                "doorway_alignment_tolerance_deg": math.degrees(
                    self.alignment_tolerance_rad
                ),
                "approach_alignment_tolerance_deg": math.degrees(
                    self.approach_alignment_tolerance_rad
                ),
                "approach_stage_capture_radius_m": self.approach_stage_capture_radius_m,
                "overspeed_stop_mps": self.overspeed_stop_mps,
                "traction_command_ceiling_mps": self.traction_command_ceiling_mps,
                "traction_gain": self.traction_gain,
                "breakaway_angular_rad_s": self.breakaway_angular_rad_s,
                "polygon_outer_radius_m": self.polygon_outer_radius_m,
                "polygon_body_circumscribed_radius_m": self.body_circumscribed_radius_m,
                "polygon_minimum_clearance_m": self.polygon_minimum_clearance_m,
                "physical_safety_credit": False,
            }
        )
        return report
