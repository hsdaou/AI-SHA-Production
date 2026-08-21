#!/usr/bin/env python3
"""Structural and cross-geometry validator for the AI-SHA PoC URDFs.

The physical robot has two driven wheels, four swivel castors and a compliant
drive carrier. The baseline URDF uses fixed spherical castor contacts and a
rigid carrier for numerical stability; that limitation is checked and reported,
not hidden.
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"aisha.urdf": 59.25, "aisha_max_payload.urdf": 69.25}
CASTORS = {"castor_fl_link", "castor_fr_link", "castor_rl_link", "castor_rr_link"}
REQUIRED_FRAMES = {
    "lidar_link",
    "front_lidar_link",
    "front_camera_link",
    "front_camera_optical_frame",
    "imu_link",
    "cargo_payload_frame",
}
CASTOR_TRAIL_M = 0.030


def rpy(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def origin(element):
    xyz, angles = [0.0] * 3, [0.0] * 3
    if element is not None:
        if element.get("xyz"):
            xyz = [float(value) for value in element.get("xyz").split()]
        if element.get("rpy"):
            angles = [float(value) for value in element.get("rpy").split()]
    return rpy(*angles), xyz


def compose(a, b):
    ra, ta = a
    rb, tb = b
    rotation = [
        [sum(ra[i][k] * rb[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    translation = [ta[i] + sum(ra[i][k] * tb[k] for k in range(3)) for i in range(3)]
    return rotation, translation


IDENT = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0])


def cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def turn(o, a, b):
        return cross_2d((a[0] - o[0], a[1] - o[1]), (b[0] - o[0], b[1] - o[1]))

    lower = []
    for point in points:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def ray_margin(point, direction, polygon):
    """Distance from an interior point to a convex polygon along direction."""
    candidates = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        segment = (end[0] - start[0], end[1] - start[1])
        denominator = cross_2d(direction, segment)
        if abs(denominator) < 1e-12:
            continue
        delta = (start[0] - point[0], start[1] - point[1])
        ray_t = cross_2d(delta, segment) / denominator
        segment_u = cross_2d(delta, direction) / denominator
        if ray_t >= -1e-9 and -1e-9 <= segment_u <= 1 + 1e-9:
            candidates.append(max(0.0, ray_t))
    if not candidates:
        raise ValueError("CG projection is outside or support polygon is invalid")
    return min(candidates)


def world_transforms(links, joints, errors):
    children = set()
    tree = {}
    for name, joint_element in joints.items():
        parent = joint_element.find("parent").attrib["link"]
        child = joint_element.find("child").attrib["link"]
        if parent not in links:
            errors.append(f"{name}: missing parent {parent}")
        if child not in links:
            errors.append(f"{name}: missing child {child}")
        if child in children:
            errors.append(f"{child}: more than one parent")
        children.add(child)
        tree.setdefault(parent, []).append((child, origin(joint_element.find("origin"))))

    roots = set(links) - children
    if roots != {"base_link"}:
        errors.append(f"expected a single root 'base_link', found {sorted(roots)}")

    world, stack = {"base_link": IDENT}, ["base_link"]
    while stack:
        parent = stack.pop()
        for child, transform in tree.get(parent, []):
            world[child] = compose(world[parent], transform)
            stack.append(child)
    return world


def check(path: Path, expected_mass: float):
    errors, warnings = [], []
    root = ET.parse(path).getroot()
    links = {element.attrib["name"]: element for element in root.findall("link")}
    joints = {element.attrib["name"]: element for element in root.findall("joint")}
    world = world_transforms(links, joints, errors)

    # Mass and full inertia-tensor sanity.
    mass = 0.0
    for name, link in links.items():
        inertial_element = link.find("inertial")
        if inertial_element is None:
            errors.append(f"{name}: missing <inertial>; Isaac may assign a default")
            continue
        link_mass = float(inertial_element.find("mass").attrib["value"])
        mass += link_mass
        inertia = inertial_element.find("inertia").attrib
        ixx, iyy, izz = (float(inertia[key]) for key in ("ixx", "iyy", "izz"))
        ixy, ixz, iyz = (float(inertia.get(key, 0)) for key in ("ixy", "ixz", "iyz"))
        if min(link_mass, ixx, iyy, izz) <= 0:
            errors.append(f"{name}: non-positive mass or principal inertia")
        elif ixx + iyy < izz - 1e-10 or ixx + izz < iyy - 1e-10 or iyy + izz < ixx - 1e-10:
            errors.append(f"{name}: inertia violates the triangle inequality")
        else:
            # Sylvester's criterion for a symmetric 3x3 inertia tensor.
            minor_2 = ixx * iyy - ixy * ixy
            determinant = (
                ixx * iyy * izz + 2 * ixy * ixz * iyz
                - ixx * iyz * iyz - iyy * ixz * ixz - izz * ixy * ixy
            )
            if minor_2 <= 0 or determinant <= 0:
                errors.append(f"{name}: inertia tensor is not positive definite")
    if not math.isclose(mass, expected_mass, abs_tol=0.02):
        errors.append(f"total mass {mass:.3f} kg, expected {expected_mass:.2f} kg")

    # Exactly two driven joints, both continuous about +Y. The URDF limit is
    # rated torque; peak torque must be separately time-limited by the controller.
    wheel_joints = sorted(name for name in joints if name.endswith("_wheel_joint"))
    if wheel_joints != ["left_wheel_joint", "right_wheel_joint"]:
        errors.append(f"expected exactly left/right_wheel_joint, found {wheel_joints}")
    for name in wheel_joints:
        wheel_joint = joints[name]
        if wheel_joint.attrib.get("type") != "continuous":
            errors.append(f"{name}: must be continuous")
        axis = wheel_joint.find("axis")
        if axis is None or axis.attrib.get("xyz") != "0 1 0":
            errors.append(f"{name}: drive axis must be '0 1 0' (+Y)")
        limit = wheel_joint.find("limit")
        if limit is None:
            errors.append(f"{name}: missing <limit>")
        else:
            effort, velocity = float(limit.attrib["effort"]), float(limit.attrib["velocity"])
            if not (0 < effort <= 6.0):
                errors.append(f"{name}: normal effort limit {effort:g} must not exceed 6 N.m rated")
            if velocity <= 0:
                errors.append(f"{name}: invalid velocity limit")

    continuous = sorted(name for name, joint_element in joints.items() if joint_element.attrib.get("type") == "continuous")
    if continuous != wheel_joints:
        errors.append(f"unexpected continuous joints: {sorted(set(continuous) - set(wheel_joints))}")

    # Contact geometry and the intentional castor proxy.
    missing_castors = CASTORS - set(links)
    if missing_castors:
        errors.append(f"missing castor links: {sorted(missing_castors)}")
    for castor in CASTORS & set(links):
        parents = [
            (name, joint_element)
            for name, joint_element in joints.items()
            if joint_element.find("child").attrib["link"] == castor
        ]
        if not parents or parents[0][1].attrib.get("type") != "fixed":
            errors.append(f"{castor}: baseline castor proxy must be fixed")
        geometry = links[castor].find("collision/geometry/sphere")
        if geometry is None:
            errors.append(f"{castor}: expected a spherical baseline contact proxy")

    # Frames required by navigation and perception.
    for frame in sorted(REQUIRED_FRAMES - set(links)):
        errors.append(f"missing required frame: {frame}")
    if "front_lidar_link" in world:
        front_lidar_z = world["front_lidar_link"][1][2]
        if not 0.22 <= front_lidar_z <= 0.35:
            errors.append(f"front_lidar_link scan frame {front_lidar_z:.3f} m is outside 0.22-0.35 m")
    if "lidar_link" in world:
        crown_z = world["lidar_link"][1][2]
        if not math.isclose(crown_z, 1.170, abs_tol=0.005):
            errors.append(f"lidar_link scan frame {crown_z:.3f} m, expected 1.170 m")

    # Centre of gravity.
    total, accumulator = 0.0, [0.0] * 3
    for name, link in links.items():
        inertial_element = link.find("inertial")
        if inertial_element is None or name not in world:
            continue
        link_mass = float(inertial_element.find("mass").attrib["value"])
        _, position = compose(world[name], origin(inertial_element.find("origin")))
        total += link_mass
        for index in range(3):
            accumulator[index] += link_mass * position[index]
    cg = [value / total for value in accumulator]
    if cg[2] > 0.45:
        warnings.append(f"CG height {cg[2]:.3f} m is above the design review threshold")
    if abs(cg[1]) > 0.005:
        errors.append(f"CG is off the centreline by {cg[1] * 1000:.1f} mm")

    # Conservative physical support polygon. The castor contact can orbit its
    # swivel axis by the trail, so caster coordinates are shrunk inward by 30 mm.
    support_points = []
    for wheel in ("left_wheel_link", "right_wheel_link"):
        if wheel in world:
            x, y, _ = world[wheel][1]
            support_points.append((x, y))
    for castor in sorted(CASTORS):
        if castor in world:
            x, y, _ = world[castor][1]
            support_points.append((
                math.copysign(max(0, abs(x) - CASTOR_TRAIL_M), x),
                math.copysign(max(0, abs(y) - CASTOR_TRAIL_M), y),
            ))
    hull = convex_hull(support_points)
    margins = {}
    try:
        for label, direction in (
            ("front", (1.0, 0.0)),
            ("rear", (-1.0, 0.0)),
            ("left", (0.0, 1.0)),
            ("right", (0.0, -1.0)),
        ):
            margins[label] = ray_margin((cg[0], cg[1]), direction, hull)
    except ValueError as exc:
        errors.append(str(exc))
        margins = {label: 0 for label in ("front", "rear", "left", "right")}

    angles = {label: math.degrees(math.atan(value / cg[2])) for label, value in margins.items()}
    if min(angles.values()) < 30:
        errors.append(f"minimum conservative static tip angle {min(angles.values()):.1f} deg is below 30 deg")

    return errors, warnings, mass, cg, margins, angles, len(links), len(joints)


ok = True
for filename, expected in EXPECTED.items():
    urdf_path = ROOT / "urdf" / filename
    if not urdf_path.exists():
        print(f"MISSING: {urdf_path}")
        ok = False
        continue
    errors, warnings, mass, cg, margins, angles, link_count, joint_count = check(urdf_path, expected)
    print(f"\n{filename}")
    print(f"  {link_count} links, {joint_count} joints, {mass:.2f} kg")
    print(f"  CG  x{cg[0]:+.4f}  y{cg[1]:+.4f}  z {cg[2]:.4f} m")
    print(
        "  conservative support margins "
        f"front {margins['front']:.3f}  rear {margins['rear']:.3f}  "
        f"left/right {min(margins['left'], margins['right']):.3f} m"
    )
    print(
        "  static tip angles "
        f"front {angles['front']:.1f}  rear {angles['rear']:.1f}  "
        f"lateral {min(angles['left'], angles['right']):.1f} deg"
    )
    for warning in warnings:
        print(f"  WARN  {warning}")
    for error in errors:
        print(f"  FAIL  {error}")
    if errors:
        ok = False

print()
if not ok:
    print("URDF validation FAILED")
    sys.exit(1)
print("URDF validation passed: Rev D baseline, 2 driven wheels + 4 castor proxies")
print("NOTE: physical drive compliance and castor swivel dynamics require separate validation")
