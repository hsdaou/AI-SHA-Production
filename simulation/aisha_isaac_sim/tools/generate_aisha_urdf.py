#!/usr/bin/env python3
"""Generate the AI-SHA proof-of-concept URDFs.

Physical architecture: two centre-line differential-drive wheels, four passive
swivel castors, a retained Prestar NF-301 deck, a guided compliant drive beam,
and a fixed mast reinforcement spine.

The URDF intentionally uses fixed low-friction spheres as castor *contact
proxies* and a rigid drive beam. Those are stable baseline approximations for
Isaac Sim, not claims that the physical castors are spherical or that drive
compliance is optional. See README.md and config/aisha_drive.yaml.

Writes urdf/aisha.urdf (design-empty) and
urdf/aisha_max_payload.urdf (10 kg proof-of-concept payload). Edit this file and
regenerate; do not hand-edit the XML.

Convention: +X forward, +Y left, +Z up. SI units. The base_link origin is at
floor level on the deck centrelines.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "urdf"

# ---------------------------------------------------------------------------
# Geometry (metres)
# ---------------------------------------------------------------------------
# V2.18 values are based on direct supplier correspondence recorded by the
# project. The package only contains a V2.0 drawing, so the V2.18 drawing and
# as-delivered measurements remain fabrication hold points.
WHEEL_R = 0.100                 # V2.18 design value: OD 200 +/-2 mm
WHEEL_W = 0.048                 # V2.18 design value: tread 48 +/-1 mm
TRACK = 0.720                   # provisional until door and V2.18 checks close
CASTOR_R = 0.065
CASTOR_X, CASTOR_Y = 0.350, 0.255   # swivel-axis locations, not contact points
CASTOR_TRAIL = 0.030            # conservative physical stability allowance

DECK_L, DECK_W = 0.910, 0.610
DECK_TOP, DECK_THK = 0.210, 0.032  # NF-301 floor height and 178 mm castor height
BATT_L, BATT_W, BATT_H = 0.331, 0.173, 0.220

TRAY_FRONT, TRAY_REAR = 0.350, -0.455
TRAY_SURFACE_Z, TRAY_SHEET_T = 0.530, 0.003
TRAY_POST_X = (-0.350, 0.245)
TRAY_POST_Y = (-0.255, 0.255)

MAST_X, MAST_SIZE, MAST_H = 0.420, 0.070, 0.600
HEAD_R, HEAD_X, HEAD_Z = 0.225, 0.500, 0.925
FRONT_LIDAR_X, FRONT_LIDAR_Z = 0.455, 0.250

PAYLOAD_KG = 10.0              # structural PoC limit, not a drivetrain claim
PAYLOAD_BOX = (0.500, 0.400, 0.150)
PAYLOAD_Z = TRAY_SURFACE_Z + PAYLOAD_BOX[2] / 2

# ---------------------------------------------------------------------------
# Design masses (kg). These remain estimates until the finished robot is
# corner-weighed. The battery value is from the supplied SBM25.6LI50 datasheet.
# ---------------------------------------------------------------------------
MASS_DECK = 11.0
MASS_STRUCTURE = 10.0
MASS_WHEEL = 3.75
MASS_CASTOR = 0.50
MASS_BATTERY = 11.25
MASS_ELECTRONICS = 2.20
MASS_TRAY_SHEET = 4.00
MASS_TRAY_POSTS = 1.00
MASS_MAST = 1.50
MASS_HEAD = 8.20               # includes compute, camera, audio and head wiring
MASS_LIDAR = 0.30
FRAME_EPSILON_MASS = 1e-4

EXPECTED_EMPTY_KG = 59.25
EXPECTED_LOADED_KG = 69.25


# ---------------------------------------------------------------------------
# Inertia helpers
# ---------------------------------------------------------------------------
def box_I(m, x, y, z):
    return (
        m * (y * y + z * z) / 12.0,
        m * (x * x + z * z) / 12.0,
        m * (x * x + y * y) / 12.0,
    )


def cyl_I_about_y(m, r, h):
    """Cylinder whose axis is +Y (a wheel)."""
    perp = m * (3 * r * r + h * h) / 12.0
    return (perp, m * r * r / 2.0, perp)


def sphere_I(m, r):
    value = 2.0 * m * r * r / 5.0
    return (value, value, value)


def inertial(
    m,
    ixx,
    iyy,
    izz,
    xyz=(0, 0, 0),
    ixy=0.0,
    ixz=0.0,
    iyz=0.0,
):
    return (
        "    <inertial>\n"
        f'      <origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" rpy="0 0 0"/>\n'
        f'      <mass value="{m:.4f}"/>\n'
        # Significant digits preserve small positive placeholder inertias.
        f'      <inertia ixx="{ixx:.10g}" ixy="{ixy:.10g}" ixz="{ixz:.10g}" '
        f'iyy="{iyy:.10g}" iyz="{iyz:.10g}" izz="{izz:.10g}"/>\n'
        "    </inertial>\n"
    )


def box_link(name, m, sx, sy, sz, rgba, com=(0, 0, 0), collision=True):
    ixx, iyy, izz = box_I(m, sx, sy, sz)
    result = (
        f'  <link name="{name}">\n'
        + inertial(m, ixx, iyy, izz, com)
        + f'    <visual><origin xyz="{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}"/>'
          f'<geometry><box size="{sx:.4f} {sy:.4f} {sz:.4f}"/></geometry>'
          f'<material name="{name}_m"><color rgba="{rgba}"/></material></visual>\n'
    )
    if collision:
        result += (
            f'    <collision><origin xyz="{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}"/>'
            f'<geometry><box size="{sx:.4f} {sy:.4f} {sz:.4f}"/></geometry></collision>\n'
        )
    return result + "  </link>\n"


def compound_box_link(name, parts, rgba):
    """Create one fixed link from axis-aligned box parts.

    Each part is (mass, (sx, sy, sz), (x, y, z), collision). Off-diagonal
    inertia terms and parallel-axis contributions are retained.
    """
    total = sum(part[0] for part in parts)
    com = tuple(sum(m * origin[k] for m, _, origin, _ in parts) / total for k in range(3))
    ixx = iyy = izz = ixy = ixz = iyz = 0.0
    for mass, size, origin, _ in parts:
        bx, by, bz = box_I(mass, *size)
        dx, dy, dz = (origin[k] - com[k] for k in range(3))
        ixx += bx + mass * (dy * dy + dz * dz)
        iyy += by + mass * (dx * dx + dz * dz)
        izz += bz + mass * (dx * dx + dy * dy)
        ixy -= mass * dx * dy
        ixz -= mass * dx * dz
        iyz -= mass * dy * dz

    result = f'  <link name="{name}">\n' + inertial(
        total, ixx, iyy, izz, com, ixy=ixy, ixz=ixz, iyz=iyz
    )
    for idx, (_, size, origin, collision) in enumerate(parts):
        sx, sy, sz = size
        ox, oy, oz = origin
        result += (
            f'    <visual name="{name}_visual_{idx}"><origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}"/>'
            f'<geometry><box size="{sx:.4f} {sy:.4f} {sz:.4f}"/></geometry>'
            f'<material name="{name}_m"><color rgba="{rgba}"/></material></visual>\n'
        )
        if collision:
            result += (
                f'    <collision name="{name}_collision_{idx}"><origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}"/>'
                f'<geometry><box size="{sx:.4f} {sy:.4f} {sz:.4f}"/></geometry></collision>\n'
            )
    return result + "  </link>\n"


def placeholder_link(name):
    return (
        f'  <link name="{name}">\n'
        + inertial(FRAME_EPSILON_MASS, 1e-8, 1e-8, 1e-8)
        + "  </link>\n"
    )


def joint(name, parent, child, xyz, jtype="fixed", axis=None, extra="", rpy=(0, 0, 0)):
    result = (
        f'  <joint name="{name}" type="{jtype}">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" '
        f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>\n'
    )
    if axis:
        result += f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n'
    return result + extra + "  </joint>\n"


def build(loaded: bool) -> str:
    xml = [
        '<?xml version="1.0"?>\n',
        '<!-- AI-SHA PoC Rev D: 2 driven wheels + 4 physical swivel castors.\n',
        '     Castor spheres and rigid drive beam are simulation proxies; see README.\n',
        '     Generated by tools/generate_aisha_urdf.py; do not hand-edit. -->\n',
        f'<robot name="aisha_poc{"_loaded" if loaded else ""}">\n',
    ]

    # Retained Prestar deck. Official platform height is 210 mm; a listed
    # 178 mm caster overall height implies a 32 mm deck structural envelope.
    xml.append(box_link(
        "base_link",
        MASS_DECK,
        DECK_L,
        DECK_W,
        DECK_THK,
        "0.55 0.57 0.60 1",
        com=(0, 0, DECK_TOP - DECK_THK / 2),
    ))

    # Conservative 10 kg allowance for the moving beam, fixed mast spine,
    # axle plates, broad backing plates, guided spring hardware and fasteners.
    # These boxes reflect load paths instead of the old solid 860x580x60 slab.
    structure_parts = [
        (2.57, (0.040, 0.580, 0.060), (0.000, 0.000, 0.140), True),
        (1.56, (0.420, 0.040, 0.045), (0.210, 0.000, 0.1555), True),
        (0.75, (0.100, 0.008, 0.100), (0.000, +0.294, 0.128), True),
        (0.75, (0.100, 0.008, 0.100), (0.000, -0.294, 0.128), True),
        (1.22, (0.150, 0.130, 0.008), (0.000, +0.230, 0.174), True),
        (1.22, (0.150, 0.130, 0.008), (0.000, -0.230, 0.174), True),
        (1.06, (0.150, 0.150, 0.006), (0.405, 0.000, 0.175), True),
        (0.87, (0.100, 0.450, 0.012), (0.000, 0.000, 0.170), False),
    ]
    xml.append(compound_box_link("structure_link", structure_parts, "0.35 0.37 0.40 1"))
    xml.append(joint("structure_joint", "base_link", "structure_link", (0, 0, 0)))

    # Drive wheels: the only actuated joints. The normal URDF effort limit is
    # continuous/rated torque. The 18 N.m peak is controller-timed, not a
    # continuously available physics limit.
    for side, sign in (("left", 1), ("right", -1)):
        ixx, iyy, izz = cyl_I_about_y(MASS_WHEEL, WHEEL_R, WHEEL_W)
        xml.append(
            f'  <link name="{side}_wheel_link">\n'
            + inertial(MASS_WHEEL, ixx, iyy, izz)
            + f'    <visual><origin rpy="-1.570796 0 0"/><geometry>'
              f'<cylinder radius="{WHEEL_R}" length="{WHEEL_W}"/></geometry>'
              f'<material name="tyre"><color rgba="0.07 0.08 0.09 1"/></material></visual>\n'
            + f'    <collision><origin rpy="-1.570796 0 0"/><geometry>'
              f'<cylinder radius="{WHEEL_R}" length="{WHEEL_W}"/></geometry></collision>\n'
            + "  </link>\n"
        )
        xml.append(joint(
            f"{side}_wheel_joint",
            "structure_link",
            f"{side}_wheel_link",
            (0.0, sign * TRACK / 2, WHEEL_R),
            jtype="continuous",
            axis=(0, 1, 0),
            extra='    <limit effort="6.0" velocity="16.755"/>\n'
                  '    <dynamics damping="0.05" friction="0.02"/>\n',
        ))

    # Physical robot: four matched 360-degree swivel castors. Baseline sim:
    # fixed low-friction spheres to avoid PhysX swivel chatter. A separate
    # high-fidelity caster/suspension test is required before sim-to-real use.
    for tag, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)):
        ixx, iyy, izz = sphere_I(MASS_CASTOR, CASTOR_R)
        xml.append(
            f'  <link name="castor_{tag}_link">\n'
            + inertial(MASS_CASTOR, ixx, iyy, izz)
            + f'    <visual><geometry><sphere radius="{CASTOR_R}"/></geometry>'
              f'<material name="castor_proxy"><color rgba="0.60 0.60 0.62 1"/></material></visual>\n'
            + f'    <collision><geometry><sphere radius="{CASTOR_R}"/></geometry></collision>\n'
            + "  </link>\n"
        )
        xml.append(joint(
            f"castor_{tag}_joint",
            "base_link",
            f"castor_{tag}_link",
            (sx * CASTOR_X, sy * CASTOR_Y, CASTOR_R),
        ))

    # Battery in a rigid cradle near the axle. Spring preload, not battery
    # location alone, establishes drive-wheel normal force.
    xml.append(box_link(
        "battery_link", MASS_BATTERY, BATT_L, BATT_W, BATT_H, "0.10 0.11 0.13 1"
    ))
    xml.append(joint(
        "battery_joint", "base_link", "battery_link", (0, 0, DECK_TOP + BATT_H / 2)
    ))

    # Driver/electrical enclosure proxy. The real under-deck installation must
    # be splash/dust protected while meeting the driver's cooling instructions.
    xml.append(box_link(
        "electronics_link", MASS_ELECTRONICS, 0.170, 0.120, 0.040, "0.20 0.22 0.24 1"
    ))
    xml.append(joint(
        "electronics_joint", "base_link", "electronics_link", (-0.120, 0, 0.140)
    ))

    # 3 mm aluminium tray plus four posts. They are separated so their mass
    # distribution is not represented as the old 25 mm solid slab.
    tray_len = TRAY_FRONT - TRAY_REAR
    xml.append(box_link(
        "cargo_tray_link",
        MASS_TRAY_SHEET,
        tray_len,
        DECK_W,
        TRAY_SHEET_T,
        "0.72 0.74 0.76 1",
    ))
    xml.append(joint(
        "cargo_tray_joint",
        "base_link",
        "cargo_tray_link",
        ((TRAY_FRONT + TRAY_REAR) / 2, 0, TRAY_SURFACE_Z - TRAY_SHEET_T / 2),
    ))

    post_h = TRAY_SURFACE_Z - TRAY_SHEET_T - DECK_TOP
    post_parts = []
    for px in TRAY_POST_X:
        for py in TRAY_POST_Y:
            post_parts.append((
                MASS_TRAY_POSTS / 4,
                (0.025, 0.025, post_h),
                (px, py, DECK_TOP + post_h / 2),
                True,
            ))
    xml.append(compound_box_link("cargo_posts_link", post_parts, "0.62 0.64 0.66 1"))
    xml.append(joint("cargo_posts_joint", "base_link", "cargo_posts_link", (0, 0, 0)))

    # Mast, head and sensors.
    xml.append(box_link(
        "mast_link", MASS_MAST, MAST_SIZE, MAST_SIZE, MAST_H, "0.70 0.72 0.74 1"
    ))
    xml.append(joint(
        "mast_joint", "base_link", "mast_link", (MAST_X, 0, DECK_TOP + MAST_H / 2)
    ))

    ixx, iyy, izz = sphere_I(MASS_HEAD, HEAD_R)
    xml.append(
        '  <link name="head_link">\n'
        + inertial(MASS_HEAD, ixx, iyy, izz)
        + f'    <visual><geometry><sphere radius="{HEAD_R}"/></geometry>'
          f'<material name="shell"><color rgba="0.93 0.92 0.88 1"/></material></visual>\n'
        + f'    <collision><geometry><sphere radius="{HEAD_R}"/></geometry></collision>\n'
        + "  </link>\n"
    )
    xml.append(joint("head_joint", "base_link", "head_link", (HEAD_X, 0, HEAD_Z)))

    # Crown LiDAR scan frame is at 1.170 m; the top of its 40 mm housing is
    # 1.190 m. It is a localisation sensor, not low-obstacle protection.
    xml.append(box_link(
        "lidar_link", MASS_LIDAR, 0.060, 0.060, 0.040, "0.12 0.13 0.15 1"
    ))
    xml.append(joint(
        "lidar_joint", "head_link", "lidar_link", (0, 0, HEAD_R + 0.020)
    ))

    # Low front perception LiDAR. Its housing clears the 210 mm deck top.
    # It is not treated as a safety-rated all-round protective scanner.
    xml.append(box_link(
        "front_lidar_link", MASS_LIDAR, 0.060, 0.060, 0.040, "0.12 0.13 0.15 1"
    ))
    xml.append(joint(
        "front_lidar_joint",
        "base_link",
        "front_lidar_link",
        (FRONT_LIDAR_X, 0, FRONT_LIDAR_Z),
    ))

    # Camera body frame (+X forward) and ROS optical frame (+Z forward, +X
    # right, +Y down). Their mass is included in MASS_HEAD.
    xml.append(placeholder_link("front_camera_link"))
    xml.append(joint(
        "front_camera_joint", "head_link", "front_camera_link", (HEAD_R - 0.020, 0, 0.020)
    ))
    xml.append(placeholder_link("front_camera_optical_frame"))
    xml.append(joint(
        "front_camera_optical_joint",
        "front_camera_link",
        "front_camera_optical_frame",
        (0, 0, 0),
        rpy=(-1.570796, 0, -1.570796),
    ))

    # IMU frame near the yaw centre on the deck. Its mass is included in the
    # electronics allowance; exact as-built XYZ must replace this placeholder.
    xml.append(placeholder_link("imu_link"))
    xml.append(joint("imu_joint", "base_link", "imu_link", (-0.120, 0.120, 0.230)))

    # Payload frame: 10 kg is the PoC structural/duty limit. The empty frame
    # keeps a tiny positive inertia for importers that reject zero inertia.
    if loaded:
        px, py, pz = PAYLOAD_BOX
        xml.append(box_link(
            "cargo_payload_frame", PAYLOAD_KG, px, py, pz, "0.80 0.70 0.45 1"
        ))
    else:
        xml.append(placeholder_link("cargo_payload_frame"))
    xml.append(joint(
        "cargo_payload_joint",
        "base_link",
        "cargo_payload_frame",
        ((TRAY_FRONT + TRAY_REAR) / 2, 0, PAYLOAD_Z),
    ))

    xml.append("</robot>\n")
    return "".join(xml)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for loaded, name in ((False, "aisha.urdf"), (True, "aisha_max_payload.urdf")):
        path = OUT / name
        path.write_text(build(loaded))
        print(f"wrote {path}")
