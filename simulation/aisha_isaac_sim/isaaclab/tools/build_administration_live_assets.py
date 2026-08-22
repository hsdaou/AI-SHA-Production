#!/usr/bin/env python3
"""Build composable administration-environment and AI-SHA shell USD assets."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ADMINISTRATION_SCENE = PACKAGE_ROOT / "scenes" / "administration.usd"
LOADED_ROBOT_USD = PACKAGE_ROOT / "usd" / "aisha_loaded.usd"
ENVIRONMENT_OUTPUT = PACKAGE_ROOT / "usd" / "administration_live_environment.usda"
SHELL_OUTPUT = PACKAGE_ROOT / "usd" / "aisha_presentation_shell.usda"
PRESENTATION_ROBOT_OUTPUT = PACKAGE_ROOT / "usd" / "aisha_loaded_presentation.usda"
REPORT_OUTPUT = PACKAGE_ROOT / "results" / "administration_live_assets_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_environment() -> None:
    if ENVIRONMENT_OUTPUT.exists():
        ENVIRONMENT_OUTPUT.unlink()
    stage = Usd.Stage.CreateNew(str(ENVIRONMENT_OUTPUT))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    environment = UsdGeom.Xform.Define(stage, "/Administration")
    stage.SetDefaultPrim(environment.GetPrim())
    relative_scene = os.path.relpath(ADMINISTRATION_SCENE, ENVIRONMENT_OUTPUT.parent)
    environment.GetPrim().GetReferences().AddReference(relative_scene, "/World")

    # The presentation scene contains a pose-replay robot and its own physics
    # scene. A live Isaac Lab environment supplies both, so exclude the nested
    # copies while retaining the full architecture, furniture and lighting.
    stage.OverridePrim("/Administration/AISHA").SetActive(False)
    stage.OverridePrim("/Administration/PhysicsScene").SetActive(False)
    environment.GetPrim().SetCustomDataByKey(
        "aisha:purpose", "live_policy_administration_environment"
    )
    environment.GetPrim().SetCustomDataByKey(
        "aisha:sourceSceneSha256", sha256(ADMINISTRATION_SCENE)
    )
    stage.GetRootLayer().customLayerData = {
        "aisha:sourceScene": relative_scene,
        "aisha:sourceSceneSha256": sha256(ADMINISTRATION_SCENE),
        "aisha:replayRobotExcluded": True,
        "aisha:nestedPhysicsSceneExcluded": True,
        "aisha:physicalRelease": False,
    }
    stage.GetRootLayer().Save()
def build_shell() -> None:
    if SHELL_OUTPUT.exists():
        SHELL_OUTPUT.unlink()
    stage = Usd.Stage.CreateNew(str(SHELL_OUTPUT))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    shell = UsdGeom.Xform.Define(stage, "/PresentationShell")
    stage.SetDefaultPrim(shell.GetPrim())

    def material(
        name: str,
        color: tuple[float, float, float],
        roughness: float,
        metallic: float = 0.0,
        emissive: tuple[float, float, float] | None = None,
    ):
        value = UsdShade.Material.Define(stage, f"/PresentationShell/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/PresentationShell/Looks/{name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        if emissive is not None:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        value.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return value

    white = material("AISHAWhite", (0.68, 0.73, 0.74), 0.20, 0.08)
    green = material("AISHAGreen", (0.03, 0.38, 0.24), 0.30, 0.05)
    black = material("AISHABlack", (0.015, 0.022, 0.025), 0.20, 0.20)
    metal = material("BrushedMetal", (0.38, 0.41, 0.43), 0.20, 0.72)
    led = material("AISHALed", (0.01, 0.24, 0.12), 0.16, emissive=(0.0, 1.35, 0.55))

    def bind(prim: Usd.Prim, visual: UsdShade.Material) -> None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(visual)

    def box(name: str, size: tuple[float, float, float], centre: tuple[float, float, float], visual) -> None:
        cube = UsdGeom.Cube.Define(stage, f"/PresentationShell/Geometry/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xform.AddScaleOp().Set(Gf.Vec3d(*size))
        bind(cube.GetPrim(), visual)

    def sphere(name: str, radius: float, centre: tuple[float, float, float], visual) -> None:
        shape = UsdGeom.Sphere.Define(stage, f"/PresentationShell/Geometry/{name}")
        shape.CreateRadiusAttr(float(radius))
        UsdGeom.Xformable(shape.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*centre))
        bind(shape.GetPrim(), visual)

    def ellipsoid(name: str, size: tuple[float, float, float], centre: tuple[float, float, float], visual) -> None:
        shape = UsdGeom.Sphere.Define(stage, f"/PresentationShell/Geometry/{name}")
        shape.CreateRadiusAttr(0.5)
        xform = UsdGeom.Xformable(shape.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xform.AddScaleOp().Set(Gf.Vec3d(*size))
        bind(shape.GetPrim(), visual)

    def cylinder(name: str, radius: float, height: float, centre: tuple[float, float, float], visual) -> None:
        shape = UsdGeom.Cylinder.Define(stage, f"/PresentationShell/Geometry/{name}")
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(height))
        UsdGeom.Xformable(shape.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*centre))
        bind(shape.GetPrim(), visual)

    ellipsoid("Body", (0.82, 0.59, 0.34), (0.02, 0.0, 0.40), white)
    ellipsoid("LowerBumper", (0.88, 0.64, 0.16), (0.03, 0.0, 0.27), black)
    box("Tray", (0.78, 0.57, 0.045), (-0.03, 0.0, 0.565), green)
    box("TrayPad", (0.68, 0.47, 0.012), (-0.05, 0.0, 0.594), black)
    box("Mast", (0.09, 0.11, 0.43), (0.40, 0.0, 0.76), white)
    ellipsoid("Head", (0.33, 0.40, 0.27), (0.50, 0.0, 0.96), white)
    box("Face", (0.022, 0.275, 0.145), (0.691, 0.0, 0.97), black)
    box("FaceStatus", (0.026, 0.060, 0.012), (0.705, -0.080, 0.920), led)
    ellipsoid("CameraAperture", (0.018, 0.040, 0.040), (0.706, 0.068, 0.985), metal)
    cylinder("LidarCollar", 0.075, 0.025, (0.50, 0.0, 1.115), metal)
    cylinder("Lidar", 0.055, 0.070, (0.50, 0.0, 1.160), black)
    box("LidarWindow", (0.038, 0.070, 0.018), (0.552, 0.0, 1.165), led)
    ellipsoid("LeftWheelCover", (0.38, 0.055, 0.23), (0.01, 0.301, 0.31), black)
    ellipsoid("RightWheelCover", (0.38, 0.055, 0.23), (0.01, -0.301, 0.31), black)
    box("LeftAccent", (0.42, 0.018, 0.055), (0.03, 0.306, 0.43), led)
    box("RightAccent", (0.42, 0.018, 0.055), (0.03, -0.306, 0.43), led)
    shell.GetPrim().SetCustomDataByKey("aisha:collision", "none_visual_only")
    shell.GetPrim().SetCustomDataByKey("aisha:purpose", "live_policy_visual_shell")
    stage.GetRootLayer().customLayerData = {
        "aisha:collision": "none_visual_only",
        "aisha:physicalRelease": False,
    }
    stage.GetRootLayer().Save()


def build_presentation_robot() -> None:
    """Compose the visual shell into base_link before PhysX parses the robot."""
    if PRESENTATION_ROBOT_OUTPUT.exists():
        PRESENTATION_ROBOT_OUTPUT.unlink()
    stage = Usd.Stage.CreateNew(str(PRESENTATION_ROBOT_OUTPUT))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    robot = UsdGeom.Xform.Define(stage, "/robot")
    stage.SetDefaultPrim(robot.GetPrim())
    loaded_stage = Usd.Stage.Open(str(LOADED_ROBOT_USD))
    if loaded_stage is None or not loaded_stage.GetDefaultPrim():
        raise RuntimeError(f"loaded robot has no default prim: {LOADED_ROBOT_USD}")
    loaded_default_prim = str(loaded_stage.GetDefaultPrim().GetPath())
    robot.GetPrim().GetReferences().AddReference(
        os.path.relpath(LOADED_ROBOT_USD, PRESENTATION_ROBOT_OUTPUT.parent),
        loaded_default_prim,
    )
    shell_link_path = Sdf.Path("/robot/base_link/presentation_shell_link")
    shell_joint_path = Sdf.Path("/robot/base_link/joints/presentation_shell_joint")
    shell_link = UsdGeom.Xform.Define(stage, shell_link_path)
    UsdPhysics.RigidBodyAPI.Apply(shell_link.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(shell_link.GetPrim())
    mass_api.CreateMassAttr(0.001)
    shell_link.GetPrim().GetReferences().AddReference(
        os.path.relpath(SHELL_OUTPUT, PRESENTATION_ROBOT_OUTPUT.parent), "/PresentationShell"
    )
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, shell_joint_path)
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path("/robot/base_link/base_link")])
    fixed_joint.CreateBody1Rel().SetTargets([shell_link_path])
    robot_description = stage.GetPrimAtPath("/robot/base_link")
    robot_description.GetRelationship("isaac:physics:robotLinks").AddTarget(shell_link_path)
    robot_description.GetRelationship("isaac:physics:robotJoints").AddTarget(shell_joint_path)
    robot.GetPrim().SetCustomDataByKey(
        "aisha:purpose", "loaded_physics_robot_with_visual_presentation_shell"
    )
    stage.GetRootLayer().customLayerData = {
        "aisha:loadedRobotSource": os.path.relpath(LOADED_ROBOT_USD, PRESENTATION_ROBOT_OUTPUT.parent),
        "aisha:loadedRobotDefaultPrim": loaded_default_prim,
        "aisha:presentationShellSource": os.path.relpath(SHELL_OUTPUT, PRESENTATION_ROBOT_OUTPUT.parent),
        "aisha:shellParent": "collisionless_fixed_link_to_base_link",
        "aisha:shellLinkMassKg": 0.001,
        "aisha:physicalRelease": False,
    }
    stage.GetRootLayer().Save()
    # Fabric registers renderable descendants when the articulation is loaded.
    # Flatten the robot and shell references into one asset so the added shell
    # is treated like the imported URDF visuals, not a late referenced child.
    flattened = stage.Flatten()
    if not flattened.Export(str(PRESENTATION_ROBOT_OUTPUT)):
        raise RuntimeError(f"could not export flattened presentation robot: {PRESENTATION_ROBOT_OUTPUT}")


def main() -> int:
    if not ADMINISTRATION_SCENE.is_file():
        raise FileNotFoundError(
            f"missing {ADMINISTRATION_SCENE}; build the administration presentation scene first"
        )
    ENVIRONMENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_environment()
    build_shell()
    build_presentation_robot()

    environment_stage = Usd.Stage.Open(str(ENVIRONMENT_OUTPUT))
    shell_stage = Usd.Stage.Open(str(SHELL_OUTPUT))
    presentation_robot_stage = Usd.Stage.Open(str(PRESENTATION_ROBOT_OUTPUT))
    checks = {
        "environment_reopens": environment_stage is not None,
        "shell_reopens": shell_stage is not None,
        "architecture_composes": bool(
            environment_stage and environment_stage.GetPrimAtPath("/Administration/Architecture")
        ),
        "replay_robot_inactive": bool(
            environment_stage
            and environment_stage.GetPrimAtPath("/Administration/AISHA")
            and not environment_stage.GetPrimAtPath("/Administration/AISHA").IsActive()
        ),
        "nested_physics_scene_inactive": bool(
            environment_stage
            and environment_stage.GetPrimAtPath("/Administration/PhysicsScene")
            and not environment_stage.GetPrimAtPath("/Administration/PhysicsScene").IsActive()
        ),
        "shell_geometry_composes": bool(
            shell_stage and shell_stage.GetPrimAtPath("/PresentationShell/Geometry/Body")
        ),
        "presentation_robot_reopens": presentation_robot_stage is not None,
        "loaded_robot_composes": bool(
            presentation_robot_stage
            and presentation_robot_stage.GetPrimAtPath("/robot/base_link/left_wheel_link")
        ),
        "shell_composes_on_fixed_link": bool(
            presentation_robot_stage
            and presentation_robot_stage.GetPrimAtPath(
                "/robot/base_link/presentation_shell_link/Geometry/Body"
            )
        ),
        "presentation_shell_fixed_joint_composes": bool(
            presentation_robot_stage
            and presentation_robot_stage.GetPrimAtPath(
                "/robot/base_link/joints/presentation_shell_joint"
            )
        ),
    }
    report = {
        "report_type": "administration_live_assets_build",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scene": str(ADMINISTRATION_SCENE),
        "source_scene_sha256": sha256(ADMINISTRATION_SCENE),
        "environment_usd": str(ENVIRONMENT_OUTPUT),
        "environment_usd_sha256": sha256(ENVIRONMENT_OUTPUT),
        "presentation_shell_usd": str(SHELL_OUTPUT),
        "presentation_shell_usd_sha256": sha256(SHELL_OUTPUT),
        "presentation_robot_usd": str(PRESENTATION_ROBOT_OUTPUT),
        "presentation_robot_usd_sha256": sha256(PRESENTATION_ROBOT_OUTPUT),
        "visual_upgrade": "administration_walkthrough_procedural_pbr_v1",
        "visual_upgrade_collision_impact": "none_visual_only",
        "checks": checks,
        "passed": all(checks.values()),
        "physical_release": False,
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ADMINISTRATION_LIVE_ENVIRONMENT={ENVIRONMENT_OUTPUT}")
    print(f"AISHA_PRESENTATION_SHELL={SHELL_OUTPUT}")
    print(f"AISHA_PRESENTATION_ROBOT={PRESENTATION_ROBOT_OUTPUT}")
    print(f"LIVE_ASSETS_REPORT={REPORT_OUTPUT}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
