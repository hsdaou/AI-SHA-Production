#!/usr/bin/env python3
"""Build deterministic flat-floor and parameterized threshold validation stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--payload", choices=("empty", "loaded"), default="loaded")
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from aisha_common import CONFIG_DIR, RESULTS_DIR, SCENES_DIR, USD_DIR, ensure_output_dirs, load_yaml, write_json


def define_physics_material(stage: Usd.Stage, path: str, values: dict[str, object]) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(values["static_friction"]))
    api.CreateDynamicFrictionAttr(float(values["dynamic_friction"]))
    api.CreateRestitutionAttr(float(values["restitution"]))
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr(str(values["friction_combine_mode"]))
    physx.CreateRestitutionCombineModeAttr(str(values["restitution_combine_mode"]))
    return material


def bind_physics_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def define_box(
    stage: Usd.Stage,
    path: str,
    *,
    size_xyz: tuple[float, float, float],
    centre_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    material: UsdShade.Material,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*centre_xyz))
    xform.AddScaleOp().Set(Gf.Vec3d(*size_xyz))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    bind_physics_material(cube.GetPrim(), material)
    return cube.GetPrim()


def configure_physics(stage: Usd.Stage, values: dict[str, object]) -> None:
    physics = values["physics"]
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(float(physics["gravity_mps2"]))
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateEnableCCDAttr(bool(physics["enable_ccd"]))
    physx.CreateEnableStabilizationAttr(bool(physics["enable_stabilization"]))
    physx.CreateEnableGPUDynamicsAttr(bool(physics["gpu_dynamics"]))
    physx.CreateBroadphaseTypeAttr(str(physics["broadphase"]))
    physx.CreateSolverTypeAttr(str(physics["solver"]))


def add_robot(stage: Usd.Stage, asset_path: str) -> None:
    robot = UsdGeom.Xform.Define(stage, "/World/AISHA")
    robot.GetPrim().GetReferences().AddReference(f"../usd/{Path(asset_path).name}")
    robot.GetPrim().SetCustomDataByKey("aisha:payloadVariant", ARGS.payload)


def bind_robot_contact_materials(stage: Usd.Stage, drive: UsdShade.Material, castor: UsdShade.Material) -> dict[str, list[str]]:
    bound = {"drive_wheel": [], "castor_low_friction": []}
    for prim in stage.TraverseAll():
        name = prim.GetName()
        if name in ("left_wheel_link", "right_wheel_link"):
            bind_physics_material(prim, drive)
            bound["drive_wheel"].append(str(prim.GetPath()))
        elif name in ("castor_fl_link", "castor_fr_link", "castor_rl_link", "castor_rr_link"):
            bind_physics_material(prim, castor)
            bound["castor_low_friction"].append(str(prim.GetPath()))
    return bound


def new_stage(path: str, config: dict[str, object]) -> tuple[Usd.Stage, dict[str, UsdShade.Material]]:
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    configure_physics(stage, config)
    materials = {
        name: define_physics_material(stage, f"/World/Looks/{name}", values)
        for name, values in config["materials"].items()
    }
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(750.0)
    sun.CreateAngleAttr(0.5)
    return stage, materials


def build_flat(config: dict[str, object], asset_path: str) -> dict[str, object]:
    path = SCENES_DIR / "validation_flat.usd"
    if path.exists():
        path.unlink()
    stage, materials = new_stage(str(path), config)
    define_box(
        stage,
        "/World/Floor",
        size_xyz=(30.0, 20.0, 0.10),
        centre_xyz=(7.5, 0.0, -0.05),
        color=(0.72, 0.74, 0.77),
        material=materials["polished_tile"],
    )
    add_robot(stage, asset_path)
    bindings = bind_robot_contact_materials(stage, materials["drive_wheel"], materials["castor_low_friction"])
    stage.GetRootLayer().customLayerData = {
        "aisha:scenePurpose": "deterministic_flat_floor_validation",
        "aisha:payloadVariant": ARGS.payload,
        "aisha:materialStatus": "simulation_starting_assumptions",
    }
    stage.GetRootLayer().Save()
    return {"path": str(path), "material_bindings": bindings}


def build_thresholds(config: dict[str, object], asset_path: str) -> dict[str, object]:
    path = SCENES_DIR / "validation_thresholds.usd"
    if path.exists():
        path.unlink()
    stage, materials = new_stage(str(path), config)
    define_box(
        stage,
        "/World/Floor",
        size_xyz=(12.0, 9.0, 0.10),
        centre_xyz=(3.0, 0.0, -0.05),
        color=(0.72, 0.74, 0.77),
        material=materials["polished_tile"],
    )
    cases = []
    for index, height_mm in enumerate((5, 10, 20)):
        y = (index - 1) * 2.5
        prim = define_box(
            stage,
            f"/World/Thresholds/H{height_mm:02d}mm",
            size_xyz=(0.10, 1.80, height_mm / 1000.0),
            centre_xyz=(2.5, y, height_mm / 2000.0),
            color=(0.90, 0.55, 0.15),
            material=materials["polished_tile"],
        )
        prim.SetCustomDataByKey("aisha:heightMm", height_mm)
        prim.SetCustomDataByKey("aisha:status", "parameterized_unmeasured_case")
        cases.append({"height_mm": height_mm, "y_m": y})
    add_robot(stage, asset_path)
    bind_robot_contact_materials(stage, materials["drive_wheel"], materials["castor_low_friction"])
    stage.GetRootLayer().customLayerData = {
        "aisha:scenePurpose": "parameterized_threshold_geometry_only",
        "aisha:contactValidationBlocked": "requires articulated compliant asset and measured caster/spring properties",
    }
    stage.GetRootLayer().Save()
    return {"path": str(path), "cases": cases, "validation_status": "blocked_high_fidelity_asset_required"}


def main() -> int:
    ensure_output_dirs()
    config = load_yaml(CONFIG_DIR / "physics_materials.yaml")
    asset = USD_DIR / ("aisha_empty.usd" if ARGS.payload == "empty" else "aisha_loaded.usd")
    if not asset.exists():
        raise FileNotFoundError(f"missing {asset}; run scripts/import_urdf.py first")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "payload": ARGS.payload,
        "flat": build_flat(config, str(asset)),
        "thresholds": build_thresholds(config, str(asset)),
    }
    output = RESULTS_DIR / "scene_build_report.json"
    write_json(output, report)
    print(f"wrote {report['flat']['path']}")
    print(f"wrote {report['thresholds']['path']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
