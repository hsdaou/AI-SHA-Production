#!/usr/bin/env python3
"""Import both canonical AI-SHA URDFs into Isaac Sim 5.1 USD assets.

Run with Isaac Sim's Python, for example:
  /path/to/isaac-sim/python.sh scripts/import_urdf.py --headless
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a window")
    parser.add_argument("--only", choices=("empty", "loaded"), help="import one payload variant")
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.version import get_version
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from aisha_common import CONFIG_DIR, RESULTS_DIR, URDF_DIR, USD_DIR, ensure_output_dirs, load_yaml, sha256_file, write_json


EXPECTED = {
    "empty": {"urdf": "aisha.urdf", "usd": "aisha_empty.usd", "mass_kg": 59.25},
    "loaded": {"urdf": "aisha_max_payload.urdf", "usd": "aisha_loaded.usd", "mass_kg": 69.25},
}
DRIVEN_JOINTS = ("left_wheel_joint", "right_wheel_joint")
REQUIRED_FRAME_NAMES = (
    "base_link",
    "lidar_link",
    "front_lidar_link",
    "front_camera_link",
    "front_camera_optical_frame",
    "imu_link",
    "cargo_payload_frame",
)


def set_import_option(config: object, name: str, value: object) -> None:
    setter = getattr(config, f"set_{name}", None)
    if callable(setter):
        setter(value)
    else:
        setattr(config, name, value)


def matching_prims(stage: Usd.Stage, names: tuple[str, ...]) -> dict[str, list[Usd.Prim]]:
    matches = {name: [] for name in names}
    for prim in stage.TraverseAll():
        if prim.GetName() in matches:
            matches[prim.GetName()].append(prim)
    return matches


def configure_wheel_drives(stage: Usd.Stage, damping: float, effort_nm: float) -> dict[str, str]:
    matches = matching_prims(stage, DRIVEN_JOINTS)
    paths: dict[str, str] = {}
    for name, prims in matches.items():
        if len(prims) != 1:
            raise RuntimeError(f"expected one {name} prim after import, found {len(prims)}")
        prim = prims[0]
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTargetVelocityAttr(0.0).Set(0.0)
        drive.CreateStiffnessAttr(0.0).Set(0.0)
        drive.CreateDampingAttr(damping).Set(damping)
        drive.CreateMaxForceAttr(effort_nm).Set(effort_nm)
        paths[name] = str(prim.GetPath())
    return paths


def configure_generated_physics_layer(
    source: Path,
    output: Path,
    materials: dict[str, dict[str, object]],
    contact_tuning: dict[str, object],
) -> tuple[list[str], dict[str, list[str]]]:
    """Create empty visual targets omitted by Isaac's modular URDF writer.

    Importer 2.4.30 authors references for mass-only frame links even though it
    does not create the referenced target. Empty Xforms preserve those frames
    without adding fake visible or collision geometry.
    """
    root = ET.parse(source).getroot()
    mass_only_names = [
        link.attrib["name"]
        for link in root.findall("link")
        if link.find("visual") is None
    ]
    physics_layer = output.parent / "configuration" / f"{output.stem}_physics.usd"
    if not physics_layer.exists():
        return [], {}
    stage = Usd.Stage.Open(str(physics_layer))
    if stage is None:
        raise RuntimeError(f"could not open generated importer layer {physics_layer}")
    for name in mass_only_names:
        UsdGeom.Xform.Define(stage, f"/visuals/{name}")

    bindings = {"drive_wheel": [], "castor_low_friction": []}
    link_materials = {
        "left_wheel_link": "drive_wheel",
        "right_wheel_link": "drive_wheel",
        "castor_fl_link": "castor_low_friction",
        "castor_fr_link": "castor_low_friction",
        "castor_rl_link": "castor_low_friction",
        "castor_rr_link": "castor_low_friction",
    }
    # Each collider scope is referenced independently by the importer. Keep the
    # relationship target inside that scope so USD reference encapsulation does
    # not discard the physics material binding.
    for link_name, material_name in link_materials.items():
        values = materials[material_name]
        material = UsdShade.Material.Define(stage, f"/colliders/{link_name}/PhysicsMaterial")
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(float(values["static_friction"]))
        api.CreateDynamicFrictionAttr(float(values["dynamic_friction"]))
        api.CreateRestitutionAttr(float(values["restitution"]))
        physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx.CreateFrictionCombineModeAttr(str(values["friction_combine_mode"]))
        physx.CreateRestitutionCombineModeAttr(str(values["restitution_combine_mode"]))
        scope = stage.GetPrimAtPath(f"/colliders/{link_name}")
        if not scope:
            raise RuntimeError(f"missing generated collider scope for {link_name}")
        for prim in Usd.PrimRange(scope):
            if prim.IsA(UsdGeom.Boundable):
                collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision_api.CreateContactOffsetAttr(float(contact_tuning["contact_offset_m"]))
                if material_name == "drive_wheel":
                    collision_api.CreateRestOffsetAttr(float(contact_tuning["drive_wheel_rest_offset_m"]))
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                bindings[material_name].append(str(prim.GetPath()))
    stage.GetRootLayer().Save()
    return mass_only_names, bindings


def inspect_asset(stage: Usd.Stage, kind: str, root_path: str, source: Path, output: Path) -> dict[str, object]:
    frame_matches = matching_prims(stage, REQUIRED_FRAME_NAMES)
    joint_matches = matching_prims(stage, DRIVEN_JOINTS)
    mass_values: list[dict[str, object]] = []
    for prim in stage.TraverseAll():
        attr = prim.GetAttribute("physics:mass")
        value = attr.Get() if attr else None
        if value is not None:
            mass_values.append({"prim": str(prim.GetPath()), "mass_kg": float(value)})

    missing_frames = [name for name, prims in frame_matches.items() if not prims]
    errors: list[str] = []
    if missing_frames:
        errors.append(f"missing frame prims: {', '.join(missing_frames)}")
    for name, prims in joint_matches.items():
        if len(prims) != 1:
            errors.append(f"expected one {name}, found {len(prims)}")

    return {
        "variant": kind,
        "source_urdf": str(source),
        "source_urdf_sha256": sha256_file(source),
        "output_usd": str(output),
        "expected_design_mass_kg": EXPECTED[kind]["mass_kg"],
        "articulation_root": root_path,
        "frame_prims": {name: [str(prim.GetPath()) for prim in prims] for name, prims in frame_matches.items()},
        "mass_attributes": mass_values,
        "mass_attribute_sum_kg": round(sum(item["mass_kg"] for item in mass_values), 6),
        "errors": errors,
    }


def import_one(
    kind: str,
    drive_config: dict[str, object],
    physics_config: dict[str, object],
) -> dict[str, object]:
    spec = EXPECTED[kind]
    source = URDF_DIR / str(spec["urdf"])
    output = USD_DIR / str(spec["usd"])
    if output.exists():
        output.unlink()

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    options = drive_config["isaac_sim_import"]
    set_import_option(import_config, "merge_fixed_joints", bool(options["merge_fixed_joints"]))
    set_import_option(import_config, "import_inertia_tensor", bool(options["import_inertia_tensor"]))
    set_import_option(import_config, "fix_base", bool(options["fix_base"]))
    set_import_option(import_config, "self_collision", bool(options["self_collision"]))
    set_import_option(import_config, "collision_from_visuals", bool(options["collision_from_visuals"]))
    set_import_option(
        import_config,
        "replace_cylinders_with_capsules",
        bool(options["replace_cylinders_with_capsules"]),
    )
    set_import_option(import_config, "convex_decomp", False)
    set_import_option(import_config, "distance_scale", 1.0)
    set_import_option(import_config, "default_drive_type", 2)  # force drive
    set_import_option(import_config, "default_drive_strength", 0.0)
    set_import_option(import_config, "default_position_drive_damping", float(options["initial_wheel_drive_damping"]))

    status, root_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(source),
        import_config=import_config,
        dest_path=str(output),
        get_articulation_root=True,
    )
    if not status or not root_path:
        raise RuntimeError(f"URDF import failed for {source}")

    for _ in range(3):
        APP.update()
    repaired_visual_targets, physics_material_bindings = configure_generated_physics_layer(
        source,
        output,
        physics_config["materials"],
        physics_config["contact_tuning"],
    )
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"could not reopen {output}")
    drive_paths = configure_wheel_drives(
        stage,
        float(options["initial_wheel_drive_damping"]),
        float(drive_config["motor"]["urdf_effort_limit_nm"]),
    )
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        # USD default prims must be root-level. The importer returns the nested
        # articulation root (/<robot>/base_link), so select /<robot> here.
        root_level_path = Sdf.Path(f"/{Sdf.Path(root_path).pathString.strip('/').split('/')[0]}")
        candidate = stage.GetPrimAtPath(root_level_path)
        if not candidate:
            raise RuntimeError(f"could not find imported robot root {root_level_path}")
        stage.SetDefaultPrim(candidate)
    stage.GetRootLayer().customLayerData = {
        **dict(stage.GetRootLayer().customLayerData),
        "aisha:sourceUrdfSha256": sha256_file(source),
        "aisha:variant": kind,
        "aisha:designMassKg": float(spec["mass_kg"]),
        "aisha:castorModel": "fixed_sphere_low_friction_proxy",
        "aisha:driveCompliance": "rigid_baseline",
    }
    stage.GetRootLayer().Save()
    report = inspect_asset(stage, kind, str(root_path), source, output)
    report["wheel_drive_prims"] = drive_paths
    report["empty_visual_reference_targets_repaired"] = repaired_visual_targets
    report["physics_material_bindings"] = physics_material_bindings
    report["contact_tuning"] = dict(physics_config["contact_tuning"])
    if report["errors"]:
        raise RuntimeError("; ".join(report["errors"]))
    return report


def main() -> int:
    ensure_output_dirs()
    enable_extension("isaacsim.asset.importer.urdf")
    APP.update()
    drive_config = load_yaml(CONFIG_DIR / "aisha_drive.yaml")
    physics_config = load_yaml(CONFIG_DIR / "physics_materials.yaml")
    variants = (ARGS.only,) if ARGS.only else ("empty", "loaded")
    report: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "isaac_sim_version": get_version()[0],
        "importer": "isaacsim.asset.importer.urdf",
        "variants": [],
    }
    for kind in variants:
        item = import_one(kind, drive_config, physics_config)
        report["variants"].append(item)
        print(f"imported {kind}: {item['output_usd']}")
    output = RESULTS_DIR / "import_report.json"
    write_json(output, report)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
