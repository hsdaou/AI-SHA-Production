#!/usr/bin/env python3
"""Convert genuine AliceVision textured meshes into native Omniverse USD layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction-root",
        type=Path,
        default=root / "tmp/phase7h_reconstruction",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "results/phase7k_photogrammetry_asset_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "scenes")
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "results/phase7k_photogrammetry_usd_build.json",
    )
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    points: list[list[float]] = []
    texture_coordinates: list[list[float]] = []
    faces: list[list[int]] = []
    face_texture_coordinates: list[list[int]] = []
    face_materials: list[str] = []
    material = "material_1001"
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                points.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("vt "):
                values = [float(value) for value in line.split()[1:3]]
                texture_coordinates.append(values)
            elif line.startswith("usemtl "):
                material = line.split()[1]
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) != 3:
                    raise ValueError(f"Phase 7K expects triangulated OBJ faces: {line[:100]}")
                vertex_face = []
                texture_face = []
                for token in tokens:
                    components = token.split("/")
                    vertex_face.append(int(components[0]) - 1)
                    texture_face.append(int(components[1]) - 1)
                faces.append(vertex_face)
                face_texture_coordinates.append(texture_face)
                face_materials.append(material)
    point_array = np.asarray(points, dtype=np.float32)
    texture_array = np.asarray(texture_coordinates, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int32)
    face_texture_array = np.asarray(face_texture_coordinates, dtype=np.int32)
    if point_array.size == 0 or texture_array.size == 0 or face_array.size == 0:
        raise RuntimeError(f"empty photogrammetry OBJ: {path}")

    # AliceVision is Y-up.  Author native survey layers as Z-up while keeping
    # their metric scale and lowering the reconstructed floor to Z=0.
    minimum_y = float(point_array[:, 1].min())
    converted = np.column_stack(
        (point_array[:, 0], -point_array[:, 2], point_array[:, 1] - minimum_y)
    ).astype(np.float32)
    # OBJ V increases upward; USD preview textures use the same convention here.
    face_varying_uv = texture_array[face_texture_array.reshape(-1)].astype(np.float32)
    return converted, face_array, face_varying_uv, face_materials


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangle = points[faces]
    face_normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normals = np.zeros_like(points, dtype=np.float32)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1.0e-12)
    return normals


def create_material(stage: Usd.Stage, cluster: str, material_name: str) -> UsdShade.Material:
    match = re.search(r"(1001|1002)$", material_name)
    if match is None:
        raise ValueError(f"unexpected AliceVision material name: {material_name}")
    index = match.group(1)
    name = f"Material{index}"
    path = f"/Survey/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/UVReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture = UsdShade.Shader.Define(stage, f"{path}/AlbedoTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(
            f"../textures/phase7k_photogrammetry/{cluster}_texture_{index}.jpg"
        )
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.64)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def build_cluster(cluster: str, obj: Path, output: Path, expected: dict) -> dict:
    points, faces, face_uvs, face_materials = parse_obj(obj)
    if len(points) != int(expected["vertices"]) or len(faces) != int(expected["faces"]):
        raise RuntimeError(f"photogrammetry geometry count drift in {cluster}")
    normals = vertex_normals(points, faces)

    stage = Usd.Stage.CreateNew(str(output.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Survey")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("aisha:phase", "PHASE7K")
    root.GetPrim().SetCustomDataByKey("aisha:cluster", cluster)
    root.GetPrim().SetCustomDataByKey("aisha:source", "AliceVision dense photogrammetry")
    root.GetPrim().SetCustomDataByKey("aisha:collision_enabled", False)
    root.GetPrim().SetCustomDataByKey("aisha:registration", "provisional_metric_similarity")
    UsdGeom.Scope.Define(stage, "/Survey/Looks")
    mesh = UsdGeom.Mesh.Define(stage, "/Survey/PhototexturedMesh")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32))
    )
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(Vt.Vec2fArray.FromNumpy(face_uvs))
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*points.min(axis=0).tolist()),
            Gf.Vec3f(*points.max(axis=0).tolist()),
        ]
    )
    face_material_array = np.asarray(face_materials)
    for material_name in sorted(set(face_materials)):
        material = create_material(stage, cluster, material_name)
        indices = np.flatnonzero(face_material_array == material_name).astype(np.int32)
        subset_name = re.sub(r"[^A-Za-z0-9_]", "_", material_name)
        subset = UsdGeom.Subset.Define(stage, f"/Survey/PhototexturedMesh/{subset_name}")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Save()
    stage = None
    return {
        "path": str(output.relative_to(PACKAGE_ROOT)),
        "sha256": sha256(output),
        "vertices": int(len(points)),
        "faces": int(len(faces)),
        "materials": sorted(set(face_materials)),
        "native_z_up_min_xyz_m": [round(float(value), 6) for value in points.min(axis=0)],
        "native_z_up_max_xyz_m": [round(float(value), 6) for value in points.max(axis=0)],
        "metric_scale": 1.0,
        "collision_enabled": False,
    }


def main() -> int:
    manifest_path = ARGS.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("passed"):
        raise RuntimeError("privacy-screened photogrammetry manifest is not accepted")
    sources = {
        "atrium_corridor": ARGS.reconstruction_root / "atrium/work/textured/texturedMesh.obj",
        "principal_office": ARGS.reconstruction_root
        / "principal/office/work/textured/texturedMesh.obj",
    }
    outputs = {}
    for cluster, source in sources.items():
        source = source.resolve()
        if sha256(source) != manifest["clusters"][cluster]["source_obj_sha256"]:
            raise RuntimeError(f"source OBJ hash drift: {source}")
        output = (ARGS.output_dir / f"phase7k_{cluster}_photogrammetry.usdc").resolve()
        outputs[cluster] = build_cluster(
            cluster,
            source,
            output,
            manifest["clusters"][cluster]["geometry"],
        )
        print(f"wrote {output}")

    report = {
        "report_type": "phase7k_photogrammetry_usd_build",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "passed": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "outputs": outputs,
        "total_vertices": sum(item["vertices"] for item in outputs.values()),
        "total_faces": sum(item["faces"] for item in outputs.values()),
        "clusters_kept_separate": True,
        "raw_dense_mesh_used_for_collision": False,
        "physical_release": False,
    }
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
