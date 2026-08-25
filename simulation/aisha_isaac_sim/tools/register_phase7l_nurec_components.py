#!/usr/bin/env python3
"""Register the Phase 7L Gaussian components to the administration world.

The walkthrough COLMAP model contains two useful disconnected reconstructions:
component 1 covers the atrium and administration corridors, while component 2
contains a second atrium pass and the captured Principal office.  This tool
matches common atrium SIFT tracks, estimates a robust Sim(3) transform between
the components, then anchors the Principal doorway/direction to the frozen
metric presentation world.

The result is a visual registration only.  It must never replace the validated
collision, LiDAR, localization or policy world.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent.parent
RNG_SEED = 7
MAIN_FRAMES = (1, 10, 20, 30, 40, 45, 50, 60, 70, 80)
PRINCIPAL_ATRIUM_FRAMES = (323, 330, 337, 344, 351)


@dataclass
class Model:
    points: dict[int, np.ndarray]
    observations: dict[int, np.ndarray]
    camera_centres: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "tmp/phase7k_photogrammetry_2fps",
    )
    parser.add_argument(
        "--metric-config",
        type=Path,
        default=ROOT / "config/measured_administration_presentation_2026-08-23.yaml",
    )
    parser.add_argument(
        "--plan-config",
        type=Path,
        default=ROOT / "config/administration_assumptions.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase7l_nurec_registration.json",
    )
    parser.add_argument("--camera-height-m", type=float, default=1.55)
    parser.add_argument("--main-component", type=int, default=1)
    parser.add_argument("--principal-component", type=int, default=2)
    return parser.parse_args()


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def frame_number(name: str) -> int:
    match = re.search(r"(\d+)", Path(name).stem)
    if match is None:
        raise ValueError(f"image name has no frame number: {name}")
    return int(match.group(1))


def load_model(source: Path, component: int) -> Model:
    text_dir = source / f"sparse/{component}_txt"
    points: dict[int, np.ndarray] = {}
    for line in (text_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        points[int(fields[0])] = np.asarray(fields[1:4], dtype=np.float64)

    observations: dict[int, np.ndarray] = {}
    camera_centres: dict[int, np.ndarray] = {}
    lines = (text_dir / "images.txt").read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        fields = lines[index].split()
        if (
            len(fields) >= 10
            and fields[0].isdigit()
            and fields[9].lower().endswith((".jpg", ".jpeg", ".png"))
        ):
            frame = frame_number(fields[9])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            translation = np.asarray(fields[5:8], dtype=np.float64)
            rotation = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
            camera_centres[frame] = -rotation.T @ translation
            point_fields = lines[index + 1].split()
            observations[frame] = np.asarray(
                [int(point_fields[offset]) for offset in range(2, len(point_fields), 3)],
                dtype=np.int64,
            )
            index += 2
        else:
            index += 1
    return Model(points, observations, camera_centres)


def gravity_from_cameras(model: Model, frames: tuple[int, ...] | None = None) -> np.ndarray:
    selected = [
        centre
        for frame, centre in model.camera_centres.items()
        if frames is None or frame in frames
    ]
    covariance = np.cov(np.asarray(selected).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    up = eigenvectors[:, 0]
    if up[1] < 0.0:
        up = -up
    return up / np.linalg.norm(up)


def horizontal_basis(up: np.ndarray) -> np.ndarray:
    first = np.asarray((1.0, 0.0, 0.0))
    first -= up * float(np.dot(up, first))
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, up))


def load_feature(
    connection: sqlite3.Connection,
    image_id: int,
    limit: int = 6000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns, blob = connection.execute(
        "SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,)
    ).fetchone()
    descriptors = np.frombuffer(blob, np.uint8).reshape(rows, columns)
    key_rows, key_columns, key_blob = connection.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)
    ).fetchone()
    keypoints = np.frombuffer(key_blob, np.float32).reshape(key_rows, key_columns)
    if key_columns >= 6:
        determinant = keypoints[:, 2] * keypoints[:, 5] - keypoints[:, 3] * keypoints[:, 4]
        score = np.sqrt(np.maximum(np.abs(determinant), 1e-9))
    else:
        score = np.ones(rows)
    indices = np.argsort(score)[-min(limit, rows) :]
    return descriptors[indices].astype(np.float32), keypoints[indices, :2], indices


def common_track_pairs(
    database: Path,
    main: Model,
    principal: Model,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    connection = sqlite3.connect(database)
    image_ids = {
        frame_number(name): image_id
        for image_id, name in connection.execute("SELECT image_id, name FROM images")
    }
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def feature(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if frame not in cache:
            cache[frame] = load_feature(connection, image_ids[frame])
        return cache[frame]

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=6),
        dict(checks=96),
    )
    cv2.setRNGSeed(RNG_SEED)
    candidates: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int, int]] = {}
    pair_stats: list[dict[str, int]] = []
    for main_frame in MAIN_FRAMES:
        if main_frame not in main.observations:
            continue
        for principal_frame in PRINCIPAL_ATRIUM_FRAMES:
            if principal_frame not in principal.observations:
                continue
            main_desc, main_keypoints, main_indices = feature(main_frame)
            principal_desc, principal_keypoints, principal_indices = feature(principal_frame)
            raw = matcher.knnMatch(main_desc, principal_desc, k=2)
            ratio_matches = [
                first for first, second in raw if first.distance < 0.76 * second.distance
            ]
            if len(ratio_matches) < 8:
                continue
            main_xy = np.float32([main_keypoints[item.queryIdx] for item in ratio_matches])
            principal_xy = np.float32(
                [principal_keypoints[item.trainIdx] for item in ratio_matches]
            )
            _, mask = cv2.findFundamentalMat(
                main_xy,
                principal_xy,
                cv2.FM_RANSAC,
                1.5,
                0.999,
            )
            if mask is None:
                continue
            usable = 0
            for match, is_inlier in zip(ratio_matches, mask.ravel() != 0):
                if not is_inlier:
                    continue
                main_index = int(main_indices[match.queryIdx])
                principal_index = int(principal_indices[match.trainIdx])
                if (
                    main_index >= len(main.observations[main_frame])
                    or principal_index >= len(principal.observations[principal_frame])
                ):
                    continue
                main_id = int(main.observations[main_frame][main_index])
                principal_id = int(principal.observations[principal_frame][principal_index])
                if main_id < 0 or principal_id < 0:
                    continue
                candidates[(main_id, principal_id)] = (
                    main.points[main_id],
                    principal.points[principal_id],
                    main_frame,
                    principal_frame,
                )
                usable += 1
            pair_stats.append(
                {
                    "main_frame": main_frame,
                    "principal_frame": principal_frame,
                    "ratio_matches": len(ratio_matches),
                    "epipolar_inliers": int(np.count_nonzero(mask)),
                    "three_dimensional_pairs": usable,
                }
            )
    connection.close()
    values = list(candidates.values())
    main_points = np.asarray([value[0] for value in values])
    principal_points = np.asarray([value[1] for value in values])
    return main_points, principal_points, pair_stats


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centre = source - source_mean
    target_centre = target - target_mean
    variance = float(np.sum(source_centre * source_centre) / len(source))
    if variance < 1e-12:
        raise ValueError("degenerate similarity sample")
    covariance = target_centre.T @ source_centre / len(source)
    left, singular, right = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(left @ right) < 0.0:
        reflection[-1, -1] = -1.0
    rotation = left @ reflection @ right
    scale = float(np.trace(np.diag(singular) @ reflection) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    best: tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for _ in range(40000):
        indices = rng.choice(len(source), 3, replace=False)
        try:
            scale, rotation, translation = fit_similarity(source[indices], target[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not 0.2 < scale < 5.0:
            continue
        errors = np.linalg.norm(
            (scale * (rotation @ source.T)).T + translation - target,
            axis=1,
        )
        inliers = errors < 0.12
        if not np.any(inliers):
            continue
        score = (int(inliers.sum()), -float(np.median(errors[inliers])))
        if best is None or score[:2] > best[:2]:
            best = (score[0], score[1], scale, rotation, translation, inliers)
    if best is None or best[0] < 10:
        raise RuntimeError("could not find a supported cross-component similarity")
    inliers = best[5]
    for threshold in (0.20, 0.15, 0.12, 0.10, 0.08):
        scale, rotation, translation = fit_similarity(source[inliers], target[inliers])
        errors = np.linalg.norm(
            (scale * (rotation @ source.T)).T + translation - target,
            axis=1,
        )
        candidate = errors < threshold
        if int(candidate.sum()) >= 10:
            inliers = candidate
    scale, rotation, translation = fit_similarity(source[inliers], target[inliers])
    errors = np.linalg.norm(
        (scale * (rotation @ source.T)).T + translation - target,
        axis=1,
    )
    return scale, rotation, translation, errors, inliers


def affine_matrix(linear: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    return np.round(matrix, 12).tolist()


def usd_row_matrix(linear: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, :3] = linear.T
    matrix[3, :3] = translation
    return np.round(matrix, 12).tolist()


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    main_model = load_model(source, args.main_component)
    principal_model = load_model(source, args.principal_component)
    main_points, principal_points, pair_stats = common_track_pairs(
        source / "database.db",
        main_model,
        principal_model,
    )
    relative_scale, relative_rotation, relative_translation, errors, inliers = robust_similarity(
        principal_points,
        main_points,
    )

    office_frames = tuple(
        frame
        for frame in principal_model.camera_centres
        if 350 <= frame <= 430 or 881 <= frame <= 897
    )
    principal_up = gravity_from_cameras(principal_model, office_frames)
    main_up = gravity_from_cameras(main_model)
    basis = horizontal_basis(principal_up)

    plan = yaml.safe_load(args.plan_config.read_text(encoding="utf-8"))
    metric = yaml.safe_load(args.metric_config.read_text(encoding="utf-8"))
    door_xy = np.asarray(
        metric["measured_visual_twin"]["registration"]["principal"]["world_anchor_xy_m"],
        dtype=np.float64,
    )
    turn = next(item for item in plan["route"]["waypoints"] if item["id"] == "principal_turn")
    turn_xy = np.asarray((turn["x_m"], turn["y_m"]), dtype=np.float64)
    ceiling_height = float(plan["plan_geometry"]["ceiling_height_m"]["value"])

    principal_heights = np.asarray(list(principal_model.points.values())) @ principal_up
    height_low, height_high = np.quantile(principal_heights, (0.01, 0.95))
    reconstructed_height_span = float(height_high - height_low)
    absolute_scale = ceiling_height / reconstructed_height_span

    # Frame 351 is still in the secretary/anteroom.  Frame 358 is the first
    # camera centre at the captured Principal office transition and is the
    # correct semantic counterpart of the registered RoomPlan entrance gap.
    door_frame = 358
    approach_frame = 337
    door_camera = principal_model.camera_centres[door_frame]
    approach_camera = principal_model.camera_centres[approach_frame]
    door_basis = basis.T @ door_camera
    approach_basis = basis.T @ approach_camera
    source_direction = approach_basis[:2] - door_basis[:2]
    world_direction = turn_xy - door_xy
    yaw = math.atan2(world_direction[1], world_direction[0]) - math.atan2(
        source_direction[1], source_direction[0]
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    world_from_basis = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    principal_linear = absolute_scale * world_from_basis @ basis.T
    world_door_camera = np.asarray((door_xy[0], door_xy[1], args.camera_height_m))
    principal_translation = world_door_camera - principal_linear @ door_camera

    # The robust match solves principal -> main.  Invert it so the main asset
    # can be composed into the same metric world as the Principal component.
    main_to_principal_linear = relative_rotation.T / relative_scale
    main_to_principal_translation = -relative_rotation.T @ relative_translation / relative_scale
    main_linear = principal_linear @ main_to_principal_linear
    main_translation = (
        principal_linear @ main_to_principal_translation + principal_translation
    )

    main_inlier_world = (main_linear @ main_points[inliers].T).T + main_translation
    principal_inlier_world = (
        principal_linear @ principal_points[inliers].T
    ).T + principal_translation
    world_errors = np.linalg.norm(main_inlier_world - principal_inlier_world, axis=1)
    gravity_residual = angle_degrees(main_linear @ main_up, principal_linear @ principal_up)
    approach_world = principal_linear @ approach_camera + principal_translation
    approach_anchor_residual = float(np.linalg.norm(approach_world[:2] - turn_xy))
    main_world_scale = float(np.mean(np.linalg.svd(main_linear, compute_uv=False)))
    principal_world_scale = float(np.mean(np.linalg.svd(principal_linear, compute_uv=False)))
    passed = bool(
        len(world_errors) >= 10
        and float(np.median(world_errors)) <= 0.10
        and float(np.quantile(world_errors, 0.95)) <= 0.25
        and gravity_residual <= 3.0
        and approach_anchor_residual <= 0.35
    )

    report = {
        "report_type": "phase7l_nurec_metric_registration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_provisional_presentation_registration" if passed else "failed",
        "passed": passed,
        "source": {
            "colmap_root": portable_path(source),
            "database": portable_path(source / "database.db"),
            "main_component": args.main_component,
            "principal_component": args.principal_component,
            "main_semantics": "atrium, reception and administration corridors",
            "principal_semantics": "shared atrium pass, captured Principal doorway and interior",
            "principal_component_registered_images": len(principal_model.camera_centres),
            "raw_media_committed": False,
        },
        "cross_component_registration": {
            "method": "SIFT track correspondences in shared atrium views plus deterministic Sim(3) RANSAC",
            "candidate_3d_pairs": len(main_points),
            "accepted_3d_pairs": int(inliers.sum()),
            "principal_to_main_scale": relative_scale,
            "principal_to_main_rotation": np.round(relative_rotation, 12).tolist(),
            "principal_to_main_translation": np.round(relative_translation, 12).tolist(),
            "native_residual_median_component1_units": float(np.median(errors[inliers])),
            "matched_pair_statistics": sorted(
                pair_stats,
                key=lambda item: item["three_dimensional_pairs"],
                reverse=True,
            )[:12],
        },
        "metric_anchor": {
            "principal_door_transition_source_frame": door_frame,
            "approach_direction_source_frame": approach_frame,
            "principal_door_world_xy_m": door_xy.tolist(),
            "principal_turn_world_xy_m": turn_xy.tolist(),
            "camera_height_m": args.camera_height_m,
            "camera_height_status": "presentation_assumption",
            "ceiling_height_m": ceiling_height,
            "ceiling_height_status": plan["plan_geometry"]["ceiling_height_m"]["status"],
            "principal_height_quantiles": [0.01, 0.95],
            "reconstructed_height_span_units": reconstructed_height_span,
            "principal_absolute_scale_m_per_unit": absolute_scale,
            "world_yaw_deg": math.degrees(yaw),
        },
        "world_transforms": {
            "principal_component": {
                "column_vector_matrix": affine_matrix(principal_linear, principal_translation),
                "usd_gf_row_vector_matrix": usd_row_matrix(principal_linear, principal_translation),
                "uniform_scale_m_per_native_unit": principal_world_scale,
            },
            "main_component": {
                "column_vector_matrix": affine_matrix(main_linear, main_translation),
                "usd_gf_row_vector_matrix": usd_row_matrix(main_linear, main_translation),
                "uniform_scale_m_per_native_unit": main_world_scale,
            },
        },
        "validation": {
            "shared_atrium_world_residual_median_m": float(np.median(world_errors)),
            "shared_atrium_world_residual_p95_m": float(np.quantile(world_errors, 0.95)),
            "shared_atrium_world_residual_max_m": float(np.max(world_errors)),
            "gravity_alignment_residual_deg": gravity_residual,
            "principal_door_anchor_residual_m": 0.0,
            "approach_direction_residual_deg": 0.0,
            "principal_turn_anchor_residual_m": approach_anchor_residual,
            "thresholds": {
                "minimum_shared_3d_pairs": 10,
                "median_residual_max_m": 0.10,
                "p95_residual_max_m": 0.25,
                "gravity_residual_max_deg": 3.0,
                "principal_turn_anchor_residual_max_m": 0.35,
            },
        },
        "layer_contract": {
            "nurec_is_visual_only": True,
            "frozen_phase7i_collision_navigation_unchanged": True,
            "raw_gaussian_used_for_collision_or_lidar": False,
            "physical_localization_credit": False,
            "physical_release": False,
            "registration_classification": "provisional presentation registration, not a certified survey control",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
