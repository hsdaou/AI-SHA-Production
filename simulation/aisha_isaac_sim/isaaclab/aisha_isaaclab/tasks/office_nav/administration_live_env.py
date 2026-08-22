"""Run the Block A sensor policy live inside the full administration USD."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from pxr import Usd, UsdPhysics

from aisha_isaaclab.assets import AISHA_PRESENTATION_CFG, SIM_PACKAGE_ROOT
from aisha_isaaclab.tasks.office_nav.block_a_sensor_env import (
    AishaBlockASensorEnv,
    AishaBlockASensorEnvCfg,
)
from aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env import (
    PHASE2_GOAL_TOLERANCES,
    TURN_DIRECTION_HINTS,
)


ADMINISTRATION_LIVE_USD = SIM_PACKAGE_ROOT / "usd" / "administration_live_environment.usda"
PRESENTATION_ROBOT_USD = SIM_PACKAGE_ROOT / "usd" / "aisha_loaded_presentation.usda"

def administration_collision_raycast_targets() -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Return only collision/navigation shapes, excluding presentation-only meshes."""
    if not ADMINISTRATION_LIVE_USD.is_file():
        return []
    stage = Usd.Stage.Open(str(ADMINISTRATION_LIVE_USD))
    if stage is None:
        raise RuntimeError(f"could not open live administration asset: {ADMINISTRATION_LIVE_USD}")
    prefix = "/Administration"
    targets = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if not path.startswith(prefix + "/") or path.startswith(prefix + "/AISHA"):
            continue
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        targets.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Administration" + path[len(prefix) :],
                is_shared=False,
                merge_prim_meshes=True,
                track_mesh_transforms=False,
            )
        )
    if not targets:
        raise RuntimeError(f"no collision raycast targets found in {ADMINISTRATION_LIVE_USD}")
    return targets


@configclass
class AishaAdministrationLiveSceneCfg(InteractiveSceneCfg):
    """Full visual/collision scene, articulated AI-SHA and policy-compatible rays."""

    administration = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Administration",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ADMINISTRATION_LIVE_USD)),
    )
    # The shell is a collisionless, negligible-mass fixed link so Fabric gives
    # it a live transform without changing the policy action space.
    robot = AISHA_PRESENTATION_CFG
    crown_lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/lidar_link",
        update_period=0.10,
        offset=MultiMeshRayCasterCfg.OffsetCfg(),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=10.0,
        ),
        max_distance=10.0,
        # Camera-facing PBR finishes and decorative meshes are intentionally
        # excluded. Otherwise a visual upgrade silently changes policy input.
        mesh_prim_paths=administration_collision_raycast_targets(),
        reference_meshes=False,
        debug_vis=False,
    )
    # Separate low forward scan for Nav2 obstacle marking. It is deliberately
    # absent from the learned policy observation, preserving the frozen policy
    # contract while exposing the real robot's intended low-front sensor role.
    front_lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_lidar_link",
        update_period=0.05,
        offset=MultiMeshRayCasterCfg.OffsetCfg(),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-60.0, 60.0),
            horizontal_res=5.0,
        ),
        max_distance=10.0,
        mesh_prim_paths=administration_collision_raycast_targets(),
        reference_meshes=False,
        debug_vis=False,
    )


@configclass
class AishaAdministrationLiveEnvCfg(AishaBlockASensorEnvCfg):
    """Policy-compatible single-scene configuration for live integration."""

    scene: AishaAdministrationLiveSceneCfg = AishaAdministrationLiveSceneCfg(
        num_envs=1,
        env_spacing=55.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    # Isolated live diagnostics should reproduce the heading inherited from
    # the preceding route segment, just like the Phase 2 turn curriculum.
    start_heading_mode = "incoming"
    goal_tolerance_m_by_segment = PHASE2_GOAL_TOLERANCES
    turn_direction_hint_rad_by_segment = TURN_DIRECTION_HINTS

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (-4.5, 4.5, 5.5)
        self.viewer.lookat = (0.0, 0.0, 0.55)


class AishaAdministrationLiveEnv(AishaBlockASensorEnv):
    """Checkpoint inference with physics and sensing active in administration.usd."""

    cfg: AishaAdministrationLiveEnvCfg

    def __init__(self, cfg: AishaAdministrationLiveEnvCfg, render_mode: str | None = None, **kwargs):
        missing = [path for path in (ADMINISTRATION_LIVE_USD, PRESENTATION_ROBOT_USD) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                "missing live administration assets; run "
                f"isaaclab/tools/build_administration_live_assets.py first: {missing}"
            )
        super().__init__(cfg, render_mode, **kwargs)
