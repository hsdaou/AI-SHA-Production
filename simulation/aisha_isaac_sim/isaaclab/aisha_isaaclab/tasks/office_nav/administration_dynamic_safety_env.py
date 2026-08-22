"""Run the frozen Phase 3M stack and Phase 3N safety actor in administration.usd."""

from __future__ import annotations

from pathlib import Path

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from aisha_isaaclab.tasks.office_nav.administration_live_env import (
    ADMINISTRATION_LIVE_USD,
    PRESENTATION_ROBOT_USD,
    AishaAdministrationLiveSceneCfg,
    administration_collision_raycast_targets,
)
from aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env import (
    AishaPhase3DynamicSafetyEnv,
    AishaPhase3DynamicSafetyEnvCfg,
    _person_proxy,
)


PHASE3N_LIVE_GOAL_TOLERANCES_M = (
    0.45,
    0.45,
    0.45,
    0.22,  # Vice-Principal visit: presentation stop tolerance, not geometry.
    0.45,
    0.45,
    0.45,
    0.45,
    0.22,  # Principal visit: presentation stop tolerance, not geometry.
    0.45,
    0.45,
    0.45,
)

PHASE3N_PRESENTATION_GOAL_TOLERANCES_M = (
    *PHASE3N_LIVE_GOAL_TOLERANCES_M[:9],
    0.20,  # Reach the principal-departure centreline before return guarding.
    *PHASE3N_LIVE_GOAL_TOLERANCES_M[10:],
)


@configclass
class AishaAdministrationDynamicSafetySceneCfg(AishaAdministrationLiveSceneCfg):
    """Walkthrough-matched administration scene with ray-visible people."""

    dynamic_obstacle_0 = _person_proxy(0)
    dynamic_obstacle_1 = _person_proxy(1)
    dynamic_obstacle_2 = _person_proxy(2)
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
        mesh_prim_paths=administration_collision_raycast_targets()
        + [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/DynamicObstacle_.*",
                is_shared=False,
                merge_prim_meshes=True,
                track_mesh_transforms=True,
            )
        ],
        reference_meshes=True,
        update_mesh_ids=False,
        debug_vis=False,
    )


@configclass
class AishaAdministrationSafetyPresentationSceneCfg(
    AishaAdministrationLiveSceneCfg
):
    """Static presentation scene; dynamic safety is proven by a separate gate."""

    dynamic_obstacle_0 = _person_proxy(0)
    dynamic_obstacle_1 = _person_proxy(1)
    dynamic_obstacle_2 = _person_proxy(2)


@configclass
class AishaAdministrationDynamicSafetyEnvCfg(AishaPhase3DynamicSafetyEnvCfg):
    """One-action live-scene gate for the packaged Phase 3N checkpoint."""

    scene: AishaAdministrationDynamicSafetySceneCfg = (
        AishaAdministrationDynamicSafetySceneCfg(
            num_envs=1,
            env_spacing=55.0,
            replicate_physics=False,
            clone_in_fabric=False,
        )
    )
    goal_tolerance_m_by_segment = PHASE3N_LIVE_GOAL_TOLERANCES_M

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (-4.5, 4.5, 5.5)
        self.viewer.lookat = (0.0, 0.0, 0.55)


@configclass
class AishaAdministrationSafetyPresentationEnvCfg(
    AishaAdministrationDynamicSafetyEnvCfg
):
    """Reproducible final-film scenario with no stochastic pedestrian proxy."""

    scene: AishaAdministrationSafetyPresentationSceneCfg = (
        AishaAdministrationSafetyPresentationSceneCfg(
            num_envs=1,
            env_spacing=55.0,
            replicate_physics=False,
            clone_in_fabric=False,
        )
    )
    dynamic_obstacle_activation_probability = 0.0
    dynamic_crossing_creep_segment_ids = ()
    predictive_stop_segment_ids = (6, 11)
    goal_tolerance_m_by_segment = PHASE3N_PRESENTATION_GOAL_TOLERANCES_M


class AishaAdministrationDynamicSafetyEnv(AishaPhase3DynamicSafetyEnv):
    """Live wheel physics, full-ring sensing, and the frozen Phase 3M stack."""

    cfg: AishaAdministrationDynamicSafetyEnvCfg

    def __init__(
        self,
        cfg: AishaAdministrationDynamicSafetyEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        missing = [
            path
            for path in (ADMINISTRATION_LIVE_USD, PRESENTATION_ROBOT_USD)
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "missing live administration assets; run "
                f"isaaclab/tools/build_administration_live_assets.py first: {missing}"
            )
        super().__init__(cfg, render_mode, **kwargs)
