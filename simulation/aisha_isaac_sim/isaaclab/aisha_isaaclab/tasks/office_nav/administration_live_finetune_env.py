"""Phase 2b curriculum against the upgraded administration collision contract."""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from aisha_isaaclab.tasks.office_nav.administration_live_env import (
    AishaAdministrationLiveEnv,
    AishaAdministrationLiveEnvCfg,
)
from aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env import (
    PHASE2_GOAL_TOLERANCES,
    TURN_DIRECTION_HINTS,
)


@configclass
class AishaAdministrationLiveFineTuneEnvCfg(AishaAdministrationLiveEnvCfg):
    """Resume PPO on the same geometry and ray contract used for final filming."""

    episode_length_s = 70.0
    linear_velocity_range_mps = (0.0, 0.50)
    start_lateral_jitter_m = 0.03
    start_yaw_jitter_rad = math.radians(15.0)
    start_heading_mode = "incoming"
    start_transition_backoff_m = 0.0
    start_linear_velocity_range_mps = (0.30, 0.50)
    goal_jitter_m = 0.03
    goal_tolerance_m_by_segment = PHASE2_GOAL_TOLERANCES
    observation_lidar_noise_std_m = 0.01
    observation_lidar_dropout_probability = 0.0025
    reward_heading_alignment = 0.02
    reward_heading_progress = 12.0
    penalty_wrong_uturn_direction = -0.05
    penalty_yaw_rate = -0.0005
    # The inherited actor saturates its forward mean during a 180-degree goal
    # reversal. Make slower sampled pivots materially better than wide arcs so
    # PPO can acquire the safe behavior before approaching either doorway.
    penalty_misaligned_forward = -0.75
    penalty_collision = -100.0
    turn_direction_hint_rad_by_segment = TURN_DIRECTION_HINTS
    # Concentrate real-scene adaptation on both office departures while still
    # rehearsing every route leg to limit catastrophic forgetting.
    segment_sampling_weights = (
        1.0,
        1.0,
        1.0,
        4.0,
        80.0,
        1.0,
        1.0,
        1.0,
        4.0,
        40.0,
        1.0,
        1.0,
    )


class AishaAdministrationLiveFineTuneEnv(AishaAdministrationLiveEnv):
    """Learned-policy adaptation with full live-scene physics and sensing."""

    cfg: AishaAdministrationLiveFineTuneEnvCfg


@configclass
class AishaAdministrationLiveRehearsalEnvCfg(AishaAdministrationLiveFineTuneEnvCfg):
    """Balanced live-scene rehearsal after the office pivot has been acquired."""

    segment_sampling_weights = (
        8.0,
        8.0,
        8.0,
        8.0,
        8.0,
        8.0,
        24.0,
        8.0,
        8.0,
        8.0,
        8.0,
        8.0,
    )
    start_transition_backoff_m_by_segment = (
        0.00,  # route origin
        0.45,
        0.45,
        0.45,
        0.20,  # Vice-Principal visit uses its tighter stop tolerance
        0.45,
        0.45,
        0.45,
        0.45,
        0.20,  # Principal visit uses its tighter stop tolerance
        0.45,
        0.45,
    )
    penalty_misaligned_forward = -0.20
    penalty_collision = -50.0


class AishaAdministrationLiveRehearsalEnv(AishaAdministrationLiveFineTuneEnv):
    """Conservative PPO rehearsal across every detailed administration route leg."""

    cfg: AishaAdministrationLiveRehearsalEnvCfg


@configclass
class AishaAdministrationLivePrincipalTurnEnvCfg(AishaAdministrationLiveRehearsalEnvCfg):
    """Acquire the chained hallway-to-Principal turn from its real handoff pose."""

    fixed_segment_id = 6
    episode_length_s = 55.0
    reward_progress = 18.0
    reward_goal_proximity = 0.02
    reward_heading_progress = 8.0
    penalty_misaligned_forward = -1.50
    # The inherited actor completes the pivot, then its raw linear mean drops
    # below -1 (the zero-speed action boundary). Penalize that action directly
    # once heading is safe so PPO can associate exploration with forward resume.
    penalty_aligned_nonforward = -0.75
    penalty_stall = -0.25


class AishaAdministrationLivePrincipalTurnEnv(AishaAdministrationLiveRehearsalEnv):
    """Focused live-scene curriculum for turn-then-resume behavior."""

    cfg: AishaAdministrationLivePrincipalTurnEnvCfg
