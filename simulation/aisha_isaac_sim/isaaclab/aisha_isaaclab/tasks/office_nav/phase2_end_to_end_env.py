"""Phase 2 curricula for removing the route turn supervisor."""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from aisha_isaaclab.tasks.office_nav.block_a_sensor_env import (
    AishaBlockASensorEnv,
    AishaBlockASensorEnvCfg,
)


TURN_DIRECTION_HINTS = (
    0.0,
    0.0,
    0.0,
    0.0,
    math.radians(30.0),
    0.0,
    0.0,
    0.0,
    0.0,
    math.radians(30.0),
    0.0,
    0.0,
)

# Office visit targets use a tighter 0.20 m tolerance so a chained run reaches
# the in-room pivot zone before changing to the departure goal. The 1.64 m
# pivot circle does not fit in the assumed 1.40 m doorway; ordinary transit
# targets retain the learned 0.45 m tolerance.
PHASE2_GOAL_TOLERANCES = (
    0.45,  # home -> east atrium exit
    0.45,  # east atrium exit -> vice-principal turn
    0.45,  # vice-principal turn -> approach
    0.20,  # vice-principal approach -> in-room visit
    0.45,  # vice-principal visit -> departure
    0.45,  # vice-principal departure -> hallway return
    0.45,  # hallway return -> principal turn
    0.45,  # principal turn -> approach
    0.20,  # principal approach -> in-room visit
    0.45,  # principal visit -> departure
    0.45,  # principal departure -> atrium return
    0.45,  # atrium return -> home
)


@configclass
class AishaPhase2TurnCurriculumEnvCfg(AishaBlockASensorEnvCfg):
    """Efficient parallel curriculum for policy-controlled route realignment."""

    episode_length_s = 70.0
    # Zero minimum speed permits a physical pivot commanded entirely through
    # the learned left/right wheel targets.
    linear_velocity_range_mps = (0.0, 0.50)
    start_lateral_jitter_m = 0.03
    start_yaw_jitter_rad = math.radians(15.0)
    start_heading_mode = "incoming"
    start_transition_backoff_m = 0.00
    start_linear_velocity_range_mps = (0.30, 0.50)
    goal_jitter_m = 0.03
    goal_tolerance_m_by_segment = PHASE2_GOAL_TOLERANCES
    observation_lidar_noise_std_m = 0.01
    observation_lidar_dropout_probability = 0.0025
    reward_heading_alignment = 0.02
    reward_heading_progress = 10.0
    penalty_wrong_uturn_direction = -0.05
    penalty_yaw_rate = -0.0005
    penalty_misaligned_forward = -0.05
    turn_direction_hint_rad_by_segment = TURN_DIRECTION_HINTS
    segment_sampling_weights = (1.0, 1.0, 1.0, 2.0, 40.0, 1.0, 1.0, 1.0, 2.0, 40.0, 1.0, 1.0)


@configclass
class AishaPhase2EndToEndRouteEnvCfg(AishaBlockASensorEnvCfg):
    """Held-out full-route gate with no turn or dwell action override."""

    episode_length_s = 240.0
    linear_velocity_range_mps = (0.0, 0.50)
    goal_tolerance_m_by_segment = PHASE2_GOAL_TOLERANCES
    route_chain_mode = True
    start_lateral_jitter_m = 0.05
    start_yaw_jitter_rad = math.radians(8.0)
    goal_jitter_m = 0.03
    observation_lidar_noise_std_m = 0.02
    observation_lidar_dropout_probability = 0.005
    reward_heading_alignment = 0.01
    reward_heading_progress = 2.0
    penalty_wrong_uturn_direction = -0.05
    penalty_yaw_rate = -0.0005
    penalty_misaligned_forward = -0.05
    turn_direction_hint_rad_by_segment = TURN_DIRECTION_HINTS


class AishaPhase2TurnCurriculumEnv(AishaBlockASensorEnv):
    """Phase 2 training environment; behaviour is configured above."""

    cfg: AishaPhase2TurnCurriculumEnvCfg


class AishaPhase2EndToEndRouteEnv(AishaBlockASensorEnv):
    """Phase 2 policy-only full-route acceptance environment."""

    cfg: AishaPhase2EndToEndRouteEnvCfg
