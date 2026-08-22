"""Phase 2c rehearsal curriculum to prevent live-adaptation forgetting."""

from __future__ import annotations

from isaaclab.utils import configclass

from aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env import (
    AishaPhase2TurnCurriculumEnv,
    AishaPhase2TurnCurriculumEnvCfg,
)


@configclass
class AishaPhase2RehearsalEnvCfg(AishaPhase2TurnCurriculumEnvCfg):
    """Rehearse the complete proxy route after administration-scene adaptation."""

    # Emphasize the hallway-to-Principal turn (segment 6), retain both office
    # departures, and sample every other leg often enough to resist forgetting.
    segment_sampling_weights = (
        8.0,
        8.0,
        8.0,
        5.0,
        8.0,
        8.0,
        24.0,
        8.0,
        5.0,
        8.0,
        8.0,
        8.0,
    )
    penalty_misaligned_forward = -0.20
    penalty_collision = -50.0


class AishaPhase2RehearsalEnv(AishaPhase2TurnCurriculumEnv):
    """Proxy-scene PPO rehearsal with the Phase 2 observation/action contract."""

    cfg: AishaPhase2RehearsalEnvCfg
