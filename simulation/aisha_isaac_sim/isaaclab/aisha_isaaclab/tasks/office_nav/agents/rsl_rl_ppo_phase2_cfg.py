"""RSL-RL PPO fine-tuning configuration for Phase 2 turn acquisition."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_sensor_cfg import (
    AishaBlockASensorPPORunnerCfg,
)


@configclass
class AishaPhase2TurnPPORunnerCfg(AishaBlockASensorPPORunnerCfg):
    seed = 245
    max_iterations = 1200
    save_interval = 50
    # Keep the experiment root compatible with the Phase 1 resume checkpoint.
    experiment_name = "aisha_block_a_sensor_nav"
    run_name = "phase2_transition_turn_seed245"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AishaPhase2RehearsalPPORunnerCfg(AishaPhase2TurnPPORunnerCfg):
    """Lower-rate PPO updates for post-adaptation route rehearsal."""

    seed = 7093
    max_iterations = 100
    save_interval = 25
    run_name = "phase2c_proxy_rehearsal_seed7093"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.7,
    )
