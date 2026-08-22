"""RSL-RL PPO configuration for the Block A sensor curriculum."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AishaBlockASensorPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 144
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "aisha_block_a_sensor_nav"
    run_name = "ld19_flush_threshold_v9"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
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
