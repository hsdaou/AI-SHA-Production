"""RSL-RL PPO configuration for Phase 3 moving-obstacle robustness."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)

from aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase2_cfg import (
    AishaPhase2RehearsalPPORunnerCfg,
)


@configclass
class AishaPhase3DynamicPPORunnerCfg(AishaPhase2RehearsalPPORunnerCfg):
    """Conservative fine-tuning from the accepted administration base policy."""

    seed = 8601
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 25
    experiment_name = "aisha_block_a_sensor_nav"
    run_name = "phase3g_conservative_open_crossings_seed8601"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.5,
    )


@configclass
class AishaPhase3Segment6PPORunnerCfg(AishaPhase3DynamicPPORunnerCfg):
    """Low-step-size recovery of the Phase 3 principal-office turn."""

    seed = 8201
    max_iterations = 300
    save_interval = 25
    run_name = "phase3c_segment6_retention_seed8201"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.06,
        entropy_coef=0.0003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.5,
    )


@configclass
class AishaPhase3Segment6SpecialistPPORunnerCfg(AishaPhase3DynamicPPORunnerCfg):
    """Very conservative PPO adaptation of the deterministic turn specialist."""

    seed = 8501
    max_iterations = 600
    save_interval = 25
    run_name = "phase3f_segment6_safety_specialist_seed8501"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.04,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.5,
    )


@configclass
class AishaPhase3RecurrentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Imitate the accepted feed-forward route policy with a one-layer GRU."""

    seed = 8801
    num_steps_per_env = 32
    max_iterations = 200
    save_interval = 25
    experiment_name = "aisha_block_a_sensor_nav"
    run_name = "phase3i_recurrent_distillation_seed8801"
    obs_groups = {"policy": ["policy"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.10,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[256, 128, 64],
        activation="elu",
        rnn_type="gru",
        # The feed-forward teacher consumes 46 observations. Matching that
        # dimension lets RSL-RL load its actor weights without shape changes.
        rnn_hidden_dim=46,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=3,
        learning_rate=3.0e-4,
        gradient_length=32,
        max_grad_norm=0.7,
        optimizer="adam",
        loss_type="mse",
    )


@configclass
class AishaPhase3RecurrentPPORunnerCfg(AishaPhase3DynamicPPORunnerCfg):
    """PPO fine-tuning after feed-forward-to-GRU policy distillation."""

    seed = 8901
    num_steps_per_env = 64
    max_iterations = 800
    save_interval = 25
    run_name = "phase3j_recurrent_ppo_seed8901"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.20,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=46,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
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


@configclass
class AishaPhase3SafetyResidualPPORunnerCfg(AishaPhase3DynamicPPORunnerCfg):
    """Recurrent PPO for the bounded slow/stop safety-residual layer."""

    seed = 9001
    num_steps_per_env = 64
    max_iterations = 600
    save_interval = 25
    run_name = "phase3k_safety_residual_seed9001"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.12,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.08,
        entropy_coef=0.0002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.7,
    )
