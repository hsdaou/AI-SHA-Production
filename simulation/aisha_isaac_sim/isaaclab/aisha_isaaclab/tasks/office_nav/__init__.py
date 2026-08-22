"""Register the AI-SHA office-door navigation task."""

import gymnasium as gym


gym.register(
    id="Isaac-AISHA-OfficeNav-Direct-v0",
    entry_point="aisha_isaaclab.tasks.office_nav.office_nav_env:AishaOfficeNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "aisha_isaaclab.tasks.office_nav.office_nav_env:AishaOfficeNavEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_cfg:AishaOfficeNavPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase2-Turn-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env:"
        "AishaPhase2TurnCurriculumEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env:"
            "AishaPhase2TurnCurriculumEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase2_cfg:"
            "AishaPhase2TurnPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase2-EndToEnd-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env:"
        "AishaPhase2EndToEndRouteEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase2_end_to_end_env:"
            "AishaPhase2EndToEndRouteEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase2_cfg:"
            "AishaPhase2TurnPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-SensorNav-Direct-v0",
    entry_point="aisha_isaaclab.tasks.office_nav.block_a_sensor_env:AishaBlockASensorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.block_a_sensor_env:AishaBlockASensorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_sensor_cfg:AishaBlockASensorPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-Administration-Live-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.administration_live_env:AishaAdministrationLiveEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.administration_live_env:"
            "AishaAdministrationLiveEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_sensor_cfg:"
            "AishaBlockASensorPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-Administration-Live-FineTune-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.administration_live_finetune_env:"
        "AishaAdministrationLiveFineTuneEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.administration_live_finetune_env:"
            "AishaAdministrationLiveFineTuneEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase2_cfg:"
            "AishaPhase2TurnPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase2-Rehearsal-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase2_rehearsal_env:"
        "AishaPhase2RehearsalEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase2_rehearsal_env:"
            "AishaPhase2RehearsalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase2_cfg:"
            "AishaPhase2RehearsalPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-Administration-Live-Rehearsal-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.administration_live_finetune_env:"
        "AishaAdministrationLiveRehearsalEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.administration_live_finetune_env:"
            "AishaAdministrationLiveRehearsalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase2_cfg:"
            "AishaPhase2RehearsalPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-Administration-Live-PrincipalTurn-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.administration_live_finetune_env:"
        "AishaAdministrationLivePrincipalTurnEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.administration_live_finetune_env:"
            "AishaAdministrationLivePrincipalTurnEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase2_cfg:"
            "AishaPhase2RehearsalPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase3-DynamicDR-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
        "AishaPhase3DynamicDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
            "AishaPhase3DynamicDREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase3_cfg:"
            "AishaPhase3DynamicPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase3-RecurrentDistill-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
        "AishaPhase3DynamicDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
            "AishaPhase3DynamicDREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase3_cfg:"
            "AishaPhase3RecurrentDistillationRunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase3-RecurrentPPO-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
        "AishaPhase3DynamicDREnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
            "AishaPhase3DynamicDREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase3_cfg:"
            "AishaPhase3RecurrentPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase3-SafetyResidual-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
        "AishaPhase3SafetyResidualEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
            "AishaPhase3SafetyResidualEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase3_cfg:"
            "AishaPhase3SafetyResidualPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase3-Segment6Rehearsal-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
        "AishaPhase3Segment6RehearsalEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
            "AishaPhase3Segment6RehearsalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase3_cfg:"
            "AishaPhase3Segment6PPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-AISHA-BlockA-Phase3-Segment6Specialist-SensorNav-Direct-v0",
    entry_point=(
        "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
        "AishaPhase3Segment6SpecialistEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.phase3_dynamic_dr_env:"
            "AishaPhase3Segment6SpecialistEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "aisha_isaaclab.tasks.office_nav.agents.rsl_rl_ppo_phase3_cfg:"
            "AishaPhase3Segment6SpecialistPPORunnerCfg"
        ),
    },
)
