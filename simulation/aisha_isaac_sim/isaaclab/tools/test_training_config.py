#!/usr/bin/env python3
"""Static contract tests for the AI-SHA Isaac Lab training foundation."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import yaml


CONFIG = Path(__file__).resolve().parents[2] / "config" / "training.yaml"
ENVIRONMENT = Path(__file__).resolve().parents[1] / "aisha_isaaclab" / "tasks" / "office_nav" / "office_nav_env.py"
EVALUATOR = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
SENSOR_ENVIRONMENT = (
    Path(__file__).resolve().parents[1]
    / "aisha_isaaclab"
    / "tasks"
    / "office_nav"
    / "block_a_sensor_env.py"
)
PHASE2_ENVIRONMENT = (
    Path(__file__).resolve().parents[1]
    / "aisha_isaaclab"
    / "tasks"
    / "office_nav"
    / "phase2_end_to_end_env.py"
)
ROUTE_PLAYER = Path(__file__).resolve().parents[1] / "scripts" / "play_block_a_route.py"
PACKAGED_PHASE2_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "aisha_phase2_policy_model_1850.pt"
)
ENSEMBLE_MANIFEST = Path(__file__).resolve().parents[1] / "checkpoints" / "administration_policy_ensemble.json"
PHASE3_ENVIRONMENT = (
    Path(__file__).resolve().parents[1]
    / "aisha_isaaclab"
    / "tasks"
    / "office_nav"
    / "phase3_dynamic_dr_env.py"
)
TASK_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "aisha_isaaclab"
    / "tasks"
    / "office_nav"
    / "__init__.py"
)
PHASE3_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "aisha_isaaclab"
    / "tasks"
    / "office_nav"
    / "agents"
    / "rsl_rl_ppo_phase3_cfg.py"
)
PHASE3_SEGMENT6_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase3_segment6_rehearsal.sh"
)
PHASE3_SEGMENT6_SPECIALIST_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase3_segment6_specialist.sh"
)
PHASE3_RECURRENT_BOOTSTRAP = (
    Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_recurrent_ppo.py"
)
PHASE3_RECURRENT_DISTILL_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase3_recurrent_distillation.sh"
)
PHASE3_RECURRENT_PPO_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase3_recurrent_ppo.sh"
)
PHASE3_SAFETY_RESIDUAL_BOOTSTRAP = (
    Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_safety_residual_ppo.py"
)
PHASE3_SAFETY_RESIDUAL_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase3_safety_residual.sh"
)
PHASE3_SAFETY_RESIDUAL_SMOKE = (
    Path(__file__).resolve().parents[1] / "scripts" / "smoke_phase3_safety_residual.py"
)
PHASE3_CLEARANCE_PLANNER_BOOTSTRAP = (
    Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_clearance_planner_ppo.py"
)
PHASE3_CLEARANCE_PLANNER_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase3_clearance_planner.sh"
)
PHASE3_CLEARANCE_PLANNER_SMOKE = (
    Path(__file__).resolve().parents[1] / "scripts" / "smoke_phase3_clearance_planner.py"
)


class TrainingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_task_uses_wheel_control(self) -> None:
        task = self.data["task"]
        self.assertEqual(task["control_mode"], "wheel_joint_velocity_targets")
        self.assertIn("root pose", task["forbidden_shortcut"])

    def test_policy_and_physics_rates_are_integral(self) -> None:
        task = self.data["task"]
        self.assertEqual(task["physics_rate_hz"] % task["policy_rate_hz"], 0)

    def test_door_clearance_matches_declared_geometry(self) -> None:
        geometry = self.data["geometry"]
        clearance = (geometry["doorway_clear_width_m"] - geometry["robot_transit_width_m"]) / 2.0
        self.assertTrue(math.isclose(clearance, geometry["nominal_clearance_per_side_m"], abs_tol=1e-9))
        self.assertIn("assumption", geometry["doorway_width_status"])

    def test_observation_and_action_dimensions(self) -> None:
        self.assertEqual(self.data["observations"]["count"], len(self.data["observations"]["terms"]))
        self.assertEqual(self.data["actions"]["count"], len(self.data["actions"]["terms"]))

    def test_episode_horizon_can_reach_goal(self) -> None:
        task = self.data["task"]
        geometry = self.data["geometry"]
        actions = self.data["actions"]
        route_length = geometry["goal_xy_m"][0] - geometry["robot_start_x_m"]
        minimum_time = route_length / actions["linear_velocity_range_mps"][1]
        self.assertGreaterEqual(task["episode_length_s"], minimum_time * 1.25)

    def test_release_boundaries_are_explicit(self) -> None:
        release = self.data["release"]
        self.assertTrue(release["foundation_policy_checkpoint_available"])
        self.assertTrue(release["held_out_foundation_evaluation_available"])
        self.assertTrue(release["sensor_policy_available"])
        self.assertTrue(release["held_out_sensor_evaluation_available"])
        self.assertTrue(release["continuous_learned_route_playback_available"])
        self.assertTrue(release["phase2_curriculum_implemented"])
        self.assertTrue(release["phase2_gpu_training_launched"])
        self.assertTrue(release["end_to_end_learned_route_available"])
        self.assertTrue(release["phase2_policy_checkpoint_available"])
        self.assertTrue(release["phase2_policy_only_acceptance_passed"])
        self.assertTrue(release["live_administration_learned_skill_ensemble_available"])
        self.assertTrue(release["live_cinematic_validation_passed"])
        self.assertTrue(release["phase3_dynamic_obstacle_curriculum_implemented"])
        self.assertTrue(release["phase3_bounded_safety_residual_training_completed"])
        self.assertFalse(release["phase3_bounded_safety_residual_checkpoint_accepted"])
        self.assertTrue(release["phase3_clearance_planner_training_completed"])
        self.assertTrue(release["phase3_clearance_planner_architecture_candidate_promoted"])
        self.assertFalse(release["phase3_clearance_planner_full_acceptance_passed"])
        self.assertFalse(release["nav2_integrated"])
        self.assertFalse(release["physical_robot_release"])

    def test_selected_phase2_checkpoint_is_packaged_and_hash_locked(self) -> None:
        selected = self.data["phase2_curriculum"]["training_stage"]
        self.assertTrue(PACKAGED_PHASE2_CHECKPOINT.is_file())
        digest = hashlib.sha256(PACKAGED_PHASE2_CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(digest, selected["selected_checkpoint_sha256"])

    def test_live_administration_policy_ensemble_is_packaged_and_hash_locked(self) -> None:
        manifest = json.loads(ENSEMBLE_MANIFEST.read_text(encoding="utf-8"))
        ensemble = self.data["phase2_curriculum"]["live_administration_integration"]
        base = ENSEMBLE_MANIFEST.parent / manifest["base_policy"]["path"]
        specialist = ENSEMBLE_MANIFEST.parent / manifest["segment_specialists"]["6"]["path"]
        self.assertEqual(ensemble["architecture"], manifest["format"].removesuffix("_v1"))
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), manifest["base_policy"]["sha256"])
        self.assertEqual(
            hashlib.sha256(specialist.read_bytes()).hexdigest(),
            manifest["segment_specialists"]["6"]["sha256"],
        )

    def test_held_out_evaluator_reads_pre_reset_outcomes(self) -> None:
        environment_source = ENVIRONMENT.read_text(encoding="utf-8")
        evaluator_source = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn('self.extras["episode_outcomes"]', environment_source)
        self.assertIn('outcomes = extras["episode_outcomes"]', evaluator_source)
        self.assertIn("deterministic_inference", evaluator_source)

    def test_sensor_evaluation_can_enforce_balanced_segment_quotas(self) -> None:
        environment_source = SENSOR_ENVIRONMENT.read_text(encoding="utf-8")
        evaluator_source = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("balanced_segment_assignment", environment_source)
        self.assertIn('"--episodes-per-segment"', evaluator_source)
        self.assertIn("equal quota per route segment", evaluator_source)

    def test_sensor_acceptance_gate_is_declared(self) -> None:
        gate = self.data["sensor_curriculum"]["held_out_acceptance_gate"]
        self.assertEqual(gate["episodes_per_segment"], 48)
        self.assertEqual(gate["office_threshold_segment_ids"], [3, 4, 8, 9])
        self.assertGreaterEqual(gate["overall_success_rate_min"], 0.80)
        self.assertLessEqual(gate["overall_collision_rate_max"], 0.15)

    def test_phase2_targets_supervisor_free_control(self) -> None:
        phase2 = self.data["phase2_curriculum"]
        self.assertEqual(phase2["control_mode"], "policy_only_wheel_joint_velocity_targets")
        self.assertIn("physical turn-supervisor action override", phase2["forbidden_shortcuts"])
        self.assertEqual(phase2["live_acceptance_gate"]["supervisor_turn_steps_allowed"], 0)
        self.assertEqual(phase2["live_acceptance_gate"]["supervisor_dwell_steps_allowed"], 0)
        self.assertTrue(phase2["full_route_acceptance_gate"]["policy_only_required"])

    def test_phase2_transition_budget_is_consistent(self) -> None:
        stage = self.data["phase2_curriculum"]["training_stage"]
        calculated = (
            stage["additional_ppo_iterations"]
            * stage["parallel_environments"]
            * stage["steps_per_environment_per_iteration"]
        )
        self.assertEqual(stage["additional_policy_transitions"], calculated)

    def test_phase2_environment_uses_physical_transition_headings(self) -> None:
        source = PHASE2_ENVIRONMENT.read_text(encoding="utf-8")
        sensor_source = SENSOR_ENVIRONMENT.read_text(encoding="utf-8")
        self.assertIn('start_heading_mode = "incoming"', source)
        self.assertIn("start_transition_backoff_m = 0.00", source)
        self.assertIn("start_linear_velocity_range_mps = (0.30, 0.50)", source)
        self.assertIn("joint_vel[:, self._wheel_ids] = start_linear_velocity", sensor_source)
        self.assertIn("self._previous_actions[env_ids, 0] = normalized_forward", sensor_source)
        self.assertIn("goal_tolerance_m_by_segment = PHASE2_GOAL_TOLERANCES", source)
        self.assertIn("self._goal_tolerances[self._segment_ids]", sensor_source)
        self.assertIn("start_yaw_jitter_rad = math.radians(15.0)", source)
        self.assertIn("reward_heading_progress = 10.0", source)
        self.assertIn("penalty_wrong_uturn_direction = -0.05", source)
        self.assertIn("penalty_misaligned_forward = -0.05", source)
        self.assertIn("linear_velocity_range_mps = (0.0, 0.50)", source)
        self.assertIn("TURN_DIRECTION_HINTS", source)
        self.assertIn("route_chain_mode = True", source)
        self.assertIn("fade_weight", sensor_source)
        self.assertIn("math.pi - 2.0 * torch.abs(hints)", sensor_source)
        self.assertIn("abs_heading_error < self.cfg.misaligned_heading_threshold_rad", sensor_source)
        self.assertIn("normalized_forward_command", sensor_source)

    def test_route_player_supports_policy_only_and_wider_camera(self) -> None:
        source = ROUTE_PLAYER.read_text(encoding="utf-8")
        self.assertIn('(\"hybrid\", \"policy-only\")', source)
        self.assertIn("(-3.8, 0.0, 2.4)", source)
        self.assertIn("for ray_offset in range(-1, 2)", source)
        self.assertIn('"manual_route_leg_follow"', source)
        self.assertIn('choices=("follow", "cinematic")', source)
        self.assertIn('"static_segment_cinematic_cameras"', source)
        self.assertIn("CINEMATIC_SHOTS", source)
        self.assertIn('"minimum_lidar_range_m"', source)
        self.assertIn('"policy_action"', source)
        self.assertIn("--segment-policy-checkpoint", source)
        self.assertIn("route_planner_selected_learned_skill_ensemble", source)
        self.assertIn("learned_sensor_policy_by_source", source)
        live_env_source = (
            Path(__file__).resolve().parents[1]
            / "aisha_isaaclab/tasks/office_nav/administration_live_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn("administration_collision_raycast_targets", live_env_source)
        self.assertIn("prim.HasAPI(UsdPhysics.CollisionAPI)", live_env_source)
        self.assertIn('live.get("route_control")', (
            Path(__file__).resolve().parent / "validate_phase2_end_to_end.py"
        ).read_text(encoding="utf-8"))

    def test_live_administration_finetune_targets_office_departures(self) -> None:
        live_source = (
            Path(__file__).resolve().parents[1]
            / "aisha_isaaclab/tasks/office_nav/administration_live_env.py"
        ).read_text(encoding="utf-8")
        finetune_source = (
            Path(__file__).resolve().parents[1]
            / "aisha_isaaclab/tasks/office_nav/administration_live_finetune_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn("administration_collision_raycast_targets", live_source)
        self.assertIn("start_linear_velocity_range_mps = (0.30, 0.50)", finetune_source)
        self.assertIn("turn_direction_hint_rad_by_segment = TURN_DIRECTION_HINTS", finetune_source)
        self.assertIn("penalty_misaligned_forward = -0.75", finetune_source)
        self.assertIn("penalty_collision = -100.0", finetune_source)
        self.assertIn("80.0", finetune_source)
        self.assertIn("40.0", finetune_source)
        self.assertIn("AishaAdministrationLiveRehearsalEnvCfg", finetune_source)
        self.assertIn("24.0", finetune_source)
        self.assertIn("start_transition_backoff_m_by_segment", finetune_source)
        self.assertIn("AishaAdministrationLivePrincipalTurnEnvCfg", finetune_source)
        self.assertIn("fixed_segment_id = 6", finetune_source)
        self.assertIn("penalty_misaligned_forward = -1.50", finetune_source)
        self.assertIn("penalty_aligned_nonforward = -0.75", finetune_source)
        sensor_env_source = (
            Path(__file__).resolve().parents[1]
            / "aisha_isaaclab/tasks/office_nav/block_a_sensor_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"aligned_nonforward"', sensor_env_source)
        self.assertIn("1.0 - normalized_forward_command", sensor_env_source)
        self.assertIn("self._start_transition_backoffs[segment_ids]", (
            Path(__file__).resolve().parents[1]
            / "aisha_isaaclab/tasks/office_nav/block_a_sensor_env.py"
        ).read_text(encoding="utf-8"))

    def test_phase2_rehearsal_samples_every_route_leg(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "aisha_isaaclab/tasks/office_nav/phase2_rehearsal_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn("segment_sampling_weights", source)
        self.assertIn("24.0", source)
        self.assertIn("penalty_misaligned_forward = -0.20", source)
        self.assertIn("penalty_collision = -50.0", source)

    def test_principal_turn_bootstrap_is_training_only(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/bootstrap_principal_turn_policy.py"
        ).read_text(encoding="utf-8")
        self.assertIn("expert demonstration gate failed", source)
        self.assertIn("actor_obs_normalizer(raw_obs)", source)
        self.assertIn("original_policy_retention_samples", source)
        self.assertIn("deterministic policy-only evaluation", source)

    def test_phase3_preserves_checkpoint_observation_contract(self) -> None:
        phase3 = self.data["phase3_curriculum"]
        source = PHASE3_ENVIRONMENT.read_text(encoding="utf-8")
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertEqual(phase3["training"]["policy_observation_count"], 46)
        self.assertTrue(phase3["training"]["checkpoint_compatible_with_phase2"])
        self.assertIn("DynamicObstacle_.*", source)
        self.assertIn("track_mesh_transforms=True", source)
        self.assertIn("ranges = self._lidar_ranges()", source)
        self.assertIn("self._lidar_ranges()", source)
        self.assertIn(phase3["gym_id"], registry)

    def test_phase3_domain_randomization_is_physical_and_bounded(self) -> None:
        phase3 = self.data["phase3_curriculum"]
        randomization = phase3["domain_randomization"]
        source = PHASE3_ENVIRONMENT.read_text(encoding="utf-8")
        self.assertEqual(randomization["action_latency_policy_steps"], [0, 2])
        self.assertLess(randomization["base_mass_scale"][0], 1.0)
        self.assertGreater(randomization["base_mass_scale"][1], 1.0)
        self.assertIn("set_masses", source)
        self.assertIn("set_inertias", source)
        self.assertIn("set_material_properties", source)
        self.assertIn("write_joint_damping_to_sim", source)
        self.assertIn("self._motor_strength", source)
        self.assertEqual(randomization["curriculum_warmup_policy_steps"], 3200)
        self.assertEqual(randomization["curriculum_ramp_policy_steps"], 11200)
        enabled = phase3["dynamic_obstacles"]["enabled_route_segments"]
        self.assertNotIn(6, enabled)
        self.assertIn("dynamic_obstacle_segment_ids = (0, 1, 2, 5, 7, 10, 11)", source)
        self.assertEqual(phase3["dynamic_obstacles"]["pedestrian_yield_radius_m"], 1.10)
        self.assertIn("self._obstacle_pause_phase", source)
        self.assertIn("yielding.float()", source)

    def test_phase3_transition_budget_and_initialization_hash(self) -> None:
        phase3 = self.data["phase3_curriculum"]
        training = phase3["training"]
        calculated = (
            training["iterations"]
            * training["parallel_environments"]
            * training["steps_per_environment_per_iteration"]
        )
        self.assertEqual(calculated, training["planned_policy_transitions"])
        checkpoint = Path(__file__).resolve().parents[1] / Path(
            phase3["initialization"]["checkpoint"]
        ).relative_to("isaaclab")
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            phase3["initialization"]["checkpoint_sha256"],
        )

    def test_phase3_acceptance_gate_does_not_claim_human_safety(self) -> None:
        phase3 = self.data["phase3_curriculum"]
        gate = phase3["acceptance_gate"]
        self.assertFalse(gate["physical_safety_claim_allowed"])
        self.assertEqual(gate["live_administration_collisions_allowed"], 0)
        self.assertGreaterEqual(gate["phase2_static_route_regression_success_rate_min"], 0.90)

    def test_phase3_segment6_recovery_retains_all_route_legs(self) -> None:
        recovery = self.data["phase3_curriculum"]["segment6_recovery"]
        source = PHASE3_ENVIRONMENT.read_text(encoding="utf-8")
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        runner = PHASE3_RUNNER.read_text(encoding="utf-8")
        launcher = PHASE3_SEGMENT6_LAUNCHER.read_text(encoding="utf-8")
        weights = recovery["segment_sampling_weights"]
        self.assertEqual(len(weights), 12)
        self.assertTrue(all(weight > 0 for weight in weights))
        self.assertEqual(weights[recovery["target_segment_id"]], 40)
        self.assertAlmostEqual(recovery["target_reset_fraction"], 40 / sum(weights))
        self.assertIn("AishaPhase3Segment6RehearsalEnvCfg", source)
        self.assertIn(recovery["gym_id"], registry)
        self.assertIn("AishaPhase3Segment6PPORunnerCfg", runner)
        self.assertIn(recovery["initialization_checkpoint_sha256"], launcher)

    def test_phase3_segment6_recovery_budget_and_checkpoint_hash(self) -> None:
        recovery = self.data["phase3_curriculum"]["segment6_recovery"]
        calculated = (
            recovery["iterations"]
            * recovery["parallel_environments"]
            * recovery["steps_per_environment_per_iteration"]
        )
        self.assertEqual(calculated, recovery["planned_policy_transitions"])
        checkpoint = Path(__file__).resolve().parents[1] / Path(
            recovery["initialization_checkpoint"]
        ).relative_to("isaaclab")
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            recovery["initialization_checkpoint_sha256"],
        )

    def test_phase3_segment6_specialist_is_hash_locked_and_fixed_skill(self) -> None:
        specialist = self.data["phase3_curriculum"]["segment6_robust_specialist"]
        source = PHASE3_ENVIRONMENT.read_text(encoding="utf-8")
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        launcher = PHASE3_SEGMENT6_SPECIALIST_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("AishaPhase3Segment6SpecialistEnvCfg", source)
        self.assertIn("fixed_segment_id = 6", source)
        self.assertIn(specialist["gym_id"], registry)
        self.assertIn(specialist["initialization_checkpoint_sha256"], launcher)
        calculated = (
            specialist["iterations"]
            * specialist["parallel_environments"]
            * specialist["steps_per_environment_per_iteration"]
        )
        self.assertEqual(calculated, specialist["planned_policy_transitions"])
        checkpoint = Path(__file__).resolve().parents[1] / Path(
            specialist["initialization_checkpoint"]
        ).relative_to("isaaclab")
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            specialist["initialization_checkpoint_sha256"],
        )

    def test_phase3_evaluator_separates_collision_classes_and_partial_gate(self) -> None:
        environment_source = PHASE3_ENVIRONMENT.read_text(encoding="utf-8")
        evaluator_source = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn('outcomes["dynamic_obstacle_collision"]', environment_source)
        self.assertIn('outcomes["static_collision"]', environment_source)
        self.assertIn("phase3_randomized_segment_subgate_only", evaluator_source)
        self.assertIn('"full_phase3_acceptance": False', evaluator_source)
        self.assertIn('is_phase3_task = "Phase3-" in args.task', evaluator_source)
        self.assertIn("env_cfg.curriculum_minimum_strength = 1.0", evaluator_source)
        self.assertIn('"phase3_curriculum_strength": phase3_curriculum_strength', evaluator_source)

    def test_phase3_recurrent_gate_preserves_sensor_contract(self) -> None:
        runner = PHASE3_RUNNER.read_text(encoding="utf-8")
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        bootstrap = PHASE3_RECURRENT_BOOTSTRAP.read_text(encoding="utf-8")
        distill_launcher = PHASE3_RECURRENT_DISTILL_LAUNCHER.read_text(encoding="utf-8")
        ppo_launcher = PHASE3_RECURRENT_PPO_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("AishaPhase3RecurrentDistillationRunnerCfg", runner)
        self.assertIn("AishaPhase3RecurrentPPORunnerCfg", runner)
        self.assertIn('rnn_type="gru"', runner)
        self.assertIn("rnn_hidden_dim=46", runner)
        self.assertIn("Phase3-RecurrentDistill", registry)
        self.assertIn("Phase3-RecurrentPPO", registry)
        self.assertIn("ActorCriticRecurrent", bootstrap)
        self.assertIn('"memory_a.", "memory_s."', bootstrap)
        self.assertIn('"observation_count": 46', bootstrap)
        self.assertIn("3ab596c61259784657b36fe4ee937da8495ce9621a43bd04e2c8a0bf6e0b1880", distill_launcher)
        self.assertIn('CHECKSUM_FILE="${CHECKPOINT%/*}/checkpoint.sha256"', ppo_launcher)

    def test_phase3_recurrent_gate_is_hash_locked_and_unaccepted(self) -> None:
        recurrent = self.data["phase3_curriculum"]["recurrent_temporal_gate"]
        self.assertEqual(recurrent["policy_observation_count"], 46)
        self.assertEqual(recurrent["ppo_training"]["policy_transitions"], 1_638_400)
        self.assertFalse(recurrent["leading_candidate"]["accepted"])
        checkpoint = Path(__file__).resolve().parents[1] / Path(
            recurrent["leading_candidate"]["checkpoint"]
        ).relative_to("isaaclab")
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            recurrent["leading_candidate"]["checkpoint_sha256"],
        )
        comparison = recurrent["same_seed_9403_comparison"]
        self.assertGreater(
            comparison["recurrent"]["dynamic_collisions"],
            comparison["feed_forward"]["dynamic_collisions"],
        )

    def test_phase3_safety_residual_has_a_hard_action_boundary(self) -> None:
        residual = self.data["phase3_curriculum"]["bounded_safety_residual_gate"]
        source = PHASE3_ENVIRONMENT.read_text(encoding="utf-8")
        runner = PHASE3_RUNNER.read_text(encoding="utf-8")
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        bootstrap = PHASE3_SAFETY_RESIDUAL_BOOTSTRAP.read_text(encoding="utf-8")
        launcher = PHASE3_SAFETY_RESIDUAL_LAUNCHER.read_text(encoding="utf-8")
        smoke = PHASE3_SAFETY_RESIDUAL_SMOKE.read_text(encoding="utf-8")
        self.assertIn("AishaPhase3SafetyResidualEnvCfg", source)
        self.assertIn("AishaPhase3SafetyResidualPPORunnerCfg", runner)
        self.assertIn("Phase3-SafetyResidual", registry)
        self.assertIn("base_forward_fraction * (1.0 - brake_fraction)", source)
        self.assertIn("base_actions[:, 1] * (1.0 - angular_attenuation)", source)
        self.assertIn("self._frozen_route_actor.requires_grad_(False)", source)
        self.assertIn(
            "52f0094674dea901b4b7f3d7717bc9c2b014a6dc2d8e22cca768f783f4a9c0c8",
            source,
        )
        self.assertIn("torch.nn.init.zeros_(final_actor_layer.weight)", bootstrap)
        self.assertIn("deterministic_zero_output", bootstrap)
        self.assertIn("FROZEN_ROUTE_SHA256", launcher)
        self.assertIn("zero_residual_is_exact_pass_through", smoke)
        self.assertFalse(residual["action_boundary"]["may_increase_forward_speed"])
        self.assertFalse(residual["action_boundary"]["may_reverse"])
        self.assertFalse(residual["action_boundary"]["may_flip_steering_sign"])
        self.assertEqual(residual["ppo_training"]["policy_transitions"], 1_228_800)
        self.assertFalse(residual["full_same_seed_comparison"]["accepted"])
        checkpoint = Path(__file__).resolve().parents[1] / Path(
            residual["full_same_seed_comparison"]["selected_checkpoint"]
        ).relative_to("isaaclab")
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            residual["full_same_seed_comparison"]["selected_checkpoint_sha256"],
        )

    def test_phase3_clearance_planner_has_independent_runtime_gates(self) -> None:
        planner = self.data["phase3_curriculum"]["clearance_planner_gate"]
        source = PHASE3_ENVIRONMENT.read_text(encoding="utf-8")
        runner = PHASE3_RUNNER.read_text(encoding="utf-8")
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        bootstrap = PHASE3_CLEARANCE_PLANNER_BOOTSTRAP.read_text(encoding="utf-8")
        launcher = PHASE3_CLEARANCE_PLANNER_LAUNCHER.read_text(encoding="utf-8")
        smoke = PHASE3_CLEARANCE_PLANNER_SMOKE.read_text(encoding="utf-8")
        evaluator = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("AishaPhase3ClearancePlannerEnvCfg", source)
        self.assertIn("AishaPhase3ClearancePlannerPPORunnerCfg", runner)
        self.assertIn("Phase3-ClearancePlanner", registry)
        self.assertIn("_predict_candidate_geometry", source)
        self.assertIn("planner_minimum_predicted_clearance_m", source)
        self.assertIn("_apply_protective_stop", source)
        self.assertIn("self._protective_stop_latched", source)
        self.assertIn("protected[:, 0]", source)
        self.assertIn("protective_stop_preserves_steering", smoke)
        self.assertIn("planner_never_accepts_below_clearance_floor", smoke)
        self.assertIn("torch.nn.init.zeros_(final_actor_layer.weight)", bootstrap)
        self.assertIn("FROZEN_ROUTE_SHA256", launcher)
        self.assertIn("signed_clearance_projected_steering_request", evaluator)
        self.assertFalse(planner["action_boundary"]["may_increase_forward_speed"])
        self.assertFalse(planner["action_boundary"]["may_reverse"])
        self.assertFalse(
            planner["action_boundary"]["steering_request_sent_directly_to_wheels"]
        )
        self.assertTrue(planner["action_boundary"]["protective_stop_can_remove_forward_motion"])
        self.assertLess(
            planner["protective_stop_contract"]["trigger_clearance_beyond_robot_envelope_m"],
            planner["protective_stop_contract"]["release_clearance_beyond_robot_envelope_m"],
        )
        self.assertEqual(planner["ppo_training"]["policy_transitions"], 1_228_800)
        comparison = planner["full_same_seed_comparison"]
        self.assertTrue(comparison["architecture_candidate_promoted"])
        self.assertFalse(comparison["full_phase3_accepted"])
        self.assertFalse(comparison["presentation_policy_replaced"])
        self.assertGreater(
            comparison["trained_model_200"]["successes"],
            comparison["original_frozen_route"]["successes"],
        )
        self.assertLess(
            comparison["trained_model_200"]["dynamic_collisions"],
            comparison["protective_stop_zero_policy"]["dynamic_collisions"],
        )
        checkpoint = Path(__file__).resolve().parents[1] / Path(
            comparison["selected_checkpoint"]
        ).relative_to("isaaclab")
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            comparison["selected_checkpoint_sha256"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
