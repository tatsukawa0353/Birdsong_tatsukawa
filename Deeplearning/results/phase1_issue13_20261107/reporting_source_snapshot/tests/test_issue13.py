"""Constant one/two-control invariants; no learning runs or result writes."""
import unittest
from unittest.mock import patch

import numpy as np

from Deeplearning.acoustics import distance
from Deeplearning.issue13_env import ConstantControlEnv, Issue13Env
from Deeplearning.model import DEFAULT_CONTROLS, HIGH, LOW
from Deeplearning.phase1_validation_env import (
    CONTROL_S, DURATION, PHYSICAL_HORIZON, RATE, TARGET_CONTROLS, ValidationEnv,
)


class Issue13EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = ValidationEnv("A")
        cls.target = env.target_audio.copy()
        env.close()

    def test_conditions_spaces_config_and_target_generation_account(self):
        self.assertIs(Issue13Env, ConstantControlEnv)
        for indices in ((0,), (0, 2)):
            env = ConstantControlEnv(indices, target_audio=self.target)
            try:
                self.assertEqual(env.config.condition, "A")
                self.assertEqual(env.config.control_indices, indices)
                self.assertEqual(env.indices, list(indices))
                self.assertEqual(env.horizon, 1)
                self.assertEqual(env.decision_horizon, 1)
                self.assertEqual(env.physical_horizon, PHYSICAL_HORIZON)
                self.assertEqual(env.action_space.shape, (len(indices),))
                self.assertEqual(env.action_space.dtype, np.dtype(np.float32))
                self.assertEqual(env.observation_space.shape, (5+len(indices),))
                self.assertEqual(env.setup_simulation_s, 0)
                self.assertEqual(env.config.reward_discount, 1)
                self.assertFalse(env.target_audio.flags.writeable)
                np.testing.assert_array_equal(env.target_audio, self.target)
                for extreme, bounds in ((-1., LOW), (1., HIGH)):
                    decoded = env.decode(np.full(len(indices), extreme))
                    np.testing.assert_array_equal(decoded[list(indices)], bounds[list(indices)])
            finally:
                env.close()
        generated = ConstantControlEnv((0, 2))
        try:
            self.assertEqual(generated.setup_simulation_s, DURATION)
            self.assertEqual(generated.integrated_s, 0)
            np.testing.assert_array_equal(generated.target_audio, self.target)
        finally:
            generated.close()

    def test_both_float64_oracles_are_exact_and_one_decision_completes_episode(self):
        for indices in ((0,), (0, 2)):
            env = ConstantControlEnv(indices, target_audio=self.target)
            try:
                initial, _ = env.reset(seed=0)
                self.assertTrue(env.observation_space.contains(initial))
                action = env.encode(TARGET_CONTROLS)
                self.assertEqual(action.dtype, np.dtype(np.float64))
                obs, reward, done, truncated, info = env.step(action)
                self.assertTrue(done)
                self.assertFalse(truncated)
                self.assertFalse(info["numerical_failure"])
                self.assertTrue(env.observation_space.contains(obs))
                np.testing.assert_array_equal(np.concatenate(env.audio), self.target)
                np.testing.assert_array_equal(env.applied_control_history,
                                              np.tile(TARGET_CONTROLS, (35, 1)))
                self.assertEqual(info["episode_metrics"]["total"], 0)
                self.assertEqual(reward, 0)
                self.assertEqual(env.completed_episodes, 1)
                self.assertEqual(env.successful_episodes, 1)
                self.assertEqual(env.decision_steps, 1)
                self.assertEqual(env.native_blocks, 35)
                self.assertEqual(env.native_blocks_completed, 35)
                self.assertAlmostEqual(env.integrated_s, DURATION)
                self.assertAlmostEqual(env.integrated_completed_s, DURATION)
                self.assertEqual(np.concatenate(env.audio).shape, (17500,))
                with self.assertRaises(RuntimeError):
                    env.step(action)
            finally:
                env.close()

    def test_one_variable_matches_historical_validation_task_a_exactly(self):
        legacy = ValidationEnv("A", target_audio=self.target)
        current = ConstantControlEnv((0,), target_audio=self.target)
        try:
            for action in (np.asarray([-.6]), np.asarray([.28]), np.asarray([.7], dtype=np.float32)):
                obs_a, _ = legacy.reset(seed=31)
                obs_b, _ = current.reset(seed=31)
                np.testing.assert_array_equal(obs_a, obs_b)
                result_a, result_b = legacy.step(action), current.step(action)
                np.testing.assert_array_equal(result_a[0], result_b[0])
                self.assertEqual(result_a[1:], result_b[1:])
                np.testing.assert_array_equal(legacy.audio, current.audio)
                np.testing.assert_array_equal(legacy.states, current.states)
                np.testing.assert_array_equal(legacy.applied_control_history,
                                              current.applied_control_history)
                self.assertEqual(legacy.loss_history, current.loss_history)
        finally:
            legacy.close()
            current.close()

    def test_two_variable_hold_matches_35_repeated_historical_c_actions(self):
        sequential = ValidationEnv("C", target_audio=self.target)
        constant = ConstantControlEnv((0, 2), target_audio=self.target)
        try:
            action = np.asarray([.13, -.12], dtype=np.float64)
            sequential.reset(seed=0)
            constant.reset(seed=0)
            constant.step(action)
            for _ in range(PHYSICAL_HORIZON):
                sequential.step(action)
            np.testing.assert_array_equal(sequential.audio, constant.audio)
            np.testing.assert_array_equal(sequential.states, constant.states)
            np.testing.assert_array_equal(sequential.applied_control_history,
                                          constant.applied_control_history)
            error = distance(self.target, np.concatenate(constant.audio), RATE,
                             scale=constant.loss_scale)["total"]
            self.assertEqual(sum(constant.rewards), -error)
            self.assertAlmostEqual(sum(sequential.rewards), sum(constant.rewards), places=13)
            self.assertEqual(len(constant.actions), 1)
            self.assertEqual(len(sequential.actions), 35)
        finally:
            sequential.close()
            constant.close()

    def test_reset_fixed_controls_and_cumulative_compute_counters(self):
        for indices in ((0,), (0, 2)):
            env = ConstantControlEnv(indices, target_audio=self.target)
            fixed = [i for i in range(5) if i not in indices]
            try:
                for episode in (1, 2):
                    obs, _ = env.reset()
                    np.testing.assert_array_equal(env.model.state, np.zeros(6))
                    np.testing.assert_array_equal(env.model.controls, DEFAULT_CONTROLS)
                    np.testing.assert_array_equal(env.decimator.state, np.zeros_like(env.decimator.state))
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(env.episode_native_blocks, 0)
                    self.assertEqual(env.actions, [])
                    decoded = env.decode(np.zeros(len(indices)))
                    np.testing.assert_array_equal(decoded[fixed], np.asarray(TARGET_CONTROLS)[fixed])
                    env.step(np.zeros(len(indices)))
                    applied = np.asarray(env.applied_control_history)
                    np.testing.assert_array_equal(applied[:, fixed],
                                                  np.tile(np.asarray(TARGET_CONTROLS)[fixed], (35, 1)))
                    self.assertEqual(env.completed_episodes, episode)
                    self.assertEqual(env.decision_steps, episode)
                    self.assertEqual(env.native_blocks, 35*episode)
                    self.assertAlmostEqual(env.integrated_s, DURATION*episode)
                    self.assertEqual(env.failures, 0)
            finally:
                env.close()

    def test_action_rng_continues_on_automatic_reset_and_repeats_with_seed(self):
        env = ConstantControlEnv((0, 2), target_audio=self.target)
        try:
            sequences = []
            for _ in range(2):
                env.reset(seed=13)
                first = env.action_space.sample()
                env.reset()
                second = env.action_space.sample()
                sequences.append(np.asarray([first, second]))
            np.testing.assert_array_equal(*sequences)
            self.assertFalse(np.array_equal(sequences[0][0], sequences[0][1]))
        finally:
            env.close()

    def test_two_variable_failure_retains_finite_one_decision_lifecycle(self):
        env = ConstantControlEnv((0, 2), target_audio=self.target)
        try:
            env.reset()
            advance = env.model.advance
            calls = []

            def fail_second_block(seconds):
                calls.append(seconds)
                if len(calls) == 2:
                    raise FloatingPointError("controlled test failure")
                return advance(seconds)

            with patch.object(env.model, "advance", side_effect=fail_second_block):
                obs, reward, done, truncated, info = env.step([0, 0])
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(reward, -1e6)
            self.assertTrue(info["numerical_failure"])
            self.assertEqual(env.decision_steps, 1)
            self.assertEqual(env.completed_episodes, 1)
            self.assertEqual(env.failures, 1)
            self.assertEqual(env.native_blocks, 2)
            self.assertEqual(env.native_blocks_completed, 1)
            self.assertAlmostEqual(env.integrated_s, 2*CONTROL_S)
        finally:
            env.close()

    def test_invalid_conditions_and_actions_rejected(self):
        for indices in ((), (2,), (0, 1), (0, 2, 3), (2, 0), None):
            with self.assertRaises(ValueError):
                ConstantControlEnv(indices, target_audio=self.target)
        env = ConstantControlEnv((0, 2), target_audio=self.target)
        try:
            env.reset()
            for action in ([], [0], [0, 0, 0], [np.nan, 0], [0, np.inf]):
                with self.assertRaises(ValueError):
                    env.step(action)
            self.assertEqual(env.decision_steps, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
