"""Invariants for the isolated synthetic Phase 1 validation protocol."""
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.env_checker import check_env

from Deeplearning.acoustics import distance
from Deeplearning.model import DEFAULT_CONTROLS, HIGH, LOW
from Deeplearning.model_env import ModelConfig, ModelEnv
from Deeplearning.phase1_validation_env import (
    CONTROL_S, DURATION, PHYSICAL_HORIZON, RATE, TARGET_CONTROLS, ValidationEnv,
)


class ValidationEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = ValidationEnv("A")
        cls.target = env.target_audio.copy()
        env.close()

    def run_actions(self, env, actions):
        env.reset(seed=0)
        for action in actions:
            _, _, done, _, info = env.step(action)
            if done:
                break
        self.assertTrue(done)
        return np.concatenate(env.audio), info

    def test_legacy_decode_unchanged_and_one_variable_fixed_control(self):
        legacy = ModelEnv(ModelConfig(control_indices=(0,)), external_target=self.target)
        validation = ValidationEnv("B", self.target)
        try:
            self.assertEqual(legacy.decode_requested([0])[2], DEFAULT_CONTROLS[2])
            self.assertEqual(validation.decode([0])[2], TARGET_CONTROLS[2])
            for action in (-1, 0, 1):
                decoded = validation.decode([action])
                np.testing.assert_array_equal(decoded[1:], np.asarray(TARGET_CONTROLS)[1:])
            self.assertEqual(validation.decode([-1])[0], LOW[0])
            self.assertEqual(validation.decode([1])[0], HIGH[0])
        finally:
            legacy.close()
            validation.close()

    def test_float64_oracles_and_original_amplitude(self):
        for task in ("A", "B", "C"):
            env = ValidationEnv(task, self.target)
            try:
                action = env.encode(TARGET_CONTROLS)
                self.assertEqual(action.dtype, np.float64)
                generated, info = self.run_actions(env, [action]*env.horizon)
                np.testing.assert_array_equal(generated, self.target)
                self.assertEqual(info["episode_metrics"]["total"], 0)
                self.assertEqual(len(generated), round(DURATION*RATE))
                self.assertEqual(len(env.applied_control_history), PHYSICAL_HORIZON)
                self.assertEqual(len(env.actions), env.horizon)
                self.assertEqual(env.native_blocks, PHYSICAL_HORIZON)
                self.assertAlmostEqual(env.integrated_s, DURATION)
                self.assertEqual(env.completed_episodes, 1)
            finally:
                env.close()

    def test_float32_oracle_is_near_numerical_precision(self):
        env = ValidationEnv("C", self.target)
        try:
            action = env.encode(TARGET_CONTROLS, dtype=np.float32)
            _, info = self.run_actions(env, [action]*env.horizon)
            self.assertLess(info["episode_metrics"]["total"], 1e-5)
        finally:
            env.close()

    def test_return_equals_negative_full_error_and_reset_preserves_accounting(self):
        for task in ("A", "B", "C"):
            env = ValidationEnv(task, self.target)
            try:
                rng = np.random.default_rng(4)
                actions = rng.uniform(-.2, .2, (env.horizon, len(env.indices)))
                audio, _ = self.run_actions(env, actions)
                error = distance(self.target, audio, RATE, scale=env.loss_scale)["total"]
                self.assertAlmostEqual(sum(env.rewards), -error, places=13)
                self.assertEqual(env.decision_steps, env.horizon)
                before = env.integrated_s
                env.reset()
                self.assertEqual(env.integrated_s, before)
                self.assertEqual(env.completed_episodes, 1)
                self.assertEqual(env.actions, [])
                self.assertEqual(env.episode_native_blocks, 0)
                self.assertEqual(env.setup_simulation_s, 0)
            finally:
                env.close()

    def test_failure_is_full_episode_penalty_and_attempts_are_counted(self):
        env = ValidationEnv("B", self.target)
        try:
            env.reset()
            env.step([0])
            with patch.object(env.model, "advance", side_effect=FloatingPointError("test")):
                obs, reward, done, truncated, info = env.step([0])
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertTrue(info["numerical_failure"])
            self.assertTrue(np.isfinite(obs).all())
            self.assertAlmostEqual(sum(env.rewards), -1e6)
            self.assertEqual(env.failures, 1)
            self.assertEqual(env.native_blocks, 2)
            self.assertEqual(env.native_blocks_completed, 1)
            self.assertAlmostEqual(env.integrated_s, 2*CONTROL_S)
            self.assertAlmostEqual(env.integrated_completed_s, CONTROL_S)
            with self.assertRaises(RuntimeError):
                env.step([0])
        finally:
            env.close()

    def test_gym_contract_target_is_read_only_and_action_validation(self):
        for task in ("A", "B", "C"):
            env = ValidationEnv(task, self.target)
            try:
                check_env(env, warn=True)
                self.assertFalse(env.target_audio.flags.writeable)
                self.assertFalse(env.target_controls.flags.writeable)
                env.reset()
                for invalid in ([np.nan]*len(env.indices), [], [0]*(len(env.indices)+1)):
                    with self.assertRaises(ValueError):
                        env.step(invalid)
            finally:
                env.close()
        with self.assertRaises(ValueError):
            ValidationEnv("invalid")

    def test_seeded_warmup_random_stream_survives_automatic_resets(self):
        env = ValidationEnv("C", self.target)
        try:
            sequences = []
            for _ in range(2):
                env.reset(seed=123)
                first = env.action_space.sample()
                env.reset()  # SB3 automatically resets with no new seed.
                second = env.action_space.sample()
                sequences.append(np.asarray([first, second]))
            np.testing.assert_array_equal(*sequences)
            self.assertFalse(np.array_equal(sequences[0][0], sequences[0][1]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
