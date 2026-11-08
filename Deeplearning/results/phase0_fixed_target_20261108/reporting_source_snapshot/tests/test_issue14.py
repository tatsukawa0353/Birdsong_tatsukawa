"""Fixed-target identity and evaluation integrity; no production run writes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from Deeplearning.acoustics import distance
from Deeplearning.sine_env import SineConfig, SineEnv, oscillate


class Issue14EnvironmentTests(unittest.TestCase):
    def test_fixed_target_is_identical_across_seeds_and_automatic_resets(self):
        reference = SineEnv(SineConfig(randomize=False))
        reference.reset(seed=10000)
        expected = (reference.frequencies.copy(), reference.amplitudes.copy(), reference.target_audio.copy())
        for seed in (0, 1, 2, 10000):
            env = SineEnv(SineConfig(randomize=False))
            try:
                for reset_seed in (seed, None, None):
                    observation, _ = env.reset(seed=reset_seed)
                    for actual, wanted in zip((env.frequencies, env.amplitudes, env.target_audio), expected):
                        np.testing.assert_array_equal(actual, wanted)
                    self.assertEqual(env.phase, 0)
                    self.assertEqual(env.index, 0)
                    np.testing.assert_array_equal(env.previous, np.zeros(2))
                    self.assertEqual(observation.shape, (5,))
                self.assertGreater(float(np.ptp(env.amplitudes)), .2)
            finally:
                env.close()
        reference.close()

    def test_factory_preserves_old_randomized_default_and_fixed_opt_in(self):
        from Deeplearning.experiments import phase0_factories
        old_train, old_eval = phase0_factories({"target": "glide"})
        fixed_train, fixed_eval = phase0_factories({"target": "glide", "randomize_training_target": False})
        environments = [factory() for factory in (old_train, old_eval, fixed_train, fixed_eval)]
        try:
            random_env, old_eval_env, fixed_train_env, fixed_eval_env = environments
            self.assertTrue(random_env.config.randomize)
            self.assertFalse(old_eval_env.config.randomize)
            random_env.reset(seed=0)
            first_random_target = random_env.target_audio.copy()
            random_env.reset()
            self.assertFalse(np.array_equal(first_random_target, random_env.target_audio))
            for env in (old_eval_env, fixed_train_env, fixed_eval_env):
                env.reset(seed=2)
                self.assertFalse(env.config.randomize)
            np.testing.assert_array_equal(old_eval_env.target_audio, fixed_train_env.target_audio)
            np.testing.assert_array_equal(fixed_train_env.target_audio, fixed_eval_env.target_audio)
        finally:
            for env in environments:
                env.close()

    def test_oracle_35_actions_and_phase_continuity(self):
        env = SineEnv(SineConfig(randomize=False))
        try:
            env.reset(seed=0)
            phase = 0.
            expected_blocks = []
            actions = env.goal_actions.copy()
            for index, action in enumerate(actions):
                f, a = env.decode(action)
                wave, phase = oscillate(f, a, phase, 480, 48000)
                expected_blocks.append(wave)
                _, reward, done, truncated, _ = env.step(action)
                self.assertEqual(done, index == 34)
                self.assertFalse(truncated)
                self.assertEqual(env.phase, phase)
                self.assertAlmostEqual(reward, -distance(env.target_audio[index*480:(index+1)*480], wave, 48000, scale=1)["total"], places=14)
            audio = np.concatenate(env.audio)
            np.testing.assert_array_equal(audio, np.concatenate(expected_blocks))
            self.assertEqual(len(audio), 16800)
            self.assertEqual(len(env.actions), 35)
            self.assertLess(distance(env.target_audio, audio, 48000, scale=1)["total"], 1e-6)
            with self.assertRaises(RuntimeError):
                env.step(actions[-1])
        finally:
            env.close()

    def test_step_reward_sum_and_full_episode_error_are_distinct(self):
        env = SineEnv(SineConfig(randomize=False))
        try:
            env.reset(seed=0)
            for _ in range(35):
                env.step(np.zeros(2, dtype=np.float32))
            full_error = distance(env.target_audio, np.concatenate(env.audio), 48000, scale=1)["total"]
            self.assertGreater(abs(-sum(env.rewards)-full_error), 1.)
            self.assertEqual(env.config.smooth_weight, 0)
        finally:
            env.close()


class Issue14EvaluationTests(unittest.TestCase):
    @staticmethod
    def settings():
        return dict(algorithms=["PPO", "TD3", "SAC"], seeds=[0, 1, 2],
                    target="glide", randomize_training_target=False, steps=64,
                    eval_interval=32, learning_rate=.0003, ppo_rollout=64,
                    threshold=.08)

    def test_evaluation_cannot_train_or_insert_replay_experience(self):
        from Deeplearning.experiments import make_agent, phase0_factories
        from Deeplearning.issue14 import evaluate_agent
        training_factory, evaluation_factory = phase0_factories(self.settings())
        env = training_factory()
        agent = make_agent("TD3", env, 0, self.settings())
        before = {key: value.detach().cpu().clone() for key, value in agent.policy.state_dict().items()}
        buffer_state = (agent.replay_buffer.pos, agent.replay_buffer.size(), agent.replay_buffer.full)
        try:
            with patch.object(agent, "train", side_effect=AssertionError("evaluation must not train")), \
                    patch.object(agent.replay_buffer, "add", side_effect=AssertionError("evaluation must not add replay experience")):
                record, snapshot = evaluate_agent(agent, evaluation_factory, 0, "initial")
            self.assertTrue(record["learner_unchanged"])
            self.assertEqual(agent.num_timesteps, 0)
            self.assertEqual(agent._n_updates, 0)
            self.assertEqual(buffer_state, (agent.replay_buffer.pos, agent.replay_buffer.size(), agent.replay_buffer.full))
            for key, value in agent.policy.state_dict().items():
                np.testing.assert_array_equal(value.detach().cpu().numpy(), before[key].numpy())
            self.assertEqual(snapshot["actions"].shape, (35, 2))
            self.assertEqual(len(snapshot["audio"]), 16800)
        finally:
            env.close()

    def test_final_curve_summary_checkpoint_and_audio_use_post_update_policy(self):
        from Deeplearning.issue14 import train_one
        from stable_baselines3 import PPO
        settings = self.settings()
        with TemporaryDirectory(prefix="birdsong-issue14-final-") as temporary:
            directory = Path(temporary) / "PPO_0"
            summary = train_one("PPO", 0, settings, directory)
            curve = json.loads((directory / "curve.json").read_text())
            callback = json.loads((directory / "callback_curve.json").read_text())
            self.assertEqual(curve[-1]["evaluation_phase"], "post_final_update")
            self.assertEqual(callback[-1]["steps"], settings["steps"])
            self.assertNotEqual(callback[-1]["parameter_sha256"], curve[-1]["parameter_sha256"])
            self.assertEqual(curve[-1]["parameter_sha256"], summary["final_parameter_sha256"])
            self.assertEqual(curve[-1]["error"], summary["error"])
            self.assertEqual(summary["training_steps"], settings["steps"])
            agent = PPO.load(directory / "agent.zip", device="cpu")
            env = SineEnv(SineConfig(randomize=False))
            try:
                observation, _ = env.reset(seed=10000)
                for _ in range(35):
                    action, _ = agent.predict(observation, deterministic=True)
                    observation, _, _, _, _ = env.step(action)
                with np.load(directory / "rollout.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive["actions"], np.asarray(env.actions))
                    np.testing.assert_array_equal(archive["audio"], np.concatenate(env.audio))
                    self.assertEqual(summary["error"], distance(archive["target"], archive["audio"], 48000, scale=1)["total"])
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
