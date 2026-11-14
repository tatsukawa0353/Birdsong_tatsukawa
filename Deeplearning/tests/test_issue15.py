"""Reward selection changes only learning scalars; evaluation stays shared."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from Deeplearning.acoustics import distance
from Deeplearning.experiments import phase0_factories
from Deeplearning.sine_env import SineConfig, SineEnv


class RewardSelectionTests(unittest.TestCase):
    def test_scalar_decomposition_and_bitwise_identical_generation(self):
        modes=("combined","spectral_only","envelope_only")
        envs=[SineEnv(SineConfig(reward_terms=m,reseed_action_space_on_reset=False)) for m in modes]
        try:
            observations=[e.reset(seed=0)[0] for e in envs]
            actions=np.random.default_rng(4).uniform(-1,1,(35,2))
            for action in actions:
                records=[e.step(action) for e in envs]
                for index,(mode,record) in enumerate(zip(modes,records)):
                    obs,reward,done,truncated,info=record
                    self.assertIsInstance(reward,float)
                    self.assertEqual(reward,-info[{"combined":"total","spectral_only":"log_spectral","envelope_only":"envelope"}[mode]])
                    np.testing.assert_array_equal(obs,records[0][0])
                    self.assertEqual(info["total"],info["log_spectral"]+info["envelope"])
                self.assertNotEqual(records[0][1],records[1][1])
            for env in envs:
                np.testing.assert_array_equal(np.concatenate(env.audio),np.concatenate(envs[0].audio))
                np.testing.assert_array_equal(env.target_audio,envs[0].target_audio)
            self.assertEqual(SineConfig().reward_terms,"combined")
            with self.assertRaises(ValueError): SineConfig(reward_terms="not_a_mode")
        finally:
            for env in envs: env.close()

    def test_all_learning_conditions_use_the_same_combined_evaluation(self):
        from Deeplearning.issue14 import evaluate_agent
        from Deeplearning.experiments import make_agent
        for mode in ("spectral_only","envelope_only"):
            settings=dict(target="glide",randomize_training_target=False,reseed_action_space_on_reset=False,
                          reward_terms=mode,learning_rate=.0003,ppo_rollout=256)
            train,evaluate=phase0_factories(settings)
            training,testing=train(),evaluate()
            try:
                self.assertEqual(training.config.reward_terms,mode)
                self.assertEqual(testing.config.reward_terms,"combined")
                agent=make_agent("PPO",training,0,settings)
                record,snapshot=evaluate_agent(agent,evaluate,0,"initial")
                expected=distance(snapshot["target"],snapshot["audio"],48000,scale=1)
                self.assertEqual(record["metrics"],expected)
                self.assertEqual(agent.num_timesteps,0)
                self.assertEqual(agent._n_updates,0)
                self.assertTrue(record["learner_unchanged"])
            finally:
                training.close(); testing.close()

    def test_oracle_and_saved_reward_trace(self):
        from Deeplearning.issue14 import train_one
        for mode in ("combined","spectral_only","envelope_only"):
            env=SineEnv(SineConfig(reward_terms=mode))
            try:
                env.reset(seed=0)
                for action in env.goal_actions.copy(): env.step(action)
                self.assertLess(distance(env.target_audio,np.concatenate(env.audio),48000,scale=1)["total"],1e-5)
            finally: env.close()
        settings=dict(target="glide",randomize_training_target=False,reseed_action_space_on_reset=False,
                      reward_terms="spectral_only",steps=256,eval_interval=128,learning_rate=.0003,ppo_rollout=64,threshold=.08)
        with TemporaryDirectory(prefix="birdsong-issue15-test-") as directory:
            p=Path(directory)/"PPO_0"
            result=train_one("PPO",0,settings,p)
            with np.load(p/"training_trace.npz") as trace:
                np.testing.assert_allclose(trace["rewards"],-trace["log_spectral"],rtol=1e-6,atol=1e-8)
                self.assertGreater(float(np.mean(trace["envelope"])),0)
            self.assertEqual(result["training_reward_terms"],"spectral_only")
            self.assertEqual(result["evaluation_reward_terms"],"combined")
            self.assertEqual(result["training_steps"],256)


    def test_weighted_lambda_half_reward_only_changes_training(self):
        with self.assertRaises(ValueError):
            SineConfig(reward_terms="weighted", reward_envelope_weight=float("nan"))
        with self.assertRaises(ValueError):
            SineConfig(reward_terms="weighted", reward_envelope_weight=-0.1)

        combined = SineEnv(SineConfig(reward_terms="combined"))
        weighted = SineEnv(SineConfig(reward_terms="weighted", reward_envelope_weight=0.5))
        try:
            a, _ = combined.reset(seed=3)
            b, _ = weighted.reset(seed=3)
            np.testing.assert_array_equal(a, b)
            action = combined.encode(1240.0, 0.45)
            first = combined.step(action)
            second = weighted.step(action)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(np.concatenate(combined.audio), np.concatenate(weighted.audio))
            metrics = second[4]
            self.assertAlmostEqual(second[1], -(metrics["log_spectral"] + 0.5 * metrics["envelope"]), places=12)
            self.assertEqual(first[1], -first[4]["total"])
        finally:
            combined.close()
            weighted.close()
        settings = dict(target="glide", randomize_training_target=False,
                        reseed_action_space_on_reset=False, reward_terms="weighted",
                        reward_envelope_weight=0.5, learning_rate=0.0003, ppo_rollout=256)
        train, evaluate = phase0_factories(settings)
        env_train, env_eval = train(), evaluate()
        try:
            self.assertEqual(env_train.config.reward_terms, "weighted")
            self.assertEqual(env_train.config.reward_envelope_weight, 0.5)
            self.assertEqual(env_eval.config.reward_terms, "combined")
            self.assertEqual(env_eval.config.reward_envelope_weight, 1.0)
        finally:
            env_train.close()
            env_eval.close()


if __name__=="__main__": unittest.main()
