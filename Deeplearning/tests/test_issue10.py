import unittest

import numpy as np

from Deeplearning.acoustics import distance
from Deeplearning.experiments import rollout
from Deeplearning.model_env import ModelConfig, ModelEnv
from Deeplearning.objectives import BalancedConfig, BalancedObjective


class Issue10Tests(unittest.TestCase):
    def setUp(self):
        self.rate = 50000
        self.t = np.arange(17500)/self.rate
        self.target = .6*np.sin(2*np.pi*1000*self.t)

    def test_oracle_silence_dc_wrong_frequency_order(self):
        obj = BalancedObjective(self.target, self.rate)
        wrong = .6*np.sin(2*np.pi*1600*self.t)
        self.assertEqual(obj.score(self.target)["total"], 0)
        wrong_error = obj.score(wrong)["total"]
        for generated in (np.zeros_like(self.target), self.target*1e-6, np.full_like(self.target, .6)):
            self.assertGreater(obj.score(generated)["total"], wrong_error)

    def test_correct_quiet_and_noise_targets_have_no_voice_bonus(self):
        silence = np.zeros_like(self.target)
        obj = BalancedObjective(silence, self.rate, scale=1)
        self.assertFalse(obj.voiced.any())
        self.assertEqual(obj.score(silence)["total"], 0)
        self.assertGreater(obj.score(self.target)["total"], 1)
        noise = np.random.default_rng(0).normal(size=len(self.target))*.1
        obj = BalancedObjective(noise, self.rate)
        self.assertGreater(np.mean(obj.voiced), .9)
        self.assertEqual(obj.score(noise)["total"], 0)
        self.assertGreater(obj.score(silence)["total"], 2)

    def test_gain_loudness_and_voicing_timing_are_penalized(self):
        obj = BalancedObjective(self.target, self.rate)
        self.assertGreater(obj.score(self.target*.5)["total"], 0)
        self.assertGreater(obj.score(self.target*10)["total"], obj.score(self.target*.5)["total"])
        gated = self.target*(self.t >= .1)*(self.t < .25)
        obj = BalancedObjective(gated, self.rate, scale=1)
        shifted = np.r_[np.zeros(1000), gated[:-1000]]
        self.assertGreater(obj.score(shifted)["total"], .01)
        self.assertGreater(obj.score(self.target)["total"], obj.score(gated)["total"])
        # Voice mask comes from the target once; generated gain cannot change it.
        before = obj.voiced.copy()
        obj.score(self.target*100)
        np.testing.assert_array_equal(before, obj.voiced)

    def test_nonoverlapping_reward_assignments_and_no_terminal_frame_inflation(self):
        obj = BalancedObjective(self.target, self.rate)
        generated = self.target*.7
        contributions, frames = [], 0
        for end in range(500, len(generated)+1, 500):
            item = obj.update(generated[:end])
            contributions.append(item["contribution"]["total"])
            frames += item["frames_added"]
        self.assertEqual(frames, len(obj.ends))
        self.assertAlmostEqual(sum(contributions), obj.score(generated)["total"], places=12)
        self.assertAlmostEqual(obj.weights.sum(), 1)
        self.assertAlmostEqual(obj.weights[-1], 1/len(self.target))
        self.assertEqual(obj.update(generated)["contribution"]["total"], 0)

    def test_prefix_features_are_causal(self):
        obj = BalancedObjective(self.target, self.rate)
        prefix = obj.frame_parts(self.target[:5000]*.8)
        full = obj.frame_parts(self.target*.8)
        np.testing.assert_allclose(prefix, full[:len(prefix)], rtol=1e-11, atol=1e-12)

    def test_padding_audit_counts_complete_odd_window_frame(self):
        from Deeplearning.issue10 import legacy_padding_counts
        self.assertEqual(legacy_padding_counts(500), dict(frames=4, padded_frames=4))
        self.assertEqual(legacy_padding_counts(1000), dict(frames=6, padded_frames=6))
        self.assertEqual(legacy_padding_counts(1500), dict(frames=7, padded_frames=6))
        self.assertEqual(legacy_padding_counts(17500), dict(frames=60, padded_frames=6))

    def test_baseline_audit_does_not_change_rewards_or_waveform(self):
        a = ModelEnv(ModelConfig())
        b = ModelEnv(ModelConfig(audit_objective=True))
        try:
            ra, wa, _ = rollout(a, lambda obs, e: np.zeros(2))
            rb, wb, _ = rollout(b, lambda obs, e: np.zeros(2))
            np.testing.assert_array_equal(wa, wb)
            np.testing.assert_array_equal(a.rewards, b.rewards)
            self.assertEqual(ra["error"], rb["error"])
            self.assertEqual(ra["metrics"], distance(a.target_audio, wa, self.rate, scale=a.loss_scale))
        finally:
            a.close()
            b.close()

    def test_aligned_discounted_return_equals_final_objective(self):
        env = ModelEnv(ModelConfig(loss_kind="balanced", reward_timing="aligned", diagnostics=True))
        try:
            result, _, _ = rollout(env, lambda obs, e: np.zeros(2))
            self.assertAlmostEqual(result["discounted_reward_cost"], result["objective_error"], places=12)
            self.assertEqual(sum(x["objective"]["frames_added"] for x in env.loss_history), len(env.objective.ends))
        finally:
            env.close()

    def test_objective_has_no_display_setting_dependency(self):
        from Deeplearning import plotting
        obj = BalancedObjective(self.target, self.rate)
        before = obj.score(self.target*.5)
        old = plotting.VMIN, plotting.VMAX
        try:
            plotting.VMIN, plotting.VMAX = -120, 0
            self.assertEqual(before, obj.score(self.target*.5))
        finally:
            plotting.VMIN, plotting.VMAX = old

    def test_invalid_config_and_shapes(self):
        with self.assertRaises(ValueError):
            BalancedConfig(missing_weight=-1)
        with self.assertRaises(ValueError):
            BalancedObjective(self.target, self.rate).score(self.target[:-1])
        with self.assertRaises(ValueError):
            ModelConfig(loss_kind="baseline", reward_timing="aligned")


if __name__ == "__main__":
    unittest.main()
