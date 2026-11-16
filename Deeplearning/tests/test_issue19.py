"""Scientific invariants specific to the restricted-domain weight comparison."""
import unittest

import numpy as np

from Deeplearning.constant020_env import episode
from Deeplearning.issue19_data import environment, curve_for, endpoints
from Deeplearning.issue19_critics import scores


class RestrictedWeightTests(unittest.TestCase):
    def test_weight_changes_only_reward_on_same_physical_candidate(self):
        results = []
        for weight in (1., .5):
            env = environment(weight)
            try:
                obs, _ = env.reset()
                np.testing.assert_array_equal(obs, [-1, 0, 0, 0, 0, 0, 0, 0])
                np.testing.assert_array_equal(env.decode([-1, -1, -1])[[0,2,3]], [2.5e7,2.5e6,49000.])
                np.testing.assert_array_equal(env.decode([1, 1, 1])[[0,2,3]], [5e7,4e6,800000.])
                results.append(episode(env, action=[.1, .2, -.5]))
            finally:
                env.close()
        a, b = results
        np.testing.assert_array_equal(a[1]['audio'], b[1]['audio'])
        self.assertEqual(a[0]['acoustic_error'], b[0]['acoustic_error'])
        self.assertAlmostEqual(a[0]['weighted_loss']-b[0]['weighted_loss'], .5*a[0]['acoustic_components']['envelope'])

    def test_random_rescore_keeps_all_candidates_and_actual_final(self):
        for seed in (0, 1, 2):
            old = curve_for('random_search', seed, 1.); new = curve_for('random_search', seed, .5)
            self.assertEqual(len(new), 4096)
            self.assertEqual([r['actions'] for r in old], [r['actions'] for r in new])
            self.assertEqual([r['acoustic_error'] for r in old], [r['acoustic_error'] for r in new])
            chosen = endpoints('random_search', seed, .5)
            self.assertEqual(chosen['final']['episodes'], 4096)
            self.assertEqual(chosen['selected']['weighted_loss'], min(r['weighted_loss'] for r in new))

    def test_critic_rank_reversal_and_signed_bias(self):
        perfect = scores([-3, -2, -1], [-3, -2, -1])
        reversed_order = scores([-3, -2, -1], [-1, -2, -3])
        optimistic = scores([-3, -2, -1], [-2, -1, 0])
        self.assertEqual(perfect['rmse'], 0.)
        self.assertEqual(perfect['pairwise_inversion_fraction'], 0.)
        self.assertEqual(reversed_order['pairwise_inversion_fraction'], 1.)
        self.assertEqual(reversed_order['spearman'], -1.)
        self.assertEqual(optimistic['bias_Q_minus_r'], 1.)

    def test_critic_ties_are_excluded_from_inversion_denominator(self):
        result = scores([-1, -1, -2], [.5, .5, .5])
        self.assertEqual(result['comparable_pairs'], 0)
        self.assertIsNone(result['pairwise_inversion_fraction'])
        self.assertIsNone(result['spearman'])


if __name__ == '__main__':
    unittest.main()
