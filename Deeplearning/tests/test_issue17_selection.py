import copy
import unittest
from Deeplearning.issue17_selection import selection_curve


class ObjectiveSelectionTests(unittest.TestCase):
    def test_weighted_selection_can_increase_common_error(self):
        points=[dict(episodes=i+1,acoustic_components=dict(log_spectral=s,envelope=e),
            acoustic_error=s+e,failed=False) for i,(s,e) in enumerate(((.30,.02),(.12,.30),(.11,.32)))]
        before=copy.deepcopy(points);half=selection_curve(points,.5);full=selection_curve(points,1.)
        self.assertEqual([p['objective_selected_trial'] for p in half],[1,2,2])
        self.assertEqual([p['common_posthoc_trial'] for p in half],[1,1,1])
        self.assertGreater(half[1]['objective_selected_common_error'],half[0]['objective_selected_common_error'])
        self.assertEqual(full[-1]['objective_selected_trial'],1)
        self.assertEqual(points,before)

    def test_failure_never_wins_on_padded_zero_components(self):
        points=[dict(episodes=1,acoustic_components=dict(log_spectral=.2,envelope=.2),acoustic_error=.4,failed=False),
                dict(episodes=2,acoustic_components=dict(log_spectral=0.,envelope=0.),acoustic_error=1e6,failed=True)]
        self.assertEqual(selection_curve(points,.5)[-1]['objective_selected_trial'],1)
