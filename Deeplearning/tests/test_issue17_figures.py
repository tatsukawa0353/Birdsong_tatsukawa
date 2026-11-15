import tempfile
from pathlib import Path
import unittest
import numpy as np
from Deeplearning.issue17_comparison_figures import comparison_curve,METHODS,COLORS


class FourMethodFigureTests(unittest.TestCase):
    def test_raw_method_positions_styles_and_inside_legend_are_preserved(self):
        trajectories={}
        for method in METHODS:
            x=np.arange(0,4097,128) if method in ('TD3','SAC') else np.arange(1,4097)
            a=np.array([np.linspace(.5,.1,len(x))+offset for offset in (0,.01,-.01)])
            trajectories[method]=(x,a)
        with tempfile.TemporaryDirectory() as directory:
            for average in (False,True):
                r=comparison_curve(trajectories,Path(directory)/f'{average}.png',limits=(-.05,.9),average=average)
                self.assertEqual(r['line_count'],4)
                self.assertEqual(r['band_count'],4 if average else 0)
                self.assertEqual(r['legend_fontsizes'],[29])
                self.assertEqual(r['tick_fontsizes'],[30])
                self.assertEqual(r['label_fontsize'],46)
                self.assertTrue(r['legend_inside'])
                self.assertFalse(r['legend_line_overlaps'] or r['legend_band_overlap'] or r['title'] or r['path_simplify'] or r['grid'])
                self.assertEqual(r['xlabel'],'Number of trials')
                self.assertEqual(r['ylabel'],'Acoustic error')
                for method in METHODS:
                    self.assertEqual(r['methods'][method]['vertices'],len(trajectories[method][0]))
                    self.assertEqual(r['methods'][method]['color'],COLORS[method])
                    self.assertGreaterEqual(r['methods'][method]['linewidth'],2.8)

    def test_missing_method_cannot_silently_produce_a_partial_comparison(self):
        with self.assertRaises(ValueError):
            comparison_curve({},Path('/tmp/unused_issue17.png'),limits=(0,1))
