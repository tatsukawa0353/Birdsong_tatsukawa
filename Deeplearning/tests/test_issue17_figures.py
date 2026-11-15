import tempfile
from pathlib import Path
import unittest
import numpy as np
from Deeplearning.issue17_figures import single_curve


class SingleConditionFigureTests(unittest.TestCase):
    def test_single_curve_large_axes_no_title_legend_or_vertex_simplification(self):
        with tempfile.TemporaryDirectory() as directory:
            x=np.arange(0,4097,128);y=np.linspace(.6,.1,len(x))
            for band in (None,(y-.02,y+.02)):
                result=single_curve(x,y,Path(directory)/('mean.png' if band else 'seed.png'),
                                    band=band,limits=(-.05,.7))
                self.assertEqual(result['line_count'],1)
                self.assertEqual(result['band_count'],int(band is not None))
                self.assertEqual(result['vertex_count'],len(x))
                self.assertGreaterEqual(result['line_width'],2.8)
                self.assertEqual(result['xlabel'],'Number of trials')
                self.assertEqual(result['ylabel'],'Acoustic error')
                self.assertEqual(result['xlabel_fontsize'],46)
                self.assertEqual(result['ylabel_fontsize'],46)
                self.assertEqual(result['tick_fontsizes'],[30])
                self.assertEqual(result['title'],'')
                self.assertFalse(result['legend'] or result['path_simplify'] or result['x_grid_visible'] or result['y_grid_visible'])
                self.assertEqual(result['xlim'],[0,4096])
                self.assertEqual(result['ylim'],[-.05,.7])
