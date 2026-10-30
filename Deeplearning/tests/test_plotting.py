"""Regression checks for the physical STFT window and boundary-free display."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal

from Deeplearning.acoustics import resample
from Deeplearning.plotting import display_spectrum, draw_spectrogram, plt


class DisplayTests(unittest.TestCase):
    def test_saved_audio_is_resampled_before_fixed_stft(self):
        scipy_spectrogram = signal.spectrogram
        for rate in (44100, 48000, 50000):
            x = np.sin(2*np.pi*1000*np.arange(round(.35*rate))/rate)
            with patch("Deeplearning.plotting.signal.spectrogram", wraps=scipy_spectrogram) as spy:
                _, times, _ = display_spectrum(x, rate)
            settings = spy.call_args.kwargs
            self.assertEqual(settings["fs"], 10_000_000)
            self.assertEqual(settings["nperseg"], 245760)
            self.assertEqual(settings["noverlap"], 184320)
            self.assertEqual(settings["window"], "blackmanharris")
            self.assertEqual(len(spy.call_args.args[0]), 3_500_000)
            self.assertAlmostEqual(times[0], .012288)
            self.assertAlmostEqual(times[-1], .331776)

    def test_only_complete_windows_match_unpadded_scipy(self):
        rate = 50000
        x = np.sin(2*np.pi*1000*np.arange(17500)/rate)
        f, t, db = display_spectrum(x, rate)
        n, overlap = 245760, 184320
        ref_f, ref_t, ref_p = signal.spectrogram(resample(x, rate, 10_000_000), fs=10_000_000, window="blackmanharris",
                                                nperseg=n, noverlap=overlap,
                                                detrend="constant", scaling="density", mode="psd")
        keep = ref_f <= 10000
        np.testing.assert_array_equal(f, ref_f[keep])
        np.testing.assert_array_equal(t, ref_t)
        np.testing.assert_array_equal(db, (10*np.log10(ref_p/ref_p.max()+1e-10))[keep])
        self.assertGreater(t[0], 0)
        self.assertLess(t[-1], .35)
        self.assertTrue(np.all(t-n/(2*10_000_000) >= -1e-12))
        self.assertTrue(np.all(t+n/(2*10_000_000) <= .35+1e-12))

    def test_color_scale_and_time_axis_exclude_both_ends(self):
        rate = 50000
        x = np.sin(2*np.pi*1000*np.arange(17500)/rate)
        _, times, _ = display_spectrum(x, rate)
        fig, ax = plt.subplots()
        try:
            mesh = draw_spectrogram(ax, x, rate)
            self.assertEqual(mesh.get_clim(), (-50.0, -20.0))
            self.assertEqual(mesh.cmap.name, "gray_r")
            np.testing.assert_array_equal(ax.get_xlim(), (times[0], times[-1]))
            self.assertEqual(ax.get_ylim(), (0, 10000))
        finally:
            plt.close(fig)

    def test_short_failed_rollout_has_no_padded_display(self):
        fig, ax = plt.subplots()
        try:
            self.assertIsNone(draw_spectrogram(ax, np.zeros(500), 50000))
            self.assertEqual(len(ax.collections), 0)
            self.assertFalse(ax.axison)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
