import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Deeplearning.issue12 import guarded_paths, inside, synthetic_fixtures, manifest
from Deeplearning.model import ROOT, LOW, HIGH
from Deeplearning.objectives import BalancedObjective
from Deeplearning.offline_diagnostics import (variant_parts, target_mask, temporal_regions,
                                             component_regions, shape_distribution, activity_metrics,
                                             ranking_changes, waveform_response, control_diagnostics,
                                             safe_correlation)


class Issue12Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"configs/issue12.json").read_text())
        self.target = synthetic_fixtures()["oracle"][0]
        self.obj = BalancedObjective(self.target, 50000, scale=1)

    def test_default_parts_and_region_contributions_reproduce_original(self):
        generated = self.target*.7
        result = variant_parts(self.obj, generated, self.config["variants"])["default"]
        np.testing.assert_allclose(result["parts"], self.obj.frame_parts(generated), rtol=1e-13, atol=1e-15)
        self.assertAlmostEqual(result["total"], self.obj.score(generated)["total"], places=13)
        labels, _ = temporal_regions(self.obj)
        regions = component_regions(self.obj, result["contributions"], labels)
        self.assertAlmostEqual(sum(r["duration_fraction"] for r in regions), 1, places=13)
        self.assertAlmostEqual(sum(r["total"] for r in regions), result["total"], places=13)
        self.assertAlmostEqual(self.obj.weights[-1], 1/17500)

    def test_mask_change_only_restricts_gated_terms(self):
        generated = self.target*0
        result = variant_parts(self.obj, generated, self.config["variants"])
        a, b = result["default"]["parts"], result["mask_10"]["parts"]
        np.testing.assert_array_equal(a[:, 2], b[:, 2])
        np.testing.assert_array_equal(a[:, 4], b[:, 4])
        self.assertTrue(np.all(b[:, [0, 1, 3]] <= a[:, [0, 1, 3]]))
        self.assertGreater(np.sum(self.obj.weights[target_mask(self.obj, .01)]),
                           np.sum(self.obj.weights[target_mask(self.obj, .1)]))

    def test_energy_weighting_does_not_change_envelope_missing_dc(self):
        generated = np.roll(self.target, 1000)
        result = variant_parts(self.obj, generated, self.config["variants"])
        a = result["default"]["parts"]
        for name in ("transport_energy", "spectral_energy"):
            b = result[name]["parts"]
            np.testing.assert_array_equal(a[:, [2, 3, 4]], b[:, [2, 3, 4]])
            self.assertTrue(np.all(b[:, 0] <= a[:, 0]+1e-15))
        np.testing.assert_array_equal(a[:, 1], result["transport_energy"]["parts"][:, 1])

    def test_oracle_quiet_noise_and_false_voicing(self):
        fixtures = synthetic_fixtures()
        for name in ("oracle", "quiet_correct", "noise_oracle"):
            target, generated = fixtures[name]
            obj = BalancedObjective(target, 50000, scale=1)
            values = variant_parts(obj, generated, self.config["variants"])
            self.assertTrue(all(v["total"] == 0 for v in values.values()))
        for name in ("quiet_false_voice", "noise_missing", "noise_wrong", "delayed", "loud"):
            target, generated = fixtures[name]
            obj = BalancedObjective(target, 50000, scale=1)
            values = variant_parts(obj, generated, self.config["variants"])
            self.assertTrue(all(v["total"] > 0 for v in values.values()), name)

    def test_high_mask_can_hide_weak_voice_but_not_all_amplitude_error(self):
        _, generated = synthetic_fixtures()["missing_weak"]
        result = variant_parts(self.obj, generated, self.config["variants"])
        self.assertLess(result["mask_10"]["total"], result["default"]["total"])
        self.assertGreater(result["mask_10"]["total"], 0)
        a = result["default"]["contributions"][:, 2].sum()
        b = result["mask_10"]["contributions"][:, 2].sum()
        self.assertEqual(a, b)
        self.assertGreater(a, 0)

    def test_q_gain_invariance_does_not_imply_whole_loss_gain_invariance(self):
        a = self.obj.target_features["power"]
        # Gain invariance holds where both sums exceed the original 1e-20
        # numerical-zero cutoff. Tiny decaying tails can cross that cutoff.
        significant = a.sum(axis=1) > 1e-16
        np.testing.assert_allclose(shape_distribution(a)[significant],
                                   shape_distribution(a*1e-4)[significant], atol=1e-14)
        tiny = np.full((1, 5), 1e-22)
        np.testing.assert_array_equal(shape_distribution(tiny), np.zeros_like(tiny))
        self.assertGreater(self.obj.score(self.target*.01)["total"], 0)
        np.testing.assert_array_equal(shape_distribution(np.zeros_like(a)), np.zeros_like(a))

    def test_masks_are_target_only_and_quiet_denominator_is_explicit(self):
        mask = target_mask(self.obj, .05).copy()
        features = self.obj._features(self.target*100, len(self.obj.ends))
        activity_metrics(self.obj, features["envelope"], .05)
        np.testing.assert_array_equal(mask, target_mask(self.obj, .05))
        quiet = BalancedObjective(self.target*0, 50000, scale=1)
        result = activity_metrics(quiet, np.zeros(len(quiet.ends)), .01)
        self.assertIsNone(result["missing_fraction"])
        self.assertEqual(result["false_voice_fraction"], 0)

    def test_timing_shift_and_boundary_partition(self):
        labels, changes = temporal_regions(self.obj)
        self.assertGreater(len(changes), 0)
        self.assertEqual(set(labels), {"body", "weak", "boundary"})
        for ratio in self.config["ratios"]:
            target = target_mask(self.obj, ratio)
            features = self.obj._features(np.r_[np.zeros(1000), self.target[:-1000]], len(self.obj.ends))
            item = activity_metrics(self.obj, features["envelope"], ratio)
            self.assertGreater(item["disagreement_fraction"], 0)
            self.assertEqual(len(target), len(labels))

    def test_rank_changes_are_within_definition_and_ties_are_averaged(self):
        ranks, changes = ranking_changes(dict(oracle=dict(default=0, alternate=0),
                                              a=dict(default=1, alternate=3), b=dict(default=2, alternate=2)))
        self.assertEqual(changes, [dict(variant="alternate", first="a", second="b")])
        ranks, _ = ranking_changes(dict(a=dict(default=1), b=dict(default=1)))
        self.assertEqual([r["rank"] for r in ranks], [1.5, 1.5])

    def test_control_statistics_exclude_structurally_fixed_b_force(self):
        controls = np.tile((LOW+HIGH)/2, (35, 1))
        controls[:, 4] = 1e7
        actions = np.zeros((35, 4))
        response = waveform_response(self.obj, self.target)
        stats, correlations = control_diagnostics(controls, actions, [0, 1, 2, 3], response)
        self.assertTrue(stats[4]["structural_fixed"])
        self.assertEqual(stats[4]["upper_boundary_fraction"], 1)
        self.assertIsNone(stats[4]["action_span"])
        self.assertTrue(all(r["control"] != "f_e_r" for r in correlations))
        self.assertTrue(all(r["correlation"] is None for r in correlations))
        self.assertIsNone(safe_correlation(np.zeros(35), np.arange(35)))
        self.assertAlmostEqual(safe_correlation(np.arange(35), np.arange(35)), 1)

    def test_private_paths_cannot_overlap_or_escape(self):
        source = ROOT/"private_runs/issue11"
        for output in (source, source/"new", ROOT/"private_runs", ROOT/"results/forbidden"):
            with self.assertRaises(ValueError):
                guarded_paths(source, output)
        with tempfile.TemporaryDirectory(dir=ROOT/"private_runs") as directory:
            directory = Path(directory)
            link = directory/"escape"
            link.symlink_to("/tmp", target_is_directory=True)
            with self.assertRaises(ValueError):
                guarded_paths(source, link/"output")
            with self.assertRaises(ValueError):
                inside(directory, "../escape")

    def test_manifest_detects_content_and_file_set_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"data").write_text("synthetic")
            first = manifest(path)
            (path/"data").write_text("changed")
            self.assertNotEqual(first, manifest(path))
            (path/"extra").write_text("new")
            self.assertNotEqual(set(first), set(manifest(path)))

    def test_invalid_offline_variants_are_rejected(self):
        with self.assertRaises(ValueError):
            variant_parts(self.obj, self.target, dict(bad=dict(ratio=.001, energy_terms=[])))
        with self.assertRaises(ValueError):
            variant_parts(self.obj, self.target, dict(bad=dict(ratio=.01, energy_terms=["missing"])))
        with self.assertRaises(ValueError):
            target_mask(self.obj, 0)


if __name__ == "__main__":
    unittest.main()
