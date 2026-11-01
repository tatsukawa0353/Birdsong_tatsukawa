import tempfile
import hashlib
from pathlib import Path
import unittest

import numpy as np
from scipy.io import wavfile

from Deeplearning.acoustics import distance
from Deeplearning.objectives import BalancedObjective

from Deeplearning.phase1_validation_audit import (
    Auditor, close_scalar, expected_schedule, parameter_nrmse,
)


class AuditHelperTests(unittest.TestCase):
    def test_finite_tolerance_rejects_nonfinite_or_material_difference(self):
        self.assertTrue(close_scalar(1+1e-11, 1))
        self.assertTrue(close_scalar(0, 0))
        for actual in (None, "invalid", np.nan, np.inf, 1.01):
            self.assertFalse(close_scalar(actual, 1))
        self.assertFalse(close_scalar(np.nan, np.nan))

    def test_checkpoints_have_one_initial_and_one_final(self):
        self.assertEqual(expected_schedule(256, 32), list(range(0, 257, 32)))
        self.assertEqual(expected_schedule(35, 10), [0, 10, 20, 30, 35])
        self.assertEqual(expected_schedule(2, 5), [0, 2])
        for budget, interval in ((0, 2), (2, 0), (2.5, 1)):
            with self.assertRaises(ValueError):
                expected_schedule(budget, interval)

    def test_parameter_error_uses_selected_ranges_and_all_physical_blocks(self):
        target = np.asarray([4, 9, 8, 7, 6.])
        low = np.zeros(5)
        high = np.asarray([10, 100, 20, 10, 10.])
        controls = np.tile(target, (35, 1))
        controls[:, 0] += 1
        controls[:, 2] += 4
        self.assertAlmostEqual(parameter_nrmse(controls, target, low, high, [0]), .1)
        self.assertAlmostEqual(parameter_nrmse(controls, target, low, high, [0, 2]), np.sqrt((.1**2+.2**2)/2))
        with self.assertRaises(ValueError):
            parameter_nrmse([], target, low, high, [0])
        with self.assertRaises(ValueError):
            parameter_nrmse(controls, target, high, low, [0])

    def test_missing_inputs_produce_structured_failure_without_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Auditor(Path(directory)).run()
        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"][0]["code"], "json_read")
        self.assertEqual(result["counts"]["rollouts_checked"], 0)

    def test_independent_rollout_audit_detects_gain_changes_and_wrong_metrics(self):
        rate = 50000
        target = 2*np.sin(2*np.pi*700*np.arange(17500)/rate)
        audio = target*.8
        controls = np.full((35, 5), 5.)
        actions = np.zeros((1, 1))
        manifest = dict(sample_rate=rate, physical_low=[0]*5,
                        physical_high=[10]*5, target_controls=[5]*5)
        objective = BalancedObjective(target, rate, scale=float(abs(target).max()))
        acoustic, balanced = distance(target, audio, rate), objective.score(audio)
        metrics = dict(acoustic_error=acoustic["total"], balanced_error=balanced["total"],
                       parameter_nrmse=0., acoustic_components=acoustic,
                       balanced_components=balanced, failed=False,
                       negative_return=acoustic["total"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.savez_compressed(output / "rollout.npz", target=target, audio=audio,
                                controls=controls, actions=actions, rate=rate)
            wavfile.write(output / "target_raw.wav", rate, target.astype(np.float32))
            wavfile.write(output / "generated_raw.wav", rate, audio.astype(np.float32))
            valid = Auditor(output)
            valid.rollout(output, metrics, "A", target, objective, manifest)
            self.assertEqual(valid.errors, [])
            wavfile.write(output / "generated_raw.wav", rate, np.clip(audio, -1, 1).astype(np.float32))
            metrics["acoustic_error"] += .1
            invalid = Auditor(output)
            invalid.rollout(output, metrics, "A", target, objective, manifest)
            codes = {error["code"] for error in invalid.errors}
            self.assertIn("wav_gain_or_clipping", codes)
            self.assertIn("numeric_mismatch", codes)

    def test_recorded_input_hashes_detect_changes_and_reject_outside_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "source.json"
            source.write_text("original")
            hashes = {"source.json": hashlib.sha256(source.read_bytes()).hexdigest()}
            auditor = Auditor(output)
            auditor.input_hashes(hashes, output, ["source.json"])
            self.assertEqual(auditor.errors, [])
            source.write_text("changed")
            auditor.input_hashes(hashes, output)
            auditor.input_hashes({"../outside": "ignored"}, output)
            self.assertEqual({e["code"] for e in auditor.errors},
                             {"input_hash_changed", "input_hash_scope"})
            self.assertEqual(auditor.counts["input_hashes_checked"], 2)


if __name__ == "__main__":
    unittest.main()
