import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Deeplearning.experiments import rollout
from Deeplearning.issue11 import activity_diagnostics, extension_cases, prepare
from Deeplearning.model import ROOT
from Deeplearning.model_env import ModelConfig, ModelEnv
from Deeplearning.phase2 import CONTROL_INDICES, model_configuration, private_output, run


class Issue11Tests(unittest.TestCase):
    def setUp(self):
        self.rate = 50000
        self.t = np.arange(17500)/self.rate
        self.target = .6*np.sin(2*np.pi*1500*self.t)

    def test_json_switch_keeps_controls_gain_and_gamma(self):
        a = json.loads((ROOT/"configs/phase2_td3_baseline.json").read_text())
        b = json.loads((ROOT/"configs/phase2_td3_balanced_aligned.json").read_text())
        for key in a:
            if key not in ("loss_kind", "reward_timing"):
                self.assertEqual(a[key], b[key])
        for case in "ABC":
            ca, cb = [model_configuration(c, case, gain=123.0) for c in (a, b)]
            self.assertEqual(ca.control_indices, CONTROL_INDICES[case])
            self.assertEqual(ca.control_indices, cb.control_indices)
            self.assertEqual(ca.output_gain, cb.output_gain)
            self.assertEqual(ca.fe_mapping, "power")
            self.assertEqual(ca.fe_power, 8)
            self.assertEqual(ca.reward_discount, cb.reward_discount)
            self.assertEqual((ca.loss_kind, ca.reward_timing), ("baseline", "context"))
            self.assertEqual((cb.loss_kind, cb.reward_timing), ("balanced", "aligned"))

    def test_old_phase2_defaults_keep_waveforms_and_rewards(self):
        # Synthetic target only: no local recording is used by any test.
        for case in "ABC":
            old = ModelEnv(ModelConfig(condition=case, control_indices=CONTROL_INDICES[case],
                                       output_gain=1e6, diagnostics=True), external_target=self.target)
            new = ModelEnv(model_configuration({}, case, gain=1e6), external_target=self.target)
            try:
                ra, wa, _ = rollout(old, lambda obs, e: np.zeros(len(e.indices)))
                rb, wb, _ = rollout(new, lambda obs, e: np.zeros(len(e.indices)))
                np.testing.assert_array_equal(wa, wb)
                np.testing.assert_array_equal(old.rewards, new.rewards)
                self.assertEqual(ra["metrics"], rb["metrics"])
                self.assertIn("objective_error", rb)
            finally:
                old.close()
                new.close()

    def test_both_profiles_share_scale_and_actual_b_right_force(self):
        for case in "ABC":
            waves, metrics, scales = [], [], []
            for loss, timing in (("baseline", "context"), ("balanced", "aligned")):
                cfg = model_configuration(dict(force_variant="power8", loss_kind=loss,
                                               reward_timing=timing, gamma=.99), case, gain=1e6)
                env = ModelEnv(cfg, external_target=self.target)
                try:
                    result, audio, _ = rollout(env, lambda obs, e: np.zeros(len(e.indices)))
                    waves.append(audio)
                    metrics.append((result["error"], result["objective_error"]))
                    scales.append(env.loss_scale)
                    if timing == "aligned":
                        self.assertAlmostEqual(result["discounted_reward_cost"], result["objective_error"], places=12)
                    if case == "B":
                        self.assertTrue(np.all(np.array(env.applied_control_history)[:, 4] == 1e7))
                finally:
                    env.close()
            np.testing.assert_array_equal(*waves)
            self.assertEqual(*metrics)
            self.assertEqual(*scales)

    def test_public_and_symlink_outputs_rejected_before_recording_read(self):
        with patch("Deeplearning.phase2.load_local_wav") as loader:
            with self.assertRaises(ValueError):
                run({}, Path("does-not-exist.wav"), ROOT/"results/forbidden", local_permission_confirmed=True)
            loader.assert_not_called()
        with tempfile.TemporaryDirectory(dir=ROOT/"private_runs") as directory:
            link = Path(directory)/"escape"
            link.symlink_to("/tmp", target_is_directory=True)
            with self.assertRaises(ValueError):
                private_output(link/"forbidden")

    def test_permission_checked_before_reading(self):
        with patch("Deeplearning.issue11.load_local_wav") as loader:
            with self.assertRaises(ValueError):
                prepare({}, "missing.wav", ROOT/"private_runs/unused", permission_confirmed=False)
            loader.assert_not_called()

    def test_target_activity_distinguishes_gaps_missing_and_noise(self):
        gated = self.target*(self.t >= .1)*(self.t < .25)
        correct, _ = activity_diagnostics(gated, gated, scale=1)
        self.assertEqual(correct["activity_disagreement_fraction"], 0)
        self.assertEqual(correct["active_underproduction"], 0)
        self.assertGreater(correct["quiet_duration_s"], .1)
        absent, _ = activity_diagnostics(gated, gated*0, scale=1)
        self.assertEqual(absent["missing_active_fraction"], 1)
        always, _ = activity_diagnostics(gated, self.target, scale=1)
        self.assertGreater(always["false_voice_quiet_fraction"], .9)
        noise = np.random.default_rng(3).normal(size=len(gated))*.1
        noisy, _ = activity_diagnostics(noise, noise, scale=1)
        self.assertGreater(noisy["target_active_fraction"], .9)
        self.assertEqual(noisy["activity_disagreement_fraction"], 0)
        quiet, _ = activity_diagnostics(gated*0, gated*0, scale=1)
        self.assertIsNone(quiet["missing_active_fraction"])
        self.assertEqual(quiet["false_voice_quiet_fraction"], 0)

    def test_extension_gate_excludes_whole_case_without_score_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = json.loads((ROOT/"configs/issue11.json").read_text())
            (output/"protocol.json").write_text(json.dumps(dict(experiment=config)))
            rows = []
            for case in "ABC":
                for profile in ("baseline", "balanced_aligned"):
                    for seed in range(3):
                        relative = f"{case}_{profile}_{seed}"
                        (output/relative).mkdir()
                        failed = case == "B" and profile == "baseline" and seed == 1
                        (output/relative/"curve.json").write_text(json.dumps([dict(failed=failed)]))
                        rows.append(dict(condition=case, relative_path=relative,
                                         result=dict(training_numerical_failures=0, failed=False, error=1e6)))
            self.assertEqual(extension_cases(output, "TD3", rows), ["A", "C"])


if __name__ == "__main__":
    unittest.main()
