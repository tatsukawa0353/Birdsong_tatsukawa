import tempfile
from pathlib import Path
import unittest

import numpy as np

from Deeplearning.experiments import train_run
from Deeplearning.model import BirdModel, DEFAULT_CONTROLS, effective_controls, build
from Deeplearning.model_env import ModelConfig, ModelEnv
from Deeplearning.phase2 import CONTROL_INDICES
from Deeplearning.issue9 import FIXED_INDICES
from Deeplearning.plotting import CONTROL_LABELS
from Deeplearning.voice_diagnostics import pooled_motion, summarize, band_audio, rms


class Issue9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()

    def test_case_b_applied_controls_match_saved_history(self):
        for case in "ABC":
            env = ModelEnv(ModelConfig(condition=case, control_indices=CONTROL_INDICES[case], diagnostics=True))
            try:
                env.reset(seed=0)
                action = np.zeros(len(env.indices), dtype=np.float32)
                requested, applied = env.decode_requested(action), env.decode(action)
                env.step(action)
                np.testing.assert_array_equal(env.applied_control_history[0], env.model.controls)
                np.testing.assert_array_equal(env.applied_control_history[0], applied)
                np.testing.assert_array_equal(env.requested_control_history[0], requested)
                if case == "B":
                    self.assertEqual(requested[4], 4.9e4)
                    self.assertEqual(applied[4], 1e7)
                else:
                    np.testing.assert_array_equal(requested, applied)
                self.assertEqual(len(env.motion_history), 1)
            finally:
                env.close()

    def test_fixed_and_variable_degrees_and_force_mapping(self):
        for case in "ABC":
            fixed = ModelEnv(ModelConfig(condition=case, control_indices=FIXED_INDICES[case]))
            power = ModelEnv(ModelConfig(condition=case, control_indices=CONTROL_INDICES[case], fe_mapping="power"))
            try:
                expected = 2 if case == "A" else 3
                self.assertEqual(fixed.action_space.shape, (expected,))
                for value in (-1.0, -.4, 0.0, .5, 1.0):
                    action = np.full(len(power.indices), value)
                    physical = power.decode(action)
                    np.testing.assert_allclose(power.encode(physical), action, atol=1e-6)
                    fixed_controls = fixed.decode(np.full(expected, value))
                    self.assertEqual(fixed_controls[3], 4.9e4)
                    self.assertEqual(fixed_controls[4], 1e7 if case == "B" else 4.9e4)
                neutral = power.decode(np.zeros(len(power.indices)))
                self.assertAlmostEqual(neutral[3], 87871.09375)
                self.assertEqual(power.decode(np.ones(len(power.indices)))[3], 1e7)
            finally:
                fixed.close()
                power.close()

    def test_full_rate_moments_match_dense_samples(self):
        with BirdModel("A") as model:
            trace = []
            pressure = []
            for _ in range(100):
                pressure.append(model.advance(1e-7)[0])
                trace.append(model.state[2:])
            model.reset()
            actual, moments = model.advance(1e-5, return_moments=True)
            np.testing.assert_array_equal(actual, pressure)
            np.testing.assert_allclose(moments["mean"], np.mean(trace, axis=0), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(moments["std"], np.std(trace, axis=0), rtol=1e-12, atol=1e-14)

    def test_equilibrium_offset_is_not_vibration(self):
        means = np.tile([-.25, 0, -.25, 0], (35, 1))
        stds = np.zeros_like(means)
        avg, std = pooled_motion(means, stds)
        np.testing.assert_allclose(avg, [-.25, 0, -.25, 0])
        np.testing.assert_allclose(std, 0, atol=1e-15)
        audio = np.full(17500, .1)
        report = summarize(audio, reference_band_rms=1, condition="C", motion_mean=means, motion_std=stds)
        self.assertFalse(report["exactly_zero"])
        self.assertFalse(report["contributing_source_motion"])
        self.assertFalse(report["sustained_acoustic"])

    def test_relative_silence_does_not_reject_noise_like_sound(self):
        rate = 50000
        t = np.arange(17500)/rate
        tone = np.sin(2*np.pi*1000*t)
        reference = rms(band_audio(tone)[5000:])
        noise = np.random.default_rng(0).normal(size=len(t))
        for wave in (tone, noise):
            report = summarize(wave, reference_band_rms=reference, condition="C")
            self.assertTrue(report["sustained_acoustic"])
        report = summarize(tone*1e-8, reference_band_rms=reference, condition="A")
        self.assertFalse(report["sustained_acoustic"])
        self.assertFalse(report["exactly_zero"])
        failed = summarize(tone, reference_band_rms=reference, condition="A", numerical_failure=True)
        self.assertFalse(failed["sustained_acoustic"])
        self.assertTrue(failed["numerical_failure"])
        self.assertEqual(failed["acoustic_state"], "numerical_failure")

    def test_all_research_labels_and_b_override_are_explicit(self):
        self.assertEqual(CONTROL_LABELS, (r"$\varepsilon_l$", r"$\varepsilon_r$", r"$p_s$", r"$f_{e,l}$", r"$f_{e,r}$"))
        before = DEFAULT_CONTROLS.copy()
        result = effective_controls(before, "B")
        self.assertEqual(result[4], 1e7)
        np.testing.assert_array_equal(before, DEFAULT_CONTROLS)
        with self.assertRaises(ValueError):
            ModelConfig(condition="B", control_indices=(0, 4))

    def test_training_counts_actual_numerical_failure_episodes(self):
        class FailingEnv(ModelEnv):
            def reset(self, **kwargs):
                obs, info = super().reset(**kwargs)
                self.model.reset(state=(1e300, 1e300, 1e300, 1e300))
                return obs, info

        settings = dict(steps=4, eval_interval=2, threshold=.08, learning_rate=.0003, ppo_rollout=4)
        with tempfile.TemporaryDirectory() as directory:
            result, _ = train_run("SAC", 0, FailingEnv, FailingEnv, settings, Path(directory))
            self.assertEqual(result["training_numerical_failures"], 4)
            self.assertEqual(result["training_episodes"], 4)
            self.assertTrue(result["failed"])
            self.assertEqual(result["error"], 1e6)


if __name__ == "__main__":
    unittest.main()
