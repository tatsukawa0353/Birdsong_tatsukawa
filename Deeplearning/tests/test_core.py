import tempfile
from pathlib import Path
import subprocess
import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from Deeplearning.acoustics import AcousticConfig, distance, spectrum, resample
from Deeplearning.acoustics import load_local_wav
from Deeplearning.model import BirdModel, DEFAULT_CONTROLS, ROOT, build
from Deeplearning.sine_env import SineEnv, SineConfig
from Deeplearning.model_env import AudioDecimator, ModelConfig, ModelEnv
from Deeplearning.evaluation import descriptors


class AcousticTests(unittest.TestCase):
    def setUp(self):
        self.fs = 48000
        t = np.arange(16800)/self.fs
        self.x = 0.6*np.sin(2*np.pi*1000*t)*(t > .04)*(t < .25)

    def test_identity_gain_frequency_timing(self):
        self.assertEqual(distance(self.x, self.x, self.fs)["total"], 0)
        t = np.arange(len(self.x))/self.fs
        variants = [self.x*.5, np.sin(2*np.pi*1600*t)*.6,
                    np.roll(self.x, round(.03*self.fs)), np.zeros_like(self.x)]
        for generated in variants:
            self.assertGreater(distance(self.x, generated, self.fs)["total"], .01)

    def test_silence_noise_invalid_inputs(self):
        zeros = np.zeros_like(self.x)
        self.assertEqual(distance(zeros, zeros, self.fs)["total"], 0)
        noise = np.random.default_rng(0).normal(size=len(self.x))*.1
        self.assertEqual(distance(noise, noise, self.fs)["total"], 0)
        for x in ([np.nan], [], [[0, 1]], [np.inf]):
            with self.assertRaises(ValueError):
                spectrum(x, self.fs)
        with self.assertRaises(ValueError):
            distance(self.x, self.x[:-1], self.fs)
        with self.assertRaises(ValueError):
            spectrum(self.x, 0)

    def test_window_seconds_and_antialiasing(self):
        cfg = AcousticConfig()
        for rate in (24000, 48000, 10000000):
            n = round(cfg.window_s*rate)
            self.assertLessEqual(abs(n/rate-cfg.window_s), .5/rate)
        t = np.arange(48000)/48000
        low = np.sin(2*np.pi*1000*t)
        high = np.sin(2*np.pi*18000*t)
        self.assertGreater(np.std(resample(low, 48000, 24000)), .6)
        self.assertLess(np.std(resample(high, 48000, 24000)), .01)

    def test_wav_duration_selection_and_pcm_scaling(self):
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"synthetic.wav"
            wavfile.write(path, 44100, np.full(15444, 1000, dtype=np.int16))
            audio, metadata = load_local_wav(path, target_rate=50000, return_metadata=True)
            self.assertEqual(len(audio), 17500)
            self.assertEqual(metadata["trimmed_end_samples"], 9)
            self.assertAlmostEqual(float(np.mean(audio[100:-100])), 1000/32768, delta=1e-5)
            wavfile.write(path, 44100, np.zeros(44100, dtype=np.int16))
            with self.assertRaises(ValueError):
                load_local_wav(path)


class SineTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        check_env(SineEnv(), warn=True)

    def test_oracle_beats_random_both_targets(self):
        for target in ("constant", "glide"):
            env = SineEnv(SineConfig(target=target))
            env.reset(seed=1)
            for _ in range(env.horizon):
                _, _, done, _, info = env.step(env.goal_actions[env.index])
            self.assertTrue(done)
            self.assertLess(info["episode_metrics"]["total"], 1e-5)
            oracle = np.concatenate(env.audio)
            env.reset(seed=1)
            for _ in range(env.horizon):
                _, _, _, _, info = env.step(env.action_space.sample())
            self.assertGreater(info["episode_metrics"]["total"], .05)
            self.assertEqual(len(oracle), 16800)
            with self.assertRaises(RuntimeError):
                env.step([0, 0])

    def test_seed_and_action_validation(self):
        a, b = SineEnv(SineConfig(randomize=True)), SineEnv(SineConfig(randomize=True))
        a.reset(seed=42)
        b.reset(seed=42)
        np.testing.assert_array_equal(a.target_audio, b.target_audio)
        for action in ([np.nan, 0], [1], [0, 0, 0]):
            with self.assertRaises(ValueError):
                a.step(action)
        for kwargs in (dict(control_s=.013), dict(sample_rate=1000), dict(duration=-1)):
            with self.assertRaises(ValueError):
                SineConfig(**kwargs)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()

    def test_original_implementation_regression(self):
        # Original CSV has six significant digits. Tolerance follows its quantization.
        repo = ROOT.parent
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            exe, csv = directory/"reference", directory/"reference.csv"
            subprocess.run(["g++", "-O3", "-std=c++11", "-I", str(repo),
                            str(ROOT/"native/reference.cpp"), str(repo/"BirdsongModel_10*10.cpp"),
                            "-o", str(exe)], check=True)
            subprocess.run([str(exe), str(csv), "350000"], check=True)
            reference = np.loadtxt(csv, delimiter=",", skiprows=1)
            with BirdModel("A") as model:
                generated = model.advance(.035)
                np.testing.assert_allclose(generated, reference[1:, 1], rtol=5.1e-6, atol=5.1e-7)
                np.testing.assert_allclose(model.state, reference[-1], rtol=5.1e-6, atol=5.1e-7)
                f = np.fft.rfft(reference[1:, 1])
                g = np.fft.rfft(generated)
                self.assertLess(np.linalg.norm(f-g)/np.linalg.norm(f), 5e-6)

    def test_chunking_reset_and_controls(self):
        with BirdModel("A") as model:
            whole = model.advance(.02)
            model.reset()
            chunks = np.concatenate([model.advance(.01), model.advance(.01)])
            np.testing.assert_array_equal(whole, chunks)
            model.reset()
            measured, rms = model.advance(.02, return_rms=True)
            np.testing.assert_array_equal(whole, measured)
            self.assertTrue(np.isfinite(rms).all())
            self.assertGreater(rms[1], 0)
            control = DEFAULT_CONTROLS.copy()
            control[0] = 4e7
            model.reset(control)
            self.assertGreater(np.max(np.abs(model.advance(.02)-whole)), 1)
            history = np.zeros(model.history_size)
            history[0] = 100
            model.reset(history=history)
            self.assertNotEqual(model.advance(1e-7)[0], whole[0])
            with self.assertRaises(ValueError):
                model.advance(.00000011)
            with self.assertRaises(ValueError):
                model.set_controls([0]*5)
        with self.assertRaises(RuntimeError):
            model.advance(.01)

    def test_conditions_are_distinct_and_b_is_fixed(self):
        signals = []
        for condition in "ABC":
            with BirdModel(condition) as model:
                signals.append(model.advance(.01))
                if condition == "B":
                    self.assertEqual(model.controls[4], 1e7)
        self.assertFalse(np.array_equal(signals[0], signals[1]))
        self.assertFalse(np.array_equal(signals[1], signals[2]))


class ModelEnvironmentTests(unittest.TestCase):
    def test_causal_filter_preserves_chunk_boundaries(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200000)
        whole = AudioDecimator().process(x)
        decimator = AudioDecimator()
        chunks = np.concatenate([decimator.process(x[:100000]), decimator.process(x[100000:])])
        np.testing.assert_array_equal(whole, chunks)

    def test_oracle_and_gymnasium_contract(self):
        env = ModelEnv()
        try:
            check_env(env, warn=True)
            env.reset(seed=0)
            for _ in range(env.horizon):
                _, _, done, _, info = env.step(env.encode(env.target_controls[env.index]))
            self.assertTrue(done)
            self.assertLess(info["episode_metrics"]["total"], 1e-5)
        finally:
            env.close()

    def test_action_dimensions_and_failure_handling(self):
        for condition, indices in (("A", (0, 2, 3)), ("B", (0, 1, 2, 3)), ("C", (0, 1, 2, 3, 4))):
            env = ModelEnv(ModelConfig(condition=condition, control_indices=indices))
            try:
                obs, _ = env.reset(seed=0)
                self.assertTrue(env.observation_space.contains(obs))
                obs, reward, done, _, info = env.step(np.zeros(len(indices)))
                self.assertTrue(np.isfinite(reward))
                self.assertTrue(env.observation_space.contains(obs))
                self.assertFalse(done)
            finally:
                env.close()
        env = ModelEnv()
        try:
            env.reset()
            env.model.reset(state=(1e300, 1e300, 1e300, 1e300))
            obs, reward, done, _, info = env.step(np.zeros(2))
            self.assertTrue(done)
            self.assertTrue(info["numerical_failure"])
            self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(env.index, 1)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(2))
        finally:
            env.close()

    def test_noise_is_described_without_exclusion(self):
        fs = 50000
        t = np.arange(17500)/fs
        tone = descriptors(np.sin(2*np.pi*1000*t), fs)
        noise = descriptors(np.random.default_rng(0).normal(size=len(t)), fs)
        self.assertAlmostEqual(tone["f0_hz"], 1000, delta=25)
        self.assertGreater(noise["se"], tone["se"])
        self.assertIsNone(noise["repository_sci"])
        self.assertEqual(noise["status"], "no_reliable_f0")

    def test_last_block_divergence_cannot_receive_a_normal_score(self):
        from Deeplearning.experiments import rollout
        env = ModelEnv()

        def fail_on_last_block(obs, environment):
            if environment.index == environment.horizon-1:
                environment.model.reset(state=(1e300, 1e300, 1e300, 1e300))
            return np.zeros(2)

        try:
            result, audio, _ = rollout(env, fail_on_last_block)
            self.assertEqual(len(audio), len(env.target_audio))
            self.assertTrue(result["failed"])
            self.assertEqual(result["error"], 1e6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
