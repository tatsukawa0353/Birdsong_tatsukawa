"""Native syrinx control with a causal audio filter shared by targets and rollouts."""
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy import signal

from .acoustics import distance
from .model import BirdModel, DT, DEFAULT_CONTROLS, LOW, HIGH, effective_controls


class AudioDecimator:
    """Continuous filter state across actions; 16-pole 15 kHz lowpass -> 50 kHz.

    This is an output filter only. Numerical integration always remains at 10 MHz.
    Both target and generated audio have the same causal filter and group delay.
    """
    sample_rate = 50000
    factor = 200

    def __init__(self):
        self.sos = signal.butter(16, 15000, fs=1/DT, output="sos")
        self.state = np.zeros((len(self.sos), 2))

    def process(self, x):
        if len(x) % self.factor:
            raise ValueError("Decimator requires an integral number of output samples")
        filtered, self.state = signal.sosfilt(self.sos, x, zi=self.state)
        return filtered[self.factor-1::self.factor]


@dataclass(frozen=True)
class ModelConfig:
    duration: float = .35
    control_s: float = .01
    sample_rate: int = 50000
    condition: str = "A"
    target: str = "constant"
    smooth_weight: float = 0.0
    control_indices: tuple = (0, 2)
    output_gain: float = 1.0
    fe_mapping: str = "linear"
    fe_power: float = 8.0
    diagnostics: bool = False

    def __post_init__(self):
        if self.condition not in "ABC" or len(self.condition) != 1:
            raise ValueError("Condition must be A, B, C")
        if self.target not in ("constant", "glide") or self.sample_rate != 50000:
            raise ValueError("Unknown target or unsupported output sample rate")
        if min(self.duration, self.control_s) <= 0 or self.smooth_weight < 0:
            raise ValueError("Invalid model timing or penalty")
        if not (np.isfinite(self.output_gain) and self.output_gain > 0):
            raise ValueError("output_gain must be finite and positive")
        if (not self.control_indices or len(set(self.control_indices)) != len(self.control_indices)
                or any(i not in range(5) for i in self.control_indices)):
            raise ValueError("Select unique control indices between 0 and 4")
        if self.condition == "B" and 4 in self.control_indices:
            raise ValueError("Condition B fixes f0_r=1e7")
        if self.fe_mapping not in ("linear", "power") or not np.isfinite(self.fe_power) or self.fe_power <= 0:
            raise ValueError("Invalid external-force mapping")
        for n in (self.duration/self.control_s, self.control_s/DT,
                  self.control_s*self.sample_rate):
            if n < 1 or abs(n-round(n)) > 1e-7:
                raise ValueError("Timing must contain integral integration and output blocks")


class ModelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=ModelConfig(), *, external_target=None):
        self.config = config
        self.horizon = round(config.duration/config.control_s)
        self.block = round(config.control_s*config.sample_rate)
        # Phase 1 uses two controls; native API exposes all five independently.
        self.indices = list(config.control_indices)
        dimensions = len(self.indices)
        self.action_space = spaces.Box(-1, 1, shape=(dimensions,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(5+dimensions,), dtype=np.float32)
        self.model = BirdModel(config.condition)
        if external_target is None:
            self.target_controls = self._target_controls()
            self.target_audio = self._generate(self.target_controls)
        else:
            x = np.asarray(external_target, dtype=float)
            if x.shape != (self.horizon*self.block,) or not np.isfinite(x).all():
                raise ValueError("External target must be finite and time-aligned")
            self.target_controls = None
            self.target_audio = x.copy()
        self.loss_scale = max(float(np.max(np.abs(self.target_audio))), 1e-8)
        self.index = self.horizon
        self.done = True

    def _target_controls(self):
        controls = np.tile(DEFAULT_CONTROLS, (self.horizon, 1))
        controls[:, 0] = 4.1e7 if self.config.target == "constant" else np.linspace(3e7, 4.6e7, self.horizon)
        controls[:, 2] = 3.4e6
        return controls

    def _generate(self, controls):
        self.model.reset()
        decimator, blocks = AudioDecimator(), []
        for c in controls:
            self.model.set_controls(c)
            blocks.append(decimator.process(self.model.advance(self.config.control_s))/self.config.output_gain)
        return np.concatenate(blocks)

    def decode_requested(self, action):
        action = np.asarray(action)
        if action.shape != (len(self.indices),) or not np.isfinite(action).all():
            raise ValueError("Action must have the configured number of finite values")
        action = np.clip(action, -1, 1)
        controls = DEFAULT_CONTROLS.copy()
        controls[self.indices] = LOW[self.indices]+(action+1)/2*(HIGH[self.indices]-LOW[self.indices])
        if self.config.fe_mapping == "power":
            for j, index in enumerate(self.indices):
                if index in (3, 4):
                    fraction = ((float(action[j])+1)/2)**self.config.fe_power
                    controls[index] = LOW[index]+fraction*(HIGH[index]-LOW[index])
        return controls

    def decode(self, action):
        return effective_controls(self.decode_requested(action), self.config.condition)

    def encode(self, controls):
        fractions = ((np.asarray(controls)[self.indices]-LOW[self.indices]) /
                     (HIGH[self.indices]-LOW[self.indices]))
        if self.config.fe_mapping == "power":
            for j, index in enumerate(self.indices):
                if index in (3, 4):
                    fractions[j] = max(0, fractions[j])**(1/self.config.fe_power)
        return (2*fractions-1).astype(np.float32)

    def _obs(self):
        state = self.model.state[2:]/np.array([.1, 200, .5, 200])
        return np.clip(np.array([2*self.index/self.horizon-1, *self.previous, *state], dtype=np.float32), -1, 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.model.reset()
        self.decimator = AudioDecimator()
        self.index = 0
        self.done = False
        self.previous = np.zeros(len(self.indices), dtype=np.float32)
        self.audio, self.actions, self.rewards, self.states = [], [], [], []
        self.requested_control_history, self.applied_control_history, self.motion_history = [], [], []
        return self._obs(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before step and after episode completion")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(self.indices),) or not np.isfinite(action).all():
            raise ValueError("Action must have the configured number of finite values")
        action = np.clip(action, -1, 1)
        requested = self.decode_requested(action)
        self.model.set_controls(requested)
        self.requested_control_history.append(requested.copy())
        self.applied_control_history.append(self.model.controls.copy())
        try:
            if self.config.diagnostics:
                raw, motion = self.model.advance(self.config.control_s, return_moments=True)
                self.motion_history.append(motion)
            else:
                raw = self.model.advance(self.config.control_s)
            wave = self.decimator.process(raw)/self.config.output_gain
        except FloatingPointError:
            self.index += 1
            self.done = True
            self.audio.append(np.zeros(self.block))
            self.actions.append(action.copy())
            self.rewards.append(-1e6)
            # Never expose nonfinite native observations to a learner.
            return np.zeros(self.observation_space.shape, dtype=np.float32), -1e6, True, False, {"numerical_failure": True}
        self.audio.append(wave)
        self.actions.append(action.copy())
        self.states.append(self.model.state.copy())
        self.index += 1
        context = min(self.index, 3)
        generated = np.concatenate(self.audio[-context:])
        target = self.target_audio[(self.index-context)*self.block:self.index*self.block]
        metrics = distance(target, generated, self.config.sample_rate, scale=self.loss_scale)
        penalty = self.config.smooth_weight*float(np.mean((action-self.previous)**2))
        reward = -metrics["total"]-penalty
        self.rewards.append(reward)
        self.previous = action.copy()
        done = self.index == self.horizon
        self.done = done
        info = dict(**metrics, smooth_penalty=penalty)
        if self.config.diagnostics:
            info["motion_std"] = motion["std"].tolist()
        if done:
            info["episode_metrics"] = distance(self.target_audio, np.concatenate(self.audio),
                                               self.config.sample_rate, scale=self.loss_scale)
        return self._obs(), float(reward), done, False, info

    def close(self):
        self.model.close()
