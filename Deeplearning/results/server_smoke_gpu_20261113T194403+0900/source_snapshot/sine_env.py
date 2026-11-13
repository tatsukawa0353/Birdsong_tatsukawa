"""Phase 0 goal-conditioned oscillator; audio rate differs from control rate."""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .acoustics import distance


@dataclass(frozen=True)
class SineConfig:
    duration: float = 0.35
    control_s: float = 0.01
    sample_rate: int = 48000
    frequency_low: float = 500.0
    frequency_high: float = 3000.0
    target: str = "glide"
    randomize: bool = False
    smooth_weight: float = 0.0

    def __post_init__(self):
        if self.target not in ("constant", "glide"):
            raise ValueError("Unknown target")
        if not (self.duration > 0 and self.control_s > 0 and self.sample_rate > 0 and
                0 < self.frequency_low < self.frequency_high < self.sample_rate/2 and
                self.smooth_weight >= 0):
            raise ValueError("Invalid oscillator configuration")
        for count in (self.duration/self.control_s, self.sample_rate*self.control_s):
            if abs(count-round(count)) > 1e-8 or count < 1:
                raise ValueError("Duration and control interval must contain integral blocks/samples")


def oscillate(frequency, amplitude, phase, count, sample_rate):
    wave = amplitude*np.sin(phase + 2*np.pi*frequency*np.arange(count)/sample_rate)
    return wave, float((phase+2*np.pi*frequency*count/sample_rate) % (2*np.pi))


class SineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=SineConfig()):
        self.config = config
        self.horizon = round(config.duration/config.control_s)
        self.block = round(config.sample_rate*config.control_s)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
        self.ready = False
        self.loss_scale = 1.0

    def encode(self, frequency, amplitude):
        c = self.config
        return np.array([2*(frequency-c.frequency_low)/(c.frequency_high-c.frequency_low)-1,
                         2*amplitude-1], dtype=np.float32)

    def decode(self, action):
        c = self.config
        return (c.frequency_low+(float(action[0])+1)/2*(c.frequency_high-c.frequency_low),
                (float(action[1])+1)/2)

    def _obs(self):
        i = min(self.index, self.horizon-1)
        return np.array([2*self.index/self.horizon-1, *self.goal_actions[i], *self.previous], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        c = self.config
        # Evaluation uses a fixed trajectory; training has independently seeded goals.
        start = float(self.np_random.uniform(800, 1500)) if c.randomize else 1000.0
        end = start if c.target == "constant" else (float(self.np_random.uniform(1700, 2400)) if c.randomize else 2000.0)
        amp = float(self.np_random.uniform(0.35, 0.7)) if c.randomize else 0.6
        fraction = np.arange(self.horizon)/max(self.horizon-1, 1)
        self.frequencies = start+(end-start)*fraction
        self.amplitudes = np.full(self.horizon, amp) if c.target == "constant" else amp*(0.8+0.2*np.cos(2*np.pi*fraction))
        self.goal_actions = np.array([self.encode(f, a) for f, a in zip(self.frequencies, self.amplitudes)])
        phase, blocks = 0.0, []
        for f, a in zip(self.frequencies, self.amplitudes):
            w, phase = oscillate(f, a, phase, self.block, c.sample_rate)
            blocks.append(w)
        self.target_audio = np.concatenate(blocks)
        self.index, self.phase = 0, 0.0
        self.previous = np.zeros(2, dtype=np.float32)
        self.audio, self.actions, self.rewards = [], [], []
        self.ready = True
        return self._obs(), {}

    def step(self, action):
        if not self.ready or self.index >= self.horizon:
            raise RuntimeError("Call reset before step and after episode completion")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must contain two finite values")
        action = np.clip(action, -1, 1)
        frequency, amplitude = self.decode(action)
        w, self.phase = oscillate(frequency, amplitude, self.phase, self.block, self.config.sample_rate)
        a = self.index*self.block
        errors = distance(self.target_audio[a:a+self.block], w, self.config.sample_rate, scale=1.0)
        smooth = self.config.smooth_weight*float(np.mean((action-self.previous)**2))
        reward = -errors["total"]-smooth
        self.audio.append(w)
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.previous = action.copy()
        self.index += 1
        done = self.index == self.horizon
        info = dict(**errors, smooth_penalty=smooth)
        if done:
            info["episode_metrics"] = distance(self.target_audio, np.concatenate(self.audio), self.config.sample_rate, scale=1.0)
        return self._obs(), float(reward), done, False, info
