"""Case A, one decision held for 0.20 s; independent of legacy 0.35 s tasks.

Native state, delay and causal filter reset together; filtering remains continuous
over all twenty 10 ms blocks. Loss and WAVs share a target-derived gain.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .acoustics import distance
from .model import BirdModel, DT, LOW, HIGH
from .model_env import AudioDecimator

RATE, DURATION, CONTROL_S, BLOCKS = 50000, .20, .01, 20
INDICES = {'constant_1d': (0,), 'constant_2d': (0, 2), 'constant_3d': (0, 2, 3)}


def render(controls, blocks=BLOCKS):
    model = BirdModel('A')
    try:
        model.reset()
        model.set_controls(controls)
        decimator = AudioDecimator()
        return np.concatenate([decimator.process(model.advance(CONTROL_S)) for _ in range(blocks)])
    finally:
        model.close()


class Constant020Env(gym.Env):
    metadata = {'render_modes': []}
    horizon = 1
    physical_horizon = BLOCKS

    def __init__(self, indices, target, controls):
        self.indices = list(indices)
        if tuple(indices) not in INDICES.values():
            raise ValueError('Expected epsilon_l; epsilon_l/ps; or epsilon_l/ps/f0_l')
        self.target_audio = np.asarray(target, dtype=np.float64).copy()
        self.fixed_controls = np.asarray(controls, dtype=np.float64).copy()
        if self.target_audio.shape != (10000,) or not np.isfinite(self.target_audio).all():
            raise ValueError('Target must contain exactly 10000 finite samples at 50 kHz (0.20 s)')
        if self.fixed_controls.shape != (5,) or not np.isfinite(self.fixed_controls).all() or np.any(self.fixed_controls<LOW) or np.any(self.fixed_controls>HIGH):
            raise ValueError('Target controls must be five finite in-range physical parameters')
        self.target_audio.setflags(write=False)
        self.fixed_controls.setflags(write=False)
        self.loss_scale = max(float(np.max(np.abs(self.target_audio))), 1e-8)
        self.action_space = spaces.Box(-1, 1, (len(indices),), np.float32)
        # All episodes start with the same physical state: a static bandit.
        self.observation_space = spaces.Box(-1, 1, (5+len(indices),), np.float32)
        self.model = BirdModel('A')
        self.decision_steps = self.completed_episodes = self.failures = 0
        self.native_blocks = self.native_blocks_completed = self.silent_episodes = 0
        self.integrated_s = self.integrated_completed_s = 0.
        self.done = True

    def encode(self, controls, dtype=np.float64):
        values = np.asarray(controls, dtype=np.float64)
        return (2*(values[self.indices]-LOW[self.indices])/(HIGH[self.indices]-LOW[self.indices])-1).astype(dtype)

    def decode(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(self.indices),) or not np.isfinite(action).all():
            raise ValueError('Action must have the configured shape and finite values')
        values = self.fixed_controls.copy()
        values[self.indices] = LOW[self.indices]+(np.clip(action,-1,1)+1)/2*(HIGH[self.indices]-LOW[self.indices])
        return values

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.model.reset()
        self.decimator = AudioDecimator()
        self.audio, self.applied_control_history, self.actions, self.rewards = [], [], [], []
        self.done = False
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = -1
        return obs, {}

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before the next candidate')
        controls = self.decode(action)
        failed = False
        self.model.set_controls(controls)
        for _ in range(BLOCKS):
            self.native_blocks += 1
            self.integrated_s += CONTROL_S
            try:
                wave = self.decimator.process(self.model.advance(CONTROL_S))
                if not np.isfinite(wave).all():
                    raise FloatingPointError('Nonfinite filtered output')
            except FloatingPointError:
                failed = True
                self.failures += 1
                break
            self.native_blocks_completed += 1
            self.integrated_completed_s += CONTROL_S
            self.audio.append(wave)
        # A failed candidate remains in the budget; padded audio is diagnostic
        # only. Never reward a shorter, apparently easier prefix.
        if failed:
            self.audio.extend([np.zeros(500)]*(BLOCKS-len(self.audio)))
        audio = np.concatenate(self.audio)
        components = distance(self.target_audio, audio, RATE, scale=self.loss_scale)
        error = 1e6 if failed else components['total']
        silent = components['generated_silent']
        self.silent_episodes += int(silent)
        self.actions = [np.asarray(action, dtype=np.float64).clip(-1,1).copy()]
        self.applied_control_history = np.tile(controls, (BLOCKS,1))
        self.rewards = [-float(error)]
        self.decision_steps += 1
        self.completed_episodes += 1
        self.done = True
        info = dict(acoustic_error=float(error), acoustic_components=components, failed=failed,
                    numerical_failure=failed, generated_silent=silent, negative_return=float(error),
                    applied_action=self.actions[0].tolist(), applied_controls=controls.tolist())
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = 1
        return obs, -float(error), True, False, info

    def close(self):
        self.model.close()


def episode(env, *, action=None, agent=None):
    obs, _ = env.reset()
    if agent is not None:
        action = agent.predict(obs, deterministic=True)[0]
    _, reward, done, truncated, metrics = env.step(action)
    assert done and not truncated and -reward == metrics['acoustic_error']
    return metrics, dict(audio=np.concatenate(env.audio).copy(),
                         physical_controls=np.asarray(env.applied_control_history).copy(),
                         actions=np.asarray(env.actions).copy())


def account(env, wall_s):
    return dict(episodes=env.completed_episodes, decision_steps=env.decision_steps,
                native_blocks_attempted=env.native_blocks, native_blocks_completed=env.native_blocks_completed,
                native_steps_attempted=env.native_blocks*round(CONTROL_S/DT),
                native_steps_completed=env.native_blocks_completed*round(CONTROL_S/DT),
                integrated_s_attempted=env.native_blocks*CONTROL_S,
                integrated_s_completed=env.native_blocks_completed*CONTROL_S,
                generated_audio_s=env.completed_episodes*DURATION, failures=env.failures,
                silent_episodes=env.silent_episodes, wall_s=float(wall_s))
