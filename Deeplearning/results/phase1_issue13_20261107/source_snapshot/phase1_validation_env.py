"""Synthetic Case A validation without changing the historical environment.

Task A is a one-decision bandit, while B and C make 35 sequential decisions.
All tasks use the same native model, reset, continuous output filter, bounds,
fixed controls and target. Rewards telescope to the negative whole-episode
``acoustics.distance`` at gamma=1; balanced loss is an independent evaluation.
"""
import gymnasium as gym
import numpy as np

from .acoustics import distance
from .model import HIGH, LOW
from .model_env import AudioDecimator, ModelConfig, ModelEnv


TARGET_CONTROLS = (4.1e7, 3.5e7, 3.4e6, 4.9e4, 4.9e4)
DURATION = .35
CONTROL_S = .01
RATE = 50000
PHYSICAL_HORIZON = round(DURATION / CONTROL_S)


class ValidationEnv(ModelEnv):
    """One or two controls, with all nonlearned controls fixed at target values.

    ``integrated_s`` and ``native_blocks`` count attempted native advances,
    including a whole requested block if native integration fails inside it.
    ``integrated_completed_s`` counts only successfully completed blocks.
    Setup target generation is recorded separately by ``setup_simulation_s``.
    All counters survive reset; episode histories do not. Float64 Oracle inputs
    are retained so representation error can be audited separately from Gym's
    float32 policy actions.
    """
    def __init__(self, task="A", target_audio=None):
        if task not in ("A", "B", "C"):
            raise ValueError("Validation task must be A, B or C")
        self.task = task
        indices = (0, 2) if task == "C" else (0,)
        super().__init__(ModelConfig(condition="A", control_indices=indices,
                                     reward_discount=1.0), external_target=target_audio)
        self.physical_horizon = PHYSICAL_HORIZON
        self.horizon = 1 if task == "A" else self.physical_horizon
        self.decision_horizon = self.horizon
        self.target_controls = self._target_controls()
        self.target_controls.setflags(write=False)
        self.target_audio = self.target_audio.copy()
        self.target_audio.setflags(write=False)
        self.fixed_controls = np.asarray(TARGET_CONTROLS, dtype=float)
        self.fixed_controls.setflags(write=False)
        self.setup_simulation_s = DURATION if target_audio is None else 0.0
        self.integrated_s = 0.0
        self.integrated_completed_s = 0.0
        self.native_blocks = 0
        self.native_blocks_completed = 0
        self.decision_steps = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.failures = 0

    def _target_controls(self):
        return np.tile(np.asarray(TARGET_CONTROLS, dtype=float), (PHYSICAL_HORIZON, 1))

    def decode(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (len(self.indices),) or not np.isfinite(action).all():
            raise ValueError("Action must contain the configured number of finite values")
        action = np.clip(action, -1, 1)
        controls = np.asarray(TARGET_CONTROLS, dtype=float).copy()
        controls[self.indices] = (LOW[self.indices] +
                                  (action + 1) / 2 * (HIGH[self.indices] - LOW[self.indices]))
        return controls

    decode_requested = decode

    def encode(self, controls, dtype=np.float64):
        controls = np.asarray(controls, dtype=np.float64)
        if controls.shape != (5,) or not np.isfinite(controls).all():
            raise ValueError("Controls must contain five finite values")
        action = (2 * (controls[self.indices] - LOW[self.indices]) /
                  (HIGH[self.indices] - LOW[self.indices]) - 1)
        return action.astype(dtype)

    def reset(self, *, seed=None, options=None):
        # Preserve the historical physical reset while following Gym's seed
        # contract: automatic reset(seed=None) must continue the random stream.
        # The old environment reseeds its action space from OS entropy on every
        # such reset, which would make off-policy warmup irreproducible.
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.model.reset()
        self.decimator = AudioDecimator()
        self.index = 0
        self.done = False
        self.previous = np.zeros(len(self.indices), dtype=np.float64)
        self.audio, self.actions, self.rewards, self.states = [], [], [], []
        self.requested_control_history = []
        self.applied_control_history = []
        self.motion_history, self.loss_history = [], []
        self.potential = 0.0
        self.episode_native_blocks = 0
        self.episode_integrated_s = 0.0
        self.episode_failed = False
        return self._obs(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before step and after episode completion")
        # Do not coerce to float32 here: Oracle replay must expose the native
        # precision limit independently from float32 learner actions.
        action = np.asarray(action, dtype=np.float64)
        requested = self.decode(action)
        action = np.clip(action, -1, 1)
        self.actions.append(action.copy())
        self.decision_steps += 1
        blocks = self.physical_horizon if self.task == "A" else 1
        failed = False
        for _ in range(blocks):
            self.model.set_controls(requested)
            self.requested_control_history.append(requested.copy())
            self.applied_control_history.append(self.model.controls.copy())
            self.native_blocks += 1
            self.episode_native_blocks += 1
            self.integrated_s = self.native_blocks * CONTROL_S
            self.episode_integrated_s = self.episode_native_blocks * CONTROL_S
            try:
                raw = self.model.advance(CONTROL_S)
                wave = self.decimator.process(raw)
            except FloatingPointError:
                self.audio.append(np.zeros(self.block))
                failed = True
                break
            self.native_blocks_completed += 1
            self.integrated_completed_s = self.native_blocks_completed * CONTROL_S
            self.audio.append(wave)
            self.states.append(self.model.state.copy())

        self.index += 1
        self.previous = action.copy()
        self.done = failed or self.index == self.horizon
        if failed:
            self.failures += 1
            self.episode_failed = True
            # The entire return is -1e6, even if a failed episode had earlier
            # finite prefix rewards. Never score a conveniently short prefix.
            reward = self.potential - 1e6
            metrics = {"total": 1e6}
            self.potential = 1e6
        else:
            audio = np.concatenate(self.audio)
            metrics = distance(self.target_audio[:len(audio)], audio, RATE,
                               scale=self.loss_scale)
            potential = len(audio) / len(self.target_audio) * metrics["total"]
            reward = self.potential - potential
            self.potential = potential
        self.rewards.append(float(reward))
        self.loss_history.append(dict(step=self.index, prefix_samples=len(self.audio)*self.block,
                                      potential=float(self.potential), reward=float(reward),
                                      prefix_metrics=metrics, numerical_failure=failed))
        if self.done:
            self.completed_episodes += 1
            self.successful_episodes += int(not failed)
        info = dict(numerical_failure=failed, native_blocks=self.episode_native_blocks,
                    integrated_s=self.episode_integrated_s)
        if self.done:
            info["episode_metrics"] = metrics
        obs = (np.zeros(self.observation_space.shape, dtype=np.float32)
               if failed else self._obs())
        return obs, float(reward), self.done, False, info
