"""Shared training/evaluation accounting for SAC, TD3 and PPO."""
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import torch

from .acoustics import distance, evaluation_features
from .plotting import comparison, learning_curves
from .sine_env import SineConfig, SineEnv


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def make_agent(name, env, seed, settings):
    torch.set_num_threads(1)
    common = dict(policy="MlpPolicy", env=env, seed=seed, device="cpu", verbose=0,
                  policy_kwargs=dict(net_arch=[32, 32]), learning_rate=settings["learning_rate"])
    if name == "PPO":
        return PPO(**common, n_steps=settings["ppo_rollout"], batch_size=64, n_epochs=5)
    common.update(buffer_size=50000, batch_size=64, learning_starts=256,
                  train_freq=1, gradient_steps=1)
    if name == "SAC":
        return SAC(**common)
    if name == "TD3":
        noise = NormalActionNoise(np.zeros(env.action_space.shape), 0.1*np.ones(env.action_space.shape))
        return TD3(**common, action_noise=noise)
    raise ValueError("Unknown algorithm")


def rollout(env, policy, *, seed=10000):
    obs, _ = env.reset(seed=seed)
    start, failed = time.perf_counter(), False
    for _ in range(env.horizon):
        action = policy(obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            failed = bool(info.get("numerical_failure", False))
            break
    audio = np.concatenate(env.audio) if env.audio else np.zeros_like(env.target_audio)
    metrics = distance(env.target_audio[:len(audio)], audio, env.config.sample_rate, scale=env.loss_scale)
    if failed or len(audio) != len(env.target_audio):
        # Failed episodes must not benefit from evaluating only a good prefix.
        metrics["total"] = 1e6
        failed = True
    return dict(error=metrics["total"], metrics=metrics, failed=failed,
                env_steps=env.index, seconds=time.perf_counter()-start,
                features=evaluation_features(audio, env.config.sample_rate)), audio, np.array(env.actions)


class EvaluationCallback(BaseCallback):
    def __init__(self, factory, interval, threshold):
        super().__init__()
        self.factory, self.interval, self.threshold = factory, interval, threshold
        self.entries = []
        self.evaluation_steps = 0
        self.evaluation_seconds = 0.0
        self.training_numerical_failures = 0
        self.training_episodes = 0

    def evaluate(self):
        start = time.perf_counter()
        env = self.factory()
        result, _, _ = rollout(env, lambda obs, _: self.model.predict(obs, deterministic=True)[0])
        env.close()
        self.evaluation_steps += result["env_steps"]
        self.entries.append(dict(steps=self.num_timesteps, **result))
        self.evaluation_seconds += time.perf_counter()-start

    def _on_training_start(self):
        self.evaluate()

    def _on_step(self):
        self.training_numerical_failures += sum(bool(i.get("numerical_failure", False))
                                               for i in self.locals.get("infos", []))
        self.training_episodes += int(np.count_nonzero(self.locals.get("dones", [])))
        if self.num_timesteps % self.interval == 0:
            self.evaluate()
        return True


def train_run(name, seed, train_factory, eval_factory, settings, output):
    output.mkdir(parents=True, exist_ok=True)
    train_env = train_factory()
    agent = make_agent(name, train_env, seed, settings)
    callback = EvaluationCallback(eval_factory, settings["eval_interval"], settings["threshold"])
    start = time.perf_counter()
    agent.learn(total_timesteps=settings["steps"], callback=callback)
    if callback.entries[-1]["steps"] != agent.num_timesteps:
        callback.evaluate()
    elapsed = time.perf_counter()-start
    evaluation = eval_factory()
    result, audio, actions = rollout(evaluation, lambda obs, _: agent.predict(obs, deterministic=True)[0])
    applied = np.array(getattr(evaluation, "applied_control_history", []))
    requested = np.array(getattr(evaluation, "requested_control_history", []))
    if not len(applied):
        applied = np.array([evaluation.decode(a) for a in actions])
        requested = applied.copy()
    motion = getattr(evaluation, "motion_history", [])
    np.savez_compressed(output/"rollout.npz", target=evaluation.target_audio, audio=audio,
                        actions=actions, sample_rate=evaluation.config.sample_rate,
                        states=np.array(getattr(evaluation, "states", [])),
                        physical_controls=applied, requested_controls=requested,
                        motion_mean=np.array([m["mean"] for m in motion]),
                        motion_std=np.array([m["std"] for m in motion]))
    agent.save(output/"agent")
    comparison(evaluation.target_audio, audio, evaluation.config.sample_rate, output/"spectrogram.png")
    reached = [e["steps"] for e in callback.entries if e["error"] <= settings["threshold"]]
    summary = dict(algorithm=name, seed=seed, requested_steps=settings["steps"],
                   training_steps=agent.num_timesteps,
                   evaluation_steps=callback.evaluation_steps+result["env_steps"],
                   training_and_eval_seconds=elapsed,
                   evaluation_seconds=callback.evaluation_seconds+result["seconds"],
                   first_threshold_step=min(reached) if reached else None,
                   initial_error=callback.entries[0]["error"], **result)
    summary.update(training_numerical_failures=callback.training_numerical_failures,
                   training_episodes=callback.training_episodes)
    write_json(output/"summary.json", summary)
    write_json(output/"curve.json", callback.entries)
    train_env.close()
    evaluation.close()
    return summary, callback.entries


def run_phase0(config, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", config)
    train_factory = lambda: SineEnv(SineConfig(randomize=True, target=config["target"]))
    eval_factory = lambda: SineEnv(SineConfig(randomize=False, target=config["target"]))
    summaries, curves, baselines = [], {}, []
    for target in ("constant", "glide"):
        for seed in config["seeds"]:
            for policy_name in ("random", "constant", "oracle"):
                env = SineEnv(SineConfig(target=target))
                rng = np.random.default_rng(seed)
                policies = dict(random=lambda obs, e: rng.uniform(-1, 1, 2),
                                constant=lambda obs, e: np.zeros(2),
                                oracle=lambda obs, e: e.goal_actions[e.index])
                result, audio, actions = rollout(env, policies[policy_name], seed=seed)
                baselines.append(dict(target=target, policy=policy_name, seed=seed, **result))
                if seed == config["seeds"][0]:
                    np.savez_compressed(output/f"{target}_{policy_name}.npz", target=env.target_audio,
                                        audio=audio, actions=actions, sample_rate=env.config.sample_rate)
    write_json(output/"baselines.json", baselines)
    for name in config["algorithms"]:
        curves[name] = []
        for seed in config["seeds"]:
            print(f"Phase 0: {name}, seed={seed}", flush=True)
            summary, entries = train_run(name, seed, train_factory, eval_factory, config, output/f"{name}_{seed}")
            summaries.append(summary)
            write_json(output/"summary.json", summaries)
            curves[name].append(([e["steps"] for e in entries], [e["error"] for e in entries]))
    learning_curves(curves, output/"learning_curves.png")
    ranking = sorted(config["algorithms"], key=lambda name: np.median([s["error"] for s in summaries if s["algorithm"] == name]))
    write_json(output/"ranking.json", dict(ranking=ranking, criterion="median final held-out acoustic error; provisional, small budget",
                                          selected=ranking[:2]))
    return summaries


def run_phase1(config, output):
    from .model_env import ModelConfig, ModelEnv

    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"config.json", config)
    factory = lambda: ModelEnv(ModelConfig(condition=config["condition"], target=config["target"]))
    summaries, curves = [], {}
    for name in config["algorithms"]:
        curves[name] = []
        for seed in config["seeds"]:
            print(f"Phase 1: {name}, seed={seed}", flush=True)
            summary, entries = train_run(name, seed, factory, factory, config, output/f"{name}_{seed}")
            summaries.append(summary)
            write_json(output/"summary.json", summaries)
            curves[name].append(([e["steps"] for e in entries], [e["error"] for e in entries]))
    learning_curves(curves, output/"learning_curves.png")
    run_model_baselines(config, output)
    return summaries


def run_model_baselines(config, output):
    from scipy.optimize import differential_evolution
    from .model_env import ModelConfig, ModelEnv

    factory = lambda: ModelEnv(ModelConfig(condition=config["condition"], target=config["target"]))
    baselines = []
    for seed in config["seeds"]:
        for name in ("constant", "random_step", "oracle"):
            env = factory()
            rng = np.random.default_rng(seed)
            policies = dict(constant=lambda obs, e: np.zeros(2),
                            random_step=lambda obs, e: rng.uniform(-1, 1, 2),
                            oracle=lambda obs, e: e.encode(e.target_controls[e.index]))
            result, audio, actions = rollout(env, policies[name], seed=seed)
            baselines.append(dict(method=name, seed=seed, **result))
            if seed == config["seeds"][0]:
                np.savez_compressed(output/f"{name}.npz", audio=audio, target=env.target_audio,
                                    actions=actions, sample_rate=env.config.sample_rate)
                comparison(env.target_audio, audio, env.config.sample_rate, output/f"{name}.png")
            env.close()
        # A candidate evaluates a full 35-step episode. No gradient/model derivatives.
        # DE: population 6 * (initial + 4 generations) = 30 episodes = 1050 steps.
        for name in (("random_search", "differential_evolution") if config["direct_episodes"] else ()):
            print(f"Phase 1: {name}, seed={seed}", flush=True)
            env = factory()
            rng = np.random.default_rng(seed)
            trace, best_error, best_audio, best_action = [], float("inf"), None, None
            best_cost = float("inf")
            start = time.perf_counter()

            def objective(action):
                nonlocal best_error, best_audio, best_action, best_cost
                result, audio, _ = rollout(env, lambda obs, e: action, seed=seed)
                cost = 1e6 if result["failed"] else -float(np.mean(env.rewards))
                if cost < best_cost:
                    best_cost = cost
                    best_error, best_audio, best_action = result["error"], audio.copy(), np.array(action)
                trace.append(dict(steps=len(trace)*env.horizon+env.horizon,
                                  error=result["error"], best_error=best_error, cost=cost,
                                  best_cost=best_cost, failed=result["failed"]))
                return cost

            if name == "random_search":
                for _ in range(config["direct_episodes"]):
                    objective(rng.uniform(-1, 1, 2))
            else:
                differential_evolution(objective, [(-1, 1)]*2, seed=seed, popsize=3,
                                       maxiter=config["direct_episodes"]//6-1,
                                       polish=False, tol=0, atol=0)
            elapsed = time.perf_counter()-start
            baselines.append(dict(method=name, seed=seed, error=best_error,
                                  optimized_cost=best_cost, objective="mean negative immediate reward, same as RL",
                                  search_env_steps=len(trace)*env.horizon, seconds=elapsed,
                                  failed_fraction=float(np.mean([e["failed"] for e in trace]))))
            write_json(output/f"{name}_{seed}_curve.json", trace)
            np.savez_compressed(output/f"{name}_{seed}.npz", audio=best_audio,
                                target=env.target_audio, action=best_action, sample_rate=env.config.sample_rate)
            env.close()
        write_json(output/"baselines.json", baselines)
    return baselines
