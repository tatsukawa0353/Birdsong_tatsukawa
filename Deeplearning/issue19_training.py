"""Nine runs, unchanged learners / candidate streams, lambda=.5 only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import time

import numpy as np

from .constant020_env import account, episode
from .fixed_budget_de import fixed_budget_de
from .issue14 import effective_hyperparameters, learner_signature
from .issue19_data import (OUT, environment, ready, folder_for, pack, metric_row,
                           save_audio, source_hashes)
from .issue15_analysis import write_json


def run_one(method, seed):
    import torch
    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.noise import NormalActionNoise
    torch.set_num_threads(1)
    manifest = ready(); config = manifest['settings']; budget = config['episodes']
    folder = folder_for(method, seed, .5)
    if (folder/'summary.json').exists():
        s = json.loads((folder/'summary.json').read_text())
        if s['source_sha256'] != source_hashes():
            raise RuntimeError('Completed run has different sources')
        return s
    folder.mkdir(parents=True, exist_ok=False)
    train = environment(.5); evaluation = environment(.5)
    curve = []; trace = []; snapshots = {}; best = None; agent = None
    eval_wall = artifact_wall = 0.; hooks = []; optimizer_steps = {}; numeric = {}
    started = time.perf_counter(); cpu_started = time.process_time()

    def record(n, metrics, values):
        nonlocal best, artifact_wall
        row = metric_row(metrics, values, n, gradient_updates=int(agent._n_updates) if agent else 0)
        curve.append(row); snapshots['final'] = values
        if best is None or row['weighted_loss'] < best['weighted_loss']:
            best = row; snapshots['selected'] = values
            if agent:
                tick = time.perf_counter(); agent.save(folder/'selected_agent.zip')
                artifact_wall += time.perf_counter()-tick
        return row

    def evaluate(n):
        nonlocal eval_wall
        before = learner_signature(agent)
        tick = time.perf_counter(); metrics, values = episode(evaluation, agent=agent)
        eval_wall += time.perf_counter()-tick
        after = learner_signature(agent)
        if before != after:
            raise RuntimeError('Independent evaluation changed learner or RNG')
        row = record(n, metrics, values)
        row.update(learner_before=before, learner_after=after)

    class FiniteLogger(Logger):
        def record(self, key, value, exclude=None):
            if isinstance(value, (float, int, np.number)):
                if not np.isfinite(value):
                    raise FloatingPointError(f'Nonfinite training metric {key}')
                stat = numeric.setdefault(key, dict(count=0, minimum=float(value), maximum=float(value)))
                stat.update(count=stat['count']+1, minimum=min(stat['minimum'], float(value)), maximum=max(stat['maximum'], float(value)))
            super().record(key, value, exclude)

    print(json.dumps(dict(stage='start', method=method, seed=seed)), flush=True)
    try:
        if method in ('TD3', 'SAC'):
            kwargs = dict(policy='MlpPolicy', env=train, seed=seed, device='cpu', verbose=0,
                policy_kwargs=dict(net_arch=config['net_arch']), learning_rate=config['learning_rate'],
                gamma=config['gamma'], batch_size=config['batch_size'], buffer_size=50000,
                learning_starts=config['warmup_episodes'], train_freq=1, gradient_steps=1)
            agent = (TD3(**kwargs, action_noise=NormalActionNoise(np.zeros(3), config['td3_noise_std']*np.ones(3)))
                     if method == 'TD3' else SAC(**kwargs, ent_coef=config['sac_ent_coef']))
            effective = effective_hyperparameters(agent); agent.set_logger(FiniteLogger(None, []))
            for label, parent, attribute in (('actor', agent.actor, 'optimizer'), ('critic', agent.critic, 'optimizer'), ('entropy', agent, 'ent_coef_optimizer')):
                optimizer = getattr(parent, attribute, None)
                if optimizer is None: continue
                optimizer_steps[label] = 0
                def pre(opt, args, kwargs):
                    for group in opt.param_groups:
                        for param in group['params']:
                            if param.grad is not None and not torch.isfinite(param.grad).all():
                                raise FloatingPointError('Nonfinite gradient')
                def post(opt, args, kwargs, label=label):
                    optimizer_steps[label] += 1
                    for group in opt.param_groups:
                        for param in group['params']:
                            if not torch.isfinite(param).all(): raise FloatingPointError('Nonfinite parameter')
                hooks.extend([optimizer.register_step_pre_hook(pre), optimizer.register_step_post_hook(post)])
            evaluate(0)

            class Evaluation(BaseCallback):
                def _on_step(self):
                    info = self.locals['infos'][0]
                    trace.append(dict(episodes=self.num_timesteps, acoustic_error=info['acoustic_error'],
                        acoustic_components=info['acoustic_components'], weighted_loss=info['weighted_loss'],
                        failed=info['failed'], sustained_voiced=info['sustained_voiced'],
                        action=info['applied_action'], physical_controls=info['applied_controls'],
                        reward=float(self.locals['rewards'][0])))
                    return True

                def _on_rollout_start(self):
                    n = train.completed_episodes
                    if 0 < n < budget and n % config['eval_every_episodes'] == 0:
                        evaluate(n)
                    if n and n % 512 == 0:
                        print(json.dumps(dict(stage='progress', method=method, seed=seed, candidates=n)), flush=True)

            tick = time.perf_counter(); previous_eval = eval_wall; previous_artifact = artifact_wall
            agent.learn(total_timesteps=budget, callback=Evaluation())
            training_wall = time.perf_counter()-tick-(eval_wall-previous_eval)-(artifact_wall-previous_artifact)
            evaluate(budget)
            tick = time.perf_counter(); agent.save(folder/'agent.zip'); artifact_wall += time.perf_counter()-tick
            assert agent.num_timesteps == budget and agent._n_updates == budget-config['warmup_episodes']
            gradient_updates = agent._n_updates; nfev = 0
        elif method == 'DE':
            effective = None; gradient_updates = 0
            rng = np.random.default_rng(seed); initial = rng.uniform(-1, 1, 3)
            def candidate(action):
                metrics, values = episode(train, action=np.asarray(action))
                row = record(len(curve)+1, metrics, values)
                if len(curve) % 512 == 0:
                    print(json.dumps(dict(stage='progress', method=method, seed=seed, candidates=len(curve))), flush=True)
                return row['weighted_loss']
            tick = time.perf_counter()
            solver = fixed_budget_de(candidate, 3, rng, initial, budget=budget,
                population_size=config['de_population'], mutation=config['de_mutation'], recombination=config['de_recombination'])
            training_wall = time.perf_counter()-tick; nfev = solver['nfev']
            assert nfev == budget and solver['error'] == best['weighted_loss']
            # Tied incumbents are allowed; archive uses first minimum J.
            write_json(folder/'de_incumbent.json', solver)
        else:
            raise ValueError(method)
        for role, expected in (('final', curve[-1]), ('selected', best)):
            tick = time.perf_counter()
            metrics, values = episode(evaluation, action=snapshots[role]['actions'][0])
            eval_wall += time.perf_counter()-tick
            assert metrics['acoustic_error'] == expected['acoustic_error']
            assert metrics['weighted_loss'] == expected['weighted_loss']
            assert np.array_equal(values['audio'], snapshots[role]['audio'])
            save_audio(folder/f'{role}.npz', values)
        pack(folder/'curve.json.gz', curve)
        if agent: pack(folder/'training_trace.json.gz', trace)
        summary = dict(method=method, seed=seed, envelope_weight=.5, initial=curve[0], final=curve[-1], selected=best,
            training=account(train, training_wall), evaluation=account(evaluation, eval_wall),
            independent_policy_evaluations=33 if agent else 0, endpoint_replays=2,
            gradient_updates=gradient_updates, optimizer_steps=optimizer_steps, optimizer_nfev=nfev,
            effective_hyperparameters=effective, numeric_training_metrics=numeric, nonfinite_detected=0,
            source_sha256=source_hashes(), target_sha256=manifest['target_sha256'],
            artifact_wall_s=artifact_wall, elapsed_s=time.perf_counter()-started, process_cpu_s=time.process_time()-cpu_started)
        write_json(folder/'summary.json', summary)
        print(json.dumps(dict(stage='complete', method=method, seed=seed, final=curve[-1]['acoustic_error'], selected=best['acoustic_error'])), flush=True)
        return summary
    except Exception as exc:
        write_json(folder/'failure.json', dict(type=type(exc).__name__, message=str(exc), candidates=train.completed_episodes))
        raise
    finally:
        for hook in hooks: hook.remove()
        train.close(); evaluation.close()


def train(workers=9):
    ready(); tick = time.perf_counter(); summaries = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_one, m, s) for m in ('TD3', 'SAC', 'DE') for s in (0, 1, 2)]
        for f in as_completed(futures):
            summaries.append(f.result())
    write_json(OUT/'execution.json', dict(workers=workers, runs=len(summaries), wall_s=time.perf_counter()-tick,
        training_candidates=sum(s['training']['episodes'] for s in summaries),
        independent_policy_evaluations=198, endpoint_replay_sounds=18, new_random_candidates=0,
        training_process_wall_s=sum(s['training']['wall_s'] for s in summaries),
        process_cpu_s=sum(s['process_cpu_s'] for s in summaries)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workers', type=int, default=9)
    a = p.parse_args(); train(a.workers)
