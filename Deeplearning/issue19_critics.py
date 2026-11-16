"""Offline, paired candidate-pool diagnostics. No simulator or learner updates."""
import argparse
import csv
import gzip
import inspect
import json
import time

import numpy as np
from scipy.stats import spearmanr

from .issue19_data import (BASE, OUT, METHODS, environment, folder_for, curve_for,
                           endpoints, pack, target_data)
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def action_key(action):
    return np.asarray(action, dtype=np.float32).tobytes()


def pool():
    """Fixed across methods / weights, balanced grid plus pre-evaluated random."""
    path = OUT/'critic_pool.csv'
    if path.exists():
        with path.open() as stream:
            return [dict(r, action=json.loads(r['action']), E_spec=float(r['E_spec']), E_env=float(r['E_env']),
                sustained_voiced=r['sustained_voiced']=='True') for r in csv.DictReader(stream)]
    with (BASE/'diagnostics/domain_grid.csv').open() as stream:
        grid = [r for r in csv.DictReader(stream) if float(r['f0_l']) <= 800000. and r['failed'] == 'False']
    env = environment(); points = []
    try:
        for voiced in (True, False):
            subset = [r for r in grid if (r['sustained_voiced']=='True') == voiced]
            chosen = sorted(set(np.linspace(0, len(subset)-1, 96).round().astype(int).tolist()
                                + sorted(range(len(subset)), key=lambda i: float(subset[i]['E_total']))[:24]))
            for i in chosen:
                r = subset[i]; c = np.array(json.loads(r['applied_controls']))
                points.append(dict(candidate=f'grid_{r["index"]}', source='physical_grid',
                    action=env.encode(c).tolist(), E_spec=float(r['E_spec']), E_env=float(r['E_env']),
                    sustained_voiced=voiced))
        for seed in (0, 1, 2):
            rows = curve_for('random_search', seed, 1.)
            chosen = set(range(0, 4096, 16))
            for weight in (1., .5):
                chosen.add(min(range(len(rows)), key=lambda i: rows[i]['acoustic_components']['log_spectral']+weight*rows[i]['acoustic_components']['envelope']))
            # Include all DE selected solutions so the pool probes high precision.
            for i in sorted(chosen):
                r = rows[i]; comp = r['acoustic_components']
                points.append(dict(candidate=f'random_s{seed}_n{r["episodes"]}', source='saved_random',
                    action=r['actions'][0], E_spec=comp['log_spectral'], E_env=comp['envelope'],
                    sustained_voiced=r['sustained_voiced']))
            r = endpoints('DE', seed, 1.)['selected']; comp = r['acoustic_components']
            points.append(dict(candidate=f'DE_lambda1_s{seed}_selected', source='saved_DE', action=r['actions'][0],
                E_spec=comp['log_spectral'], E_env=comp['envelope'], sustained_voiced=r['sustained_voiced']))
    finally:
        env.close()
    write_csv(path, [dict(p, action=json.dumps(p['action'])) for p in points])
    return points


def scores(actual, predicted):
    actual = np.asarray(actual); predicted = np.asarray(predicted)
    difference = predicted-actual
    i, j = np.triu_indices(len(actual), k=1)
    truth = actual[i]-actual[j]; estimate = predicted[i]-predicted[j]
    comparable = (truth != 0) & (estimate != 0)
    count = int(comparable.sum())
    rho = float(spearmanr(actual, predicted).statistic) if len(actual)>1 and np.ptp(actual)>0 and np.ptp(predicted)>0 else None
    return dict(n=len(actual), rmse=float(np.sqrt(np.mean(difference**2))), mae=float(np.mean(abs(difference))),
        bias_Q_minus_r=float(np.mean(difference)), spearman=rho,
        pairwise_inversion_fraction=float(np.mean(truth[comparable]*estimate[comparable]<0)) if count else None,
        comparable_pairs=count)


def diagnose(weights):
    import torch
    from stable_baselines3 import TD3, SAC
    torch.set_num_threads(1)
    tick = time.perf_counter(); points = pool(); predictions = []; metrics = []; actors = []
    implementations = {m: inspect.getsource(c.train) for m, c in (('TD3', TD3), ('SAC', SAC))}
    for weight in weights:
        for method in ('TD3', 'SAC'):
            for seed in (0, 1, 2):
                folder = folder_for(method, seed, weight)
                agent = {'TD3':TD3, 'SAC':SAC}[method].load(folder/'agent.zip', device='cpu')
                agent.policy.set_training_mode(False)
                trace = read_record(folder/'training_trace.json')
                sampled = {action_key(r.get('action', r.get('applied_action'))) for r in trace}
                own = []
                for r in trace[::32]:
                    comp = r.get('acoustic_components', r.get('components'))
                    own.append(dict(candidate=f'own_n{r["episodes"]}', source='own_training',
                        action=r['action'], E_spec=comp['log_spectral'], E_env=comp['envelope'],
                        sustained_voiced=r['sustained_voiced']))
                final = endpoints(method, seed, weight)['final']; comp = final['acoustic_components']
                actor = dict(candidate='final_actor', source='actor', action=final['actions'][0],
                    E_spec=comp['log_spectral'], E_env=comp['envelope'], sustained_voiced=final['sustained_voiced'])
                candidates = points+own+[actor]
                actions = np.asarray([p['action'] for p in candidates], dtype=np.float32)
                # Env.reset() gives [-1,0,...,0]. No VecNormalize; action space
                # [-1,1] means SB3 scale_action is the identity.
                obs = np.zeros((len(actions), 8), dtype=np.float32); obs[:, 0] = -1
                assert agent.observation_space.shape == (8,) and agent.action_space.shape == (3,)
                scaled_actions = agent.policy.scale_action(actions)
                np.testing.assert_allclose(scaled_actions, actions, rtol=0, atol=1.2e-7)
                deterministic = agent.predict(obs[-1], deterministic=True)[0]
                assert np.array_equal(deterministic, actions[-1])
                with torch.no_grad():
                    q1, q2 = [q.cpu().numpy().ravel().astype(float) for q in agent.critic(torch.as_tensor(obs), torch.as_tensor(scaled_actions))]
                actual = -np.array([p['E_spec']+weight*p['E_env'] for p in candidates])
                common = np.array([p['E_spec']+p['E_env'] for p in candidates])
                qs = dict(Q1=q1, Q2=q2, minQ=np.minimum(q1, q2))
                seen = np.array([action_key(a) in sampled for a in actions])
                fixed = np.arange(len(candidates)) < len(points)
                low = fixed & (actual >= np.quantile(actual[:len(points)], .9))
                subsets = {'fixed_pool':fixed, 'not_exactly_sampled':fixed & ~seen,
                           'own_training':np.array([p['source']=='own_training' for p in candidates]),
                           'low_J_decile':low}
                for label, mask in subsets.items():
                    for name, q in qs.items():
                        if mask.any(): metrics.append(dict(method=method, seed=seed, weight=weight,
                            subset=label, critic=name, **scores(actual[mask], q[mask])))
                objective_q = q1 if method == 'TD3' else qs['minQ']
                best_actual = int(np.argmax(actual[:len(points)])); best_q = int(np.argmax(objective_q[:len(points)]))
                nonoracle = np.flatnonzero(common[:len(points)] > 1e-8)
                best_nonoracle = int(nonoracle[np.argmax(actual[nonoracle])])
                actors.append(dict(method=method, seed=seed, weight=weight,
                    actor_E_total=common[-1], actor_J=-actual[-1], actor_Q1=q1[-1], actor_Q2=q2[-1],
                    actor_action=json.dumps(actor['action']), actor_controls=json.dumps(final['physical_controls']),
                    true_best_candidate=points[best_actual]['candidate'], true_best_J=-actual[best_actual],
                    true_best_Q1=q1[best_actual], true_best_Q2=q2[best_actual],
                    critic_preferred_candidate=points[best_q]['candidate'], critic_preferred_J=-actual[best_q],
                    pool_max_actor_objective_Q_gap=objective_q[best_q]-objective_q[-1],
                    actor_J_minus_pool_best=actual[best_actual]-actual[-1],
                    true_best_rank_by_actor_Q=int(1+np.sum(objective_q[:len(points)]>objective_q[best_actual])),
                    best_nonoracle_candidate=points[best_nonoracle]['candidate'], best_nonoracle_J=-actual[best_nonoracle],
                    best_nonoracle_rank_by_actor_Q=int(1+np.sum(objective_q[:len(points)]>objective_q[best_nonoracle])),
                    actor_objective='Q1' if method=='TD3' else 'expected minQ + entropy (deterministic diagnostic only)',
                    entropy_alpha=float(agent.log_ent_coef.exp().item()) if method=='SAC' else None,
                    fixed_pool_size=len(points), fixed_pool_exact_sample_overlap=int((fixed & seen).sum())))
                for i, p in enumerate(candidates):
                    predictions.append(dict(dict(p, action=json.dumps(p['action'])), method=method, seed=seed, weight=weight,
                        actual_reward=actual[i], E_total=common[i],
                        Q1=q1[i], Q2=q2[i], minQ=qs['minQ'][i],
                        exact_training_action_float32=bool(seen[i])))
    suffix = 'baseline' if weights == (1.,) else 'all'
    write_csv(OUT/f'critic_metrics_{suffix}.csv', metrics)
    write_csv(OUT/f'actor_diagnostics_{suffix}.csv', actors)
    # Complete predictions are one gzip CSV, not thousands of files.
    fields = list(predictions[0])
    with gzip.open(OUT/f'critic_predictions_{suffix}.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader(); writer.writerows(predictions)
    write_json(OUT/f'critic_execution_{suffix}.json', dict(generated_sounds=0, learner_updates=0,
        fixed_pool=len(points), models=len(weights)*6, wall_s=time.perf_counter()-tick,
        source_sha256=digest(__file__), sb3_train_source=implementations,
        diagnostic_rule='Fixed grid/random/DE pool defined before lambda=.5 training; 128 own training points shown separately; unseen means no exact float32 action match, not independent statistical holdout',
        quantization='Critic uses policy.scale_action(float32 actions), identity on [-1,1] up to 1.2e-7 arithmetic rounding; rewards from original float64 actions; no new simulator calls. Exact training match refers to saved environment actions, replay-buffer normalization may differ by rounding.',
        target_semantics='All transitions terminal: Q1/Q2 supervised targets r; bootstrap and next-step entropy zero. TD3 actor maximizes Q1; SAC actor minimizes alpha*logpi-minQ for sampled actions.'))
    print(f'Offline critic diagnostics complete: {len(weights)*6} models, {len(points)} fixed candidates, no new sounds', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--baseline-only', action='store_true')
    a = p.parse_args(); diagnose((1.,) if a.baseline_only else (1., .5))
