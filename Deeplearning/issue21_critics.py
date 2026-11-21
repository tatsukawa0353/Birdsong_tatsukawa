"""Evaluate all widths on exactly the saved Issue19 pool, without new sounds."""
import csv
import gzip
import inspect
import json
import time

import numpy as np

from .issue19_critics import pool,scores,action_key
from .issue21_data import OUT,PRIOR,METHODS,WIDTHS,SEEDS,folder_for,endpoints,ready
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def diagnose():
    import torch
    from stable_baselines3 import TD3,SAC
    torch.set_num_threads(1);manifest=ready();tick=time.perf_counter();points=pool()
    assert len(points)==1005
    metrics=[];actors=[];predictions=[]
    for method in METHODS:
        for width in WIDTHS:
            for seed in SEEDS:
                folder=folder_for(method,width,seed)
                agent={'TD3':TD3,'SAC':SAC}[method].load(folder/'agent.zip',device='cpu')
                agent.policy.set_training_mode(False)
                trace=read_record(folder/'training_trace.json')
                seen={action_key(p['action']) for p in trace};own=[]
                for p in trace[::32]:
                    comp=p.get('acoustic_components',p.get('components'))
                    own.append(dict(candidate=f'own_n{p["episodes"]}',source='own_training',action=p['action'],
                        E_spec=comp['log_spectral'],E_env=comp['envelope'],sustained_voiced=p['sustained_voiced']))
                final=endpoints(method,width,seed)['final'];comp=final['acoustic_components']
                actor=dict(candidate='final_actor',source='actor',action=final['actions'][0],
                    E_spec=comp['log_spectral'],E_env=comp['envelope'],sustained_voiced=final['sustained_voiced'])
                candidates=points+own+[actor]
                actions=np.array([p['action'] for p in candidates],dtype=np.float32)
                observations=np.zeros((len(candidates),8),dtype=np.float32);observations[:,0]=-1
                scaled=agent.policy.scale_action(actions)
                np.testing.assert_allclose(scaled,actions,rtol=0,atol=1.2e-7)
                np.testing.assert_array_equal(agent.predict(observations[-1],deterministic=True)[0],actions[-1])
                with torch.no_grad():
                    q1,q2=[q.cpu().numpy().ravel().astype(float) for q in agent.critic(torch.as_tensor(observations),torch.as_tensor(scaled))]
                q_values=dict(Q1=q1,Q2=q2,minQ=np.minimum(q1,q2))
                actual=-np.array([p['E_spec']+p['E_env'] for p in candidates])
                fixed=np.arange(len(candidates))<len(points)
                matches=np.array([action_key(a) in seen for a in actions])
                masks=dict(fixed_pool=fixed,not_exactly_sampled=fixed&~matches,
                    own_training=np.array([p['source']=='own_training' for p in candidates]),
                    low_J_decile=fixed&(actual>=np.quantile(actual[:len(points)],.9)))
                for subset,mask in masks.items():
                    for critic,q in q_values.items():
                        if mask.any():metrics.append(dict(method=method,width=width,seed=seed,subset=subset,critic=critic,**scores(actual[mask],q[mask])))
                q_actor=q1 if method=='TD3' else q_values['minQ']
                nonoracle=np.flatnonzero(-actual[:len(points)]>1e-8)
                best=int(nonoracle[np.argmax(actual[nonoracle])]);max_q=int(np.argmax(q_actor[:len(points)]))
                actors.append(dict(method=method,width=width,seed=seed,actor_E_total=-actual[-1],
                    actor_Q1=q1[-1],actor_Q2=q2[-1],actor_action=json.dumps(actor['action']),
                    actor_controls=json.dumps(final['physical_controls']),
                    best_nonoracle_candidate=points[best]['candidate'],best_nonoracle_error=-actual[best],
                    best_nonoracle_rank_by_actor_Q=int(1+np.sum(q_actor[:len(points)]>q_actor[best])),
                    critic_preferred_candidate=points[max_q]['candidate'],critic_preferred_error=-actual[max_q],
                    pool_max_Q_minus_actor_Q=q_actor[max_q]-q_actor[-1],
                    fixed_pool_size=len(points),fixed_pool_exact_sample_matches=int((fixed&matches).sum()),
                    actor_objective='Q1' if method=='TD3' else 'expected minQ + entropy; deterministic output is only a diagnostic',
                    entropy_alpha=float(agent.log_ent_coef.exp().item()) if method=='SAC' else None))
                for i,p in enumerate(candidates):
                    predictions.append(dict(dict(p,action=json.dumps(p['action'])),method=method,width=width,seed=seed,
                        actual_reward=actual[i],Q1=q1[i],Q2=q2[i],minQ=q_values['minQ'][i],exact_training_action_float32=bool(matches[i])))
    # The width32 inference must reproduce the already saved diagnosis, not just
    # the old acoustic endpoints. This also tests pool/normalization semantics.
    with (PRIOR/'critic_metrics_all.csv').open() as stream:
        old={(r['method'],int(r['seed']),r['subset'],r['critic']):r for r in csv.DictReader(stream) if float(r['weight'])==1.}
    maximum=0.
    for r in metrics:
        if r['width']!=32:continue
        previous=old[r['method'],r['seed'],r['subset'],r['critic']]
        for field in ('rmse','mae','bias_Q_minus_r','spearman','pairwise_inversion_fraction'):
            delta=abs(r[field]-float(previous[field]));assert delta<1e-14,(field,delta)
            maximum=max(maximum,delta)
    write_csv(OUT/'critic_metrics.csv',metrics);write_csv(OUT/'actor_diagnostics.csv',actors)
    with gzip.open(OUT/'critic_predictions.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]),lineterminator='\n');writer.writeheader();writer.writerows(predictions)
    write_json(OUT/'critic_execution.json',dict(models=len(METHODS)*len(WIDTHS)*len(SEEDS),fixed_candidates=len(points),generated_sounds=0,learner_updates=0,
        pool_path=manifest['critic_pool_path'],pool_sha256=manifest['critic_pool_sha256'],
        baseline_diagnostic_max_difference=maximum,wall_s=time.perf_counter()-tick,source_sha256=digest(__file__),
        score_implementation_sha256=digest(PRIOR.parent.parent/'issue19_critics.py'),
        sb3_train_implementation={m:inspect.getsource(cls.train) for m,cls in (('TD3',TD3),('SAC',SAC))},
        normalization='Fixed reset observation [-1,0,...,0]; no VecNormalize; policy.scale_action(float32(a)); same restricted mapping; rewards from original float64 simulation',
        independence='Fixed Issue19 stratified pool, including known oracle and 3 saved DE anchors; unchanged data, no new DE calculation; exact action match is not a statistical independence guarantee',
        semantics='All transitions terminal, target Q1/Q2=r. TD3 actor uses Q1; SAC actor expected minQ with entropy, deterministic output need not maximize diagnostic minQ.'))
    print('Critic diagnosis complete: 24 models x1005 fixed candidates; baseline reproduced; no new sounds',flush=True)


if __name__=='__main__':diagnose()
