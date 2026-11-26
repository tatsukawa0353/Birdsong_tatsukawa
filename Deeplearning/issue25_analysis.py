"""Independent LR/replay audits, paired summaries and Final fixed-pool Critic scores."""
from .result_paths import resolve as resolve_result_path
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from .issue25_data import (ROOT,BASE,OUT,RATES,SEEDS,NAMES,HALF_WIDTH,BASE_LR,CRITIC_RATES,
    ready,folder_for,curve_for,chosen,environment,load_checkpoint,predict_q,q_action_gradient,policy_statistics,pack)
from .issue25_agent import make_agent,effective_difference
from .issue14 import learner_signature
from .constant020_env import episode,account
from .issue19_critics import action_key,scores
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest
from .issue25_offline import scores as fast_scores
from .issue25_data import fixed_gradients,offline_folder,OFFLINE_UPDATES
from .issue21_training import make_agent as stock_agent


def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def audit():
    import torch
    torch.set_num_threads(1);m=ready();tick=time.perf_counter();replays=[];indexes=[];accounts=[];accounting=[]
    assert m['versions']=={p:importlib.metadata.version(p) for p in m['versions']}
    candidates=json.loads((BASE/'fixed_candidates.json').read_text())
    for rate in RATES:
        for seed in SEEDS:
            folder=folder_for(rate,seed);s=json.loads((folder/'summary.json').read_text());curve=curve_for(rate,seed)
            trace=read_record(folder/'training_trace.json');updates=read_record(folder/'updates.json');new=rate!=BASE_LR
            baseline=json.loads((folder_for(BASE_LR,seed)/'summary.json').read_text())
            for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
            if s.get('status')=='failed':
                assert (folder/'failure.json').exists() and 'final' not in s
                accounting.append(dict(critic_lr=rate,seed=seed,status='failed',training_candidates=len(trace),independent_evaluations=len(curve),
                    gradient_updates=s['gradient_updates'],failure=json.dumps(s['failure']),evaluation_to_buffer_insertions=0))
                continue
            assert len(trace)==4096 and len(updates)==4080 and len(curve)==33
            assert s['training']['episodes']==4096 and s['gradient_updates']==4080 and s['nonfinite_detected']==0
            assert s['optimizer_steps']==dict(actor=4080,critic=4080,entropy=4080)
            assert s['evaluation']['episodes']==33 and s['independent_policy_evaluations']==33
            assert [p['episodes'] for p in curve]==list(range(0,4097,128))
            assert trace[:16]==read_record(folder_for(BASE_LR,seed)/'training_trace.json')[:16]
            assert curve[0]['learner_before']['policy_sha256']==baseline['initial']['learner_before']['policy_sha256']
            if new:
                assert s['optimizer_lr_checks']==dict(actor=4080,critic=4080,entropy=4080)
                assert s['source_sha256']==m['source_sha256'] and s['warmup_exact']
                normalized=json.loads(json.dumps(s['effective_hyperparameters']));normalized['optimizer_param_groups']['critic'][0]['lr']=BASE_LR
                assert normalized==baseline['effective_hyperparameters']
            expected_lrs=dict(actor=[BASE_LR],critic=[rate],entropy=[BASE_LR]);previous_action=None
            for u in updates:
                assert u['trial']==u['update']+16
                if new:assert u['optimizer_lrs']==u['before']['optimizer_lrs']==expected_lrs
                before=np.array(u['before']['action']);after=np.array(u['after']['action'])
                np.testing.assert_array_equal(after-before,u['delta_action'])
                if previous_action is not None:np.testing.assert_array_equal(before,previous_action)
                previous_action=after
            for p in curve:
                assert p['learner_before']==p['learner_after']
                if new:assert p['optimizer_lrs']==expected_lrs
                comp=p['acoustic_components'];expected=1e6 if p['failed'] else .75*comp['log_spectral']+.25*comp['envelope']
                assert p['weighted_loss']==expected and p['acoustic_error']==comp['log_spectral']+comp['envelope']
                u=[u for u in updates if u['trial']<=p['episodes']]
                if u:np.testing.assert_array_equal(u[-1]['after']['action'],p['policy_statistics']['deterministic_action'])
            env=environment(.75);t=time.perf_counter();cache={}
            try:
                for p in trace:
                    assert p['reward']==float(np.float32(-p['weighted_loss']))
                    np.testing.assert_array_equal(env.decode(p['action']),p['physical_controls'])
                    c=p['physical_controls'];assert c[1]==3.5e7 and c[4]==49000.
                roles=chosen(curve)
                if new and s.get('first_success') is not None:roles['first_success']=s['first_success']
                for role,p in roles.items():
                    model_path=folder/f'{role}.pt';agent,meta=load_checkpoint(model_path,seed,m['settings']);signature=learner_signature(agent)
                    assert meta['trial']==p['episodes'] and meta['J']==p['weighted_loss']
                    action=agent.predict(np.array([-1,0,0,0,0,0,0,0],np.float32),deterministic=True)[0]
                    np.testing.assert_array_equal(action,p['actions'][0]);key=action_key(action);generated=key not in cache
                    if generated:cache[key]=episode(env,agent=agent)
                    metrics,values=cache[key]
                    assert metrics['weighted_loss']==p['weighted_loss'] and metrics['acoustic_error']==p['acoustic_error']
                    with np.load(folder/f'{role}.npz') as z:
                        np.testing.assert_array_equal(values['audio'],z['audio']);np.testing.assert_array_equal(values['physical_controls'],z['physical_controls'])
                    stats=policy_statistics(agent)
                    for k in ('alpha','latent_log_std','normalized_action_std_mc','squashed_entropy_mc'):
                        np.testing.assert_allclose(stats[k],p['policy_statistics'][k],rtol=0,atol=1e-7)
                    np.testing.assert_array_equal(q_action_gradient(agent,action),p['actor_minQ_gradient'])
                    if role=='final':
                        q1,q2=predict_q(agent,[c['action'] for c in candidates])
                        if new:record=read_record(folder/'final_q.json')
                        else:record=[p for p in read_record(folder/'q_history.json') if p['trial']==4096]
                        np.testing.assert_array_equal(q1,[r['Q1'] for r in record]);np.testing.assert_array_equal(q2,[r['Q2'] for r in record])
                    assert signature==learner_signature(agent)
                    replays.append(dict(critic_lr=rate,seed=seed,role=role,trial=p['episodes'],E_total=p['acoustic_error'],J=p['weighted_loss'],
                        independent_sound_generated=generated,waveform_exact=True,Q_exact=True,learner_unchanged=True))
                    indexes.append(dict(critic_lr=rate,seed=seed,role=role,trial=p['episodes'],path=str(model_path.relative_to(ROOT)),
                        sha256=digest(model_path),audio_path=str((folder/f'{role}.npz').relative_to(ROOT)),audio_sha256=digest(folder/f'{role}.npz')))
            finally:accounts.append(account(env,time.perf_counter()-t));env.close()
            accounting.append(dict(critic_lr=rate,seed=seed,training_candidates=4096,independent_evaluations=33,gradient_updates=4080,
                status='complete',
                optimizer_steps_each=4080,lr_checks_each=4080 if new else '',lr_provenance='measured all updates/evaluations' if new else 'frozen stock scheduler/config; not a newly measured history',
                evaluation_to_buffer_insertions=0,training_failures=s['training']['failures'],evaluation_failures=s['evaluation']['failures']))
    write_csv(OUT/'endpoint_replays.csv',replays);write_csv(OUT/'run_accounting.csv',accounting);write_json(OUT/'checkpoint_index.json',indexes)
    write_json(OUT/'audit.json',dict(passed=True,model_reloads=len(indexes),generated_sounds=sum(a['episodes'] for a in accounts),accounts=accounts,
        complete_online_runs=sum(r['status']=='complete' for r in accounting),failed_online_runs=sum(r['status']=='failed' for r in accounting),
        new_lr_checks_each_optimizer=24480,new_learning_evaluations=198,warmup16_exact_all6=True,
        critic_only_effective_setting_difference=True,diagnostics_preserve_rng_optimizer_replay=True,
        source_sha256=digest(__file__),wall_s=time.perf_counter()-tick))
    print('Online audit passed:retained models/waves and all intended Critic LR checks; failures remain explicit',flush=True)


def tables():
    ready();endpoints=[];evaluations=[];dynamics=[];stability=[];motion=[];intervals=[];lr_history=[];update_rows=[];groups=[];paired=[]
    for rate in RATES:
        for seed in SEEDS:
            folder=folder_for(rate,seed);c=curve_for(rate,seed);selection=chosen(c) if c else {};trace=read_record(folder/'training_trace.json');updates=read_record(folder/'updates.json')
            summary=json.loads((folder/'summary.json').read_text())
            if summary.get('status')=='failed':
                for role in ('final','selected','posthoc'):endpoints.append(dict(critic_lr=rate,seed=seed,role=role,status='failed',trial=None,E_total=None,E_spec=None,E_env=None,J=None,sustained_voiced=False,failed=True))
                stability.append(dict(critic_lr=rate,seed=seed,status='failed',failure=json.dumps(summary['failure'])))
                continue
            baseline_trace=read_record(folder_for(BASE_LR,seed)/'training_trace.json')
            first_difference=next((p['episodes'] for p,b in zip(trace,baseline_trace) if p!=b),None)
            good=[p for p in c if p['acoustic_error']<=.1 and p['sustained_voiced'] and not p['failed']]
            good_samples=[p for p in trace if p['acoustic_error']<=.1 and p['sustained_voiced'] and not p['failed']]
            # Raw threshold and sustained threshold are both retained, never drop failures.
            raw_good=[p for p in c if p['acoustic_error']<=.1];raw_sample=[p for p in trace if p['acoustic_error']<=.1]
            for role,p in selection.items():
                comp=p['acoustic_components'];endpoints.append(dict(critic_lr=rate,seed=seed,role=role,status='complete',trial=p['episodes'],E_total=p['acoustic_error'],
                    E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],sustained_voiced=p['sustained_voiced'],failed=p['failed'],
                    alpha=p['policy_statistics']['alpha'],entropy=p['policy_statistics']['squashed_entropy_mc'],
                    **{name:p['physical_controls'][i] for name,i in zip(NAMES,(0,2,3))}))
            a,b,h=selection['selected'],selection['final'],selection['posthoc']
            after=[p for p in c if good and p['episodes']>=good[0]['episodes']]
            stability.append(dict(critic_lr=rate,seed=seed,status='complete',first_E_le_0p10=raw_good[0]['episodes'] if raw_good else None,
                first_training_trajectory_difference_from_baseline=first_difference,
                first_voiced_E_le_0p10=good[0]['episodes'] if good else None,policy_threshold_count=len(raw_good),voiced_policy_threshold_count=len(good),
                training_threshold_count=len(raw_sample),voiced_training_threshold_count=len(good_samples),first_training_success=good_samples[0]['episodes'] if good_samples else None,
                best_training_E=min(p['acoustic_error'] for p in trace),selected_trial=a['episodes'],posthoc_trial=h['episodes'],
                selected_minus_final_E=a['acoustic_error']-b['acoustic_error'],selected_minus_final_J=a['weighted_loss']-b['weighted_loss'],
                posthoc_minus_final_E=h['acoustic_error']-b['acoustic_error'],posthoc_minus_selected_E=h['acoustic_error']-a['acoustic_error'],
                final_success=bool(b['acoustic_error']<=.1 and b['sustained_voiced'] and not b['failed']),
                retention_evaluations=len(after),retention_successes=sum(p['acoustic_error']<=.1 and p['sustained_voiced'] and not p['failed'] for p in after),
                retention_fraction=sum(p['acoustic_error']<=.1 and p['sustained_voiced'] and not p['failed'] for p in after)/len(after) if after else None))
            for p in c:
                comp=p['acoustic_components'];st=p['policy_statistics'];n=p['episodes']
                evaluations.append(dict(critic_lr=rate,seed=seed,trial=n,E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],sustained_voiced=p['sustained_voiced'],failed=p['failed']))
                dynamics.append(dict(critic_lr=rate,seed=seed,trial=n,alpha=st['alpha'],entropy=st['squashed_entropy_mc'],gaussian_entropy=st['gaussian_entropy'],
                    entropy_mc_se=st['entropy_mc_se'],**{f'{field}_{name}':st[field][i] for field in ('normalized_action_std_mc','latent_log_std','latent_mean') for i,name in enumerate(NAMES)}))
                actual=p.get('optimizer_lrs',dict(actor=[BASE_LR],critic=[BASE_LR],entropy=[BASE_LR]))
                lr_history.append(dict(critic_lr_condition=rate,seed=seed,trial=n,kind='independent_evaluation',
                    source='measured' if rate!=BASE_LR else 'baseline scheduler/config provenance',**{name+'_lr':v[0] for name,v in actual.items()}))
            d=np.array([u['delta_action_l2'] for u in updates]);da=np.array([u['delta_action'] for u in updates]);physical=da*HALF_WIDTH
            ms=dict(critic_lr=rate,seed=seed,updates=len(d),median_delta_action_l2=float(np.median(d)),max_delta_action_l2=float(d.max()),
                p10_delta_action_l2=float(np.quantile(d,.1)),p90_delta_action_l2=float(np.quantile(d,.9)),p99_delta_action_l2=float(np.quantile(d,.99)),
                total_path_length=float(d.sum()),net_initial_final=float(np.linalg.norm(np.sum(da,axis=0))))
            for i,name in enumerate(NAMES):
                ms.update({f'median_abs_delta_physical_{name}':float(np.median(abs(physical[:,i]))),f'max_abs_delta_physical_{name}':float(np.max(abs(physical[:,i])))})
            motion.append(ms)
            for u in updates:
                row=dict(critic_lr=rate,seed=seed,trial=u['trial'],delta_action_l2=u['delta_action_l2'],
                    **{f'delta_action_{name}':u['delta_action'][i] for i,name in enumerate(NAMES)},
                    **{f'delta_physical_{name}':u['delta_action'][i]*HALF_WIDTH[i] for i,name in enumerate(NAMES)},
                    actor_loss=u['actor_loss'],critic_loss=u['critic_loss'],ent_coef_loss=u['ent_coef_loss'],ent_coef_used=u['ent_coef'])
                update_rows.append(row)
                actual=u.get('optimizer_lrs',dict(actor=[BASE_LR],critic=[BASE_LR],entropy=[BASE_LR]))
                lr_history.append(dict(critic_lr_condition=rate,seed=seed,trial=u['trial'],kind='optimizer_update',
                    source='measured pre/post all3 optimizers' if rate!=BASE_LR else 'baseline scheduler/config provenance',**{name+'_lr':v[0] for name,v in actual.items()}))
            for start,end in zip(c,c[1:]):
                recent=[u for u in updates if start['episodes']<u['trial']<=end['episodes']]
                da=np.array(end['actions'][0])-start['actions'][0];summed=np.sum([u['delta_action'] for u in recent],axis=0)
                np.testing.assert_allclose(da,summed,rtol=0,atol=1.2e-7);path=sum(u['delta_action_l2'] for u in recent);net=float(np.linalg.norm(da))
                row=dict(critic_lr=rate,seed=seed,start=start['episodes'],end=end['episodes'],updates=len(recent),path_length=path,net_distance=net,
                    path_minus_net=path-net,path_over_net=path/net if net else None,delta_E=end['acoustic_error']-start['acoustic_error'],delta_J=end['weighted_loss']-start['weighted_loss'],
                    **{key:float(np.mean([u[key] for u in recent])) for key in ('actor_loss','critic_loss','ent_coef_loss')})
                for i,name in enumerate(NAMES):row.update({f'net_delta_action_{name}':float(da[i]),f'net_delta_physical_{name}':float(da[i]*HALF_WIDTH[i])})
                intervals.append(row)
    for rate in RATES:
        for role in ('final','selected','posthoc'):
            subset=[r for r in endpoints if r['critic_lr']==rate and r['role']==role];g=dict(critic_lr=rate,role=role,voiced=sum(r['sustained_voiced'] for r in subset),failures=sum(r['failed'] for r in subset))
            g['failed_runs']=sum(r['status']=='failed' for r in subset)
            for key in ('E_total','E_spec','E_env','J'):
                if g['failed_runs']:g.update({key+'_mean':None,key+'_sd':None});continue
                data=np.array([r[key] for r in subset]);g.update({key+'_mean':float(data.mean()),key+'_sd':float(data.std(ddof=1))})
            g['final_goal_met']=role=='final' and not g['failed_runs'] and g['E_total_mean']<=.1 and g['voiced']==3 and g['failures']==0;groups.append(g)
    for rate in CRITIC_RATES:
        for seed in SEEDS:
            current=next(r for r in endpoints if r['critic_lr']==rate and r['seed']==seed and r['role']=='final')
            base=next(r for r in endpoints if r['critic_lr']==BASE_LR and r['seed']==seed and r['role']=='final')
            if current['status']=='failed':paired.append(dict(critic_lr=rate,seed=seed,status='failed'));continue
            a=next(r for r in motion if r['critic_lr']==rate and r['seed']==seed);b=next(r for r in motion if r['critic_lr']==BASE_LR and r['seed']==seed)
            paired.append(dict(critic_lr=rate,seed=seed,status='complete',lr_ratio=rate/BASE_LR,**{'delta_'+key:current[key]-base[key] for key in ('E_total','E_spec','E_env','J')},
                median_movement_ratio=a['median_delta_action_l2']/b['median_delta_action_l2'],max_movement_ratio=a['max_delta_action_l2']/b['max_delta_action_l2'],
                path_length_ratio=a['total_path_length']/b['total_path_length']))
    for filename,rows in (('comparison_per_seed.csv',endpoints),('comparison_summary.csv',groups),('policy_evaluations.csv',evaluations),
        ('policy_dynamics.csv',dynamics),('stability_and_acquisition.csv',stability),('movement_per_seed.csv',motion),('update_intervals.csv',intervals),('paired_differences.csv',paired)):
        write_csv(OUT/filename,rows)
    import gzip
    for filename,rows in (('optimizer_learning_rates.csv.gz',lr_history),('actor_updates.csv.gz',update_rows)):
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        (OUT/filename).write_bytes(gzip.compress(buffer.getvalue().encode(),mtime=0))
    write_json(OUT/'table_execution.json',dict(generated_sounds=0,learning_updates=0,cohorts=9,evaluations=297,updates=36720,source_sha256=digest(__file__)))
    print('Nine-cohort paired, acquisition/retention, movement, temperature and effective-LR tables complete',flush=True)


def critics():
    import torch
    torch.set_num_threads(1);m=ready();tick=time.perf_counter();candidates=json.loads((BASE/'fixed_candidates.json').read_text());truth=np.array([p['reward'] for p in candidates]);rows=[];metrics=[];ranks=[];time_metrics=[];gradients=[];own_arrays={};failed=[]
    for rate in RATES:
        for seed in SEEDS:
            folder=folder_for(rate,seed);summary=json.loads((folder/'summary.json').read_text())
            if summary.get('status')=='failed':failed.append(dict(critic_lr=rate,seed=seed,status='failed'));continue
            q=[p for p in read_record(folder/'q_history.json') if p['trial']==4096] if rate==BASE_LR else read_record(folder/'final_q.json')
            assert [p['candidate_id'] for p in q]==[p['candidate_id'] for p in candidates]
            heads={h:np.array([p[h] for p in q]) for h in ('Q1','Q2','minQ')}
            for subset,mask in (('all99',np.ones(99,bool)),('own_baseline33',np.array([p['source_seed']==seed for p in candidates])),('lowJ10',truth>=np.quantile(truth,.9))):
                for head,pred in heads.items():metrics.append(dict(critic_lr=rate,seed=seed,subset=subset,head=head,**scores(truth[mask],pred[mask])))
            for i,p in enumerate(candidates):rows.append(dict(critic_lr=rate,seed=seed,**{k:v for k,v in p.items() if k not in ('action','physical_controls')},**{h:float(v[i]) for h,v in heads.items()}))
            best=int(np.argmax(truth));preferred=int(np.argmax(heads['minQ']))
            ranks.append(dict(critic_lr=rate,seed=seed,true_best=candidates[best]['candidate_id'],true_best_J=-truth[best],true_best_minQ=float(heads['minQ'][best]),
                true_best_rank=int(1+(heads['minQ']>heads['minQ'][best]).sum()),predicted_best=candidates[preferred]['candidate_id'],predicted_best_true_J=-truth[preferred]))
            agent,_=load_checkpoint(folder/'final.pt',seed,m['settings']);before=learner_signature(agent)
            trace=read_record(folder/'training_trace.json');actual=-np.array([p['weighted_loss'] for p in trace]);a=np.array([p['action'] for p in trace],np.float32)
            q1,q2=predict_q(agent,a);own_heads=dict(Q1=q1,Q2=q2,minQ=np.minimum(q1,q2))
            for subset,mask in (('own_training4096',np.ones(len(trace),bool)),('own_recent1024',np.arange(len(trace))>=len(trace)-1024),('own_training_lowJ10',actual>=np.quantile(actual,.9))):
                for head,pred in own_heads.items():metrics.append(dict(critic_lr=rate,seed=seed,subset=subset,head=head,**fast_scores(actual[mask],pred[mask])))
            key=f'lr_{rate:.0e}_seed_{seed}';own_arrays.update({key+'_actions':a,key+'_rewards':actual,key+'_Q1':q1,key+'_Q2':q2})
            grad=fixed_gradients(agent,[p['action'] for p in candidates]);gradients.extend(gradient_rows('online',rate,seed,grad,candidates))
            assert before==learner_signature(agent)
            history=read_record(folder/'q_history.json')
            for n in range(0,4097,128):
                h=[p for p in history if p['trial']==n]
                for head in ('Q1','Q2','minQ'):
                    pred=np.array([p[head] for p in h])
                    for subset,mask in (('all99',np.ones(99,bool)),('lowJ10',truth>=np.quantile(truth,.9))):
                        time_metrics.append(dict(critic_lr=rate,seed=seed,trial=n,head=head,subset=subset,**fast_scores(truth[mask],pred[mask])))
    pack(OUT/'critic_predictions.json.gz',rows);write_csv(OUT/'critic_metrics.csv',metrics);write_csv(OUT/'critic_ranks.csv',ranks)
    write_csv(OUT/'critic_time_metrics.csv',time_metrics);write_csv(OUT/'online_gradient_comparison.csv',gradients)
    np.savez_compressed(OUT/'own_visit_predictions.npz',**own_arrays)
    write_json(OUT/'critic_execution.json',dict(generated_sounds=0,learning_updates=0,candidates=99,predictions=len(rows),wall_s=time.perf_counter()-tick,
        failed_final_models=failed,source_sha256=digest(__file__),limitation='Same99 retrospective candidates and own real training visits; gradient probes at legacy anchors only; no new native calls'))
    print('Online Critic:fixed99,time history,own4096 and legacy-anchor gradients;0 native calls',flush=True)


def gradient_rows(kind,rate,seed,gradients,candidates):
    rows=[]
    for r in csv_rows(BASE/'local_gradients.csv'):
        anchor=int(r['anchor']);i=next(i for i,p in enumerate(candidates) if p['source_seed']==2 and p['source_trial']==anchor);j=NAMES.index(r['variable'])
        actual=float(r['true_reward_gradient']);predicted=float(gradients[i,j])
        rows.append(dict(kind=kind,critic_lr=rate,seed=seed,anchor=anchor,variable=r['variable'],step=float(r['step']),
            true_centered_reward_gradient=actual,analytic_minQ_gradient=predicted,absolute_error=abs(predicted-actual),sign_agreement=actual*predicted>0))
    return rows


def offline_audit():
    import torch
    torch.set_num_threads(1);m=ready();tick=time.perf_counter();final=[];histories=[];indexes=[];ranks=[];gradients=[];predictions=[];comparisons={};failures=[]
    with np.load(OUT/'offline_data.npz') as z:
        data={k:z[k].copy() for k in z.files}
    candidates=json.loads((BASE/'fixed_candidates.json').read_text())
    def obs(n):
        x=torch.zeros((n,8));x[:,0]=-1;return x
    for seed in SEEDS:
        comparisons[seed]=dict(initial=set(),batches=set())
        for rate in RATES:
            folder=offline_folder(rate,seed);s=json.loads((folder/'summary.json').read_text())
            for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
            comparisons[seed]['initial'].add(s['initial_critic_sha256']);comparisons[seed]['batches'].add(s['batch_index_sha256'])
            if s['status']=='failed':failures.append(dict(rate=rate,seed=seed,failure=s['failure']));continue
            assert s['updates']==OFFLINE_UPDATES and s['native_calls']==s['actor_updates']==s['temperature_updates']==0
            assert s['actor_unchanged'] and s['temperature_unchanged'] and s['global_rng_unchanged']
            history=read_record(folder/'history.json');trace=read_record(folder/'training_trace.json')
            assert len(trace)==OFFLINE_UPDATES and all(p['lr']==rate for p in trace)
            assert sorted({p['update'] for p in history})==m['offline']['evaluation_updates']
            for row in history:histories.append(dict(critic_lr=rate,seed=seed,**row))
            for row in history:
                if row['update']==OFFLINE_UPDATES:final.append(dict(critic_lr=rate,seed=seed,**row))
            env=environment(.75)
            try:agent=stock_agent('SAC',128,seed,env,m['settings'])
            finally:env.close()
            for split in ('train','test'):
                np.testing.assert_array_equal(agent.policy.scale_action(data[split+'_raw_actions']),data[split+'_actions'])
            checkpoint=torch.load(folder/'critic.pt',map_location='cpu',weights_only=True);assert 'actor' not in checkpoint
            agent.critic.load_state_dict(checkpoint['critic']);signature=learner_signature(agent)
            with np.load(folder/'predictions.npz') as saved:
                with torch.no_grad():
                    for split,prefix in (('train','train'),('test','holdout99')):
                        a=torch.as_tensor(data[split+'_actions']);q1,q2=agent.critic(obs(len(a)),a)
                        heads=dict(Q1=q1[:,0].numpy().astype(float),Q2=q2[:,0].numpy().astype(float));heads['minQ']=np.minimum(heads['Q1'],heads['Q2'])
                        for head,p in heads.items():np.testing.assert_array_equal(p,saved[prefix+'_'+head])
                        actual=data[split+'_rewards']
                        for subset,mask in (('all',np.ones(len(a),bool)),('lowJ_decile',actual>=np.quantile(actual,.9))):
                            for head,p in heads.items():
                                metric=fast_scores(actual[mask],p[mask]);old=next(r for r in final if r['critic_lr']==rate and r['seed']==seed and r['split']==prefix and r['subset']==subset and r['head']==head)
                                for key,value in metric.items():assert old[key]==value,(key,old[key],value)
                        if split=='test':
                            best=int(np.argmax(actual));preferred=int(np.argmax(heads['minQ']))
                            ranks.append(dict(critic_lr=rate,seed=seed,true_best=candidates[best]['candidate_id'],true_best_rank=int(1+(heads['minQ']>heads['minQ'][best]).sum()),
                                predicted_best=candidates[preferred]['candidate_id'],predicted_best_true_J=-actual[preferred]))
                            for i,p in enumerate(candidates):predictions.append(dict(critic_lr=rate,seed=seed,candidate_id=p['candidate_id'],reward=actual[i],**{h:float(q[i]) for h,q in heads.items()}))
                grad=fixed_gradients(agent,data['test_actions']);np.testing.assert_array_equal(grad,saved['holdout_gradients']);gradients.extend(gradient_rows('offline',rate,seed,grad,candidates))
            assert learner_signature(agent)==signature
            indexes.append(dict(critic_lr=rate,seed=seed,path=str((folder/'critic.pt').relative_to(ROOT)),sha256=digest(folder/'critic.pt')))
        assert len(comparisons[seed]['initial'])==len(comparisons[seed]['batches'])==1,'Initial weights/batches differed by LR'
    write_csv(OUT/'offline_final_metrics.csv',final);write_csv(OUT/'offline_history.csv',histories);write_csv(OUT/'offline_ranks.csv',ranks);write_csv(OUT/'offline_gradient_comparison.csv',gradients)
    pack(OUT/'offline_holdout_predictions.json.gz',predictions);write_json(OUT/'offline_checkpoint_index.json',indexes)
    write_json(OUT/'offline_audit.json',dict(passed=True,model_reloads=len(indexes),generated_sounds=0,updates=sum(json.loads((offline_folder(r,s)/'summary.json').read_text())['updates'] for r in RATES for s in SEEDS),
        same_initial_weights_and_batch_indices_each_seed=True,actor_temperature_global_rng_unchanged=True,failed_runs=failures,
        source_sha256=digest(__file__),wall_s=time.perf_counter()-tick))
    print('Offline audit:shared initial Critic/batches, all saved predictions and9 Critic-only reloads;0 native calls',flush=True)


def integrity():
    ready();changed=[];deleted=[]
    if not (OUT/'integrity.json').exists():write_json(OUT/'integrity.json',dict(status='running'))
    protected=csv_rows(OUT/'protected_before.csv')
    for r in protected:
        p=resolve_result_path(ROOT/r['path'])
        if not p.exists():deleted.append(r['path'])
        elif digest(p)!=r['sha256']:changed.append(r['path'])
    links=0;broken=[]
    for p in (OUT/'README.md',OUT/'REPORT.md',OUT/'PLAN.md',OUT/'REPRODUCE.md'):
        if not p.exists():continue
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            links+=1
            if not (p.parent/dest.split('#')[0]).exists():broken.append(dict(file=p.name,link=dest))
    for name,root in (('artifact_checksums.json',OUT),('implementation_checksums.json',ROOT)):
        if (OUT/name).exists():
            for p,h in json.loads((OUT/name).read_text()).items():assert digest(root/p)==h,p
    if (OUT/'listening.zip').exists():
        with zipfile.ZipFile(OUT/'listening.zip') as archive:
            for row in csv_rows(OUT/'listening_index.csv'):
                rate,wave=wavfile.read(io.BytesIO(archive.read(row['archive_entry'])))
                assert rate==50000 and wave.shape==(10000,) and wave.dtype==np.float32
                with np.load(resolve_result_path(ROOT/row['raw_audio_path'])) as raw:np.testing.assert_array_equal(wave,(raw[row['raw_audio_key']]*float(row['shared_gain'])).astype(np.float32))
                assert hashlib.sha256(wave.tobytes()).hexdigest()==row['sample_sha256']
    assert json.loads((OUT/'audit.json').read_text())['passed']
    assert json.loads((OUT/'offline_audit.json').read_text())['passed']
    result=dict(passed=not changed and not deleted and not broken,original_files_checked=len(protected),original_files_changed=changed,
        original_files_deleted=deleted,local_links_checked=links,broken_links=broken,source_sha256=digest(__file__))
    write_json(OUT/'integrity.json',result);assert result['passed'],result
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('audit','tables','critics','offline_audit','integrity'));a=p.parse_args();globals()[a.stage]()
