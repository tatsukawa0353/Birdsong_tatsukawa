"""Reward-aware endpoint audits and offline Q diagnostics; no old-file writes."""
from .result_paths import resolve as resolve_result_path
import argparse
import csv
import gzip
import importlib.metadata
import json
import re
import time

import numpy as np

from .constant020_env import episode,account
from .issue14 import effective_hyperparameters,learner_signature
from .issue21_training import network_geometry
from .issue19_critics import pool,scores
from .issue22_data import (ROOT,BASE,OUT,CONDITIONS,WEIGHTS,SEEDS,POOL,ready,environment,
    folder_for,curve_for,endpoints,label,coefficients,objective,policy_statistics)
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def loss_for(condition,spec,env):
    return float(spec+env) if condition=='legacy_sum' else objective(spec,env,condition)


def audit():
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1);manifest=ready();started=time.perf_counter()
    assert digest(BASE/'manifest.json')==manifest['baseline_manifest_sha256']
    assert manifest['versions']==json.loads((BASE/'manifest.json').read_text())['versions']
    assert manifest['versions']=={p:importlib.metadata.version(p) for p in manifest['versions']}
    replays=[];sources=[];accounts=[];accounting=[];legacy_statistics=[]
    for seed in SEEDS:
        old_folder=folder_for('legacy_sum',seed);old=json.loads((old_folder/'summary.json').read_text())
        old_trace=read_record(old_folder/'training_trace.json')
        for condition in CONDITIONS:
            folder=folder_for(condition,seed);s=json.loads((folder/'summary.json').read_text())
            curve=curve_for(condition,seed);trace=read_record(folder/'training_trace.json');new=condition!='legacy_sum'
            assert len(trace)==4096 and [p['episodes'] for p in trace]==list(range(1,4097))
            assert [p['episodes'] for p in curve]==list(range(0,4097,128))
            assert s['training']['episodes']==4096 and s['gradient_updates']==4080
            assert s['training']['native_steps_completed']==8192000000 and s['training']['failures']==0 and s['nonfinite_detected']==0
            assert s['optimizer_steps']==dict(actor=4080,critic=4080,entropy=4080)
            assert s['effective_hyperparameters']==old['effective_hyperparameters']
            assert curve[0]['learner_before']['policy_sha256']==old['initial']['learner_before']['policy_sha256']
            assert [p['action'] for p in trace[:16]]==[p['action'] for p in old_trace[:16]]
            if new:
                assert s['source_sha256']==manifest['source_sha256']
                assert s['independent_policy_evaluations']==33 and s['evaluation']['episodes']==35
                for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
            env=environment(condition);tick=time.perf_counter();cache={};audio_cache={}
            try:
                for p in trace:
                    comp=p.get('acoustic_components',p.get('components'))
                    assert not p['failed'] and np.isfinite(p['acoustic_error'])
                    assert p['acoustic_error']==comp['log_spectral']+comp['envelope']
                    assert p['weighted_loss']==loss_for(condition,comp['log_spectral'],comp['envelope'])
                    assert p['reward']==float(np.float32(-p['weighted_loss']))
                    np.testing.assert_array_equal(env.decode(p['action']),p['physical_controls'])
                    c=p['physical_controls'];assert c[1]==3.5e7 and c[4]==4.9e4
                    assert all(lo<=c[i]<=hi for lo,i,hi in ((2.5e7,0,5e7),(2.5e6,2,4e6),(49000.,3,800000.)))
                    if new:assert np.isfinite(p['alpha_before_update']) and p['alpha_before_update']>0
                for p in curve:
                    assert p['learner_before']==p['learner_after']
                    comp=p['acoustic_components'];assert p['weighted_loss']==loss_for(condition,comp['log_spectral'],comp['envelope'])
                    if new:assert p['spectral_coefficient']+p['envelope_coefficient']==1.
                chosen=endpoints(condition,seed)
                assert chosen['selected']['episodes']==s['selected']['episodes']
                if new:assert chosen['posthoc']['episodes']==s['posthoc']['episodes']
                for role,expected in chosen.items():
                    model_path=None;audio_path=None;stats=None
                    if role!='posthoc':
                        model_path=folder/('agent.zip' if role=='final' else 'selected_agent.zip');audio_path=folder/f'{role}.npz'
                        agent=SAC.load(model_path,device='cpu');network_geometry(agent,128)
                        assert all(p.device.type=='cpu' and torch.isfinite(p).all() for p in agent.policy.parameters())
                        effective=json.loads(json.dumps(effective_hyperparameters(agent)))
                        for key,value in s['effective_hyperparameters'].items():
                            if key!='initial_entropy_coefficient':assert effective[key]==value,key
                        action=agent.predict(env.reset()[0],deterministic=True)[0]
                        np.testing.assert_array_equal(action,expected['actions'][0])
                        before=learner_signature(agent);stats=policy_statistics(agent);assert before==learner_signature(agent)
                        if new:
                            for key in ('alpha','squashed_entropy_mc','gaussian_entropy','entropy_mc_se'):
                                assert abs(stats[key]-expected['policy_statistics'][key])<1e-7,(key,condition,seed)
                        else:legacy_statistics.append(dict(condition=condition,seed=seed,role=role,trial=expected['episodes'],**stats))
                    else:action=np.asarray(expected['actions'][0])
                    key=np.asarray(action,dtype=np.float64).tobytes();reused=key in cache
                    if not reused:
                        measured,wave=episode(env,action=action);cache[key]=(measured,wave)
                    metrics,values=cache[key]
                    assert metrics['acoustic_error']==expected['acoustic_error'] and metrics['weighted_loss']==expected['weighted_loss']
                    assert metrics['sustained_voiced']==expected['sustained_voiced']
                    for part in ('log_spectral','envelope'):assert metrics['acoustic_components'][part]==expected['acoustic_components'][part]
                    if audio_path:
                        with np.load(audio_path) as z:
                            np.testing.assert_array_equal(values['audio'],z['audio']);np.testing.assert_array_equal(values['physical_controls'],z['physical_controls'])
                        audio_cache[key]=audio_path
                    elif key in audio_cache:audio_path=audio_cache[key]
                    replays.append(dict(condition=label(condition),seed=seed,role=role,trial=expected['episodes'],
                        E_total=metrics['acoustic_error'],J=metrics['weighted_loss'],sustained_voiced=metrics['sustained_voiced'],
                        model_reloaded=model_path is not None,sound_reused_within_run=reused,error_difference=0.,waveform_max_difference=0. if audio_path else None))
                    sources.append(dict(condition=label(condition),seed=seed,role=role,trial=expected['episodes'],
                        model_path=str(model_path.relative_to(ROOT)) if model_path else None,model_sha256=digest(model_path) if model_path else None,
                        audio_path=str(audio_path.relative_to(ROOT)) if audio_path else None,audio_sha256=digest(audio_path) if audio_path else None,
                        action=expected['actions'][0],posthoc_is_action_diagnostic=role=='posthoc'))
            finally:accounts.append(account(env,time.perf_counter()-tick));env.close()
            accounting.append(dict(condition=label(condition),seed=seed,new_training_candidates=4096 if new else 0,
                training_candidates=4096,evaluation_opportunities=33,source_training_wall_s=s['training']['wall_s'],
                source_evaluation_sounds=s['evaluation']['episodes'],source_process_cpu_s=s['process_cpu_s'],
                source_summary=str((folder/'summary.json').relative_to(ROOT))))
    write_csv(OUT/'endpoint_replays.csv',replays);write_csv(OUT/'run_accounting.csv',accounting)
    write_json(OUT/'endpoint_sources.json',sources);write_json(OUT/'legacy_policy_statistics.json',legacy_statistics)
    prior=json.loads((OUT/'audit.json').read_text()) if (OUT/'audit.json').exists() else {}
    passes=prior.get('passes',[])+[dict(generated_sounds=sum(a['episodes'] for a in accounts),wall_s=time.perf_counter()-started,source_sha256=digest(__file__))]
    write_json(OUT/'audit.json',dict(passed=True,runs=12,new_runs=9,training_candidates=36864,independent_policy_evaluations=297,
        training_endpoint_replays=18,model_reloads=24,endpoint_records=len(replays),generated_sounds=sum(a['episodes'] for a in accounts),
        cumulative_generated_sounds=sum(p['generated_sounds'] for p in passes),passes=passes,
        only_learner_change='sum-one reward definition',warmup_and_initial_policies_match=True,
        coefficient_sums=1,numerical_failures=0,source_sha256=digest(__file__)))
    integrity();print('Audit passed: nine x4096; Final/Selected models and posthoc actions; original files preserved',flush=True)


def diagnose():
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1);m=ready();points=pool();started=time.perf_counter();rows=[];predictions=[]
    for condition in CONDITIONS:
        for seed in SEEDS:
            agent=SAC.load(folder_for(condition,seed)/'agent.zip',device='cpu');agent.policy.set_training_mode(False)
            actions=np.asarray([p['action'] for p in points],dtype=np.float32);obs=np.zeros((len(points),8),dtype=np.float32);obs[:,0]=-1
            with torch.no_grad():q1,q2=[q.cpu().numpy().ravel().astype(float) for q in agent.critic(torch.as_tensor(obs),torch.as_tensor(agent.policy.scale_action(actions)))]
            actual=-np.array([loss_for(condition,p['E_spec'],p['E_env']) for p in points]);scale=float(actual.std(ddof=1))
            masks=dict(fixed_pool=np.ones(len(points),dtype=bool),low_J_decile=actual>=np.quantile(actual,.9))
            for head,q in (('Q1',q1),('Q2',q2),('minQ',np.minimum(q1,q2))):
                for subset,mask in masks.items():
                    metrics=scores(actual[mask],q[mask]);rows.append(dict(condition=label(condition),seed=seed,critic=head,subset=subset,
                        **metrics,reward_std_fixed_pool=scale,rmse_over_reward_std=metrics['rmse']/scale))
            for i,p in enumerate(points):predictions.append(dict(condition=label(condition),seed=seed,candidate=p['candidate'],source=p['source'],
                E_spec=p['E_spec'],E_env=p['E_env'],actual_reward=actual[i],Q1=q1[i],Q2=q2[i],minQ=min(q1[i],q2[i])))
    write_csv(OUT/'critic_metrics.csv',rows)
    with gzip.open(OUT/'critic_predictions.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]),lineterminator='\n');writer.writeheader();writer.writerows(predictions)
    write_json(OUT/'critic_execution.json',dict(models=12,candidates=len(points),generated_sounds=0,learner_updates=0,
        pool_sha256=m['critic_pool_sha256'],wall_s=time.perf_counter()-started,source_sha256=digest(__file__),
        target='Q1/Q2 against each own -J; all transitions terminal. SAC actor still uses current entropy term.',
        interpretation='Raw RMSE/bias across different coefficients/scales are not directly comparable. Standardized RMSE also uses different target rankings, not a causal capacity test.',
        pool_limit='Same saved stratified Issue19 pool including oracle/three old DE anchors; no new candidates or simulations; not a statistically independent test population'))
    print('Q diagnostics complete: twelve models, fixed1005 pool, own reward targets; zero new sounds',flush=True)


def integrity():
    ready();checked=0
    with (OUT/'protected_before.csv').open() as stream:
        for r in csv.DictReader(stream):assert digest(resolve_result_path(ROOT/r['path']))==r['sha256'],r['path'];checked+=1
    links=0;broken=[]
    for doc in OUT.glob('*.md'):
        for match in re.finditer(r'\]\(([^)]+)\)',doc.read_text()):
            ref=match[1].split('#')[0]
            if not ref or '://' in ref:continue
            links+=1
            if not (doc.parent/ref).exists():broken.append(dict(document=doc.name,path=ref))
    assert not broken,broken
    for r in json.loads((OUT/'endpoint_sources.json').read_text()) if (OUT/'endpoint_sources.json').exists() else []:
        for kind in ('model','audio'):
            if r[kind+'_path']:assert digest(resolve_result_path(ROOT/r[kind+'_path']))==r[kind+'_sha256']
    images=list((OUT/'figures').glob('*.png'));assert len(images)<=10
    if (OUT/'figure_sources.json').exists():
        registry=json.loads((OUT/'figure_sources.json').read_text())['images'];assert len(registry)==len(images)==8
        for r in registry:assert digest(OUT/r['path'])==r['sha256']
    sample_rows=0;counts={}
    if (OUT/'training_samples.csv.gz').exists():
        with gzip.open(OUT/'training_samples.csv.gz','rt') as stream:
            for r in csv.DictReader(stream):
                w=float(r['lambda']);spec,env,total,J,reward=[float(r[k]) for k in ('E_spec','E_env','E_total','J','reward_float32')]
                assert abs(total-spec-env)<1e-14 and J==objective(spec,env,w) and reward==float(np.float32(-J))
                counts.setdefault((w,int(r['seed'])),[]).append(int(r['trial']));sample_rows+=1
        assert len(counts)==9 and all(ns==list(range(1,4097)) for ns in counts.values())
    write_json(OUT/'integrity.json',dict(passed=True,original_files_checked=checked,original_files_changed=0,original_files_deleted=0,
        local_links_checked=links,broken_links=broken,images=len(images),training_csv_rows=sample_rows,source_sha256=digest(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('audit','critic','integrity'));a=p.parse_args()
    {'audit':audit,'critic':diagnose,'integrity':integrity}[a.action]()
