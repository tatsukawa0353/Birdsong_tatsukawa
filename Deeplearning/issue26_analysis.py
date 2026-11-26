"""Replay, sampler reconstruction and descriptive statistics for Issue26 only."""
from .result_paths import resolve as resolve_result_path
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from .issue26_data import (ROOT,OUT,BASE,TARGET,CONDITIONS,NEW,SEEDS,PREFIX,RADII,LOW,HIGH,F0_BINS,
    environment,ready,folder_for,curve_for,chosen,half_width,load_checkpoint,save_audio,target_data)
from .issue26_agent import make_agent,continuous_local,BehaviorSAC
from .issue26_training import comparable
from .constant020_env import episode,account
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def csv_rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def compressed_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));stream=io.StringIO(newline='')
    w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)
    Path(path).write_bytes(gzip.compress(stream.getvalue().encode(),compresslevel=6,mtime=0))


def source_for(p):
    return p['proposal']['source'] if 'proposal' in p else 'prefix_warmup' if p['episodes']<=16 else 'prefix_sac' if p['episodes']<=PREFIX else 'sac'


def best_record(p):
    return dict(trial=p['episodes'],J=p['weighted_loss'],E_total=p['acoustic_error'],physical_controls=p['physical_controls'],
        action=p['action'],sustained_voiced=p['sustained_voiced'],failed=p['failed'])


def action_fields(action,controls):
    return dict(a_epsilon_l=action[0],a_ps=action[1],a_f0_l=action[2],epsilon_l=controls[0],ps=controls[2],f0_l=controls[3])


def metrics_fields(p):
    return dict(E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],E_env=p['acoustic_components']['envelope'],
        J=p['weighted_loss'],voiced=p['sustained_voiced'],failed=p['failed'])


def audit():
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1);m=ready();tick=time.perf_counter();assert BehaviorSAC.train is SAC.train
    replay_rows=[];accounts=[];sound_index=[];run_rows=[];prefix_rows=[];sampler_rows=[]
    (OUT/'baseline_replays').mkdir(exist_ok=True)
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder=folder_for(condition,seed);s=json.loads((folder/'summary.json').read_text());curve=curve_for(condition,seed)
            trace=read_record(folder/'training_trace.json');updates=read_record(folder/'updates.json');new=condition in NEW
            for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,(condition,seed,p)
            if s.get('status')=='failed':
                assert (folder/'failure.json').exists() and 'final' not in s
                run_rows.append(dict(condition=condition,seed=seed,status='failed',training_candidates=len(trace),evaluations=len(curve),failure=s['failure']))
                continue
            assert len(trace)==4096 and len(curve)==33 and len(updates)==4080
            assert [p['episodes'] for p in curve]==list(range(0,4097,128))
            assert s['optimizer_steps']==dict(actor=4080,critic=4080,entropy=4080)
            assert s['training']['episodes']==4096 and s['evaluation']['episodes']==33
            if new:
                assert s['replay_transition_checks']==4096 and s['optimizer_lr_checks']==dict(actor=4080,critic=4080,entropy=4080)
                assert s['source_sha256']==m['source_sha256']
            env=environment(condition);mapping=make_agent(condition,seed,env,m['settings'])
            choice=np.random.default_rng(260026+seed);candidate=np.random.default_rng(260126+seed);incumbent=None;local_count=0;local_attempts=0
            try:
                for i,p in enumerate(trace,1):
                    assert p['episodes']==i and p['reward']==float(np.float32(-p['weighted_loss']))
                    assert np.isfinite(p['action']).all() and np.max(np.abs(p['action']))<=1
                    np.testing.assert_array_equal(env.decode(p['action']),p['physical_controls'])
                    comp=p['acoustic_components'];assert p['acoustic_error']==(1e6 if p['failed'] else comp['log_spectral']+comp['envelope'])
                    assert p['weighted_loss']==(1e6 if p['failed'] else .75*comp['log_spectral']+.25*comp['envelope'])
                    improved=incumbent is None or p['weighted_loss']<incumbent['J']
                    if new:
                        q=p['proposal'];assert q['ref_before']==incumbent and q['local_sampling_preserved_learner_rng']
                        np.testing.assert_array_equal(q['env_action'],p['action']);np.testing.assert_array_equal(q['replay_action'],p['stored_action'])
                        np.testing.assert_array_equal(mapping.policy.scale_action(np.asarray(p['action'],np.float32)),p['stored_action'])
                        if condition in RADII and i>PREFIX:
                            draw=float(choice.random());assert q['local_draw']==draw
                            expected_local=draw<.8;assert (q['source']=='local')==expected_local
                            if expected_local:
                                center=np.asarray(incumbent['physical_controls'])[[0,2,3]]
                                lower=np.maximum(LOW,center-RADII[condition]);upper=np.minimum(HIGH,center+RADII[condition])
                                actual=np.asarray(p['physical_controls'])[[0,2,3]];requested=np.asarray(q['requested_controls'])
                                assert np.all(actual>=lower) and np.all(actual<=upper) and np.all(requested>=lower) and np.all(requested<=upper)
                                np.testing.assert_array_equal(q['local_lower'],lower);np.testing.assert_array_equal(q['local_upper'],upper)
                                a,b,expected=continuous_local(env,mapping.policy,center,RADII[condition],candidate)
                                np.testing.assert_array_equal(a[0],p['action']);np.testing.assert_array_equal(b[0],p['stored_action'])
                                for k,v in expected.items():assert q[k]==v,(condition,seed,i,k)
                                local_count+=1;local_attempts+=q['draw_attempts']
                            else:assert q['stock_action']==p['action']
                        else:
                            assert q['local_draw'] is None and q['source']!='local' and q['stock_action']==p['action']
                        assert p['improved']==improved and p['replay_pos']==i-1
                    if improved:incumbent=best_record(p)
                    if new:assert p['ref_after']==incumbent
                if new:assert s['training_best']==incumbent
                for p in curve:
                    assert p['learner_before']==p['learner_after']
                    if new:
                        assert p['observer_before']==p['observer_after']
                        assert p['optimizer_lrs']==dict(actor=[3e-4],critic=[3e-4],entropy=[3e-4])
                        expect=min(trace[:p['episodes']],key=lambda x:x['weighted_loss']) if p['episodes'] else None
                        assert p['training_incumbent']==(best_record(expect) if expect else None)
                        np.testing.assert_array_equal(p['physical_action_std_mc'],np.array(p['policy_statistics']['normalized_action_std_mc'])*half_width(condition))
                if condition in RADII:
                    btrace=read_record(folder_for('B',seed)/'training_trace.json');bcurve={p['episodes']:p for p in curve_for('B',seed)}
                    assert [comparable(p) for p in trace[:PREFIX]]==[comparable(p) for p in btrace[:PREFIX]]
                    for n in (128,256):
                        p=next(p for p in curve if p['episodes']==n)
                        assert p['learner_before']==bcurve[n]['learner_before']
                    assert s['prefix_comparison']['passed'];prefix_rows.append(dict(condition=condition,seed=seed,training_rows=256,learner_states=2,passed=True))
                for j,u in enumerate(updates,1):
                    assert u['trial']==j+16 and u['update']==j
                    np.testing.assert_array_equal(np.array(u['after']['action'])-u['before']['action'],u['delta_action'])
                    if j>1:np.testing.assert_array_equal(u['before']['action'],updates[j-2]['after']['action'])
                    if new:assert u['optimizer_lrs']==dict(actor=[3e-4],critic=[3e-4],entropy=[3e-4])
                sampler_rows.append(dict(condition=condition,seed=seed,local_candidates=local_count,uniform_attempts=local_attempts,
                    rounded_edge_redraws=local_attempts-local_count,actual_action_reconstruction_rows=len(trace),passed=True))
                t=time.perf_counter();endpoints=chosen(curve)
                roles=['final','selected','posthoc']+(['prefix256'] if condition=='B' else [])
                for role in roles:
                    p=next(p for p in curve if p['episodes']==256) if role=='prefix256' else endpoints[role]
                    agent,meta=load_checkpoint(folder/f'{role}.pt',condition,seed,m['settings'])
                    metrics,values=episode(env,agent=agent)
                    np.testing.assert_array_equal(values['actions'][0],p['actions'][0]);assert metrics['weighted_loss']==p['weighted_loss']
                    with np.load(folder/f'{role}.npz') as z:
                        for k in ('audio','physical_controls','actions'):np.testing.assert_array_equal(values[k],z[k])
                    assert meta['J']==p['weighted_loss'] and meta['trial']==p['episodes']
                    replay_rows.append(dict(condition=condition,seed=seed,role=role,trial=p['episodes'],audio_max_error=0.,J_error=0.,model_action_exact=True))
                    sound_index.append(dict(condition=condition,seed=seed,role=role,trial=p['episodes'],raw_npz=str((folder/f'{role}.npz').relative_to(ROOT)),
                        **metrics_fields(p),**action_fields(p['actions'][0],p['physical_controls'])))
                best=min(trace,key=lambda p:p['weighted_loss']);metrics,values=episode(env,action=best['action'])
                assert metrics['weighted_loss']==best['weighted_loss'];assert metrics['sustained_voiced']==best['sustained_voiced']
                if condition=='A':raw=OUT/'baseline_replays'/f'seed_{seed}_training_best.npz';save_audio(raw,values)
                else:
                    raw=folder/'training_best.npz'
                    with np.load(raw) as z:
                        for k in ('audio','physical_controls','actions'):np.testing.assert_array_equal(values[k],z[k])
                replay_rows.append(dict(condition=condition,seed=seed,role='training_best',trial=best['episodes'],audio_max_error=0.,J_error=0.,model_action_exact='action-only'))
                sound_index.append(dict(condition=condition,seed=seed,role='training_best',trial=best['episodes'],raw_npz=str(raw.relative_to(ROOT)),
                    **metrics_fields(best),**action_fields(best['action'],best['physical_controls'])))
                accounts.append(dict(condition=condition,seed=seed,**account(env,time.perf_counter()-t)))
                run_rows.append(dict(condition=condition,seed=seed,status='complete',new_full_training=new,training_candidates=4096,evaluations=33,
                    gradient_updates=4080,Replay_transition_checks=4096 if new else 'legacy audit',evaluation_to_Replay_insertions=0,
                    training_voiced=sum(p['sustained_voiced'] for p in trace),training_failed=sum(p['failed'] for p in trace),training_best_trial=best['episodes']))
            finally:env.close()
    patterns=[]
    for seed in SEEDS:
        c=read_record(folder_for('C',seed)/'training_trace.json');d=read_record(folder_for('D',seed)/'training_trace.json')
        assert [source_for(p) for p in c]==[source_for(p) for p in d];patterns.append(dict(seed=seed,choice_patterns_equal=True,rows=4096))
    write_csv(OUT/'model_replay_checks.csv',replay_rows);write_csv(OUT/'sound_index.csv',sound_index);write_csv(OUT/'run_accounting.csv',run_rows)
    write_csv(OUT/'prefix_checks.csv',prefix_rows);write_csv(OUT/'sampler_checks.csv',sampler_rows)
    record=dict(passed=True,generated_sounds=sum(a['episodes'] for a in accounts),model_replays=sum(r['role']!='training_best' for r in replay_rows),
        training_action_replays=sum(r['role']=='training_best' for r in replay_rows),accounts=accounts,prefix=prefix_rows,choice_patterns=patterns,
        source_sha256=digest(__file__),training_inherited_from_standard_SAC=True,wall_s=time.perf_counter()-tick)
    write_json(OUT/'audit.json',record);print(json.dumps(dict(audit=True,sounds=record['generated_sounds'])),flush=True);return record


def aggregate():
    ready();endpoints=[];arrivals=[];distributions=[];bins=[];sources=[];centers=[];local=[];candidates=[];policies=[];movements=[];intervals=[];local_stats=[]
    for c in CONDITIONS:
        for seed in SEEDS:
            folder=folder_for(c,seed);summary=json.loads((folder/'summary.json').read_text());curve=curve_for(c,seed);trace=read_record(folder/'training_trace.json')
            if summary.get('status')=='failed':continue
            updates=read_record(folder/'updates.json');best=min(trace,key=lambda p:p['weighted_loss']);prefix=min(trace[:PREFIX],key=lambda p:p['weighted_loss'])
            for role,p in dict(**chosen(curve),training_best=best).items():
                action=p['action'] if role=='training_best' else p['actions'][0]
                endpoints.append(dict(condition=c,seed=seed,role=role,trial=p['episodes'],**metrics_fields(p),**action_fields(action,p['physical_controls'])))
            tr=next((p for p in trace[PREFIX:] if p['weighted_loss']<=.5*prefix['weighted_loss']),None)
            ea=next((p for p in curve if p['acoustic_error']<=.5*curve[0]['acoustic_error']),None)
            ja=next((p for p in curve if p['episodes']>=PREFIX and p['weighted_loss']<=1.25*prefix['weighted_loss']),None)
            selected=chosen(curve)['selected'];final=curve[-1]
            arrivals.append(dict(condition=c,seed=seed,prefix256_best_J=prefix['weighted_loss'],prefix_best_trial=prefix['episodes'],
                first_training_half_prefix_J=tr['episodes'] if tr else None,training_half_prefix_J_reached=bool(tr),
                first_actor_half_initial_E=ea['episodes'] if ea else None,actor_half_initial_E_reached=bool(ea),
                first_actor_near_prefix_J=ja['episodes'] if ja else None,actor_near_prefix_J_reached=bool(ja),
                actor_near_prefix_J_retention_fraction=sum(p['weighted_loss']<=1.25*prefix['weighted_loss'] for p in curve if p['episodes']>=ja['episodes'])/sum(p['episodes']>=ja['episodes'] for p in curve) if ja else None,
                final_half_initial_E=final['acoustic_error']<=.5*curve[0]['acoustic_error'],final_near_prefix_J=final['weighted_loss']<=1.25*prefix['weighted_loss'],
                final_J_to_prefix_J=final['weighted_loss']/prefix['weighted_loss'],
                final_minus_selected_E=final['acoustic_error']-selected['acoustic_error'],final_minus_selected_J=final['weighted_loss']-selected['weighted_loss'],
                final_minus_training_best_J=final['weighted_loss']-best['weighted_loss'],selected_minus_training_best_J=selected['weighted_loss']-best['weighted_loss'],
                training_best_trial=best['episodes'],training_best_voiced=best['sustained_voiced']))
            physical=np.array([p['physical_controls'] for p in trace]);f0=physical[:,3];voice=np.array([p['sustained_voiced'] for p in trace]);fail=np.array([p['failed'] for p in trace])
            d=dict(condition=c,seed=seed,candidates=len(trace),voiced=int(voice.sum()),voiced_fraction=float(voice.mean()),failed=int(fail.sum()),silent=summary['training']['silent_episodes'],
                **{f'f0_q{int(q*100):02d}':float(np.quantile(f0,q)) for q in (0,.05,.25,.5,.75,.95,1)})
            for threshold in (800000,2500000,5000000):
                mask=f0>threshold;d.update({f'above_{threshold}_count':int(mask.sum()),f'above_{threshold}_fraction':float(mask.mean()),
                    f'above_{threshold}_voiced':int(voice[mask].sum()),f'above_{threshold}_voiced_fraction':float(voice[mask].mean()) if mask.any() else None})
            distributions.append(d)
            for lo,hi in zip(F0_BINS[:-1],F0_BINS[1:]):
                mask=(f0>=lo if lo==F0_BINS[0] else f0>lo)&(f0<=hi);v=voice[mask]
                bins.append(dict(condition=c,seed=seed,lower=lo,upper=hi,count=int(mask.sum()),voiced=int(v.sum()),voiced_fraction=float(v.mean()) if len(v) else None,
                    epsilon_mean=float(physical[mask,0].mean()) if mask.any() else None,ps_mean=float(physical[mask,2].mean()) if mask.any() else None,
                    best_J=min((p['weighted_loss'] for p,m in zip(trace,mask) if m),default=None)))
            incumbent=None;prev_f0=None;source_items={}
            for p in trace:
                n=p['episodes'];source=source_for(p);improved=incumbent is None or p['weighted_loss']<incumbent['J'];q=p.get('proposal',{})
                source_items.setdefault(source,[]).append((p,improved));fields=action_fields(p['action'],p['physical_controls'])
                candidates.append(dict(condition=c,seed=seed,trial=n,source=source,strict_incumbent_improvement=improved,
                    reference_trial=incumbent['trial'] if incumbent else None,reference_J=incumbent['J'] if incumbent else None,**metrics_fields(p),**fields))
                if source=='local':
                    deviation=q['local_deviation'];radius=RADII[c];actual=np.asarray(p['physical_controls'])[[0,2,3]];center=np.asarray(incumbent['physical_controls'])[[0,2,3]]
                    local.append(dict(condition=c,seed=seed,trial=n,reference_trial=incumbent['trial'],reference_J=incumbent['J'],J=p['weighted_loss'],voiced=p['sustained_voiced'],
                        improved=improved,center_f0=center[2],actual_f0=actual[2],delta_previous_f0=float(actual[2]-prev_f0),
                        **{f'delta_{name}':deviation[j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                        **{f'fraction_radius_{name}':deviation[j]/radius[j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                        **{f'truncated_{name}':bool(q['local_lower'][j]>center[j]-radius[j] or q['local_upper'][j]<center[j]+radius[j]) for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                        draw_attempts=q['draw_attempts'],rounding_f0=q['rounding_error'][2]))
                if improved:
                    incumbent=best_record(p);centers.append(dict(condition=c,seed=seed,trial=n,source=source,**metrics_fields(p),**fields))
                prev_f0=p['physical_controls'][3]
            for source,items in source_items.items():
                ps=[p for p,_ in items];hit=sum(h for _,h in items)
                sources.append(dict(condition=c,seed=seed,source=source,count=len(ps),fraction_all=len(ps)/4096,
                    fraction_after256=len(ps)/3840 if source in ('local','sac') else None,strict_improvements=hit,hit_rate=hit/len(ps),
                    voiced=sum(p['sustained_voiced'] for p in ps),failed=sum(p['failed'] for p in ps),best_J=min(p['weighted_loss'] for p in ps)))
            for i,p in enumerate(curve):
                stats=p['policy_statistics'];current=min(trace[:p['episodes']],key=lambda x:x['weighted_loss']) if p['episodes'] else None
                policies.append(dict(condition=c,seed=seed,trial=p['episodes'],**metrics_fields(p),**action_fields(p['actions'][0],p['physical_controls']),
                    alpha=stats['alpha'],gaussian_entropy=stats['gaussian_entropy'],squashed_entropy_mc=stats['squashed_entropy_mc'],entropy_mc_se=stats['entropy_mc_se'],
                    **{f'normalized_std_{name}':stats['normalized_action_std_mc'][j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                    **{f'physical_std_{name}':stats['normalized_action_std_mc'][j]*half_width(c)[j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                    **{f'latent_mean_{name}':stats['latent_mean'][j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                    **{f'latent_std_{name}':stats['latent_std'][j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                    training_incumbent_J=current['weighted_loss'] if current else None))
                if i:
                    last=curve[i-1];us=[u for u in updates if last['episodes']<u['trial']<=p['episodes']]
                    delta=np.asarray(p['actions'][0])-last['actions'][0]
                    intervals.append(dict(condition=c,seed=seed,start=last['episodes'],end=p['episodes'],updates=len(us),
                        normalized_path=sum(u['delta_action_l2'] for u in us),normalized_net=float(np.linalg.norm(delta)),
                        **{f'physical_net_{name}':delta[j]*half_width(c)[j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                        **{key:float(np.mean([u[key] for u in us])) for key in ('actor_loss','critic_loss','ent_coef_loss')},
                        delta_E=p['acoustic_error']-last['acoustic_error'],delta_J=p['weighted_loss']-last['weighted_loss']))
            for u in updates:
                movements.append(dict(condition=c,seed=seed,trial=u['trial'],update=u['update'],alpha=u['after']['alpha'],
                    **{key:u[key] for key in ('delta_action_l2','actor_loss','critic_loss','ent_coef_loss')},
                    **{f'delta_a_{name}':u['delta_action'][j] for j,name in enumerate(('epsilon_l','ps','f0_l'))},
                    **{f'delta_physical_{name}':u['delta_action'][j]*half_width(c)[j] for j,name in enumerate(('epsilon_l','ps','f0_l'))}))
            if c in RADII:
                rs=[r for r in local if r['condition']==c and r['seed']==seed];f0=np.array([p['physical_controls'][3] for p in trace])
                changes=np.flatnonzero(np.r_[True,np.diff(np.minimum.accumulate([p['weighted_loss'] for p in trace]))<0])+1
                r=dict(condition=c,seed=seed,local_count=len(rs),local_fraction_after256=len(rs)/3840,local_fraction_all=len(rs)/4096,
                    rounding_redraws=sum(r['draw_attempts']-1 for r in rs),f0_radius=RADII[c][2],
                    consecutive_f0_steps_over_radius=int((abs(np.diff(f0)[255:])>RADII[c][2]).sum()),
                    local_consecutive_f0_steps_over_radius=sum(abs(r['delta_previous_f0'])>RADII[c][2] for r in rs),
                    incumbent_updates=len(changes),longest_incumbent_plateau=int(np.diff(np.r_[changes,4097]).max()),
                    nonvoiced_incumbent_updates=sum(not r['voiced'] for r in centers if r['condition']==c and r['seed']==seed))
                for name in ('epsilon_l','ps','f0_l'):
                    values=np.array([r['delta_'+name] for r in rs]);r.update({f'{name}_deviation_q{int(q*100):02}':float(np.quantile(values,q)) for q in (0,.05,.5,.95,1)})
                local_stats.append(r)
    groups=[]
    for c in CONDITIONS:
        for role in ('final','selected','posthoc','training_best'):
            rows=[r for r in endpoints if r['condition']==c and r['role']==role];g=dict(condition=c,role=role,complete_seeds=len(rows),all3_complete=len(rows)==3)
            if len(rows)==3:
                for field in ('E_total','E_spec','E_env','J'):
                    v=np.array([r[field] for r in rows]);g.update({field+'_mean':float(v.mean()),field+'_sd':float(v.std(ddof=1))})
                g.update(voiced_seeds=sum(r['voiced'] for r in rows),failed_seeds=sum(r['failed'] for r in rows))
            groups.append(g)
    paired=[]
    for left,right in (('B','A'),('C','B'),('D','B'),('D','C')):
        for seed in SEEDS:
            for role in ('final','selected','posthoc','training_best'):
                a=next((r for r in endpoints if (r['condition'],r['seed'],r['role'])==(left,seed,role)),None)
                b=next((r for r in endpoints if (r['condition'],r['seed'],r['role'])==(right,seed,role)),None)
                if a and b:paired.append(dict(comparison=left+'−'+right,seed=seed,role=role,delta_E=a['E_total']-b['E_total'],delta_J=a['J']-b['J']))
    for name,rows in (('endpoints',endpoints),('summary_by_condition',groups),('arrivals_and_retention',arrivals),('candidate_distribution',distributions),
        ('f0_bins',bins),('behavior_sources',sources),('center_history',centers),('policy_dynamics',policies),('update_intervals',intervals),('paired_differences',paired),('local_sampling_summary',local_stats)):
        write_csv(OUT/f'{name}.csv',rows)
    for name,rows in (('training_candidates',candidates),('local_displacements',local),('actor_updates',movements)):compressed_csv(OUT/f'{name}.csv.gz',rows)
    write_json(OUT/'aggregate_execution.json',dict(generated_sounds=0,source_sha256=digest(__file__),candidate_rows=len(candidates),eval_rows=len(policies),
        actor_update_rows=len(movements),local_rows=len(local),no_error_threshold_gate=True))
    return groups


def listening():
    rows=csv_rows('sound_index.csv');target,_=target_data();gain=1./float(np.max(abs(target)));items=[]
    archive=OUT/'listening_wav.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def add(name,audio,raw):
            wave=(audio*gain).astype(np.float32);assert np.isfinite(wave).all() and len(wave)==10000
            stream=io.BytesIO();wavfile.write(stream,50000,wave);z.writestr(name,stream.getvalue())
            items.append(dict(wav=name,source=raw,raw_key='target' if name=='Target.wav' else 'audio',gain=gain,peak=float(abs(wave).max()),samples=10000,sample_rate=50000,clipped_samples=0))
        add('Target.wav',target,str(TARGET.relative_to(ROOT)))
        for r in rows:
            raw=resolve_result_path(ROOT/r['raw_npz'])
            with np.load(raw) as a:add(f"{r['condition']}_seed{r['seed']}_{r['role']}.wav",a['audio'],r['raw_npz'])
    write_csv(OUT/'wav_index.csv',items);return items


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('audit','aggregate','listening'));a=p.parse_args()
    {'audit':audit,'aggregate':aggregate,'listening':listening}[a.stage]()
