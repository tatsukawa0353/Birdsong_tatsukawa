"""Audit retained policies, fixed-candidate Q history and bounded local sensitivity."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import re
import time

import numpy as np

from .issue23_data import (ROOT,BASE,OUT,SEEDS,NAMES,HALF_WIDTH,LOCAL_ANCHORS,LOCAL_STEPS,ready,
    folder_for,curve_for,chosen,old_folder,old_curve,environment,load_checkpoint,
    predict_q,q_action_gradient,policy_statistics,save_audio)
from .issue14 import learner_signature
from .constant020_env import episode,account
from .issue19_critics import scores,action_key
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def audit():
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1);m=ready();tick=time.perf_counter()
    assert digest(BASE/'manifest.json')==m['baseline_manifest_sha256']
    assert m['versions']=={p:importlib.metadata.version(p) for p in m['versions']}
    candidates=json.loads((OUT/'fixed_candidates.json').read_text());replays=[];accounts=[];checkpoints=[];accounting=[]
    for seed in SEEDS:
        folder=folder_for(seed);s=json.loads((folder/'summary.json').read_text());curve=curve_for(seed);points={p['episodes']:p for p in curve}
        trace=read_record(folder/'training_trace.json');updates=read_record(folder/'updates.json');qh=read_record(folder/'q_history.json')
        for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
        assert len(trace)==4096 and len(updates)==4080 and len(qh)==33*99
        assert s['training']['native_steps_completed']==8192000000 and s['training']['failures']==0
        assert s['gradient_updates']==4080 and s['optimizer_steps']==dict(actor=4080,critic=4080,entropy=4080)
        assert s['evaluation']['episodes']==s['independent_policy_evaluations']==33
        assert [p['episodes'] for p in curve]==list(range(0,4097,128))
        assert trace==read_record(old_folder(.75,seed)/'training_trace.json')
        assert s['comparison_to_issue22']['training_rows_exact']==4096 and s['comparison_to_issue22']['first_difference'] is None
        assert s['comparison_to_issue22']['evaluation_signatures_exact']==33
        previous_action=None
        for u in updates:
            assert u['trial']==u['update']+16
            before=np.array(u['before']['action']);after=np.array(u['after']['action'])
            np.testing.assert_array_equal(after-before,u['delta_action'])
            if previous_action is not None:np.testing.assert_array_equal(before,previous_action)
            previous_action=after
        for p,old in zip(curve,old_curve(.75,seed)):
            assert p['learner_before']==p['learner_after']
            assert all(p[k]==old[k] for k in ('actions','physical_controls','acoustic_error','weighted_loss','policy_statistics'))
            u=[u for u in updates if u['trial']<=p['episodes']]
            if u:np.testing.assert_array_equal(u[-1]['after']['action'],p['policy_statistics']['deterministic_action'])
            comp=p['acoustic_components'];assert p['weighted_loss']==.75*comp['log_spectral']+.25*comp['envelope']
            assert p['acoustic_error']==comp['log_spectral']+comp['envelope']
        selected=chosen(curve);env=environment(.75);t=time.perf_counter();cache={}
        roles=[(r,folder/f'{r}.pt',p) for r,p in selected.items()]
        if seed==2:roles.extend((f'at_{n}',folder/f'at_{n}.pt',points[n]) for n in (2944,3072,3840))
        try:
            for role,path,p in roles:
                agent,meta=load_checkpoint(path,seed,m['settings']);signature=learner_signature(agent)
                assert meta['trial']==p['episodes'] and meta['J']==p['weighted_loss']
                action=agent.predict(np.array([-1.,0,0,0,0,0,0,0],np.float32),deterministic=True)[0]
                np.testing.assert_array_equal(action,p['actions'][0]);key=action_key(action)
                generated=key not in cache
                if generated:cache[key]=episode(env,agent=agent)
                metrics,values=cache[key]
                assert metrics['weighted_loss']==p['weighted_loss'] and metrics['acoustic_error']==p['acoustic_error']
                np.testing.assert_array_equal(values['physical_controls'][0],p['physical_controls'])
                if role in selected:
                    with np.load(folder/f'{role}.npz') as z:
                        np.testing.assert_array_equal(z['audio'],values['audio']);np.testing.assert_array_equal(z['physical_controls'],values['physical_controls'])
                if seed==2 and role=='at_3072':save_audio(OUT/'seed2_at_3072.npz',values)
                stats=policy_statistics(agent)
                for k in ('alpha','latent_log_std','normalized_action_std_mc','squashed_entropy_mc'):
                    np.testing.assert_allclose(stats[k],p['policy_statistics'][k],atol=1e-7,rtol=0)
                q1,q2=predict_q(agent,[c['action'] for c in candidates]);record=[q for q in qh if q['trial']==p['episodes']]
                np.testing.assert_array_equal(q1,[q['Q1'] for q in record]);np.testing.assert_array_equal(q2,[q['Q2'] for q in record])
                np.testing.assert_allclose(q_action_gradient(agent,action),p['actor_minQ_gradient'],rtol=0,atol=0)
                assert signature==learner_signature(agent)
                if role in ('selected','final'):
                    old=SAC.load(old_folder(.75,seed)/('agent.zip' if role=='final' else 'selected_agent.zip'),device='cpu')
                    for current,prior in ((agent.actor,old.actor),(agent.critic,old.critic)):
                        for k,v in current.state_dict().items():assert torch.equal(v,prior.state_dict()[k])
                replays.append(dict(seed=seed,role=role,trial=p['episodes'],E_total=p['acoustic_error'],J=p['weighted_loss'],
                    independent_sound_generated=generated,waveform_exact=True,action_exact=True,Q1_Q2_exact=True,learner_unchanged=True))
                checkpoints.append(dict(seed=seed,role=role,trial=p['episodes'],path=str(path.relative_to(ROOT)),sha256=digest(path)))
        finally:accounts.append(account(env,time.perf_counter()-t));env.close()
        accounting.append(dict(seed=seed,training_candidates=4096,independent_evaluations=33,gradient_updates=4080,
            evaluation_to_buffer_insertions=0,exact_training_rows=4096,exact_evaluation_rows=33,optimizer_steps_each=4080,
            initial_env_rng_ignored_for_cross_run_only=True,all_within_run_signatures_match=True,
            saved_inference_checkpoints=len(roles),sampled_training_voiced=sum(p['sustained_voiced'] for p in trace)))
    write_csv(OUT/'endpoint_replays.csv',replays);write_csv(OUT/'run_accounting.csv',accounting)
    write_json(OUT/'checkpoint_index.json',checkpoints)
    write_json(OUT/'audit.json',dict(passed=True,model_reloads=len(checkpoints),generated_sounds=sum(a['episodes'] for a in accounts),
        accounts=accounts,exact_training_rows=12288,exact_evaluation_rows=99,optimizer_updates=12240,
        old_endpoint_actor_critic_weights_exact=True,evaluation_diagnostic_checkpoint_preserve_learner_rng_buffer=True,
        initial_cross_run_env_rng_exception='Only initial unused OS RNG; reset applies seed; full signatures equal from128 onwards',
        source_sha256=digest(__file__),wall_s=time.perf_counter()-tick))
    print('Audit passed: 12288 training samples and99 evaluations reproduce Issue22;12 checkpoints,8 unique sounds',flush=True)


def critics():
    ready();tick=time.perf_counter();candidates=json.loads((OUT/'fixed_candidates.json').read_text())
    actual=np.array([c['reward'] for c in candidates]);actions=np.array([c['action'] for c in candidates]);rows=[];metrics=[];ranks=[];intervals=[];coverage=[]
    for seed in SEEDS:
        folder=folder_for(seed);curve=curve_for(seed);qhistory=read_record(folder/'q_history.json');updates=read_record(folder/'updates.json')
        trace=read_record(folder/'training_trace.json');training_actions=np.array([t['action'] for t in trace])
        # Range fractions: [-1,1] has total width2 in each normalized coordinate.
        distances=np.linalg.norm((actions[:,None,:]-training_actions[None,:,:])/2,axis=2)
        current_points={p['episodes']:p for p in curve};selected=chosen(curve)
        for point in curve:
            n=point['episodes'];q=[p for p in qhistory if p['trial']==n]
            assert [p['candidate_id'] for p in q]==[p['candidate_id'] for p in candidates]
            heads={h:np.array([p[h] for p in q]) for h in ('Q1','Q2','minQ')}
            own=np.array([c['source_seed']==seed for c in candidates]);late=own & np.array([c['source_trial']>=2048 for c in candidates])
            low=actual>=np.quantile(actual,.9)
            for subset,mask in (('all99',np.ones(99,bool)),('own33',own),('own_late17',late),('lowJ10',low)):
                for h,pred in heads.items():metrics.append(dict(seed=seed,trial=n,subset=subset,head=h,**scores(actual[mask],pred[mask])))
            q_actor=point['actor_minQ'];actual_actor=-point['weighted_loss']
            for i,c in enumerate(candidates):
                d=distances[i,:n];exact_count=int(np.count_nonzero(d==0))
                rows.append(dict(seed=seed,trial=n,**{k:v for k,v in c.items() if k not in ('action','physical_controls')},
                    **{h:float(p[i]) for h,p in heads.items()},exact_action_training_occurrences=exact_count))
                if c['source_seed']==seed and c['source_trial'] in {selected[r]['episodes'] for r in selected}|({2944,3072,3840,4096} if seed==2 else set()):
                    coverage.append(dict(seed=seed,trial=n,candidate_id=c['candidate_id'],source_trial=c['source_trial'],
                        nearest_training_range_fraction_l2=float(d.min()) if n else None,exact_occurrences=exact_count,
                        within_0p005=int((d<=.005).sum()),within_0p01=int((d<=.01).sum()),within_0p05=int((d<=.05).sum())))
            for role,chosen_point in selected.items():
                i=next(i for i,c in enumerate(candidates) if c['source_seed']==seed and c['source_trial']==chosen_point['episodes'])
                ranks.append(dict(seed=seed,trial=n,reference_role=role,candidate_id=candidates[i]['candidate_id'],
                    true_reward=actual[i],actor_true_reward=actual_actor,actor_minQ=q_actor,
                    reference_minQ=float(heads['minQ'][i]),reference_Q1=float(heads['Q1'][i]),reference_Q2=float(heads['Q2'][i]),
                    minQ_rank_all99=int(1+(heads['minQ']>heads['minQ'][i]).sum()),
                    minQ_rank_own33=int(1+((heads['minQ']>heads['minQ'][i])&own).sum()),
                    true_reference_minus_actor=actual[i]-actual_actor,
                    predicted_reference_minus_actor=float(heads['minQ'][i]-q_actor),
                    reference_bias=float(heads['minQ'][i]-actual[i])))
            previous=current_points.get(n-128)
            if previous:
                recent=[u for u in updates if n-128<u['trial']<=n];da=np.array(point['actions'][0])-previous['actions'][0]
                summed=np.sum([u['delta_action'] for u in recent],axis=0)
                np.testing.assert_allclose(da,summed,rtol=0,atol=1.2e-7)
                row=dict(seed=seed,start=n-128,end=n,delta_E_total=point['acoustic_error']-previous['acoustic_error'],
                    delta_J=point['weighted_loss']-previous['weighted_loss'],delta_action_l2=float(np.linalg.norm(da)),
                    actor_path_length=float(sum(u['delta_action_l2'] for u in recent)),max_update_l2=max(u['delta_action_l2'] for u in recent),
                    mean_actor_loss=float(np.mean([u['actor_loss'] for u in recent])),mean_critic_loss=float(np.mean([u['critic_loss'] for u in recent])),
                    alpha_start=previous['policy_statistics']['alpha'],alpha_end=point['policy_statistics']['alpha'],
                    entropy_start=previous['policy_statistics']['squashed_entropy_mc'],entropy_end=point['policy_statistics']['squashed_entropy_mc'])
                for i,name in enumerate(NAMES):
                    row.update({f'delta_action_{name}':float(da[i]),f'delta_physical_{name}':float(da[i]*HALF_WIDTH[i]),
                        f'std_start_{name}':previous['policy_statistics']['normalized_action_std_mc'][i],
                        f'std_end_{name}':point['policy_statistics']['normalized_action_std_mc'][i]})
                intervals.append(row)
    from .issue23_data import pack
    pack(OUT/'critic_predictions.json.gz',rows);write_csv(OUT/'critic_metrics.csv',metrics)
    write_csv(OUT/'candidate_ranks.csv',ranks);write_csv(OUT/'candidate_coverage.csv',coverage);write_csv(OUT/'update_intervals.csv',intervals)
    write_json(OUT/'critic_execution.json',dict(generated_sounds=0,learning_updates=0,points=99,predictions=len(rows),
        metrics=len(metrics),wall_s=time.perf_counter()-tick,source_sha256=digest(__file__),
        limitation='Retrospective fixed policy-trajectory set; no global calibration or causal intervention; deterministic Q excludes SAC expected entropy objective'))
    print('Fixed-candidate Critic/time/coverage analysis complete; no additional simulator calls',flush=True)


def local():
    import torch
    torch.set_num_threads(1);m=ready()
    if (OUT/'local_execution.json').exists():
        old=json.loads((OUT/'local_execution.json').read_text())
        for p,h in old['artifact_sha256'].items():assert digest(OUT/p)==h,p
        print('Completed36-sound sensitivity preserved',flush=True);return
    tick=time.perf_counter();points={p['episodes']:p for p in curve_for(2)};env=environment(.75);rows=[];audio=[];gradients=[]
    try:
        for n in LOCAL_ANCHORS:
            center=points[n];action=np.array(center['actions'][0]);path=folder_for(2)/(f'at_{n}.pt' if n!=4096 else 'final.pt')
            agent,meta=load_checkpoint(path,2,m['settings']);signature=learner_signature(agent)
            for h in LOCAL_STEPS:
                for i,name in enumerate(NAMES):
                    pair=[]
                    for direction in (-1,1):
                        candidate=action.copy();candidate[i]+=direction*h
                        assert np.all(abs(candidate)<1),'No boundary clipping in this local diagnostic'
                        metrics,values=episode(env,action=candidate);audio.append(values['audio'])
                        q1,q2=predict_q(agent,[candidate])
                        row=dict(index=len(rows),seed=2,anchor=n,variable=name,step=h,direction=direction,
                            action=json.dumps(candidate.tolist()),physical_controls=json.dumps(values['physical_controls'][0].tolist()),
                            E_total=metrics['acoustic_error'],E_spec=metrics['acoustic_components']['log_spectral'],E_env=metrics['acoustic_components']['envelope'],
                            J=metrics['weighted_loss'],center_E_total=center['acoustic_error'],center_J=center['weighted_loss'],
                            delta_E_total=metrics['acoustic_error']-center['acoustic_error'],delta_J=metrics['weighted_loss']-center['weighted_loss'],
                            Q1=float(q1[0]),Q2=float(q2[0]),minQ=float(min(q1[0],q2[0])),failed=metrics['failed'])
                        rows.append(row);pair.append(row)
                    gradients.append(dict(anchor=n,variable=name,step=h,
                        true_reward_gradient=-(pair[1]['J']-pair[0]['J'])/(2*h),
                        predicted_minQ_gradient=(pair[1]['minQ']-pair[0]['minQ'])/(2*h),
                        analytic_minQ_gradient=center['actor_minQ_gradient'][i],
                        E_gradient=(pair[1]['E_total']-pair[0]['E_total'])/(2*h),
                        max_abs_delta_E=max(abs(p['delta_E_total']) for p in pair),
                        max_abs_delta_J=max(abs(p['delta_J']) for p in pair)))
            assert signature==learner_signature(agent)
        assert len(rows)==36==m['local_sensitivity']['maximum_sounds']
        write_csv(OUT/'local_sensitivity.csv',rows);write_csv(OUT/'local_gradients.csv',gradients)
        np.savez_compressed(OUT/'local_audio.npz',audio=np.asarray(audio),rate=50000,duration_s=.2)
        write_json(OUT/'local_execution.json',dict(generated_sounds=36,account=account(env,time.perf_counter()-tick),
            normalized_steps=list(LOCAL_STEPS),anchors=list(LOCAL_ANCHORS),source_sha256=digest(__file__),
            artifact_sha256={p:digest(OUT/p) for p in ('local_sensitivity.csv','local_gradients.csv','local_audio.npz')}))
    finally:env.close()
    print('Bounded local sensitivity complete:36 sounds; no extra search or learning',flush=True)


def integrity():
    ready();changed=[];deleted=[]
    # The report links this check's own output even on the first invocation.
    if not (OUT/'integrity.json').exists():write_json(OUT/'integrity.json',dict(status='running'))
    with (OUT/'protected_before.csv').open() as f:protected=list(csv.DictReader(f))
    for r in protected:
        path=ROOT/r['path']
        if not path.exists():deleted.append(r['path'])
        elif digest(path)!=r['sha256']:changed.append(r['path'])
    broken=[];links=0
    for p in (OUT/'README.md',OUT/'REPORT.md',OUT/'STAGE_A.md',OUT/'REPRODUCE.md'):
        if not p.exists():continue
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            links+=1
            if not (p.parent/dest.split('#')[0]).exists():broken.append(dict(file=p.name,link=dest))
    for seed in SEEDS:
        folder=folder_for(seed);s=json.loads((folder/'summary.json').read_text())
        for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
    if (OUT/'artifact_checksums.json').exists():
        for p,h in json.loads((OUT/'artifact_checksums.json').read_text()).items():assert digest(OUT/p)==h,p
    if (OUT/'implementation_checksums.json').exists():
        for p,h in json.loads((OUT/'implementation_checksums.json').read_text()).items():assert digest(ROOT/p)==h,p
    if (OUT/'listening.zip').exists():
        import hashlib
        import io
        import zipfile
        from scipy.io import wavfile
        with (OUT/'listening_index.csv').open() as f:index=list(csv.DictReader(f))
        with zipfile.ZipFile(OUT/'listening.zip') as z:
            for row in index:
                rate,wave=wavfile.read(io.BytesIO(z.read(row['archive_entry'])))
                assert rate==50000 and wave.shape==(10000,) and wave.dtype==np.float32
                with np.load(folder_for(int(row['seed']))/f"{row['role']}.npz") as raw:
                    np.testing.assert_array_equal(wave,(raw['audio']*float(row['shared_gain'])).astype(np.float32))
                assert hashlib.sha256(wave.tobytes()).hexdigest()==row['sample_sha256']
    for name in ('audit','stage_a_audit'):
        assert json.loads((OUT/f'{name}.json').read_text())['passed']
    assert json.loads((OUT/'local_execution.json').read_text())['generated_sounds']==36
    passed=not changed and not deleted and not broken
    write_json(OUT/'integrity.json',dict(passed=passed,original_files_checked=len(protected),original_files_changed=changed,
        original_files_deleted=deleted,local_links_checked=links,broken_links=broken,source_sha256=digest(__file__)))
    assert passed,(changed,deleted,broken)
    return passed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('audit','critics','local','integrity'))
    a=p.parse_args();globals()[a.stage]()
