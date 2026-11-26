"""Exact endpoint replay, paired parameter-error statistics and native0 summaries."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from .issue27g_data import ROOT,OUT,CONTROLS,LOW,HIGH,METHODS,SEEDS,folder,ready,target_path,target_data,load_checkpoint,indices
from .issue27g_env import CaseBEnv
from .issue27g_native import render,CHANNELS
from .issue27f_prepare import features
from .issue17_artifacts import read_record
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest
from .acoustics import distance

ROLES=('final','selected','training_best','posthoc')
def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def summaries():return [read(p) for p in sorted((OUT/'runs').glob('*/*/*/*/summary.json'))]
def export_trace(name,v):
    path=OUT/f'timeseries/{name}.csv.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s',*CHANNELS]);w.writerows(np.column_stack((np.arange(1,10001)/50000,v['diagnostic_samples'])))
def diagnose(name,v,metadata):
    p=v['diagnostic_samples'];st=v['diagnostic_moments'];assert p.shape==(10000,12) and np.isfinite(p).all();np.testing.assert_array_equal(st[:,49],np.full(20,100000))
    np.testing.assert_array_equal(st[:,:4],v['motion_mean']);np.testing.assert_array_equal(st[:,12:16],v['motion_std'])
    bound=64*np.finfo(float).eps*max(1.,float(abs(p).max()));assert st[:,48].max()<bound
    np.testing.assert_allclose(p[:,7],p[:,4]+p[:,5]+p[:,6],rtol=0,atol=bound)
    f=features(v);last=st[-1];base=dict(name=name,**metadata)
    state=[dict(**base,channel=ch,native_tail10ms_mean=float(last[j]),native_tail10ms_std=float(last[12+j]),detrended_sampled_tail20ms_std=f['motion_detrended_std'][j]) for j,ch in enumerate(CHANNELS[:4])]
    source=dict(**base,**{ch+'_'+key:float(last[j+off]) for j,ch in enumerate(CHANNELS[4:],4) for key,off in (('mean',0),('std',12))},
        decomposition_max_error=float(st[:,48].max()),left_motion_proxy=f['left_motion_periodic_proxy'],right_motion_proxy=f['right_motion_periodic_proxy'],own_peak_voiced=f['eligible'])
    blocks=[dict(**base,block=i+1,channel=ch,mean=float(s[j]),std=float(s[12+j]),minimum=float(s[24+j]),maximum=float(s[36+j])) for i,s in enumerate(st) for j,ch in enumerate(CHANNELS)]
    return state,source,blocks

def physics():
    gates=read(OUT/'preflight/gate.json');peak=max(float(abs(target_data(t)[0]).max()) for t in CONTROLS);gain=1/peak;(OUT/'audio').mkdir(exist_ok=True)
    states=[];sources=[];blocks=[];audio=[];summary=[]
    for t,c in CONTROLS.items():
        v=load(target_path(t));state,source,block=diagnose(t,v,dict(target=t,role='Target'));states+=state;sources.append(source);blocks+=block;export_trace(t,v)
        wavfile.write(OUT/f'audio/{t}_Target.wav',50000,(v['audio']*gain).astype(np.float32));audio.append(dict(name=t+'_Target',wav=f'audio/{t}_Target.wav',raw_npz=str(target_path(t).relative_to(OUT)),gain=gain))
        summary.append(dict(target=t,**{k:v for k,v in gates[t]['features'].items() if k!='motion_detrended_std'},controls=c.tolist(),passed=gates[t]['passed'],right_source_mean=source['source_r_mean'],right_source_std=source['source_r_std']))
    write_csv(OUT/'target_summary.csv',summary);write_csv(OUT/'target_motion.csv',states);write_csv(OUT/'target_sources.csv',sources);write_csv(OUT/'target_native_blocks.csv',blocks);write_csv(OUT/'audio_index.csv',audio)
    # Saved preflight sensitivity only; no new parameter evaluations.
    pair=[load(OUT/f'preflight/T1_right_epsilon_{int(e)}.npz')['audio'] for e in (2e7,5e7)]
    write_json(OUT/'preflight/right_sensitivity_saved_data.json',dict(native_sounds=0,max_abs_difference=float(abs(pair[0]-pair[1]).max()),
        rms_difference_over_T1_peak=float(np.sqrt(np.mean((pair[0]-pair[1])**2))/abs(target_data('T1')[0]).max()),
        tail_rms_difference_over_T1_peak=float(np.sqrt(np.mean((pair[0][-1000:]-pair[1][-1000:])**2))/abs(target_data('T1')[0]).max())))
    write_json(OUT/'physics_audit.json',dict(passed=True,targets_checked=3,all_oracles_exact=True,old_Targets_regenerated=0,right_source_output_nonzero=True,additional_native_sounds=0,audio_gain=gain))

def endpoints():
    torch.set_num_threads(1);manifest=ready();runs=summaries();assert len(runs)==manifest['research_runs'] and all(s['status']=='complete' for s in runs)
    audit=OUT/'audit';audit.mkdir(exist_ok=True)
    if (audit/'ledger.json').exists():raise FileExistsError('Preserve audit ledger; never silently repeat replays')
    ledger=[];out=[];curves=[];account=[];states=[];sources=[];blocks=[];exploration=[];audio=[]
    gain=read(OUT/'physics_audit.json')['audio_gain']
    for s in runs:
        t=s['target'];dim=s['dim'];m=s['method'];seed=s['seed'];p=folder(t,dim,m,seed);target,c=target_data(t);scale=float(abs(target).max());env=CaseBEnv(t,dim)
        try:
            assert s['training']['episodes']==s['transition_checks']==4096 and s['source_sha256']==manifest['source_sha256']
            for name,h in s['artifact_sha256'].items():assert digest(p/name)==h
            curve=read_record(p/'curve.json');trace=read_record(p/'training_trace.json');events=read_record(p/'optimizer_events.json');assert len(trace)==4096 and len(curve)==33
            assert [r['episodes'] for r in curve]==list(range(0,4097,128));assert s['selected']['episodes']==min(curve,key=lambda r:r['weighted_loss'])['episodes'];assert s['posthoc']['episodes']==min(curve,key=lambda r:r['acoustic_error'])['episodes']
            assert s['training_best']['episodes']==min(trace,key=lambda r:r['weighted_loss'])['episodes'];assert len(events)==2*sum(s['optimizer_steps'].values())
            for r in curve:
                n=r['episodes'];assert r['learner_before']==r['learner_after'];assert r['gradient_updates']==(n//128*10 if m=='PPO' else max(0,n-16))
                assert all(x==3e-4 for a in r['optimizer_lrs'].values() for x in a)
                curves.append(dict(target=t,dim=dim,method=m,seed=seed,trial=n,E_spec=r['acoustic_components']['log_spectral'],E_env=r['acoustic_components']['envelope'],E_total=r['acoustic_error'],J=r['weighted_loss'],voiced=r['sustained_voiced'],
                    action=r['actions'][0],physical_controls=r['physical_controls'],optimizer_steps=r['optimizer_steps']))
            for i,r in enumerate(trace,1):
                assert r['episodes']==i and r['stored_reward']==float(np.float32(-r['weighted_loss']))
                a=np.asarray(r['action'],np.float32);stored=np.asarray(r['stored_action'],np.float32)
                np.testing.assert_array_equal(np.clip(stored,-1,1) if m=='PPO' else -1+.5*(stored+1)*2,a);np.testing.assert_array_equal(env.decode(a),r['physical_controls'])
            control=np.array([r['physical_controls'] for r in trace]);assert np.all(control[:,4]==c[4]) and np.all(np.ptp(control[:,indices(dim)],axis=0)>0)
            if dim==3:assert np.all(control[:,1]==c[1])
            exploration.append(dict(target=t,dim=dim,method=m,seed=seed,voiced_training_fraction=float(np.mean([r['sustained_voiced'] for r in trace])),actual_min_controls=control.min(0).tolist(),actual_max_controls=control.max(0).tolist(),
                f0_at_lower_fraction=float(np.mean(control[:,3]==49000)),f0_at_upper_fraction=float(np.mean(control[:,3]==800000)),executed_any_boundary_fraction=float(np.mean(np.any(abs(np.array([r['action'] for r in trace]))==1,axis=1))),
                initial_policy_std=curve[0]['policy_statistics']['normalized_action_std_mc'],final_policy_std=curve[-1]['policy_statistics']['normalized_action_std_mc'],final_policy_statistics=curve[-1]['policy_statistics']))
            account.append(dict(target=t,dim=dim,method=m,seed=seed,training_steps=4096,evaluations=33,training_wall_s=s['training']['wall_s'],evaluation_wall_s=s['evaluation']['wall_s'],process_cpu_s=s['process_cpu_s'],
                failures=s['training']['failures'],nonvoiced=s['training']['nonvoiced_episodes'],optimizer_steps=s['optimizer_steps'],internal_update_counter=s['internal_update_counter']))
            for role in ROLES:
                r=s[role];z=load(p/f'{role}.npz');action=z['actions'][0];physical=env.decode(action)
                if role!='training_best':
                    agent,meta=load_checkpoint(p/f'{role}.pt',t,dim,m,seed);pred=agent.predict(np.array([-1,0,0,0,0,0,0,0],np.float32),deterministic=True)[0]
                    np.testing.assert_array_equal(pred,action);assert meta['trial']==r['episodes']
                else:np.testing.assert_array_equal(action,trace[r['episodes']-1]['action'])
                name=f'{t}_{dim}D_{m}_s{seed}_{role}';ledger.append(dict(name=name,status='started'));write_json(audit/'ledger.json',ledger)
                v,coeff=render(physical,detailed=True)
                for key in ('audio','physical_controls','state_end_blocks','source_end_blocks'):np.testing.assert_array_equal(v[key],z[key])
                comp=distance(target,v['audio'],50000,scale=scale);j=.75*comp['log_spectral']+.25*comp['envelope'];assert comp==r['acoustic_components'] and j==r['weighted_loss'] and comp['total']==r['acoustic_error']
                np.savez_compressed(audit/f'{name}.npz',**v);state,source,block=diagnose(name,v,dict(target=t,dim=dim,method=m,seed=seed,role=role));states+=state;sources.append(source);blocks+=block
                if seed==1 and role in ('final','selected'):export_trace(name,v)
                error=abs(physical-c);normalized=error/(HIGH-LOW)
                row=dict(target=t,dim=dim,method=m,seed=seed,role=role,trial=r['episodes'],E_spec=comp['log_spectral'],E_env=comp['envelope'],E_total=comp['total'],J=j,voiced=r['sustained_voiced'],
                    epsilon_l=float(physical[0]),epsilon_r=float(physical[1]),ps=float(physical[2]),f0_l=float(physical[3]),f0_r=float(physical[4]),
                    common3_error_mean=float(normalized[[0,2,3]].mean()),right_epsilon_fixed=dim==3,left_motion=source['left_motion_proxy'],right_motion=source['right_motion_proxy'])
                for index,key in ((0,'epsilon_l'),(2,'ps'),(3,'f0_l')):row[key+'_abs_error']=float(error[index]);row[key+'_width_normalized_error']=float(normalized[index])
                row['epsilon_r_abs_error']=float(error[1]) if dim==4 else None;row['epsilon_r_width_normalized_error']=float(normalized[1]) if dim==4 else None
                out.append(row);wavfile.write(OUT/f'audio/{name}.wav',50000,(v['audio']*gain).astype(np.float32));audio.append(dict(name=name,wav=f'audio/{name}.wav',raw_npz=f'audit/{name}.npz',gain=gain))
                ledger[-1].update(status='complete',native_steps=2000000,exact=True);write_json(audit/'ledger.json',ledger)
            print(json.dumps(dict(audit_run=[t,dim,m,seed],endpoint_sounds=len(ledger),exact=True)),flush=True)
        finally:env.close()
    write_csv(OUT/'endpoints.csv',out);write_csv(OUT/'learning_curves.csv',curves);write_csv(OUT/'training_accounting.csv',account);write_csv(OUT/'exploration.csv',exploration)
    write_csv(OUT/'endpoint_motion.csv',states);write_csv(OUT/'endpoint_sources.csv',sources);write_csv(OUT/'endpoint_native_blocks.csv',blocks);write_csv(OUT/'audio_index.csv',rows('audio_index.csv')+audio)
    write_json(audit/'endpoints.json',dict(passed=True,research_runs=len(runs),independent_replay_sounds=len(ledger),exact_wave_state_control_loss=True,transition_rows_checked=4096*len(runs),evaluation_invariance_checks=33*len(runs),
        all_right_source_output_nonzero=all(abs(s['source_r_mean'])>0 and abs(s['output_r_mean'])>0 for s in sources)))
    aggregate(out,account)

def aggregate(out,account):
    keys=('E_spec','E_env','E_total','J','voiced','epsilon_l','epsilon_r','ps','f0_l','epsilon_l_abs_error','ps_abs_error','f0_l_abs_error','epsilon_l_width_normalized_error','ps_width_normalized_error','f0_l_width_normalized_error','common3_error_mean')
    means=[];paired=[]
    for t in CONTROLS:
        for dim in (3,4):
            for m in METHODS:
                for role in ROLES:
                    group=[r for r in out if (r['target'],r['dim'],r['method'],r['role'])==(t,dim,m,role)];assert len(group)==3
                    row=dict(target=t,dim=dim,method=m,role=role,seeds=3)
                    for key in keys+(('epsilon_r_abs_error','epsilon_r_width_normalized_error') if dim==4 else ()):
                        v=np.array([r[key] for r in group],float);offset=v-v[0];row[key+'_mean']=float(v[0]+offset.mean());row[key+'_sample_sd']=float(offset.std(ddof=1))
                    means.append(row)
        for m in METHODS:
            for role in ROLES:
                for seed in SEEDS:
                    a=next(r for r in out if (r['target'],r['dim'],r['method'],r['seed'],r['role'])==(t,3,m,seed,role));b=next(r for r in out if (r['target'],r['dim'],r['method'],r['seed'],r['role'])==(t,4,m,seed,role))
                    paired.append(dict(target=t,method=m,seed=seed,role=role,**{key+'_4D_minus_3D':b[key]-a[key] for key in ('E_spec','E_env','E_total','J','common3_error_mean')},epsilon_r_4D_abs_error=b['epsilon_r_abs_error'],epsilon_r_4D_width_normalized_error=b['epsilon_r_width_normalized_error']))
    pm=[]
    for t in CONTROLS:
        for m in METHODS:
            for role in ROLES:
                group=[r for r in paired if (r['target'],r['method'],r['role'])==(t,m,role)];r=dict(target=t,method=m,role=role)
                for k in ('E_spec_4D_minus_3D','E_env_4D_minus_3D','E_total_4D_minus_3D','J_4D_minus_3D','common3_error_mean_4D_minus_3D'):
                    v=np.array([x[k] for x in group]);r[k+'_mean']=float(v.mean());r[k+'_sample_sd']=float(v.std(ddof=1))
                pm.append(r)
    timing=[]
    for t in CONTROLS:
        for d in (3,4):
            for m in METHODS:
                g=[r for r in account if (r['target'],r['dim'],r['method'])==(t,d,m)];r=dict(target=t,dim=d,method=m)
                for k in ('training_wall_s','evaluation_wall_s','process_cpu_s','nonvoiced'):
                    v=np.array([x[k] for x in g]);r[k+'_mean']=float(v.mean());r[k+'_sample_sd']=float(v.std(ddof=1))
                timing.append(r)
    write_csv(OUT/'method_summary.csv',means);write_csv(OUT/'paired_differences.csv',paired);write_csv(OUT/'paired_summary.csv',pm);write_csv(OUT/'timing_summary.csv',timing)

def recovery_prefix():
    checks=[];(OUT/'audit').mkdir(exist_ok=True)
    for p in sorted((OUT/'interrupted_attempt/runs').glob('*/*/*/*/*.pt')):
        r=torch.load(p,map_location='cpu',weights_only=True);t=r['target'];d=r['dim'];m=r['method'];seed=int(p.parent.name.split('_')[-1])
        curve=read_record(folder(t,d,m,seed)/'curve.json');point=next(x for x in curve if x['episodes']==r['trial'])
        np.testing.assert_array_equal(point['actions'][0],r['action']);assert point['weighted_loss']==r['J'] and point['acoustic_error']==r['E_total']
        checks.append(dict(source=str(p.relative_to(OUT)),trial=r['trial'],exact_action_and_losses=True))
    assert len(checks)==54
    write_json(OUT/'audit/recovery_prefix.json',dict(passed=True,interrupted_checkpoints_checked=54,additional_native_sounds=0,checks=checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('physics','endpoints','recovery_prefix'));a=p.parse_args();physics() if a.stage=='physics' else endpoints() if a.stage=='endpoints' else recovery_prefix()
