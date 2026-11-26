"""Saved-data diagnostics and exactly36 independently rendered endpoint audits."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from .issue27f_data import ROOT,OUT,OLD,TARGET,METHODS,SEEDS,folder,ready,load_checkpoint,target_data
from .issue27f_prepare import CONFIG,features
from .issue27f_native import render,CHANNELS
from .issue27f_env import CaseBEnv
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue17_artifacts import read_record
from .issue16 import digest
from .acoustics import distance

ROLES=('final','selected','training_best','posthoc')
def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def records():return read(OUT/'preflight/candidates.json')
def summaries():return [read(folder(m,s)/'summary.json') for m in METHODS for s in SEEDS]

def diagnostic(name,z,metadata):
    points=z['diagnostic_samples'];stats=z['diagnostic_moments'];last=stats[-1]
    assert points.shape==(10000,12) and np.isfinite(points).all()
    np.testing.assert_array_equal(stats[:,49],np.full(20,100000))
    np.testing.assert_array_equal(z['motion_mean'],stats[:,:4]);np.testing.assert_array_equal(z['motion_std'],stats[:,12:16])
    bound=np.finfo(float).eps*64*max(1.,float(abs(points).max()))
    assert stats[:,48].max()<bound
    np.testing.assert_allclose(points[:,7],points[:,4]+points[:,5]+points[:,6],rtol=0,atol=bound)
    tail=points[-1000:];base=dict(name=name,**metadata)
    f=features(z)
    motion=[dict(**base,channel=channel,mean_native=float(last[j]),std_native=float(last[12+j]),minimum=float(last[24+j]),maximum=float(last[36+j]),
        detrended_std_last20ms=f['motion_detrended_std'][j],sample_end=float(points[-1,j])) for j,channel in enumerate(CHANNELS[:4])]
    source=dict(**base,**{channel+'_'+key:float(last[offset+j]) for j,channel in enumerate(CHANNELS[4:],4) for key,offset in (('mean',0),('std',12))},
        decomposition_max_error=float(stats[:,48].max()),left_motion=f['left_motion_periodic_proxy'],right_motion=f['right_motion_periodic_proxy'])
    damping=[]
    for side,index in (('left',0),('right',2)):
        nonlinear=2e8*tail[:,index]**2;effective=2e4+nonlinear-.01*(z['controls'][2]-tail[:,7])
        damping.append(dict(**base,side=side,x_tail_mean=float(tail[:,index].mean()),x_tail_std=float(tail[:,index].std()),
            nonlinear_damping_mean=float(nonlinear.mean()),nonlinear_damping_min=float(nonlinear.min()),nonlinear_damping_max=float(nonlinear.max()),
            effective_velocity_coefficient_mean=float(effective.mean()),effective_velocity_coefficient_min=float(effective.min()),effective_velocity_coefficient_max=float(effective.max()),negative_fraction=float(np.mean(effective<0))))
    dest=OUT/f'timeseries/{name}.csv.gz';dest.parent.mkdir(exist_ok=True)
    with gzip.open(dest,'wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['time_s',*CHANNELS]);writer.writerows(np.column_stack((np.arange(1,10001)/50000,points)))
    return motion,source,damping

def physics():
    (OUT/'audio').mkdir(exist_ok=True);target,controls=target_data();gain=1/float(abs(target).max());summary=[];motions=[];sources=[];damping=[];audio=[]
    for r in records():
        name=r['candidate'];z=load(OUT/r['raw_npz']);f=features(z)
        assert f['eligible']==r['eligible'] and f['late_ac_ratio']==r['late_ac_ratio']
        mt,so,da=diagnostic(name,z,dict(group=r['group'],epsilon_l=r['epsilon_l'],ps=r['ps'],f0_l=r['f0_l'],f0_r=r['f0_r']))
        motions.extend(mt);sources.append(so);damping.extend(da);summary.append({k:v for k,v in r.items() if k!='native_coefficients'})
        path=OUT/f'audio/{name}.wav';wavfile.write(path,50000,(z['audio']*gain).astype(np.float32));audio.append(dict(name=name,wav=str(path.relative_to(OUT)),raw_npz=r['raw_npz'],gain=gain))
    # Standalone Target files are explicit, byte-identical with its candidate wave.
    wavfile.write(OUT/'audio/Target.wav',50000,(target*gain).astype(np.float32));audio.append(dict(name='Target',wav='audio/Target.wav',raw_npz='preflight/target.npz',gain=gain))
    write_csv(OUT/'candidate_summary.csv',summary);write_csv(OUT/'tail_motion.csv',motions);write_csv(OUT/'source_statistics.csv',sources);write_csv(OUT/'damping_diagnostics.csv',damping);write_csv(OUT/'audio_index.csv',audio)
    # Old9 are only read, no physics generation or rewriting.
    old_damping=[];old_sources=[];old_motion=[]
    for r in read(OLD/'preflight/candidates.json'):
        z=load(OLD/r['raw_npz']);mt,so,da=diagnostic('old_'+r['candidate'],z,dict(group=1,epsilon_l=r['epsilon_l'],ps=r['ps'],f0_l=3e5,f0_r=1e7))
        old_motion.extend(mt);old_sources.append(so);old_damping.extend(da)
    write_csv(OUT/'old_tail_motion_readonly.csv',old_motion);write_csv(OUT/'old_source_statistics_readonly.csv',old_sources);write_csv(OUT/'old_damping_readonly.csv',old_damping)
    write_json(OUT/'physics_audit.json',dict(passed=True,new_candidates=18,sustained=sum(r['eligible'] for r in summary),old_sounds_read_only=9,
        additional_native_sounds=0,all_right_source_nonzero=all(abs(s['source_r_mean'])>0 and s['source_r_std']>0 for s in sources),
        maximum_decomposition_error=max(s['decomposition_max_error'] for s in sources),target_controls=controls.tolist(),common_audio_gain=gain))

def endpoints():
    torch.set_num_threads(1);manifest=ready();runs=summaries();assert len(runs)==9 and all(s['status']=='complete' for s in runs)
    audit=OUT/'audit';audit.mkdir(exist_ok=True)
    if (audit/'endpoints.json').exists():raise FileExistsError('Preserve completed36 endpoint replays')
    if (audit/'ledger.json').exists():raise FileExistsError('Preserve partial endpoint replay ledger; do not silently repeat')
    target,_=target_data();scale=float(abs(target).max());rows=[];curve_rows=[];motion=[];sources=[];damping=[];ledger=[];audio=[];accounting=[]
    env=CaseBEnv()
    try:
        for s in runs:
            m=s['method'];seed=s['seed'];p=folder(m,seed);curve=read_record(p/'curve.json');trace=read_record(p/'training_trace.json');events=read_record(p/'optimizer_events.json')
            assert s['source_sha256']==manifest['source_sha256'] and s['training']['episodes']==4096 and s['transition_checks']==4096
            assert len(curve)==33 and [r['episodes'] for r in curve]==list(range(0,4097,128)) and len(trace)==4096
            assert s['selected']['episodes']==min(curve,key=lambda r:r['weighted_loss'])['episodes']
            assert s['posthoc']['episodes']==min(curve,key=lambda r:r['acoustic_error'])['episodes']
            assert s['training_best']['episodes']==min(trace,key=lambda r:r['weighted_loss'])['episodes']
            for r in curve:
                assert r['learner_before']==r['learner_after']
                n=r['episodes'];expected=(max(0,n-16) if m!='PPO' else n//128*10)
                assert r['gradient_updates']==expected
                assert all(x==.0003 for values in r['optimizer_lrs'].values() for x in values)
                curve_rows.append(dict(method=m,seed=seed,trial=n,E_spec=r['acoustic_components']['log_spectral'],E_env=r['acoustic_components']['envelope'],E_total=r['acoustic_error'],J=r['weighted_loss'],voiced=r['sustained_voiced'],updates=r['gradient_updates']))
            for i,t in enumerate(trace,1):
                assert t['episodes']==i and t['stored_reward']==float(np.float32(-t['weighted_loss']))
                action=np.asarray(t['action'],np.float32);stored=np.asarray(t['stored_action'],np.float32)
                if m=='PPO':np.testing.assert_array_equal(np.clip(stored,-1,1),action)
                else:np.testing.assert_array_equal(2*((stored+1)/2)-1,action)
                np.testing.assert_array_equal(env.decode(action),t['physical_controls'])
            assert len(events)==2*sum(s['optimizer_steps'].values())
            for name,h in s['artifact_sha256'].items():assert digest(p/name)==h
            accounting.append(dict(method=m,seed=seed,training_steps=4096,evaluations=33,training_wall_s=s['training']['wall_s'],evaluation_wall_s=s['evaluation']['wall_s'],process_cpu_s=s['process_cpu_s'],
                nonvoiced=s['training']['nonvoiced_episodes'],failures=s['training']['failures'],**{k+'_steps':v for k,v in s['optimizer_steps'].items()}))
            for role in ROLES:
                r=s[role];z=load(p/f'{role}.npz');action=z['actions'][0];physical=env.decode(action)
                if role!='training_best':
                    agent,meta=load_checkpoint(p/f'{role}.pt',m,seed,CONFIG)
                    predicted=agent.predict(np.array([-1,0,0,0,0,0,0,0],np.float32),deterministic=True)[0]
                    np.testing.assert_array_equal(predicted,action);assert meta['trial']==r['episodes']
                else:np.testing.assert_array_equal(action,trace[r['episodes']-1]['action'])
                np.testing.assert_array_equal(np.tile(physical,(20,1)),z['physical_controls'])
                name=f'{m}_s{seed}_{role}';entry=dict(name=name,status='started');ledger.append(entry);write_json(audit/'ledger.json',ledger)
                values,coeff=render(physical,'B',detailed=True)
                for key in ('audio','physical_controls','state_end_blocks','source_end_blocks'):np.testing.assert_array_equal(values[key],z[key])
                comp=distance(target,values['audio'],50000,scale=scale);j=.75*comp['log_spectral']+.25*comp['envelope']
                assert comp==r['acoustic_components'] and j==r['weighted_loss'] and comp['total']==r['acoustic_error']
                np.savez_compressed(audit/f'{name}.npz',**values)
                mt,so,da=diagnostic(name,values,dict(method=m,seed=seed,role=role));motion.extend(mt);sources.append(so);damping.extend(da)
                row=dict(method=m,seed=seed,role=role,trial=r['episodes'],E_spec=comp['log_spectral'],E_env=comp['envelope'],E_total=comp['total'],J=j,
                    voiced=r['sustained_voiced'],epsilon_l=float(physical[0]),ps=float(physical[2]),f0_l=float(physical[3]),f0_r=float(physical[4]),left_motion=so['left_motion'],right_motion=so['right_motion'],source_r_mean=so['source_r_mean'],source_r_std=so['source_r_std'])
                rows.append(row);wavfile.write(OUT/f'audio/{name}.wav',50000,(values['audio']/scale).astype(np.float32));audio.append(dict(name=name,wav=f'audio/{name}.wav',raw_npz=f'audit/{name}.npz',gain=1/scale))
                entry.update(status='complete',native_steps=2000000,exact=True);write_json(audit/'ledger.json',ledger)
                print(json.dumps(dict(endpoint=name,E=comp['total'],exact=True)),flush=True)
    finally:env.close()
    means=[]
    for m in METHODS:
        for role in ROLES:
            subset=[r for r in rows if r['method']==m and r['role']==role];out=dict(method=m,role=role,seeds=3)
            for key in ('E_spec','E_env','E_total','J','epsilon_l','ps','f0_l','voiced'):
                v=np.array([r[key] for r in subset],float);out[key+'_mean']=float(v.mean());out[key+'_sample_sd']=float(v.std(ddof=1))
            means.append(out)
    write_csv(OUT/'endpoints.csv',rows);write_csv(OUT/'method_summary.csv',means);write_csv(OUT/'learning_curves.csv',curve_rows);write_csv(OUT/'training_accounting.csv',accounting)
    write_csv(OUT/'endpoint_motion.csv',motion);write_csv(OUT/'endpoint_sources.csv',sources);write_csv(OUT/'endpoint_damping.csv',damping)
    with (OUT/'audio_index.csv').open() as f:existing=list(csv.DictReader(f))
    write_csv(OUT/'audio_index.csv',existing+audio)
    write_json(audit/'endpoints.json',dict(passed=True,endpoints=36,additional_native_sounds=36,exact_wave_state_controls_and_loss=True,actor_checkpoints_checked=27,
        training_best_candidates_checked=9,transition_rows_checked=36864,evaluation_invariance_checks=297,all_right_contributions_nonzero=all(abs(s['source_r_mean'])>0 and s['source_r_std']>0 for s in sources)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('physics','endpoints'));a=parser.parse_args();physics() if a.stage=='physics' else endpoints()
