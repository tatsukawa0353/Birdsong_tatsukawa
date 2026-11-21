"""Independent reloads, fixed-budget checks and protected-original verification."""
import argparse
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import re
import time

import numpy as np

from .constant020_env import account,episode
from .issue14 import effective_hyperparameters
from .issue21_data import (ROOT,BASE,OUT,METHODS,WIDTHS,SEEDS,ready,environment,
    folder_for,curve_for,endpoints,target_data)
from .issue21_training import network_geometry
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def audit():
    import torch
    from stable_baselines3 import TD3,SAC
    torch.set_num_threads(1);manifest=ready();tick=time.perf_counter()
    replays=[];sources=[];accounts=[];runs=[]
    target,controls=target_data();assert controls.tolist()==[4.1e7,3.5e7,3.4e6,3e5,4.9e4]
    baseline_versions=json.loads((BASE/'manifest.json').read_text())['versions']
    assert manifest['versions']==baseline_versions
    assert {name:importlib.metadata.version(name) for name in baseline_versions}==baseline_versions
    assert digest(BASE/'manifest.json')==manifest['baseline_manifest_sha256']
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    for method in METHODS:
        for seed in SEEDS:
            baseline_folder=folder_for(method,32,seed)
            baseline=json.loads((baseline_folder/'summary.json').read_text())
            baseline_trace=read_record(baseline_folder/'training_trace.json')
            for width in WIDTHS:
                folder=folder_for(method,width,seed);s=json.loads((folder/'summary.json').read_text())
                curve=curve_for(method,width,seed);trace=read_record(folder/'training_trace.json')
                assert len(trace)==4096 and [p['episodes'] for p in trace]==list(range(1,4097))
                assert [p['episodes'] for p in curve]==list(range(0,4097,128))
                assert s['training']['episodes']==4096 and s['gradient_updates']==4080
                assert s['training']['failures']==0 and s['nonfinite_detected']==0
                assert s['training']['native_steps_completed']==8192000000
                assert s['optimizer_steps']['critic']==4080 and s['optimizer_steps']['actor']==(2040 if method=='TD3' else 4080)
                if method=='SAC':assert s['optimizer_steps']['entropy']==4080
                assert dict(s['effective_hyperparameters'],net_arch=[32,32])==baseline['effective_hyperparameters']
                if width!=32:
                    assert s['source_sha256']==manifest['source_sha256']
                    for name,h in s['artifact_sha256'].items():assert digest(folder/name)==h,name
                    assert s['independent_policy_evaluations']==33 and s['evaluation']['episodes']==35
                assert [p['action'] for p in trace[:16]]==[p['action'] for p in baseline_trace[:16]],'Warmup changed'
                env=environment();job_tick=time.perf_counter()
                try:
                    for p in trace:
                        comp=p.get('acoustic_components',p.get('components'))
                        assert not p['failed'] and np.isfinite(p['acoustic_error']) and p['weighted_loss']==p['acoustic_error']
                        assert p['acoustic_error']==comp['log_spectral']+comp['envelope']
                        c=np.asarray(p['physical_controls']);np.testing.assert_array_equal(env.decode(p['action']),c)
                        assert c[1]==controls[1] and c[4]==controls[4]
                        assert all(low<=value<=high for low,value,high in zip((2.5e7,2.5e6,49000.),c[[0,2,3]],(5e7,4e6,800000.)))
                    for p in curve:
                        assert p['learner_before']==p['learner_after']
                        assert p['envelope_weight']==1. and p['weighted_loss']==p['acoustic_error']
                    chosen=endpoints(method,width,seed)
                    expected_selected=s['best'] if width==32 else s['selected']
                    assert chosen['selected']['episodes']==expected_selected['episodes']
                    for role,expected in chosen.items():
                        model_path=(folder/('agent.zip' if role=='final' else 'best/agent.zip')) if width==32 else folder/('agent.zip' if role=='final' else 'selected_agent.zip')
                        audio_path=(folder/('rollout.npz' if role=='final' else 'best/rollout.npz')) if width==32 else folder/f'{role}.npz'
                        agent={'TD3':TD3,'SAC':SAC}[method].load(model_path,device='cpu')
                        assert all(p.device.type=='cpu' and torch.isfinite(p).all() for p in agent.policy.parameters())
                        geometry=network_geometry(agent,width)
                        if width!=32:assert geometry==s['network_geometry']
                        actual_effective=json.loads(json.dumps(effective_hyperparameters(agent)))
                        for key,value in s['effective_hyperparameters'].items():
                            if key!='initial_entropy_coefficient':assert actual_effective[key]==value,(method,width,key)
                        metrics,values=episode(env,agent=agent)
                        assert metrics['acoustic_error']==expected['acoustic_error'] and metrics['weighted_loss']==expected['weighted_loss']
                        for key in ('log_spectral','envelope'):assert metrics['acoustic_components'][key]==expected['acoustic_components'][key]
                        assert metrics['sustained_voiced']==expected['sustained_voiced']
                        np.testing.assert_array_equal(values['actions'][0],expected['actions'][0])
                        with np.load(audio_path) as z:
                            np.testing.assert_array_equal(z['audio'],values['audio'])
                            np.testing.assert_array_equal(z['physical_controls'],values['physical_controls'])
                        replays.append(dict(method=method,width=width,seed=seed,role=role,
                            E_total=metrics['acoustic_error'],E_spec=metrics['acoustic_components']['log_spectral'],
                            E_env=metrics['acoustic_components']['envelope'],sustained_voiced=metrics['sustained_voiced'],
                            error_difference=0.,waveform_max_difference=0.,model_reloaded=True))
                        sources.append(dict(method=method,width=width,seed=seed,role=role,
                            model_path=str(model_path.relative_to(ROOT)),model_sha256=digest(model_path),
                            audio_path=str(audio_path.relative_to(ROOT)),audio_sha256=digest(audio_path)))
                finally:accounts.append(account(env,time.perf_counter()-job_tick));env.close()
                runs.append(dict(method=method,width=width,seed=seed,training_candidates=4096,
                    new_training_candidates=0 if width==32 else 4096,independent_evaluations=33,
                    source_training_wall_s=s['training']['wall_s'],source_evaluation_sounds=s['evaluation']['episodes'],
                    source_process_cpu_s=s['process_cpu_s'],training_failures=s['training']['failures'],
                    source_summary=str((folder/'summary.json').relative_to(ROOT))))
    assert len(replays)==48 and sum(r['new_training_candidates'] for r in runs)==73728
    write_csv(OUT/'endpoint_replays.csv',replays);write_csv(OUT/'run_accounting.csv',runs);write_json(OUT/'endpoint_sources.json',sources)
    prior=json.loads((OUT/'audit.json').read_text()) if (OUT/'audit.json').exists() else {}
    passes=prior.get('passes',[])+[dict(generated_sounds=48,wall_s=time.perf_counter()-tick,source_sha256=digest(__file__))]
    write_json(OUT/'audit.json',dict(passed=True,runs=24,new_runs=18,baseline_retrained=False,
        new_training_candidates=73728,independent_policy_evaluations=594,training_endpoint_replays=36,
        audit_generated_sounds=sum(a['episodes'] for a in accounts),cumulative_audit_generated_sounds=sum(p['generated_sounds'] for p in passes),
        passes=passes,model_reloads=48,all_actors_twins_and_targets_match_registered_widths=True,
        only_effective_hyperparameter_changed='net_arch',warmup_actions_match=True,
        numerical_failures=0,native_library_sha256=manifest['native_library_sha256'],source_sha256=digest(__file__)))
    integrity()
    print('Audit passed: 18 x4096 new candidates; 48 exact endpoint/model reloads; only net_arch differs',flush=True)


def integrity():
    """No simulation: check originals, source links, CSV and saved artifact hashes."""
    ready();checked=0
    with (OUT/'protected_before.csv').open() as stream:
        for r in csv.DictReader(stream):
            assert digest(ROOT/r['path'])==r['sha256'],r['path'];checked+=1
    broken=[];links=0
    for doc in OUT.glob('*.md'):
        for m in re.finditer(r'\]\(([^)]+)\)',doc.read_text()):
            ref=m[1].split('#')[0]
            if not ref or '://' in ref:continue
            links+=1
            if not (doc.parent/ref).exists():broken.append(dict(document=doc.name,path=ref))
    assert not broken,broken
    for r in json.loads((OUT/'endpoint_sources.json').read_text()) if (OUT/'endpoint_sources.json').exists() else []:
        assert digest(ROOT/r['audio_path'])==r['audio_sha256']
        assert digest(ROOT/r['model_path'])==r['model_sha256']
    images=list((OUT/'figures').glob('*.png'));assert len(images)<=10
    if (OUT/'figure_sources.json').exists():
        figure_sources=json.loads((OUT/'figure_sources.json').read_text())['images']
        assert len(images)==len(figure_sources)==8
        for r in figure_sources:assert digest(OUT/r['path'])==r['sha256']
    csv_rows=0;counts={}
    if (OUT/'new_training_samples.csv.gz').exists():
        with gzip.open(OUT/'new_training_samples.csv.gz','rt') as stream:
            for row in csv.DictReader(stream):
                key=(row['method'],int(row['width']),int(row['seed']));counts.setdefault(key,[]).append(int(row['trial']))
                spec,env,total,reward=[float(row[k]) for k in ('E_spec','E_env','E_total','reward_float32')]
                assert np.isfinite([spec,env,total,reward]).all() and abs(total-spec-env)<1e-14
                assert reward==float(np.float32(-total))
                csv_rows+=1
        assert len(counts)==18 and all(ns==list(range(1,4097)) for ns in counts.values())
    if (OUT/'critic_execution.json').exists():
        diagnosis=json.loads((OUT/'critic_execution.json').read_text())
        assert diagnosis['models']==24 and diagnosis['fixed_candidates']==1005 and diagnosis['generated_sounds']==0
        assert diagnosis['baseline_diagnostic_max_difference']<1e-14
    write_json(OUT/'integrity.json',dict(passed=True,original_files_checked=checked,original_files_changed=0,original_files_deleted=0,
        broken_links=broken,local_links_checked=links,images=len(images),new_training_csv_rows=csv_rows,source_sha256=digest(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--integrity-only',action='store_true')
    a=p.parse_args();integrity() if a.integrity_only else audit()
