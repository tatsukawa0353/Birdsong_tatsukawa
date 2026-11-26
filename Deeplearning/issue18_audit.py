"""Independent budget/domain checks and saved-model reloads for Issue18."""
from .result_paths import source_hashes_match
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time
import numpy as np
from .issue18_diagnostics import OUT,BASE
from .issue18_domain_env import DomainEnv
from .issue18_training import hashes
from .issue17 import load_ready
from .issue17_artifacts import read_record
from .issue17_selection import selection_curve
from .issue16 import digest
from .issue15_analysis import write_json
from .constant020_env import episode,account


def replay_job(item):
    method,seed=item
    _,target,controls=load_ready(OUT)
    upper=json.loads((OUT/'domain_decision.json').read_text())['restricted_f0_high']
    env=DomainEnv((0,2,3),target,controls,f0_high=upper)
    folder=OUT/'runs/constant_3d'/method/'lambda1'/f'seed_{seed}'
    summary=json.loads((folder/'summary.json').read_text());result=[];tick=time.perf_counter()
    try:
        for role,rel,expected in (('final','',summary['final']),('selected_best','best',summary['best'])):
            with np.load(folder/rel/'rollout.npz') as z:audio=z['audio'].copy();action=z['actions'][0].copy();physical=z['physical_controls'].copy()
            if method in ('TD3','SAC'):
                import torch
                from stable_baselines3 import TD3,SAC
                agent={'TD3':TD3,'SAC':SAC}[method].load(folder/rel/'agent.zip',device='cpu')
                assert all(p.device.type=='cpu' and torch.isfinite(p).all() for p in agent.policy.parameters())
                metrics,values=episode(env,agent=agent)
                assert np.array_equal(action,values['actions'][0]),'reloaded policy action changed'
            else:metrics,values=episode(env,action=action)
            assert np.array_equal(audio,values['audio']) and np.array_equal(physical,values['physical_controls'])
            assert metrics['acoustic_error']==expected['acoustic_error']
            result.append(dict(method=method,seed=seed,role=role,error_difference=0.,waveform_max_difference=0.,
                model_reloaded=method in ('TD3','SAC'),common_error=metrics['acoustic_error']))
        return dict(replays=result,account=account(env,time.perf_counter()-tick))
    finally:env.close()


def audit():
    start=time.perf_counter();manifest,target,controls=load_ready(OUT);checks=0
    plan=json.loads((OUT/'diagnostics/plan.json').read_text())
    from .issue17 import ROOT
    assert digest(ROOT/'issue18_diagnostics.py')==plan['diagnostic_source_sha256']
    with (OUT/'diagnostics/domain_grid.csv').open() as stream:grid=list(csv.DictReader(stream))
    assert len(grid)==3388 and all(np.isfinite(float(r['E_total'])) for r in grid)
    for row in grid:
        applied=json.loads(row['applied_controls'])
        assert max(abs(applied[i]-float(row[name])) for i,name in ((0,'epsilon_l'),(2,'ps'),(3,'f0_l')))<1e-6
        checks+=1
    voiced=[r for r in grid if float(r['late_ac_rms_over_target_peak'])>.05 and r['failed']=='False']
    maximum=max(float(r['f0_l']) for r in voiced)
    assert len(voiced)==702
    assert min(1e7,float(np.ceil(1.5*maximum/1e5)*1e5))==manifest['settings']['domain_restriction']['restricted_f0_high']
    checks+=4
    assert source_hashes_match(hashes(), manifest['range_adapter_source_sha256'])
    assert digest(OUT/'domain_decision.json')==manifest['domain_decision_sha256']
    oracle_env=DomainEnv((0,2,3),target,controls,f0_high=manifest['settings']['domain_restriction']['restricted_f0_high'])
    oracles=[]
    try:
        for dtype in (np.float64,np.float32):
            metrics,values=episode(oracle_env,action=oracle_env.encode(controls,dtype=dtype))
            assert not metrics['failed'] and metrics['sustained_voiced']
            assert metrics['acoustic_error']<1e-4
            oracles.append(dict(action_dtype=np.dtype(dtype).name,common_error=metrics['acoustic_error'],
                physical_controls=values['physical_controls'][0].tolist(),
                relative_waveform_max_difference=float(abs(values['audio']-target).max()/abs(target).max())))
        assert oracle_env.decode(np.array([0,0,-1]))[3]==49000
        assert oracle_env.decode(np.array([0,0,1]))[3]==800000
        checks+=6
    finally:oracle_env.close()
    write_json(OUT/'diagnostics/restricted_mapping_oracles.json',oracles)
    summaries=[];restart_checkpoints=0
    for method in ('TD3','SAC','DE','random_search'):
        for seed in (0,1,2):
            folder=OUT/'runs/constant_3d'/method/'lambda1'/f'seed_{seed}'
            original=BASE/'runs/constant_3d'/method/'lambda1'/f'seed_{seed}'
            s=json.loads((folder/'summary.json').read_text());old=json.loads((original/'summary.json').read_text())
            points=selection_curve(read_record(folder/'curve.json'),1.)
            assert s['training']['episodes']==4096 and s['training']['native_steps_completed']==8192000000;checks+=2
            assert s['nonfinite_detected']==0 and s['training']['failures']==0;checks+=2
            expected=list(range(0,4097,128)) if method in ('TD3','SAC') else list(range(1,4097))
            assert [p['episodes'] for p in points]==expected;checks+=1
            assert s['best']['acoustic_error']==min(p['acoustic_error'] for p in points);checks+=1
            assert s['final']['episodes']==4096 and s['final']['acoustic_error']==points[-1]['acoustic_error'];checks+=1
            training=read_record(folder/'training_trace.json') if method in ('TD3','SAC') else points
            upper=manifest['settings']['domain_restriction']['restricted_f0_high']
            for p in training:
                assert 49000.-1e-6<=p['physical_controls'][3]<=upper+1e-6
                assert np.isfinite(p['acoustic_error']) and np.isfinite(p['weighted_loss']);checks+=2
            if method in ('TD3','SAC'):
                previous=read_record(original/'training_trace.json')
                assert s['effective_hyperparameters']==old['effective_hyperparameters'];checks+=1
                assert s['gradient_updates']==4080 and s['evaluation']['episodes']==33;checks+=2
                assert points[0]['learner_before']['policy_sha256']==old['initial']['learner_before']['policy_sha256'];checks+=1
                assert [p['action'] for p in training[:16]]==[p['action'] for p in previous[:16]];checks+=1
                for p in points:
                    assert p['learner_before']==p['learner_after'];checks+=1
            else:
                previous=read_record(original/'curve.json')
                length=4096 if method=='random_search' else 8
                assert [p['actions'] for p in points[:length]]==[p['actions'] for p in previous[:length]];checks+=1
                assert s['optimizer_nfev']==4096 and s['gradient_updates']==0;checks+=2
            summaries.append(s)
            interrupted=OUT/'interrupted_attempt_1'/folder.relative_to(OUT)
            for checkpoint in ((256,512,1024,2048) if interrupted.exists() else ()):
                old_point=json.loads((interrupted/'checkpoints'/str(checkpoint)/'metrics.json').read_text())
                new_point=next(p for p in points if p['episodes']==checkpoint)
                for field in ('actions','physical_controls','acoustic_error','weighted_loss'):
                    assert old_point[field]==new_point[field],('restart changed',method,seed,checkpoint,field)
                    checks+=1
                restart_checkpoints+=1
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(replay_job,[(m,s) for m in ('TD3','SAC','DE','random_search') for s in (0,1,2)]))
    replays=[r for result in results for r in result['replays']]
    result=dict(passed=True,checks=checks,training_candidates=sum(s['training']['episodes'] for s in summaries),
        independent_training_evaluation_sounds=sum(s['evaluation']['episodes'] for s in summaries),
        audit_generated_sounds=sum(r['account']['episodes'] for r in results)+2,
        model_reloads=sum(r['model_reloaded'] for r in replays),independent_replays=replays,
        original_learner_settings_match=True,paired_initial_policies_and_warmup_actions_match=True,
        interrupted_restart_matching_checkpoints=restart_checkpoints,
        diagnostic_range_frozen_before_training=True,wall_s=time.perf_counter()-start)
    write_json(OUT/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='independent_replays'}),flush=True)


if __name__=='__main__':audit()
