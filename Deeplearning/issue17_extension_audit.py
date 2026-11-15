"""Independent selection/reuse/new-DE audit; original54 runs remain immutable."""
import json
from pathlib import Path
import time
import numpy as np

from .acoustics import distance
from .constant020_env import Constant020Env,INDICES,account,episode
from .issue15_analysis import write_json
from .issue16 import digest
from .issue17 import DEST,ROOT,load_ready
from .issue17_artifacts import read_record
from .issue17_direct_weights import WORK,EXTENSION,current_sources
from .issue17_extension_data import load_runs


def verify_preserved(evidence):
    mapping=json.loads((evidence/'protected_before.json').read_text())
    changed=[p for p,h in mapping.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    if changed:raise RuntimeError(f'Protected preexisting artifacts changed:{changed[:4]}')
    return len(mapping)


def prefix_minimum_indices(values):
    values=np.asarray(values)
    strict=values<np.r_[np.inf,np.minimum.accumulate(values)[:-1]]
    return np.maximum.accumulate(np.where(strict,np.arange(len(values)),0))


def audit(stage=WORK):
    stage=Path(stage);evidence=stage if (stage/'manifest.json').exists() and stage!=DEST else EXTENSION
    start=time.perf_counter();checks=0;errors=[];replays=[];accounts=[]
    manifest=json.loads((evidence/'manifest.json').read_text())
    if manifest['numerical_sources']!=current_sources():raise RuntimeError('Extension numerical source changed')
    base,target,controls=load_ready(DEST);runs=load_runs(stage)
    def check(value,code,path):
        nonlocal checks
        checks+=1
        if not value:errors.append(dict(code=code,path=str(path)))
    check(len(runs)==72,'72 profiles',stage)
    for (c,m,seed,w),data in runs.items():
        folder=data['folder'];curve=data['curve'];summary=data['summary']
        expected=list(range(0,4097,128)) if m in ('TD3','SAC') else list(range(1,4097))
        check([p['episodes'] for p in curve]==expected,'all observations retained',folder)
        common=np.array([p['acoustic_error'] for p in curve])
        objective=np.array([1e6 if p['failed'] else p['acoustic_components']['log_spectral']+w*p['acoustic_components']['envelope'] for p in curve])
        ji,ei=prefix_minimum_indices(objective),prefix_minimum_indices(common)
        check(np.array_equal([p['objective_selected_index'] for p in curve],ji),'minJ first-tie archive',folder)
        check(np.array_equal([p['common_posthoc_index'] for p in curve],ei),'separate posthoc common minimum',folder)
        check(np.array_equal([p['objective_selected_common_error'] for p in curve],common[ji]),'common error of J-selected incumbent',folder)
        check(np.array_equal([p['common_posthoc_error'] for p in curve],common[ei]),'posthoc minimum values',folder)
        check(np.isfinite(objective).all() and np.isfinite(common).all(),'finite objective and common error',folder)
        if m in ('TD3','SAC') or w==1:continue
        original=read_record(DEST/'runs'/c/m/'lambda1'/f'seed_{seed}'/'curve.json')
        if m=='DE':
            check(summary['training']['episodes']==4096 and summary['optimizer_nfev']==4096,'exact newDE budget',folder)
            check(summary['training']['native_steps_completed']==8192000000,'newDE native steps',folder)
            check(summary['gradient_updates']==0 and summary['nonfinite_detected']==0,'no RL updates',folder)
            for a,b in zip(curve[:8],original[:8]):
                check(a['actions']==b['actions'] and a['physical_controls']==b['physical_controls'] and
                      a['acoustic_components']==b['acoustic_components'],'paired initial DE population',folder)
            selected=json.loads((folder/'de_selected_incumbent.json').read_text())
            check(abs(selected['weighted_loss']-objective.min())<1e-12,'solver optimizes J0.5',folder)
            stages=[('initial',curve[0]),('final',curve[-1]),('best_objective',curve[ji[-1]]),
                    ('best_common_posthoc',curve[ei[-1]]),('solver_returned',None)]
        else:
            reuse=json.loads((folder/'source_reuse.json').read_text())
            source=DEST/reuse['source']
            check(reuse['candidates_reused']==4096 and reuse['candidates_regenerated']==0,'entire Random pool reused',folder)
            check(summary['training']['native_steps_attempted']==0 and summary['new_candidate_sounds']==0,'no Random candidate simulation',folder)
            check(digest(source/'curve.json.gz')==reuse['source_curve_sha256'],'source Random curve unchanged',folder)
            for a,b in zip(curve,original):
                check(a['actions']==b['actions'] and a['physical_controls']==b['physical_controls'] and
                      a['acoustic_components']==b['acoustic_components'],'same Random pool and order',folder)
            stages=[('best_objective',curve[ji[-1]])]
            # Reused snapshots remain byte-identical to protected lambda1 data.
            check(digest(folder/'rollout.npz')==digest(source/'rollout.npz'),'Random final waveform reused',folder)
        check(digest(folder/'curve.json.gz')==summary['curve_sha256'],'new curve checksum',folder)
        env=Constant020Env(INDICES[c],target,controls,envelope_weight=.5);tick=time.perf_counter()
        try:
            for name,record in stages:
                destination=folder if name=='final' else folder/name
                with np.load(destination/'rollout.npz') as z:
                    audio,actions,physical=z['audio'].copy(),z['actions'].copy(),z['physical_controls'].copy()
                    check(np.array_equal(z['target'],target),'same target',destination)
                    check(int(z['rate'])==50000 and audio.shape==(10000,),'0.20s at50kHz',destination)
                check(np.array_equal(physical,np.tile(physical[0],(20,1))),'constant controls',destination)
                comp=distance(target,audio,50000)
                weighted=comp['log_spectral']+.5*comp['envelope']
                if record is not None:
                    check(abs(comp['total']-record['acoustic_error'])<1e-12,'saved common error',destination)
                    check(abs(weighted-record['weighted_loss'])<1e-12,'saved weighted objective',destination)
                metrics,values=episode(env,action=actions[0])
                check(np.array_equal(audio,values['audio']) and np.array_equal(physical,values['physical_controls']),
                      'independent native replay exact',destination)
                check(metrics['acoustic_error']==comp['total'],'replayed common error',destination)
                replays.append(dict(condition=c,method=m,seed=seed,stage=name,common_error=comp['total'],weighted_loss=weighted,
                                    waveform_max_difference=float(abs(audio-values['audio']).max())))
        finally:
            accounts.append(dict(condition=c,method=m,seed=seed,**account(env,time.perf_counter()-tick)));env.close()
    protected=verify_preserved(evidence)
    result=dict(passed=not errors,checks=checks,errors=errors,profiles=72,new_DE_runs=9,derived_Random_runs=9,
        RL_retrained=False,Random_candidates_regenerated=0,protected_files_verified=protected,
        baseline_independent_audit_sha256=digest(DEST/'audit.json'),baseline_audit_checks=json.loads((DEST/'audit.json').read_text())['checks'],
        baseline_validation='Prior independent audit plus current byte-for-byte preservation; no repeated RL training or replay',
        independent_replays=replays,audit_generation=accounts,
        audit_generated_sounds=sum(a['episodes'] for a in accounts),
        audit_native_steps=sum(a['native_steps_attempted'] for a in accounts),wall_s=time.perf_counter()-start)
    destination=evidence/'audit.json' if evidence==WORK else Path('/tmp/issue17-direct-weight-audit.json')
    write_json(destination,result)
    if errors:raise RuntimeError(f'Extension audit failed:{errors[:4]}')
    print(json.dumps({k:result[k] for k in ('passed','checks','profiles','audit_generated_sounds','protected_files_verified')}),flush=True)
    return result
