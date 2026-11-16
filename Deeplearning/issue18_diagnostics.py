"""Physical-only domain survey and saved-history analysis; never writes Issue17."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import numpy as np

from .constant020_env import Constant020Env,episode,account
from .issue17 import ROOT,DEST as BASE,load_ready
from .issue17_artifacts import read_record
from .issue17_extension_data import load_runs,write_csv
from .issue15_analysis import write_json
from .issue16 import digest
from .model import LOW,HIGH

OUT=ROOT/'results/phase1_3d_domain_diagnostics'
_ENV=None


def worker_init():
    global _ENV
    with np.load(OUT/'diagnostics/target.npz') as z:target,controls=z['target'],z['controls']
    _ENV=Constant020Env((0,2,3),target,controls)


def measure(item):
    index,controls=item
    metrics,values=episode(_ENV,action=_ENV.encode(controls))
    return dict(index=index,epsilon_l=float(controls[0]),ps=float(controls[2]),f0_l=float(controls[3]),
        applied_controls=values['physical_controls'][0].tolist(),E_total=metrics['acoustic_error'],
        E_spec=metrics['acoustic_components']['log_spectral'],E_env=metrics['acoustic_components']['envelope'],
        sustained_voiced=metrics['sustained_voiced'],late_ac_rms_over_target_peak=metrics['late_ac_rms_over_target_peak'],
        failed=metrics['failed'],peak_over_target_peak=float(abs(values['audio']).max()/_ENV.loss_scale))


def survey(workers=9):
    if OUT.exists():raise FileExistsError('Existing Issue18 output preserved')
    base,target,controls=load_ready(BASE)
    (OUT/'diagnostics').mkdir(parents=True)
    shutil.copy2(BASE/'diagnostics/target.npz',OUT/'diagnostics/target.npz')
    f0=np.unique(np.r_[np.linspace(LOW[3],HIGH[3],41),np.geomspace(LOW[3],HIGH[3],41),
                      np.linspace(LOW[3],1e6,41),controls[3]])
    eps=np.unique(np.r_[np.linspace(LOW[0],HIGH[0],9),controls[0]])
    ps=np.unique(np.r_[np.linspace(LOW[2],HIGH[2],9),controls[2]])
    pairs={(float(e),float(controls[2])) for e in eps}|{(float(controls[0]),float(p)) for p in ps}
    pairs|={(float(e),float(p)) for e in np.linspace(LOW[0],HIGH[0],3) for p in np.linspace(LOW[2],HIGH[2],3)}
    candidates=[]
    for e,p in sorted(pairs):
        for f in f0:
            c=controls.copy();c[[0,2,3]]=[e,p,f];candidates.append(c)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='physical voicing domain, no optimization',
        base_manifest_sha256=digest(BASE/'manifest.json'),target_sha256=digest(OUT/'diagnostics/target.npz'),
        source_sha256=base['source_sha256'],diagnostic_source_sha256=digest(__file__),
        f0_grid=f0.tolist(),epsilon_grid=eps.tolist(),ps_grid=ps.tolist(),joint_pairs=sorted(pairs),
        candidates=len(candidates),proxy='std(last1000samples)/target_peak > .05, no numerical failure',
        range_decision_rule='Keep epsilon_l and ps full ranges and f0_l lower49000. Upper=min(1e7,ceil(1.5*largest_voiced_f0/1e5)*1e5). Use proxy only, never E_total or proximity to target.',
        limitations='Cross-sections and nine joint anchor pairs, not a complete 3D volume survey')
    write_json(OUT/'diagnostics/plan.json',plan)
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
        futures=[pool.submit(measure,(i,c)) for i,c in enumerate(candidates)]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%200==0:print(json.dumps(dict(stage='physical_survey',completed=len(rows),total=len(candidates))),flush=True)
    rows.sort(key=lambda r:r['index']);write_csv(OUT/'diagnostics/domain_grid.csv',rows)
    voiced=[r for r in rows if r['sustained_voiced']]
    if not voiced:raise RuntimeError('No voiced survey points; do not invent a restriction')
    maximum=max(r['f0_l'] for r in voiced);upper=min(float(HIGH[3]),float(np.ceil(1.5*maximum/1e5)*1e5))
    assert upper>controls[3] and upper>LOW[3]
    decision=dict(frozen_before_training_utc=datetime.now(timezone.utc).isoformat(),
        rule=plan['range_decision_rule'],largest_voiced_f0=maximum,restricted_f0_low=float(LOW[3]),
        restricted_f0_high=upper,original_f0_high=float(HIGH[3]),epsilon_l_range=[float(LOW[0]),float(HIGH[0])],
        ps_range=[float(LOW[2]),float(HIGH[2])],target_f0=float(controls[3]),
        target_fraction_restricted=float((controls[3]-LOW[3])/(upper-LOW[3])),
        survey_voiced_points=len(voiced),survey_points=len(rows),voiced_fraction_on_this_nonuniform_grid=len(voiced)/len(rows),
        note='Grid is nonuniform; this fraction is NOT a physical-volume probability. Upper includes 50% safety margin beyond all observed voiced anchors.',
        same_target_and_reward=True,lambda_for_followup=1.,budget=4096,seeds=[0,1,2],methods=['TD3','SAC','DE','random_search'])
    write_json(OUT/'domain_decision.json',decision)
    write_json(OUT/'diagnostics/execution.json',dict(candidates=len(rows),native_steps=len(rows)*2000000,
        wall_s=time.perf_counter()-start,workers=workers,numerical_failures=sum(r['failed'] for r in rows)))
    worker_init();representatives=[]
    from .issue17 import save_snapshot
    for f in (49000.,150000.,300000.,500000.,1e6,5e6,1e7):
        c=controls.copy();c[3]=f;metrics,values=episode(_ENV,action=_ENV.encode(c))
        folder=OUT/'diagnostics/representative_audio'/f'f0_{int(f)}'
        save_snapshot(folder,values,target,controls);write_json(folder/'metrics.json',metrics)
        representatives.append(dict(f0_l=f,**metrics))
    write_json(OUT/'diagnostics/representatives.json',representatives)
    _ENV.close()
    print(json.dumps(dict(stage='domain_decided',**decision)),flush=True)


def reanalyze():
    runs=load_runs(BASE);summaries=[];distributions=[]
    for (c,m,seed,w),data in runs.items():
        if c!='constant_3d':continue
        folder=data['folder'];curve=data['curve']
        training=read_record(folder/'training_trace.json') if m in ('TD3','SAC') else curve
        for role,points in (('training_samples',training),('independent_policy_evaluations',curve)) if m in ('TD3','SAC') else [('actual_candidates',curve)]:
            errors=np.array([p['acoustic_error'] for p in points]);physical=np.array([p['physical_controls'] for p in points])
            j=np.array([p['weighted_loss'] for p in points]);best=int(j.argmin());voiced=[p for p in points if p['sustained_voiced']]
            summaries.append(dict(method=m,seed=seed,envelope_weight=w,role=role,observations=len(points),
                voiced_count=len(voiced),voiced_fraction=len(voiced)/len(points),min_J_trial=points[best]['episodes'],
                min_J_common_error=float(errors[best]),min_common_error=float(errors.min()),
                min_J_sustained_voiced=points[best]['sustained_voiced'],
                f0_median=float(np.median(physical[:,3])),f0_upper95=float(np.quantile(physical[:,3],.95)),
                f0_upper_boundary_fraction=float(np.mean(physical[:,3]>=LOW[3]+.95*(HIGH[3]-LOW[3]))),
                final_error=float(errors[-1]),final_sustained_voiced=points[-1]['sustained_voiced'],
                first_voiced_trial=voiced[0]['episodes'] if voiced else None))
            for p in points:
                comp=p.get('acoustic_components',p.get('components'))
                distributions.append(dict(method=m,seed=seed,envelope_weight=w,role=role,trial=p['episodes'],
                    epsilon_l=p['physical_controls'][0],ps=p['physical_controls'][2],f0_l=p['physical_controls'][3],
                    E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],
                    sustained_voiced=p['sustained_voiced'],failed=p['failed']))
    write_csv(OUT/'diagnostics/existing_history_summary.csv',summaries)
    write_csv(OUT/'diagnostics/existing_history_points.csv',distributions)
    print('Saved separate training-sample / independent-policy histories',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('survey','reanalyze'));p.add_argument('--workers',type=int,default=9)
    a=p.parse_args();survey(a.workers) if a.action=='survey' else reanalyze()
