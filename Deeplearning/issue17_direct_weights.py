"""Add only DE lambda=.5; rescore the saved Random pool. Never construct an RL agent."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .constant020_env import Constant020Env,INDICES,account,episode
from .fixed_budget_de import fixed_budget_de
from .issue15_analysis import write_json
from .issue16 import digest
from .issue17 import ROOT,DEST,load_ready,save_snapshot
from .issue17_artifacts import read_record
from .issue17_selection import selection_curve

WORK=ROOT/'.issue17-work/direct_weights'
EXTENSION=DEST/'direct_weight_extension'
NUMERICAL=('issue17_direct_weights.py','issue17_selection.py')
MUTABLE=('README.md','REPORT.md','comparison_aggregate.csv','comparison_per_seed.csv','comparison_curves.csv',
         'checkpoint_comparison.csv','lambda_effects.csv','figure_sources.json','figure_audit.json','visual_review.json',
         'report_timing.json','checksums.json','storage_summary.json','issue_update_status.json','issue_update_proposal.md')


def current_sources():
    return {name:digest(ROOT/name) for name in NUMERICAL}


def write_curve(path,records):
    payload=json.dumps(records,ensure_ascii=False,allow_nan=False).encode()
    packed=gzip.compress(payload,mtime=0)
    assert gzip.decompress(packed)==payload
    Path(path).write_bytes(packed)


def prepare():
    if WORK.exists() or EXTENSION.exists():raise FileExistsError('Existing extension preserved')
    base,_,_=load_ready(DEST)
    WORK.mkdir(parents=True)
    settings=dict(base['settings'],direct_envelope_weights=[1.,.5],new_DE_runs=9,derived_random_runs=9)
    write_json(WORK/'config.json',settings)
    write_json(WORK/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        workflow='issue17_all_methods_weight_extension',base_manifest_sha256=digest(DEST/'manifest.json'),
        base_source_sha256=base['source_sha256'],numerical_sources=current_sources(),
        target_sha256=base['target_sha256'],new_training='9 DE runs only; lambda=.5;4096 candidates;seeds0/1/2',
        random_reuse='All4096 candidates from corresponding lambda1 run; no candidate regeneration',
        selection='min J_lambda, ties first; common E_total post-hoc minimum stored separately',
        original_RL_and_lambda1_direct_runs_retrained=False))
    protected={}
    for path in (ROOT/'results').rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        if path.is_relative_to(DEST):
            rel=path.relative_to(DEST).as_posix()
            if rel in MUTABLE or rel.startswith(('figures/learning_curves/','figures/best_observed/')):continue
        protected[str(path.relative_to(ROOT))]=digest(path)
    write_json(WORK/'protected_before.json',protected)
    old=[dict(path=str(p.relative_to(DEST)),sha256=digest(p),bytes=p.stat().st_size)
         for name in ('learning_curves','best_observed') for p in (DEST/'figures'/name).iterdir() if p.is_file()]
    assert len(old)==72 and all(p['path'].endswith('.png') for p in old)
    write_json(WORK/'old_single_curve_images.json',old)
    shutil.copy2(DEST/'comparison_per_seed.csv',WORK/'baseline_comparison_per_seed.csv')
    print('Extension prepared; existing numerical artifacts protected',flush=True)


def ready():
    manifest=json.loads((WORK/'manifest.json').read_text())
    if manifest['numerical_sources']!=current_sources():raise RuntimeError('Extension numerical sources changed')
    base,target,controls=load_ready(DEST)
    if digest(DEST/'manifest.json')!=manifest['base_manifest_sha256']:raise RuntimeError('Base manifest changed')
    return base,target,controls


def record_metric(metrics,values,n):
    return dict(episodes=n,**metrics,actions=values['actions'].tolist(),
                physical_controls=values['physical_controls'][0].tolist(),gradient_updates=0,
                evaluation_phase='actual_training_candidate')


def de_one(condition,seed):
    base,target,controls=ready();config=base['settings'];indices=INDICES[condition]
    folder=WORK/'runs'/condition/'DE/lambda0p5'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'config.json',dict(settings=config,condition=condition,method='DE',seed=seed,
        envelope_weight=.5,target_sha256=base['target_sha256'],base_source_sha256=base['source_sha256']))
    train=Constant020Env(indices,target,controls,envelope_weight=.5)
    evaluation=Constant020Env(indices,target,controls,envelope_weight=.5)
    raw=[];snapshots={};best_j=best_e=None;artifact_wall=eval_wall=0.
    start=time.perf_counter();cpu=time.process_time();rng=np.random.default_rng(seed)
    initial=rng.uniform(-1,1,len(indices))
    print(json.dumps(dict(stage='DE_start',condition=condition,seed=seed)),flush=True)

    def candidate(action):
        nonlocal best_j,best_e,artifact_wall
        metrics,values=episode(train,action=np.asarray(action));n=len(raw)+1
        row=record_metric(metrics,values,n);raw.append(row)
        if n==1:snapshots['initial']=values
        snapshots['final']=values
        if best_j is None or row['weighted_loss']<best_j['weighted_loss']:
            best_j=row;snapshots['best_objective']=values
        if best_e is None or row['acoustic_error']<best_e['acoustic_error']:
            best_e=row;snapshots['best_common_posthoc']=values
        if n in config['checkpoints'] and n<config['episodes']:
            tick=time.perf_counter();cp=folder/'checkpoints'/str(n)
            save_snapshot(cp,values,target,controls);write_json(cp/'metrics.json',row)
            artifact_wall+=time.perf_counter()-tick
        if n%256==0:
            write_json(folder/'progress.json',dict(episodes=n,common_error=row['acoustic_error'],
                selected_weighted_loss=best_j['weighted_loss'],elapsed_s=time.perf_counter()-start))
        if n%1024==0:print(json.dumps(dict(stage='DE_progress',condition=condition,seed=seed,trials=n)),flush=True)
        return row['weighted_loss']

    try:
        tick=time.perf_counter()
        solver=fixed_budget_de(candidate,len(indices),rng,initial,budget=config['episodes'],
            population_size=config['de_population'],mutation=config['de_mutation'],recombination=config['de_recombination'])
        training_wall=time.perf_counter()-tick-artifact_wall
        curve=selection_curve(raw,.5)
        assert len(curve)==4096 and solver['nfev']==4096
        assert abs(solver['error']-best_j['weighted_loss'])<1e-12
        # The library's tie resolution can return a different tied candidate.
        # Preserve that solution, without silently using common error to break ties.
        tick=time.perf_counter();solver_metrics,solver_values=episode(evaluation,action=solver['action'])
        eval_wall+=time.perf_counter()-tick
        snapshots['solver_returned']=solver_values
        solver.update(weighted_loss=solver_metrics['weighted_loss'],common_error=solver_metrics['acoustic_error'],
                      objective='E_spec +0.5 E_env',tie_policy_for_comparison_archive='first minimum J')
        write_json(folder/'de_selected_incumbent.json',solver)
        replay=[]
        for name,expected in (('final',raw[-1]),('best_objective',best_j),('best_common_posthoc',best_e)):
            tick=time.perf_counter();metrics,values=episode(evaluation,action=snapshots[name]['actions'][0])
            eval_wall+=time.perf_counter()-tick
            assert metrics['acoustic_error']==expected['acoustic_error'] and metrics['weighted_loss']==expected['weighted_loss']
            replay.append(dict(stage=name,common_error=metrics['acoustic_error'],weighted_loss=metrics['weighted_loss'],difference=0.))
        tick=time.perf_counter()
        for name,values in snapshots.items():save_snapshot(folder if name=='final' else folder/name,values,target,controls)
        write_curve(folder/'curve.json.gz',curve)
        write_json(folder/'replay.json',replay)
        summary=dict(condition=condition,method='DE',seed=seed,envelope_weight=.5,initial=curve[0],final=curve[-1],
            objective_selected=curve[curve[-1]['objective_selected_index']],
            common_posthoc_best=curve[curve[-1]['common_posthoc_index']],
            training=account(train,training_wall),evaluation=account(evaluation,eval_wall),
            training_candidates=4096,new_candidate_sounds=4096,candidate_pool_reused=False,
            gradient_updates=0,optimizer_steps={},optimizer_nfev=solver['nfev'],
            source_sha256=current_sources(),target_sha256=base['target_sha256'],
            nonfinite_detected=0,artifact_wall_s=artifact_wall+time.perf_counter()-tick,
            elapsed_s=time.perf_counter()-start,process_cpu_s=time.process_time()-cpu,
            rollout_sha256=digest(folder/'rollout.npz'),curve_sha256=digest(folder/'curve.json.gz'))
        write_json(folder/'summary.json',summary)
        print(json.dumps(dict(stage='DE_complete',condition=condition,seed=seed,
            candidate4096=summary['final']['acoustic_error'],selected_common=summary['objective_selected']['acoustic_error'])),flush=True)
        return summary
    except Exception as exc:
        write_json(folder/'failure.json',dict(error=str(exc),training=account(train,time.perf_counter()-start)))
        raise
    finally:train.close();evaluation.close()


def link_snapshot(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('rollout.npz','target.wav','generated.wav'):
        p=source/name
        if p.is_symlink() or not p.resolve().is_relative_to(DEST):raise RuntimeError('Unexpected source')
        os.link(p,destination/name)


def random_one(condition,seed):
    base,target,controls=ready();start=time.perf_counter()
    source=DEST/'runs'/condition/'random_search/lambda1'/f'seed_{seed}'
    prior=json.loads((source/'summary.json').read_text());original=read_record(source/'curve.json')
    assert len(original)==4096 and [r['episodes'] for r in original]==list(range(1,4097))
    curve=selection_curve(original,.5);folder=WORK/'runs'/condition/'random_search/lambda0p5'/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    objective=curve[curve[-1]['objective_selected_index']];common=curve[curve[-1]['common_posthoc_index']]
    rescoring_wall=time.perf_counter()-start
    evaluation=Constant020Env(INDICES[condition],target,controls,envelope_weight=.5)
    try:
        link_snapshot(source,folder);link_snapshot(source/'initial',folder/'initial')
        link_snapshot(source/'best',folder/'best_common_posthoc')
        reuse={1:source/'initial',4096:source,prior['best']['episodes']:source/'best'}
        tick=time.perf_counter()
        if objective['episodes'] in reuse:
            link_snapshot(reuse[objective['episodes']],folder/'best_objective')
            audio_reused=True
        else:
            metrics,values=episode(evaluation,action=objective['actions'][0])
            assert metrics['acoustic_error']==objective['acoustic_error']
            assert metrics['weighted_loss']==objective['weighted_loss']
            save_snapshot(folder/'best_objective',values,target,controls)
            audio_reused=False
        replay_wall=time.perf_counter()-tick
        write_curve(folder/'curve.json.gz',curve)
        write_json(folder/'source_reuse.json',dict(source=str(source.relative_to(DEST)),
            source_curve_sha256=digest(source/'curve.json.gz'),source_summary_sha256=digest(source/'summary.json'),
            candidates_reused=4096,candidates_regenerated=0,same_order_and_actions=True,
            artifact_storage='hard links to unchanged lambda1 snapshots, except newly rendered weighted-selected solution',
            selected_audio_reused=audio_reused))
        summary=dict(condition=condition,method='random_search',seed=seed,envelope_weight=.5,
            initial=curve[0],final=curve[-1],objective_selected=objective,common_posthoc_best=common,
            training=dict(episodes=0,decision_steps=0,native_steps_attempted=0,native_steps_completed=0,wall_s=rescoring_wall),
            evaluation=account(evaluation,replay_wall),source_sampling_account=prior['training'],
            training_candidates=4096,new_candidate_sounds=0,candidate_pool_reused=True,
            gradient_updates=0,optimizer_steps={},optimizer_nfev=0,nonfinite_detected=0,
            rescoring_wall_s=rescoring_wall,elapsed_s=time.perf_counter()-start,
            source_sha256=current_sources(),target_sha256=base['target_sha256'],
            rollout_sha256=digest(folder/'rollout.npz'),curve_sha256=digest(folder/'curve.json.gz'))
        write_json(folder/'summary.json',summary)
        write_json(folder/'config.json',dict(condition=condition,method='random_search',seed=seed,
            envelope_weight=.5,settings=base['settings'],generation='reused all saved lambda1 candidates'))
        return summary
    finally:evaluation.close()


def run(workers):
    ready();start=time.perf_counter();summaries=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(de_one,c,s) for c in INDICES for s in (0,1,2)]
        for future in as_completed(futures):
            summaries.append(future.result());write_json(WORK/'DE_summary.json',summaries)
    de_wall=time.perf_counter()-start
    random=[random_one(c,s) for c in INDICES for s in (0,1,2)]
    write_json(WORK/'random_summary.json',random)
    write_json(WORK/'execution.json',dict(new_DE_runs=9,derived_random_runs=9,workers=workers,
        new_training_candidates=9*4096,reused_random_candidates=9*4096,random_candidates_regenerated=0,
        DE_parallel_wall_s=de_wall,extension_wall_s=time.perf_counter()-start,
        DE_training_process_wall_s=sum(s['training']['wall_s'] for s in summaries),
        random_rescoring_wall_s=sum(s['rescoring_wall_s'] for s in random),
        new_evaluation_sounds=sum(s['evaluation']['episodes'] for s in summaries+random)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--workers',type=int,default=9);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run(args.workers)
