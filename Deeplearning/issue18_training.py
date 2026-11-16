"""Reuse the unmodified Issue17 learner in isolated child processes, new domain."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from functools import partial
import importlib.metadata
import json
import multiprocessing
import shutil
import time

from . import issue17 as original
from .issue18_diagnostics import OUT,BASE
from .issue18_domain_env import DomainEnv
from .issue15_analysis import write_json
from .issue16 import digest

ADAPTER_SOURCES=('issue18_training.py','issue18_domain_env.py')


def hashes():return {p:digest(original.ROOT/p) for p in ADAPTER_SOURCES}


def prepare():
    if (OUT/'manifest.json').exists():raise FileExistsError('Existing training manifest preserved')
    base,target,controls=original.load_ready(BASE)
    decision=json.loads((OUT/'domain_decision.json').read_text())
    if decision['restricted_f0_high']>=decision['original_f0_high']:raise RuntimeError('Survey does not justify a restriction')
    config=dict(base['settings'],conditions=['constant_3d'],rl_envelope_weights=[1.],
                domain_restriction=decision)
    shutil.copy2(BASE/'diagnostics/review.json',OUT/'diagnostics/review.json')
    write_json(OUT/'config.json',config)
    write_json(OUT/'manifest.json',dict(base,issue=18,created_utc=datetime.now(timezone.utc).isoformat(),
        settings=config,range_adapter_source_sha256=hashes(),
        domain_decision_sha256=digest(OUT/'domain_decision.json'),base_manifest_sha256=digest(BASE/'manifest.json'),
        baseline='Existing full-domain Issue17 runs read only; no baseline retraining',
        implementation='Unmodified issue17.run_one; per-child-process env factory supplies only the f0_l range adapter',
        reward='Unmodified -(E_spec+E_env); only lambda1 follow-up; target, physics and all learner settings unchanged'))
    print('Prepared 12 restricted-domain runs; lambda1 only',flush=True)


def run_one(method,seed):
    manifest=json.loads((OUT/'manifest.json').read_text())
    if hashes()!=manifest['range_adapter_source_sha256']:raise RuntimeError('Frozen range adapter changed')
    if digest(OUT/'domain_decision.json')!=manifest['domain_decision_sha256']:raise RuntimeError('Domain changed after preregistration')
    # The substitution is confined to this spawned process. Original files and
    # the base experiment are never mutated, and the learner code is identical.
    original.Constant020Env=partial(DomainEnv,f0_high=manifest['settings']['domain_restriction']['restricted_f0_high'])
    return original.run_one(OUT,'constant_3d',method,seed,1.)


def train(workers=12):
    start=time.perf_counter();summaries=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,m,s) for m in ('TD3','SAC','DE','random_search') for s in (0,1,2)]
        for f in as_completed(futures):
            summaries.append(f.result());write_json(OUT/'summary.json',summaries)
    write_json(OUT/'execution.json',dict(runs=len(summaries),workers=workers,wall_s=time.perf_counter()-start,
        training_candidates=sum(s['training']['episodes'] for s in summaries),
        evaluation_sounds=sum(s['evaluation']['episodes'] for s in summaries),
        training_process_wall_s=sum(s['training']['wall_s'] for s in summaries),
        evaluation_process_wall_s=sum(s['evaluation']['wall_s'] for s in summaries),
        process_cpu_s=sum(s['process_cpu_s'] for s in summaries)))
    from .issue17_artifacts import compress_records
    compress_records(OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','train'));p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();prepare() if a.action=='prepare' else train(a.workers)
