"""Issue 16: the final, preregistered Phase 0 learning-rate comparison.

Only the six 1e-4 runs are trained. The 3e-4 cohort is read, never redrawn.
Use --prepare, --train, --finish separately to retain reviewable gates/artifacts.
"""
from .result_paths import resolve as resolve_result_path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from .issue14 import diagnostics, train_one
from .issue14_audit import Auditor
from .issue15_analysis import write_json, write_csv

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'results/Phase0/synthetic/reward_ablation/weighted_lambda_0p5'
DEST = BASE.parent / 'learning_rate_comparison'
WORK = ROOT / '.issue16-work/artifacts'
ALGORITHMS, SEEDS = ('TD3', 'SAC'), (0, 1, 2)
SCIENCE = ('sine_env.py', 'experiments.py', 'acoustics.py', 'issue14.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_snapshot():
    return {str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'results').rglob('*')
            if p.is_file() and not p.is_symlink()}


def verify_protected(output):
    entries = json.loads((output/'protected_before.json').read_text())
    changed = [name for name, sha in entries.items() if not (resolve_result_path(ROOT/name)).is_file() or digest(resolve_result_path(ROOT/name)) != sha]
    if changed:
        raise RuntimeError(f'Existing results changed: {changed}')
    return len(entries)


def prepare():
    if WORK.exists() or DEST.exists():
        raise FileExistsError('Use the existing staged results; overwriting is forbidden')
    base = json.loads((BASE/'config.json').read_text())
    old = json.loads((BASE/'manifest.json').read_text())
    assert base['learning_rate'] == .0003 and base['steps'] == 102400
    assert base['eval_interval'] == 2048 and base['reward_envelope_weight'] == .5
    hashes = {name: digest(resolve_result_path(ROOT/name)) for name in SCIENCE}
    assert hashes == {name: old['source_sha256'][name] for name in SCIENCE}
    versions = {name: importlib.metadata.version(name) for name in old['versions']}
    assert versions == old['versions']
    WORK.mkdir(parents=True)
    settings = dict(base, algorithms=list(ALGORITHMS), learning_rate=.0001)
    write_json(WORK/'config.json', settings)
    write_json(WORK/'protected_before.json', protected_snapshot())
    manifest = dict(issue=16, created_utc=datetime.now(timezone.utc).isoformat(),
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    git_status_before=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
                    source_sha256=hashes, runner_sha256=digest(__file__), versions=versions,
                    baseline=str(BASE.relative_to(ROOT)), baseline_manifest_sha256=digest(BASE/'manifest.json'),
                    baseline_retrained=False, changed_training_hyperparameters={'learning_rate':[.0003,.0001]},
                    training_reward='-(local E_spec + 0.5 * local E_env)',
                    evaluation='full 0.35 s E_spec + E_env; lambda=1; post-final-update endpoint',
                    registered_training_runs=6, device='cpu', torch_threads_per_worker=1)
    write_json(WORK/'manifest.json', manifest)
    diagnostics(WORK, settings)
    # A separate 320-step run exercises updates, weighted rewards, and save/reload.
    smoke = dict(settings, steps=320, eval_interval=160)
    summary = train_one('TD3', 0, smoke, WORK.parent/'smoke_TD3_0')
    assert summary['training_steps'] == 320 and summary['optimizer_steps']['critic'] == 64
    from .experiments import phase0_factories
    from .issue14 import evaluate_agent
    from stable_baselines3 import TD3
    agent = TD3.load(WORK.parent/'smoke_TD3_0/agent.zip', device='cpu')
    replay, _ = evaluate_agent(agent, phase0_factories(settings)[1], 320, 'reloaded')
    assert replay['error'] == summary['error']
    with np.load(WORK.parent/'smoke_TD3_0/training_trace.npz') as trace:
        np.testing.assert_allclose(trace['rewards'], -(trace['log_spectral']+.5*trace['envelope']), rtol=1e-6, atol=1e-7)
    write_json(WORK/'smoke.json', dict(passed=True, environment_steps=320,
               optimizer_steps=summary['optimizer_steps'], reload_error_difference=replay['error']-summary['error'],
               independent_of_registered_runs=True))
    print(f'Prepared and smoke-tested {WORK}', flush=True)


def train(workers):
    settings = json.loads((WORK/'config.json').read_text())
    manifest = json.loads((WORK/'manifest.json').read_text())
    assert manifest['source_sha256'] == {name:digest(resolve_result_path(ROOT/name)) for name in SCIENCE}
    assert json.loads((WORK/'smoke.json').read_text())['passed']
    started = time.perf_counter()
    summaries = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(train_one, name, seed, settings, WORK/f'{name}_{seed}')
                   for name in ALGORITHMS for seed in SEEDS]
        for future in as_completed(futures):
            summaries.append(future.result())
            write_json(WORK/'summary.json', summaries)
    write_json(WORK/'execution.json', dict(completed_runs=len(summaries), concurrent_run_wall_s=time.perf_counter()-started,
               training_environment_steps=sum(s['training_steps'] for s in summaries),
               evaluation_environment_steps=sum(s['evaluation_steps'] for s in summaries),
               training_process_wall_s=sum(s['training_wall_s'] for s in summaries),
               evaluation_process_wall_s=sum(s['evaluation_wall_s'] for s in summaries)))


def normalize_lr(values):
    result = deepcopy(values)
    result['learning_rate'] = 'ONLY_CHANGED_PARAMETER'
    for groups in result['optimizer_param_groups'].values():
        for group in groups:
            group['lr'] = 'ONLY_CHANGED_PARAMETER'
    return result


class LearningRateAuditor(Auditor):
    def effective_settings(self, values, agent, name, seed, path):
        baseline = json.loads((BASE/f'{name}_{seed}/summary.json').read_text())['effective_hyperparameters']
        self.check(normalize_lr(values) == normalize_lr(baseline), 'baseline_effective_settings_except_lr', path)
        self.scalar(values['learning_rate'], .0001, path/'recorded_learning_rate')
        self.scalar(agent.learning_rate, .0001, path/'loaded_learning_rate')
        for label in ('actor', 'critic'):
            for group in getattr(agent, label).optimizer.param_groups:
                self.scalar(group['lr'], .0001, path/f'{label}_actual_optimizer_lr')
        if name == 'SAC':
            for group in agent.ent_coef_optimizer.param_groups:
                self.scalar(group['lr'], .0001, path/'entropy_actual_optimizer_lr')


def audit(output):
    auditor = LearningRateAuditor(output)
    auditor.counts['runs_expected'] = 6
    auditor.oracle()
    for name in ALGORITHMS:
        for seed in SEEDS:
            folder = output/f'{name}_{seed}'
            auditor.audit_run(folder/'summary.json', name, seed)
            auditor.counts['runs_found'] += 1
            baseline = json.loads((BASE/f'{name}_{seed}/summary.json').read_text())
            summary = auditor.summaries[(name,seed)]
            auditor.check(summary['initial_record']['parameter_sha256'] == baseline['initial_record']['parameter_sha256'],
                          'identical_initial_policy', folder)
            with np.load(folder/'training_trace.npz') as new, np.load(BASE/f'{name}_{seed}/training_trace.npz') as old:
                auditor.exact_array(new['actions'][:256], old['actions'][:256], folder/'identical_warmup_actions')
                auditor.exact_array(new['rewards'][:256], old['rewards'][:256], folder/'identical_warmup_rewards')
                auditor.check(np.allclose(new['rewards'], -(new['log_spectral']+.5*new['envelope']),rtol=1e-6,atol=1e-7),
                              'weighted_local_training_reward', folder)
    result = dict(passed=not auditor.errors, checks=auditor.checks, counts=auditor.counts,
                  errors=auditor.errors, final_timing_comparisons=auditor.final_timing_comparisons,
                  protected_files_verified=verify_protected(output))
    write_json(output/'audit.json', result)
    if not result['passed']:
        raise RuntimeError(f'Audit failed: {auditor.errors[:3]}')
    return result


def finish():
    from .issue16_report import report
    if DEST.exists():
        raise FileExistsError(DEST)
    audit(WORK)
    report(WORK, BASE)
    verify_protected(WORK)
    WORK.rename(DEST)
    print(f'Published audited results: {DEST}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','train','finish','audit'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'train': train(args.workers)
    elif args.action == 'finish': finish()
    else: audit(DEST if DEST.exists() else WORK)
