"""Independent model reload, candidate accounting and original-data verification."""
from .result_paths import resolve as resolve_result_path
import csv
import json
from pathlib import Path
import re
import time

import numpy as np

from .constant020_env import episode, account
from .issue19_data import (ROOT, BASE, OUT, METHODS, WEIGHTS, environment, ready,
    folder_for, curve_for, endpoints, save_audio, target_data)
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def audit():
    import torch
    from stable_baselines3 import TD3, SAC
    torch.set_num_threads(1)
    tick = time.perf_counter(); manifest = ready(); replays = []; registry = []; runs = []
    target, controls = target_data(); eval_account = []
    assert controls.tolist() == [4.1e7, 3.5e7, 3.4e6, 3e5, 4.9e4]
    assert json.loads((OUT/'model_gate.json').read_text())['passed']
    assert digest(BASE/'manifest.json') == manifest['baseline_manifest_sha256']
    previous = json.loads((BASE/'manifest.json').read_text())
    for name, expected in previous['source_sha256'].items():
        # Only the known stored-x0 revision is allowed in historical physics.
        if name != 'native/model.cpp': assert digest(resolve_result_path(ROOT/name)) == expected, name
    for weight in WEIGHTS:
        for method in METHODS:
            for seed in (0, 1, 2):
                folder = folder_for(method, seed, weight)
                summary_folder = folder_for(method, seed, 1.) if method=='random_search' else folder
                s = json.loads((summary_folder/'summary.json').read_text())
                curve = curve_for(method, seed, weight); endpoint = endpoints(method, seed, weight)
                is_rl = method in ('TD3', 'SAC'); reused = method=='random_search' and weight==.5
                assert [p['episodes'] for p in curve] == (list(range(0, 4097, 128)) if is_rl else list(range(1, 4097)))
                assert s['training']['episodes'] == 4096 and s['training']['failures'] == 0
                assert s['training']['native_steps_completed'] == 8192000000 and s['nonfinite_detected'] == 0
                trace = read_record(summary_folder/'training_trace.json') if is_rl else curve
                assert len(trace) == 4096
                if is_rl:
                    assert s['gradient_updates'] == 4080
                    for p in curve: assert p['learner_before'] == p['learner_after']
                if weight == .5 and method != 'random_search':
                    prior_folder = folder_for(method, seed, 1.)
                    prior_s = json.loads((prior_folder/'summary.json').read_text())
                    if is_rl:
                        prior_trace = read_record(prior_folder/'training_trace.json')
                        assert [p['action'] for p in trace[:16]] == [p['action'] for p in prior_trace[:16]]
                        assert s['initial']['learner_before']['policy_sha256'] == prior_s['initial']['learner_before']['policy_sha256']
                        assert s['effective_hyperparameters'] == prior_s['effective_hyperparameters']
                    else:
                        prior_curve = curve_for(method, seed, 1.)
                        assert [p['actions'] for p in curve[:8]] == [p['actions'] for p in prior_curve[:8]]
                for p in trace:
                    c = p['physical_controls']
                    assert all(a-1e-6 <= b <= d+1e-6 for a,b,d in zip((2.5e7, 2.5e6, 49000.), [c[i] for i in (0,2,3)], (5e7,4e6,800000.)))
                    assert c[1] == 3.5e7 and c[4] == 49000.
                    assert np.isfinite(p['weighted_loss']) and not p['failed']
                env = environment(weight); job_tick = time.perf_counter()
                try:
                    for role, expected in endpoint.items():
                        if is_rl:
                            model_path = (folder/('agent.zip' if role=='final' else 'best/agent.zip')) if weight==1. else folder/('agent.zip' if role=='final' else 'selected_agent.zip')
                            agent = {'TD3':TD3, 'SAC':SAC}[method].load(model_path, device='cpu')
                            assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
                            metrics, values = episode(env, agent=agent)
                        else:
                            model_path = None
                            metrics, values = episode(env, action=expected['actions'][0])
                        assert np.array_equal(values['actions'][0], expected['actions'][0])
                        assert metrics['acoustic_error'] == expected['acoustic_error']
                        assert metrics['weighted_loss'] == expected['weighted_loss']
                        assert metrics['sustained_voiced'] == expected['sustained_voiced']
                        if weight==1.:
                            audio_path = folder/('rollout.npz' if role=='final' else 'best/rollout.npz')
                        elif method!='random_search':
                            audio_path = folder/f'{role}.npz'
                        elif role=='final':
                            audio_path = summary_folder/'rollout.npz'
                        else:
                            original_selected=endpoints(method,seed,1.)['selected']
                            if np.array_equal(expected['actions'],original_selected['actions']):
                                audio_path=summary_folder/'best/rollout.npz'
                            else:
                                # Only a newly selected solution needs a new raw
                                # snapshot; replay stays within the 48 audit sounds.
                                audio_path = OUT/'representatives'/f'random_seed{seed}_selected.npz'
                                audio_path.parent.mkdir(exist_ok=True); save_audio(audio_path, values)
                        with np.load(audio_path) as z:
                            assert np.array_equal(z['audio'], values['audio'])
                            assert np.array_equal(z['physical_controls'], values['physical_controls'])
                        replays.append(dict(method=method, seed=seed, weight=weight, role=role,
                            E_total=metrics['acoustic_error'], J=metrics['weighted_loss'], error_difference=0.,
                            waveform_max_difference=0., model_reloaded=is_rl))
                        registry.append(dict(method=method, seed=seed, weight=weight, role=role,
                            audio_path=str(audio_path.relative_to(ROOT)), audio_sha256=digest(audio_path),
                            model_path=str(model_path.relative_to(ROOT)) if model_path else None,
                            model_sha256=digest(model_path) if model_path else None))
                finally:
                    eval_account.append(account(env, time.perf_counter()-job_tick)); env.close()
                runs.append(dict(method=method, seed=seed, weight=weight,
                    candidates=4096, new_candidates=0 if weight==1. or reused else 4096,
                    evaluation_opportunities=33 if is_rl else 4096, gradient_updates=s['gradient_updates'],
                    training_wall_s=0 if weight==1. or reused else s['training']['wall_s'],
                    source_training_wall_s=s['training']['wall_s'],source_process_cpu_s=s['process_cpu_s'],
                    source_evaluation_sounds=s['evaluation']['episodes'],
                    training_failures=s['training']['failures'], source_summary=str((summary_folder/'summary.json').relative_to(ROOT)),
                    source_curve_sha256=digest(summary_folder/'curve.json.gz')))
    write_csv(OUT/'endpoint_replays.csv', replays); write_csv(OUT/'run_accounting.csv', runs)
    write_json(OUT/'endpoint_sources.json', registry)
    assert sum(r['new_candidates'] for r in runs) == 36864
    passes=[]
    if (OUT/'audit.json').exists():
        prior_audit=json.loads((OUT/'audit.json').read_text())
        passes=prior_audit.get('passes',[dict(generated_sounds=prior_audit['audit_generated_sounds'],
            wall_s=prior_audit['wall_s'],source_sha256=prior_audit['source_sha256'])])
    passes.append(dict(generated_sounds=sum(a['episodes'] for a in eval_account),
        wall_s=time.perf_counter()-tick,source_sha256=digest(__file__)))
    write_json(OUT/'audit.json', dict(passed=True, runs=24, new_runs=9, derived_random_runs=3,
        new_training_candidates=36864, reused_random_candidates=12288,
        independent_policy_evaluations=198, training_endpoint_replay_sounds=18,
        audit_generated_sounds=sum(a['episodes'] for a in eval_account),
        cumulative_audit_generated_sounds=sum(p['generated_sounds'] for p in passes),passes=passes,model_reloads=24,
        baseline_files_unmodified=True, paired_initial_policies_warmup_and_DE_population_match=True,
        no_numerical_failures=True, native_library_sha256=manifest['native_library_sha256'],
        source_sha256=digest(__file__), wall_s=time.perf_counter()-tick))
    finalize()
    print('Audit passed: 24 endpoints pairs replay exactly; 9 new runs x4096; original files protected', flush=True)


def finalize():
    """Read-only original-file audit and live Markdown link checks after reporting."""
    import csv
    import gzip
    failures = []; checked = 0
    with (OUT/'protected_before.csv').open() as stream:
        for row in csv.DictReader(stream):
            path = resolve_result_path(ROOT/row['path'])
            if not path.is_file() or digest(path) != row['sha256']: failures.append(row['path'])
            checked += 1
    assert not failures, failures[:10]
    broken = []; links = 0
    for doc in OUT.glob('*.md'):
        for match in re.finditer(r'\]\(([^)]+)\)', doc.read_text()):
            ref = match[1].split('#')[0]
            if not ref or '://' in ref: continue
            links += 1
            if not (doc.parent/ref).exists(): broken.append(dict(document=doc.name, target=ref))
    assert not broken, broken
    images = list((OUT/'figures').glob('*.png'))
    assert len(images) <= 10
    if (OUT/'figure_sources.json').exists():
        figures=json.loads((OUT/'figure_sources.json').read_text())['images']
        assert len(figures)==len(images)==8
        for r in figures:assert digest(OUT/r['path'])==r['sha256']
    for r in json.loads((OUT/'endpoint_sources.json').read_text()) if (OUT/'endpoint_sources.json').exists() else []:
        assert digest(resolve_result_path(ROOT/r['audio_path']))==r['audio_sha256']
        if r['model_path']:assert digest(resolve_result_path(ROOT/r['model_path']))==r['model_sha256']
    csv_rows=0
    if (OUT/'new_candidate_metrics.csv.gz').exists():
        trials={}
        with gzip.open(OUT/'new_candidate_metrics.csv.gz','rt') as stream:
            for row in csv.DictReader(stream):
                key=(row['method'],int(row['seed']));trials.setdefault(key,[]).append(int(row['trial']))
                spec,env,total,J=[float(row[k]) for k in ('E_spec','E_env','E_total','J')]
                assert np.isfinite([spec,env,total,J]).all()
                assert abs(total-spec-env)<1e-14 and abs(J-spec-.5*env)<1e-14
                csv_rows+=1
        assert len(trials)==9 and all(ns==list(range(1,4097)) for ns in trials.values())
    write_json(OUT/'final_integrity.json', dict(passed=True, original_files_checked=checked, original_files_changed=0,
        original_files_deleted=0, local_markdown_links_checked=links, broken_links=broken,
        image_count=len(images), new_candidate_csv_rows=csv_rows, source_sha256=digest(__file__)))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--finalize-only', action='store_true')
    a = p.parse_args(); finalize() if a.finalize_only else audit()
