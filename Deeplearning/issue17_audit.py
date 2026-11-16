"""Audit the revised 4096-candidate/54-run experiment without learning."""
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.io import wavfile
from .acoustics import distance
from .constant020_env import Constant020Env, INDICES, RATE, account, episode
from .issue15_analysis import write_json
from .issue16 import digest, verify_protected
from .issue17 import load_ready, run_specs, weight_label, WORK
from .model import LOW, HIGH
from .issue17_artifacts import read_record,compress_records


def audit(output):
    if (Path(output)/'visualization_revision/audit.json').exists():
        from .issue17_visualization import audit as audit_visualization
        return audit_visualization()
    if (Path(output)/'direct_weight_extension/manifest.json').exists():
        from .issue17_extension_audit import audit as audit_extended
        return audit_extended(Path(output))
    import torch
    from stable_baselines3 import TD3, SAC
    torch.set_num_threads(1)
    output = Path(output)
    started = time.perf_counter()
    errors, reloads, accounts, runs = [], [], [], {}
    checks = 0
    manifest, target, controls = load_ready(output)
    config = manifest['settings']
    if output==WORK:
        if len(list((output/'runs').glob('*/*/*/seed_*/summary.json')))!=54:
            raise RuntimeError('Do not compress or audit incomplete runs')
        compress_records(output)
    budget, scale = config['episodes'], float(abs(target).max())

    def check(value, code, path, actual=None):
        nonlocal checks
        checks += 1
        if not value:
            errors.append(dict(code=code, path=str(path.relative_to(output)), actual=actual))

    def scalar(a, b, code, path, atol=1e-12, rtol=0.):
        check(bool(np.isfinite(a) and np.isfinite(b) and abs(a-b) <= atol+rtol*abs(b)),
              code, path, dict(actual=float(a), expected=float(b)))

    specs = run_specs(config)
    expected = {output/'runs'/c/m/weight_label(w)/f'seed_{s}'/'summary.json' for c,m,s,w in specs}
    found = set((output/'runs').glob('*/*/*/seed_*/summary.json'))
    check(found == expected, 'exact_54_run_set', output, len(found))
    check(budget == 4096 and len(specs) == 54, 'registered_budget', output)
    check(manifest['target_sha256'] == manifest['supersedes']['pilot_target_sha256'],
          'unchanged_pilot_target_archive', output)
    schedule = list(range(0, budget+1, config['eval_every_episodes']))
    for index, (c,m,seed,w) in enumerate(specs):
        folder = output/'runs'/c/m/weight_label(w)/f'seed_{seed}'
        s = json.loads((folder/'summary.json').read_text())
        curve = read_record(folder/'curve.json')
        trace = read_record(folder/'training_trace.json') if m in ('TD3','SAC') else curve
        rc = json.loads((folder/'config.json').read_text())
        runs[(c,m,seed,w)] = s, trace
        check(rc['settings'] == config and rc['target_sha256'] == manifest['target_sha256'], 'common_config_target', folder)
        check(s['source_sha256'] == manifest['source_sha256'], 'frozen_sources', folder)
        check(s['envelope_weight'] == w == rc['envelope_weight'], 'weight_identity', folder)
        check(len(trace) == budget and [r['episodes'] for r in trace] == list(range(1,budget+1)), 'complete_trace', folder)
        check(s['training']['episodes'] == s['training']['decision_steps'] == s['training_candidates'] == budget,
              '4096_environment_steps_not_updates', folder)
        for name in ('training','evaluation'):
            a = s[name]
            check(a['native_steps_attempted'] == a['native_blocks_attempted']*100000, 'native_attempt_accounting', folder)
            check(a['native_steps_completed'] == a['native_blocks_completed']*100000, 'native_complete_accounting', folder)
            if a['failures'] == 0:
                check(a['native_blocks_completed'] == a['native_blocks_attempted'] == a['episodes']*20,
                      '20_blocks_per_sound', folder)
        check(s['nonfinite_detected'] == 0, 'no_nonfinite_updates', folder)
        for r in trace:
            comp = r['components'] if m in ('TD3','SAC') else r['acoustic_components']
            check(all(np.isfinite(comp[k]) for k in ('log_spectral','envelope','total')), 'finite_components', folder)
            weighted = 1e6 if r['failed'] else comp['log_spectral']+w*comp['envelope']
            scalar(r['weighted_loss'], weighted, 'weighted_loss_formula', folder)
            if not r['failed']:
                scalar(r['acoustic_error'], comp['log_spectral']+comp['envelope'], 'lambda_independent_error', folder)
            if m in ('TD3','SAC'):
                scalar(r['reward'], -weighted, 'actual_float32_reward', folder, atol=1e-7, rtol=1e-7)
        check(s['initial'] == curve[0] and s['final'] == curve[-1], 'summary_curve_endpoints', folder)
        best = min(curve, key=lambda r:r['acoustic_error'])
        scalar(s['best']['acoustic_error'], best['acoustic_error'], 'best_common_error', folder)
        check(s['best']['episodes'] == best['episodes'], 'best_checkpoint_position', folder)
        check(digest(folder/'rollout.npz') == s['rollout_sha256'], 'final_rollout_hash', folder)
        if m in ('TD3','SAC'):
            check([r['episodes'] for r in curve] == schedule, 'predeclared_33_evaluations', folder)
            check(s['evaluation']['episodes'] == len(schedule), 'evaluation_cost', folder)
            updates = budget-config['warmup_episodes']
            check(s['gradient_updates'] == updates, 'gradient_iterations', folder)
            calls = dict(actor=updates//2, critic=updates) if m=='TD3' else dict(actor=updates,critic=updates,entropy=updates)
            check(s['optimizer_steps'] == calls, 'actual_optimizer_calls', folder)
            for r in curve:
                check(r['learner_before'] == r['learner_after'], 'evaluation_keeps_learner_rng', folder)
                check(r['gradient_updates'] == max(0,r['episodes']-16), 'post_update_timing', folder)
            check(curve[-1]['evaluation_phase'] == 'post_final_update', 'final_after_all_updates', folder)
            effective = s['effective_hyperparameters']
            for key,value in dict(learning_rate=.0003,gamma=1.,batch_size=32,learning_starts=16,
                                  buffer_size=50000,gradient_steps=1,seed=seed,torch_threads=1).items():
                scalar(effective[key],value,'effective_'+key,folder)
            check(effective['net_arch'] == [32,32] and effective['device'] == 'cpu', 'network_device', folder)
            if m=='SAC':
                scalar(effective['initial_entropy_coefficient'],.01,'initial_entropy',folder,atol=1e-8)
            check(digest(folder/'agent.zip') == s['model_sha256'], 'final_model_hash', folder)
            check(bool(s['numeric_training_metrics']) and all(np.isfinite(v[k])
                  for v in s['numeric_training_metrics'].values() for k in ('minimum','maximum')), 'finite_losses', folder)
        else:
            check(w == 1 and s['optimizer_nfev'] == budget and s['evaluation']['episodes'] == 2,
                  'direct_standard_objective_exact_budget', folder)
            check(s['final']['episodes'] == budget and s['final']['evaluation_phase'] == 'actual_training_candidate',
                  'actual_4096th_candidate_primary', folder)
            minimum = np.minimum.accumulate([r['acoustic_error'] for r in curve])
            check(np.array_equal(minimum,[r['best_observed_error'] for r in curve]), 'candidate_vs_incumbent', folder)
            for name,record in (('final_candidate_replay.json',s['final']),('best_candidate_replay.json',s['best'])):
                replay = json.loads((folder/name).read_text())
                scalar(replay['acoustic_error'],record['acoustic_error'],'direct_independent_replay',folder)
            if m=='DE':
                selected = json.loads((folder/'de_selected_incumbent.json').read_text())
                scalar(selected['error'],s['best']['acoustic_error'],'de_selected_incumbent',folder)
                check(selected['nfev'] == budget and selected['convergence_stops_disabled'], 'de_exact_budget', folder)
        stages = [('initial',folder/'initial',s['initial']),('best',folder/'best',s['best']),('final',folder,s['final'])]
        for n in config['checkpoints']:
            cp = folder/'checkpoints'/str(n)
            record = json.loads((cp/'metrics.json').read_text())
            check(record['episodes'] == n, 'required_checkpoint_present', cp)
            if n < budget:
                matching = next(r for r in curve if r['episodes']==n)
                scalar(record['acoustic_error'],matching['acoustic_error'],'checkpoint_curve',cp)
                stages.append((str(n),cp,record))
            else:
                check(record['final_artifacts_reused'] and record['artifact_directory']=='../..',
                      'final_without_duplicate_arrays',cp)
        env = Constant020Env(INDICES[c],target,controls,envelope_weight=w)
        tick = time.perf_counter()
        try:
            for stage,destination,record in stages:
                with np.load(destination/'rollout.npz') as data:
                    audio = data['audio'].copy()
                    actions = data['actions'].copy()
                    physical = data['physical_controls'].copy()
                    check(np.array_equal(data['target'],target),'identical_target_waveform',destination)
                    check(int(data['rate'])==50000 and float(data['duration_s'])==.2 and audio.shape==(10000,),
                          'duration_sample_rate',destination)
                check(physical.shape==(20,5) and np.array_equal(physical,np.tile(physical[0],(20,1))),
                      'constant_control_twenty_blocks',destination)
                check(np.isfinite(audio).all() and np.all(physical>=LOW) and np.all(physical<=HIGH),'finite_bounds',destination)
                other = [i for i in range(5) if i not in INDICES[c]]
                check(np.array_equal(physical[:,other],np.tile(controls[other],(20,1))),'nonlearned_controls_fixed',destination)
                calc = distance(target,audio,RATE,scale=scale)
                for k in ('log_spectral','envelope','total'):
                    scalar(calc[k],record['acoustic_components'][k],'recomputed_'+k,destination)
                if not record['failed']:
                    scalar(calc['total'],record['acoustic_error'],'recomputed_common_error',destination)
                pe = np.sqrt(np.mean(((physical[0]-controls)/(HIGH-LOW))[list(INDICES[c])]**2))
                scalar(record['parameter_nrmse'],pe,'parameter_nrmse',destination)
                rate,wav = wavfile.read(destination/'generated.wav')
                check(rate==RATE and np.array_equal(wav,(audio/scale).astype(np.float32)),'common_gain_wav',destination)
                agent = None
                if m in ('TD3','SAC'):
                    agent = {'TD3':TD3,'SAC':SAC}[m].load(destination/'agent.zip',device='cpu')
                    check(all(torch.isfinite(p).all() for p in agent.policy.parameters()),'finite_loaded_network',destination)
                    check(agent.policy.net_arch==[32,32] and agent.gamma==1 and agent.learning_rate==.0003,
                          'loaded_hyperparameters',destination)
                    check(agent.num_timesteps==record['episodes'] and agent._n_updates==record['gradient_updates'],
                          'loaded_checkpoint_counters',destination)
                metrics,values = episode(env,agent=agent,action=actions[0] if agent is None else None)
                check(np.array_equal(actions,values['actions']),'reloaded_actions',destination)
                check(np.array_equal(physical,values['physical_controls']),'reloaded_controls',destination)
                check(np.array_equal(audio,values['audio']),'reloaded_waveform_exact',destination)
                scalar(metrics['acoustic_error'],record['acoustic_error'],'reload_common_error',destination)
                scalar(metrics['weighted_loss'],record['weighted_loss'],'reload_weighted_loss',destination)
                reloads.append(dict(condition=c,method=m,seed=seed,envelope_weight=w,stage=stage,
                    error_difference=metrics['acoustic_error']-record['acoustic_error'],
                    max_waveform_abs_difference=float(abs(audio-values['audio']).max())))
        finally:
            accounts.append(dict(condition=c,method=m,seed=seed,envelope_weight=w,**account(env,time.perf_counter()-tick)))
            env.close()
        if (index+1)%6==0:
            print(json.dumps(dict(stage='audit_progress',runs=index+1,total=len(specs))),flush=True)
    for c in config['conditions']:
        for m in ('TD3','SAC'):
            for seed in config['seeds']:
                (a,ta),(b,tb) = runs[(c,m,seed,1.)],runs[(c,m,seed,.5)]
                path = output/'runs'/c/m
                check(a['effective_hyperparameters']==b['effective_hyperparameters'],'paired_effective_settings',path)
                check(a['initial']['learner_before']['policy_sha256']==b['initial']['learner_before']['policy_sha256'],
                      'paired_initial_policy',path)
                check([r['action'] for r in ta[:16]]==[r['action'] for r in tb[:16]],'paired_warmup_actions',path)
                check([r['components'] for r in ta[:16]]==[r['components'] for r in tb[:16]],'paired_initial_physics',path)
    result = dict(passed=not errors,checks=checks,errors=errors,completed_runs=len(found),
        protected_files_verified=verify_protected(output),reload_comparisons=reloads,
        separate_audit_generation=accounts,audit_generated_sounds=sum(a['episodes'] for a in accounts),
        audit_native_steps=sum(a['native_steps_attempted'] for a in accounts),audit_wall_s=time.perf_counter()-started,
        audit_scope='4096-candidate revision; legacy retirement and relocation recorded separately')
    destination = output/'audit.json' if output==WORK or not (output/'audit.json').exists() else Path('/tmp')/f'issue17-audit-{os.getpid()}.json'
    write_json(destination,result)
    if errors:
        raise RuntimeError(f'Issue 17 audit failed: {errors[:3]} (record: {destination})')
    print(json.dumps(dict(stage='audit_passed',checks=checks,runs=len(found),
                         replayed_sounds=result['audit_generated_sounds'],record=str(destination))),flush=True)
    return result
