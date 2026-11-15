"""Independent saved-artifact and model-reload audit; no training calls."""
import json
import time
import numpy as np
from scipy.io import wavfile

from .acoustics import distance
from .constant020_env import Constant020Env,INDICES,RATE,account,episode
from .issue15_analysis import write_json
from .issue16 import digest,verify_protected
from .issue17 import load_ready
from .model import LOW,HIGH


def audit(output):
    import torch
    from stable_baselines3 import TD3,SAC
    torch.set_num_threads(1)
    started=time.perf_counter();errors=[];checks=0;reloads=[];accounts=[]
    manifest,target,controls=load_ready(output)
    config=manifest['settings'];scale=float(abs(target).max())

    def check(value,code,path,actual=None):
        nonlocal checks
        checks+=1
        if not value:errors.append(dict(code=code,path=str(path.relative_to(output)),actual=actual))

    def close(a,b,code,path,atol=1e-12):
        check(bool(np.isfinite(a) and np.isfinite(b) and abs(a-b)<=atol),code,path,
              dict(actual=float(a),expected=float(b)))

    paths=sorted((output/'runs').glob('*/*/seed_*/summary.json'))
    check(len(paths)==36,'36_runs',output, len(paths))
    for path in paths:
        folder=path.parent;s=json.loads(path.read_text());c=s['condition'];m=s['method'];seed=s['seed']
        curve=json.loads((folder/'curve.json').read_text())
        trace=json.loads((folder/'training_trace.json').read_text())
        run_config=json.loads((folder/'config.json').read_text())
        check(run_config['settings']==config and run_config['target_sha256']==manifest['target_sha256'],
              'shared_config_and_target',folder)
        check(s['source_sha256']==manifest['source_sha256'],'same_numerical_sources',folder)
        check(len(trace)==256 and [r['episodes'] for r in trace]==list(range(1,257)),'training_budget_and_schedule',folder)
        check(s['training']['episodes']==s['training']['decision_steps']==s['training_candidates']==256,'one_decision_budget',folder)
        check(s['training']['native_steps_attempted']==s['training']['native_blocks_attempted']*100000,
              'native_budget_separate',folder)
        if s['training']['failures']==0:
            check(s['training']['native_blocks_completed']==s['training']['native_blocks_attempted']==5120,
                  '20_native_blocks_per_candidate',folder)
        check(s['nonfinite_detected']==0,'finite_training',folder)
        for r in trace:
            check(np.isfinite(r['acoustic_error']),'finite_candidate_error',folder)
            if m in ('TD3','SAC'):
                close(r['reward'],-r['acoustic_error'],'full_episode_reward',folder,atol=max(1e-7,abs(r['acoustic_error'])*1e-7))
        check(s['initial']==curve[0] and s['final']==curve[-1],'summary_curve_endpoints',folder)
        best=min(curve,key=lambda r:r['acoustic_error'])
        close(s['best']['acoustic_error'],best['acoustic_error'],'best_observed',folder)
        check(s['best']['episodes']==best['episodes'],'best_step',folder)
        check(digest(folder/'rollout.npz')==s['rollout_sha256'],'final_rollout_hash',folder)
        if m in ('TD3','SAC'):
            check([r['episodes'] for r in curve]==list(range(0,257,8)),'sparse_eval_schedule',folder)
            check(s['evaluation']['episodes']==33,'extra_evaluation_budget',folder)
            check(s['gradient_updates']==240,'gradient_iterations',folder)
            expected=dict(actor=120,critic=240) if m=='TD3' else dict(actor=240,critic=240,entropy=240)
            check(s['optimizer_steps']==expected,'actual_optimizer_calls',folder)
            for r in curve:
                check(r['learner_before']==r['learner_after'],'evaluation_keeps_learner_rng',folder)
                n=r['episodes'];expected_updates=240 if n==256 else max(0,n-17)
                check(r['gradient_updates']==expected_updates,'evaluation_update_timing',folder)
            check(curve[-1]['evaluation_phase']=='post_final_update','final_post_update',folder)
            effective=s['effective_hyperparameters']
            for key,value in dict(learning_rate=.0003,gamma=1.,batch_size=32,learning_starts=16,
                                  buffer_size=50000,gradient_steps=1,seed=seed,torch_threads=1).items():
                close(effective[key],value,'effective_'+key,folder)
            check(effective['net_arch']==[32,32] and effective['device']=='cpu','architecture_device',folder)
            if m=='SAC':close(effective['initial_entropy_coefficient'],.01,'initial_entropy',folder,atol=1e-8)
            check(digest(folder/'agent.zip')==s['model_sha256'],'model_hash',folder)
        else:
            check(len(curve)==256 and s['optimizer_nfev']==256 and s['evaluation']['episodes']==1,
                  'direct_candidate_evaluation_budget',folder)
            check(s['final']['episodes']==256 and s['final']['evaluation_phase']=='actual_training_candidate',
                  'actual_last_candidate_primary',folder)
            minimum=np.minimum.accumulate([r['acoustic_error'] for r in curve])
            check(np.array_equal(minimum,[r['best_observed_error'] for r in curve]),'actual_vs_best_so_far',folder)
            if m=='DE':
                selected=json.loads((folder/'de_selected_incumbent.json').read_text())
                close(selected['error'],s['best']['acoustic_error'],'de_incumbent_auxiliary',folder)
                check(selected['nfev']==256,'de_no_early_termination',folder)
        # Replay all three distinct saved stages, using the saved model for RL
        # and saved actions for direct search. This runs in a separate process
        # from the training pool and recomputes metrics from saved raw audio.
        env=Constant020Env(INDICES[c],target,controls)
        tick=time.perf_counter()
        try:
            for stage,record in (('initial',s['initial']),('best',s['best']),('final',s['final'])):
                destination=folder if stage=='final' else folder/stage
                with np.load(destination/'rollout.npz') as data:
                    audio=data['audio'].copy();actions=data['actions'].copy();physical=data['physical_controls'].copy()
                    check(np.array_equal(data['target'],target),'same_target_audio',destination)
                    check(int(data['rate'])==50000 and float(data['duration_s'])==.2 and audio.shape==(10000,),
                          'duration_rate_metadata',destination)
                    check(np.array_equal(data['target_controls'],np.tile(controls,(20,1))),
                          'same_target_physical_controls',destination)
                check(np.isfinite(audio).all(),'finite_audio',destination)
                check(physical.shape==(20,5) and np.array_equal(physical,np.tile(physical[0],(20,1))),
                      'constant_physical_controls',destination)
                check(np.all(physical>=LOW) and np.all(physical<=HIGH),'physical_bounds',destination)
                other=[i for i in range(5) if i not in INDICES[c]]
                check(np.array_equal(physical[:,other],np.tile(controls[other],(20,1))),
                      'unlearned_controls_fixed_to_target',destination)
                calc=distance(target,audio,RATE,scale=scale)
                if not record['failed']:
                    close(calc['total'],record['acoustic_error'],'saved_waveform_metric',destination)
                for key in ('log_spectral','envelope','total'):
                    close(calc[key],record['acoustic_components'][key],'saved_component_'+key,destination)
                rate,wav=wavfile.read(destination/'generated.wav')
                check(rate==RATE and np.array_equal(wav,(audio/scale).astype(np.float32)),
                      'wav_common_target_gain',destination)
                agent=None
                if m in ('TD3','SAC'):
                    agent={'TD3':TD3,'SAC':SAC}[m].load(destination/'agent.zip',device='cpu')
                    check(all(torch.isfinite(p).all() for p in agent.policy.parameters()),'finite_loaded_parameters',destination)
                    check(agent.policy.net_arch==[32,32] and agent.gamma==1 and agent.learning_rate==.0003,
                          'loaded_settings',destination)
                    for label in ('actor','critic'):
                        check(all(g['lr']==.0003 for g in getattr(agent,label).optimizer.param_groups),
                              'loaded_optimizer_lr',destination)
                metrics,values=episode(env,agent=agent,action=actions[0] if agent is None else None)
                check(np.array_equal(actions,values['actions']),'reload_actions',destination)
                check(np.array_equal(physical,values['physical_controls']),'reload_controls',destination)
                check(np.array_equal(audio,values['audio']),'reload_waveform_exact',destination)
                close(metrics['acoustic_error'],record['acoustic_error'],'reload_error',destination)
                reloads.append(dict(condition=c,method=m,seed=seed,stage=stage,
                    error_difference=metrics['acoustic_error']-record['acoustic_error'],
                    max_waveform_abs_difference=float(abs(audio-values['audio']).max())))
        finally:
            accounts.append(dict(condition=c,method=m,seed=seed,**account(env,time.perf_counter()-tick)))
            env.close()
    result=dict(passed=not errors,checks=checks,errors=errors,completed_runs=len(paths),
        protected_files_verified=verify_protected(output),reload_comparisons=reloads,
        separate_audit_generation=accounts,audit_generated_sounds=sum(a['episodes'] for a in accounts),
        audit_native_steps=sum(a['native_steps_attempted'] for a in accounts),audit_wall_s=time.perf_counter()-started)
    write_json(output/'audit.json',result)
    if errors:raise RuntimeError(f'Issue 17 audit failed: {errors[:3]}')
    print(json.dumps(dict(stage='audit_passed',checks=checks,runs=len(paths),replayed_sounds=result['audit_generated_sounds'])),flush=True)
    return result
