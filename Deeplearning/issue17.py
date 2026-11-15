"""Issue 17: fixed-budget, one-action Case A comparison at 0.20 seconds.

Staged preparation (including a physical gate), 1D/2D then gated 3D, independent
audit/report, and atomic publication. No legacy experiment file is modified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
from scipy.io import wavfile
from .fixed_budget_de import fixed_budget_de

from .constant020_env import Constant020Env, INDICES, RATE, account, episode
from .issue15_analysis import write_json
from .issue16 import digest, protected_snapshot, verify_protected
from .model import LIBRARY, LOW, HIGH

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT/'configs/issue17_constant020.json'
WORK = ROOT/'.issue17-work/artifacts4096'
DEST = ROOT/'results/phase1_constant_020s'
SOURCES = ('constant020_env.py','issue17.py','issue17_diagnostics.py','model.py','model_env.py',
           'native/model.cpp','acoustics.py','configs/issue17_constant020.json','issue14.py','fixed_budget_de.py')
OLD_PHASE1 = ('phase1','phase1_extended','phase1_terminal_baselines',
              'phase1_validation_20261031','phase1_issue13_20261107','phase1_constant_020s')


def weight_label(weight):
    return 'lambda1' if weight == 1 else 'lambda0p5'


def run_specs(config):
    return [(c,m,s,w) for c in config['conditions'] for m in config['methods']
            for w in (config['rl_envelope_weights'] if m in ('TD3','SAC') else [1.0])
            for s in config['seeds']]


def source_hashes():
    return {name:digest(ROOT/name) for name in SOURCES}


def save_snapshot(folder, values, target, controls):
    folder.mkdir(parents=True, exist_ok=True)
    scale = float(np.max(np.abs(target)))
    np.savez_compressed(folder/'rollout.npz', **values, target=target,
                        target_controls=np.tile(controls,(20,1)), rate=RATE, duration_s=.2, listening_gain=1/scale)
    wavfile.write(folder/'target.wav',RATE,(target/scale).astype(np.float32))
    wavfile.write(folder/'generated.wav',RATE,(values['audio']/scale).astype(np.float32))


def prepare():
    if WORK.exists():
        raise FileExistsError('Existing staged results are preserved')
    prior=json.loads((DEST/'manifest.json').read_text())
    if prior['settings']['episodes'] != 256:
        raise RuntimeError('Only the explicitly superseded 256-candidate pilot can be replaced')
    settings = json.loads(CONFIG.read_text())
    old = json.loads((ROOT/'configs/issue13_reproduction.json').read_text())
    for key in ('seeds','warmup_episodes','learning_rate','batch_size','net_arch',
                'gamma','sac_ent_coef','td3_noise_std','de_population','de_mutation','de_recombination'):
        assert settings[key] == old[key], key
    assert settings['episodes'] == 4096 and settings['eval_every_episodes'] == 128
    assert len(run_specs(settings)) == 54
    WORK.mkdir(parents=True)
    write_json(WORK/'config.json',settings)
    inventory=protected_snapshot()
    replaceable=lambda n:any(n.startswith('results/'+p+'/') for p in OLD_PHASE1)
    write_json(WORK/'protected_before.json',{n:h for n,h in inventory.items() if not replaceable(n)})
    write_json(WORK/'phase1_inventory_before.json',{n:dict(sha256=h,bytes=(ROOT/n).stat().st_size)
               for n,h in inventory.items() if replaceable(n)})
    from .issue17_diagnostics import prepare_diagnostics
    review=prepare_diagnostics(WORK/'diagnostics')
    with np.load(DEST/'diagnostics/target.npz') as old_target, np.load(WORK/'diagnostics/target.npz') as fresh:
        assert np.array_equal(old_target['target'],fresh['target'])
        assert np.array_equal(old_target['controls'],fresh['controls'])
    import shutil
    shutil.copy2(DEST/'diagnostics/target.npz',WORK/'diagnostics/target.npz')
    write_json(WORK/'manifest.json',dict(issue=17, created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        settings=settings, source_sha256=source_hashes(), native_library_sha256=digest(LIBRARY),
        versions={name:importlib.metadata.version(name) for name in
                  ('numpy','scipy','torch','stable-baselines3','gymnasium','matplotlib','SciencePlots')},
        platform=platform.platform(), processor=platform.processor(), device='cpu', torch_threads_per_worker=1,
        target_sha256=digest(WORK/'diagnostics/target.npz'), target_controls=review['controls'],
        control_order=['epsilon_l','epsilon_r','ps','f0_l','f0_r'], low=LOW.tolist(), high=HIGH.tolist(),
        reward='-(full E_spec + lambda * full E_env), gamma=1; RL lambda=1/0.5, DE/Random lambda=1 only',
        failure_policy='keep failed candidate, pad diagnostic waveform, reward -1e6; no retries/exclusion',
        extreme_reward_policy='finite rewards preserved without clipping; nonfinite loss/gradients/parameters abort and retain failure log',
        native_failure_accounting='attempted steps count entire requested 10 ms block; completed steps exclude interrupted block',
        condition_order='all dimensions passed physical gates before training; exact fixed budget for all 54 runs',
        supersedes=dict(experiment=prior['settings']['experiment_id'],episodes=256,
                        pilot_manifest_sha256=digest(DEST/'manifest.json'),pilot_target_sha256=prior['target_sha256']),
        endpoint='final deterministic policy or actual candidate 4096; DE incumbent/best are auxiliary'))
    print(json.dumps(dict(stage='prepared',passed=review['passed'],proceed_3d=review['proceed_3d'],
                         target_controls=review['controls'])),flush=True)


def load_ready(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['source_sha256'] != source_hashes() or manifest['native_library_sha256'] != digest(LIBRARY):
        raise RuntimeError('Frozen experiment sources or native library changed')
    review=json.loads((output/'diagnostics/review.json').read_text())
    if not review['passed']:
        raise RuntimeError('Physical gate did not pass')
    with np.load(output/'diagnostics/target.npz') as data:
        target, controls=data['target'].copy(),data['controls'].copy()
    return manifest,target,controls


def run_one(output, condition, method, seed, envelope_weight=1.0):
    import torch
    from stable_baselines3 import TD3,SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.noise import NormalActionNoise
    from .issue14 import effective_hyperparameters, learner_signature
    torch.set_num_threads(1)
    manifest,target,controls=load_ready(output)
    config=manifest['settings']; budget=config['episodes']; indices=INDICES[condition]
    if method not in ('TD3','SAC') and envelope_weight != 1:
        raise ValueError('Direct baselines use lambda=1 only')
    folder=output/'runs'/condition/method/weight_label(envelope_weight)/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'config.json',dict(condition=condition,method=method,seed=seed,envelope_weight=envelope_weight,settings=config,
               indices=list(indices),source_sha256=manifest['source_sha256'],target_sha256=manifest['target_sha256']))
    train_env=Constant020Env(indices,target,controls,envelope_weight=envelope_weight)
    eval_env=Constant020Env(indices,target,controls,envelope_weight=envelope_weight)
    started=time.perf_counter(); cpu_started=time.process_time()
    curve=[]; trace=[]; snapshots={}; best=None; agent=None; hooks=[]
    eval_wall=artifact_wall=0.; optimizer_steps={}; numeric={}

    def record(n,metrics,values,phase):
        nonlocal best,artifact_wall
        row=dict(episodes=n,**metrics,evaluation_phase=phase,
                 gradient_updates=int(agent._n_updates) if agent is not None else 0,
                 actions=values['actions'].tolist(),physical_controls=values['physical_controls'][0].tolist())
        row['best_observed_error']=min(metrics['acoustic_error'],best['acoustic_error'] if best else float('inf'))
        curve.append(row)
        snapshots['final']={key:value.copy() for key,value in values.items()}
        if len(curve)==1:
            snapshots['initial']=snapshots['final']
        if best is None or metrics['acoustic_error'] < best['acoustic_error']:
            best=row.copy(); snapshots['best']=snapshots['final']
            if agent is not None:
                tick=time.perf_counter(); (folder/'best').mkdir(exist_ok=True)
                agent.save(folder/'best/agent.zip'); artifact_wall+=time.perf_counter()-tick
        if n in config['checkpoints'] and n < budget:
            tick=time.perf_counter(); checkpoint=folder/'checkpoints'/str(n)
            checkpoint.mkdir(parents=True)
            if agent is not None:agent.save(checkpoint/'agent.zip')
            save_snapshot(checkpoint,values,target,controls)
            write_json(checkpoint/'metrics.json',row)
            artifact_wall+=time.perf_counter()-tick
        if n and n%128==0:
            write_json(folder/'progress.json',dict(episodes=n,error=metrics['acoustic_error'],
                envelope_weight=envelope_weight,gradient_updates=row['gradient_updates'],
                elapsed_s=time.perf_counter()-started,native_blocks=train_env.native_blocks))
            if n%512==0:print(json.dumps(dict(stage='progress',condition=condition,method=method,
                seed=seed,weight=envelope_weight,trials=n,error=metrics['acoustic_error'])),flush=True)
        return row

    def evaluate(n,phase,action=None):
        nonlocal eval_wall
        before=learner_signature(agent) if agent else None
        tick=time.perf_counter()
        metrics,values=episode(eval_env,agent=agent,action=action)
        eval_wall+=time.perf_counter()-tick
        after=learner_signature(agent) if agent else None
        if before != after:
            raise RuntimeError('Evaluation mutated training state or RNG')
        row=record(n,metrics,values,phase)
        if agent:
            row['learner_before']=before;row['learner_after']=after
        return row

    class FiniteLogger(Logger):
        def record(self,key,value,exclude=None):
            if isinstance(value,(float,int,np.number)):
                if not np.isfinite(value): raise FloatingPointError(f'Nonfinite training metric {key}')
                stat=numeric.setdefault(key,dict(count=0,minimum=float(value),maximum=float(value)))
                stat.update(count=stat['count']+1,minimum=min(stat['minimum'],float(value)),maximum=max(stat['maximum'],float(value)))
            super().record(key,value,exclude)

    print(json.dumps(dict(stage='start',condition=condition,method=method,seed=seed,weight=envelope_weight)),flush=True)
    try:
        if method in ('TD3','SAC'):
            kwargs=dict(policy='MlpPolicy',env=train_env,seed=seed,device='cpu',verbose=0,
                        policy_kwargs=dict(net_arch=config['net_arch']),learning_rate=config['learning_rate'],
                        gamma=config['gamma'],batch_size=config['batch_size'],buffer_size=50000,
                        learning_starts=config['warmup_episodes'],train_freq=1,gradient_steps=1)
            if method=='TD3':
                agent=TD3(**kwargs,action_noise=NormalActionNoise(np.zeros(len(indices)),config['td3_noise_std']*np.ones(len(indices))))
            else:
                agent=SAC(**kwargs,ent_coef=config['sac_ent_coef'])
            effective=effective_hyperparameters(agent)
            agent.set_logger(FiniteLogger(None,[]))
            for label,parent,attribute in (('actor',agent.actor,'optimizer'),('critic',agent.critic,'optimizer'),('entropy',agent,'ent_coef_optimizer')):
                optimizer=getattr(parent,attribute,None)
                if optimizer is None: continue
                optimizer_steps[label]=0
                def pre(opt,args,kwargs):
                    for group in opt.param_groups:
                        for param in group['params']:
                            if param.grad is not None and not torch.isfinite(param.grad).all():
                                raise FloatingPointError('Nonfinite gradient')
                def post(opt,args,kwargs,label=label):
                    optimizer_steps[label]+=1
                    for group in opt.param_groups:
                        for param in group['params']:
                            if not torch.isfinite(param).all(): raise FloatingPointError('Nonfinite parameter')
                hooks.extend([optimizer.register_step_pre_hook(pre),optimizer.register_step_post_hook(post)])
            evaluate(0,'initial')
            tick=time.perf_counter(); (folder/'initial').mkdir(exist_ok=True)
            agent.save(folder/'initial/agent.zip'); artifact_wall+=time.perf_counter()-tick

            class SparseEvaluation(BaseCallback):
                def _on_step(self):
                    info=self.locals['infos'][0]
                    trace.append(dict(episodes=self.num_timesteps,acoustic_error=info['acoustic_error'],
                        components=info['acoustic_components'],failed=info['failed'],
                        weighted_loss=info['weighted_loss'],sustained_voiced=info['sustained_voiced'],
                        parameter_nrmse=info['parameter_nrmse'],
                        action=info['applied_action'],
                        reward=float(self.locals['rewards'][0]),
                        physical_controls=info['applied_controls']))
                    return True

                def _on_rollout_start(self):
                    n=train_env.completed_episodes
                    if 0<n<budget and n%config['eval_every_episodes']==0:
                        evaluate(n,'post_gradient_update')

            tick=time.perf_counter(); eval_before,artifact_before=eval_wall,artifact_wall
            agent.learn(total_timesteps=budget,callback=SparseEvaluation())
            training_wall=time.perf_counter()-tick-(eval_wall-eval_before)-(artifact_wall-artifact_before)
            evaluate(budget,'post_final_update')
            assert agent.num_timesteps==budget
            assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
            gradient_updates=int(agent._n_updates)
            optimizer_nfev=0
        else:
            effective=None; gradient_updates=0; optimizer_nfev=0
            rng=np.random.default_rng(seed)
            initial=rng.uniform(-1,1,len(indices))
            # Initial is the first counted candidate (no free objective calls).
            def candidate(action):
                nonlocal optimizer_nfev
                metrics,values=episode(train_env,action=np.asarray(action))
                optimizer_nfev+=1
                row=record(optimizer_nfev,metrics,values,'actual_training_candidate')
                return metrics['acoustic_error']
            tick=time.perf_counter()
            if method=='random_search':
                candidate(initial)
                for _ in range(budget-1): candidate(rng.uniform(-1,1,len(indices)))
            elif method=='DE':
                result=fixed_budget_de(candidate,len(indices),rng,initial,budget=budget,
                    population_size=config['de_population'],mutation=config['de_mutation'],recombination=config['de_recombination'])
                assert result['nfev']==budget
                write_json(folder/'de_selected_incumbent.json',result)
            else: raise ValueError(method)
            training_wall=time.perf_counter()-tick-artifact_wall
            assert optimizer_nfev==budget
            final=curve[-1]; before=final['acoustic_error']
            tick=time.perf_counter()
            metrics,values=episode(eval_env,action=snapshots['final']['actions'][0])
            eval_wall+=time.perf_counter()-tick
            assert metrics['acoustic_error']==before
            independent_final=dict(**metrics,actions=values['actions'].tolist(),
                                   physical_controls=values['physical_controls'][0].tolist())
            # Keep the last candidate as the primary endpoint; do not replace it
            # with the optimizer incumbent or minimum of previous candidates.
            write_json(folder/'final_candidate_replay.json',independent_final)
            tick=time.perf_counter()
            best_metrics,_=episode(eval_env,action=snapshots['best']['actions'][0])
            eval_wall+=time.perf_counter()-tick
            assert best_metrics['acoustic_error']==best['acoustic_error']
            write_json(folder/'best_candidate_replay.json',best_metrics)

        tick=time.perf_counter()
        if agent is not None: agent.save(folder/'agent.zip')
        for stage,values in snapshots.items():
            save_snapshot(folder if stage=='final' else folder/stage,values,target,controls)
        write_json(folder/'curve.json',curve)
        if agent is not None:write_json(folder/'training_trace.json',trace)
        checkpoint=folder/'checkpoints'/str(budget);checkpoint.mkdir(parents=True)
        write_json(checkpoint/'metrics.json',dict(curve[-1],artifact_directory='../..',final_artifacts_reused=True))
        artifact_wall+=time.perf_counter()-tick
        summary=dict(condition=condition,method=method,seed=seed,envelope_weight=envelope_weight,optimizer_dimension=len(indices),
            initial=curve[0],final=curve[-1],best=best,training=account(train_env,training_wall),
            evaluation=account(eval_env,eval_wall),training_candidates=budget,
            gradient_updates=gradient_updates,optimizer_steps=optimizer_steps,optimizer_nfev=optimizer_nfev,
            effective_hyperparameters=effective,numeric_training_metrics=numeric,nonfinite_detected=0,
            artifact_wall_s=artifact_wall,elapsed_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu_started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            returned_solution='final deterministic actor' if agent else f'actual candidate {budget} (not best-so-far)',
            target_sha256=manifest['target_sha256'],source_sha256=manifest['source_sha256'],
            model_sha256=digest(folder/'agent.zip') if agent else None,rollout_sha256=digest(folder/'rollout.npz'))
        write_json(folder/'summary.json',summary)
        print(json.dumps(dict(stage='complete',condition=condition,method=method,seed=seed,
                             weight=envelope_weight,
                             final=summary['final']['acoustic_error'],best=best['acoustic_error'],elapsed_s=summary['elapsed_s'])),flush=True)
        return summary
    except Exception as exc:
        write_json(folder/'failure.json',dict(type=type(exc).__name__,message=str(exc),
            training=account(train_env,time.perf_counter()-started),evaluation=account(eval_env,eval_wall)))
        raise
    finally:
        for hook in hooks: hook.remove()
        train_env.close();eval_env.close()


def train(workers):
    manifest,_,_=load_ready(WORK); config=manifest['settings']
    started=time.perf_counter(); summaries=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,WORK,c,m,s,w) for c,m,s,w in run_specs(config)]
        for future in as_completed(futures):
            summaries.append(future.result());write_json(WORK/'summary.json',summaries)
    write_json(WORK/'execution.json',dict(completed_runs=len(summaries),workers=workers,
        concurrent_run_wall_s=time.perf_counter()-started,
        training_candidates=sum(s['training']['episodes'] for s in summaries),
        evaluation_sounds=sum(s['evaluation']['episodes'] for s in summaries),
        training_process_wall_s=sum(s['training']['wall_s'] for s in summaries),
        evaluation_process_wall_s=sum(s['evaluation']['wall_s'] for s in summaries),
        save_process_wall_s=sum(s['artifact_wall_s'] for s in summaries),
        process_cpu_s=sum(s['process_cpu_s'] for s in summaries)))


def finish():
    from .issue17_audit import audit
    from .issue17_report import report
    audit(WORK)
    report(WORK)
    verify_protected(WORK)
    from .issue17_cleanup import publish
    publish(WORK,DEST)
    print(f'Published audited results: {DEST}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','finish','audit'))
    parser.add_argument('--workers',type=int,default=27)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='train':train(args.workers)
    elif args.action=='finish':finish()
    else:
        from .issue17_audit import audit
        audit(WORK if WORK.exists() else DEST)
