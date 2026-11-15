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
from scipy.optimize import differential_evolution

from .constant020_env import Constant020Env, INDICES, RATE, account, episode
from .issue15_analysis import write_json
from .issue16 import digest, protected_snapshot, verify_protected
from .model import LIBRARY, LOW, HIGH

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT/'configs/issue17_constant020.json'
WORK = ROOT/'.issue17-work/artifacts'
DEST = ROOT/'results/phase1_constant_020s'
SOURCES = ('constant020_env.py','issue17.py','issue17_diagnostics.py','model.py','model_env.py',
           'native/model.cpp','acoustics.py','configs/issue17_constant020.json','issue14.py')


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
    if WORK.exists() or DEST.exists():
        raise FileExistsError('Existing staged/published results are preserved')
    settings = json.loads(CONFIG.read_text())
    old = json.loads((ROOT/'configs/issue13_reproduction.json').read_text())
    for key in ('seeds','episodes','warmup_episodes','learning_rate','batch_size','net_arch',
                'gamma','sac_ent_coef','td3_noise_std','de_population','de_mutation','de_recombination'):
        assert settings[key] == old[key], key
    assert settings['episodes'] == 256 and settings['eval_every_episodes'] == 8
    WORK.mkdir(parents=True)
    write_json(WORK/'config.json',settings)
    write_json(WORK/'protected_before.json',protected_snapshot())
    from .issue17_diagnostics import prepare_diagnostics
    review=prepare_diagnostics(WORK/'diagnostics')
    write_json(WORK/'manifest.json',dict(issue=17, created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        settings=settings, source_sha256=source_hashes(), native_library_sha256=digest(LIBRARY),
        versions={name:importlib.metadata.version(name) for name in
                  ('numpy','scipy','torch','stable-baselines3','gymnasium','matplotlib','SciencePlots')},
        platform=platform.platform(), processor=platform.processor(), device='cpu', torch_threads_per_worker=1,
        target_sha256=digest(WORK/'diagnostics/target.npz'), target_controls=review['controls'],
        control_order=['epsilon_l','epsilon_r','ps','f0_l','f0_r'], low=LOW.tolist(), high=HIGH.tolist(),
        reward='negative full 0.20 s acoustics.distance (log PSD + 5 ms RMS envelope), gamma=1',
        failure_policy='keep failed candidate, pad diagnostic waveform, reward -1e6; no retries/exclusion',
        extreme_reward_policy='finite rewards preserved without clipping; nonfinite loss/gradients/parameters abort and retain failure log',
        native_failure_accounting='attempted steps count entire requested 10 ms block; completed steps exclude interrupted block',
        condition_order='finish 1D and 2D cohorts before gated 3D; no budget extension',
        endpoint='final deterministic policy or actual candidate 256; DE incumbent/best are auxiliary'))
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


def run_one(output, condition, method, seed):
    import torch
    from stable_baselines3 import TD3,SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.noise import NormalActionNoise
    from .issue14 import effective_hyperparameters, learner_signature
    torch.set_num_threads(1)
    manifest,target,controls=load_ready(output)
    config=manifest['settings']; budget=config['episodes']; indices=INDICES[condition]
    folder=output/'runs'/condition/method/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'config.json',dict(condition=condition,method=method,seed=seed,settings=config,
               indices=list(indices),source_sha256=manifest['source_sha256'],target_sha256=manifest['target_sha256']))
    train_env=Constant020Env(indices,target,controls)
    eval_env=Constant020Env(indices,target,controls)
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

    print(json.dumps(dict(stage='start',condition=condition,method=method,seed=seed)),flush=True)
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
                        action=info['applied_action'],
                        reward=float(self.locals['rewards'][0]),
                        physical_controls=info['applied_controls']))
                    return True

                def _on_rollout_end(self):
                    n=train_env.completed_episodes
                    if 0<n<budget and n%config['eval_every_episodes']==0:
                        evaluate(n,'pre_gradient_update')

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
                trace.append(dict(row))
                return metrics['acoustic_error']
            tick=time.perf_counter()
            if method=='random_search':
                candidate(initial)
                for _ in range(budget-1): candidate(rng.uniform(-1,1,len(indices)))
            elif method=='DE':
                pop=config['de_population']; population=rng.uniform(-1,1,(pop,len(indices))); population[0]=initial
                result=differential_evolution(candidate,[(-1,1)]*len(indices),init=population,
                    maxiter=budget//pop-1,polish=False,tol=0,atol=0,
                    mutation=tuple(config['de_mutation']),recombination=config['de_recombination'],rng=rng)
                assert result.nfev==budget
                write_json(folder/'de_selected_incumbent.json',dict(action=result.x.tolist(),error=float(result.fun),
                    nfev=int(result.nfev),nit=int(result.nit),success=bool(result.success),message=str(result.message),
                    role='DE selected incumbent; auxiliary, not relabeled as actual candidate 256'))
            else: raise ValueError(method)
            training_wall=time.perf_counter()-tick
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
        if agent is not None: agent.save(folder/'agent.zip')
        for stage,values in snapshots.items():
            save_snapshot(folder if stage=='final' else folder/stage,values,target,controls)
        write_json(folder/'curve.json',curve)
        write_json(folder/'training_trace.json',trace)
        artifact_wall+=time.perf_counter()-tick
        summary=dict(condition=condition,method=method,seed=seed,optimizer_dimension=len(indices),
            initial=curve[0],final=curve[-1],best=best,training=account(train_env,training_wall),
            evaluation=account(eval_env,eval_wall),training_candidates=budget,
            gradient_updates=gradient_updates,optimizer_steps=optimizer_steps,optimizer_nfev=optimizer_nfev,
            effective_hyperparameters=effective,numeric_training_metrics=numeric,nonfinite_detected=0,
            artifact_wall_s=artifact_wall,elapsed_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu_started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            returned_solution='final deterministic actor' if agent else 'actual candidate 256 (not best-so-far)',
            target_sha256=manifest['target_sha256'],source_sha256=manifest['source_sha256'],
            model_sha256=digest(folder/'agent.zip') if agent else None,rollout_sha256=digest(folder/'rollout.npz'))
        write_json(folder/'summary.json',summary)
        print(json.dumps(dict(stage='complete',condition=condition,method=method,seed=seed,
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
    started=time.perf_counter(); summaries=[]; cohorts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for conditions in (('constant_1d','constant_2d'),('constant_3d',)):
            if conditions==('constant_3d',):
                assert len(summaries)==24 and json.loads((WORK/'diagnostics/review.json').read_text())['proceed_3d']
            tick=time.perf_counter()
            futures=[pool.submit(run_one,WORK,c,m,s) for c in conditions for m in config['methods'] for s in config['seeds']]
            for future in as_completed(futures):
                summaries.append(future.result());write_json(WORK/'summary.json',summaries)
            cohorts.append(dict(conditions=list(conditions),wall_s=time.perf_counter()-tick))
    write_json(WORK/'execution.json',dict(completed_runs=len(summaries),cohorts=cohorts,
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
    if DEST.exists(): raise FileExistsError(DEST)
    audit(WORK)
    report(WORK)
    verify_protected(WORK)
    WORK.rename(DEST)
    print(f'Published audited results: {DEST}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','finish','audit'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='train':train(args.workers)
    elif args.action=='finish':finish()
    else:
        from .issue17_audit import audit
        audit(DEST if DEST.exists() else WORK)
