"""Change Actor and twin Critic hidden widths only; preserve the original learner."""
from .result_paths import source_hashes_match
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .constant020_env import account, episode
from .issue14 import effective_hyperparameters, learner_signature
from .issue15_analysis import write_json
from .issue16 import digest
from .issue21_data import (OUT, METHODS, SEEDS, environment, ready, folder_for,
                           source_hashes, pack, metric_row, save_audio)


def make_agent(method, width, seed, env, config):
    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.noise import NormalActionNoise
    if method not in METHODS or width not in (32, 64, 128, 256):
        raise ValueError('Unregistered method or width')
    kwargs = dict(policy='MlpPolicy', env=env, seed=seed, device='cpu', verbose=0,
        policy_kwargs=dict(net_arch=[width, width]), learning_rate=config['learning_rate'],
        gamma=config['gamma'], batch_size=config['batch_size'], buffer_size=50000,
        learning_starts=config['warmup_episodes'], train_freq=1, gradient_steps=1)
    return (TD3(**kwargs, action_noise=NormalActionNoise(np.zeros(3),config['td3_noise_std']*np.ones(3)))
            if method=='TD3' else SAC(**kwargs, ent_coef=config['sac_ent_coef']))


def network_geometry(agent, width):
    """Inspect actual Linear layers, including target networks, not just net_arch."""
    import torch
    def linear(module):
        return [[m.in_features, m.out_features] for m in module.modules() if isinstance(m,torch.nn.Linear)]
    td3 = hasattr(agent, 'actor_target')
    actor = linear(agent.actor.mu if td3 else agent.actor.latent_pi)
    expected_actor = [[8,width],[width,width]]+([[width,3]] if td3 else [])
    assert actor == expected_actor, actor
    critics = [linear(q) for q in agent.critic.q_networks]
    assert critics == [[[11,width],[width,width],[width,1]]]*2, critics
    targets = [linear(q) for q in agent.critic_target.q_networks]
    assert targets == critics
    if td3:
        assert linear(agent.actor_target.mu) == actor
    else:
        assert linear(agent.actor.mu)==[[width,3]] and linear(agent.actor.log_std)==[[width,3]]
    return dict(actor=actor, Q1=critics[0], Q2=critics[1], critic_targets=targets,
        actor_parameters=sum(p.numel() for p in agent.actor.parameters()),
        critic_parameters=sum(p.numel() for p in agent.critic.parameters()))


def run_one(method, width, seed, output=OUT):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    torch.set_num_threads(1)
    output = Path(output); manifest = ready(output); config = manifest['settings']; budget=config['episodes']
    if width == 32:
        raise ValueError('The 32-wide baseline must never be retrained')
    folder=folder_for(method,width,seed,output)
    if (folder/'summary.json').exists():
        summary=json.loads((folder/'summary.json').read_text())
        assert source_hashes_match(summary['source_sha256'], source_hashes())
        for name,h in summary['artifact_sha256'].items():assert digest(folder/name)==h,name
        return summary
    folder.mkdir(parents=True,exist_ok=False)
    train=environment();evaluation=environment();curve=[];trace=[];snapshots={};best=None
    started=time.perf_counter();cpu_started=time.process_time();eval_wall=artifact_wall=0.
    hooks=[];steps={};numeric={}
    try:
        agent=make_agent(method,width,seed,train,config)
        effective=json.loads(json.dumps(effective_hyperparameters(agent)));geometry=network_geometry(agent,width)
        baseline=json.loads((folder_for(method,32,seed)/'summary.json').read_text())['effective_hyperparameters']
        assert dict(effective,net_arch=[32,32])==baseline, 'More than network width changed'
        write_json(folder/'config.json',dict(method=method,width=width,seed=seed,envelope_weight=1.,
            settings=config,effective_hyperparameters=effective,network_geometry=geometry,
            source_sha256=source_hashes(),target_sha256=manifest['target_sha256']))

        class FiniteLogger(Logger):
            def record(self,key,value,exclude=None):
                if isinstance(value,(float,int,np.number)):
                    if not np.isfinite(value):raise FloatingPointError(f'Nonfinite training metric {key}')
                    stat=numeric.setdefault(key,dict(count=0,minimum=float(value),maximum=float(value)))
                    stat.update(count=stat['count']+1,minimum=min(stat['minimum'],float(value)),maximum=max(stat['maximum'],float(value)))
                super().record(key,value,exclude)
        agent.set_logger(FiniteLogger(None,[]))
        for label,parent,attribute in (('actor',agent.actor,'optimizer'),('critic',agent.critic,'optimizer'),('entropy',agent,'ent_coef_optimizer')):
            optimizer=getattr(parent,attribute,None)
            if optimizer is None:continue
            steps[label]=0
            def pre(opt,args,kwargs):
                for group in opt.param_groups:
                    for param in group['params']:
                        if param.grad is not None and not torch.isfinite(param.grad).all():raise FloatingPointError('Nonfinite gradient')
            def post(opt,args,kwargs,label=label):
                steps[label]+=1
                for group in opt.param_groups:
                    for param in group['params']:
                        if not torch.isfinite(param).all():raise FloatingPointError('Nonfinite parameter')
            hooks.extend([optimizer.register_step_pre_hook(pre),optimizer.register_step_post_hook(post)])

        def evaluate(n):
            nonlocal best,eval_wall,artifact_wall
            before=learner_signature(agent);tick=time.perf_counter()
            metrics,values=episode(evaluation,agent=agent);eval_wall+=time.perf_counter()-tick
            after=learner_signature(agent)
            if before!=after:raise RuntimeError('Evaluation mutated learner, buffer or RNG')
            row=metric_row(metrics,values,n,gradient_updates=int(agent._n_updates),learner_before=before,learner_after=after)
            curve.append(row);snapshots['final']=values
            if best is None or row['weighted_loss']<best['weighted_loss']:
                best=row;snapshots['selected']=values;tick=time.perf_counter()
                agent.save(folder/'selected_agent.zip');artifact_wall+=time.perf_counter()-tick

        class Evaluation(BaseCallback):
            def _on_step(self):
                info=self.locals['infos'][0]
                trace.append(dict(episodes=self.num_timesteps,acoustic_error=info['acoustic_error'],
                    acoustic_components=info['acoustic_components'],weighted_loss=info['weighted_loss'],
                    failed=info['failed'],sustained_voiced=info['sustained_voiced'],
                    action=info['applied_action'],physical_controls=info['applied_controls'],
                    reward=float(self.locals['rewards'][0])))
                return True
            def _on_rollout_start(self):
                n=train.completed_episodes
                if 0<n<budget and n%config['eval_every_episodes']==0:evaluate(n)
                if n and n%512==0:print(json.dumps(dict(stage='progress',method=method,width=width,seed=seed,candidates=n)),flush=True)

        print(json.dumps(dict(stage='start',method=method,width=width,seed=seed)),flush=True)
        evaluate(0);tick=time.perf_counter();eval_before=eval_wall;artifact_before=artifact_wall
        agent.learn(total_timesteps=budget,callback=Evaluation())
        training_wall=time.perf_counter()-tick-(eval_wall-eval_before)-(artifact_wall-artifact_before)
        evaluate(budget)
        assert agent.num_timesteps==budget and agent._n_updates==budget-config['warmup_episodes']
        tick=time.perf_counter();agent.save(folder/'agent.zip');artifact_wall+=time.perf_counter()-tick
        for role,expected in (('final',curve[-1]),('selected',best)):
            tick=time.perf_counter();metrics,values=episode(evaluation,action=snapshots[role]['actions'][0]);eval_wall+=time.perf_counter()-tick
            assert metrics['acoustic_error']==expected['acoustic_error'] and metrics['weighted_loss']==expected['weighted_loss']
            assert np.array_equal(values['audio'],snapshots[role]['audio'])
            save_audio(folder/f'{role}.npz',values)
        pack(folder/'curve.json.gz',curve);pack(folder/'training_trace.json.gz',trace)
        files=('config.json','agent.zip','selected_agent.zip','final.npz','selected.npz','curve.json.gz','training_trace.json.gz')
        summary=dict(method=method,width=width,seed=seed,envelope_weight=1.,initial=curve[0],final=curve[-1],selected=best,
            training=account(train,training_wall),evaluation=account(evaluation,eval_wall),
            independent_policy_evaluations=len(curve),endpoint_replays=2,gradient_updates=int(agent._n_updates),
            optimizer_steps=steps,effective_hyperparameters=effective,network_geometry=geometry,
            numeric_training_metrics=numeric,nonfinite_detected=0,source_sha256=source_hashes(),
            target_sha256=manifest['target_sha256'],artifact_sha256={name:digest(folder/name) for name in files},
            artifact_wall_s=artifact_wall,elapsed_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu_started)
        write_json(folder/'summary.json',summary)
        print(json.dumps(dict(stage='complete',method=method,width=width,seed=seed,final=curve[-1]['acoustic_error'],selected=best['acoustic_error'])),flush=True)
        return summary
    except Exception as exc:
        write_json(folder/'failure.json',dict(type=type(exc).__name__,message=str(exc),
            training=account(train,time.perf_counter()-started),evaluation=account(evaluation,eval_wall)))
        raise
    finally:
        for hook in hooks:hook.remove()
        train.close();evaluation.close()


def train(workers=18,output=OUT):
    output=Path(output);ready(output)
    if (output/'execution.json').exists():
        print('Completed execution preserved; use audit/report for existing results',flush=True)
        return
    tick=time.perf_counter();summaries=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,m,w,s,output) for w in (64,128,256) for m in METHODS for s in SEEDS]
        for f in as_completed(futures):summaries.append(f.result())
    write_json(output/'execution.json',dict(runs=len(summaries),workers=workers,wall_s=time.perf_counter()-tick,
        training_candidates=sum(s['training']['episodes'] for s in summaries),
        independent_policy_evaluations=sum(s['independent_policy_evaluations'] for s in summaries),
        endpoint_replays=sum(s['endpoint_replays'] for s in summaries),
        training_process_wall_s=sum(s['training']['wall_s'] for s in summaries),
        evaluation_process_wall_s=sum(s['evaluation']['wall_s'] for s in summaries),
        process_cpu_s=sum(s['process_cpu_s'] for s in summaries)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=18)
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();train(a.workers,a.output)
