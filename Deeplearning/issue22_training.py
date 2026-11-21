"""Nine SAC128 runs; change the specified sum-one terminal objective only."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .constant020_env import episode,account
from .issue14 import effective_hyperparameters,learner_signature
from .issue21_training import make_agent,network_geometry
from .issue22_data import (OUT,WEIGHTS,SEEDS,environment,ready,folder_for,label,
    source_hashes,pack,metric_row,save_audio,policy_statistics,coefficients)
from .issue15_analysis import write_json
from .issue16 import digest


def run_one(weight,seed,output=OUT):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    torch.set_num_threads(1);output=Path(output);manifest=ready(output);config=manifest['settings'];budget=config['episodes']
    if weight not in WEIGHTS:raise ValueError('Only the registered sum-one weights may be trained')
    folder=folder_for(weight,seed,output)
    if (folder/'summary.json').exists():
        s=json.loads((folder/'summary.json').read_text());assert s['source_sha256']==source_hashes()
        for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h
        return s
    folder.mkdir(parents=True,exist_ok=False);train=environment(weight);evaluation=environment(weight)
    started=time.perf_counter();cpu=time.process_time();eval_wall=artifact_wall=0.
    curve=[];trace=[];snapshots={};best=None;hooks=[];steps={};numeric={}
    try:
        agent=make_agent('SAC',128,seed,train,config)
        effective=json.loads(json.dumps(effective_hyperparameters(agent)));geometry=network_geometry(agent,128)
        prior=json.loads((folder_for('legacy_sum',seed)/'summary.json').read_text())
        assert effective==prior['effective_hyperparameters'],'Non-reward learner condition changed'
        write_json(folder/'config.json',dict(seed=seed,method='SAC',net_arch=[128,128],spectral_weight=weight,
            coefficients=coefficients(weight),settings=config,effective_hyperparameters=effective,
            network_geometry=geometry,source_sha256=source_hashes(),target_sha256=manifest['target_sha256']))
        class FiniteLogger(Logger):
            def record(self,key,value,exclude=None):
                if isinstance(value,(float,int,np.number)):
                    if not np.isfinite(value):raise FloatingPointError(f'Nonfinite training metric {key}')
                    item=numeric.setdefault(key,dict(count=0,minimum=float(value),maximum=float(value)))
                    item.update(count=item['count']+1,minimum=min(item['minimum'],float(value)),maximum=max(item['maximum'],float(value)))
                super().record(key,value,exclude)
        agent.set_logger(FiniteLogger(None,[]))
        for name,parent,attribute in (('actor',agent.actor,'optimizer'),('critic',agent.critic,'optimizer'),('entropy',agent,'ent_coef_optimizer')):
            opt=getattr(parent,attribute);steps[name]=0
            def pre(opt,args,kwargs):
                for group in opt.param_groups:
                    for p in group['params']:
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
            def post(opt,args,kwargs,name=name):
                steps[name]+=1
                for group in opt.param_groups:
                    for p in group['params']:
                        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite parameter')
            hooks.extend([opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)])

        def evaluate(n):
            nonlocal best,eval_wall,artifact_wall
            before=learner_signature(agent);tick=time.perf_counter()
            metrics,values=episode(evaluation,agent=agent);eval_wall+=time.perf_counter()-tick
            policy=policy_statistics(agent);after=learner_signature(agent)
            assert before==after,'Policy evaluation/statistics mutated learner or RNG'
            row=metric_row(metrics,values,n,gradient_updates=int(agent._n_updates),policy_statistics=policy,
                learner_before=before,learner_after=after)
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
                    reward=float(self.locals['rewards'][0]),alpha_before_update=float(agent.log_ent_coef.exp().item())))
                return True
            def _on_rollout_start(self):
                n=train.completed_episodes
                if 0<n<budget and n%config['eval_every_episodes']==0:evaluate(n)
                if n and n%512==0:print(json.dumps(dict(stage='progress',weight=weight,seed=seed,candidates=n)),flush=True)

        print(json.dumps(dict(stage='start',weight=weight,seed=seed)),flush=True)
        evaluate(0);assert curve[0]['learner_before']['policy_sha256']==prior['initial']['learner_before']['policy_sha256']
        tick=time.perf_counter();e0=eval_wall;a0=artifact_wall
        agent.learn(total_timesteps=budget,callback=Evaluation())
        training_wall=time.perf_counter()-tick-(eval_wall-e0)-(artifact_wall-a0)
        evaluate(budget);assert agent.num_timesteps==budget and agent._n_updates==budget-config['warmup_episodes']
        tick=time.perf_counter();agent.save(folder/'agent.zip');artifact_wall+=time.perf_counter()-tick
        for role,expected in (('final',curve[-1]),('selected',best)):
            tick=time.perf_counter();metrics,values=episode(evaluation,action=snapshots[role]['actions'][0]);eval_wall+=time.perf_counter()-tick
            assert metrics['weighted_loss']==expected['weighted_loss'] and metrics['acoustic_error']==expected['acoustic_error']
            np.testing.assert_array_equal(values['audio'],snapshots[role]['audio']);save_audio(folder/f'{role}.npz',values)
        pack(folder/'curve.json.gz',curve);pack(folder/'training_trace.json.gz',trace)
        posthoc=min(curve,key=lambda p:p['acoustic_error'])
        files=('config.json','agent.zip','selected_agent.zip','final.npz','selected.npz','curve.json.gz','training_trace.json.gz')
        summary=dict(method='SAC',net_arch=[128,128],spectral_weight=weight,seed=seed,
            initial=curve[0],final=curve[-1],selected=best,posthoc=posthoc,
            training=account(train,training_wall),evaluation=account(evaluation,eval_wall),
            independent_policy_evaluations=len(curve),endpoint_replays=2,gradient_updates=int(agent._n_updates),
            optimizer_steps=steps,effective_hyperparameters=effective,network_geometry=geometry,
            numeric_training_metrics=numeric,nonfinite_detected=0,source_sha256=source_hashes(),
            target_sha256=manifest['target_sha256'],artifact_sha256={p:digest(folder/p) for p in files},
            elapsed_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu,artifact_wall_s=artifact_wall)
        write_json(folder/'summary.json',summary)
        print(json.dumps(dict(stage='complete',weight=weight,seed=seed,final=curve[-1]['acoustic_error'],selected=best['acoustic_error'],posthoc=posthoc['acoustic_error'])),flush=True)
        return summary
    except Exception as exc:
        write_json(folder/'failure.json',dict(type=type(exc).__name__,message=str(exc),
            training=account(train,time.perf_counter()-started),evaluation=account(evaluation,eval_wall)))
        raise
    finally:
        for hook in hooks:hook.remove()
        train.close();evaluation.close()


def train(workers=9,output=OUT):
    output=Path(output);ready(output)
    if (output/'execution.json').exists():
        print('Completed execution preserved; use audit/report',flush=True);return
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,w,s,output) for w in WEIGHTS for s in SEEDS]
        for f in as_completed(futures):results.append(f.result())
    write_json(output/'execution.json',dict(runs=len(results),workers=workers,wall_s=time.perf_counter()-started,
        training_candidates=sum(r['training']['episodes'] for r in results),
        independent_policy_evaluations=sum(r['independent_policy_evaluations'] for r in results),
        endpoint_replays=sum(r['endpoint_replays'] for r in results),
        training_process_wall_s=sum(r['training']['wall_s'] for r in results),
        evaluation_process_wall_s=sum(r['evaluation']['wall_s'] for r in results),
        process_cpu_s=sum(r['process_cpu_s'] for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=9)
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();train(a.workers,a.output)
