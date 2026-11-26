"""Exact4096-step Case B learning; evaluation happens after completed updates."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from .issue27g_data import OUT,METHODS,SEEDS,half_width,hashes,ready,folder,save_values,pack,metric_row,chosen,save_checkpoint
from .issue27g_env import CaseBEnv,episode
from .issue27g_agent import make_agent,optimizers,lrs,geometry,effective,signature,statistics
from .issue25_agent import numerical_check
from .constant020_env import account
from .issue15_analysis import write_json
from .issue16 import digest

def run_one(target,dim,method,seed,output=OUT):
    torch.set_num_threads(1);output=Path(output);manifest=ready(output);config=manifest['settings'];budget=config['episodes'];path=folder(target,dim,method,seed,output)
    if (path/'summary.json').exists():
        s=json.loads((path/'summary.json').read_text());assert s['source_sha256']==hashes()
        for name,h in s['artifact_sha256'].items():assert digest(path/name)==h
        return s
    path.mkdir(parents=True,exist_ok=False);train=CaseBEnv(target,dim);evaluation=CaseBEnv(target,dim,record_moments=True)
    tick=time.perf_counter();cpu=time.process_time();eval_wall=artifact_wall=0.;curve=[];snapshots={};best={};logs=[];optimizer_events=[];hooks=[];steps={};lr_checks={};numeric_max={}
    try:
        agent=make_agent(method,seed,train,config);net=geometry(agent,method);settings=effective(agent,method);lrs(agent,method)
        write_json(path/'config.json',dict(target=target,dim=dim,method=method,seed=seed,settings=config,network_geometry=net,effective_hyperparameters=settings,source_sha256=hashes()))
        class FiniteLogger(Logger):
            def record(self,key,value,exclude=None):
                if isinstance(value,(int,float,np.number)):
                    if not np.isfinite(value):raise FloatingPointError(f'Nonfinite logger metric:{key}')
                    if key.startswith('train/'):logs.append(dict(trial=agent.num_timesteps,key=key,value=float(value)))
                super().record(key,value,exclude)
        agent.set_logger(FiniteLogger(None,[]))
        for name,opt in optimizers(agent,method).items():
            steps[name]=lr_checks[name]=0;numeric_max[name]={}
            def check(opt,grad,name):
                q=numerical_check(opt,gradients=grad)
                for k,v in q.items():numeric_max[name][k]=max(numeric_max[name].get(k,0.),v)
                return q
            def pre(opt,args,kwargs,name=name):
                lrs(agent,method);lr_checks[name]+=1;q=check(opt,True,name)
                optimizer_events.append(dict(trial=agent.num_timesteps,optimizer=name,step=steps[name]+1,phase='before',lrs=lrs(agent,method),**q))
            def post(opt,args,kwargs,name=name):
                steps[name]+=1;lrs(agent,method);q=check(opt,False,name)
                optimizer_events.append(dict(trial=agent.num_timesteps,optimizer=name,step=steps[name],phase='after',lrs=lrs(agent,method),**q))
            hooks.extend([opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)])
        def evaluate(n):
            nonlocal eval_wall,artifact_wall
            before=signature(agent);started=time.perf_counter();metrics,values=episode(evaluation,agent=agent);eval_wall+=time.perf_counter()-started
            stats=statistics(agent,method);r=metric_row(metrics,values,n,policy_statistics=stats,gradient_updates=int(agent._n_updates),optimizer_steps=dict(steps),optimizer_lrs=lrs(agent,method),
                physical_action_std_mc=(np.asarray(stats['normalized_action_std_mc'])*half_width(dim)).tolist(),training_incumbent=dict(train.best_candidate) if train.best_candidate else None)
            curve.append(r);snapshots['final']=values;started=time.perf_counter()
            for role,key in (('selected','weighted_loss'),('posthoc','acoustic_error')):
                if role not in best or r[key]<best[role][key]:best[role]=r;snapshots[role]=values;save_checkpoint(path/f'{role}.pt',agent,method,r)
            if n==budget:save_checkpoint(path/'final.pt',agent,method,r)
            artifact_wall+=time.perf_counter()-started;after=signature(agent);assert before==after,'Evaluation/log/save changed learner/Buffer/RNG/training best'
            r.update(learner_before=before,learner_after=after)
        class Evaluation(BaseCallback):
            def _on_step(self):return True
            def _on_rollout_start(self):
                n=train.completed_episodes
                if n and n<budget and n%config['eval_every_episodes']==0:evaluate(n)
                if n and n%512==0:print(json.dumps(dict(stage='progress',target=target,dim=dim,method=method,seed=seed,candidates=n)),flush=True)
        print(json.dumps(dict(stage='start',target=target,dim=dim,method=method,seed=seed,budget=budget)),flush=True);evaluate(0)
        started=time.perf_counter();e0=eval_wall;a0=artifact_wall
        agent.learn(total_timesteps=budget,callback=Evaluation());training_wall=time.perf_counter()-started-(eval_wall-e0)-(artifact_wall-a0)
        evaluate(budget)
        assert agent.num_timesteps==train.completed_episodes==agent.transition_checks==len(agent.transition_rows)==budget
        if method=='SAC':expected=dict(actor=budget-16,critic=budget-16,entropy=budget-16);assert agent._n_updates==budget-16
        elif method=='TD3':expected=dict(actor=(budget-16)//2,critic=budget-16);assert agent._n_updates==budget-16
        else:expected=dict(policy_value=(budget//128)*40);assert agent._n_updates==(budget//128)*10 and budget%128==0
        assert steps==lr_checks==expected,(steps,expected)
        assert len(curve)==1+budget//config['eval_every_episodes']
        for role in ('final','selected','posthoc'):save_values(path/f'{role}.npz',snapshots[role])
        save_values(path/'training_best.npz',train.best_values)
        for name,records in (('curve',curve),('training_trace',agent.transition_rows),('optimizer_events',optimizer_events),('training_metrics',logs)):pack(path/f'{name}.json.gz',records)
        s=dict(status='complete',target=target,dim=dim,method=method,seed=seed,initial=curve[0],**chosen(curve),training_best=dict(train.best_candidate),
            training=account(train,training_wall),evaluation=account(evaluation,eval_wall),independent_policy_evaluations=len(curve),optimizer_steps=steps,optimizer_lr_checks=lr_checks,
            internal_update_counter=int(agent._n_updates),PPO_rollouts=budget//128 if method=='PPO' else None,transition_checks=agent.transition_checks,native_checks=train.native_checks,
            network_geometry=net,effective_hyperparameters=settings,numerical_maxima=numeric_max,source_sha256=hashes(),elapsed_s=time.perf_counter()-tick,process_cpu_s=time.process_time()-cpu,
            artifact_sha256={p.name:digest(p) for p in sorted(path.iterdir()) if p.is_file()})
        write_json(path/'summary.json',s);print(json.dumps(dict(stage='complete',target=target,dim=dim,method=method,seed=seed,Final_E=s['final']['acoustic_error'],Final_voiced=s['final']['sustained_voiced'])),flush=True);return s
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),actual_candidates=train.completed_episodes);write_json(path/'failure.json',failure)
        for name,records in (('curve',curve),('training_trace',getattr(agent,'transition_rows',[]) if 'agent' in locals() else []),('optimizer_events',optimizer_events),('training_metrics',logs)):pack(path/f'{name}.json.gz',records)
        s=dict(status='failed',target=target,dim=dim,method=method,seed=seed,failure=failure,training=account(train,time.perf_counter()-tick),evaluation=account(evaluation,eval_wall),
            independent_policy_evaluations=len(curve),optimizer_steps=steps,source_sha256=hashes());write_json(path/'summary.json',s)
        print(json.dumps(dict(stage='failed',target=target,dim=dim,method=method,seed=seed,failure=failure)),flush=True);return s
    finally:
        for h in hooks:h.remove()
        train.close();evaluation.close()

def train(workers=27):
    manifest=ready()
    if (OUT/'execution.json').exists():raise FileExistsError('Existing execution preserved')
    cases=manifest['eligible_cases'];tick=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=min(workers,len(cases)*9),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,t,d,m,s) for t,d in cases for m in METHODS for s in SEEDS]
        for f in as_completed(futures):results.append(f.result())
    write_json(OUT/'execution.json',dict(runs=len(results),planned_runs=54,complete_runs=sum(s['status']=='complete' for s in results),failed_runs=sum(s['status']=='failed' for s in results),
        training_candidates=sum(s['training']['episodes'] for s in results),independent_policy_evaluations=sum(s['independent_policy_evaluations'] for s in results),
        wall_s=time.perf_counter()-tick,workers=workers,source_sha256=hashes()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=27);a=p.parse_args();train(a.workers)
