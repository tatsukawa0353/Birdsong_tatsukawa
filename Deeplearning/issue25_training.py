"""Six Critic-only LR interventions with per-update LR and state diagnostics."""
from .result_paths import source_hashes_match
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .constant020_env import episode,account
from .issue14 import learner_signature
from .issue21_training import network_geometry
from .issue25_agent import make_agent,effective_difference,assert_lrs,optimizer_lrs,CRITIC_RATES,BASE_LR,numerical_check
from .issue25_data import (BASE,OUT,SEEDS,environment,ready,folder_for,curve_for,source_hashes,
    pack,metric_row,save_audio,policy_statistics,chosen,deterministic_state,predict_q,q_action_gradient)
from .issue23_data import save_checkpoint
from .issue17_artifacts import read_record
from .issue15_analysis import write_json
from .issue16 import digest


def run_one(rate,seed,output=OUT):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    torch.set_num_threads(1);output=Path(output);manifest=ready(output);config=manifest['settings'];budget=config['episodes']
    if rate not in CRITIC_RATES:raise ValueError('The baseline must never be retrained')
    folder=folder_for(rate,seed,output)
    if (folder/'summary.json').exists():
        s=json.loads((folder/'summary.json').read_text());assert source_hashes_match(s['source_sha256'], source_hashes())
        for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
        return s
    folder.mkdir(parents=True,exist_ok=False);train=environment(.75);evaluation=environment(.75)
    started=time.perf_counter();cpu=time.process_time();eval_wall=artifact_wall=diagnostic_wall=0.
    curve=[];trace=[];updates=[];snapshots={};best={};hooks=[];steps={};numeric={};update_before={};lr_checks={}
    candidates=json.loads((BASE/'fixed_candidates.json').read_text());final_q=[];q_history=[];numeric_state={};numeric_maxima={};first_success=None
    try:
        agent=make_agent(rate,seed,train,config);prior=json.loads((folder_for(BASE_LR,seed)/'summary.json').read_text())
        effective=effective_difference(agent,prior['effective_hyperparameters'],rate);geometry=network_geometry(agent,128)
        write_json(folder/'config.json',dict(seed=seed,critic_learning_rate=rate,settings=config,
            effective_hyperparameters=effective,network_geometry=geometry,initial_lrs=assert_lrs(agent,rate),source_sha256=source_hashes()))
        class DiagnosticLogger(Logger):
            def record(self,key,value,exclude=None):
                if isinstance(value,(int,float,np.number)):
                    if not np.isfinite(value):raise FloatingPointError(f'Nonfinite metric {key}')
                    item=numeric.setdefault(key,dict(count=0,minimum=float(value),maximum=float(value)))
                    item.update(count=item['count']+1,minimum=min(item['minimum'],float(value)),maximum=max(item['maximum'],float(value)))
                    if updates and key in ('train/actor_loss','train/critic_loss','train/ent_coef_loss','train/ent_coef'):
                        assert updates[-1]['trial']==agent.num_timesteps;updates[-1][key.split('/')[1]]=float(value)
                super().record(key,value,exclude)
        agent.set_logger(DiagnosticLogger(None,[]))
        for name,parent,attribute in (('actor',agent.actor,'optimizer'),('critic',agent.critic,'optimizer'),('entropy',agent,'ent_coef_optimizer')):
            opt=getattr(parent,attribute);steps[name]=0;lr_checks[name]=0
            def pre(opt,args,kwargs,name=name):
                nonlocal diagnostic_wall
                lrs=assert_lrs(agent,rate);lr_checks[name]+=1
                state=numerical_check(opt,gradients=True);numeric_state[name+'_pre']=state
                maximum=numeric_maxima.setdefault(name,{k:0. for k in state})
                for k,v in state.items():maximum[k]=max(maximum[k],v)
                for group in opt.param_groups:
                    for p in group['params']:
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
                if name=='actor':
                    t=time.perf_counter();update_before.clear();update_before.update(deterministic_state(agent));update_before['optimizer_lrs']=lrs
                    diagnostic_wall+=time.perf_counter()-t
            def post(opt,args,kwargs,name=name):
                nonlocal diagnostic_wall
                assert_lrs(agent,rate);steps[name]+=1
                state=numerical_check(opt);numeric_state[name+'_post']=state
                maximum=numeric_maxima.setdefault(name,{k:0. for k in state})
                for k,v in state.items():maximum[k]=max(maximum[k],v)
                for group in opt.param_groups:
                    for p in group['params']:
                        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite parameter')
                if name=='actor':
                    t=time.perf_counter();after=deterministic_state(agent);da=np.array(after['action'])-update_before['action']
                    updates.append(dict(trial=agent.num_timesteps,update=steps[name],before=dict(update_before),after=after,
                        delta_action=da.tolist(),delta_action_l2=float(np.linalg.norm(da)),optimizer_lrs=optimizer_lrs(agent),numeric_state=dict(numeric_state)))
                    diagnostic_wall+=time.perf_counter()-t
            hooks.extend([opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)])

        def evaluate(n):
            nonlocal eval_wall,artifact_wall,diagnostic_wall,final_q,first_success
            before=learner_signature(agent);t=time.perf_counter();metrics,values=episode(evaluation,agent=agent);eval_wall+=time.perf_counter()-t
            t=time.perf_counter();policy=policy_statistics(agent);q1,q2=predict_q(agent,values['actions'])
            row=metric_row(metrics,values,n,gradient_updates=int(agent._n_updates),policy_statistics=policy,optimizer_lrs=assert_lrs(agent,rate),
                actor_Q1=float(q1[0]),actor_Q2=float(q2[0]),actor_minQ=float(min(q1[0],q2[0])),actor_minQ_gradient=q_action_gradient(agent,values['actions'][0]))
            previous_n=curve[-1]['episodes'] if curve else 0;recent=[u for u in updates if previous_n<u['trial']<=n]
            row['previous_interval']=dict(updates=len(recent),actor_path_length=sum(u['delta_action_l2'] for u in recent),
                mean_actor_loss=float(np.mean([u['actor_loss'] for u in recent])) if recent else None,
                mean_critic_loss=float(np.mean([u['critic_loss'] for u in recent])) if recent else None,
                mean_ent_coef_loss=float(np.mean([u['ent_coef_loss'] for u in recent])) if recent else None)
            qa,qb=predict_q(agent,[c['action'] for c in candidates])
            current_q=[dict(candidate_id=c['candidate_id'],Q1=float(qa[i]),Q2=float(qb[i]),minQ=float(min(qa[i],qb[i]))) for i,c in enumerate(candidates)]
            q_history.extend(dict(trial=n,**p) for p in current_q)
            if n==budget:final_q=current_q
            diagnostic_wall+=time.perf_counter()-t;curve.append(row);snapshots['final']=values;t=time.perf_counter()
            for role,key in (('selected','weighted_loss'),('posthoc','acoustic_error')):
                if role not in best or row[key]<best[role][key]:
                    best[role]=row;snapshots[role]=values;save_checkpoint(folder/f'{role}.pt',agent,row)
            if first_success is None and row['acoustic_error']<=.1 and row['sustained_voiced'] and not row['failed']:
                first_success=row;snapshots['first_success']=values;save_checkpoint(folder/'first_success.pt',agent,row)
            if n==budget:save_checkpoint(folder/'final.pt',agent,row)
            artifact_wall+=time.perf_counter()-t;after=learner_signature(agent)
            assert before==after,'Evaluation/logging/checkpoint mutated learner, RNG or Replay'
            row.update(learner_before=before,learner_after=after)

        class Evaluation(BaseCallback):
            def _on_step(self):
                info=self.locals['infos'][0];trace.append(dict(episodes=self.num_timesteps,acoustic_error=info['acoustic_error'],
                    acoustic_components=info['acoustic_components'],weighted_loss=info['weighted_loss'],failed=info['failed'],
                    sustained_voiced=info['sustained_voiced'],action=info['applied_action'],physical_controls=info['applied_controls'],
                    reward=float(self.locals['rewards'][0]),alpha_before_update=float(agent.log_ent_coef.exp())))
                return True
            def _on_rollout_start(self):
                n=train.completed_episodes
                if 0<n<budget and n%config['eval_every_episodes']==0:evaluate(n)
                if n and n%512==0:print(json.dumps(dict(stage='progress',rate=rate,seed=seed,candidates=n)),flush=True)

        print(json.dumps(dict(stage='start',rate=rate,seed=seed)),flush=True)
        evaluate(0);assert curve[0]['learner_before']['policy_sha256']==prior['initial']['learner_before']['policy_sha256']
        assert curve[0]['weighted_loss']==prior['initial']['weighted_loss']
        t=time.perf_counter();e0=eval_wall;a0=artifact_wall;d0=diagnostic_wall
        agent.learn(total_timesteps=budget,callback=Evaluation())
        training_wall=time.perf_counter()-t-(eval_wall-e0)-(artifact_wall-a0)-(diagnostic_wall-d0)
        evaluate(budget);assert agent.num_timesteps==budget and agent._n_updates==budget-config['warmup_episodes']
        previous=read_record(folder_for(BASE_LR,seed)/'training_trace.json')
        assert trace[:config['warmup_episodes']]==previous[:config['warmup_episodes']]
        assert len(updates)==budget-config['warmup_episodes'] and evaluation.completed_episodes==len(curve)
        assert all(all(k in u for k in ('actor_loss','critic_loss','ent_coef_loss','ent_coef')) for u in updates)
        for role in ('final','selected','posthoc'):save_audio(folder/f'{role}.npz',snapshots[role])
        if first_success is not None:save_audio(folder/'first_success.npz',snapshots['first_success'])
        for name,records in (('curve',curve),('training_trace',trace),('updates',updates),('final_q',final_q),('q_history',q_history)):pack(folder/f'{name}.json.gz',records)
        s=dict(status='complete',seed=seed,critic_learning_rate=rate,initial=curve[0],first_success=first_success,**chosen(curve),training=account(train,training_wall),
            evaluation=account(evaluation,eval_wall),independent_policy_evaluations=len(curve),gradient_updates=int(agent._n_updates),
            optimizer_steps=steps,optimizer_lr_checks=lr_checks,effective_hyperparameters=effective,network_geometry=geometry,
            final_lrs=assert_lrs(agent,rate),warmup_exact=True,nonfinite_detected=0,numeric_training_metrics=numeric,numeric_maxima=numeric_maxima,
            source_sha256=source_hashes(),elapsed_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu,
            artifact_wall_s=artifact_wall,diagnostic_wall_s=diagnostic_wall,
            artifact_sha256={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()})
        write_json(folder/'summary.json',s)
        print(json.dumps(dict(stage='complete',rate=rate,seed=seed,final=s['final']['acoustic_error'],selected=s['selected']['acoustic_error'],posthoc=s['posthoc']['acoustic_error'])),flush=True)
        return s
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),trial=train.completed_episodes,optimizer_steps=steps,numeric_state=numeric_state)
        write_json(folder/'failure.json',failure)
        for name,records in (('curve',curve),('training_trace',trace),('updates',updates),('q_history',q_history)):pack(folder/f'{name}.json.gz',records)
        for role in ('selected','posthoc','first_success'):
            if role in snapshots:save_audio(folder/f'{role}.npz',snapshots[role])
        if curve:save_audio(folder/'last_evaluated.npz',snapshots['final'])
        s=dict(status='failed',seed=seed,critic_learning_rate=rate,planned_candidates=budget,failure=failure,
            training=account(train,time.perf_counter()-started),evaluation=account(evaluation,eval_wall),
            independent_policy_evaluations=len(curve),gradient_updates=int(agent._n_updates) if 'agent' in locals() else 0,
            optimizer_steps=steps,optimizer_lr_checks=lr_checks,numeric_maxima=numeric_maxima,source_sha256=source_hashes(),
            artifact_sha256={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()})
        write_json(folder/'summary.json',s);print(json.dumps(dict(stage='failed',rate=rate,seed=seed,failure=failure)),flush=True)
        return s
    finally:
        for h in hooks:h.remove()
        train.close();evaluation.close()


def train(workers=6,output=OUT):
    output=Path(output);ready(output)
    if (output/'execution.json').exists():print('Completed six runs preserved; use analysis/report',flush=True);return
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=min(workers,6),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,r,s,output) for r in CRITIC_RATES for s in SEEDS]
        for f in as_completed(futures):results.append(f.result())
    write_json(output/'execution.json',dict(runs=len(results),wall_s=time.perf_counter()-started,workers=min(workers,6),
        complete_runs=sum(r['status']=='complete' for r in results),failed_runs=sum(r['status']=='failed' for r in results),
        training_candidates=sum(r['training']['episodes'] for r in results),independent_policy_evaluations=sum(r['independent_policy_evaluations'] for r in results),
        gradient_updates=sum(r['gradient_updates'] for r in results),source_sha256=source_hashes()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=6);a=p.parse_args();train(a.workers)
