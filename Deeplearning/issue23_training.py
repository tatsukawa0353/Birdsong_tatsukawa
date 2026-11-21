"""Three unchanged SAC128 runs with read-only diagnostics and separate retention."""
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
from .issue23_data import (ROOT,OUT,SEEDS,environment,ready,folder_for,old_folder,old_curve,
    source_hashes,pack,metric_row,save_audio,policy_statistics,chosen,deterministic_state,
    predict_q,q_action_gradient,save_checkpoint)
from .issue17_artifacts import read_record
from .issue15_analysis import write_json
from .issue16 import digest


def run_one(seed,output=OUT):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    torch.set_num_threads(1);output=Path(output);manifest=ready(output);config=manifest['settings'];budget=config['episodes']
    folder=folder_for(seed,output)
    if (folder/'summary.json').exists():
        summary=json.loads((folder/'summary.json').read_text());assert summary['source_sha256']==source_hashes()
        for p,h in summary['artifact_sha256'].items():assert digest(folder/p)==h,p
        return summary
    folder.mkdir(parents=True,exist_ok=False);train=environment(.75);evaluation=environment(.75)
    started=time.perf_counter();cpu=time.process_time();eval_wall=artifact_wall=diagnostic_wall=0.
    curve=[];trace=[];updates=[];q_history=[];snapshots={};best={};hooks=[];steps={};numeric={};update_before={}
    comparison=dict(training_rows_exact=0,evaluation_metrics_exact=0,evaluation_signatures_exact=0,
        first_difference=None,maximum_training_E_difference=0.,maximum_evaluation_E_difference=0.,
        initial_signature_excludes=['training_env_rng_sha256'],initial_env_rng_equal=None)
    previous_trace=read_record(old_folder(.75,seed)/'training_trace.json');previous_curve={p['episodes']:p for p in old_curve(.75,seed)}
    candidates=json.loads((output/'fixed_candidates.json').read_text())
    def mismatch(kind,n):
        if comparison['first_difference'] is None:comparison['first_difference']=dict(kind=kind,trial=n)
    try:
        agent=make_agent('SAC',128,seed,train,config)
        effective=json.loads(json.dumps(effective_hyperparameters(agent)));geometry=network_geometry(agent,128)
        prior=json.loads((old_folder(.75,seed)/'summary.json').read_text())
        assert effective==prior['effective_hyperparameters'],'Non-selection learner condition changed'
        write_json(folder/'config.json',dict(seed=seed,method='SAC',spectral_weight=.75,settings=config,
            effective_hyperparameters=effective,network_geometry=geometry,source_sha256=source_hashes()))

        class DiagnosticLogger(Logger):
            def record(self,key,value,exclude=None):
                if isinstance(value,(int,float,np.number)):
                    if not np.isfinite(value):raise FloatingPointError(f'Nonfinite metric {key}')
                    item=numeric.setdefault(key,dict(count=0,minimum=float(value),maximum=float(value)))
                    item.update(count=item['count']+1,minimum=min(item['minimum'],float(value)),maximum=max(item['maximum'],float(value)))
                    if updates and key in ('train/actor_loss','train/critic_loss','train/ent_coef_loss','train/ent_coef'):
                        assert updates[-1]['trial']==agent.num_timesteps
                        updates[-1][key.split('/')[1]]=float(value)
                super().record(key,value,exclude)
        agent.set_logger(DiagnosticLogger(None,[]))
        for name,parent,attribute in (('actor',agent.actor,'optimizer'),('critic',agent.critic,'optimizer'),('entropy',agent,'ent_coef_optimizer')):
            opt=getattr(parent,attribute);steps[name]=0
            def pre(opt,args,kwargs,name=name):
                nonlocal diagnostic_wall
                for group in opt.param_groups:
                    for p in group['params']:
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
                if name=='actor':
                    tick=time.perf_counter();update_before.clear();update_before.update(deterministic_state(agent));diagnostic_wall+=time.perf_counter()-tick
            def post(opt,args,kwargs,name=name):
                nonlocal diagnostic_wall
                steps[name]+=1
                for group in opt.param_groups:
                    for p in group['params']:
                        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite parameter')
                if name=='actor':
                    tick=time.perf_counter();after=deterministic_state(agent)
                    da=np.array(after['action'])-update_before['action']
                    updates.append(dict(trial=agent.num_timesteps,update=steps[name],before=dict(update_before),after=after,
                        delta_action=da.tolist(),delta_action_l2=float(np.linalg.norm(da))))
                    diagnostic_wall+=time.perf_counter()-tick
            hooks.extend([opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)])

        def evaluate(n):
            nonlocal eval_wall,artifact_wall,diagnostic_wall
            before=learner_signature(agent);tick=time.perf_counter()
            metrics,values=episode(evaluation,agent=agent);eval_wall+=time.perf_counter()-tick
            tick=time.perf_counter();policy=policy_statistics(agent)
            row=metric_row(metrics,values,n,gradient_updates=int(agent._n_updates),policy_statistics=policy)
            q1,q2=predict_q(agent,[p['action'] for p in candidates])
            current_q1,current_q2=predict_q(agent,values['actions'])
            row.update(actor_Q1=float(current_q1[0]),actor_Q2=float(current_q2[0]),actor_minQ=float(min(current_q1[0],current_q2[0])),
                actor_minQ_gradient=q_action_gradient(agent,values['actions'][0]))
            previous_n=curve[-1]['episodes'] if curve else 0
            recent=[u for u in updates if previous_n<u['trial']<=n]
            row['previous_interval']=dict(updates=len(recent),
                actor_path_length=sum(u['delta_action_l2'] for u in recent),
                mean_actor_loss=float(np.mean([u['actor_loss'] for u in recent])) if recent else None,
                mean_critic_loss=float(np.mean([u['critic_loss'] for u in recent])) if recent else None)
            for i,c in enumerate(candidates):
                q_history.append(dict(trial=n,candidate_id=c['candidate_id'],Q1=float(q1[i]),Q2=float(q2[i]),minQ=float(min(q1[i],q2[i]))))
            diagnostic_wall+=time.perf_counter()-tick
            curve.append(row);snapshots['final']=values
            tick=time.perf_counter()
            for role,key in (('selected','weighted_loss'),('posthoc','acoustic_error')):
                if role not in best or row[key]<best[role][key]:
                    best[role]=row;snapshots[role]=values;save_checkpoint(folder/f'{role}.pt',agent,row)
            if seed==2 and n in (2944,3072,3840):save_checkpoint(folder/f'at_{n}.pt',agent,row)
            if n==budget:save_checkpoint(folder/'final.pt',agent,row)
            artifact_wall+=time.perf_counter()-tick
            after=learner_signature(agent);assert before==after,'Logging/evaluation/checkpoint changed learner, RNG or Replay'
            row.update(learner_before=before,learner_after=after)
            if n in previous_curve:
                old=previous_curve[n]
                exact=all(row[k]==old[k] for k in ('actions','physical_controls','acoustic_components','acoustic_error','weighted_loss','policy_statistics'))
                comparison['evaluation_metrics_exact']+=int(exact)
                old_sig=old['learner_before']
                if n==0:
                    # SB3 queues the training reset seed; this unused RNG is still
                    # OS-initialized before learn(). From the first reset it is seeded.
                    comparison['initial_env_rng_equal']=before['training_env_rng_sha256']==old_sig['training_env_rng_sha256']
                    same=all(before[k]==old_sig[k] for k in before if k!='training_env_rng_sha256')
                else:same=before==old_sig
                comparison['evaluation_signatures_exact']+=int(same)
                comparison['maximum_evaluation_E_difference']=max(comparison['maximum_evaluation_E_difference'],abs(row['acoustic_error']-old['acoustic_error']))
                if not exact:mismatch('evaluation_metrics',n)
                if not same:mismatch('learner_signature',n)
            if n in (0,128):
                write_json(folder/f'prefix_{n}.json',dict(trial=n,comparison=comparison,
                    exact_so_far=comparison['first_difference'] is None,stage_a_sha256=manifest['stage_a_sha256']))
                print(json.dumps(dict(stage='prefix_check',seed=seed,trial=n,comparison=comparison)),flush=True)

        class Evaluation(BaseCallback):
            def _on_step(self):
                info=self.locals['infos'][0]
                row=dict(episodes=self.num_timesteps,acoustic_error=info['acoustic_error'],acoustic_components=info['acoustic_components'],
                    weighted_loss=info['weighted_loss'],failed=info['failed'],sustained_voiced=info['sustained_voiced'],
                    action=info['applied_action'],physical_controls=info['applied_controls'],
                    reward=float(self.locals['rewards'][0]),alpha_before_update=float(agent.log_ent_coef.exp()))
                old=previous_trace[self.num_timesteps-1];same=row==old
                comparison['training_rows_exact']+=int(same)
                comparison['maximum_training_E_difference']=max(comparison['maximum_training_E_difference'],abs(row['acoustic_error']-old['acoustic_error']))
                if not same:mismatch('training_sample',self.num_timesteps)
                trace.append(row);return True
            def _on_rollout_start(self):
                n=train.completed_episodes
                if 0<n<budget and n%config['eval_every_episodes']==0:evaluate(n)
                if n and n%512==0:print(json.dumps(dict(stage='progress',seed=seed,candidates=n)),flush=True)

        print(json.dumps(dict(stage='start',seed=seed,stage_a_sha256=manifest['stage_a_sha256'])),flush=True)
        evaluate(0);assert curve[0]['learner_before']['policy_sha256']==prior['initial']['learner_before']['policy_sha256']
        tick=time.perf_counter();e0=eval_wall;a0=artifact_wall;d0=diagnostic_wall
        agent.learn(total_timesteps=budget,callback=Evaluation())
        training_wall=time.perf_counter()-tick-(eval_wall-e0)-(artifact_wall-a0)-(diagnostic_wall-d0)
        evaluate(budget);assert agent.num_timesteps==budget and agent._n_updates==budget-config['warmup_episodes']
        for role in ('final','selected','posthoc'):save_audio(folder/f'{role}.npz',snapshots[role])
        for name,records in (('curve',curve),('training_trace',trace),('updates',updates),('q_history',q_history)):
            pack(folder/f'{name}.json.gz',records)
        assert all(all(k in u for k in ('actor_loss','critic_loss','ent_coef_loss','ent_coef')) for u in updates)
        assert len(updates)==budget-config['warmup_episodes']
        assert evaluation.completed_episodes==len(curve),'No learning experiences may enter evaluation'
        endpoints=chosen(curve)
        summary=dict(seed=seed,initial=curve[0],**endpoints,training=account(train,training_wall),
            evaluation=account(evaluation,eval_wall),independent_policy_evaluations=len(curve),gradient_updates=int(agent._n_updates),
            optimizer_steps=steps,effective_hyperparameters=effective,network_geometry=geometry,numeric_training_metrics=numeric,
            nonfinite_detected=0,source_sha256=source_hashes(),comparison_to_issue22=comparison,
            elapsed_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu,artifact_wall_s=artifact_wall,
            diagnostic_wall_s=diagnostic_wall,checkpoint_format='actor+online twin critic inference weights; no resume data',
            artifact_sha256={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()})
        write_json(folder/'summary.json',summary)
        print(json.dumps(dict(stage='complete',seed=seed,final=curve[-1]['acoustic_error'],comparison=comparison)),flush=True)
        return summary
    except Exception as exc:
        write_json(folder/'failure.json',dict(type=type(exc).__name__,message=str(exc),
            training=account(train,time.perf_counter()-started),evaluation=account(evaluation,eval_wall)))
        raise
    finally:
        for h in hooks:h.remove()
        train.close();evaluation.close()


def train(workers=3,output=OUT):
    output=Path(output);ready(output)
    if (output/'execution.json').exists():
        print('Completed reproduction preserved; use analysis/report',flush=True);return
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=min(workers,3),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,s,output) for s in SEEDS]
        for f in as_completed(futures):results.append(f.result())
    write_json(output/'execution.json',dict(runs=len(results),wall_s=time.perf_counter()-started,workers=min(workers,3),
        training_candidates=sum(r['training']['episodes'] for r in results),
        independent_policy_evaluations=sum(r['independent_policy_evaluations'] for r in results),
        gradient_updates=sum(r['gradient_updates'] for r in results),source_sha256=source_hashes(),
        exact_reproductions=sum(r['comparison_to_issue22']['first_difference'] is None for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();train(a.workers)
