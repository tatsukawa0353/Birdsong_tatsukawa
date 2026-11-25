"""Stage B wide SAC, followed by both frozen behavior-mixture interventions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .constant020_env import episode,account
from .issue14 import effective_hyperparameters,learner_signature
from .issue21_training import network_geometry
from .issue25_agent import numerical_check,assert_lrs,BASE_LR
from .issue26_agent import make_agent,observer_signature
from .issue26_data import (BASE,OUT,NEW,SEEDS,PREFIX,environment,ready,folder_for,curve_for,source_hashes,
    pack,metric_row,save_audio,policy_statistics,chosen,deterministic_state,save_checkpoint,half_width)
from .issue17_artifacts import read_record
from .issue15_analysis import write_json
from .issue16 import digest


def comparable(row):
    return {k:row[k] for k in ('episodes','acoustic_error','acoustic_components','weighted_loss','failed','sustained_voiced','action','physical_controls','reward','alpha_before_update')}


def run_one(condition,seed,output=OUT):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    torch.set_num_threads(1);output=Path(output);manifest=ready(output);config=manifest['settings'];budget=config['episodes']
    if condition not in NEW:raise ValueError('A is read-only')
    folder=folder_for(condition,seed,output)
    if (folder/'summary.json').exists():
        s=json.loads((folder/'summary.json').read_text());assert s['source_sha256']==source_hashes()
        for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
        return s
    folder.mkdir(parents=True,exist_ok=False);train=environment(condition);evaluation=environment(condition)
    started=time.perf_counter();cpu=time.process_time();eval_wall=artifact_wall=diagnostic_wall=0.
    curve=[];updates=[];snapshots={};best={};hooks=[];steps={};lr_checks={};numeric={};maxima={};before_update={}
    prefix_comparison=dict(training_rows=0,evaluation_rows=0,learner_states=0,passed=None)
    reference_trace=None;reference_curve={}
    if condition!='B':
        reference_trace=read_record(folder_for('B',seed,output)/'training_trace.json')
        reference_curve={p['episodes']:p for p in curve_for('B',seed,output)}
    try:
        agent=make_agent(condition,seed,train,config);prior=json.loads((folder_for('A',seed)/'summary.json').read_text())
        effective=json.loads(json.dumps(effective_hyperparameters(agent)));assert effective==prior['effective_hyperparameters']
        geometry=network_geometry(agent,128);assert_lrs(agent,BASE_LR)
        write_json(folder/'config.json',dict(condition=condition,seed=seed,settings=config,effective_hyperparameters=effective,
            network_geometry=geometry,source_sha256=source_hashes()))
        class DiagnosticLogger(Logger):
            def record(self,key,value,exclude=None):
                if isinstance(value,(int,float,np.number)):
                    if not np.isfinite(value):raise FloatingPointError(f'Nonfinite metric:{key}')
                    item=numeric.setdefault(key,dict(count=0,minimum=float(value),maximum=float(value)))
                    item.update(count=item['count']+1,minimum=min(item['minimum'],float(value)),maximum=max(item['maximum'],float(value)))
                    if updates and key in ('train/actor_loss','train/critic_loss','train/ent_coef_loss','train/ent_coef'):
                        assert updates[-1]['trial']==agent.num_timesteps;updates[-1][key.split('/')[1]]=float(value)
                super().record(key,value,exclude)
        agent.set_logger(DiagnosticLogger(None,[]))
        for name,parent,attribute in (('actor',agent.actor,'optimizer'),('critic',agent.critic,'optimizer'),('entropy',agent,'ent_coef_optimizer')):
            opt=getattr(parent,attribute);steps[name]=lr_checks[name]=0
            def record_numeric(optimizer,gradient,name):
                state=numerical_check(optimizer,gradients=gradient);maximum=maxima.setdefault(name,{k:0. for k in state})
                for k,v in state.items():maximum[k]=max(maximum[k],v)
                return state
            def pre(opt,args,kwargs,name=name):
                nonlocal diagnostic_wall
                assert_lrs(agent,BASE_LR);lr_checks[name]+=1;record_numeric(opt,True,name)
                if name=='actor':
                    t=time.perf_counter();before_update.clear();before_update.update(deterministic_state(agent));diagnostic_wall+=time.perf_counter()-t
            def post(opt,args,kwargs,name=name):
                nonlocal diagnostic_wall
                steps[name]+=1;assert_lrs(agent,BASE_LR);record_numeric(opt,False,name)
                if name=='actor':
                    t=time.perf_counter();after=deterministic_state(agent);da=np.array(after['action'])-before_update['action']
                    updates.append(dict(trial=agent.num_timesteps,update=steps[name],before=dict(before_update),after=after,
                        delta_action=da.tolist(),delta_action_l2=float(np.linalg.norm(da)),optimizer_lrs=assert_lrs(agent,BASE_LR)))
                    diagnostic_wall+=time.perf_counter()-t
            hooks.extend([opt.register_step_pre_hook(pre),opt.register_step_post_hook(post)])

        def evaluate(n):
            nonlocal eval_wall,artifact_wall,diagnostic_wall
            before=observer_signature(agent);t=time.perf_counter();metrics,values=episode(evaluation,agent=agent);eval_wall+=time.perf_counter()-t
            t=time.perf_counter();policy=policy_statistics(agent)
            row=metric_row(metrics,values,n,gradient_updates=int(agent._n_updates),policy_statistics=policy,
                physical_action_std_mc=(np.array(policy['normalized_action_std_mc'])*half_width(condition)).tolist(),
                optimizer_lrs=assert_lrs(agent,BASE_LR),training_incumbent=dict(agent.incumbent) if agent.incumbent else None)
            recent=[u for u in updates if (curve[-1]['episodes'] if curve else 0)<u['trial']<=n]
            row['previous_interval']=dict(updates=len(recent),path_length=sum(u['delta_action_l2'] for u in recent),
                **{key:float(np.mean([u[key] for u in recent])) if recent else None for key in ('actor_loss','critic_loss','ent_coef_loss')})
            diagnostic_wall+=time.perf_counter()-t;curve.append(row);snapshots['final']=values;t=time.perf_counter()
            for role,key in (('selected','weighted_loss'),('posthoc','acoustic_error')):
                if role not in best or row[key]<best[role][key]:best[role]=row;snapshots[role]=values;save_checkpoint(folder/f'{role}.pt',agent,row,condition)
            if condition=='B' and n==PREFIX:save_checkpoint(folder/'prefix256.pt',agent,row,condition);save_audio(folder/'prefix256.npz',values)
            if n==budget:save_checkpoint(folder/'final.pt',agent,row,condition)
            artifact_wall+=time.perf_counter()-t;after=observer_signature(agent);assert before==after,'Evaluation/log/save mutated learner or behavior RNG/center'
            row.update(learner_before=before['learner'],learner_after=after['learner'],observer_before=before,observer_after=after)
            if reference_trace is not None and n in reference_curve and 0<n<=config.get('local_start',PREFIX):
                old=reference_curve[n]
                assert all(row[k]==old[k] for k in ('actions','physical_controls','acoustic_error','weighted_loss','policy_statistics'))
                assert row['learner_before']==old['learner_before'];assert before['choice_rng']==old['observer_before']['choice_rng'] and before['candidate_rng']==old['observer_before']['candidate_rng']
                prefix_comparison['evaluation_rows']+=1;prefix_comparison['learner_states']+=1
            if n==config.get('local_start',PREFIX) and reference_trace is not None:
                assert [comparable(p) for p in agent.candidate_log[:n]]==[comparable(p) for p in reference_trace[:n]]
                prefix_comparison.update(training_rows=n,passed=True);write_json(folder/'prefix_check.json',prefix_comparison)

        class Evaluation(BaseCallback):
            def _on_step(self):return True
            def _on_rollout_start(self):
                n=train.completed_episodes
                if 0<n<budget and n%config['eval_every_episodes']==0:evaluate(n)
                if n and n%512==0:print(json.dumps(dict(stage='progress',condition=condition,seed=seed,candidates=n)),flush=True)

        print(json.dumps(dict(stage='start',condition=condition,seed=seed)),flush=True);evaluate(0)
        assert curve[0]['learner_before']['policy_sha256']==prior['initial']['learner_before']['policy_sha256']
        t=time.perf_counter();e0=eval_wall;a0=artifact_wall;d0=diagnostic_wall
        agent.learn(total_timesteps=budget,callback=Evaluation());training_wall=time.perf_counter()-t-(eval_wall-e0)-(artifact_wall-a0)-(diagnostic_wall-d0)
        evaluate(budget);assert len(agent.candidate_log)==budget==agent.transition_checks and agent._n_updates==budget-config['warmup_episodes']
        assert train.best_candidate['weighted_loss']==agent.incumbent['J'];assert train.best_candidate['trial']==agent.incumbent['trial']
        for role in ('final','selected','posthoc'):save_audio(folder/f'{role}.npz',snapshots[role])
        save_audio(folder/'training_best.npz',train.best_values)
        for name,records in (('curve',curve),('training_trace',agent.candidate_log),('updates',updates)):pack(folder/f'{name}.json.gz',records)
        s=dict(status='complete',condition=condition,seed=seed,initial=curve[0],**chosen(curve),training_best=dict(agent.incumbent),
            training=account(train,training_wall),evaluation=account(evaluation,eval_wall),independent_policy_evaluations=len(curve),
            gradient_updates=int(agent._n_updates),optimizer_steps=steps,optimizer_lr_checks=lr_checks,replay_transition_checks=agent.transition_checks,
            prefix_comparison=prefix_comparison,numeric_maxima=maxima,effective_hyperparameters=effective,network_geometry=geometry,
            source_sha256=source_hashes(),elapsed_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu,
            artifact_sha256={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()})
        write_json(folder/'summary.json',s);print(json.dumps(dict(stage='complete',condition=condition,seed=seed,final=s['final']['acoustic_error'],training_best_J=s['training_best']['J'])),flush=True);return s
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),trial=train.completed_episodes);write_json(folder/'failure.json',failure)
        for name,records in (('curve',curve),('training_trace',agent.candidate_log if 'agent' in locals() else []),('updates',updates)):pack(folder/f'{name}.json.gz',records)
        s=dict(status='failed',condition=condition,seed=seed,failure=failure,training=account(train,time.perf_counter()-started),evaluation=account(evaluation,eval_wall),
            independent_policy_evaluations=len(curve),gradient_updates=int(agent._n_updates) if 'agent' in locals() else 0,source_sha256=source_hashes())
        write_json(folder/'summary.json',s);print(json.dumps(dict(stage='failed',condition=condition,seed=seed,failure=failure)),flush=True);return s
    finally:
        for h in hooks:h.remove()
        train.close();evaluation.close()


def stage_b_record(output):
    rows=[]
    for c in ('A','B'):
        for seed in SEEDS:
            s=json.loads((folder_for(c,seed,output)/'summary.json').read_text())
            rows.append(dict(condition=c,seed=seed,status=s.get('status','complete'),final_E=s.get('final',{}).get('acoustic_error'),
                final_J=s.get('final',{}).get('weighted_loss'),training=s['training'],evaluations=s['independent_policy_evaluations']))
    body='# Stage B — 広域通常SACを先行\n\n'+json.dumps(rows,ensure_ascii=False,indent=2)+'\n\nBの結果によらずC/Dを同じ凍結計画で全て実施する。0.10の閾値による打切りや追加調整はしない。独立再生監査はStage Dでまとめる。\n'
    (Path(output)/'STAGE_B.md').write_text(body);write_json(Path(output)/'stage_b_review.json',dict(rows=rows,report_sha256=digest(Path(output)/'STAGE_B.md'),source_sha256=digest(__file__)))


def train(stage,workers=6,output=OUT):
    output=Path(output);ready(output);name='wide_execution.json' if stage=='wide' else 'local_execution.json'
    if (output/name).exists():print('Completed stage preserved',flush=True);return
    if stage=='local':
        assert (output/'wide_execution.json').exists() and (output/'stage_b_review.json').exists()
        assert json.loads((output/'stage_b_review.json').read_text())['report_sha256']==digest(output/'STAGE_B.md')
    conditions=('B',) if stage=='wide' else ('C','D');tick=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=min(workers,len(conditions)*3),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_one,c,s,output) for c in conditions for s in SEEDS]
        for f in as_completed(futures):results.append(f.result())
    record=dict(runs=len(results),complete_runs=sum(s['status']=='complete' for s in results),failed_runs=sum(s['status']=='failed' for s in results),
        training_candidates=sum(s['training']['episodes'] for s in results),independent_policy_evaluations=sum(s['independent_policy_evaluations'] for s in results),
        gradient_updates=sum(s['gradient_updates'] for s in results),wall_s=time.perf_counter()-tick,source_sha256=source_hashes())
    write_json(output/name,record)
    if stage=='wide':stage_b_record(output)
    else:
        wide=json.loads((output/'wide_execution.json').read_text());write_json(output/'execution.json',dict(runs=wide['runs']+record['runs'],
            complete_runs=wide['complete_runs']+record['complete_runs'],failed_runs=wide['failed_runs']+record['failed_runs'],
            training_candidates=wide['training_candidates']+record['training_candidates'],independent_policy_evaluations=wide['independent_policy_evaluations']+record['independent_policy_evaluations'],
            gradient_updates=wide['gradient_updates']+record['gradient_updates'],wall_s=wide['wall_s']+record['wall_s'],local_runs_executed_despite_B_outcome=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('wide','local'));p.add_argument('--workers',type=int,default=6);a=p.parse_args();train(a.stage,a.workers)
