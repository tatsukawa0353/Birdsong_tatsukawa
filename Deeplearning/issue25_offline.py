"""Fixed-data terminal-reward regression; never SAC learning or native rendering."""
from .result_paths import source_hashes_match
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import random
import time

import numpy as np
from scipy.stats import kendalltau,spearmanr

from .issue25_data import (BASE,OUT,RATES,SEEDS,OFFLINE_EVALS,environment,offline_folder,ready,source_hashes,pack,fixed_gradients)
from .issue25_agent import BASE_LR,numerical_check
from .issue21_training import make_agent
from .issue14 import _state_digest
from .issue15_analysis import write_json
from .issue16 import digest


def scores(actual,predicted):
    """Exact tie-excluding inversion rate via Kendall counts, O(n log n)."""
    a=np.asarray(actual,float);p=np.asarray(predicted,float);difference=p-a;n=len(a);pairs=n*(n-1)//2
    def ties(values,axis=None):
        _,count=np.unique(values,return_counts=True,axis=axis)
        return int(np.sum(count*(count-1)//2))
    ta,tp=ties(a),ties(p);joint=ties(np.column_stack([a,p]),axis=0);comparable=pairs-ta-tp+joint
    rho=float(spearmanr(a,p).statistic) if n>1 and np.ptp(a)>0 and np.ptp(p)>0 else None
    if comparable:
        tau=float(kendalltau(a,p,variant='b').statistic)
        discordant=round((comparable-tau*np.sqrt(float(pairs-ta)*float(pairs-tp)))/2)
        inversion=discordant/comparable
    else:inversion=None
    return dict(n=n,rmse=float(np.sqrt(np.mean(difference**2))),mae=float(np.mean(abs(difference))),
        bias_Q_minus_r=float(difference.mean()),spearman=rho,pairwise_inversion_fraction=inversion,comparable_pairs=comparable)


def rng_digest():
    import torch
    return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=4)+torch.get_rng_state().numpy().tobytes()).hexdigest()


def offline_fit(rate,seed,output=OUT):
    import torch
    torch.set_num_threads(1);output=Path(output);m=ready(output);settings=m['offline'];budget=settings['updates'];folder=offline_folder(rate,seed,output)
    if (folder/'summary.json').exists():
        s=json.loads((folder/'summary.json').read_text());assert source_hashes_match(s['source_sha256'], source_hashes())
        for p,h in s['artifact_sha256'].items():assert digest(folder/p)==h,p
        return s
    folder.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();history=[];trace=[];maxima={};completed=0
    with np.load(output/'offline_data.npz') as z:
        train_a=z['train_actions'];train_r=z['train_rewards'];test_a=z['test_actions'];test_r=z['test_rewards']
    with np.load(output/'offline_batches.npz') as z:batch_indices=z[f'seed_{seed}'].copy()
    assert batch_indices.shape==(budget,32)
    env=environment(.75)
    try:agent=make_agent('SAC',128,seed,env,m['settings'])
    finally:env.close()
    optimizer=agent.critic.optimizer
    for group in optimizer.param_groups:group['lr']=rate
    actor_before=_state_digest(agent.actor.state_dict());temperature_before=_state_digest(agent.log_ent_coef);global_before=rng_digest()
    initial_hash=_state_digest(agent.critic.state_dict());optimizer_initial=optimizer.state_dict()
    config=dict(rate=rate,seed=seed,updates=budget,batch_size=32,initial_critic_sha256=initial_hash,
        batch_index_sha256=hashlib.sha256(batch_indices.tobytes()).hexdigest(),train_points=len(train_a),holdout_points=len(test_a),
        optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer_initial['param_groups']],
        source_sha256=source_hashes(),objective='0.5*(MSE(Q1,float32(r))+MSE(Q2,float32(r))); terminal target r, no bootstrap',
        fixed_data=True,actor_updates=0,temperature_updates=0,native_calls=0)
    write_json(folder/'config.json',config)
    train_a=torch.as_tensor(train_a);test_a=torch.as_tensor(test_a);train_y=torch.as_tensor(train_r.astype(np.float32)).reshape(-1,1)
    def obs(n):
        x=torch.zeros((n,8));x[:,0]=-1;return x
    train_obs=obs(len(train_a));test_obs=obs(len(test_a));schedule=settings['evaluation_updates']
    def evaluate(n):
        before=rng_digest();records=[];predictions={}
        with torch.no_grad():
            for split,a,x,r in (('train',train_a,train_obs,train_r),('holdout99',test_a,test_obs,test_r)):
                q1,q2=agent.critic(x,a);heads=dict(Q1=q1[:,0].numpy().astype(float),Q2=q2[:,0].numpy().astype(float));heads['minQ']=np.minimum(heads['Q1'],heads['Q2'])
                low=r>=np.quantile(r,.9)
                for subset,mask in (('all',np.ones(len(r),bool)),('lowJ_decile',low)):
                    for head,p in heads.items():records.append(dict(update=n,split=split,subset=subset,head=head,**scores(r[mask],p[mask])))
                predictions[split]=heads
        assert rng_digest()==before
        history.extend(records);return predictions
    try:
        final=evaluate(0)
        for i,indices in enumerate(batch_indices,1):
            assert [g['lr'] for g in optimizer.param_groups]==[rate]
            q1,q2=agent.critic(train_obs[indices],train_a[indices]);target=train_y[indices]
            loss=.5*(torch.nn.functional.mse_loss(q1,target)+torch.nn.functional.mse_loss(q2,target))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite offline Critic loss')
            optimizer.zero_grad();loss.backward();pre=numerical_check(optimizer,gradients=True)
            optimizer.step();completed=i;post=numerical_check(optimizer)
            for state in (pre,post):
                for k,v in state.items():maxima[k]=max(maxima.get(k,0.),v)
            trace.append(dict(update=i,loss=float(loss.detach()),lr=rate,**pre))
            if i in schedule:final=evaluate(i)
        assert completed==budget and len(trace)==budget
        assert actor_before==_state_digest(agent.actor.state_dict()) and temperature_before==_state_digest(agent.log_ent_coef)
        assert global_before==rng_digest(),'Offline fit unexpectedly consumed a global RNG'
        gradients=fixed_gradients(agent,test_a.numpy())
        torch.save(dict(critic=agent.critic.state_dict(),rate=rate,seed=seed,updates=budget,
            format='Issue25 offline twin Critic only; never connected to online Actor'),folder/'critic.pt')
        np.savez_compressed(folder/'predictions.npz',**{f'{split}_{head}':values for split,heads in final.items() for head,values in heads.items()},holdout_gradients=gradients)
        status='complete';failure=None
    except Exception as exc:
        status='failed';failure=dict(type=type(exc).__name__,message=str(exc),completed_updates=completed);write_json(folder/'failure.json',failure)
    pack(folder/'history.json.gz',history);pack(folder/'training_trace.json.gz',trace)
    s=dict(status=status,rate=rate,seed=seed,updates=completed,planned_updates=budget,initial_critic_sha256=initial_hash,
        batch_index_sha256=config['batch_index_sha256'],numeric_maxima=maxima,actor_unchanged=actor_before==_state_digest(agent.actor.state_dict()),
        temperature_unchanged=temperature_before==_state_digest(agent.log_ent_coef),global_rng_unchanged=global_before==rng_digest(),
        native_calls=0,actor_updates=0,temperature_updates=0,failure=failure,source_sha256=source_hashes(),wall_s=time.perf_counter()-tick,
        artifact_sha256={p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()})
    write_json(folder/'summary.json',s)
    print(json.dumps(dict(stage='offline_complete',rate=rate,seed=seed,status=status,updates=completed)),flush=True);return s


def train(workers=3,output=OUT):
    output=Path(output);ready(output)
    if (output/'offline_execution.json').exists():print('Completed offline execution preserved',flush=True);return
    online=json.loads((output/'execution.json').read_text());assert online['runs']==6,'Stage B must finish before Stage C'
    tick=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=min(workers,9),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(offline_fit,r,s,output) for r in RATES for s in SEEDS]
        for f in as_completed(futures):results.append(f.result())
    write_json(output/'offline_execution.json',dict(runs=9,complete_runs=sum(r['status']=='complete' for r in results),
        failed_runs=sum(r['status']=='failed' for r in results),updates=sum(r['updates'] for r in results),native_calls=0,
        actor_updates=0,temperature_updates=0,wall_s=time.perf_counter()-tick,workers=min(workers,9),source_sha256=source_hashes()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);a=p.parse_args();train(a.workers)
