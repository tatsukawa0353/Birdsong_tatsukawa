"""Preregistered three Targets, paired3D/4D Case B experiment."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import json
from pathlib import Path
import numpy as np
from .issue16 import digest
from .issue19_data import pack,metric_row
from .issue23_data import chosen
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/Phase1/CaseB/3targets_3d4d'
OLD=ROOT/'results/Phase1/CaseB/target_diagnostics/low_forces'
METHODS=('SAC','TD3','PPO');SEEDS=(0,1,2);DIMS=(3,4)
CONTROLS={'T1':np.array([3.5e7,3.5e7,3e6,49000.,1e7]),'T2':np.array([3.5e7,3.5e7,3e6,49000.,49000.]),'T3':np.array([3.5e7,2.1e7,3e6,49000.,49000.])}
LOW=np.array([2.5e7,2e7,2.5e6,49000.,49000.]);HIGH=np.array([5e7,5e7,4e6,800000.,1e7])
CONFIG=dict(episodes=4096,eval_every_episodes=128,learning_rate=3e-4,gamma=1.,batch_size=32,buffer_size=50000,warmup_episodes=16,train_freq=1,gradient_steps=1,
    sac_ent_coef='auto_0.01',sac_target_entropy='-action_dim',td3_noise_std=.1,ppo_n_steps=128,ppo_n_epochs=10,ppo_clip_range=.2)
CORE=('issue27g_data.py','issue27g_native.py','issue27g_env.py','issue27g_agent.py','issue27g_training.py','issue27g_prepare.py',
    'issue27f_agent.py','issue27f_prepare.py','issue27f_native.py','issue27f_data.py','issue27_probe.cpp','model.py','native/model.cpp','model_env.py','acoustics.py',
    'constant020_env.py','issue18_domain_env.py','issue14.py','issue19_data.py','issue23_data.py','issue22_data.py','issue25_agent.py')
def hashes():return {n:digest(resolve_result_path(ROOT/n)) for n in CORE}
def target_path(target):
    if target not in CONTROLS:raise ValueError(target)
    return OUT/f'preflight/{target}/target.npz'
def target_data(target):
    with np.load(target_path(target)) as z:return z['audio'].copy(),z['controls'].copy()
def indices(dim):
    if dim not in DIMS:raise ValueError(dim)
    return [0,2,3] if dim==3 else [0,1,2,3]
def half_width(dim):return (HIGH[indices(dim)]-LOW[indices(dim)])/2
def folder(target,dim,method,seed,output=OUT):
    assert target in CONTROLS and dim in DIMS and method in METHODS and seed in SEEDS
    return Path(output)/'runs'/target/f'{dim}D'/method/f'seed_{seed}'
def ready(output=OUT):
    from .model import LIBRARY
    from .issue27g_native import PROBE
    m=json.loads((Path(output)/'manifest.json').read_text());assert m['preflight_passed'] and source_hashes_match(m['source_sha256'], hashes())
    assert m['native_library_sha256']==digest(LIBRARY) and m['probe_sha256']==digest(PROBE) and m['plan_sha256']==digest(OUT/'PLAN.md')
    assert all(digest(target_path(t))==h for t,h in m['target_sha256'].items());return m
def save_values(path,values):np.savez_compressed(path,**values,sample_rate=50000,duration_s=.2,model_condition='B')
def save_checkpoint(path,agent,method,row):
    import torch
    env=agent.get_env().envs[0].unwrapped
    torch.save(dict(method=method,target=env.target_name,dim=env.dim,policy=agent.policy.state_dict(),log_ent_coef=agent.log_ent_coef.detach().clone() if method=='SAC' else None,
        trial=row['episodes'],action=row['actions'][0],J=row['weighted_loss'],E_total=row['acoustic_error'],condition='B',target_sha256=digest(target_path(env.target_name)),
        format='Inference weights only; optimizer/Replay/rollout resume state not included'),Path(path))
def load_checkpoint(path,target,dim,method,seed):
    import torch
    from .issue27g_env import CaseBEnv
    from .issue27g_agent import make_agent
    env=CaseBEnv(target,dim)
    try:agent=make_agent(method,seed,env,CONFIG,audited=False)
    finally:env.close()
    r=torch.load(path,map_location='cpu',weights_only=True)
    assert (r['target'],r['dim'],r['method'],r['condition'],r['target_sha256'])==(target,dim,method,'B',digest(target_path(target)))
    agent.policy.load_state_dict(r['policy'])
    if method=='SAC':
        with torch.no_grad():agent.log_ent_coef.copy_(r['log_ent_coef'])
    return agent,r
