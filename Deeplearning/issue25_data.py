"""Frozen online/offline cohorts and existing model-generated data only."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import json
from pathlib import Path

import numpy as np

from .issue23_data import (ROOT,OUT as BASE,TARGET,SEEDS,NAMES,HALF_WIDTH,environment,chosen,
    policy_statistics,deterministic_state,predict_q,q_action_gradient,load_checkpoint,
    source_hashes as baseline_hashes,pack,metric_row,save_audio,target_data)
from .issue24_data import OUT as PREVIOUS,sb3_hashes
from .issue25_agent import BASE_LR,CRITIC_RATES
from .issue17_artifacts import read_record
from .issue16 import digest

OUT=ROOT/'results/Phase1/CaseA/learning_rate/critic'
RATES=(BASE_LR,*CRITIC_RATES)
OFFLINE_UPDATES=4080
OFFLINE_EVALS=tuple([0,*range(128,OFFLINE_UPDATES,128),OFFLINE_UPDATES])
SOURCES=('issue25_agent.py','issue25_data.py','issue25_training.py','issue25_offline.py','issue25_prepare.py',
    'issue24_agent.py','issue24_data.py',*baseline_hashes())


def source_hashes():return {p:digest(resolve_result_path(ROOT/p)) for p in SOURCES}


def label(rate):return {BASE_LR:'critic_lr_3e-4',1e-4:'critic_lr_1e-4',1e-3:'critic_lr_1e-3'}[rate]


def folder_for(rate,seed,output=OUT):
    if rate not in RATES or seed not in SEEDS:raise ValueError('Unregistered LR/seed')
    return BASE/'runs'/f'seed_{seed}' if rate==BASE_LR else Path(output)/'runs'/label(rate)/f'seed_{seed}'


def offline_folder(rate,seed,output=OUT):return Path(output)/'offline'/label(rate)/f'seed_{seed}'


def curve_for(rate,seed,output=OUT):return read_record(folder_for(rate,seed,output)/'curve.json')


def ready(output=OUT):
    from .model import LIBRARY
    output=Path(output);m=json.loads((output/'manifest.json').read_text())
    assert source_hashes_match(m['source_sha256'], source_hashes()),'Frozen numerical implementation changed'
    assert m['native_library_sha256']==digest(LIBRARY) and m['target_sha256']==digest(TARGET)
    assert m['baseline_manifest_sha256']==digest(BASE/'manifest.json')
    assert m['previous_manifest_sha256']==digest(PREVIOUS/'manifest.json')
    assert m['sb3_implementation_sha256']==sb3_hashes()
    assert m['plan_sha256']==digest(output/'PLAN.md')
    for p,h in m['data_sha256'].items():assert digest(output/p)==h,p
    for p,h in m['local_reference_sha256'].items():assert digest(BASE/p)==h,p
    return m


def fixed_gradients(agent,actions):
    import torch
    obs=torch.zeros((len(actions),8));obs[:,0]=-1
    a=torch.tensor(agent.policy.scale_action(np.asarray(actions,np.float32)),requires_grad=True)
    q1,q2=agent.critic(obs,a)
    return torch.autograd.grad(torch.minimum(q1,q2).sum(),a)[0].detach().numpy()
