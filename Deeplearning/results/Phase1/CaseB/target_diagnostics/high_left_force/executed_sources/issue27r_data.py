"""Revised Issue27, independent of the preserved failed-Target experiment."""
import json
from pathlib import Path
import numpy as np
from .issue16 import digest
from .issue19_data import pack,metric_row
from .issue17_artifacts import read_record
from .issue23_data import chosen

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/phase1_caseB_self_target_020s_methods'
BASE=ROOT/'results/phase1_caseA_sac128_stability_diagnostics'
OLD=ROOT/'results/phase1_caseB_self_target_020s_sac128'
TARGET=OUT/'preflight/target.npz'
METHODS=('SAC','TD3','PPO');SEEDS=(0,1,2)
GRID_E=(3e7,3.5e7,4.1e7);GRID_P=(3e6,3.4e6,3.8e6)
HALF_WIDTH=np.array([12500000.,750000.,375500.])
CORE=('issue27r_data.py','issue27r_native.py','issue27r_prepare.py','issue27r_env.py','issue27r_agent.py','issue27r_training.py',
    'issue27_probe.cpp','model.py','native/model.cpp','model_env.py','acoustics.py','constant020_env.py','issue18_domain_env.py',
    'issue14.py','issue19_data.py','issue23_data.py','issue22_data.py','issue25_agent.py')
def hashes():return {name:digest(ROOT/name) for name in CORE}
def target_data():
    with np.load(TARGET) as z:return z['audio'].copy(),z['controls'].copy()
def folder(method,seed,output=OUT):
    if method not in METHODS or seed not in SEEDS:raise ValueError('Unregistered method/seed')
    return Path(output)/'runs'/method/f'seed_{seed}'
def curve_for(method,seed):return read_record(folder(method,seed)/'curve.json')
def save_values(path,values):np.savez_compressed(path,**values,sample_rate=50000,duration_s=.2,model_condition='B')
def ready(output=OUT):
    from .model import LIBRARY
    from .issue27r_native import PROBE
    p=Path(output);m=json.loads((p/'manifest.json').read_text())
    assert m['preflight_passed'] and m['source_sha256']==hashes()
    assert m['native_library_sha256']==digest(LIBRARY) and m['probe_sha256']==digest(PROBE)
    assert m['target_sha256']==digest(TARGET) and m['plan_sha256']==digest(OUT/'PLAN.md')
    return m

def save_checkpoint(path,agent,method,row):
    import torch
    torch.save(dict(method=method,policy=agent.policy.state_dict(),log_ent_coef=agent.log_ent_coef.detach().clone() if method=='SAC' else None,
        trial=row['episodes'],action=row['actions'][0],J=row['weighted_loss'],E_total=row['acoustic_error'],condition='B',target_sha256=digest(TARGET),
        format='Inference policy/state only; no optimizer/Replay/rollout resume state'),Path(path))
def load_checkpoint(path,method,seed,config):
    import torch
    from .issue27r_env import CaseBEnv
    from .issue27r_agent import make_agent
    env=CaseBEnv()
    try:agent=make_agent(method,seed,env,config,audited=False)
    finally:env.close()
    record=torch.load(path,map_location='cpu',weights_only=True)
    assert record['condition']=='B' and record['method']==method and record['target_sha256']==digest(TARGET)
    agent.policy.load_state_dict(record['policy'])
    if method=='SAC':
        with torch.no_grad():agent.log_ent_coef.copy_(record['log_ent_coef'])
    return agent,record
