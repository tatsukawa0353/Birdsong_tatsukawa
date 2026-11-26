"""Case B self-Target experiment; all legacy Case A outputs stay read only."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import json
from pathlib import Path
import numpy as np
from .issue16 import digest
from .issue22_data import policy_statistics
from .issue23_data import deterministic_state,chosen
from .issue19_data import pack,metric_row
from .issue17_artifacts import read_record

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/Phase1/CaseB/target_diagnostics/initial_stop'
BASE=ROOT/'results/Phase1/CaseA/stability_diagnostics'
SEEDS=(0,1,2)
CONTROLS=np.array([4.1e7,3.5e7,3.4e6,3e5,1e7])
INDICES=[0,2,3]
HALF_WIDTH=np.array([12500000.,750000.,375500.])
TARGET=OUT/'preflight/target.npz'
SOURCES=('issue27_data.py','issue27_native.py','issue27_probe.cpp','issue27_env.py','issue27_prepare.py',
    'issue21_training.py','issue22_data.py','issue23_data.py','issue25_agent.py','issue24_agent.py','issue24_data.py',
    'issue19_data.py','issue18_domain_env.py','issue14.py','model.py','model_env.py','native/model.cpp','constant020_env.py','acoustics.py')

def source_hashes():return {p:digest(resolve_result_path(ROOT/p)) for p in SOURCES}
def target_data():
    with np.load(TARGET) as z:return z['audio'].copy(),z['controls'].copy()
def folder_for(seed,output=OUT):
    if seed not in SEEDS:raise ValueError('Only registered seeds0,1,2')
    return Path(output)/'runs'/f'seed_{seed}'
def curve_for(seed):return read_record(folder_for(seed)/'curve.json')
def save_values(path,values):np.savez_compressed(path,**values,sample_rate=50000,duration_s=.2,model_condition='B')
def ready(output=OUT):
    from .model import LIBRARY
    from .issue27_native import PROBE
    p=Path(output);m=json.loads((p/'manifest.json').read_text())
    if not m['preflight_passed']:raise RuntimeError('Case B Target failed the physical gate; no training is authorized under this protocol')
    assert source_hashes_match(m['source_sha256'], source_hashes()) and m['native_library_sha256']==digest(LIBRARY)
    assert m['target_sha256']==digest(TARGET) and m['probe_sha256']==digest(PROBE)
    assert m['plan_sha256']==digest(p/'PLAN.md') and m['preflight_passed']
    return m

def save_checkpoint(path,agent,row):
    import torch
    torch.save(dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),alpha=row['policy_statistics']['alpha'],
        trial=row['episodes'],action=row['actions'][0],J=row['weighted_loss'],E_total=row['acoustic_error'],condition='B',
        target_sha256=digest(TARGET),format='Inference Actor/online Critic only; not optimizer/Replay resume state'),Path(path))

def load_checkpoint(path,seed,config):
    import torch
    from .issue27_env import CaseBEnv
    from .issue21_training import make_agent
    env=CaseBEnv()
    try:agent=make_agent('SAC',128,seed,env,config)
    finally:env.close()
    meta=torch.load(path,map_location='cpu',weights_only=True)
    assert meta['condition']=='B' and meta['target_sha256']==digest(TARGET)
    agent.actor.load_state_dict(meta['actor']);agent.critic.load_state_dict(meta['critic'])
    with torch.no_grad():agent.log_ent_coef.fill_(np.log(meta['alpha']))
    return agent,meta
