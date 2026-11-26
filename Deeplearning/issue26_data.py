"""Wide-domain candidate-generation experiment; legacy outputs remain read only."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import json
from pathlib import Path

import numpy as np

from .issue23_data import (ROOT,OUT as BASE,TARGET,SEEDS,NAMES,chosen,policy_statistics,
    deterministic_state,pack,metric_row,save_audio,source_hashes as baseline_hashes,target_data)
from .issue22_data import SumOneEnv
from .issue24_data import sb3_hashes
from .issue16 import digest
from .issue17_artifacts import read_record

OUT=ROOT/'results/Phase1/CaseA/exploration_strategy'
CONDITIONS=('A','B','C','D')
NEW=('B','C','D')
PREFIX=256
LOCAL_PROBABILITY=.8
RADII={'C':np.array([625000.,37500.,50000.]),'D':np.array([625000.,37500.,250000.])}
LOW=np.array([2.5e7,2.5e6,49000.]);HIGH=np.array([5e7,4e6,1e7])
F0_BINS=(49000.,800000.,2500000.,5000000.,10000000.)
SOURCES=('issue26_data.py','issue26_agent.py','issue26_training.py','issue26_prepare.py',
    'issue25_agent.py','issue24_agent.py','issue24_data.py',*baseline_hashes())


class ExperimentEnv(SumOneEnv):
    def __init__(self,condition):
        if condition not in CONDITIONS:raise ValueError('Unknown condition')
        super().__init__(.75);self.domain_high[3]=800000. if condition=='A' else 1e7
        self.best_candidate=None;self.best_values=None

    def step(self,action):
        obs,r,terminated,truncated,info=super().step(action)
        if self.best_candidate is None or info['weighted_loss']<self.best_candidate['weighted_loss']:
            self.best_candidate=dict(trial=self.completed_episodes,**info)
            self.best_values=dict(audio=np.concatenate(self.audio).copy(),physical_controls=np.asarray(self.applied_control_history).copy(),actions=np.asarray(self.actions).copy())
        return obs,r,terminated,truncated,info


def environment(condition):return ExperimentEnv(condition)


def folder_for(condition,seed,output=OUT):
    if condition not in CONDITIONS or seed not in SEEDS:raise ValueError('Unregistered condition/seed')
    return BASE/'runs'/f'seed_{seed}' if condition=='A' else Path(output)/'runs'/condition/f'seed_{seed}'


def curve_for(condition,seed,output=OUT):return read_record(folder_for(condition,seed,output)/'curve.json')


def half_width(condition):return np.array([12500000.,750000.,375500. if condition=='A' else 4975500.])


def source_hashes():return {p:digest(resolve_result_path(ROOT/p)) for p in SOURCES}


def ready(output=OUT):
    from .model import LIBRARY
    output=Path(output);m=json.loads((output/'manifest.json').read_text())
    assert source_hashes_match(m['source_sha256'], source_hashes()),'Frozen numerical implementation changed'
    assert m['native_library_sha256']==digest(LIBRARY) and m['target_sha256']==digest(TARGET)
    assert m['baseline_manifest_sha256']==digest(BASE/'manifest.json') and m['sb3_implementation_sha256']==sb3_hashes()
    assert m['plan_sha256']==digest(output/'PLAN.md')
    return m


def save_checkpoint(path,agent,row,condition):
    import torch
    torch.save(dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),condition=condition,trial=row['episodes'],
        action=row['actions'][0],alpha=row['policy_statistics']['alpha'],J=row['weighted_loss'],E_total=row['acoustic_error'],
        training_incumbent=agent.incumbent,format='Issue26 inference Actor+online twin Critic; no optimizer/Replay/sampler-resume state'),Path(path))


def load_checkpoint(path,condition,seed,config):
    import torch
    from .issue21_training import make_agent
    env=environment(condition)
    try:agent=make_agent('SAC',128,seed,env,config)
    finally:env.close()
    data=torch.load(Path(path),map_location='cpu',weights_only=True)
    agent.actor.load_state_dict(data['actor']);agent.critic.load_state_dict(data['critic'])
    with torch.no_grad():agent.log_ent_coef.fill_(np.log(data['alpha']))
    return agent,data
