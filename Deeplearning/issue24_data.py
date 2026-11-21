"""Immutable baseline, local-sensitivity references and fixed Actor LR cohorts."""
import json
import hashlib
import inspect
from pathlib import Path

from .issue23_data import (ROOT,OUT as BASE,TARGET,SEEDS,NAMES,HALF_WIDTH,environment,
    chosen,policy_statistics,deterministic_state,predict_q,q_action_gradient,load_checkpoint,
    source_hashes as baseline_hashes,pack,metric_row,save_audio,target_data)
from .issue24_agent import BASE_LR,ACTOR_RATES
from .issue16 import digest
from .issue17_artifacts import read_record

OUT=ROOT/'results/phase1_caseA_sac128_actor_lr'
RATES=(BASE_LR,*ACTOR_RATES)
SOURCES=('issue24_agent.py','issue24_data.py','issue24_training.py',*baseline_hashes())


def source_hashes():return {p:digest(ROOT/p) for p in SOURCES}


def sb3_hashes():
    from stable_baselines3 import SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
    return {name:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for name,fn in
        (('SAC.train',SAC.train),('BaseAlgorithm._update_learning_rate',BaseAlgorithm._update_learning_rate))}


def label(rate):return {BASE_LR:'actor_lr_3e-4',1e-4:'actor_lr_1e-4',3e-5:'actor_lr_3e-5'}[rate]


def folder_for(rate,seed,output=OUT):
    if rate not in RATES or seed not in SEEDS:raise ValueError('Unregistered LR/seed')
    return BASE/'runs'/f'seed_{seed}' if rate==BASE_LR else Path(output)/'runs'/label(rate)/f'seed_{seed}'


def curve_for(rate,seed,output=OUT):return read_record(folder_for(rate,seed,output)/'curve.json')


def ready(output=OUT):
    from .model import LIBRARY
    m=json.loads((Path(output)/'manifest.json').read_text())
    assert m['source_sha256']==source_hashes(),'Frozen experiment implementation changed'
    assert m['native_library_sha256']==digest(LIBRARY) and m['target_sha256']==digest(TARGET)
    assert m['baseline_manifest_sha256']==digest(BASE/'manifest.json')
    if 'sb3_implementation_sha256' in m:assert m['sb3_implementation_sha256']==sb3_hashes()
    assert m['candidates_sha256']==digest(BASE/'fixed_candidates.json')
    for p,h in m['local_reference_sha256'].items():assert digest(BASE/p)==h,p
    return m
