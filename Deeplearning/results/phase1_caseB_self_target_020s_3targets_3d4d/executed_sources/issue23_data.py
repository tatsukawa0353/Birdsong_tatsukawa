"""Fixed inputs and non-mutating diagnostics for the Issue23 reproduction."""
import json
from pathlib import Path

import numpy as np

from .issue22_data import (ROOT,OUT as BASE,TARGET,SEEDS,environment,policy_statistics,
    folder_for as old_folder,curve_for as old_curve,source_hashes as prior_hashes,
    pack,metric_row,save_audio,target_data)
from .issue16 import digest

OUT=ROOT/'results/phase1_caseA_sac128_stability_diagnostics'
NAMES=('epsilon_l','ps','f0_l')
HALF_WIDTH=np.array([12500000.,750000.,375500.])
SOURCES=('issue23_data.py','issue23_training.py',*prior_hashes())
LOCAL_STEPS=(.002,.01)
LOCAL_ANCHORS=(2944,3840,4096)


def source_hashes():return {p:digest(ROOT/p) for p in SOURCES}


def folder_for(seed,output=OUT):
    if seed not in SEEDS:raise ValueError('Only the three registered SAC seeds are allowed')
    return Path(output)/'runs'/f'seed_{seed}'


def curve_for(seed,output=OUT):
    from .issue17_artifacts import read_record
    return read_record(folder_for(seed,output)/'curve.json')


def chosen(points):
    return dict(final=points[-1],selected=min(points,key=lambda p:p['weighted_loss']),
        posthoc=min(points,key=lambda p:p['acoustic_error']))


def ready(output=OUT):
    from .model import LIBRARY
    m=json.loads((Path(output)/'manifest.json').read_text())
    assert m['source_sha256']==source_hashes(),'Frozen reproduction implementation changed'
    assert m['native_library_sha256']==digest(LIBRARY) and m['target_sha256']==digest(TARGET)
    assert m['stage_a_sha256']==digest(Path(output)/'STAGE_A.md')
    assert m['candidates_sha256']==digest(Path(output)/'fixed_candidates.json')
    return m


def fixed_candidates():
    """Retrospective, fixed before fitting; never replay-buffer experience."""
    return [dict(candidate_id=f's{s}_t{p["episodes"]}',source_seed=s,source_trial=p['episodes'],
        action=p['actions'][0],physical_controls=p['physical_controls'],
        E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],
        E_env=p['acoustic_components']['envelope'],J=p['weighted_loss'],reward=-p['weighted_loss'])
        for s in SEEDS for p in old_curve(.75,s)]


def deterministic_state(agent):
    """No distribution sampling, mode changes, environment steps or RNG use."""
    import torch
    with torch.no_grad():
        obs=torch.zeros((1,8));obs[:,0]=-1
        mean,log_std,_=agent.actor.get_action_dist_params(obs)
        action=torch.tanh(mean)
        q1,q2=agent.critic(obs,action)
        return dict(action=action[0].tolist(),log_std=log_std[0].tolist(),
            alpha=float(agent.log_ent_coef.exp()),Q1=float(q1),Q2=float(q2),minQ=float(torch.minimum(q1,q2)))


def predict_q(agent,actions):
    import torch
    a=np.asarray(actions,dtype=np.float32)
    scaled=agent.policy.scale_action(a)
    np.testing.assert_allclose(scaled,a,rtol=0,atol=1.2e-7)
    with torch.no_grad():
        obs=torch.zeros((len(a),8));obs[:,0]=-1
        q1,q2=agent.critic(obs,torch.as_tensor(scaled))
    return q1[:,0].numpy().astype(float),q2[:,0].numpy().astype(float)


def q_action_gradient(agent,action):
    """Gradient with respect to normalized action; do not populate parameter .grad."""
    import torch
    obs=torch.zeros((1,8));obs[:,0]=-1
    a=torch.tensor(agent.policy.scale_action(np.asarray([action],np.float32)),requires_grad=True)
    q1,q2=agent.critic(obs,a);q=torch.minimum(q1,q2)
    return torch.autograd.grad(q.sum(),a)[0][0].detach().tolist()


def save_checkpoint(path,agent,row):
    """Diagnostic/inference weights only; no optimizer, replay or resume claim."""
    import torch
    torch.save(dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),
        trial=row['episodes'],action=row['actions'][0],alpha=row['policy_statistics']['alpha'],
        J=row['weighted_loss'],E_total=row['acoustic_error'],
        format='Issue23 actor+online twin critic; inference/diagnosis only'),Path(path))


def load_checkpoint(path,seed,config):
    import torch
    from .issue21_training import make_agent
    env=environment(.75)
    try:agent=make_agent('SAC',128,seed,env,config)
    finally:env.close()
    data=torch.load(Path(path),map_location='cpu',weights_only=True)
    agent.actor.load_state_dict(data['actor']);agent.critic.load_state_dict(data['critic'])
    with torch.no_grad():agent.log_ent_coef.fill_(np.log(data['alpha']))
    return agent,data
