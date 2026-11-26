"""Coefficient-sum-one rewards; immutable SAC128 baseline and shared diagnostics."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import json
from pathlib import Path

import numpy as np

from .issue17 import ROOT
from .issue17_artifacts import read_record
from .issue18_domain_env import DomainEnv
from .issue19_data import pack,metric_row,save_audio
from .issue21_data import target_data
from .issue16 import digest

BASE=ROOT/'results/Phase1/CaseA/network_width'
TARGET=ROOT/'results/Phase1/CaseA/domain_diagnostics/diagnostics/target.npz'
POOL=ROOT/'results/Phase1/CaseA/reward_weights/restricted_domain/critic_pool.csv'
OUT=ROOT/'results/Phase1/CaseA/reward_weights/sac128'
WEIGHTS=(.5,.75,.9)
CONDITIONS=('legacy_sum',*WEIGHTS)
SEEDS=(0,1,2)
SOURCES=('issue22_data.py','issue22_training.py','issue21_training.py','issue21_data.py',
    'issue19_data.py','issue18_domain_env.py','constant020_env.py','issue14.py',
    'model.py','model_env.py','native/model.cpp','acoustics.py')
ENTROPY_SAMPLES=4096
ENTROPY_SEED=220026


def source_hashes():return {p:digest(resolve_result_path(ROOT/p)) for p in SOURCES}


def coefficients(weight):
    if not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('lambda must be in [0,1]')
    return float(weight),1.-float(weight)


def objective(spec,env,weight):
    a,b=coefficients(weight)
    if not np.isfinite([spec,env]).all():raise FloatingPointError('Nonfinite acoustic component')
    return float(a*spec+b*env)


class SumOneEnv(DomainEnv):
    """Change only terminal reward; physics, reset, mapping and common loss stay intact."""
    def __init__(self,weight):
        self.spectral_weight,self.envelope_coefficient=coefficients(weight)
        target,controls=target_data()
        super().__init__((0,2,3),target,controls,envelope_weight=1.,f0_high=800000.)

    def step(self,action):
        obs,_,terminated,truncated,metrics=super().step(action)
        comp=metrics['acoustic_components']
        loss=1e6 if metrics['failed'] else objective(comp['log_spectral'],comp['envelope'],self.spectral_weight)
        metrics.pop('envelope_weight')
        metrics.update(weighted_loss=loss,J_lambda=loss,negative_return=loss,
            spectral_coefficient=self.spectral_weight,envelope_coefficient=self.envelope_coefficient,
            reward_definition='-(lambda E_spec + (1-lambda) E_env); coefficients sum 1, no factor 2')
        self.rewards=[-loss]
        return obs,-loss,terminated,truncated,metrics


def environment(condition):
    if condition=='legacy_sum':
        target,controls=target_data()
        return DomainEnv((0,2,3),target,controls,envelope_weight=1.,f0_high=800000.)
    return SumOneEnv(condition)


def label(condition):return 'legacy_sum' if condition=='legacy_sum' else f'lambda_{condition:.2f}'.replace('.','p')


def folder_for(condition,seed,output=OUT):
    if condition not in CONDITIONS or seed not in SEEDS:raise ValueError('Unregistered condition/seed')
    return (BASE/'runs/SAC/width_128'/f'seed_{seed}' if condition=='legacy_sum' else Path(output)/'runs'/label(condition)/f'seed_{seed}')


def curve_for(condition,seed,output=OUT):return read_record(folder_for(condition,seed,output)/'curve.json')


def endpoints(condition,seed,output=OUT):
    points=curve_for(condition,seed,output)
    return dict(final=points[-1],selected=min(points,key=lambda p:p['weighted_loss']),
        posthoc=min(points,key=lambda p:p['acoustic_error']))


def policy_statistics(agent):
    """Fixed local quadrature samples; never consume learner/global RNG or simulate."""
    import torch
    assert not agent.use_sde
    obs=torch.zeros((1,8),dtype=torch.float32,device='cpu');obs[:,0]=-1
    with torch.no_grad():
        mean,log_std,_=agent.actor.get_action_dist_params(obs)
        std=log_std.exp()
        z=torch.as_tensor(np.random.default_rng(ENTROPY_SEED).standard_normal((ENTROPY_SAMPLES,3)).astype(np.float32))
        latent=mean+std*z;actions=torch.tanh(latent)
        log_probability=torch.distributions.Normal(mean,std).log_prob(latent).sum(dim=1)
        log_probability-=torch.log(1-actions**2+agent.actor.action_dist.epsilon).sum(dim=1)
        entropy=-log_probability
        result=dict(alpha=float(agent.log_ent_coef.exp().item()),latent_mean=mean[0].tolist(),
            latent_log_std=log_std[0].tolist(),latent_std=std[0].tolist(),
            normalized_action_std_mc=actions.std(dim=0,unbiased=True).tolist(),
            squashed_entropy_mc=float(entropy.mean()),entropy_mc_se=float(entropy.std(unbiased=True)/np.sqrt(ENTROPY_SAMPLES)),
            gaussian_entropy=float((log_std+.5*np.log(2*np.pi*np.e)).sum()),
            deterministic_action=torch.tanh(mean)[0].tolist(),entropy_samples=ENTROPY_SAMPLES,entropy_seed=ENTROPY_SEED)
        assert np.isfinite([result['alpha'],result['squashed_entropy_mc'],result['gaussian_entropy']]).all()
        return result


def ready(output=OUT):
    from .model import LIBRARY
    m=json.loads((Path(output)/'manifest.json').read_text())
    assert source_hashes_match(m['source_sha256'], source_hashes()),'Frozen training implementation changed'
    assert m['native_library_sha256']==digest(LIBRARY) and m['target_sha256']==digest(TARGET)
    assert m['critic_pool_sha256']==digest(POOL)
    return m
