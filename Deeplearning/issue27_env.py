"""Explicit Case B constant3D environment; old Case A constructors are untouched.

The failed self-Target gate prevents training in this study. This independent
environment is checked without integrating any additional sounds.
"""
import gymnasium as gym
import json
import numpy as np
from .model import BirdModel,LOW,HIGH
from .model_env import AudioDecimator
from .issue18_domain_env import DomainEnv
from .issue27_data import target_data,OUT
from .issue27_native import snapshot
from .acoustics import distance


class CaseBEnv(gym.Env):
    metadata={'render_modes':[]};horizon=1;physical_horizon=20
    encode=DomainEnv.encode;decode=DomainEnv.decode

    def __init__(self):
        target,controls=target_data();self.target_audio=target;self.fixed_controls=controls
        assert target.shape==(10000,) and np.isfinite(target).all()
        assert controls[1]==3.5e7 and controls[4]==1e7
        self.target_audio.setflags(write=False);self.fixed_controls.setflags(write=False)
        self.indices=[0,2,3];self.domain_low=LOW.copy();self.domain_high=HIGH.copy();self.domain_high[3]=800000.
        self.loss_scale=max(float(abs(target).max()),1e-8)
        self.action_space=gym.spaces.Box(-1,1,(3,),np.float32);self.observation_space=gym.spaces.Box(-1,1,(8,),np.float32)
        self.model=BirdModel('B');self.decision_steps=self.completed_episodes=self.failures=0
        self.native_blocks=self.native_blocks_completed=self.silent_episodes=self.nonvoiced_episodes=0
        self.integrated_s=self.integrated_completed_s=0.;self.done=True;self.best_candidate=None;self.best_values=None
        self.native_checks=0

    def check(self,reset=False):
        value=snapshot(self.model,reset=reset);self.native_checks+=1;return value

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.action_space.seed(seed)
        self.model.reset(self.fixed_controls);self.check(reset=True);self.decimator=AudioDecimator()
        self.audio=[];self.actions=[];self.applied_control_history=[];self.motion_means=[];self.motion_stds=[];self.states=[];self.source_terms=[]
        self.done=False;obs=np.zeros(8,np.float32);obs[0]=-1;return obs,{}

    def step(self,action):
        gate=json.loads((OUT/'preflight/physics.json').read_text())
        if not gate['passed']:raise RuntimeError('Case B Target preflight failed; learning/integration is disabled')
        if self.done:raise RuntimeError('Reset before each constant candidate')
        controls=self.decode(action);assert controls[1]==3.5e7 and controls[4]==1e7
        self.model.set_controls(controls);self.check();failed=False
        for _ in range(20):
            self.native_blocks+=1;self.integrated_s+=.01
            try:
                raw,moments=self.model.advance(.01,return_moments=True);wave=self.decimator.process(raw)
                if not np.isfinite(wave).all():raise FloatingPointError('Nonfinite filtered Case B output')
            except FloatingPointError:failed=True;self.failures+=1;break
            self.native_blocks_completed+=1;self.integrated_completed_s+=.01;self.audio.append(wave)
            self.motion_means.append(moments['mean']);self.motion_stds.append(moments['std']);state=self.check()
            self.states.append(state['state']);self.source_terms.append([state['source_l'],state['source_r']])
        if failed:self.audio.extend([np.zeros(500)]*(20-len(self.audio)))
        audio=np.concatenate(self.audio);comp=distance(self.target_audio,audio,50000,scale=self.loss_scale)
        j=1e6 if failed else .75*comp['log_spectral']+.25*comp['envelope'];e=1e6 if failed else comp['total']
        voiced=bool(np.std(audio[-1000:])/self.loss_scale>.05) and not failed
        self.actions=[np.asarray(action,np.float64).clip(-1,1)];self.applied_control_history=np.tile(controls,(20,1))
        self.decision_steps+=1;self.completed_episodes+=1;self.silent_episodes+=int(comp['generated_silent']);self.nonvoiced_episodes+=int(not voiced);self.done=True
        info=dict(acoustic_error=float(e),acoustic_components=comp,weighted_loss=float(j),failed=failed,sustained_voiced=voiced,
            generated_silent=comp['generated_silent'],applied_action=self.actions[0].tolist(),applied_controls=controls.tolist(),model_condition='B',
            late_ac_rms_over_target_peak=float(np.std(audio[-1000:])/self.loss_scale),motion_mean=self.motion_means[-1].tolist() if self.motion_means else None,
            motion_std=self.motion_stds[-1].tolist() if self.motion_stds else None,source_end=self.source_terms[-1] if self.source_terms else None)
        if self.best_candidate is None or j<self.best_candidate['weighted_loss']:
            self.best_candidate=dict(trial=self.completed_episodes,**info)
            self.best_values=dict(audio=audio.copy(),physical_controls=self.applied_control_history.copy(),actions=np.asarray(self.actions).copy(),
                state_end_blocks=np.asarray(self.states).copy(),motion_mean=np.asarray(self.motion_means).copy(),motion_std=np.asarray(self.motion_stds).copy(),source_end_blocks=np.asarray(self.source_terms).copy())
        obs=np.zeros(8,np.float32);obs[0]=1;return obs,-float(j),True,False,info

    def close(self):self.model.close()
