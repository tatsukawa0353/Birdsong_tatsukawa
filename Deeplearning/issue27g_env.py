"""Independent constant3D Case B environment tied to the newly frozen Target."""
import json
import gymnasium as gym
import numpy as np
from .model import LOW,HIGH
from .model_env import AudioDecimator
from .issue18_domain_env import DomainEnv
from .issue27g_data import OUT,target_path,target_data
from .issue27g_native import ConfiguredCaseB,snapshot
from .issue16 import digest
from .acoustics import distance

class CaseBEnv(gym.Env):
    metadata={'render_modes':[]};horizon=1;physical_horizon=20
    encode=DomainEnv.encode;decode=DomainEnv.decode
    def __init__(self,target_name,dim,record_moments=False):
        gate=json.loads((OUT/'preflight/gate.json').read_text());assert gate[target_name]['passed'] and gate[target_name]['target_sha256']==digest(target_path(target_name))
        self.target_name=target_name;self.dim=dim;assert dim in (3,4)
        target,controls=target_data(target_name);assert target.shape==(10000,) and np.isfinite(target).all()
        self.target_audio=target;self.fixed_controls=controls;self.target_audio.setflags(write=False);self.fixed_controls.setflags(write=False)
        self.indices=[0,2,3] if dim==3 else [0,1,2,3];self.domain_low=LOW.copy();self.domain_high=HIGH.copy();self.domain_low[1]=2e7;self.domain_high[3]=800000.
        self.loss_scale=max(float(abs(target).max()),1e-8);self.record_moments=record_moments
        self.action_space=gym.spaces.Box(-1,1,(dim,),np.float32);self.observation_space=gym.spaces.Box(-1,1,(8,),np.float32)
        self.model=ConfiguredCaseB(float(controls[4]));self.model.reset(controls);self.decision_steps=self.completed_episodes=self.failures=0
        self.native_blocks=self.native_blocks_completed=self.silent_episodes=self.nonvoiced_episodes=0
        self.integrated_s=self.integrated_completed_s=0.;self.done=True;self.native_checks=0
        self.best_candidate=None;self.best_values=None;self.last_info=None;self.last_values=None
        self.check()
    def check(self,reset=False):
        s=snapshot(self.model,reset=reset);self.native_checks+=1
        assert s['epsilon_r']==self.model.controls[1] and s['f0_r']==self.fixed_controls[4]
        if self.dim==3:assert s['epsilon_r']==self.fixed_controls[1]
        assert self.model.condition=='B'
        return s
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.action_space.seed(seed)
        self.model.reset(self.fixed_controls);self.check(reset=True);self.decimator=AudioDecimator()
        self.audio=[];self.actions=[];self.applied_control_history=[];self.states=[];self.source_terms=[];self.means=[];self.stds=[]
        self.done=False;obs=np.zeros(8,np.float32);obs[0]=-1;return obs,{}
    def step(self,action):
        if self.done:raise RuntimeError('Reset for every constant candidate')
        a=np.asarray(action,np.float64);assert a.shape==(self.dim,) and np.isfinite(a).all() and np.max(abs(a))<=1
        c=self.decode(a);self.model.set_controls(c);self.check();failed=False
        for _ in range(20):
            self.native_blocks+=1;self.integrated_s+=.01
            try:
                if self.record_moments:
                    raw,m=self.model.advance(.01,return_moments=True);self.means.append(m['mean']);self.stds.append(m['std'])
                else:raw=self.model.advance(.01)
                audio=self.decimator.process(raw)
                if not np.isfinite(audio).all():raise FloatingPointError('Nonfinite filtered output')
            except FloatingPointError:failed=True;self.failures+=1;break
            self.native_blocks_completed+=1;self.integrated_completed_s+=.01;self.audio.append(audio)
            s=self.check();self.states.append(s['state']);self.source_terms.append([s['source_l'],s['source_r']])
        if failed:self.audio.extend([np.zeros(500)]*(20-len(self.audio)))
        audio=np.concatenate(self.audio);comp=distance(self.target_audio,audio,50000,scale=self.loss_scale)
        j=1e6 if failed else .75*comp['log_spectral']+.25*comp['envelope'];e=1e6 if failed else comp['total']
        voiced=bool(np.std(audio[-1000:])/self.loss_scale>.05) and not failed
        self.actions=[a.copy()];self.applied_control_history=np.tile(c,(20,1));self.decision_steps+=1;self.completed_episodes+=1;self.done=True
        self.silent_episodes+=int(comp['generated_silent']);self.nonvoiced_episodes+=int(not voiced)
        info=dict(acoustic_error=float(e),acoustic_components=comp,weighted_loss=float(j),failed=failed,sustained_voiced=voiced,
            generated_silent=comp['generated_silent'],applied_action=a.tolist(),applied_controls=c.tolist(),model_condition='B',
            late_ac_rms_over_target_peak=float(np.std(audio[-1000:])/self.loss_scale))
        values=dict(audio=audio.copy(),actions=np.asarray(self.actions).copy(),physical_controls=self.applied_control_history.copy(),
            state_end_blocks=np.asarray(self.states).copy(),source_end_blocks=np.asarray(self.source_terms).copy())
        if self.record_moments:values.update(motion_mean=np.asarray(self.means).copy(),motion_std=np.asarray(self.stds).copy())
        self.last_info=info;self.last_values=values
        if self.best_candidate is None or j<self.best_candidate['weighted_loss']:
            self.best_candidate=dict(episodes=self.completed_episodes,**info);self.best_values={k:v.copy() for k,v in values.items()}
        obs=np.zeros(8,np.float32);obs[0]=1;return obs,-float(j),True,False,info
    def close(self):self.model.close()

def episode(env,*,agent=None,action=None):
    obs,_=env.reset()
    if agent is not None:action=agent.predict(obs,deterministic=True)[0]
    _,r,done,truncated,info=env.step(action);assert done and not truncated and r==-info['weighted_loss']
    return info,{k:v.copy() for k,v in env.last_values.items()}
