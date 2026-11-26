"""Change only the f0_l action-to-physical mapping; keep the original reward."""
import numpy as np
from .constant020_env import Constant020Env
from .model import LOW,HIGH


class DomainEnv(Constant020Env):
    def __init__(self,indices,target,controls,*,envelope_weight=1.,f0_high):
        if not LOW[3]<f0_high<=HIGH[3]:raise ValueError('Invalid restricted upper bound')
        super().__init__(indices,target,controls,envelope_weight=envelope_weight)
        self.domain_low=LOW.copy();self.domain_high=HIGH.copy();self.domain_high[3]=f0_high

    def encode(self,controls,dtype=np.float64):
        c=np.asarray(controls,dtype=np.float64)
        return (2*(c[self.indices]-self.domain_low[self.indices])/(self.domain_high[self.indices]-self.domain_low[self.indices])-1).astype(dtype)

    def decode(self,action):
        a=np.asarray(action,dtype=np.float64)
        if a.shape!=(len(self.indices),) or not np.isfinite(a).all():raise ValueError('Invalid action')
        c=self.fixed_controls.copy()
        c[self.indices]=self.domain_low[self.indices]+(np.clip(a,-1,1)+1)/2*(self.domain_high[self.indices]-self.domain_low[self.indices])
        return c
