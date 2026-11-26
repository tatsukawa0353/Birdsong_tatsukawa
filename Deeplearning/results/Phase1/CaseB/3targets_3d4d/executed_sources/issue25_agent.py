"""Stock SAC updates, changing only Critic LR after SB3's common schedule."""
import json

from stable_baselines3 import SAC

from .issue14 import effective_hyperparameters
from .issue24_agent import optimizer_lrs

BASE_LR=3e-4
CRITIC_RATES=(1e-4,1e-3)
LIMITS=dict(gradient_l2=1e6,parameter_abs=1e6,optimizer_moment_abs=1e12)


def assert_lrs(agent,rate):
    actual=optimizer_lrs(agent)
    assert actual==dict(actor=[BASE_LR],critic=[rate],entropy=[BASE_LR]),actual
    return actual


class CriticRateSAC(SAC):
    def __init__(self,*args,critic_learning_rate=BASE_LR,**kwargs):
        if critic_learning_rate not in (BASE_LR,*CRITIC_RATES):raise ValueError('Unregistered Critic LR')
        self.critic_learning_rate=float(critic_learning_rate);super().__init__(*args,**kwargs)
        if self.learning_rate!=BASE_LR:raise ValueError('The shared schedule must remain3e-4')
        self._set_critic_rate()

    def _set_critic_rate(self):
        for group in self.critic.optimizer.param_groups:group['lr']=self.critic_learning_rate

    def _update_learning_rate(self,optimizers):
        super()._update_learning_rate(optimizers)
        assert self.lr_schedule(self._current_progress_remaining)==BASE_LR
        self._set_critic_rate();assert_lrs(self,self.critic_learning_rate)


def make_agent(rate,seed,env,config):
    if config['learning_rate']!=BASE_LR:raise ValueError('Changing the shared learning_rate is forbidden')
    return CriticRateSAC('MlpPolicy',env=env,seed=seed,device='cpu',verbose=0,
        policy_kwargs=dict(net_arch=[128,128]),learning_rate=BASE_LR,gamma=config['gamma'],
        batch_size=config['batch_size'],buffer_size=50000,learning_starts=config['warmup_episodes'],
        train_freq=1,gradient_steps=1,ent_coef=config['sac_ent_coef'],critic_learning_rate=rate)


def effective_difference(agent,baseline,rate):
    actual=json.loads(json.dumps(effective_hyperparameters(agent)));normalized=json.loads(json.dumps(actual))
    assert actual['optimizer_param_groups']['critic'][0]['lr']==rate
    normalized['optimizer_param_groups']['critic'][0]['lr']=BASE_LR
    assert normalized==baseline,'A non-Critic-LR setting changed';assert_lrs(agent,rate)
    return actual


def numerical_check(optimizer,gradients=False):
    """Observe, never clip. Same predeclared emergency stops for every LR."""
    import torch
    norm2=0.;parameter_abs=moment_abs=0.
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if not torch.isfinite(parameter).all():raise FloatingPointError('Nonfinite parameter')
            parameter_abs=max(parameter_abs,float(parameter.detach().abs().max()))
            if gradients and parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():raise FloatingPointError('Nonfinite gradient')
                norm2+=float(parameter.grad.detach().norm())**2
    for state in optimizer.state.values():
        for name,value in state.items():
            if isinstance(value,torch.Tensor):
                if not torch.isfinite(value).all():raise FloatingPointError(f'Nonfinite optimizer state:{name}')
                if name in ('exp_avg_sq','max_exp_avg_sq') and bool((value<0).any()):raise FloatingPointError('Negative second moment')
                if name!='step':moment_abs=max(moment_abs,float(value.abs().max()))
    result=dict(gradient_l2=norm2**.5,parameter_abs=parameter_abs,optimizer_moment_abs=moment_abs)
    for key,value in result.items():
        if value>LIMITS[key]:raise FloatingPointError(f'Emergency numerical stop:{key}={value} > {LIMITS[key]}')
    return result
