"""Preserve SB3 SAC.train; override only the Actor optimizer's constant LR."""
import json

from stable_baselines3 import SAC

from .issue14 import effective_hyperparameters

BASE_LR=3e-4
ACTOR_RATES=(1e-4,3e-5)


def optimizer_lrs(agent):
    return {name:[float(group['lr']) for group in optimizer.param_groups] for name,optimizer in
        (('actor',agent.actor.optimizer),('critic',agent.critic.optimizer),('entropy',agent.ent_coef_optimizer))}


def assert_lrs(agent,actor_rate):
    actual=optimizer_lrs(agent)
    assert actual==dict(actor=[actor_rate],critic=[BASE_LR],entropy=[BASE_LR]),actual
    return actual


class ActorRateSAC(SAC):
    def __init__(self,*args,actor_learning_rate=BASE_LR,**kwargs):
        if actor_learning_rate not in (BASE_LR,*ACTOR_RATES):raise ValueError('Unregistered Actor LR')
        self.actor_learning_rate=float(actor_learning_rate)
        super().__init__(*args,**kwargs)
        if self.learning_rate!=BASE_LR:raise ValueError('Shared/Critic/temperature LR must stay3e-4')
        self._set_actor_rate()

    def _set_actor_rate(self):
        for group in self.actor.optimizer.param_groups:group['lr']=self.actor_learning_rate

    def _update_learning_rate(self,optimizers):
        # Keep SB3's schedule/logging and all non-Actor updates unchanged.
        super()._update_learning_rate(optimizers)
        assert self.lr_schedule(self._current_progress_remaining)==BASE_LR
        self._set_actor_rate()
        assert_lrs(self,self.actor_learning_rate)


def make_agent(actor_rate,seed,env,config):
    if config['learning_rate']!=BASE_LR:raise ValueError('Changing shared learning_rate is forbidden')
    return ActorRateSAC('MlpPolicy',env=env,seed=seed,device='cpu',verbose=0,
        policy_kwargs=dict(net_arch=[128,128]),learning_rate=BASE_LR,
        gamma=config['gamma'],batch_size=config['batch_size'],buffer_size=50000,
        learning_starts=config['warmup_episodes'],train_freq=1,gradient_steps=1,
        ent_coef=config['sac_ent_coef'],actor_learning_rate=actor_rate)


def effective_difference(agent,baseline,actor_rate):
    actual=json.loads(json.dumps(effective_hyperparameters(agent)))
    normalized=json.loads(json.dumps(actual))
    # Everything, including shared LR schedule and other optimizer options,
    # must equal the saved baseline after removing the ONE intended difference.
    assert actual['optimizer_param_groups']['actor'][0]['lr']==actor_rate
    normalized['optimizer_param_groups']['actor'][0]['lr']=BASE_LR
    assert normalized==baseline,'A non-Actor-LR effective setting changed'
    assert_lrs(agent,actor_rate)
    return actual
