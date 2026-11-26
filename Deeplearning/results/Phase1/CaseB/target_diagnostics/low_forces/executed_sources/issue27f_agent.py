"""Stock SAC/TD3/PPO updates with observation-only transition/LR diagnostics."""
import numpy as np
import torch
from stable_baselines3 import SAC,TD3,PPO
from stable_baselines3.common.noise import NormalActionNoise
from .issue14 import learner_signature,_state_digest
from .issue22_data import policy_statistics as sac_statistics

def row(info,n):
    return dict(episodes=n,acoustic_error=info['acoustic_error'],acoustic_components=info['acoustic_components'],weighted_loss=info['weighted_loss'],
        failed=info['failed'],sustained_voiced=info['sustained_voiced'],generated_silent=info['generated_silent'],
        action=info['applied_action'],physical_controls=info['applied_controls'],reward=float(np.float32(-info['weighted_loss'])))

class StoreAudit:
    def __init__(self,*args,mapping_env,**kwargs):
        self.mapping_env=mapping_env;self.transition_rows=[];self.transition_checks=0;super().__init__(*args,**kwargs)
    def _store_transition(self,replay_buffer,buffer_action,new_obs,reward,dones,infos):
        pos=replay_buffer.pos;p=infos[0];super()._store_transition(replay_buffer,buffer_action,new_obs,reward,dones,infos)
        stored=replay_buffer.actions[pos,0];executed=np.asarray(p['applied_action'],np.float32)
        np.testing.assert_array_equal(stored,buffer_action[0])
        np.testing.assert_array_equal(self.policy.unscale_action(stored),executed)
        # Stock float32 scaling is not bitwise invertible. Observe its forward
        # mapping exactly; never rewrite an action or Replay entry for an audit.
        inverse=self.policy.scale_action(executed)
        np.testing.assert_allclose(inverse,stored,rtol=0,atol=2*np.finfo(np.float32).eps)
        np.testing.assert_array_equal(self.mapping_env.decode(executed),p['applied_controls'])
        assert float(replay_buffer.rewards[pos,0])==float(reward[0])==float(np.float32(-p['weighted_loss']))
        assert bool(dones[0]) and replay_buffer.dones[pos,0]==1 and replay_buffer.timeouts[pos,0]==0
        r=row(p,self.num_timesteps);r.update(buffer_pos=pos,stored_action=stored.tolist(),stored_reward=float(replay_buffer.rewards[pos,0]),
            buffer_kind='Replay',inverse_roundtrip_error=float(np.max(abs(inverse-stored))),
            alpha_before_update=float(self.log_ent_coef.detach().exp()) if hasattr(self,'log_ent_coef') else None)
        self.transition_rows.append(r);self.transition_checks+=1

class ObservedSAC(StoreAudit,SAC):pass
class ObservedTD3(StoreAudit,TD3):pass

class ObservedPPO(PPO):
    def __init__(self,*args,mapping_env,**kwargs):
        self.mapping_env=mapping_env;self.transition_rows=[];self.transition_checks=0;super().__init__(*args,**kwargs)
        self._original_buffer_add=self.rollout_buffer.add;self.rollout_buffer.add=self._audited_add
    def _audited_add(self,obs,action,reward,episode_start,value,log_prob):
        pos=self.rollout_buffer.pos;p=self.mapping_env.last_info;assert p is not None
        np.testing.assert_array_equal(np.clip(action[0],-1,1),p['applied_action'])
        np.testing.assert_array_equal(self.mapping_env.decode(p['applied_action']),p['applied_controls'])
        self._original_buffer_add(obs,action,reward,episode_start,value,log_prob)
        stored=self.rollout_buffer.actions[pos,0];np.testing.assert_array_equal(stored,action[0])
        assert float(self.rollout_buffer.rewards[pos,0])==float(reward[0])==float(np.float32(-p['weighted_loss']))
        assert self.rollout_buffer.episode_starts[pos,0]==1
        r=row(p,self.num_timesteps);r.update(buffer_pos=pos,stored_action=stored.tolist(),stored_reward=float(self.rollout_buffer.rewards[pos,0]),
            buffer_kind='Rollout_unclipped_Gaussian',sampled_action=action[0].tolist(),sampled_log_prob=float(log_prob.detach().cpu().numpy()[0]),
            action_was_clipped=bool(np.any(abs(action[0])>1)))
        self.transition_rows.append(r);self.transition_checks+=1

def make_agent(method,seed,env,config,audited=True):
    if method not in ('SAC','TD3','PPO'):raise ValueError('Unregistered algorithm')
    kwargs=dict(policy='MlpPolicy',env=env,seed=seed,device='cpu',verbose=0,learning_rate=3e-4,gamma=1.,batch_size=32)
    if audited:kwargs['mapping_env']=env
    if method=='PPO':
        kwargs.update(policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])),n_steps=128,n_epochs=10,clip_range=.2)
        return (ObservedPPO if audited else PPO)(**kwargs)
    kwargs.update(policy_kwargs=dict(net_arch=[128,128]),buffer_size=50000,learning_starts=16,train_freq=1,gradient_steps=1)
    if method=='SAC':return (ObservedSAC if audited else SAC)(**kwargs,ent_coef='auto_0.01',target_entropy=-3)
    return (ObservedTD3 if audited else TD3)(**kwargs,action_noise=NormalActionNoise(np.zeros(3),.1*np.ones(3)))

def optimizers(agent,method):
    if method=='PPO':return dict(policy_value=agent.policy.optimizer)
    d=dict(actor=agent.actor.optimizer,critic=agent.critic.optimizer)
    if method=='SAC':d['entropy']=agent.ent_coef_optimizer
    return d
def lrs(agent,method):
    actual={k:[float(g['lr']) for g in o.param_groups] for k,o in optimizers(agent,method).items()}
    assert all(all(x==3e-4 for x in r) for r in actual.values()),actual
    return actual
def geometry(agent,method):
    linear=lambda m:[[x.in_features,x.out_features] for x in m.modules() if isinstance(x,torch.nn.Linear)]
    count=lambda m:sum(p.numel() for p in m.parameters())
    if method=='PPO':
        pi=linear(agent.policy.mlp_extractor.policy_net);vf=linear(agent.policy.mlp_extractor.value_net)
        assert pi==vf==[[8,128],[128,128]] and linear(agent.policy.action_net)==[[128,3]] and linear(agent.policy.value_net)==[[128,1]]
        actor=count(agent.policy.mlp_extractor.policy_net)+count(agent.policy.action_net)+agent.policy.log_std.numel()
        critic=count(agent.policy.mlp_extractor.value_net)+count(agent.policy.value_net)
        assert (actor,critic,count(agent.policy))==(18054,17793,35847)
        return dict(actor_layers=pi+[[128,3]],value_layers=vf+[[128,1]],actor_parameters=actor,value_parameters=critic,total_policy_parameters=count(agent.policy),
            activation='Tanh',log_std_parameters=3,optimizer_parameters=35847)
    a=linear(agent.actor.latent_pi) if method=='SAC' else linear(agent.actor.mu)
    assert a==([[8,128],[128,128]] if method=='SAC' else [[8,128],[128,128],[128,3]])
    q=[linear(m) for m in agent.critic.q_networks];assert q==[[[11,128],[128,128],[128,1]]]*2
    assert [linear(m) for m in agent.critic_target.q_networks]==q
    ac=count(agent.actor);qc=count(agent.critic);total=count(agent.policy)
    assert (ac,qc,total)==((18438,36354,91146) if method=='SAC' else (18051,36354,108810)),(ac,qc,total)
    return dict(actor_layers=a,critic_layers=q,actor_parameters=ac,critic_parameters=qc,total_policy_parameters=total,
        critic_target_parameters=count(agent.critic_target),actor_target_parameters=count(agent.actor_target) if method=='TD3' else 0,activation='ReLU',
        entropy_parameter_count=1 if method=='SAC' else 0)

def effective(agent,method):
    fields=('learning_rate','gamma','batch_size','buffer_size','learning_starts','gradient_steps','tau','policy_delay','target_policy_noise','target_noise_clip',
        'ent_coef','target_entropy','n_steps','n_epochs','gae_lambda','ent_coef','vf_coef','max_grad_norm','normalize_advantage','target_kl','use_sde')
    result={key:getattr(agent,key) for key in fields if hasattr(agent,key)}
    if method=='PPO':result.update(clip_range=float(agent.clip_range(1.)),clip_range_vf=agent.clip_range_vf)
    else:result['train_freq']=dict(frequency=agent.train_freq.frequency,unit=agent.train_freq.unit.value)
    result.update(device=str(agent.device),net_arch=agent.policy.net_arch,optimizer_groups={k:[{n:v for n,v in g.items() if n!='params'} for g in opt.param_groups] for k,opt in optimizers(agent,method).items()})
    assert result['gamma']==1 and result['learning_rate']==3e-4 and result['batch_size']==32 and result['device']=='cpu'
    if method=='PPO':assert result['n_steps']==128 and result['n_epochs']==10 and result['clip_range']==.2
    else:assert result['learning_starts']==16 and result['buffer_size']==50000 and result['gradient_steps']==1 and result['train_freq']['frequency']==1
    return result

def signature(agent):
    return dict(learner=learner_signature(agent),training_best=_state_digest(getattr(agent.get_env().envs[0].unwrapped,'best_candidate',None)),
        transition_rows=len(getattr(agent,'transition_rows',[])),transition_checks=getattr(agent,'transition_checks',0))

def statistics(agent,method):
    if method=='SAC':return dict(method=method,**sac_statistics(agent))
    obs=torch.zeros((1,8));obs[:,0]=-1;z=np.random.default_rng(220026).standard_normal((4096,3))
    with torch.no_grad():
        if method=='PPO':
            distribution=agent.policy.get_distribution(obs).distribution
            mean=distribution.mean[0].numpy();std=distribution.stddev[0].numpy();entropy=float(distribution.entropy().sum())
        else:mean=agent.actor(obs)[0].numpy();std=np.full(3,.1);entropy=None
    sampled=mean+std*z;executed=np.clip(sampled,-1,1)
    action=agent.predict(np.array([-1,0,0,0,0,0,0,0],np.float32),deterministic=True)[0]
    return dict(method=method,latent_mean=mean.tolist(),latent_std=std.tolist(),normalized_action_std_mc=executed.std(0,ddof=1).tolist(),
        deterministic_action=action.tolist(),clip_fraction=float(np.mean(np.any(abs(sampled)>1,axis=1))),gaussian_entropy=entropy,
        distribution='Gaussian clipped at environment' if method=='PPO' else 'Deterministic Actor plus fixed Normal noise .1',entropy_samples=4096,entropy_seed=220026,alpha=None)
