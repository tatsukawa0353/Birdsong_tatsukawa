"""Original SB3 updates; action-size-aware construction and read-only audits."""
import numpy as np
import torch
from stable_baselines3 import SAC,TD3,PPO
from stable_baselines3.common.noise import NormalActionNoise
from .issue27f_agent import ObservedSAC,ObservedTD3,ObservedPPO,optimizers,lrs,effective,signature

def make_agent(method,seed,env,config,audited=True):
    n=env.action_space.shape[0];assert n in (3,4)
    kwargs=dict(policy='MlpPolicy',env=env,seed=seed,device='cpu',verbose=0,learning_rate=3e-4,gamma=1.,batch_size=32)
    if audited:kwargs['mapping_env']=env
    if method=='PPO':
        kwargs.update(policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])),n_steps=128,n_epochs=10,clip_range=.2)
        return (ObservedPPO if audited else PPO)(**kwargs)
    kwargs.update(policy_kwargs=dict(net_arch=[128,128]),buffer_size=50000,learning_starts=16,train_freq=1,gradient_steps=1)
    if method=='SAC':return (ObservedSAC if audited else SAC)(**kwargs,ent_coef='auto_0.01',target_entropy=-n)
    if method=='TD3':return (ObservedTD3 if audited else TD3)(**kwargs,action_noise=NormalActionNoise(np.zeros(n),.1*np.ones(n)))
    raise ValueError(method)

def geometry(agent,method):
    n=agent.action_space.shape[0];linear=lambda m:[[x.in_features,x.out_features] for x in m.modules() if isinstance(x,torch.nn.Linear)]
    count=lambda m:sum(p.numel() for p in m.parameters());base=17664
    if method=='PPO':
        pi=linear(agent.policy.mlp_extractor.policy_net);vf=linear(agent.policy.mlp_extractor.value_net)
        assert pi==vf==[[8,128],[128,128]] and linear(agent.policy.action_net)==[[128,n]] and linear(agent.policy.value_net)==[[128,1]]
        ac=count(agent.policy.mlp_extractor.policy_net)+count(agent.policy.action_net)+agent.policy.log_std.numel();vc=count(agent.policy.mlp_extractor.value_net)+count(agent.policy.value_net)
        assert ac==base+n*130 and vc==17793 and count(agent.policy)==ac+vc
        return dict(actor_layers=pi+[[128,n]],value_layers=vf+[[128,1]],actor_parameters=ac,value_parameters=vc,total_policy_parameters=count(agent.policy),activation='Tanh',log_std_parameters=n)
    a=linear(agent.actor.latent_pi) if method=='SAC' else linear(agent.actor.mu)
    assert a==([[8,128],[128,128]] if method=='SAC' else [[8,128],[128,128],[128,n]])
    q=[linear(m) for m in agent.critic.q_networks];assert q==[[[8+n,128],[128,128],[128,1]]]*2
    assert [linear(m) for m in agent.critic_target.q_networks]==q
    ac=count(agent.actor);qc=count(agent.critic);expected_actor=base+n*129*(2 if method=='SAC' else 1);expected_q=2*((8+n)*128+128+16512+129)
    assert ac==expected_actor and qc==expected_q and count(agent.policy)==ac*(2 if method=='TD3' else 1)+qc*2
    if method=='SAC':assert agent.target_entropy==-n
    return dict(actor_layers=a,critic_layers=q,actor_parameters=ac,critic_parameters=qc,total_policy_parameters=count(agent.policy),activation='ReLU',
        critic_target_parameters=count(agent.critic_target),actor_target_parameters=count(agent.actor_target) if method=='TD3' else 0,entropy_parameter_count=1 if method=='SAC' else 0)

def statistics(agent,method):
    n=agent.action_space.shape[0];obs=torch.zeros((1,8));obs[:,0]=-1
    z=np.random.default_rng(220026).standard_normal((4096,n))
    with torch.no_grad():
        if method=='SAC':
            mean,log_std,_=agent.actor.get_action_dist_params(obs);std=log_std.exp();latent=mean+std*torch.as_tensor(z.astype(np.float32));actions=torch.tanh(latent)
            logp=torch.distributions.Normal(mean,std).log_prob(latent).sum(1)-torch.log(1-actions**2+agent.actor.action_dist.epsilon).sum(1)
            return dict(method=method,alpha=float(agent.log_ent_coef.exp()),latent_mean=mean[0].tolist(),latent_log_std=log_std[0].tolist(),latent_std=std[0].tolist(),
                normalized_action_std_mc=actions.std(0,unbiased=True).tolist(),squashed_entropy_mc=float(-logp.mean()),entropy_mc_se=float(logp.std(unbiased=True)/np.sqrt(4096)),
                gaussian_entropy=float((log_std+.5*np.log(2*np.pi*np.e)).sum()),deterministic_action=torch.tanh(mean)[0].tolist(),entropy_samples=4096,entropy_seed=220026)
        if method=='PPO':
            distribution=agent.policy.get_distribution(obs).distribution;mean=distribution.mean[0].numpy();std=distribution.stddev[0].numpy();entropy=float(distribution.entropy().sum())
        else:mean=agent.actor(obs)[0].numpy();std=np.full(n,.1);entropy=None
    sampled=mean+std*z;executed=np.clip(sampled,-1,1);action=agent.predict(obs.numpy()[0],deterministic=True)[0]
    return dict(method=method,latent_mean=mean.tolist(),latent_std=std.tolist(),normalized_action_std_mc=executed.std(0,ddof=1).tolist(),deterministic_action=action.tolist(),
        clip_fraction=float(np.mean(np.any(abs(sampled)>1,axis=1))),gaussian_entropy=entropy,alpha=None,entropy_samples=4096,entropy_seed=220026)
