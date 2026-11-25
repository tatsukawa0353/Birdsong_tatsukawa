"""Off-policy SAC behavior mixture with exact executed-action Replay recording."""
import hashlib
import pickle
import random
import numpy as np
from stable_baselines3 import SAC

from .issue14 import learner_signature,_state_digest
from .issue26_data import LOW,HIGH,RADII,PREFIX,LOCAL_PROBABILITY


def learner_rng():
    import torch
    return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=4)+torch.get_rng_state().numpy().tobytes()).hexdigest()


def observer_signature(agent):
    return dict(learner=learner_signature(agent),choice_rng=_state_digest(agent.choice_rng.bit_generator.state),
        candidate_rng=_state_digest(agent.candidate_rng.bit_generator.state),incumbent=_state_digest(agent.incumbent),
        candidates=len(agent.candidate_log))


def continuous_local(env,policy,center,radius,rng):
    lower=np.maximum(LOW,np.asarray(center)-radius);upper=np.minimum(HIGH,np.asarray(center)+radius)
    assert np.all(upper>lower)
    for attempt in range(1,17):
        requested=rng.uniform(lower,upper)
        controls=env.fixed_controls.copy();controls[[0,2,3]]=requested
        normalized=env.encode(controls,dtype=np.float32).reshape(1,3)
        buffer_action=policy.scale_action(normalized).astype(np.float32)
        action=policy.unscale_action(buffer_action).astype(np.float32)
        actual=env.decode(action[0])[[0,2,3]]
        # Rare float32 edge rounding is resampled before simulation, never clipped.
        if np.all(actual>=lower) and np.all(actual<=upper):
            return action,buffer_action,dict(requested_controls=requested.tolist(),local_lower=lower.tolist(),local_upper=upper.tolist(),
                local_deviation=(actual-center).tolist(),rounding_error=(actual-requested).tolist(),draw_attempts=attempt)
    raise FloatingPointError('Float32 local candidate mapping repeatedly left its sampled interval')


class BehaviorSAC(SAC):
    def __init__(self,*args,condition,mapping_env,local_start=PREFIX,**kwargs):
        self.condition=condition;self.mapping_env=mapping_env;self.local_start=local_start
        self.incumbent=None;self.candidate_log=[];self.last_proposal=None;self.transition_checks=0
        super().__init__(*args,**kwargs)
        self.choice_rng=np.random.default_rng(260026+int(self.seed))
        self.candidate_rng=np.random.default_rng(260126+int(self.seed))

    def _sample_action(self,learning_starts,action_noise=None,n_envs=1):
        assert n_envs==1 and action_noise is None
        action,buffer_action=super()._sample_action(learning_starts,action_noise,n_envs)
        rng_before_local=learner_rng()
        proposal=dict(source='prefix_warmup' if self.num_timesteps<learning_starts else 'prefix_sac' if self.num_timesteps<self.local_start else 'sac',
            stock_action=action[0].tolist(),ref_before=None if self.incumbent is None else dict(self.incumbent),local_draw=None)
        if self.condition in RADII and self.num_timesteps>=self.local_start:
            draw=float(self.choice_rng.random());proposal['local_draw']=draw
            if draw<LOCAL_PROBABILITY:
                assert self.incumbent is not None
                center=np.asarray(self.incumbent['physical_controls'])[[0,2,3]]
                action,buffer_action,local=continuous_local(self.mapping_env,self.policy,center,RADII[self.condition],self.candidate_rng)
                proposal.update(source='local',**local)
        proposal.update(env_action=action[0].tolist(),replay_action=buffer_action[0].tolist())
        assert learner_rng()==rng_before_local,'Local sampler consumed learner RNG'
        proposal['local_sampling_preserved_learner_rng']=True
        self.last_proposal=proposal
        return action,buffer_action

    def _store_transition(self,replay_buffer,buffer_action,new_obs,reward,dones,infos):
        info=infos[0];proposal=self.last_proposal;assert proposal is not None
        np.testing.assert_array_equal(buffer_action[0],proposal['replay_action'])
        np.testing.assert_array_equal(info['applied_action'],proposal['env_action'])
        np.testing.assert_array_equal(self.policy.scale_action(np.asarray(proposal['env_action'],np.float32)),buffer_action[0])
        np.testing.assert_array_equal(self.mapping_env.decode(info['applied_action']),info['applied_controls'])
        pos=replay_buffer.pos
        super()._store_transition(replay_buffer,buffer_action,new_obs,reward,dones,infos)
        np.testing.assert_array_equal(replay_buffer.actions[pos,0],buffer_action[0])
        assert float(replay_buffer.rewards[pos,0])==float(reward[0])==float(np.float32(-info['weighted_loss']))
        improved=self.incumbent is None or info['weighted_loss']<self.incumbent['J']
        if improved:self.incumbent=dict(trial=self.num_timesteps,J=info['weighted_loss'],E_total=info['acoustic_error'],
            physical_controls=info['applied_controls'],action=info['applied_action'],sustained_voiced=info['sustained_voiced'],failed=info['failed'])
        self.transition_checks+=1
        self.candidate_log.append(dict(episodes=self.num_timesteps,acoustic_error=info['acoustic_error'],acoustic_components=info['acoustic_components'],
            weighted_loss=info['weighted_loss'],failed=info['failed'],sustained_voiced=info['sustained_voiced'],action=info['applied_action'],
            physical_controls=info['applied_controls'],reward=float(reward[0]),alpha_before_update=float(self.log_ent_coef.detach().exp()),
            proposal=proposal,improved=improved,ref_after=dict(self.incumbent),replay_pos=pos,stored_action=replay_buffer.actions[pos,0].tolist()))


def make_agent(condition,seed,env,config):
    return BehaviorSAC('MlpPolicy',env=env,seed=seed,device='cpu',verbose=0,policy_kwargs=dict(net_arch=[128,128]),
        learning_rate=config['learning_rate'],gamma=config['gamma'],batch_size=config['batch_size'],buffer_size=50000,
        learning_starts=config['warmup_episodes'],train_freq=1,gradient_steps=1,ent_coef=config['sac_ent_coef'],
        condition=condition,mapping_env=env,local_start=config.get('local_start',PREFIX))
