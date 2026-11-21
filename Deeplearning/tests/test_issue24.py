"""Actor-only scheduling, unchanged baseline updates and retained inference policies."""
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger

from Deeplearning.issue14 import learner_signature
from Deeplearning.issue15_analysis import write_json
from Deeplearning.issue16 import digest
from Deeplearning.issue21_training import make_agent as stock_agent
from Deeplearning.issue24_agent import (ActorRateSAC,make_agent,assert_lrs,effective_difference,BASE_LR,ACTOR_RATES)
from Deeplearning.issue24_data import (BASE,TARGET,source_hashes,environment,folder_for,curve_for,load_checkpoint)
from Deeplearning.issue24_training import run_one
from Deeplearning.model import LIBRARY
from Deeplearning.constant020_env import episode


class SyntheticTerminal(gym.Env):
    """No syrinx calls: same observation/action shape for testing optimizer steps."""
    def __init__(self):
        self.observation_space=gym.spaces.Box(-1,1,(8,),np.float32);self.action_space=gym.spaces.Box(-1,1,(3,),np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);return np.array([-1,0,0,0,0,0,0,0],np.float32),{}
    def step(self,action):
        return np.zeros(8,np.float32),-float(np.square(action).sum()),True,False,{}


class ActorRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.config=json.loads((BASE/'manifest.json').read_text())['settings']

    def test_stock_train_method_and_schedule_do_not_reset_actor_rate(self):
        self.assertIs(ActorRateSAC.train,SAC.train)
        for rate in ACTOR_RATES:
            env=SyntheticTerminal()
            try:
                a=make_agent(rate,0,env,self.config);a.set_logger(Logger(None,[]))
                for progress in (1.,.8,.1,0.):
                    a._current_progress_remaining=progress
                    a._update_learning_rate([a.actor.optimizer,a.critic.optimizer,a.ent_coef_optimizer]);assert_lrs(a,rate)
                steps=[]
                hook=a.actor.optimizer.register_step_pre_hook(lambda opt,args,kwargs:steps.append(assert_lrs(a,rate)))
                try:a.learn(32)
                finally:hook.remove()
                self.assertEqual(len(steps),16);assert_lrs(a,rate)
            finally:env.close()

    def test_baseline_rate_is_bitwise_equal_to_stock_after_32_steps(self):
        signatures=[]
        for factory in (lambda e:stock_agent('SAC',128,2,e,self.config),lambda e:make_agent(BASE_LR,2,e,self.config)):
            env=SyntheticTerminal()
            try:
                a=factory(env);a.set_logger(Logger(None,[]));a.learn(32);signatures.append(learner_signature(a))
            finally:env.close()
        self.assertEqual(signatures[0],signatures[1])

    def test_first_update_changes_only_actor_weights_not_critic_or_temperature(self):
        records=[]
        for rate in (BASE_LR,*ACTOR_RATES):
            env=SyntheticTerminal()
            try:
                a=make_agent(rate,1,env,self.config);a.set_logger(Logger(None,[]));a.learn(17)
                records.append(dict(actor={k:v.clone() for k,v in a.actor.state_dict().items()},
                    critic={k:v.clone() for k,v in a.critic.state_dict().items()},alpha=a.log_ent_coef.clone()))
            finally:env.close()
        for r in records[1:]:
            self.assertTrue(all(torch.equal(v,r['critic'][k]) for k,v in records[0]['critic'].items()))
            self.assertTrue(torch.equal(records[0]['alpha'],r['alpha']))
            self.assertTrue(any(not torch.equal(v,r['actor'][k]) for k,v in records[0]['actor'].items()))

    def test_shared_lr_change_rejected_and_effective_difference_is_only_actor_lr(self):
        env=SyntheticTerminal()
        try:
            old=json.loads((folder_for(BASE_LR,0)/'summary.json').read_text())['effective_hyperparameters']
            for rate in ACTOR_RATES:effective_difference(make_agent(rate,0,env,self.config),old,rate)
            with self.assertRaises(ValueError):make_agent(1e-4,0,env,dict(self.config,learning_rate=1e-4))
        finally:env.close()

    def test_short_native_training_lrs_warmup_and_three_endpoint_replays(self):
        with tempfile.TemporaryDirectory(prefix='issue24-test-') as d:
            out=Path(d);config=dict(self.config,episodes=32,eval_every_episodes=16)
            write_json(out/'manifest.json',dict(settings=config,source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),
                target_sha256=digest(TARGET),baseline_manifest_sha256=digest(BASE/'manifest.json'),
                candidates_sha256=digest(BASE/'fixed_candidates.json'),local_reference_sha256={}))
            for rate in ACTOR_RATES:
                s=run_one(rate,0,out);c=curve_for(rate,0,out);folder=folder_for(rate,0,out)
                self.assertEqual(len(c),3);self.assertEqual(s['optimizer_lr_checks'],dict(actor=16,critic=16,entropy=16))
                self.assertEqual(s['evaluation']['episodes'],3);self.assertTrue(s['warmup_exact'])
                self.assertTrue(all(p['optimizer_lrs']==dict(actor=[rate],critic=[BASE_LR],entropy=[BASE_LR]) for p in c))
                env=environment(.75)
                try:
                    for role in ('final','selected','posthoc'):
                        a,meta=load_checkpoint(folder/f'{role}.pt',0,config);m,v=episode(env,agent=a)
                        self.assertEqual(m['weighted_loss'],s[role]['weighted_loss'])
                        with np.load(folder/f'{role}.npz') as z:np.testing.assert_array_equal(z['audio'],v['audio'])
                finally:env.close()


if __name__=='__main__':unittest.main()
