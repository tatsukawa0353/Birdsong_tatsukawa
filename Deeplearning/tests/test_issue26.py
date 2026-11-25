"""Continuous local behavior, exact 256-prefix and executed-action Replay tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from Deeplearning.issue18_domain_env import DomainEnv
from Deeplearning.issue21_training import make_agent as stock_agent
from Deeplearning.issue14 import learner_signature
from Deeplearning.issue16 import digest
from Deeplearning.issue15_analysis import write_json
from Deeplearning.issue26_data import (BASE,TARGET,SEEDS,LOW,HIGH,RADII,source_hashes,sb3_hashes,environment,folder_for,load_checkpoint,target_data)
from Deeplearning.issue26_agent import make_agent,continuous_local,observer_signature
from Deeplearning.issue26_training import run_one,comparable
from Deeplearning.constant020_env import episode
from Deeplearning.model import LIBRARY,LOW as FULL_LOW,HIGH as FULL_HIGH


class SyntheticWide(gym.Env):
    encode=DomainEnv.encode;decode=DomainEnv.decode
    def __init__(self,constant=False):
        self.fixed_controls=target_data()[1];self.indices=[0,2,3];self.domain_low=FULL_LOW.copy();self.domain_high=FULL_HIGH.copy();self.constant=constant
        self.action_space=gym.spaces.Box(-1,1,(3,),np.float32);self.observation_space=gym.spaces.Box(-1,1,(8,),np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.action_space.seed(seed)
        return np.array([-1,0,0,0,0,0,0,0],np.float32),{}
    def step(self,action):
        spec=float(np.sum((np.asarray(action)-[.2,-.1,-.8])**2));env=.2*spec;j=1. if self.constant else .75*spec+.25*env
        info=dict(acoustic_error=spec+env,acoustic_components=dict(log_spectral=spec,envelope=env,total=spec+env),weighted_loss=j,
            failed=False,sustained_voiced=False if self.constant else bool(action[2]<.2),applied_action=np.asarray(action,float).tolist(),applied_controls=self.decode(action).tolist())
        return np.array([1,0,0,0,0,0,0,0],np.float32),-j,True,False,info


class LocalBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.config=json.loads((BASE/'manifest.json').read_text())['settings']

    def test_all3_seeds_stock_B_C_D_first256_match_full_state_and_history(self):
        for seed in SEEDS:
            reference=None;reference_trace=None
            for condition in ('stock','B','C','D'):
                env=SyntheticWide()
                try:
                    agent=stock_agent('SAC',128,seed,env,self.config) if condition=='stock' else make_agent(condition,seed,env,self.config)
                    agent.set_logger(Logger(None,[]));trace=[]
                    class Capture(BaseCallback):
                        def _on_step(self):
                            i=self.locals['infos'][0];trace.append(dict(action=i['applied_action'],controls=i['applied_controls'],J=i['weighted_loss'],reward=float(self.locals['rewards'][0])))
                            return True
                    agent.learn(256,callback=Capture());signature=learner_signature(agent)
                    if reference is None:reference=signature;reference_trace=trace
                    else:self.assertEqual(signature,reference);self.assertEqual(trace,reference_trace);self.assertTrue(all(p['proposal']['source']!='local' for p in agent.candidate_log))
                finally:env.close()

    def test_intersection_uniform_is_dense_not_grid_or_boundary_clipping(self):
        env=SyntheticWide();agent=make_agent('C',0,env,self.config);center=np.array([LOW[0]+1,HIGH[1]-1,LOW[2]+1])
        for condition in ('C','D'):
            rng=np.random.default_rng(26);actual=[];boundary=0
            for _ in range(4096):
                action,replay,p=continuous_local(env,agent.policy,center,RADII[condition],rng);c=env.decode(action[0])[[0,2,3]]
                self.assertTrue(np.all(c>=p['local_lower']) and np.all(c<=p['local_upper']));actual.append(c);boundary+=int(c[2]==LOW[2])
                np.testing.assert_array_equal(agent.policy.scale_action(action),replay)
            actual=np.array(actual);self.assertGreater(len(np.unique(actual[:,2])),3000);self.assertLess(boundary,10)
            lower=np.maximum(LOW,center-RADII[condition]);upper=np.minimum(HIGH,center+RADII[condition])
            self.assertTrue(np.all(abs(actual.mean(0)-(lower+upper)/2)<=.03*(upper-lower)))
        env.close()

    def test_shared_physical_controls_reproduce_same_sound_across_domains(self):
        a=environment('A');b=environment('B');controls=np.array([4e7,3.5e7,3.25e6,49000.,49000.])
        try:
            ma,va=episode(a,action=a.encode(controls));mb,vb=episode(b,action=b.encode(controls))
            np.testing.assert_array_equal(va['physical_controls'],vb['physical_controls']);np.testing.assert_array_equal(va['audio'],vb['audio'])
            self.assertEqual(ma['weighted_loss'],mb['weighted_loss'])
        finally:a.close();b.close()

    def test_first_tied_nonvoiced_sample_remains_center_and_eval_cannot_replace(self):
        env=SyntheticWide(constant=True)
        try:
            a=make_agent('C',1,env,dict(self.config,local_start=16));a.set_logger(Logger(None,[]));a.learn(32)
            self.assertEqual(a.incumbent['trial'],1);self.assertFalse(a.incumbent['sustained_voiced'])
            before=observer_signature(a);a.predict(np.array([-1,0,0,0,0,0,0,0],np.float32),deterministic=True)
            from Deeplearning.issue22_data import policy_statistics
            policy_statistics(a);self.assertEqual(observer_signature(a),before)
        finally:env.close()

    def test_normal_component_allows_steps_beyond_local_radius(self):
        env=SyntheticWide()
        try:
            a=make_agent('C',0,env,self.config);a.num_timesteps=256;a._last_obs=np.array([[-1,0,0,0,0,0,0,0]],np.float32)
            a.incumbent=dict(physical_controls=[4e7,3.5e7,3.25e6,50000.,49000.],J=.1,trial=1)
            class Choice:
                def random(self):return .99
            a.choice_rng=Choice();action,_=a._sample_action(16)
            self.assertEqual(a.last_proposal['source'],'sac');self.assertGreater(abs(env.decode(action[0])[3]-50000),50000)
        finally:env.close()

    def test_native_short_B_C_D_replay_prefix_and_four_saved_sounds(self):
        summaries=[];traces={}
        with tempfile.TemporaryDirectory(prefix='issue26-native-test-') as d:
            out=Path(d);(out/'PLAN.md').write_text('test-only plan')
            write_json(out/'manifest.json',dict(settings=dict(self.config,episodes=32,eval_every_episodes=16,local_start=16),
                source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),
                baseline_manifest_sha256=digest(BASE/'manifest.json'),sb3_implementation_sha256=sb3_hashes(),plan_sha256=digest(out/'PLAN.md')))
            for condition in ('B','C','D'):
                summary=run_one(condition,0,out);self.assertEqual(summary['status'],'complete');summaries.append(summary)
                self.assertEqual(summary['replay_transition_checks'],32);self.assertEqual(summary['optimizer_steps'],dict(actor=16,critic=16,entropy=16))
                if condition!='B':self.assertTrue(summary['prefix_comparison']['passed'])
                folder=folder_for(condition,0,out)
                from Deeplearning.issue17_artifacts import read_record
                traces[condition]=read_record(folder/'training_trace.json')
                env=environment(condition)
                try:
                    for role in ('final','selected','posthoc'):
                        agent,meta=load_checkpoint(folder/f'{role}.pt',condition,0,self.config);metrics,values=episode(env,agent=agent)
                        self.assertEqual(metrics['weighted_loss'],summary[role]['weighted_loss'])
                        with np.load(folder/f'{role}.npz') as z:np.testing.assert_array_equal(z['audio'],values['audio'])
                    with np.load(folder/'training_best.npz') as z:
                        metrics,values=episode(env,action=z['actions'][0]);np.testing.assert_array_equal(z['audio'],values['audio'])
                        self.assertEqual(metrics['weighted_loss'],summary['training_best']['J'])
                finally:env.close()
            self.assertEqual([p['proposal']['source'] for p in traces['C']],[p['proposal']['source'] for p in traces['D']])
            self.assertTrue(any(p['proposal']['source']=='local' and p['proposal']['stock_action']!=p['stored_action'] for p in traces['C']))
        write_json(Path('/tmp/birdsong-issue26-native-test.json'),dict(native_test_passed=True,tests=6,command='python -m Deeplearning.tests.test_issue26',
            synthetic_terminal_episodes=3104,prefix_seeds=3,prefix_steps=256,prefix_conditions=['stock','B','C','D'],
            pure_local_candidate_draws=8192,native_sound_evaluations=119,native_training_candidates=96,native_policy_evaluations=9,
            native_replay_sounds=12,native_shared_domain_sounds=2,test_source_sha256=digest(Path(__file__))))


if __name__=='__main__':
    import sys
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalBehaviorTests))
    if result.wasSuccessful():
        p=Path('/tmp/birdsong-issue26-native-test.json');r=json.loads(p.read_text());r.update(passed=True,test_source_sha256=digest(Path(__file__)))
        write_json(Path('/tmp/birdsong-issue26-preflight-tests.json'),r)
    sys.exit(0 if result.wasSuccessful() else 1)
