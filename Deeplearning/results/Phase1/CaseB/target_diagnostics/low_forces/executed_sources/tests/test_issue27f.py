"""Stock-update parity and native smoke tests, separately counted from4096 runs."""
import json
from pathlib import Path
import tempfile
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import Logger
from Deeplearning.issue27f_data import ROOT,OUT,TARGET,hashes,target_data,folder,load_checkpoint
from Deeplearning.issue27f_prepare import CONFIG
from Deeplearning.issue27f_env import CaseBEnv,episode
from Deeplearning.issue27f_agent import make_agent,geometry,effective,signature
from Deeplearning.issue27f_native import ConfiguredCaseB,snapshot,PROBE
from Deeplearning.issue27f_training import run_one
from Deeplearning.issue18_domain_env import DomainEnv
from Deeplearning.issue14 import learner_signature
from Deeplearning.model import BirdModel,LOW,HIGH,LIBRARY
from Deeplearning.issue16 import digest
from Deeplearning.issue15_analysis import write_json
from Deeplearning.issue17_artifacts import read_record

class Synthetic(gym.Env):
    encode=DomainEnv.encode;decode=DomainEnv.decode
    def __init__(self):
        self.indices=[0,2,3];self.domain_low=LOW.copy();self.domain_high=HIGH.copy();self.domain_high[3]=800000.;self.fixed_controls=target_data()[1]
        self.action_space=gym.spaces.Box(-1,1,(3,),np.float32);self.observation_space=gym.spaces.Box(-1,1,(8,),np.float32);self.completed_episodes=0;self.last_info=None;self.best_candidate=None
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.action_space.seed(seed)
        return np.array([-1,0,0,0,0,0,0,0],np.float32),{}
    def step(self,action):
        self.completed_episodes+=1;s=float(np.square(np.asarray(action)-[-.6,-.33,-1.]).sum());e=.2*s;j=.75*s+.25*e
        p=dict(acoustic_error=s+e,acoustic_components=dict(log_spectral=s,envelope=e,total=s+e),weighted_loss=j,failed=False,sustained_voiced=False,generated_silent=False,
            applied_action=np.asarray(action,float).tolist(),applied_controls=self.decode(action).tolist());self.last_info=p
        if self.best_candidate is None or j<self.best_candidate['weighted_loss']:self.best_candidate=dict(episodes=self.completed_episodes,**p)
        return np.array([1,0,0,0,0,0,0,0],np.float32),-j,True,False,p

class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_18_points_oracle_and_native_right_controls(self):
        records=json.loads((OUT/'preflight/candidates.json').read_text());self.assertEqual(len(records),18)
        self.assertEqual({r['f0_r'] for r in records},{49000.,1e7});self.assertTrue(all(r['eligible'] for r in records))
        for force in (49000.,1e7):
            model=ConfiguredCaseB(force)
            try:
                model.reset([3e7,3.5e7,3e6,49000.,1e7 if force==49000 else 49000]);s=snapshot(model,reset=True)
                self.assertEqual(s['f0_r'],force);self.assertEqual(s['alpha_r'],1.2e6);self.assertEqual(s['beta_r'],1500);self.assertEqual(model.condition,'B')
            finally:model.close()
        original=BirdModel('B')
        try:original.reset([3e7,3.5e7,3e6,49000,49000]);self.assertEqual(original.controls[4],1e7)
        finally:original.close()
        with np.load(TARGET) as t,np.load(OUT/'preflight/oracle.npz') as o:
            for key in ('audio','state_end_blocks','motion_mean','motion_std'):np.testing.assert_array_equal(t[key],o[key])

    def test_standard_algorithms_match_observed_weights_optimizers_RNG_and_buffers(self):
        for method in ('SAC','TD3','PPO'):
            signatures=[];trajectories=[]
            for observed in (False,True):
                env=Synthetic()
                try:
                    agent=make_agent(method,0,env,CONFIG,audited=observed);agent.set_logger(Logger(None,[]));agent.learn(128)
                    signatures.append(learner_signature(agent));geometry(agent,method);effective(agent,method)
                    if observed:
                        self.assertEqual(agent.transition_checks,128)
                        if method=='PPO':self.assertTrue(any(r['action_was_clipped'] for r in agent.transition_rows))
                finally:env.close()
            self.assertEqual(signatures[0],signatures[1],method)

    def test_all_seeds_initialization_and_physical_mapping(self):
        for method in ('SAC','TD3','PPO'):
            hashes_by_seed=[]
            for seed in (0,1,2):
                env=CaseBEnv()
                try:
                    agent=make_agent(method,seed,env,CONFIG);geometry(agent,method);effective(agent,method)
                    hashes_by_seed.append(learner_signature(agent)['policy_sha256']);self.assertEqual(agent.num_timesteps,0)
                    for action in (np.full(3,-1.),np.zeros(3),np.ones(3)):
                        c=env.decode(action);self.assertEqual(c[4],target_data()[1][4]);self.assertEqual(c[1],3.5e7)
                        np.testing.assert_allclose(env.encode(c),action,atol=1e-15,rtol=0)
                finally:env.close()
            self.assertEqual(len(set(hashes_by_seed)),3)

    def test_native128_smoke_all_methods_replay_updates_and_saved_models(self):
        smoke=OUT/'preflight/smoke'
        if smoke.exists():raise FileExistsError('A previous smoke attempt must be preserved and counted separately')
        smoke.mkdir();write_json(smoke/'manifest.json',dict(preflight_passed=True,source_sha256=hashes(),native_library_sha256=digest(LIBRARY),probe_sha256=digest(PROBE),
            target_sha256=digest(TARGET),plan_sha256=digest(OUT/'PLAN.md'),settings=dict(CONFIG,episodes=128,eval_every_episodes=128)))
        counts=[];replays=[]
        for method in ('SAC','TD3','PPO'):
            s=run_one(method,0,smoke);self.assertEqual(s['status'],'complete',s.get('failure'));self.assertEqual(s['training']['episodes'],128);self.assertEqual(s['independent_policy_evaluations'],2)
            self.assertEqual(s['transition_checks'],128);p=folder(method,0,smoke);env=CaseBEnv(record_moments=True)
            try:
                for role in ('final','selected','posthoc'):
                    agent,meta=load_checkpoint(p/f'{role}.pt',method,0,CONFIG);metrics,values=episode(env,agent=agent)
                    self.assertEqual(metrics['weighted_loss'],s[role]['weighted_loss'])
                    with np.load(p/f'{role}.npz') as z:
                        for key in ('audio','actions','physical_controls','state_end_blocks','motion_mean','motion_std'):np.testing.assert_array_equal(values[key],z[key])
                    replays.append(dict(method=method,role=role,exact=True))
                with np.load(p/'training_best.npz') as z:
                    metrics,values=episode(env,action=z['actions'][0]);np.testing.assert_array_equal(values['audio'],z['audio']);self.assertEqual(metrics['weighted_loss'],s['training_best']['weighted_loss'])
                replays.append(dict(method=method,role='training_best',exact=True))
            finally:env.close()
            counts.append(dict(method=method,training=128,independent_evaluations=2,replays=4,optimizer_steps=s['optimizer_steps']))
        write_json(OUT/'preflight/smoke_audit.json',dict(passed=True,additional_native_sounds=402,training_candidates=384,independent_evaluations=6,saved_replays=12,runs=counts,replays=replays))

if __name__=='__main__':
    names=('test_18_points_oracle_and_native_right_controls','test_standard_algorithms_match_observed_weights_optimizers_RNG_and_buffers',
        'test_all_seeds_initialization_and_physical_mapping','test_native128_smoke_all_methods_replay_updates_and_saved_models')
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestSuite(MethodTests(n) for n in names))
    history=json.loads((OUT/'preflight/synthetic_precision_failure.json').read_text())
    synthetic_previous=history['initial_suite_synthetic_episodes']+history['failed_SAC_reproduction_synthetic_episodes']+history['TD3_reproduction_synthetic_episodes']
    if result.wasSuccessful():write_json(OUT/'preflight/tests.json',dict(passed=True,tests=result.testsRun,source_sha256=hashes(),test_source_sha256=digest(Path(__file__)),synthetic_episodes=768+synthetic_previous,successful_suite_synthetic_episodes=768,previous_synthetic_episodes=synthetic_previous,
        native_smoke_sounds=402,training_research_runs=0,command='python -m Deeplearning.tests.test_issue27f'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
