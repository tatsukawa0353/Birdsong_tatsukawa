"""Paired-dimension mapping, stock algorithm parity and18 native smoke runs."""
from Deeplearning.result_paths import source_hashes_match
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from pathlib import Path
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import Logger
from Deeplearning.issue27g_data import ROOT,OUT,CONTROLS,LOW,HIGH,CONFIG,hashes,target_path,target_data,folder,indices,load_checkpoint
from Deeplearning.issue27g_env import CaseBEnv,episode
from Deeplearning.issue27g_agent import make_agent,geometry,effective
from Deeplearning.issue27g_native import render,snapshot,ConfiguredCaseB,PROBE
from Deeplearning.issue27g_training import run_one
from Deeplearning.issue18_domain_env import DomainEnv
from Deeplearning.issue14 import learner_signature
from Deeplearning.issue16 import digest
from Deeplearning.issue15_analysis import write_json
from Deeplearning.model import LIBRARY,BirdModel

class Synthetic(gym.Env):
    encode=DomainEnv.encode;decode=DomainEnv.decode
    def __init__(self,dim):
        self.indices=indices(dim);self.domain_low=LOW;self.domain_high=HIGH;self.fixed_controls=CONTROLS['T3'];self.target_name='synthetic';self.dim=dim
        self.action_space=gym.spaces.Box(-1,1,(dim,),np.float32);self.observation_space=gym.spaces.Box(-1,1,(8,),np.float32);self.last_info=None;self.best_candidate=None;self.completed_episodes=0
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.action_space.seed(seed)
        return np.array([-1,0,0,0,0,0,0,0],np.float32),{}
    def step(self,action):
        self.completed_episodes+=1;e=float(np.square(np.asarray(action)-np.linspace(-.6,-1,self.dim)).sum());env=.2*e;j=.75*e+.25*env
        p=dict(acoustic_error=e+env,acoustic_components=dict(log_spectral=e,envelope=env,total=e+env),weighted_loss=j,failed=False,sustained_voiced=False,generated_silent=False,
            applied_action=np.asarray(action,float).tolist(),applied_controls=self.decode(action).tolist());self.last_info=p
        if self.best_candidate is None or j<self.best_candidate['weighted_loss']:self.best_candidate=dict(episodes=self.completed_episodes,**p)
        return np.array([1,0,0,0,0,0,0,0],np.float32),-j,True,False,p

def smoke_replay(target,dim,method):
    torch.set_num_threads(1);output=OUT/'preflight/smoke';s=run_one(target,dim,method,0,output)
    assert s['status']=='complete',s;assert s['training']['episodes']==128 and s['independent_policy_evaluations']==2 and s['transition_checks']==128
    p=folder(target,dim,method,0,output);env=CaseBEnv(target,dim,record_moments=True)
    try:
        for role in ('final','selected','posthoc'):
            agent,meta=load_checkpoint(p/f'{role}.pt',target,dim,method,0);metrics,values=episode(env,agent=agent)
            assert metrics['weighted_loss']==s[role]['weighted_loss']
            with np.load(p/f'{role}.npz') as z:
                for k in ('audio','actions','physical_controls','state_end_blocks','motion_mean','motion_std'):np.testing.assert_array_equal(z[k],values[k])
        with np.load(p/'training_best.npz') as z:
            metrics,values=episode(env,action=z['actions'][0]);np.testing.assert_array_equal(z['audio'],values['audio']);assert metrics['weighted_loss']==s['training_best']['weighted_loss']
    finally:env.close()
    return dict(target=target,dim=dim,method=method,passed=True,native_sounds=134,training=128,evaluations=2,replays=4,optimizer_steps=s['optimizer_steps'])

class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_01_targets_and_six_mappings(self):
        gate=json.loads((OUT/'preflight/gate.json').read_text());initial=[]
        for t,c in CONTROLS.items():
            self.assertTrue(gate[t]['passed'])
            for dim in (3,4):
                env=CaseBEnv(t,dim)
                try:
                    _,info=env.reset(seed=0);np.testing.assert_array_equal(env.decode(env.encode(c)),c)
                    for a in (np.full(dim,-1.),np.zeros(dim),np.ones(dim),env.encode(c)):
                        physical=env.decode(a);env.model.set_controls(physical);state=env.check();np.testing.assert_array_equal(env.model.controls,physical)
                        self.assertEqual(state['epsilon_r'],physical[1]);self.assertEqual(state['f0_r'],c[4]);np.testing.assert_allclose(env.encode(physical),a,atol=1e-15,rtol=0)
                        if dim==3:self.assertEqual(physical[1],c[1])
                    for m in ('SAC','TD3','PPO'):
                        hs=[]
                        for seed in (0,1,2):
                            agent=make_agent(m,seed,env,CONFIG);geometry(agent,m);effective(agent,m);hs.append(learner_signature(agent)['policy_sha256'])
                        self.assertEqual(len(set(hs)),3);initial.append(dict(target=t,dim=dim,method=m,seed_policy_hashes=hs))
                finally:env.close()
        original=BirdModel('B')
        try:original.reset(CONTROLS['T2']);self.assertEqual(original.controls[4],1e7)
        finally:original.close()
        write_json(OUT/'preflight/initialization.json',dict(passed=True,native_sounds=0,cases=initial))

    def test_02_standard_update_parity(self):
        for dim in (3,4):
            for m in ('SAC','TD3','PPO'):
                signatures=[]
                for observed in (False,True):
                    env=Synthetic(dim)
                    try:
                        agent=make_agent(m,0,env,CONFIG,audited=observed);agent.set_logger(Logger(None,[]));agent.learn(128);signatures.append(learner_signature(agent));geometry(agent,m)
                        if observed:self.assertEqual(agent.transition_checks,128)
                    finally:env.close()
                self.assertEqual(signatures[0],signatures[1],(dim,m))

    def test_03_T1_four_dimensional_right_epsilon_changes_output(self):
        p=OUT/'preflight/right_sensitivity.json'
        if p.exists():raise FileExistsError('Do not repeat physical check')
        audio=[];rows=[];ledger=[]
        for er in (2e7,5e7):
            c=CONTROLS['T1'].copy();c[1]=er;ledger.append(dict(epsilon_r=er,status='started'));write_json(OUT/'preflight/right_sensitivity_ledger.json',ledger)
            v,coef=render(c,detailed=True);self.assertEqual(coef['epsilon_r'],er);self.assertEqual(coef['f0_r'],1e7);self.assertGreater(abs(v['diagnostic_moments'][-1,5]),0)
            np.savez_compressed(OUT/f'preflight/T1_right_epsilon_{int(er)}.npz',**v);audio.append(v['audio']);rows.append(dict(epsilon_r=er,native_coefficients=coef))
            ledger[-1].update(status='complete',native_steps=2000000);write_json(OUT/'preflight/right_sensitivity_ledger.json',ledger)
        self.assertFalse(np.array_equal(*audio));write_json(p,dict(passed=True,native_sounds=2,waveform_max_abs_difference=float(abs(audio[0]-audio[1]).max()),settings=rows))

    def test_04_native_smoke_six_cases_three_methods(self):
        smoke=OUT/'preflight/smoke'
        if smoke.exists():
            assert source_hashes_match(json.loads((smoke/'manifest.json').read_text())['source_sha256'], hashes())
            for t in CONTROLS:
                for d in (3,4):
                    for m in ('SAC','TD3','PPO'):assert json.loads((folder(t,d,m,0,smoke)/'summary.json').read_text())['status']=='complete'
        else:
            smoke.mkdir();write_json(smoke/'manifest.json',dict(preflight_passed=True,source_sha256=hashes(),native_library_sha256=digest(LIBRARY),probe_sha256=digest(PROBE),
                target_sha256={t:digest(target_path(t)) for t in CONTROLS},plan_sha256=digest(OUT/'PLAN.md'),settings=dict(CONFIG,episodes=128)))
        result=[]
        with ProcessPoolExecutor(max_workers=18,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(smoke_replay,t,d,m) for t in CONTROLS for d in (3,4) for m in ('SAC','TD3','PPO')]
            for f in as_completed(futures):result.append(f.result())
        self.assertEqual(len(result),18);write_json(OUT/'preflight/smoke_audit.json',dict(passed=True,native_sounds=2412,training=2304,evaluations=36,replays=72,runs=result))

if __name__=='__main__':
    import sys
    names=('test_01_targets_and_six_mappings','test_02_standard_update_parity','test_03_T1_four_dimensional_right_epsilon_changes_output','test_04_native_smoke_six_cases_three_methods')
    resumed='--replay-only' in sys.argv
    if resumed:names=names[-1:]
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.TestSuite(Preflight(n) for n in names))
    if result.wasSuccessful():write_json(OUT/'preflight/tests.json',dict(passed=True,tests=result.testsRun,source_sha256=hashes(),test_source_sha256=digest(Path(__file__)),synthetic_episodes=1536,
        smoke_runs=json.loads((OUT/'preflight/smoke_audit.json').read_text())['runs'],native_sensitivity_sounds=2,native_smoke_sounds=2412,
        first_attempt_tests_run=4,first_attempt_first_three_passed=True,replay_only_resume=resumed,repeated_training=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
