"""Critic-only scheduling, high-LR safety, data isolation and fixed regression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import Logger

from Deeplearning.tests.test_issue24 import SyntheticTerminal
from Deeplearning.issue14 import learner_signature
from Deeplearning.issue15_analysis import write_json
from Deeplearning.issue16 import digest
from Deeplearning.issue21_training import make_agent as stock_agent
from Deeplearning.issue19_critics import scores as brute_scores
from Deeplearning.issue25_agent import (CriticRateSAC,make_agent,assert_lrs,effective_difference,numerical_check,LIMITS,BASE_LR,CRITIC_RATES)
from Deeplearning.issue25_data import (BASE,PREVIOUS,TARGET,source_hashes,sb3_hashes,environment,folder_for,load_checkpoint)
from Deeplearning.issue25_prepare import build_data
from Deeplearning.issue25_training import run_one
from Deeplearning.issue25_offline import offline_fit,scores
from Deeplearning.model import LIBRARY,BirdModel
from Deeplearning.constant020_env import episode


class CriticRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.config=json.loads((BASE/'manifest.json').read_text())['settings']

    def manifest(self,out,offline=False):
        (out/'PLAN.md').write_text('test-only fixed plan before execution')
        if offline:build_data(out,updates=32)
        write_json(out/'manifest.json',dict(settings=dict(self.config,episodes=32,eval_every_episodes=16),
            source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),
            baseline_manifest_sha256=digest(BASE/'manifest.json'),previous_manifest_sha256=digest(PREVIOUS/'manifest.json'),
            sb3_implementation_sha256=sb3_hashes(),plan_sha256=digest(out/'PLAN.md'),data_sha256={},local_reference_sha256={},
            offline=dict(updates=32,evaluation_updates=[0,16,32])))

    def test_stock_method_and_baseline_state_rng_replay_equal_after32(self):
        self.assertIs(CriticRateSAC.train,SAC.train);signatures=[]
        for factory in (lambda e:stock_agent('SAC',128,2,e,self.config),lambda e:make_agent(BASE_LR,2,e,self.config)):
            env=SyntheticTerminal()
            try:
                a=factory(env);a.set_logger(Logger(None,[]));a.learn(32);signatures.append(learner_signature(a))
            finally:env.close()
        self.assertEqual(signatures[0],signatures[1])

    def test_scheduler_and_real_steps_keep_only_critic_lr_changed(self):
        for rate in CRITIC_RATES:
            env=SyntheticTerminal()
            try:
                a=make_agent(rate,0,env,self.config);a.set_logger(Logger(None,[]))
                baseline=json.loads((folder_for(BASE_LR,0)/'summary.json').read_text())['effective_hyperparameters'];effective_difference(a,baseline,rate)
                for progress in (1.,.8,.1,0.):
                    a._current_progress_remaining=progress;a._update_learning_rate([a.actor.optimizer,a.critic.optimizer,a.ent_coef_optimizer]);assert_lrs(a,rate)
                checks=[];hook=a.critic.optimizer.register_step_pre_hook(lambda opt,args,kwargs:checks.append(assert_lrs(a,rate)))
                try:a.learn(32)
                finally:hook.remove()
                self.assertEqual(len(checks),16);assert_lrs(a,rate)
                with self.assertRaises(ValueError):make_agent(rate,0,env,dict(self.config,learning_rate=rate))
            finally:env.close()

    def test_same_first_critic_gradients_and_adam_moments_all_rates(self):
        records=[]
        for rate in (BASE_LR,*CRITIC_RATES):
            env=SyntheticTerminal()
            try:
                a=make_agent(rate,1,env,self.config);a.set_logger(Logger(None,[]));grad=[]
                hook=a.critic.optimizer.register_step_pre_hook(lambda opt,args,kwargs:grad.extend(p.grad.clone() for g in opt.param_groups for p in g['params']))
                try:a.learn(17)
                finally:hook.remove()
                states=[{k:v.clone() for k,v in s.items()} for s in a.critic.optimizer.state.values()]
                records.append(dict(grad=grad,state=states,alpha=a.log_ent_coef.clone()))
            finally:env.close()
        for r in records[1:]:
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(records[0]['grad'],r['grad'])))
            self.assertTrue(all(torch.equal(s[k],r['state'][i][k]) for i,s in enumerate(records[0]['state']) for k in s))
            self.assertTrue(torch.equal(records[0]['alpha'],r['alpha']))

    def test_fast_exact_rank_metrics_match_brute_with_ties(self):
        rng=np.random.default_rng(25)
        for n in (1,3,20,80):
            for _ in range(5):
                a=rng.integers(-3,4,n);p=rng.integers(-3,4,n);actual=brute_scores(a,p);fast=scores(a,p)
                for key,value in actual.items():
                    if value is None:self.assertIsNone(fast[key])
                    else:self.assertAlmostEqual(fast[key],value,places=12)

    def test_fixed_data_batches_critic_only_and_zero_native_regression(self):
        with tempfile.TemporaryDirectory(prefix='issue25-offline-test-') as d:
            out=Path(d);self.manifest(out,offline=True);summaries=[]
            with patch.object(BirdModel,'advance',side_effect=AssertionError('Offline native call forbidden')):
                for rate in (BASE_LR,*CRITIC_RATES):summaries.append(offline_fit(rate,2,out))
            self.assertEqual(len({s['batch_index_sha256'] for s in summaries}),1);self.assertEqual(len({s['initial_critic_sha256'] for s in summaries}),1)
            self.assertTrue(all(s['status']=='complete' and s['updates']==32 and s['actor_unchanged'] and s['temperature_unchanged'] and s['global_rng_unchanged'] and s['native_calls']==0 for s in summaries))
            with np.load(out/'offline_data.npz') as z:self.assertEqual(z['train_actions'].shape,(12288,3));self.assertEqual(z['test_actions'].shape,(99,3))

    def test_high_lr_all3_seeds_native_smoke_and_retained_endpoints(self):
        summaries=[]
        with tempfile.TemporaryDirectory(prefix='issue25-native-test-') as d:
            out=Path(d);self.manifest(out)
            for rate,seed in ((1e-4,0),(1e-3,0),(1e-3,1),(1e-3,2)):
                s=run_one(rate,seed,out);summaries.append(s)
                self.assertEqual(s['status'],'complete');self.assertEqual(s['optimizer_lr_checks'],dict(actor=16,critic=16,entropy=16));self.assertTrue(s['warmup_exact'])
                env=environment(.75)
                try:
                    for role in ('final','selected','posthoc'):
                        folder=folder_for(rate,seed,out);a,meta=load_checkpoint(folder/f'{role}.pt',seed,self.config);metrics,values=episode(env,agent=a)
                        self.assertEqual(metrics['weighted_loss'],s[role]['weighted_loss'])
                        with np.load(folder/f'{role}.npz') as z:np.testing.assert_array_equal(z['audio'],values['audio'])
                finally:env.close()
        write_json(Path('/tmp/birdsong-issue25-preflight-tests.json'),dict(passed=True,tests=8,
            command='python -m unittest Deeplearning.tests.test_issue25 -v',
            native_sound_evaluations=171,native_training_candidates=145,native_independent_evaluations=14,native_endpoint_replays=12,
            native_failure_injection_candidates=17,synthetic_terminal_episodes=179,offline_test_fits=3,offline_test_updates=96,
            high_lr_native_smoke=[dict(seed=s['seed'],rate=s['critic_learning_rate'],updates=s['gradient_updates'],numeric_maxima=s['numeric_maxima']) for s in summaries],
            test_source_sha256=digest(Path(__file__))))

    def test_emergency_failure_retained_and_never_renamed_final(self):
        with tempfile.TemporaryDirectory(prefix='issue25-stop-test-') as d:
            out=Path(d);self.manifest(out)
            with patch('Deeplearning.issue25_training.numerical_check',side_effect=FloatingPointError('injected emergency stop')):
                s=run_one(1e-3,1,out)
            self.assertEqual(s['status'],'failed');self.assertEqual(s['training']['episodes'],17);self.assertNotIn('final',s)
            self.assertFalse((folder_for(1e-3,1,out)/'final.pt').exists());self.assertTrue((folder_for(1e-3,1,out)/'failure.json').exists())

    def test_gradient_and_optimizer_state_emergency_limits(self):
        parameter=torch.nn.Parameter(torch.ones(1));opt=torch.optim.Adam([parameter],lr=1e-3);parameter.grad=torch.full((1,),1e7)
        with self.assertRaises(FloatingPointError):numerical_check(opt,gradients=True)
        parameter.grad=torch.ones(1);opt.step();opt.state[parameter]['exp_avg_sq'].fill_(-1)
        with self.assertRaises(FloatingPointError):numerical_check(opt)


if __name__=='__main__':unittest.main()
