"""Reward scale, invariant physics, non-mutating entropy and weighted selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import SAC

from Deeplearning.constant020_env import episode
from Deeplearning.issue14 import learner_signature
from Deeplearning.issue15_analysis import write_json
from Deeplearning.issue16 import digest
from Deeplearning.issue21_training import make_agent
from Deeplearning.issue22_data import (BASE,TARGET,POOL,WEIGHTS,coefficients,objective,
    environment,policy_statistics,source_hashes,folder_for,endpoints,curve_for)
from Deeplearning.issue22_training import run_one
from Deeplearning.model import LIBRARY


class SumOneRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=json.loads((BASE/'manifest.json').read_text())['settings']

    def test_coefficients_and_exact_half_old_objective(self):
        pairs=np.random.default_rng(71).uniform(0,4,(2000,2))
        for weight in WEIGHTS:
            a,b=coefficients(weight);self.assertEqual(a+b,1.)
            for spec,env in pairs:
                self.assertEqual(objective(spec,env,weight),a*spec+b*env)
                if weight==.5:self.assertEqual(objective(spec,env,weight),(spec+env)/2)
        for value in (-.1,1.1,np.nan,np.inf):
            with self.assertRaises(ValueError):coefficients(value)

    def test_physics_and_common_metric_identical_across_all_rewards(self):
        outputs=[]
        for condition in ('legacy_sum',*WEIGHTS):
            env=environment(condition)
            try:outputs.append(episode(env,action=[.1,.2,-.5]))
            finally:env.close()
        for metrics,values in outputs[1:]:
            np.testing.assert_array_equal(values['audio'],outputs[0][1]['audio'])
            self.assertEqual(metrics['acoustic_error'],outputs[0][0]['acoustic_error'])
            self.assertNotIn('envelope_weight',metrics)
        self.assertEqual(outputs[1][0]['weighted_loss'],outputs[0][0]['weighted_loss']/2)

    def test_failed_candidate_retains_existing_hard_failure_penalty(self):
        env=environment(.9)
        try:
            env.reset()
            with patch.object(env.model,'advance',side_effect=FloatingPointError('injected')):
                _,reward,done,truncated,info=env.step([0,0,0])
            self.assertEqual(reward,-1e6);self.assertTrue(done and info['failed']);self.assertFalse(truncated)
            self.assertEqual(env.completed_episodes,1)
        finally:env.close()

    def test_statistics_do_not_consume_learner_rng_or_modify_state(self):
        env=environment(.5)
        try:
            agent=make_agent('SAC',128,0,env,self.config)
            before=learner_signature(agent);a=policy_statistics(agent);b=policy_statistics(agent)
            self.assertEqual(learner_signature(agent),before);self.assertEqual(a,b)
            self.assertTrue(all(0<=v<=1.01 for v in a['normalized_action_std_mc']))
            self.assertAlmostEqual(a['alpha'],.01,places=7)
        finally:env.close()

    def test_minJ_selection_and_posthoc_common_minimum_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);folder=folder_for(.9,0,out);folder.mkdir(parents=True)
            rows=[dict(episodes=n,weighted_loss=objective(spec,env,.9),acoustic_error=spec+env)
                  for n,spec,env in ((0,.1,.5),(128,.2,.1),(4096,.25,.15))]
            write_json(folder/'curve.json',rows);p=endpoints(.9,0,out)
            self.assertEqual(p['selected']['episodes'],0)
            self.assertEqual(p['posthoc']['episodes'],128)
            self.assertEqual(p['final']['episodes'],4096)

    def test_short_training_and_saved_models_use_the_requested_objective(self):
        with tempfile.TemporaryDirectory(prefix='issue22-test-') as d:
            out=Path(d);config=dict(self.config,episodes=32,eval_every_episodes=16)
            write_json(out/'manifest.json',dict(settings=config,source_sha256=source_hashes(),
                native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),critic_pool_sha256=digest(POOL)))
            for weight in (.5,.9):
                summary=run_one(weight,0,out);curve=curve_for(weight,0,out);folder=folder_for(weight,0,out)
                self.assertEqual([p['episodes'] for p in curve],[0,16,32])
                self.assertEqual(summary['training']['episodes'],32);self.assertEqual(summary['evaluation']['episodes'],5)
                self.assertEqual(summary['gradient_updates'],16)
                self.assertEqual(summary['selected']['weighted_loss'],min(p['weighted_loss'] for p in curve))
                self.assertEqual(summary['posthoc']['acoustic_error'],min(p['acoustic_error'] for p in curve))
                env=environment(weight)
                try:
                    for role in ('final','selected'):
                        model=SAC.load(folder/('agent.zip' if role=='final' else 'selected_agent.zip'),device='cpu')
                        metrics,values=episode(env,agent=model)
                        self.assertEqual(metrics['weighted_loss'],summary[role]['weighted_loss'])
                        with np.load(folder/f'{role}.npz') as z:np.testing.assert_array_equal(values['audio'],z['audio'])
                finally:env.close()


if __name__=='__main__':unittest.main()
