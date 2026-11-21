"""Width-only changes, actual network geometry, and end-to-end compact runs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import TD3,SAC

from Deeplearning.constant020_env import episode
from Deeplearning.issue14 import effective_hyperparameters
from Deeplearning.issue15_analysis import write_json
from Deeplearning.issue16 import digest
from Deeplearning.issue21_data import (BASE,PRIOR,environment,folder_for,source_hashes,curve_for)
from Deeplearning.issue21_training import make_agent,network_geometry,run_one
from Deeplearning.model import LIBRARY


class NetworkWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=json.loads((BASE/'manifest.json').read_text())['settings']

    def test_width_is_only_effective_change_for_both_methods(self):
        for method in ('TD3','SAC'):
            baseline=json.loads((folder_for(method,32,0)/'summary.json').read_text())['effective_hyperparameters']
            for width in (64,128,256):
                env=environment()
                try:
                    agent=make_agent(method,width,0,env,self.config)
                    effective=json.loads(json.dumps(effective_hyperparameters(agent)))
                    self.assertEqual(dict(effective,net_arch=[32,32]),baseline)
                    geometry=network_geometry(agent,width)
                    self.assertEqual(geometry['Q1'],geometry['Q2'])
                finally:env.close()

    def test_detects_accidentally_unchanged_twin_critic(self):
        env=environment()
        try:
            agent=make_agent('TD3',64,0,env,self.config)
            agent.critic.q_networks[1][0]=torch.nn.Linear(11,32)
            with self.assertRaises(AssertionError):network_geometry(agent,64)
        finally:env.close()

    def test_short_runs_preserve_budget_evaluation_selection_and_model_reloads(self):
        with tempfile.TemporaryDirectory(prefix='issue21-test-') as directory:
            out=Path(directory)
            config=dict(self.config,episodes=32,eval_every_episodes=16,checkpoints=[])
            write_json(out/'manifest.json',dict(settings=config,source_sha256=source_hashes(),
                native_library_sha256=digest(LIBRARY),target_sha256=digest(BASE/'diagnostics/target.npz'),
                critic_pool_sha256=digest(PRIOR/'critic_pool.csv')))
            for method,width in (('TD3',64),('SAC',256)):
                summary=run_one(method,width,0,out)
                points=curve_for(method,width,0,out)
                self.assertEqual([p['episodes'] for p in points],[0,16,32])
                self.assertEqual(summary['training']['episodes'],32)
                self.assertEqual(summary['evaluation']['episodes'],5)
                self.assertEqual(summary['gradient_updates'],16)
                self.assertEqual(summary['selected']['weighted_loss'],min(p['weighted_loss'] for p in points))
                self.assertTrue(all(p['learner_before']==p['learner_after'] for p in points))
                env=environment();folder=folder_for(method,width,0,out)
                try:
                    for role in ('final','selected'):
                        model={'TD3':TD3,'SAC':SAC}[method].load(folder/('agent.zip' if role=='final' else 'selected_agent.zip'),device='cpu')
                        metrics,values=episode(env,agent=model)
                        self.assertEqual(metrics['acoustic_error'],summary[role]['acoustic_error'])
                        with np.load(folder/f'{role}.npz') as z:np.testing.assert_array_equal(z['audio'],values['audio'])
                finally:env.close()
                # Completed run reuse must not rewrite original models or timing.
                before=(folder/'summary.json').read_bytes()
                self.assertEqual(run_one(method,width,0,out),summary)
                self.assertEqual((folder/'summary.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
