"""Diagnostic retention must preserve the actual SAC learning trajectory."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from Deeplearning.issue14 import learner_signature
from Deeplearning.issue15_analysis import write_json
from Deeplearning.issue16 import digest
from Deeplearning.issue21_training import make_agent
from Deeplearning.issue23_data import (BASE,TARGET,environment,source_hashes,fixed_candidates,
    deterministic_state,predict_q,q_action_gradient,save_checkpoint,load_checkpoint,chosen,curve_for,folder_for)
from Deeplearning.issue23_stage_a import correlation
from Deeplearning.issue23_training import run_one
from Deeplearning.model import LIBRARY
from Deeplearning.constant020_env import episode


class StabilityDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.config=json.loads((BASE/'manifest.json').read_text())['settings']

    def test_queries_and_checkpoint_preserve_learner_rng_and_gradients(self):
        env=environment(.75)
        try:
            agent=make_agent('SAC',128,2,env,self.config)
            before=learner_signature(agent)
            state=deterministic_state(agent);q1,q2=predict_q(agent,[state['action'],[.1,.2,-.4]])
            grad=q_action_gradient(agent,state['action'])
            self.assertTrue(np.isfinite([*q1,*q2,*grad]).all())
            self.assertTrue(all(p.grad is None for p in agent.policy.parameters()))
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'policy.pt'
                row=dict(episodes=0,actions=[state['action']],policy_statistics=dict(alpha=state['alpha']),weighted_loss=1.,acoustic_error=2.)
                save_checkpoint(p,agent,row)
                self.assertEqual(learner_signature(agent),before)
                loaded,meta=load_checkpoint(p,2,self.config)
                self.assertEqual(meta['trial'],0)
                np.testing.assert_array_equal(predict_q(loaded,[state['action']]),predict_q(agent,[state['action']]))
                self.assertEqual(deterministic_state(loaded)['action'],state['action'])
                data=torch.load(p,weights_only=True)
                self.assertNotIn('optimizer',data);self.assertNotIn('replay_buffer',data)
        finally:env.close()

    def test_minJ_and_minE_ties_keep_first_point_and_final_separate(self):
        points=[dict(episodes=n,weighted_loss=j,acoustic_error=e) for n,j,e in ((0,.3,.4),(128,.1,.3),(256,.1,.2),(4096,.2,.5))]
        p=chosen(points)
        self.assertEqual(p['selected']['episodes'],128);self.assertEqual(p['posthoc']['episodes'],256);self.assertEqual(p['final']['episodes'],4096)

    def test_constant_correlations_are_missing_not_spurious(self):
        self.assertIsNone(correlation([1,1,1],[1,2,3]));self.assertEqual(correlation([1,2,3],[3,2,1]),-1.)

    def test_short_reproduction_is_exact_and_all_three_policies_reload(self):
        with tempfile.TemporaryDirectory(prefix='issue23-test-') as d:
            out=Path(d);(out/'STAGE_A.md').write_text('test-only predeclared recording')
            write_json(out/'fixed_candidates.json',fixed_candidates())
            config=dict(self.config,episodes=32,eval_every_episodes=16)
            write_json(out/'manifest.json',dict(settings=config,source_sha256=source_hashes(),
                native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),
                stage_a_sha256=digest(out/'STAGE_A.md'),candidates_sha256=digest(out/'fixed_candidates.json')))
            s=run_one(2,out);c=curve_for(2,out);folder=folder_for(2,out)
            self.assertEqual(s['comparison_to_issue22']['first_difference'],None)
            self.assertEqual(s['comparison_to_issue22']['training_rows_exact'],32)
            self.assertEqual(s['training']['episodes'],32);self.assertEqual(s['evaluation']['episodes'],3)
            self.assertEqual(s['optimizer_steps'],dict(actor=16,critic=16,entropy=16))
            self.assertEqual(len(c),3)
            env=environment(.75)
            try:
                for role in ('selected','posthoc','final'):
                    agent,meta=load_checkpoint(folder/f'{role}.pt',2,config)
                    m,v=episode(env,agent=agent)
                    self.assertEqual(m['weighted_loss'],s[role]['weighted_loss']);self.assertEqual(m['acoustic_error'],s[role]['acoustic_error'])
                    with np.load(folder/f'{role}.npz') as z:np.testing.assert_array_equal(v['audio'],z['audio'])
                    self.assertEqual(meta['trial'],s[role]['episodes'])
            finally:env.close()


if __name__=='__main__':unittest.main()
