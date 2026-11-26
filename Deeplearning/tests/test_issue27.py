"""Case B read-only physics checks; never integrate/train past the failed gate."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from Deeplearning.issue27_data import ROOT,OUT,BASE,CONTROLS,TARGET
from Deeplearning.issue27_env import CaseBEnv
from Deeplearning.issue27_native import snapshot
from Deeplearning.model import BirdModel,LIBRARY,effective_controls
from Deeplearning.issue16 import digest
from Deeplearning.issue14 import effective_hyperparameters,learner_signature
from Deeplearning.issue21_training import make_agent,network_geometry
from Deeplearning.issue25_agent import assert_lrs
from Deeplearning.issue15_analysis import write_json


class CaseBPreflightTests(unittest.TestCase):
    initial_checks=[]
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_live_B_coefficients_and_forced_right_controls(self):
        for condition in ('A','B'):
            model=BirdModel(condition)
            try:
                c=CONTROLS.copy();c[4]=49000;model.reset(c);s=snapshot(model,expect_B=condition=='B',reset=True)
                self.assertEqual(s['f0_r'],1e7 if condition=='B' else 49000)
                self.assertEqual((s['alpha_r'],s['beta_r']),(1.2e6,1500.) if condition=='B' else (0.,0.))
                self.assertEqual(s['history_max_abs'],0);self.assertEqual(model.state[0],0)
            finally:model.close()

    def test_3D_mapping_fixed_right_and_zero_reset(self):
        env=CaseBEnv()
        try:
            for a in (np.full(3,-1.),np.zeros(3),np.ones(3),np.array([.28,.2,-.33])):
                c=env.decode(a);self.assertEqual(c[1],3.5e7);self.assertEqual(c[4],1e7)
                np.testing.assert_allclose(env.encode(c),a,rtol=0,atol=3e-16)
                self.assertTrue(49000<=c[3]<=800000)
            for seed in (0,1,2):
                obs,_=env.reset(seed=seed);np.testing.assert_array_equal(obs,[-1,0,0,0,0,0,0,0]);self.assertEqual(env.check()['f0_r'],1e7)
                self.assertEqual(env.native_blocks,0)
        finally:env.close()

    def test_failed_gate_prevents_any_episode_or_native_advance(self):
        self.assertFalse(json.loads((OUT/'preflight/physics.json').read_text())['passed'])
        env=CaseBEnv()
        try:
            env.reset(seed=0);before=env.model.state.copy()
            with self.assertRaisesRegex(RuntimeError,'preflight failed'):env.step(np.zeros(3))
            np.testing.assert_array_equal(before,env.model.state)
            self.assertEqual(env.native_blocks,0);self.assertEqual(env.completed_episodes,0)
        finally:env.close()

    def test_saved_oracle_is_exact_B_and_new_Target_is_not_A(self):
        with np.load(TARGET) as t,np.load(OUT/'preflight/oracle.npz') as r:
            for name in ('audio','controls','physical_controls','motion_mean','motion_std','state_end_blocks'):
                np.testing.assert_array_equal(t[name],r[name])
            self.assertEqual(t['audio'].dtype,np.float64);self.assertEqual(t['audio'].shape,(10000,))
            np.testing.assert_array_equal(t['controls'],CONTROLS)
            ratio=float(t['audio'][-1000:].std()/abs(t['audio']).max());self.assertLess(ratio,.05)
        self.assertNotEqual(digest(TARGET),digest(ROOT/'results/phase1_3d_domain_diagnostics/diagnostics/target.npz'))

    def test_saved_right_contribution_not_removed_or_forced_static(self):
        with np.load(OUT/'preflight/oracle.npz') as b,np.load(OUT/'preflight/same_controls_caseA.npz') as a:
            bm=b['diagnostic_moments'];am=a['diagnostic_moments']
            self.assertGreater(abs(bm[-1,5]),1);self.assertGreater(bm[-1,14],0);self.assertGreater(bm[-1,15],0)
            np.testing.assert_array_equal(am[:,5],np.zeros(20));self.assertLess(bm[:,48].max(),1e-7)
            self.assertFalse(np.array_equal(b['audio'],a['audio']))

    def test_actual_library_exactly_rebuilds_from_current_source(self):
        record=json.loads((OUT/'native/build.json').read_text())
        self.assertTrue(record['unchanged_source_build_binary_exact'])
        self.assertEqual(digest(LIBRARY),record['loaded_library_sha256'])
        self.assertEqual(digest(OUT/'native/libsource_reference.so'),digest(LIBRARY))

    def test_all3_initial_SAC_settings_and_seeds_without_learning(self):
        config=json.loads((BASE/'manifest.json').read_text())['settings'];hashes=[]
        for seed in (0,1,2):
            env=CaseBEnv()
            try:
                agent=make_agent('SAC',128,seed,env,config);baseline=json.loads((BASE/f'runs/seed_{seed}/summary.json').read_text())
                self.assertEqual(json.loads(json.dumps(effective_hyperparameters(agent))),baseline['effective_hyperparameters']);assert_lrs(agent,3e-4)
                h=learner_signature(agent)['policy_sha256'];self.assertEqual(h,baseline['initial']['learner_before']['policy_sha256']);hashes.append(h)
                network_geometry(agent,128);self.assertEqual(agent.num_timesteps,0);self.assertEqual(agent._n_updates,0);self.assertEqual(agent.replay_buffer.pos,0)
                self.initial_checks.append(dict(seed=seed,model='B',effective_settings_exact=True,initial_policy_exact=True,training_candidates=0,gradient_updates=0))
            finally:env.close()
        self.assertEqual(len(set(hashes)),3)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CaseBPreflightTests))
    if result.wasSuccessful():write_json(OUT/'preflight/tests.json',dict(passed=True,tests=result.testsRun,generated_sounds=0,training_candidates=0,gradient_updates=0,
        source_sha256=digest(Path(__file__)),initialization_checks=CaseBPreflightTests.initial_checks,
        replay_update_tests='not run: physical Target gate failed before learning',command='python -m Deeplearning.tests.test_issue27'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
