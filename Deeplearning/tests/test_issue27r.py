"""Revised Stage A checks; no new sounds, method smoke tests or learning."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
from Deeplearning.issue27r_data import ROOT,OUT,OLD,TARGET,GRID_E,GRID_P,require_target,ready,target_data
from Deeplearning.issue27r_native import snapshot,PROBE
from Deeplearning.issue27r_prepare import select
from Deeplearning.model import BirdModel,LIBRARY
from Deeplearning.issue16 import digest
from Deeplearning.issue15_analysis import write_json


class RevisedPhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.records=json.loads((OUT/'preflight/candidates.json').read_text())

    def test_all9_prespecified_points_and_effective_B_coefficients(self):
        self.assertEqual(len(self.records),9)
        self.assertEqual({(r['epsilon_l'],r['ps']) for r in self.records},{(e,p) for e in GRID_E for p in GRID_P})
        for r in self.records:
            self.assertFalse(r['failed']);self.assertFalse(r['generated_silent']);c=r['native_coefficients']
            self.assertEqual([c[k] for k in ('alpha_l','beta_l','alpha_r','beta_r')],[1.2e6,1500,1.2e6,1500])
            self.assertEqual(c['f0_r'],1e7);self.assertEqual(c['epsilon_r'],3.5e7);self.assertEqual(c['history_max_abs'],0)

    def test_each_ratio_recomputes_and_all_primary_proxies_fail(self):
        for r in self.records:
            with np.load(OUT/r['raw_npz']) as z:
                a=z['audio'];self.assertEqual(a.shape,(10000,));self.assertEqual(a.dtype,np.float64);self.assertTrue(np.isfinite(a).all())
                np.testing.assert_array_equal(z['controls'],[r['epsilon_l'],3.5e7,r['ps'],3e5,1e7])
                ratio=float(a[-1000:].std()/abs(a).max());x=np.arange(1000)/50000;residual=a[-1000:]-np.polyval(np.polyfit(x,a[-1000:],1),x)
                self.assertEqual(ratio,r['late_ac_ratio']);self.assertAlmostEqual(float(residual.std()/abs(a).max()),r['detrended_ac_ratio'],places=14)
                self.assertLess(ratio,.05);self.assertFalse(r['eligible'])

    def test_native_step_counts_motion_and_right_contribution(self):
        for r in self.records:
            with np.load(OUT/r['raw_npz']) as z:
                q=z['diagnostic_moments'];p=z['diagnostic_samples']
                self.assertEqual(q.shape,(20,50));self.assertEqual(p.shape,(10000,12))
                np.testing.assert_array_equal(q[:,49],np.full(20,100000))
                np.testing.assert_array_equal(z['motion_mean'],q[:,:4]);np.testing.assert_array_equal(z['motion_std'],q[:,12:16])
                self.assertNotEqual(q[-1,5],0);self.assertNotEqual(q[-1,9],0);self.assertGreater(q[-1,14],0);self.assertGreater(q[-1,15],0)
                self.assertLess(q[:,48].max(),1e-7)
                np.testing.assert_allclose(p[:,7],p[:,4]+p[:,5]+p[:,6],rtol=0,atol=1e-7)

    def test_live_memory_B_and_right_force_override_without_integration(self):
        model=BirdModel('B')
        try:
            model.reset([3e7,3.5e7,3e6,3e5,49000]);r=snapshot(model,reset=True)
            self.assertEqual(r['alpha_r'],1.2e6);self.assertEqual(r['beta_r'],1500);self.assertEqual(r['f0_r'],1e7)
            self.assertEqual(model.state[0],0)
        finally:model.close()

    def test_previous_c21_condition_is_exactly_reproduced_read_only(self):
        with np.load(OUT/'preflight/candidates/c21.npz') as new,np.load(OLD/'preflight/oracle.npz') as old:
            for key in ('audio','controls','physical_controls','state_end_blocks','motion_mean','motion_std','diagnostic_samples','diagnostic_moments'):
                np.testing.assert_array_equal(new[key],old[key])

    def test_no_target_or_training_can_be_started_after_failed_grid(self):
        self.assertFalse(select());self.assertFalse(TARGET.exists());self.assertFalse((OUT/'runs').exists())
        for f in (require_target,ready,target_data):
            with self.assertRaisesRegex(RuntimeError,'All nine'):f()

    def test_executed_scan_and_original_binary_source_unchanged(self):
        b=json.loads((OUT/'native/build.json').read_text());self.assertTrue(b['unchanged_source_build_binary_exact'])
        self.assertEqual(digest(LIBRARY),b['loaded_library_sha256']);self.assertEqual(digest(OUT/'native/libsource_reference.so'),digest(LIBRARY));self.assertEqual(digest(PROBE),b['probe_sha256'])
        self.assertEqual(digest(ROOT/'native/model.cpp'),b['native_source_sha256']);self.assertEqual(digest(ROOT/'issue27_probe.cpp'),b['probe_source_sha256'])
        def scan_ast(path):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='scan'),include_attributes=False)
        self.assertEqual(scan_ast(ROOT/'issue27r_prepare.py'),scan_ast(OUT/'executed_sources/issue27r_prepare.py'))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RevisedPhysicalTests))
    if result.wasSuccessful():write_json(OUT/'preflight/tests.json',dict(passed=True,tests=result.testsRun,additional_native_sounds=0,training_runs=0,
        method_smoke_tests='not started: no voiced Target',source_sha256=digest(Path(__file__)),command='python -m Deeplearning.tests.test_issue27r'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
