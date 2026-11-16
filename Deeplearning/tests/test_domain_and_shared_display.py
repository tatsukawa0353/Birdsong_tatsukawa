import unittest
import numpy as np
from Deeplearning.acoustic_views import relative_db,rms_envelope
from Deeplearning.acoustics import distance
from Deeplearning.issue18_domain_env import DomainEnv
from Deeplearning.constant020_env import Constant020Env,episode
from Deeplearning.issue17 import DEST,load_ready


class SharedDisplayTests(unittest.TestCase):
    def test_half_amplitude_stays_six_db_quieter(self):
        power=np.array([[1.,.1],[.01,.001]])
        np.testing.assert_allclose(relative_db(power*.25,1.)-relative_db(power,1.),-6.020599913279624)

    def test_rms_difference_is_existing_envelope_loss(self):
        t=np.arange(10000)/50000;a=np.sin(2*np.pi*1000*t);b=.4*a
        expected=np.mean(abs(rms_envelope(a,1.)-rms_envelope(b,1.)))
        self.assertEqual(expected,distance(a,b,50000,scale=1.)['envelope'])


class DomainTests(unittest.TestCase):
    def test_mapping_changes_only_f0_and_keeps_same_physical_reward(self):
        _,target,controls=load_ready(DEST)
        broad=Constant020Env((0,2,3),target,controls);narrow=DomainEnv((0,2,3),target,controls,f0_high=800000.)
        try:
            np.testing.assert_array_equal(broad.decode(np.zeros(3))[[0,1,2,4]],narrow.decode(np.zeros(3))[[0,1,2,4]])
            self.assertEqual(narrow.decode(np.ones(3))[3],800000.)
            for f in (300000.,400000.):
                c=controls.copy();c[3]=f
                a,_=episode(broad,action=broad.encode(c));b,_=episode(narrow,action=narrow.encode(c))
                self.assertAlmostEqual(a['weighted_loss'],b['weighted_loss'],places=9)
                self.assertEqual(a['sustained_voiced'],b['sustained_voiced'])
        finally:broad.close();narrow.close()
