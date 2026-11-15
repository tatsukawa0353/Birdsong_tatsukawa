"""Physical duration, reset, reward, failure and auto-reset trace invariants."""
import unittest
from unittest.mock import patch
import numpy as np

from Deeplearning.constant020_env import Constant020Env,INDICES,episode,render
from Deeplearning.model import LOW,HIGH


class Constant020Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controls=np.array([4.1e7,3.5e7,3.4e6,300000.,49000.])
        cls.target=render(cls.controls)

    def test_each_dimension_oracle_duration_constant_controls_and_reset(self):
        for indices in INDICES.values():
            env=Constant020Env(indices,self.target,self.controls)
            try:
                obs,_=env.reset(seed=0)
                self.assertTrue(env.observation_space.contains(obs))
                action=env.encode(self.controls)
                m,s=episode(env,action=action)
                self.assertLess(m['acoustic_error'],1e-9)
                self.assertEqual(s['audio'].shape,(10000,))
                self.assertEqual(env.horizon,1)
                self.assertEqual(env.native_blocks,20)
                np.testing.assert_allclose(s['physical_controls'],np.tile(self.controls,(20,1)),rtol=1e-15)
                episode(env,action=np.zeros(len(indices)))
                _,again=episode(env,action=action)
                np.testing.assert_array_equal(s['audio'],again['audio'])
                self.assertEqual(env.native_blocks_completed,60)
                self.assertEqual(env.rewards,[-m['acoustic_error']])
                for end,bounds in ((-1,LOW),(1,HIGH)):
                    decoded=env.decode(np.full(len(indices),end))
                    np.testing.assert_array_equal(decoded[list(indices)],bounds[list(indices)])
            finally:env.close()

    def test_rejects_old_duration_and_invalid_actions(self):
        with self.assertRaises(ValueError):
            Constant020Env((0,),np.zeros(17500),self.controls)
        env=Constant020Env((0,),self.target,self.controls)
        try:
            env.reset()
            for action in ([float('nan')],[0,0]):
                with self.assertRaises(ValueError):env.step(action)
        finally:env.close()

    def test_weight_changes_reward_without_changing_waveform_or_common_error(self):
        outputs=[]
        for weight in (1.,.5):
            env=Constant020Env((0,2),self.target,self.controls,envelope_weight=weight)
            try:
                metrics,values=episode(env,action=[0.,0.])
                components=metrics['acoustic_components']
                self.assertEqual(metrics['acoustic_error'],components['log_spectral']+components['envelope'])
                self.assertEqual(env.rewards[0],-(components['log_spectral']+weight*components['envelope']))
                self.assertEqual(metrics['envelope_weight'],weight)
                outputs.append((metrics,values))
            finally:env.close()
        self.assertEqual(outputs[0][0]['acoustic_error'],outputs[1][0]['acoustic_error'])
        self.assertNotEqual(outputs[0][0]['weighted_loss'],outputs[1][0]['weighted_loss'])
        np.testing.assert_array_equal(outputs[0][1]['audio'],outputs[1][1]['audio'])

    def test_failure_retains_candidate_budget_without_scoring_short_prefix(self):
        env=Constant020Env((0,),self.target,self.controls)
        try:
            env.reset()
            with patch.object(env.model,'advance',side_effect=FloatingPointError('injected')):
                obs,reward,done,truncated,info=env.step([0])
            self.assertEqual(reward,-1e6)
            self.assertTrue(done and info['failed'])
            self.assertEqual(env.native_blocks,1)
            self.assertEqual(env.native_blocks_completed,0)
            self.assertEqual(env.completed_episodes,1)
            self.assertEqual(np.concatenate(env.audio).shape,(10000,))
            self.assertTrue(np.isfinite(obs).all())
            with self.assertRaises(RuntimeError):env.step([0])
            m,_=episode(env,action=env.encode(self.controls))
            self.assertEqual(m['acoustic_error'],0)
        finally:env.close()

    def test_sb3_auto_reset_retains_terminal_candidate_metadata_and_updates(self):
        import torch
        from stable_baselines3 import SAC,TD3
        from stable_baselines3.common.callbacks import BaseCallback
        torch.set_num_threads(1)
        for cls in (TD3,SAC):
            env=Constant020Env((0,),self.target,self.controls)
            try:
                agent=cls('MlpPolicy',env,seed=0,device='cpu',learning_starts=16,
                          buffer_size=128,batch_size=8,train_freq=1,gradient_steps=1,
                          gamma=1,policy_kwargs=dict(net_arch=[32,32]),verbose=0)
                rows=[]
                class Capture(BaseCallback):
                    def _on_step(self):
                        rows.append(self.locals['infos'][0]);return True
                agent.learn(24,callback=Capture())
                self.assertEqual(len(rows),24)
                self.assertEqual(env.completed_episodes,24)
                self.assertEqual(agent._n_updates,8)
                self.assertTrue(all(len(r['applied_action'])==1 and len(r['applied_controls'])==5 for r in rows))
                self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy.parameters()))
            finally:env.close()


if __name__=='__main__':unittest.main()
