"""Validate coefficient sums, actual library and reward-only comparison before fitting."""
import argparse
import ctypes
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from .issue14 import effective_hyperparameters,learner_signature
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .constant020_env import render,episode
from .issue21_training import make_agent,network_geometry
from .issue22_data import (ROOT,BASE,OUT,TARGET,POOL,WEIGHTS,SEEDS,coefficients,objective,
    source_hashes,target_data,environment,folder_for,policy_statistics,ENTROPY_SEED,ENTROPY_SAMPLES)
from .issue16 import digest
from .model import LIBRARY


def prepare(before_path=Path('/tmp/birdsong-issue22-before.json'),output=OUT):
    import torch
    torch.set_num_threads(1);output=Path(output)
    if output.exists():raise FileExistsError('Existing results preserved')
    before=json.loads(Path(before_path).read_text())
    for r in before['files']:assert digest(ROOT/r['path'])==r['sha256'],r['path']
    output.mkdir(parents=True);write_csv(output/'protected_before.csv',before.pop('files'));write_json(output/'storage_before.json',before)
    tick=time.perf_counter();previous=json.loads((BASE/'manifest.json').read_text())
    for name,h in previous['source_sha256'].items():assert digest(ROOT/name)==h,name
    assert digest(LIBRARY)==previous['native_library_sha256'] and digest(TARGET)==previous['target_sha256']
    versions={p:importlib.metadata.version(p) for p in previous['versions']};assert versions==previous['versions']
    config=dict(previous['settings']);config.update(experiment_id='issue22_sac128_sum_one_reward',methods=['SAC'],net_arch=[128,128],
        spectral_weights=list(WEIGHTS),checkpoints=[])
    for key in ('new_widths','baseline_width','rl_envelope_weights'):config.pop(key,None)
    # Probe the compiled reset assignments without changing the live library.
    with tempfile.TemporaryDirectory(prefix='issue22-probe-') as directory:
        temp=Path(directory);source=temp/'probe.cpp';source.write_text((ROOT/'native/model.cpp').read_text()+'''
extern "C" double issue22_constant(int which){Model m;double c[5]={4.1e7,3.5e7,3.4e6,3e5,4.9e4},s[4]={0,0,0,0};std::vector<double> h(m.history_size,0);bird_reset(&m,c,s,h.data(),m.history_size,0);return which==0?m.left.params.x0:which==1?m.left.params.D:which==2?m.right.params.alpha:m.right.params.beta;}
''');lib=temp/'probe.so'
        subprocess.run(['g++','-O3','-std=c++11','-fPIC','-shared',str(source),'-o',str(lib)],check=True)
        probe=ctypes.CDLL(str(lib));probe.issue22_constant.restype=ctypes.c_double
        constants=[probe.issue22_constant(i) for i in range(4)];assert constants==[.02,.01,0.,0.]
    values=np.random.default_rng(22).uniform(0,3,(1000,2));coefficient_checks=[]
    for weight in WEIGHTS:
        a,b=coefficients(weight);assert a+b==1.
        losses=np.array([objective(s,e,weight) for s,e in values]);np.testing.assert_allclose(losses,a*values[:,0]+b*values[:,1],rtol=0,atol=0)
        if weight==.5:np.testing.assert_array_equal(losses,values.sum(1)/2)
        coefficient_checks.append(dict(lambda_spectral=weight,spectral=a,envelope=b,sum=a+b,extra_multiplier=1.))
    target,controls=target_data();assert controls.tolist()==[4.1e7,3.5e7,3.4e6,3e5,4.9e4]
    np.testing.assert_array_equal(render(controls),target)
    comparisons=[];policies=[]
    for weight,seed in zip(WEIGHTS,SEEDS):
        env=environment(weight)
        try:
            prior=json.loads((folder_for('legacy_sum',seed)/'summary.json').read_text())
            metrics,wave=episode(env,action=prior['final']['actions'][0])
            with np.load(folder_for('legacy_sum',seed)/'final.npz') as z:np.testing.assert_array_equal(wave['audio'],z['audio'])
            assert metrics['acoustic_error']==prior['final']['acoustic_error']
            assert metrics['weighted_loss']==objective(metrics['acoustic_components']['log_spectral'],metrics['acoustic_components']['envelope'],weight)
            agent=make_agent('SAC',128,seed,env,config)
            effective=json.loads(json.dumps(effective_hyperparameters(agent)));assert effective==prior['effective_hyperparameters']
            geometry=network_geometry(agent,128);before_stats=learner_signature(agent);stats=policy_statistics(agent)
            assert before_stats==learner_signature(agent) and before_stats['policy_sha256']==prior['initial']['learner_before']['policy_sha256']
            comparisons.append(dict(weight=weight,seed=seed,waveform_max_difference=0.,common_error_difference=0.,J=metrics['weighted_loss']))
            policies.append(dict(seed=seed,network_geometry=geometry,effective_hyperparameters=effective,initial_statistics=stats))
        finally:env.close()
    write_json(output/'preflight.json',dict(passed=True,generated_sounds=4,coefficient_checks=coefficient_checks,
        lambda0p5_is_exact_half_old_sum=True,common_waveforms_unchanged=comparisons,model_constants=constants,
        initial_policies_match=True,statistics_preserve_rng_and_learner=True,policies=policies,
        wall_s=time.perf_counter()-tick,source_sha256=digest(__file__)))
    write_json(output/'manifest.json',dict(issue=22,created_utc=datetime.now(timezone.utc).isoformat(),git_head=before['git_head'],
        settings=config,source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),
        target_sha256=digest(TARGET),target_controls=controls.tolist(),critic_pool_sha256=digest(POOL),
        baseline_manifest_sha256=digest(BASE/'manifest.json'),versions=versions,device='cpu',torch_threads=1,
        reward='-(lambda E_spec + (1-lambda) E_env); sum of coefficients=1; no multiplier',
        coefficient_conditions=coefficient_checks,legacy_reference='Issue21 SAC128: -(E_spec+E_env), coefficient sum2; read only',
        final='Post-final-update deterministic policy at candidate4096',selected='First minimum J_lambda among33 independent evaluations',
        posthoc='First minimum common E_total among33 evaluations; diagnostic only, no extra saved policy/return-to-best',
        selected_minus_final='E_total(Selected minJ) - E_total(Final); negative means final common error is worse',
        entropy_estimator=dict(samples=ENTROPY_SAMPLES,seed=ENTROPY_SEED,distribution='SB3 diagonal Gaussian + tanh, epsilon=1e-6',
            sampling='Fixed local NumPy RNG, same standard-normal samples across evaluations; no global RNG or learning mutation',
            limitation='Monte Carlo estimate of continuous squashed entropy may be negative; conditional action std, not noise hyperparameter'),
        failure_policy='Preserve existing numerical-failure penalty1e6; retain candidate, no retries/exclusions',
        scope='Case A constant3D only; same target/domain/reset/learners as Issue21; reward definition only',
        source_link='https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/22'))
    print('Preflight passed: coefficient sums1, lambda.5 exactly half old sum, four sounds exact, SAC128 settings unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue22-before.json'))
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();prepare(a.before,a.output)
