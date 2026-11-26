"""Freeze the single-factor protocol and verify original physics before learning."""
from .result_paths import resolve as resolve_result_path
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

from . import model
from .constant020_env import render,episode
from .issue14 import effective_hyperparameters
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue19_protection import workspace_inventory
from .issue21_data import (ROOT,BASE,PRIOR,OUT,METHODS,WIDTHS,SEEDS,source_hashes,environment,target_data,folder_for)
from .issue21_training import make_agent,network_geometry
from .issue16 import digest


def prepare(before_path=Path('/tmp/birdsong-issue21-before.json'),output=OUT,resume=False):
    import torch
    torch.set_num_threads(1);output=Path(output)
    if output.exists() and not (resume and (output/'protected_before.csv').is_file()
            and (output/'storage_before.json').is_file() and not (output/'manifest.json').exists()
            and not (output/'runs').exists()):raise FileExistsError('Existing study preserved')
    before=json.loads(Path(before_path).read_text())
    for r in before['files']:
        if digest(resolve_result_path(ROOT/r['path']))!=r['sha256']:raise RuntimeError(f'Original file changed: {r["path"]}')
    output.mkdir(parents=True,exist_ok=resume)
    files=before.pop('files')
    if not (output/'protected_before.csv').exists():
        write_csv(output/'protected_before.csv',files);write_json(output/'storage_before.json',before)
    else:assert json.loads((output/'storage_before.json').read_text())==before
    tick=time.perf_counter();original=json.loads((BASE/'manifest.json').read_text())
    prior=json.loads((PRIOR/'manifest.json').read_text())
    for name,h in original['source_sha256'].items():
        if name!='native/model.cpp':assert digest(resolve_result_path(ROOT/name))==h,name
    for name in ('native/model.cpp','model.py','model_env.py','constant020_env.py','issue18_domain_env.py','acoustics.py'):
        assert digest(resolve_result_path(ROOT/name))==prior['source_sha256'][name],name
    assert digest(model.LIBRARY)==prior['native_library_sha256'],'Actual library differs from numerically verified Issue19 build'
    assert digest(BASE/'diagnostics/target.npz')==prior['target_sha256']
    target,controls=target_data();assert controls.tolist()==[4.1e7,3.5e7,3.4e6,3e5,4.9e4]
    # Probe actual reset assignments in an isolated build without modifying the
    # library used by original results or by the new training processes.
    with tempfile.TemporaryDirectory(prefix='issue21-probe-') as directory:
        temp=Path(directory);source=temp/'probe.cpp'
        source.write_text((ROOT/'native/model.cpp').read_text()+'''
extern "C" double issue21_constant(int which) {
 Model m; double c[5]={4.1e7,3.5e7,3.4e6,3e5,4.9e4},s[4]={0,0,0,0};
 std::vector<double> h(m.history_size,0); bird_reset(&m,c,s,h.data(),m.history_size,0);
 return which==0?m.left.params.x0:which==1?m.left.params.D:which==2?m.right.params.alpha:m.right.params.beta;
}
''')
        lib=temp/'probe.so';subprocess.run(['g++','-O3','-std=c++11','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
        probe=ctypes.CDLL(str(lib));probe.issue21_constant.restype=ctypes.c_double
        constants=[probe.issue21_constant(i) for i in range(4)];assert constants==[.02,.01,0.,0.]
    assert np.array_equal(render(controls),target)
    comparisons=[]
    for method,seed in (('TD3',0),('TD3',1),('SAC',0),('SAC',2)):
        with np.load(folder_for(method,32,seed)/'rollout.npz') as z:
            audio=render(z['physical_controls'][0]);assert np.array_equal(audio,z['audio'])
        comparisons.append(dict(method=method,seed=seed,max_waveform_difference=0.))
    config=dict(original['settings'])
    config.update(experiment_id='issue21_caseA_restricted3d_network_width',methods=list(METHODS),
        new_widths=[64,128,256],baseline_width=32,rl_envelope_weights=[1.],checkpoints=[])
    assert config['episodes']==4096 and config['eval_every_episodes']==128
    architectures=[];env=environment()
    try:
        for method in METHODS:
            baseline=json.loads((folder_for(method,32,0)/'summary.json').read_text())['effective_hyperparameters']
            for width in WIDTHS:
                agent=make_agent(method,width,0,env,config)
                effective=json.loads(json.dumps(effective_hyperparameters(agent)));assert dict(effective,net_arch=[32,32])==baseline
                geometry=network_geometry(agent,width)
                if width==32:
                    action=agent.predict(env.reset()[0],deterministic=True)[0]
                    expected=json.loads((folder_for(method,32,0)/'summary.json').read_text())['initial']['actions'][0]
                    assert np.array_equal(action,expected),'Same-width initial actor is not reproduced'
                architectures.append(dict(method=method,width=width,effective_hyperparameters=effective,network_geometry=geometry))
    finally:env.close()
    previous_attempts=json.loads((output/'preflight_attempts.json').read_text()) if (output/'preflight_attempts.json').exists() else []
    write_json(output/'preflight.json',dict(passed=True,model_constants=dict(x0_cm=constants[0],D=constants[1],alpha_r=constants[2],beta_r=constants[3]),
        same_target_exact=True,baseline_waveforms_exact=comparisons,generated_sounds=5,
        architecture_checks=architectures,only_effective_hyperparameter_changed='net_arch',
        source_sha256=digest(__file__),wall_s=time.perf_counter()-tick,
        previous_attempts=previous_attempts,total_generated_sounds=5+sum(r['generated_sounds'] for r in previous_attempts)))
    write_json(output/'manifest.json',dict(issue=21,created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=before['git_head'],settings=config,source_sha256=source_hashes(),
        native_library_sha256=digest(model.LIBRARY),target_sha256=digest(BASE/'diagnostics/target.npz'),
        target_controls=controls.tolist(),baseline_manifest_sha256=digest(BASE/'manifest.json'),
        critic_pool_path=str((PRIOR/'critic_pool.csv').relative_to(ROOT)),critic_pool_sha256=digest(PRIOR/'critic_pool.csv'),
        versions={n:importlib.metadata.version(n) for n in original['versions']},device='cpu',torch_threads_per_worker=1,
        case='A: alpha_r=beta_r=0; constant3D, 0.20s, native dt1e-7, initial state/delay zero',
        variable_indices=[0,2,3],domain_low=[2.5e7,2.5e6,49000.],domain_high=[5e7,4e6,800000.],
        reward='-(E_spec+E_env); envelope_weight=1; one action and terminated=True',
        evaluation_schedule='0,128,...,3968 post previous gradient update; 4096 post final update; 33 independent evaluations',
        endpoints='Final post-final-update deterministic policy; Selected first minimum of 33 independent evaluations',
        only_numerical_factor_changed='Actor and both online/target Critic hidden widths [32,32] -> [64,64]/[128,128]/[256,256]',
        baseline='Issue18 restricted3D lambda1, read only, no baseline relearning or redraw',
        limitations='Same seeds do not imply matching weight initialization or action trajectories across different architectures',
        source_links=dict(issue='https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/21',
            context='https://app.notion.com/p/41f6906b3ff081c3ba69e92cedfafb95'),
        forbidden_scope=['Case B','time-varying control','new DE runs','DE comparison plots','real recordings','noise/reward/LR tuning']))
    print('Prepared 18 runs; only net_arch differs; five full waveforms exact; originals protected',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue21-before.json'))
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--resume-preparation',action='store_true')
    a=p.parse_args();prepare(a.before,a.output,a.resume_preparation)
