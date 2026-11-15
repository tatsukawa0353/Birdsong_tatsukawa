from concurrent.futures import ProcessPoolExecutor
import multiprocessing,json,shutil
from pathlib import Path
import time
import numpy as np
from Deeplearning.issue17 import WORK,run_one,weight_label
from Deeplearning.constant020_env import Constant020Env,episode
from stable_baselines3 import TD3,SAC

if __name__=='__main__':
    smoke=WORK.parent/'smoke4096';smoke.mkdir()
    (smoke/'diagnostics').mkdir()
    for name in ('target.npz','review.json'):
        shutil.copy2(WORK/'diagnostics'/name,smoke/'diagnostics'/name)
    manifest=json.loads((WORK/'manifest.json').read_text())
    manifest['settings']=dict(manifest['settings'],episodes=32,eval_every_episodes=8,checkpoints=[16,32])
    manifest['validation_smoke_only']=True
    (smoke/'manifest.json').write_text(json.dumps(manifest))
    start=time.perf_counter()
    specs=[('TD3',1.),('TD3',.5),('SAC',1.),('SAC',.5),('DE',1.),('random_search',1.)]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        summaries=list(pool.map(run_one,[smoke]*6,['constant_1d']*6,[m for m,w in specs],[0]*6,[w for m,w in specs]))
    with np.load(smoke/'diagnostics/target.npz') as z:target,controls=z['target'].copy(),z['controls'].copy()
    for s in summaries:
        m,w=s['method'],s['envelope_weight'];folder=smoke/'runs/constant_1d'/m/weight_label(w)/'seed_0'
        assert s['training']['episodes']==32
        if m in ('TD3','SAC'):
            env=Constant020Env((0,),target,controls,envelope_weight=w)
            model={'TD3':TD3,'SAC':SAC}[m].load(folder/'agent.zip',device='cpu')
            metrics,_=episode(env,agent=model);env.close()
            assert metrics['acoustic_error']==s['final']['acoustic_error']
            assert s['gradient_updates']==16
            curve=json.loads((folder/'curve.json').read_text());assert [r['episodes'] for r in curve]==[0,8,16,24,32]
            assert [r['gradient_updates'] for r in curve]==[0,0,0,8,16]
            trace=json.loads((folder/'training_trace.json').read_text())
            for r in trace:
                expected=-(r['components']['log_spectral']+w*r['components']['envelope'])
                assert abs(r['reward']-expected)<1e-7
    for m in ('TD3','SAC'):
        a=[json.loads((smoke/'runs/constant_1d'/m/weight_label(w)/'seed_0/training_trace.json').read_text()) for w in (1.,.5)]
        assert [r['action'] for r in a[0][:16]]==[r['action'] for r in a[1][:16]]
    result=dict(passed=True,registered_budget=False,runs=6,training_candidates=192,
                evaluation_sounds=sum(s['evaluation']['episodes'] for s in summaries),
                reload_check_sounds=4,wall_s=time.perf_counter()-start,
                checks=['weighted reward','paired warmup','post-update evaluation','model reload','fixed DE budget','checkpoint serialization'])
    (WORK/'smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
