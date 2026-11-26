"""Verify reuse, installed SB3 scheduler behavior and initial Actor-only intervention."""
from .result_paths import resolve as resolve_result_path
import argparse
from datetime import datetime,timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .issue14 import learner_signature
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest
from .issue21_training import make_agent as original_agent
from .issue24_agent import make_agent,effective_difference,assert_lrs,ACTOR_RATES,BASE_LR
from .issue24_data import ROOT,BASE,OUT,TARGET,SEEDS,source_hashes,sb3_hashes,folder_for,environment


def prepare(before_path=Path('/tmp/birdsong-issue24-before.json'),output=OUT):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.logger import Logger
    from .model import LIBRARY
    torch.set_num_threads(1);output=Path(output);tick=time.perf_counter()
    if output.exists():raise FileExistsError('Existing study preserved')
    before=json.loads(Path(before_path).read_text())
    for r in before['files']:assert digest(resolve_result_path(ROOT/r['path']))==r['sha256'],r['path']
    previous=json.loads((BASE/'manifest.json').read_text())
    for p,h in previous['source_sha256'].items():assert digest(resolve_result_path(ROOT/p))==h,p
    assert digest(TARGET)==previous['target_sha256'] and digest(LIBRARY)==previous['native_library_sha256']
    assert json.loads((BASE/'audit.json').read_text())['passed']
    versions={p:importlib.metadata.version(p) for p in previous['versions']};assert versions==previous['versions']
    output.mkdir(parents=True);write_csv(output/'protected_before.csv',before.pop('files'));write_json(output/'storage_before.json',before)
    config=dict(previous['settings']);config.update(experiment_id='issue24_sac128_actor_lr',actor_learning_rates=list(ACTOR_RATES))
    inspections=[]
    for seed in SEEDS:
        old=json.loads((folder_for(BASE_LR,seed)/'summary.json').read_text())
        for p,h in old['artifact_sha256'].items():assert digest(folder_for(BASE_LR,seed)/p)==h,p
        assert old['comparison_to_issue22']['first_difference'] is None and old['training']['episodes']==4096
        env=environment(.75)
        try:
            original=original_agent('SAC',128,seed,env,config)
            initial=learner_signature(original)
            groups=lambda a:{n:[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups] for n,opt in
                (('actor',a.actor.optimizer),('critic',a.critic.optimizer),('entropy',a.ent_coef_optimizer))}
            reference_groups=groups(original)
            for rate in ACTOR_RATES:
                agent=make_agent(rate,seed,env,config);effective_difference(agent,old['effective_hyperparameters'],rate)
                sig=learner_signature(agent)
                assert sig['policy_sha256']==initial['policy_sha256']==old['initial']['learner_before']['policy_sha256']
                assert sig['rng_sha256']==initial['rng_sha256'] and sig['entropy_sha256']==initial['entropy_sha256']
                actual_groups=groups(agent);normalized=json.loads(json.dumps(actual_groups));normalized['actor'][0]['lr']=BASE_LR
                assert normalized==json.loads(json.dumps(reference_groups))
                agent.set_logger(Logger(None,[]));history=[]
                for progress in (1.,.5,0.):
                    agent._current_progress_remaining=progress
                    agent._update_learning_rate([agent.actor.optimizer,agent.critic.optimizer,agent.ent_coef_optimizer])
                    history.append(dict(progress_remaining=progress,optimizer_lrs=assert_lrs(agent,rate)))
                inspections.append(dict(seed=seed,rate=rate,initial_policy_exact=True,initial_global_rng_exact=True,
                    only_actor_optimizer_lr_differs=True,schedule_checks=history))
        finally:env.close()
    # Keep the exact installed primary implementation rather than rely on API lore.
    implementation='SAC.train\n'+inspect.getsource(SAC.train)+'\nBaseAlgorithm._update_learning_rate\n'+inspect.getsource(BaseAlgorithm._update_learning_rate)
    (output/'sb3_learning_rate_source.txt').write_text(implementation)
    local_refs={p:digest(BASE/p) for p in ('local_sensitivity.csv','local_gradients.csv','local_execution.json')}
    write_json(output/'preflight.json',dict(passed=True,generated_sounds=0,initial_checks=inspections,
        stock_SAC_train_inherited_unchanged=True,baseline_reuse_exact=True,baseline_runs=3,new_baseline_research_runs=0,
        scheduler='Call stock scheduler at shared3e-4, then apply constant Actor LR; verify all three optimizers before every optimizer.step',
        source_sha256=digest(__file__),wall_s=time.perf_counter()-tick))
    write_json(output/'manifest.json',dict(issue=24,created_utc=datetime.now(timezone.utc).isoformat(),git_head=before['git_head'],
        settings=config,source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),versions=versions,
        baseline_manifest_sha256=digest(BASE/'manifest.json'),candidates_sha256=digest(BASE/'fixed_candidates.json'),local_reference_sha256=local_refs,
        sb3_source_sha256=digest(output/'sb3_learning_rate_source.txt'),sb3_implementation_sha256=sb3_hashes(),research_runs=6,baseline_new_runs=0,
        actor_rates=list(ACTOR_RATES),critic_lr=BASE_LR,entropy_optimizer_lr=BASE_LR,
        selected='First minJ among33 evaluations',posthoc='First min common E among33; separate posthoc selection',final='4096 post-final-update',
        additional_local_sounds=0,source_link='https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/24'))
    print('Preflight passed: baseline reuse, initial weights/RNG and Actor-only LR; stock scheduler tested at progress1/.5/0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue24-before.json'));a=p.parse_args();prepare(a.before)
