"""Infrastructure recovery with unchanged seeds/protocol and durable counters."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import multiprocessing
from pathlib import Path
import re
import shutil
import time
from .issue27g_data import OUT,ROOT,METHODS,SEEDS,ready,hashes
from .issue16 import digest
from .issue15_analysis import write_json

def preserve_interruption():
    archive=OUT/'interrupted_attempt'
    if archive.exists():return
    log=Path('/tmp/birdsong-issue27g-training.log');progress={}
    for line in log.read_text().splitlines():
        r=json.loads(line);key=(r['target'],r['dim'],r['method'],r['seed'])
        if r['stage']=='start':progress[key]=0
        elif r['stage']=='progress':progress[key]=r['candidates']
        else:raise RuntimeError('Expected no completed or failed run before infrastructure interruption')
    assert len(progress)==27 and not list((OUT/'runs').glob('*/*/*/*/summary.json'))
    archive.mkdir();shutil.copyfile(log,archive/'stdout.log');shutil.copyfile(OUT/'manifest.json',archive/'manifest.json');shutil.move(str(OUT/'runs'),str(archive/'runs'))
    bounds=[dict(target=t,dim=d,method=m,seed=s,completed_training_lower=n,completed_training_upper=n+512,
        independent_evaluations_lower=1+n//128,independent_evaluations_upper=1+(n+512)//128,in_progress_sounds_upper=1) for (t,d,m,s),n in sorted(progress.items())]
    result=dict(status='interrupted_SIGTERM_exit143',completed_runs=0,started_runs=27,unstarted_runs=27,exact_final_counts_available=False,bounds=bounds,
        completed_training_lower=sum(r['completed_training_lower'] for r in bounds),completed_training_upper=sum(r['completed_training_upper'] for r in bounds),
        evaluation_lower=sum(r['independent_evaluations_lower'] for r in bounds),evaluation_upper=sum(r['independent_evaluations_upper'] for r in bounds),
        in_progress_sounds_upper=27,reason='Termination cause not established; exit143 observed. Last512-step logs bound completed calls; no exact terminal counters survived.',
        recovery='Same54 registered conditions/seeds. First27 repeated from original initialization, remaining27 first execution. No seed replacement, tuning, or extension of4096 effective steps.')
    write_json(archive/'interruption.json',result)
    (OUT/'RECOVERY.md').write_text('''# 実行セッション中断と復旧

最初の27runが途中で実行セッションのSIGTERM/終了143により中断した。数値異常のfailure記録や完了summaryはなく、原因は確定していない。中断前の全モデル/config/ログはinterrupted_attemptへ保存し、削除・成功結果への上書きはしない。次の27runは未開始だった。

同じTarget・seed・物理・報酬・4096step・全ハイパーパラメータを維持して54runを実行する。中断27本は同じ初期seedからの再計算であり、成功seedへの差し替えや精度に基づく延長ではない。途中重みだけからoptimizer/Bufferを復元したとは主張しない。正式な最終54runの会計と、別途発生した中断計算を両方記録する。

中断時の正確な終了stepは保存されていないため、最後の512step進捗と次の進捗の間の範囲で報告する。独立評価と実行中1音/workerも分けて上限を記す。中断計算を0回扱いしたり、合計回数を偽の確定値にしない。

復旧時は元の環境stepを変更せず、完了後に原子的なcounterファイルを書き出す観測用subclassを追加する。実行条件・元コアのhashは変更しない。新ランチャのhashはrecovery_manifestに記録。長い実行セッションを定期的にpollし状態を確認する。
''')

def counted_run(t,d,m,s):
    from . import issue27g_training as training
    from .issue27g_env import CaseBEnv
    root=OUT/f'progress/{t}_{d}D_{m}_s{s}';root.mkdir(parents=True,exist_ok=True)
    class CountedEnv(CaseBEnv):
        def __init__(self,*args,**kwargs):
            self.counter_kind='evaluation' if kwargs.get('record_moments',False) else 'training';super().__init__(*args,**kwargs)
        def step(self,action):
            result=super().step(action)
            dest=root/(self.counter_kind+'.json');temp=dest.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(completed_episodes=self.completed_episodes,native_blocks_completed=self.native_blocks_completed,failures=self.failures,
                last_J=self.last_info['weighted_loss'],last_controls=self.last_info['applied_controls'])))
            temp.replace(dest);return result
    training.CaseBEnv=CountedEnv
    return training.run_one(t,d,m,s)

def run(workers):
    manifest=ready();preserve_interruption();assert not (OUT/'execution.json').exists()
    write_json(OUT/'recovery_manifest.json',dict(source_sha256=digest(Path(__file__)),frozen_core_sha256=hashes(),original_manifest_sha256=digest(OUT/'manifest.json'),
        same_seeds_and_hyperparameters=True,observer_only_change='Call original environment.step then atomically persist completed counters; no RNG/buffer/optimizer/action changes',workers=workers))
    tick=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(counted_run,t,d,m,s) for t,d in manifest['eligible_cases'] for m in METHODS for s in SEEDS]
        for f in as_completed(futures):results.append(f.result())
    write_json(OUT/'execution.json',dict(runs=len(results),planned_runs=54,complete_runs=sum(s['status']=='complete' for s in results),failed_runs=sum(s['status']=='failed' for s in results),
        training_candidates=sum(s['training']['episodes'] for s in results),independent_policy_evaluations=sum(s['independent_policy_evaluations'] for s in results),
        wall_s=time.perf_counter()-tick,workers=workers,source_sha256=hashes(),interrupted_attempt_separate='interrupted_attempt/interruption.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=27);a=p.parse_args();run(a.workers)
