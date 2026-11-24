"""Stage A: audit immutable data, independent LRs and budgets before any full run."""
import argparse
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from .issue25_data import (ROOT,BASE,PREVIOUS,OUT,TARGET,SEEDS,RATES,OFFLINE_UPDATES,OFFLINE_EVALS,
    source_hashes,sb3_hashes,folder_for,environment,pack)
from .issue25_agent import BASE_LR,CRITIC_RATES,LIMITS,make_agent,effective_difference,assert_lrs
from .issue21_training import make_agent as original_agent
from .issue14 import learner_signature
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def build_data(output,updates=OFFLINE_UPDATES):
    output=Path(output);points=[];arrays={};source_files={}
    for split in ('train','test'):
        subset=[]
        for seed in SEEDS:
            if split=='train':
                path=folder_for(BASE_LR,seed)/'training_trace.json.gz';rows=read_record(path.with_suffix(''))
                from .issue22_data import folder_for as previous_folder
                assert rows==read_record(previous_folder(.75,seed)/'training_trace.json')
            else:path=folder_for(BASE_LR,seed)/'curve.json.gz';rows=read_record(path.with_suffix(''))
            source_files[str(path.relative_to(ROOT))]=digest(path)
            for p in rows:
                a=np.asarray(p['action'] if split=='train' else p['actions'][0],np.float32)
                scaled=(2.*((a-(-1.))/(1.-(-1.)))-1.).astype(np.float32)
                comp=p['acoustic_components'];reward=-p['weighted_loss'];assert not p['failed']
                assert p['weighted_loss']==.75*comp['log_spectral']+.25*comp['envelope']
                if split=='train':assert p['reward']==float(np.float32(reward))
                identity=dict(state=[-1.,0.,0.,0.,0.,0.,0.,0.],raw_action=a.tolist(),critic_action=scaled.tolist(),
                    reward=reward,replay_reward=float(np.float32(reward)),E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'])
                point=dict(index=len(subset),split=split,source_seed=seed,source_trial=p['episodes'],source_path=str(path.relative_to(ROOT)),
                    point_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                    raw_action_sha256=hashlib.sha256(a.tobytes()).hexdigest(),critic_input_sha256=hashlib.sha256(scaled.tobytes()).hexdigest(),**identity)
                subset.append(point);points.append(point)
        arrays[split+'_actions']=np.array([p['critic_action'] for p in subset],np.float32)
        arrays[split+'_raw_actions']=np.array([p['raw_action'] for p in subset],np.float32)
        arrays[split+'_rewards']=np.array([p['reward'] for p in subset],np.float64)
        arrays[split+'_source_seeds']=np.array([p['source_seed'] for p in subset],np.int32)
    assert arrays['train_actions'].shape==(12288,3) and arrays['test_actions'].shape==(99,3)
    candidates=json.loads((BASE/'fixed_candidates.json').read_text())
    np.testing.assert_array_equal(arrays['test_raw_actions'],[p['action'] for p in candidates]);np.testing.assert_array_equal(arrays['test_rewards'],[p['reward'] for p in candidates])
    train=[p for p in points if p['split']=='train'];test=[p for p in points if p['split']=='test']
    raw_train={p['raw_action_sha256'] for p in train};feature_train={p['critic_input_sha256'] for p in train}
    distances=np.linalg.norm((arrays['test_actions'][:,None,:].astype(float)-arrays['train_actions'][None,:,:])/2,axis=2)
    nearest=[]
    for i,p in enumerate(test):
        d=distances[i];j=int(np.argmin(d));nearest.append(dict(source_seed=p['source_seed'],source_trial=p['source_trial'],
            nearest_train_index=j,nearest_train_seed=train[j]['source_seed'],nearest_train_trial=train[j]['source_trial'],
            nearest_range_fraction_l2=float(d[j]),exact_raw_overlap=p['raw_action_sha256'] in raw_train,
            exact_critic_input_overlap=p['critic_input_sha256'] in feature_train,
            **{f'within_{radius:g}':int((d<=radius).sum()) for radius in (.001,.005,.01,.05)}))
    train_nearest=cKDTree(arrays['train_actions'].astype(float)/2).query(arrays['train_actions'].astype(float)/2,k=2)[0][:,1]
    threshold=float(np.quantile(-arrays['test_rewards'],.1))
    statistics=dict(train_points=12288,holdout_points=99,source_seeds={s:4096 for s in SEEDS},holdout_per_seed=33,
        duplicate_train_raw=12288-len(raw_train),duplicate_train_critic_inputs=12288-len(feature_train),
        duplicate_holdout_raw=99-len({p['raw_action_sha256'] for p in test}),
        train_holdout_exact_raw_overlap=sum(p['raw_action_sha256'] in raw_train for p in test),
        train_holdout_exact_feature_overlap=sum(p['critic_input_sha256'] in feature_train for p in test),
        nearest_holdout_distance_quantiles={str(q):float(np.quantile(distances.min(1),q)) for q in (0,.25,.5,.75,1)},
        nearest_train_distance_quantiles={str(q):float(np.quantile(train_nearest,q)) for q in (0,.25,.5,.75,1)},
        lowJ_holdout_decile_cutoff=threshold,train_at_or_below_holdout_lowJ_cutoff=int((-arrays['train_rewards']<=threshold).sum()),
        train_E_le_0p10_by_seed={s:sum(p['E_total']<=.1 for p in train if p['source_seed']==s) for s in SEEDS},
        source_files=source_files,training_duplicate_family='Issue23 exactly reproduces Issue22 lambda.75; use Issue23 only, never pool both copies',
        leakage_limits='Holdout never sampled for fitting/early stopping; same target, correlated trajectories and near neighbors remain; not IID/new-target generalization',
        input='constant reset observation8 + normalized SB3 float32 action3, no standardization/reweighting',
        source_sha256=digest(__file__))
    np.savez_compressed(output/'offline_data.npz',**arrays)
    batches={f'seed_{s}':np.random.default_rng(250025+s).integers(0,12288,size=(updates,32),dtype=np.int32) for s in SEEDS}
    np.savez_compressed(output/'offline_batches.npz',**batches)
    buffer=io.StringIO();columns=list(points[0]);writer=csv.DictWriter(buffer,fieldnames=columns,lineterminator='\n');writer.writeheader()
    for p in points:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,list) else v for k,v in p.items()})
    (output/'offline_points.csv.gz').write_bytes(gzip.compress(buffer.getvalue().encode(),mtime=0))
    write_csv(output/'holdout_nearest_neighbors.csv',nearest);write_json(output/'dataset_audit.json',statistics)
    return statistics


def prepare(before_path=Path('/tmp/birdsong-issue25-before.json'),output=OUT):
    import torch
    from stable_baselines3.common.logger import Logger
    from .model import LIBRARY
    torch.set_num_threads(1);output=Path(output);tick=time.perf_counter()
    if output.exists():raise FileExistsError('Existing study preserved')
    before=json.loads(Path(before_path).read_text())
    for p in before['files']:assert digest(ROOT/p['path'])==p['sha256'],p['path']
    old=json.loads((BASE/'manifest.json').read_text());previous=json.loads((PREVIOUS/'manifest.json').read_text())
    assert previous['baseline_manifest_sha256']==digest(BASE/'manifest.json')
    for p,h in old['source_sha256'].items():assert digest(ROOT/p)==h,p
    for p,h in previous['source_sha256'].items():assert digest(ROOT/p)==h,p
    assert digest(LIBRARY)==old['native_library_sha256'] and digest(TARGET)==old['target_sha256']
    versions={p:importlib.metadata.version(p) for p in old['versions']};assert versions==old['versions']==previous['versions']
    test_record=json.loads(Path('/tmp/birdsong-issue25-preflight-tests.json').read_text());assert test_record['passed']
    output.mkdir(parents=True);write_csv(output/'protected_before.csv',before.pop('files'));write_json(output/'storage_before.json',before)
    config=dict(old['settings']);config.update(experiment_id='issue25_sac128_critic_lr',critic_learning_rates=list(CRITIC_RATES))
    initial_checks=[]
    for seed in SEEDS:
        baseline=json.loads((folder_for(BASE_LR,seed)/'summary.json').read_text())
        for p,h in baseline['artifact_sha256'].items():assert digest(folder_for(BASE_LR,seed)/p)==h,p
        env=environment(.75)
        try:
            ref=original_agent('SAC',128,seed,env,config);ref_sig=learner_signature(ref)
            def groups(a):return {n:[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups] for n,opt in
                (('actor',a.actor.optimizer),('critic',a.critic.optimizer),('entropy',a.ent_coef_optimizer))}
            ref_groups=json.loads(json.dumps(groups(ref)))
            for rate in CRITIC_RATES:
                agent=make_agent(rate,seed,env,config);effective_difference(agent,baseline['effective_hyperparameters'],rate);sig=learner_signature(agent)
                assert sig['policy_sha256']==ref_sig['policy_sha256']==baseline['initial']['learner_before']['policy_sha256']
                assert sig['rng_sha256']==ref_sig['rng_sha256'] and sig['entropy_sha256']==ref_sig['entropy_sha256']
                normalized=json.loads(json.dumps(groups(agent)));normalized['critic'][0]['lr']=BASE_LR;assert normalized==ref_groups
                agent.set_logger(Logger(None,[]));lrs=[]
                for progress in (1.,.5,0.):
                    agent._current_progress_remaining=progress;agent._update_learning_rate([agent.actor.optimizer,agent.critic.optimizer,agent.ent_coef_optimizer]);lrs.append(assert_lrs(agent,rate))
                initial_checks.append(dict(seed=seed,rate=rate,initial_policy_rng_exact=True,only_critic_lr_differs=True,schedule_lrs=lrs))
        finally:env.close()
    dataset=build_data(output);write_json(output/'test_execution.json',test_record)
    plan=f'''# Stage A — 計算前に固定したIssue25計画

既存Issue23の基準3seedとIssue24が参照した基準manifest/hashを照合した。新規フル学習はCritic LR1e-4/1e-3×3seed×4096候補の6runのみ。Actor/温度LRと共通scheduleは3e-4、他設定・模型・Target・報酬・範囲は不変。全optimizer更新直前/直後と33独立評価で実効LRを検査する。

高LRの初期nativeテストは3seed×32候補、低LRはseed0×32候補を事前確認し、全て成功。通常の有限値検査に加え、全LRで同じ緊急停止基準を固定：勾配L2>{LIMITS['gradient_l2']:g}、重み絶対値>{LIMITS['parameter_abs']:g}、Adam moment絶対値>{LIMITS['optimizer_moment_abs']:g}、非有限値または負の二次momentで停止する。クリップ・置換・失敗seedの削除は行わず、使用予算と失敗を残す。

Final=4096最終更新後、Selected=33点の最初のminJ、posthoc=同33点の最初のmin共通E。初回E≤.10かつ持続の方策も1本だけ保持。全33時点で同じ99候補へのQ1/Q2/minQ、Actor行動・∇a minQ・alpha/std/entropy、全Actor更新差分と3LR/数値検査を記録する。独立評価はReplayに入れず、保存方策に戻さない。

固定回帰は既存Issue23学習経験12288件を全てtrain、独立評価99候補をholdoutとして完全隔離する。Issue22は同じデータの再現元であるため重複して加えない。各回帰runは全3source seedの同じtrainを使い、回帰seed0/1/2は初期Criticとbatch列のseedを指す。構造/twin-Q/Adam/batch32/損失を固定し、Critic LR3e-4/1e-4/1e-3×3seedの9本を各{OFFLINE_UPDATES}更新する。オンライン完了後に開始し、Actor・温度・target Criticを学習せず、オンラインActorへ接続しない。

初期Criticは同seedの標準SACから取り、全LRで同じ重み。mini-batch index列はlocal NumPy seed250025+seedで先に生成し、全LRで同じ列を使う。更新目標はfloat32(r)（SB3 Replayと同じ丸め）、損失0.5*(MSE(Q1,r)+MSE(Q2,r))。報告は元float64 rも用いる。更新数は評価値によらず固定、早期停止・holdout選抜・サンプル追加/重み付けは0。診断時刻は{list(OFFLINE_EVALS)}、Final回帰だけを主比較する。

点ごとのhashと重複・近傍重複を[dataset_audit](dataset_audit.json)と[近傍CSV](holdout_nearest_neighbors.csv)へ保存した。raw完全重複{dataset['train_holdout_exact_raw_overlap']}、Critic入力完全重複{dataset['train_holdout_exact_feature_overlap']}。同じTarget・相関した方策軌道・近傍点は共有し、新Target/一様領域の汎化性能とは呼ばない。holdoutの結果で今回のLRや更新数を変更しない。

追加の局所鳴管計算上限は**0音**。Issue23の既存36音と3固定anchorの真の中心差分だけを参照し、現在のCriticの勾配を同じ既存点で照会できる。新しい訪問点の真の勾配を測ったとは言わない。全99候補・lowJ10・実際の自身の学習経験を別集計する。

研究用のStage B/C実行前にこの計画・データ・実装hashをmanifestへ凍結する。数値異常は失敗として保持し、欠けたFinalを途中方策へ置き換えない。
'''
    (output/'PLAN.md').write_text(plan)
    write_json(output/'preflight.json',dict(passed=True,generated_sounds=0,initial_checks=initial_checks,
        baseline_reuse_exact=True,high_lr_native_smoke_passed=True,numerical_limits=LIMITS,source_sha256=digest(__file__),wall_s=time.perf_counter()-tick))
    write_json(output/'manifest.json',dict(issue=25,created_utc=datetime.now(timezone.utc).isoformat(),git_head=before['git_head'],settings=config,
        source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),versions=versions,
        baseline_manifest_sha256=digest(BASE/'manifest.json'),previous_manifest_sha256=digest(PREVIOUS/'manifest.json'),
        sb3_implementation_sha256=sb3_hashes(),plan_sha256=digest(output/'PLAN.md'),
        data_sha256={p:digest(output/p) for p in ('offline_data.npz','offline_batches.npz','offline_points.csv.gz','dataset_audit.json','holdout_nearest_neighbors.csv')},
        local_reference_sha256={p:digest(BASE/p) for p in ('local_sensitivity.csv','local_gradients.csv','local_execution.json')},
        numerical_limits=LIMITS,online_runs=6,baseline_new_runs=0,local_native_sounds=0,
        offline=dict(runs=9,updates=OFFLINE_UPDATES,batch_size=32,evaluation_updates=list(OFFLINE_EVALS),train_points=12288,holdout_points=99),
        source_link='https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/25'))
    print('Stage A frozen: Critic-only LR verified;12288 train /99 isolated holdout;6 online runs then9 offline fits,4080 updates each',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue25-before.json'));a=p.parse_args();prepare(a.before)
