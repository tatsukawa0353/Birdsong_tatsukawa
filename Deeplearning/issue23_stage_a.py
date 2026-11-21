"""Existing-artifact analysis and explicit decision before any research retraining."""
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from .issue23_data import (ROOT,BASE,OUT,TARGET,SEEDS,NAMES,HALF_WIDTH,LOCAL_STEPS,LOCAL_ANCHORS,
    source_hashes,fixed_candidates,old_folder,old_curve,chosen,environment,policy_statistics,predict_q)
from .issue14 import learner_signature
from .constant020_env import episode,account
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return None if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def prepare(before=Path('/tmp/birdsong-issue23-before.json'),output=OUT):
    import torch
    from stable_baselines3 import SAC
    from .model import LIBRARY
    torch.set_num_threads(1);output=Path(output);tick=time.perf_counter()
    if output.exists():raise FileExistsError('Existing study preserved')
    inventory=json.loads(Path(before).read_text())
    for r in inventory['files']:assert digest(ROOT/r['path'])==r['sha256'],r['path']
    prior=json.loads((BASE/'manifest.json').read_text())
    for p,h in prior['source_sha256'].items():assert digest(ROOT/p)==h,p
    assert digest(LIBRARY)==prior['native_library_sha256'] and digest(TARGET)==prior['target_sha256']
    assert prior['versions']=={p:importlib.metadata.version(p) for p in prior['versions']}
    output.mkdir(parents=True);write_csv(output/'protected_before.csv',inventory.pop('files'))
    write_json(output/'storage_before.json',inventory)
    candidates=fixed_candidates();write_json(output/'fixed_candidates.json',candidates)
    evaluations=[];correlations=[];endpoints=[];audits=[];accounts=[];supplement=[]
    with (BASE/'policy_evaluations.csv').open() as f:csv_eval=list(csv.DictReader(f))
    with (BASE/'policy_dynamics.csv').open() as f:csv_dyn=list(csv.DictReader(f))
    for seed in SEEDS:
        points=old_curve(.75,seed);selection=chosen(points);previous=None
        for p in points:
            n=p['episodes'];s=p['policy_statistics'];a=np.array(p['actions'][0]);physical=np.array(p['physical_controls'])[[0,2,3]]
            match=[r for r in csv_eval if r['condition']=='lambda_0p75' and int(r['seed'])==seed and int(r['trial'])==n]
            dyn=[r for r in csv_dyn if r['condition']=='lambda_0p75' and int(r['seed'])==seed and int(r['trial'])==n]
            assert len(match)==len(dyn)==1 and float(match[0]['E_total'])==p['acoustic_error']
            assert float(match[0]['J'])==p['weighted_loss'] and float(dyn[0]['alpha'])==s['alpha']
            row=dict(seed=seed,trial=n,E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],
                E_env=p['acoustic_components']['envelope'],J=p['weighted_loss'],alpha=s['alpha'],
                entropy=s['squashed_entropy_mc'],gaussian_entropy=s['gaussian_entropy'],
                selected=n==selection['selected']['episodes'],posthoc=n==selection['posthoc']['episodes'],final=n==4096,
                sustained_voiced=p['sustained_voiced'],delta_action_l2=None,delta_range_fraction_l2=None,delta_E_total=None)
            for i,name in enumerate(NAMES):
                row.update({f'action_{name}':float(a[i]),f'physical_{name}':float(physical[i]),
                    f'std_{name}':s['normalized_action_std_mc'][i],f'log_std_{name}':s['latent_log_std'][i],
                    f'delta_action_{name}':None,f'delta_physical_{name}':None})
            if previous is not None:
                da=a-np.array(previous['actions'][0]);dp=physical-np.array(previous['physical_controls'])[[0,2,3]]
                np.testing.assert_allclose(dp,da*HALF_WIDTH,rtol=1e-11,atol=1e-7)
                row.update(delta_action_l2=float(np.linalg.norm(da)),delta_range_fraction_l2=float(np.linalg.norm(da/2)),
                    delta_E_total=p['acoustic_error']-previous['acoustic_error'],delta_J=p['weighted_loss']-previous['weighted_loss'])
                for i,name in enumerate(NAMES):row.update({f'delta_action_{name}':float(da[i]),f'delta_physical_{name}':float(dp[i])})
            evaluations.append(row);previous=p
        for role,p in selection.items():
            endpoints.append(dict(seed=seed,role=role,trial=p['episodes'],E_total=p['acoustic_error'],
                E_spec=p['acoustic_components']['log_spectral'],E_env=p['acoustic_components']['envelope'],J=p['weighted_loss'],
                selected_minus_final=p['acoustic_error']-selection['final']['acoustic_error']))
        for window,lower in (('all',0),('late',2048)):
            rows=[r for r in evaluations if r['seed']==seed and r['trial']>=lower]
            for field in ('alpha','entropy',*[f'std_{n}' for n in NAMES],*[f'log_std_{n}' for n in NAMES]):
                x=np.array([r[field] for r in rows]);y=np.array([r['E_total'] for r in rows])
                for kind,xx,yy in (('level_same_time',x,y),('difference_same_interval',np.diff(x),np.diff(y)),
                    ('level_predicts_next_delta',x[:-1],np.diff(y)),('prior_delta_predicts_next_delta',np.diff(x)[:-1],np.diff(y)[1:])):
                    correlations.append(dict(seed=seed,window=window,field=field,alignment=kind,n=len(xx),spearman=correlation(xx,yy)))
            change=[r for r in rows if r['delta_action_l2'] is not None]
            correlations.append(dict(seed=seed,window=window,field='delta_action_l2',alignment='absolute_delta_E',n=len(change),
                spearman=correlation([r['delta_action_l2'] for r in change],[abs(r['delta_E_total']) for r in change])))
        env=environment(.75);t=time.perf_counter()
        try:
            for role in ('final','selected'):
                folder=old_folder(.75,seed);model_path=folder/('agent.zip' if role=='final' else 'selected_agent.zip')
                agent=SAC.load(model_path,device='cpu');p=selection[role];before_sig=learner_signature(agent)
                metrics,values=episode(env,agent=agent);stats=policy_statistics(agent)
                q1,q2=predict_q(agent,[c['action'] for c in candidates]);assert learner_signature(agent)==before_sig
                assert metrics['weighted_loss']==p['weighted_loss'] and metrics['acoustic_error']==p['acoustic_error']
                np.testing.assert_array_equal(values['actions'][0],p['actions'][0])
                with np.load(folder/f'{role}.npz') as z:
                    np.testing.assert_array_equal(values['audio'],z['audio']);np.testing.assert_array_equal(values['physical_controls'],z['physical_controls'])
                for k in ('alpha','latent_log_std','normalized_action_std_mc','squashed_entropy_mc'):
                    np.testing.assert_allclose(stats[k],p['policy_statistics'][k],atol=1e-7,rtol=0)
                audits.append(dict(seed=seed,role=role,trial=p['episodes'],model_path=str(model_path.relative_to(ROOT)),
                    model_sha256=digest(model_path),waveform_exact=True,controls_exact=True,metrics_exact=True,learner_unchanged=True))
        finally:accounts.append(account(env,time.perf_counter()-t));env.close()
    for method,seed in (('TD3',0),('SAC',1)):
        p=ROOT/f'results/phase1_caseA_3d_network_width/runs/{method}/width_128/seed_{seed}/summary.json'
        s=json.loads(p.read_text());supplement.append(dict(method=method,seed=seed,source=str(p.relative_to(ROOT)),
            final=s['final']['acoustic_error'],selected=s['selected']['acoustic_error'],trial=s['selected']['episodes']))
    write_csv(output/'stage_a_evaluations.csv',evaluations);write_csv(output/'stage_a_correlations.csv',correlations)
    write_csv(output/'stage_a_endpoints.csv',endpoints);write_csv(output/'other_methods.csv',supplement)
    write_json(output/'stage_a_audit.json',dict(passed=True,model_reloads=6,generated_sounds=6,endpoints=audits,accounts=accounts,
        intermediate_critic_available=False,old_intermediate_policy_available=False,source_sha256=digest(__file__),wall_s=time.perf_counter()-tick))
    ep_table=table(['seed','採用','試行','E_total','E_spec','E_env','J','採用−Final E'],
        [[r['seed'],r['role'],r['trial'],*[f"{r[k]:.7f}" for k in ('E_total','E_spec','E_env','J')],f"{r['selected_minus_final']:+.7f}"] for r in endpoints])
    body=f'''# Stage A — 既存成果物による診断（新規学習前）

Issue #22のλ=.75のCSVと圧縮JSONを照合し、全99評価を読み込んだ。Stage Aで新規学習は0。保存済みFinal/Selectedの6モデルをロードし、6音を独立再生した。行動・20ブロックの制御・float64波形・J・共通Eが完全一致し、温度/分散も一致した。

{ep_table}

seed0は33点すべてで共通E>.10であり、評価した方策の中では到達自体が課題。未評価の127試行間に良い方策が存在しなかったとは断定しない。seed1はSelected .09137とFinal .09599で、到達後の乖離は.00462と小さい。seed2は2944試行のposthoc .03587から3072試行に悪化し、3840試行のminJ .04656を経てFinal .08283へ再び悪化する。良い方策への到達と、その後の維持を分ける必要がある。Finalはseed1/2で≤.10だが3seed平均.14745で暫定目標は未達。

[全seedの損失・行動・差分](stage_a_evaluations.csv) / [相関・時刻の整列](stage_a_correlations.csv) / [既存endpoint監査](stage_a_audit.json)。Δaは128候補間隔の決定論的評価行動の差で、学習時に採った確率的行動の差や1更新の制限値ではない。物理差はΔa×[12500000,750000,375500]、全域幅に対する差はΔa/2。単位の異なる物理量を直接L2で合成しない。

相関はseed別、全33点と後半17点（≥2048）で、同時刻の水準、同一区間の一次差分、前時刻水準→次区間ΔE、前区間差分→次区間ΔEを計算した。標本が少なく自己相関・共通トレンド・多重比較があり、検定や因果推論には使わない。alphaの上昇と行動stdの上昇を同一視せず、3座標を別に評価する。33点の間は未観測で、線は時系列の接続のみ。

## Stage Bが必要な理由と事前固定する追加記録

既存データには中間Actor/Critic重みがなく、2944/3072/3840時点のQやActorの1更新による逸脱を復元できない。保存済みFinal/Selected Criticで過去候補を評価することは可能だが、それを過去のCriticと呼ばない。したがって同条件の3seed×4096候補だけを再現する。学習率・報酬・範囲・entropy目標・幅・batch等は変更しない。

開始時は実効設定・初期policy署名・評価を照合し、実行中は全4096学習サンプルの行動/報酬/音響値、全33評価の学習器/RNG/buffer署名を旧runと照合する。SB3は学習開始時のresetで環境seedを適用するため、初期時点だけ未使用・OS初期化のtraining_env_rng署名は比較から除外し、その不一致自体を記録する。128以降は除外せず全署名を比較する。初回128候補でprefix一致を記録し、完全一致しない場合は差の時点を残して別軌跡として扱う。診断による不変性は初期点を含む評価・保存前後の全署名でも検査する。

選択規則を事前固定：Selected=33点の最初のminJ、posthoc=同33点の最初のmin共通E（学習目標とは別の事後選別）、Final=4096候補の最終更新後。各基準のActorとonline twin Criticの推論用重み・行動・波形を別々に保持し、学習器に戻さない。optimizer/Replayは再開しないため保存しない。加えてseed2の2944/3072/3840時点を診断用に保存する。Final4096は基準別保持を使う。

旧33点×3seedの99個の行動と実測−Jを固定候補にする。新runの各33時点で全候補へのQ1/Q2/minQ、Actor出力へのQ・∇a minQ、alpha/std/entropy、直前区間のactor/critic/温度loss、全4080 Actor更新の前後出力を記録する。固定候補は回顧的で良い解を含む偏った集合であり、学習器へ一切渡さず、空間全体の較正や介入の因果効果は主張しない。

局所感度はseed2の2944/3840/4096の3既存行動に限定し、正規化座標3変数×±(.002,.01)で最大36追加音。刻みは物理量でε_l±(25000,125000)、p_s±(1500,7500)、f0_l±(751,3755)。中心値は既存ログ/独立再生を使い、追加の無制限探索は行わない。局所感度を空間全体へ一般化しない。

他手法の補助事実：{'; '.join(f"{r['method']}128 seed{r['seed']} Selected({r['trial']}) {r['selected']:.5f} → Final {r['final']:.5f}" for r in supplement)}。報酬係数が1+1の別実験であり、SAC固有とは断定しない。TD3新規計算は0。
'''
    (output/'STAGE_A.md').write_text(body);(output/'REPORT.md').write_text(body)
    config=dict(prior['settings']);config.update(experiment_id='issue23_sac128_stability',spectral_weights=[.75])
    write_json(output/'manifest.json',dict(issue=23,created_utc=datetime.now(timezone.utc).isoformat(),settings=config,
        source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),
        baseline_manifest_sha256=digest(BASE/'manifest.json'),versions=prior['versions'],
        stage_a_sha256=digest(output/'STAGE_A.md'),candidates_sha256=digest(output/'fixed_candidates.json'),
        stage_b_reason='Intermediate Critic/Actor weights and per-update outputs absent; Stage A recorded before fitting',
        research_runs_max=3,training_candidates_max=12288,evaluations_per_run=33,
        checkpoints=dict(selected='first min J',posthoc='first min common E; posthoc selection',final='post final update',extra={'2':[2944,3072,3840]}),
        local_sensitivity=dict(seed=2,anchors=list(LOCAL_ANCHORS),normalized_steps=list(LOCAL_STEPS),maximum_sounds=36),
        reference_issue='https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/23'))
    print('Stage A saved before fitting; six old endpoints replayed exactly; intermediate Critic absent -> three reproduction runs justified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue23-before.json'))
    a=p.parse_args();prepare(a.before)
