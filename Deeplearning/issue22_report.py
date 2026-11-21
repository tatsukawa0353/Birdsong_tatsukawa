"""Report sum-one objectives, common error, stability and entropy without refitting."""
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from .issue22_data import (ROOT,BASE,OUT,TARGET,CONDITIONS,WEIGHTS,SEEDS,label,
    folder_for,curve_for,endpoints,target_data,ready)
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest
from .issue19_protection import workspace_inventory


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def statistics(rows):
    result=[]
    for c in CONDITIONS:
        for role in ('final','selected','posthoc'):
            a=[r for r in rows if r['condition']==label(c) and r['role']==role]
            g=dict(condition=label(c),role=role,n=3,voiced=sum(r['sustained_voiced'] for r in a))
            for field in ('E_total','E_spec','E_env','J','alpha','squashed_entropy_mc','parameter_nrmse_restricted'):
                y=np.array([r[field] for r in a]);g[field+'_mean']=float(y.mean());g[field+'_sd']=float(y.std(ddof=1))
            g['goal_met']=role=='final' and g['E_total_mean']<=.1 and g['voiced']==3
            result.append(g)
    return result


def listening():
    target,_=target_data();gain=1/abs(target).max();members={};index=[]
    with zipfile.ZipFile(OUT/'new_listening_audio.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for r in json.loads((OUT/'endpoint_sources.json').read_text()):
            if r['condition']=='legacy_sum' or r['role']=='posthoc':continue
            with np.load(ROOT/r['audio_path']) as z:audio=z['audio'].copy()
            stream=io.BytesIO();wavfile.write(stream,50000,(audio*gain).astype(np.float32));data=stream.getvalue();h=hashlib.sha256(data).hexdigest()
            name=f'{r["condition"]}_seed{r["seed"]}_{r["role"]}.wav'
            if h not in members:
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data);members[h]=name
            index.append(dict(condition=r['condition'],seed=r['seed'],role=r['role'],member=members[h],sha256=h))
    write_json(OUT/'listening_audio.json',dict(gain=gain,sample_rate=50000,unique_members=len(members),endpoints=index,
        independent_normalization=False,raw_target_path=str(TARGET.relative_to(ROOT)),generated_sounds=0,
        meaning='Same Target-derived gain for every new sound; raw float64 audio preserved in NPZ; old sounds referenced, not copied'))


def freeze():
    from .issue22_analysis import integrity
    for name in ('artifact_checksums.json','storage_after.json'):
        if not (OUT/name).exists():(OUT/name).write_text('{}\n')
    integrity()
    new=[ROOT/p for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines() if (ROOT/p).is_file()]
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True).splitlines()
    before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total']
    for _ in range(4):
        current=workspace_inventory();size=sum(p.stat().st_size for p in new)
        record=dict(workspace=current,new_git_candidate_files=len(new),new_git_candidate_bytes=size,status=status,
            modified_tracked_files=sum(r[:2].strip() not in ('??','D') for r in status),deleted_tracked_files=sum('D' in r[:2] for r in status),
            measurement='Logical regular files including ignored files; .git/symlinks excluded; new candidates are untracked, not staged or committed')
        write_json(OUT/'storage_after.json',record)
        block='<!-- ISSUE22_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[
            ['作業ツリー通常ファイル数',before['files'],current['total']['files']],
            ['論理容量（ignored含む）',f"{before['bytes']/1e6:.2f} MB",f"{current['total']['bytes']/1e6:.2f} MB"],
            ['Git新規追加候補','0（クリーン）',f'{len(new)}個 / {size/1e6:.2f} MB'],
            ['既存追跡ファイルの変更 / 削除','0 / 0',f"{record['modified_tracked_files']} / {record['deleted_tracked_files']}"]])+'\n<!-- ISSUE22_STORAGE_END -->'
        p=OUT/'REPORT.md';body=p.read_text();p.write_text(re.sub(r'<!-- ISSUE22_STORAGE_BEGIN -->.*?<!-- ISSUE22_STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- ISSUE22_STORAGE_BEGIN -->' in body else body+'\n'+block+'\n')
        write_json(OUT/'artifact_checksums.json',dict(excludes=['artifact_checksums.json','storage_after.json'],
            implementation_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.glob('issue22*.py'))},
            artifacts={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')}))


def report():
    ready();started=time.perf_counter();_,controls=target_data();rows=[];policy=[];dynamics=[];stability=[]
    legacy={(r['seed'],r['role']):r for r in json.loads((OUT/'legacy_policy_statistics.json').read_text())}
    for condition in CONDITIONS:
        for seed in SEEDS:
            chosen=endpoints(condition,seed)
            for role,p in chosen.items():
                comp=p['acoustic_components'];physical=np.asarray(p['physical_controls']);delta=physical-controls
                stats=p['policy_statistics'] if 'policy_statistics' in p else legacy[seed,'final' if role=='final' else 'selected']
                rows.append(dict(condition=label(condition),lambda_spectral='' if condition=='legacy_sum' else condition,
                    spec_coefficient=1. if condition=='legacy_sum' else condition,env_coefficient=1. if condition=='legacy_sum' else 1-condition,
                    seed=seed,role=role,trial=p['episodes'],E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],
                    sustained_voiced=p['sustained_voiced'],failed=p['failed'],epsilon_l=physical[0],ps=physical[2],f0_l=physical[3],
                    epsilon_l_error=delta[0],ps_error=delta[2],f0_l_error=delta[3],parameter_nrmse_original=p['parameter_nrmse'],
                    parameter_nrmse_restricted=float(np.sqrt(np.mean((delta[[0,2,3]]/[2.5e7,1.5e6,751000.])**2))),
                    alpha=stats['alpha'],squashed_entropy_mc=stats['squashed_entropy_mc'],entropy_mc_se=stats['entropy_mc_se']))
            f,s,h=chosen['final'],chosen['selected'],chosen['posthoc']
            stability.append(dict(condition=label(condition),seed=seed,selected_trial=s['episodes'],posthoc_trial=h['episodes'],
                selected_minus_final_E_total=s['acoustic_error']-f['acoustic_error'],selected_minus_final_J=s['weighted_loss']-f['weighted_loss'],
                posthoc_minus_selected_E_total=h['acoustic_error']-s['acoustic_error'],posthoc_minus_final_E_total=h['acoustic_error']-f['acoustic_error']))
            for p in curve_for(condition,seed):
                comp=p['acoustic_components'];policy.append(dict(condition=label(condition),seed=seed,trial=p['episodes'],
                    E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],
                    sustained_voiced=p['sustained_voiced'],gradient_updates=p['gradient_updates']))
                if condition!='legacy_sum':
                    stats=p['policy_statistics'];dynamics.append(dict(condition=label(condition),seed=seed,trial=p['episodes'],source='independent_evaluation',
                        alpha=stats['alpha'],gaussian_entropy=stats['gaussian_entropy'],squashed_entropy_mc=stats['squashed_entropy_mc'],entropy_mc_se=stats['entropy_mc_se'],
                        **{f'{name}_{dimension}':stats[name][i] for name in ('latent_mean','latent_std','latent_log_std','normalized_action_std_mc') for i,dimension in enumerate(('epsilon_l','ps','f0_l'))},
                        **{f'normalized_action_variance_mc_{dimension}':stats['normalized_action_std_mc'][i]**2 for i,dimension in enumerate(('epsilon_l','ps','f0_l'))}))
    for p in legacy.values():
        dynamics.append(dict(condition='legacy_sum',seed=p['seed'],trial=p['trial'],source='saved_'+p['role']+'_model',
            alpha=p['alpha'],gaussian_entropy=p['gaussian_entropy'],squashed_entropy_mc=p['squashed_entropy_mc'],entropy_mc_se=p['entropy_mc_se'],
            **{f'{name}_{dimension}':p[name][i] for name in ('latent_mean','latent_std','latent_log_std','normalized_action_std_mc') for i,dimension in enumerate(('epsilon_l','ps','f0_l'))},
            **{f'normalized_action_variance_mc_{dimension}':p['normalized_action_std_mc'][i]**2 for i,dimension in enumerate(('epsilon_l','ps','f0_l'))}))
    groups=statistics(rows);paired=[]
    for condition in WEIGHTS:
        for reference in ('legacy_sum',.5):
            if condition==reference:continue
            for seed in SEEDS:
                for role in ('final','selected','posthoc'):
                    a=next(r for r in rows if (r['condition'],r['seed'],r['role'])==(label(reference),seed,role))
                    b=next(r for r in rows if (r['condition'],r['seed'],r['role'])==(label(condition),seed,role))
                    paired.append(dict(condition=label(condition),reference=label(reference),seed=seed,role=role,
                        **{'delta_'+field:b[field]-a[field] for field in ('E_total','E_spec','E_env')},
                        voiced_reference=a['sustained_voiced'],voiced_condition=b['sustained_voiced']))
    for path,data in (('comparison_per_seed.csv',rows),('comparison_aggregate.csv',groups),('policy_evaluations.csv',policy),
        ('policy_dynamics.csv',dynamics),('selected_final_stability.csv',stability),('paired_differences.csv',paired)):write_csv(OUT/path,data)
    with gzip.open(OUT/'training_samples.csv.gz','wt',newline='') as stream:
        columns=['lambda','seed','trial','E_total','E_spec','E_env','J','reward_float32','alpha_before_update','sustained_voiced','action','physical_controls']
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for weight in WEIGHTS:
            for seed in SEEDS:
                for p in read_record(folder_for(weight,seed)/'training_trace.json'):
                    comp=p['acoustic_components'];writer.writerow(dict({'lambda':weight},seed=seed,trial=p['episodes'],
                        E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],reward_float32=p['reward'],
                        alpha_before_update=p['alpha_before_update'],sustained_voiced=p['sustained_voiced'],
                        action=json.dumps(p['action']),physical_controls=json.dumps(p['physical_controls'])))
    listening()
    from .issue22_figures import render
    images=render(rows,stability)
    fmt=lambda g,f:f"{g[f+'_mean']:.5f} ± {g[f+'_sd']:.5f}"
    main=table(['条件','採用','E_total 平均±SD','E_spec 平均±SD','E_env 平均±SD','自身J 平均±SD','持続/3'],
        [[g['condition'],g['role'],*[fmt(g,f) for f in ('E_total','E_spec','E_env','J')],g['voiced']] for g in groups])
    final=[g for g in groups if g['role']=='final' and g['condition']!='legacy_sum'];old=next(g for g in groups if g['role']=='final' and g['condition']=='legacy_sum')
    best=min(final,key=lambda g:g['E_total_mean']);passed=[g for g in final if g['goal_met']]
    goal=('暫定目標を満たす新式条件：'+', '.join(g['condition'] for g in passed)+'。') if passed else '新式3条件のいずれも、Final平均E_total≤.10かつ持続発声3/3の暫定目標は満たさなかった。'
    eligible=[g for g in final if g['voiced']==3]
    candidate=min(eligible,key=lambda g:g['E_total_mean']) if eligible else None
    adoption=(f"新式内の次段階の比較基準候補は{candidate['condition']}（Final {fmt(candidate,'E_total')}）。ただし3seedの範囲での候補であり、一般的な最適重みや採用確定を意味しない。" if candidate and candidate['E_total_mean']<old['E_total_mean'] else
        '旧報酬を超えるFinal平均と全seed持続を両立する採用根拠は得られなかった。新式の永続的な既定値への採用は提案せず、旧条件と今回の全seed結果を保持する。')
    individual=[]
    for c in CONDITIONS:
        for seed in SEEDS:
            a={r['role']:r for r in rows if r['condition']==label(c) and r['seed']==seed};g=next(r for r in stability if r['condition']==label(c) and r['seed']==seed)
            individual.append([label(c),seed,*[f"{a[role]['E_total']:.7g}" for role in ('final','selected','posthoc')],a['selected']['trial'],a['posthoc']['trial'],
                f"{g['selected_minus_final_E_total']:+.7g}",a['final']['sustained_voiced'],a['selected']['sustained_voiced']])
    with (OUT/'critic_metrics.csv').open() as stream:critics=list(csv.DictReader(stream))
    ctable=[]
    for condition in CONDITIONS:
        for head in ('Q1','Q2','minQ'):
            a=[r for r in critics if r['condition']==label(condition) and r['critic']==head and r['subset']=='fixed_pool']
            low=[r for r in critics if r['condition']==label(condition) and r['critic']==head and r['subset']=='low_J_decile']
            mean=lambda data,field:f"{np.mean([float(r[field]) for r in data if r[field]]):.4f}"
            ctable.append([label(condition),head,mean(a,'rmse'),mean(a,'rmse_over_reward_std'),mean(a,'spearman'),mean(a,'pairwise_inversion_fraction'),mean(low,'bias_Q_minus_r')])
    entropy_table=[]
    for c in CONDITIONS:
        for role in ('final','selected'):
            g=next(r for r in groups if r['condition']==label(c) and r['role']==role)
            entropy_table.append([label(c),role,fmt(g,'alpha'),fmt(g,'squashed_entropy_mc')])
    pair_table=[]
    for condition in WEIGHTS:
        for reference in ('legacy_sum',.5):
            if condition==reference:continue
            data=[r for r in paired if r['condition']==label(condition) and r['reference']==label(reference) and r['role']=='final']
            values=[np.array([r['delta_'+field] for r in data]) for field in ('E_total','E_spec','E_env')]
            pair_table.append([label(condition),label(reference),*[f'{v.mean():+.5f} ± {v.std(ddof=1):.5f}' for v in values]])
    audit=json.loads((OUT/'audit.json').read_text());execution=json.loads((OUT/'execution.json').read_text())
    with (OUT/'protected_before.csv').open() as stream:protected=sum(1 for _ in csv.DictReader(stream))
    body=f'''# Issue22 — SAC128・係数和1の報酬重み比較

## 結果と採用判断

{goal}

新式のFinal平均が最小だったのは{best['condition']}：E_total={fmt(best,'E_total')}、持続{best['voiced']}/3。旧報酬SAC128の参考値は{fmt(old,'E_total')}、持続{old['voiced']}/3。

{adoption}

今回のλ=.75は旧条件に比べてFinal平均E_specが0.17329→0.10602へ下がった一方、E_envは0.01185→0.04142へ増えた。共通誤差はseed1/2で改善したがseed0は0.26352で、平均改善と全seedの高精度は別である。λ=.90のFinal平均は0.20089で、スペクトル係数を増やすほど良くなる結果ではなかった。

同じ重み比のλ=.50でも、Final共通誤差は旧0.18515→0.23515に悪化した。一方、学習用Jは0.11758と旧Jより小さい。これはスケールの異なるJを改善指標にできない具体例であり、重み比の効果とも解釈しない。

この比較は学習報酬の定義を変えた9runだけ。Selected→Finalの差は記録するが、原因を同定したり、良い方策へ自動復帰する仕組みを導入したりしていない。次Issueでは今回の候補条件を事前固定して、必要なら安定性の診断・保持機構・探索の介入を別々に検証する。今回の結果だけから唯一の原因や一般的優位を断定しない。

## 報酬の定義とスケール

新式は `Jλ=λE_spec+(1−λ)E_env`、`reward=−Jλ`。λ=.50/.75/.90の係数は(.50,.50)/(.75,.25)/(.90,.10)、すべて和1。**補正の2倍は掛けない**。主評価は全条件共通の `E_total=E_spec+E_env`。

旧Issue21は係数(1,1)の`legacy_sum`で、旧報酬の「λ=1」やIssue19の`E_spec+wE_env`と新式λを混同しない。新式λ=.50は同じ候補について旧Jの正確な半分で、重み比は同じでも損失の大きさが違う。旧条件との差を「重み比だけの効果」と呼ばない。新式内も、成分の尺度が違うため係数和1が実際の報酬分布や温度との比まで一定にするわけではない。

1000組の成分値と実際の音で係数和・半分の恒等式を検証し、目標と旧Final3音の全サンプル・共通評価が一致した。[事前検証](preflight.json)。既存の数値失敗ペナルティ1e6は保持し、実際の研究runの数値失敗は0。テストで注入した失敗は別記する。

## 固定条件

Case A、右α_r=β_r=0、x0=.02 cm/D=.01、0.20秒・50kHz・10000sample、native dt=1e-7、状態/遅延履歴0。目標の既存波形/hashと制御 `[4.1e7,3.5e7,3.4e6,3e5,4.9e4]` を保持。可変はε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]、右ε_r=3.5e7/f0_r=4.9e4固定。f0_lはHzでなく開閉外力係数。モデル内部量を実測張力やPaと直接同一視しない。

SACのみ、Actor/Critic [128,128]、lr3e-4、batch32、gamma1、buffer50000、warmup16、train_freq1、gradient_steps1、ent_coef=auto_0.01、CPU1thread/worker。旧SAC128の実効設定・初期方策・最初の16行動が一致。後の行動軌道が同一とは仮定しない。各runは4096候補・4080回のActor/Critic/温度更新。評価は0,128,...,4096の33回で、4096は最終更新後。TD3/DE/Random、Case B、時変制御、実録音、範囲・entropy目標・初期温度・探索機構の変更は行わない。[manifest](manifest.json) / [各runの実効設定](runs/) / [監査](audit.json)。

## Final・Selected・posthoc

Finalは最終更新後の決定論的方策。Selectedは33独立評価中で**その条件のJを最小化**した最初の点。posthocは同じ33点の共通E_total最小で、診断値のみ。posthocのための追加学習や学習器への復帰、途中重みの大量保存は行わない。posthocの行動は保存制御から再評価し、Final/Selectedと同じ行動なら検証済みの再生を共用する。

{main}

平均±SDは3seedの標本SD（ddof=1）。暫定目標はFinal平均≤.10かつ持続3/3で、全seedの誤差≤.10や聴覚的同一性の基準ではない。持続proxyは末尾1000sampleのAC標準偏差/Target peak>.05かつ数値失敗なし。生理学的な有声判定とは別。失敗・非持続seedを除外しない。

![33点の独立評価](figures/evaluation_curves.png)

![各endpointと安定性](figures/endpoints_and_stability.png)

対応seedのFinal差（条件−参照）。Jの差で改善を判定せず、共通誤差と成分を比べる。

{table(['条件','参照','ΔE_total 平均±SD','ΔE_spec 平均±SD','ΔE_env 平均±SD'],pair_table)}

## 全seedとSelected−Final

**Selected−Final = E_total(Selected minJ)−E_total(Final)**。負ならFinalの共通誤差が悪化。正ならFinalの共通誤差はSelectedより良いが、採用基準は異なるので矛盾しない。`selected_minus_final_J`はminJ選択のため常に0以下。posthoc−Final/posthoc−Selectedも別列に記録する。

例えばλ=.90・seed1のSelectedは試行3200でJ=0.19595、E_spec=0.21100、E_env=0.06053、E_total=0.27153。FinalはJ=0.20005、E_spec=0.22080、E_env=0.01329、E_total=0.23409だった。SelectedはJの基準では正しく、Selected−Finalの共通誤差は+0.03744。この点を共通誤差の小さいFinalやposthocへ差し替えない。

{table(['条件','seed','Final E','Selected E','posthoc E','Selected試行','posthoc試行','Selected−Final E','Final持続','Selected持続'],individual)}

[全成分・J・制御・物理誤差](comparison_per_seed.csv) / [対応差](paired_differences.csv) / [安定性の符号付き差](selected_final_stability.csv) / [33点の独立評価](policy_evaluations.csv)。物理量NRMSEは従来の全域幅と今回の制限幅を区別し、音響的な一致から制御値の一意な復元は主張しない。

![物理制御の集約](figures/physical_controls.png)

## 温度・分散・entropy

新規runでは全33評価点でα、pre-tanh Gaussianのmean/log_std/std、tanh後の正規化行動標準偏差、Gaussian entropy、squashed entropy推定を保存。後者は固定の局所NumPy RNG・4096標準正規サンプルとSB3と同じtanh Jacobian補正（epsilon1e-6）を使う。グローバルRNG、学習器、optimizer、bufferの評価前後署名が一致し、追加の鳴管音は0。分散は条件付き方策分布の推定で、探索ノイズの設定を変えた値ではない。

entropyは正規化行動座標の連続分布の微分entropy（nats）で負になり得る。epsilon付き推定と有限MCの誤差を含む。MC SEは各点の積分誤差の目安で、3seedのSDや一般化の信頼区間ではない。同じMCサンプルを全条件で使う。旧条件には33点の温度/分散履歴がないため、保存Final/Selectedモデルから得た実測2点/seedだけを表示し、補間しない。

{table(['条件','採用','α 平均±SD','squashed entropy 平均±SD [nats]'],entropy_table)}

Final平均αは旧0.01048に対してλ=.50/.75/.90で0.00700/0.00996/0.00429だった。損失が半分になるλ=.50でも、温度が旧条件の正確な半分へ自動調整されたわけではない。λ=.75のFinal平均entropy推定は−4.849 nats（旧−3.524）で、座標別の行動分散も変化した。これらは実際の方策状態の違いであり、温度・分散の変化だけを性能差の原因と断定しない。

![方策と温度の変化](figures/policy_dynamics.png)

![学習報酬の大きさと温度](figures/reward_temperature_scale.png)

学習サンプルと独立評価は分離。[全学習サンプルCSV](training_samples.csv.gz)には実際のfloat32報酬と当該更新前α、[方策診断CSV](policy_dynamics.csv)には評価後の方策状態を保存。報酬図は128学習サンプルのbin平均。α/平均|r|は相対スケールの記述であり、entropy支配や性能差の原因を証明する比ではない。

終端1stepなのでCritic targetのbootstrap/次状態entropyは消えるが、SAC Actorの現在行動の `α logπ−minQ` は残る。報酬の縮小はQとentropyの相対関係にも影響し、auto温度が常に同じ比を保つ保証はない。今回観測した温度・分散・安定性の関連は、重み以外を介入していない回顧的な観測であり、原因同定を先取りしない。

## 補助Critic診断

同じ既存1005候補プールに対して各条件の **r=−J** とQ1/Q2/minQを比較した。学習・新しい候補抽選・鳴管再計算は0。旧格子・Random・DEの保存診断点と既知oracleを含む層別プールで、統計的に独立な母集団や物理空間の発生確率ではない。各λの下位J10%は異なる点を含み得る。

以下は3seedで各指標を計算した平均。**raw RMSE/biasは異なる報酬尺度の間で単純比較しない**。RMSE/報酬SDも目標関数と順位が違うため、因果的な精度改善の証明にはならない。順位反転は真値/予測の同値対を除く。低J biasはQ−rで、負なら過小評価。

{table(['条件','Critic','raw RMSE','RMSE/報酬SD','Spearman','反転率','低J bias'],ctable)}

![自身の報酬とのCritic整合](figures/critic_calibration.png)

[seed別Critic指標](critic_metrics.csv) / [全予測](critic_predictions.csv.gz) / [実行条件](critic_execution.json)。図の各パネルは自身のrの軸スケールを使う。SACの決定論的出力と確率的Actorの目的も同一視しない。

## 代表音と保存

全条件でseed1を固定し、FinalとSelectedを別図にした。Target最大full-band PSDに対する共通dB範囲・カラーバー、共通Target peak基準の5ms RMS、位相合わせなし。40–60msの波形は表示だけの切出しで、保存音/報酬は全0.20秒。PSD表示は10MHz、Blackman-Harris窓245760/overlap184320、有効窓のみ、0–10kHz。色/振幅軸は全24保存Final/Selectedで共通。[図の出所](figure_sources.json)。

![Final代表音](figures/audio_final.png)

![Selected minJ代表音](figures/audio_selected.png)

新規9runのFinal/Selectedモデル、float64元振幅NPZ、全4096学習行動と33方策評価を保持。聴取は全音に同じTarget由来gainを適用した[WAV ZIP](new_listening_audio.zip) / [対応表](listening_audio.json)。完全同一WAVは共用し、旧モデル/音声は[元ファイルを参照](endpoint_sources.json)してコピーしない。posthocは保存33点の行動と数値の診断で、対応時点の追加model checkpointを後から作ったとは扱わない。

## 実行・監査・保護

新規9run×4096=36864学習音、独立方策評価297音、学習後replay18音。事前整合4音、テスト用モデル評価145回（注入失敗1回）は別枠。監査の実鳴管再生は今回{audit['generated_sounds']}音・累計{audit['cumulative_generated_sounds']}音で、同一行動を再利用して余分な再計算を抑えた。保存Final/Selectedモデルのreload24回、posthocを含む36endpointの制御・誤差・発声を照合。Critic、温度診断、描画、WAV化は追加鳴管音0。wall時間{execution['wall_s']:.2f}秒、worker数{execution['workers']}。[実行記録](execution.json) / [run別時間](run_accounting.csv) / [監査](audit.json) / [再生記録](endpoint_replays.csv) / [17テスト](tests.json)。

着手前に旧成果・既存コード{protected}ファイルをhash化し、変更/削除0を確認。root READMEとignoreだけを更新した。途中checkpointやseedごとのPNGを大量生成せず、主要図8枚。旧Issue17〜21の図・モデル・数値・目標は変更しない。commit/push、Git履歴書換え、Issueのクローズは行っていない。[最終整合](integrity.json) / [成果物hash](artifact_checksums.json) / [変更前](storage_before.json) / [変更後](storage_after.json)。

## 限界と次Issue

{adoption}

3seed・固定人工目標の比較であり、聴覚的一致、物理制御の一意性、実音声への一般化や一般的優位は未確定。発声proxyとスペクトル精度を分けて読む。f0_l≤800000は人工持続音の診断的制約で、無音/閉鎖や実録音、Case Bへ一般化しない。Selected→Finalの保持・Replay優先化・変化上限・探索/entropyハイパーパラメータ変更は別Issueとし、本Issueでは追加実験していない。[元の指示](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/22)。
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'README.md').write_text('''# Issue22 — SAC128の報酬係数和1比較

[結果と採用判断](REPORT.md)。λ=.50/.75/.90の9run、旧Issue21の1+1報酬はreadonly参考。

- [条件・source hash](manifest.json)、[係数和と物理の事前検証](preflight.json)
- [全seed](comparison_per_seed.csv)、[平均±標本SD](comparison_aggregate.csv)、[対応seed差](paired_differences.csv)
- [Selected−Final](selected_final_stability.csv)、[33点の方策評価](policy_evaluations.csv)
- [α・分散・entropy](policy_dynamics.csv)、[学習サンプル](training_samples.csv.gz)
- [補助Critic指標](critic_metrics.csv)、[診断条件](critic_execution.json)
- [モデル/行動再生監査](audit.json)、[旧成果・リンク・CSV整合](integrity.json)、[成果物hash](artifact_checksums.json)
- [図8枚の出所](figure_sources.json)、[モデル/波形の出所](endpoint_sources.json)
- [聴取用WAV ZIP](new_listening_audio.zip)、[共通gainと対応表](listening_audio.json)
- [再現手順](REPRODUCE.md)、[変更前](storage_before.json)、[変更後](storage_after.json)
''')
    (OUT/'REPRODUCE.md').write_text('''# 再現・検証

リポジトリ直下、既存のDeeplearning/.venv-rlを使う。prepareは既存出力先を拒否し、未完了runも上書きしない。完了済みrunはsourceと成果物hashが合う場合のみ再利用し、全実行済みのtrainは元execution.jsonを保持して終了する。旧成果を更新・再学習しない。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue22-mpl
# 本計算前、元ファイルのhash snapshotを保存してから新規prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_prepare --before /tmp/birdsong-issue22-before.json
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_training --workers 9
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_analysis critic
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue22_report
```

保存済みモデル/行動の再生はanalysis audit（実再生回数を別記）、鳴管計算なしのチェックはanalysis integrity。描き直し/表の再計算はissue22_report、同じ候補のQ推論だけならanalysis critic。報告段階はいずれも学習しない。

新規実験では、着手前のworkspace数・Git状態・既存研究ファイル/既存コードのpath/bytes/sha256をJSON（storage_before/protected_beforeのschema）に記録し、--beforeへ渡す。別出力先はprepare/trainingの--outputで指定し、解析のOUTは別checkoutでその新規先に固定してからprepareする。元の正本は保持する。

新式のlambdaはスペクトル係数、包絡は1-lambda。旧式のenvelope_weightではない。実効SAC設定/層構造は各runのconfig.json、全4096学習サンプルはtraining_trace.json.gz、33方策評価はcurve.json.gz。元のfloat64波形はfinal.npz/selected.npz、対応重みはagent.zip/selected_agent.zip。posthocは曲線のmin common Eの点・行動診断で、対応する途中modelを追加保存したものではない。

α/分散/entropyは評価曲線のpolicy_statisticsへ保存。固定MC4096点、ローカルseed202622、SB3と同じtanhのepsilon1e-6補正。新しい環境音を生成せず、グローバルRNG/学習状態を変えない。旧条件の温度等は保存Final/Selectedの2点/seedのみ。WAVは全新規音で同じTarget peakの逆数をgainとし、個別正規化やクリップをしない。
''')
    write_json(OUT/'report_execution.json',dict(wall_s=time.perf_counter()-started,generated_sounds=0,images=len(images),source_sha256=digest(__file__)))
    print('Tables and eight figures generated; checking final integrity and freezing artifact hashes',flush=True)
    freeze()


if __name__=='__main__':report()
