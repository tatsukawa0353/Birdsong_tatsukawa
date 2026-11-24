"""Report the Critic LR intervention with Final-first conclusions and offline artifacts."""
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from .issue25_data import ROOT,BASE,OUT,TARGET,RATES,BASE_LR,SEEDS,folder_for,offline_folder,target_data,ready,OFFLINE_UPDATES
from .issue25_analysis import csv_rows,tables,integrity
from .issue23_stage_a import table as plain_table
from .issue19_protection import workspace_inventory
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def table(headers,rows):
    escape=lambda value:str(value).replace('|','\\|')
    return plain_table([escape(h) for h in headers],[[escape(v) for v in row] for row in rows])


def listening():
    target,_=target_data();gain=1/float(np.max(abs(target)));seen={};index=[]
    with zipfile.ZipFile(OUT/'listening.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        def add(audio,name,rate,seed,role,path,key):
            wave=(audio*gain).astype(np.float32);sha=hashlib.sha256(wave.tobytes()).hexdigest();entry=seen.get(sha)
            if entry is None:
                entry=name;buffer=io.BytesIO();wavfile.write(buffer,50000,wave);archive.writestr(entry,buffer.getvalue());seen[sha]=entry
            index.append(dict(critic_lr=rate,seed=seed,role=role,archive_entry=entry,shared_gain=gain,sample_sha256=sha,
                raw_audio_path=str(path.relative_to(ROOT)),raw_audio_key=key))
        add(target,'target.wav','','','target',TARGET,'target')
        for rate in RATES:
            for seed in SEEDS:
                for role in ('final','selected','posthoc','first_success'):
                    path=folder_for(rate,seed)/f'{role}.npz'
                    if not path.exists():continue
                    with np.load(path) as z:audio=z['audio']
                    add(audio,f'lr_{rate:.0e}_seed_{seed}_{role}.wav',rate,seed,role,path,'audio')
    write_csv(OUT/'listening_index.csv',index);return len(seen)


def freeze():
    checks=OUT/'artifact_checksums.json';write_json(checks,{})
    write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in
        [*sorted(ROOT.glob('issue25*.py')),ROOT/'issue25_findings.md',ROOT/'tests/test_issue25.py']})
    integrity();write_json(OUT/'storage_after.json',{})
    repository=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    for _ in range(5):
        status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True)
        new=[line[3:] for line in status.splitlines() if line.startswith('?? ')]
        record=dict(workspace=workspace_inventory(),new_git_candidates=len(new),new_git_candidate_bytes=sum((repository/p).stat().st_size for p in new),
            modified_tracked_files=sum(not line.startswith('?? ') for line in status.splitlines()),deleted_tracked_files=sum('D' in line[:2] for line in status.splitlines()),
            measurement='Logical regular files including ignored; .git/symlinks excluded; nothing staged/committed')
        write_json(OUT/'storage_after.json',record)
        before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total'];after=record['workspace']['total']
        block='<!-- ISSUE25_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[
            ['作業ツリー通常ファイル数',before['files'],after['files']],['論理容量（ignored含む）',f"{before['bytes']/1e6:.2f} MB",f"{after['bytes']/1e6:.2f} MB"],
            ['Git新規追加候補','0（クリーン）',f"{record['new_git_candidates']}個 / {record['new_git_candidate_bytes']/1e6:.2f} MB"],
            ['既存追跡ファイルの変更 / 削除','0 / 0',f"{record['modified_tracked_files']} / {record['deleted_tracked_files']}"]])+'\n<!-- ISSUE25_STORAGE_END -->'
        p=OUT/'REPORT.md';body=p.read_text();p.write_text(re.sub(r'<!-- ISSUE25_STORAGE_BEGIN -->.*?<!-- ISSUE25_STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- ISSUE25_STORAGE_BEGIN -->' in body else body+'\n'+block+'\n')
        write_json(checks,{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    record['workspace']=workspace_inventory();write_json(OUT/'storage_after.json',record)


def report():
    ready();tick=time.perf_counter();tables()
    endpoints=csv_rows(OUT/'comparison_per_seed.csv');groups=csv_rows(OUT/'comparison_summary.csv');stability=csv_rows(OUT/'stability_and_acquisition.csv')
    motion=csv_rows(OUT/'movement_per_seed.csv');paired=csv_rows(OUT/'paired_differences.csv');critics=csv_rows(OUT/'critic_metrics.csv')
    off=csv_rows(OUT/'offline_final_metrics.csv');offranks=csv_rows(OUT/'offline_ranks.csv');ranks=csv_rows(OUT/'critic_ranks.csv')
    dataset=json.loads((OUT/'dataset_audit.json').read_text());audit=json.loads((OUT/'audit.json').read_text());oaudit=json.loads((OUT/'offline_audit.json').read_text())
    execution=json.loads((OUT/'execution.json').read_text());oexecution=json.loads((OUT/'offline_execution.json').read_text())
    fmt=lambda r,k:f"{float(r[k+'_mean']):.5f} ± {float(r[k+'_sd']):.5f}" if r[k+'_mean'] else '未定義（失敗seed含む）'
    group_table=table(['Critic LR','採用','E_total平均±SD','E_spec平均±SD','E_env平均±SD','J平均±SD','持続/3','失敗run'],
        [[r['critic_lr'],r['role'],*[fmt(r,k) for k in ('E_total','E_spec','E_env','J')],r['voiced'],r['failed_runs']] for r in groups])
    endpoint_table=table(['Critic LR','seed','採用','試行','E_total','E_spec','E_env','J','持続'],
        [[r['critic_lr'],r['seed'],r['role'],r['trial'],*[f"{float(r[k]):.7f}" if r[k] else '失敗/欠測' for k in ('E_total','E_spec','E_env','J')],r['sustained_voiced']] for r in endpoints])
    acquisition=table(['Critic LR','seed','学習でE≤.10 /4096','Actor初回E≤.10','Actor成功数/33','Final成功','到達後保持率','Selected−Final E'],
        [[r['critic_lr'],r['seed'],r.get('training_threshold_count','失敗'),r.get('first_E_le_0p10') or '未到達',r.get('voiced_policy_threshold_count','欠測'),r.get('final_success','欠測'),
            f"{r['retention_successes']}/{r['retention_evaluations']}={float(r['retention_fraction']):.3f}" if r.get('retention_fraction') else '未定義',
            f"{float(r['selected_minus_final_E']):+.6f}" if r.get('selected_minus_final_E') else '欠測'] for r in stability])
    parallel=[];offpairs=[];gradient_summary=[]
    def find(data,rate,seed,**filters):
        return next((r for r in data if float(r['critic_lr'])==rate and int(r['seed'])==seed and all(r[k]==v for k,v in filters.items())),None)
    for rate in RATES:
        for seed in SEEDS:
            e=find(endpoints,rate,seed,role='final');q=find(critics,rate,seed,subset='all99',head='minQ')
            t=find(off,rate,seed,split='train',subset='all',head='minQ');h=find(off,rate,seed,split='holdout99',subset='all',head='minQ')
            low=find(off,rate,seed,split='holdout99',subset='lowJ_decile',head='minQ');rank=find(offranks,rate,seed)
            parallel.append(dict(critic_lr=rate,seed=seed,online_Final_E=e['E_total'] if e else None,
                online_fixed99_RMSE=q['rmse'] if q else None,online_fixed99_rho=q['spearman'] if q else None,
                offline_train_RMSE=t['rmse'] if t else None,offline_holdout_RMSE=h['rmse'] if h else None,
                offline_holdout_rho=h['spearman'] if h else None,offline_lowJ_RMSE=low['rmse'] if low else None,
                offline_lowJ_bias=low['bias_Q_minus_r'] if low else None,offline_true_best_rank=rank['true_best_rank'] if rank else None))
            if rate!=BASE_LR:
                base=find(off,BASE_LR,seed,split='holdout99',subset='all',head='minQ')
                offpairs.append(dict(critic_lr=rate,seed=seed,delta_holdout_RMSE=float(h['rmse'])-float(base['rmse']) if h and base else None,
                    delta_holdout_rho=float(h['spearman'])-float(base['spearman']) if h and base and h['spearman'] and base['spearman'] else None))
    gradients=csv_rows(OUT/'online_gradient_comparison.csv')+csv_rows(OUT/'offline_gradient_comparison.csv')
    for kind in ('online','offline'):
        for rate in RATES:
            for step in (.002,.01):
                rows=[r for r in gradients if r['kind']==kind and float(r['critic_lr'])==rate and float(r['step'])==step]
                if not rows:continue
                errors=np.array([float(r['analytic_minQ_gradient'])-float(r['true_centered_reward_gradient']) for r in rows])
                gradient_summary.append(dict(kind=kind,critic_lr=rate,step=step,components=len(rows),
                    rmse=float(np.sqrt(np.mean(errors**2))),mae=float(np.mean(abs(errors))),sign_agreement=sum(r['sign_agreement']=='True' for r in rows)/len(rows)))
    write_csv(OUT/'parallel_online_offline.csv',parallel);write_csv(OUT/'offline_paired_differences.csv',offpairs);write_csv(OUT/'gradient_summary.csv',gradient_summary)
    joint_table=table(['Critic LR','seed','Online Final E','Online固定99 RMSE','Onlineρ','Offline train RMSE','Offline holdout RMSE','Offlineρ','Offline lowJ RMSE','lowJ bias','最良候補順位'],
        [[r['critic_lr'],r['seed'],*[f"{float(r[k]):.5f}" if r[k] is not None and r[k]!='' else '欠測' for k in
            ('online_Final_E','online_fixed99_RMSE','online_fixed99_rho','offline_train_RMSE','offline_holdout_RMSE','offline_holdout_rho','offline_lowJ_RMSE','offline_lowJ_bias')],r['offline_true_best_rank']] for r in parallel])
    wavs=listening()
    from .issue25_figures import render
    images=render();findings=(ROOT/'issue25_findings.md').read_text()
    new_summaries=[json.loads((folder_for(r,s)/'summary.json').read_text()) for r in RATES if r!=BASE_LR for s in SEEDS]
    native=sum(s['training']['native_steps_completed']+s['evaluation']['native_steps_completed'] for s in new_summaries)+sum(a['native_steps_completed'] for a in audit['accounts'])
    protected=len(csv_rows(OUT/'protected_before.csv'));sounds=execution['training_candidates']+execution['independent_policy_evaluations']+audit['generated_sounds']
    goals=[r['critic_lr'] for r in groups if r['role']=='final' and r['final_goal_met']=='True']
    goal='暫定目標（Final平均E_total≤.10かつ持続3/3）を満たしたCritic LR：'+', '.join(goals)+'。' if goals else '今回、Final平均E_total≤.10かつ持続3/3の暫定目標を満たす条件はなかった。'
    body=f'''# Issue25 — Critic LR：オンラインSACと固定データ回帰を分けた比較

## 結果と判断

{goal}

{findings}

## Stage Aと変更要因の監査

[計算前に凍結したPLAN](PLAN.md)と[manifest](manifest.json)に、データ分割・更新数・選択規則・数値停止基準・局所追加0音を記録した。Issue24が参照した基準manifestがIssue23の完全再現済み3seedと一致することも照合。基準Critic3e-4は読み取り専用で参照し、新規フル学習0。新規オンラインはCritic1e-4/1e-3×3seedの6run、Actor/温度/共通scheduleは3e-4。

専用CriticRateSACはSAC.trainを標準実装から継承し、_update_learning_rateの標準処理の後にCritic optimizerだけを指定値へ戻す。Actor/温度LR・Adam設定・構造・更新則は変えない。初期の実効設定はCritic LRを基準に戻すと全項目一致。初期policy・RNG・温度が一致し、warmup16の行動/報酬も一致した。基準LRでの短い同seed検査では重み・optimizer・RNG・Replay（行動/報酬含む）が完全一致。Critic LR変更後はActor勾配や経験分布も変わり得るため軌跡一致は要求しない。[事前検証](preflight.json) / [実効LR全履歴](optimizer_learning_rates.csv.gz)。

全optimizer更新直前/直後と33評価点でLRを検査。勾配/重み/Adam状態を観測し、非有限値、負の二次moment、または勾配L2>1e6・重み絶対値>1e6・moment絶対値>1e12で停止する。これは緊急停止基準で、クリップや収束判定ではない。失敗時は使用予算・失敗理由を保持し、条件を削除/置換せず、途中方策をFinalにしない。実際のオンラインは完了{execution['complete_runs']}、失敗{execution['failed_runs']}run。[run会計](run_accounting.csv) / [実行記録](execution.json)。

## 固定した課題・目標

Case A（右音源の寄与α_r=β_r=0）、一定3D、1episode1行動、0.20s/50kHz/10000sample、native dt=1e-7、x0=.02cm/D=.01、初期状態/遅延履歴0。Target・物理/音響コード・実ライブラリhashは既存と同じ。ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]、右ε_r=3.5e7/f0_r=49000。f0_lは基本周波数Hzではなくラビア開閉外力係数。模型内部量を実測Paや張力と同一視しない。

報酬r=−J、J=.75E_spec+.25E_env、係数和1・2倍補正なし。共通主評価E_total=E_spec+E_env。SAC Actor/Critic[128,128]、Actor/温度LR3e-4、ent_coef=auto_0.01、target_entropy=-3、batch32、buffer50000、warmup16、gamma1、train_freq1、gradient_steps1、tau/target更新/Replay方式/CPU1threadを固定。範囲・entropy目標・ノイズ・優先Replay・batch・幅・報酬・Case B・時変・実録音・教師づけ・最良方策への復帰を追加していない。

## 実験I — オンラインSAC：Finalが主評価

Final=4096候補の最終更新後。Selected=0,128,...,4096の33独立評価の最初のminJ。posthoc=同33点の最初のmin共通Eで、学習目標とは別の事後選別。初回のE≤.10かつ持続方策も必要最小限の追加checkpointとして保持する。独立評価はReplayに入れず、保存方策へ戻さない。

{group_table}

3seed平均±標本SD（ddof=1）。失敗runを含む条件の3seed Final平均は未定義とし、完了seedだけの平均へ置換しない。持続proxyは末尾1000sampleのAC標準偏差/Target peak>.05かつ数値失敗なしで、生理的な有声判定ではない。暫定目標はFinal平均≤.10かつ持続3/3で、途中最良や全seed個別の閾値とは別。[全成分と制御CSV](comparison_per_seed.csv) / [対応seed差](paired_differences.csv)。

![全seed・4誤差・共通の軸](figures/error_histories.png)

![Finalと保存基準を区別](figures/endpoints_and_stability.png)

{endpoint_table}

### 到達と保持・Replay経験

{acquisition}

学習経験の良い1候補、独立評価の決定論的Actorの到達、Finalの維持を区別する。保持率は初回E≤.10かつ持続の評価点からFinalまでの成功点数/全評価点数（初回含む）。未到達は未定義、33点の間の未評価状態を捏造しない。純E閾値と持続条件付きの両方は[到達CSV](stability_and_acquisition.csv)に残す。Selected−Finalが負ならFinalの共通Eが悪いが、minJとminEの違いも別途解釈する。

全Actor更新前後の固定観測からのtanh(mean)、正規化3D行動差・物理差・3optimizer LR・数値状態・実際のbatch32 lossを記録した。Δ物理量=Δa×[12500000,750000,375500]で、単位の異なる差を直接L2合成しない。128候補の経路長Σ||Δa||と正味距離||a_end−a_start||を分ける。Actor optimizerの前後では当該stepのCritic/温度更新は済んでおり、決定論的1点のQと、確率的batchで計算したActor lossは同じではない。[更新要約](movement_per_seed.csv) / [全更新CSV](actor_updates.csv.gz) / [区間差・loss](update_intervals.csv)。

alpha/std/entropyは同じ33時点で記録。tanh後stdとentropyはlocal NumPy seed220026の同じ4096サンプルで推定し、学習RNGを消費しない。alphaを探索stdと同一視せず、微分entropyは負になり得る。[方策推移](policy_dynamics.csv)。

![温度と座標別std](figures/temperature_and_exploration.png)

### オンラインCritic

同じ既存99行動のQ1/Q2/minQを全33時点で記録した。全99、lowJ10、同seedの基準33候補、実際の自身の学習4096経験、直近1024、学習lowJ10を分ける。訪問経験への当てはまりと、旧候補への予測は同じ集合ではない。[指標CSV](critic_metrics.csv) / [時系列](critic_time_metrics.csv) / [最良候補の順位](critic_ranks.csv) / [自身の訪問点予測](own_visit_predictions.npz)。

![オンラインFinal Q1/Q2](figures/final_critic_calibration.png)

## 実験II — 固定データ上のCritic回帰

既存Issue23の学習経験3seed×4096={dataset['train_points']}件をtrain、独立評価済み99行動をholdoutに隔離した。Issue23はIssue22 λ=.75の完全再現なので両方を重複投入しない。各回帰seedは同じ全3source seedのtrainを使い、初期Criticとbatch列だけのseedを指す。[データ監査](dataset_audit.json) / [点ごとのhash・出典](offline_points.csv.gz) / [固定配列](offline_data.npz) / [固定batch列](offline_batches.npz)。

raw行動とCritic入力のtrain内・holdout内完全重複は0、train/holdout完全重複も0。ただし近傍重複は存在し、holdoutからtrainへの全域幅比L2距離の中央値は{dataset['nearest_holdout_distance_quantiles']['0.5']:.7f}、最小{dataset['nearest_holdout_distance_quantiles']['0']:.7f}。同じTarget・旧方策軌道・領域の偏りを共有し、独立な新Targetや一様3D汎化テストとは呼ばない。[全候補の近傍距離・半径内経験数](holdout_nearest_neighbors.csv)。

holdout低J10%のJ境界は{dataset['lowJ_holdout_decile_cutoff']:.7f}。それ以下のtrain点は{dataset['train_at_or_below_holdout_lowJ_cutoff']}件（{100*dataset['train_at_or_below_holdout_lowJ_cutoff']/dataset['train_points']:.2f}%）で、低J密度は偏る。これはこの訪問データ内の比率であり、物理領域の体積確率ではない。補充・低J重み付け・holdoutの学習混入は0。

MLP/twin-Q/入力11特徴量（reset観測8＋正規化行動3）・Adam・batch32・損失・4080更新を固定し、LR3e-4/1e-4/1e-3×3seedの9本を学習した。各seed内で初期Criticとmini-batch index列のhashが全LR一致。batchは事前local seed250025+seedの一様復元抽出、全LRで同じ列。目標はSB3 Replayと同じfloat32(r)、損失0.5*(MSE(Q1,r)+MSE(Q2,r))。報告誤差は元float64 rに対して計算する。1step終端なのでbootstrap/次状態entropyは0。

回帰のActor/温度/target Criticは更新せず、Actorへ接続しない。グローバルRNGも学習前後一致。更新数・LRはholdout結果で変更せず、早期停止や最良holdout checkpoint選抜もない。33診断時点の最後は4080**回帰更新**で、オンラインの4096候補とは別の軸。[全回帰時系列](offline_history.csv) / [Final全指標](offline_final_metrics.csv) / [対応差](offline_paired_differences.csv) / [9本の実行数](offline_execution.json)。

![固定回帰のtrain/holdout・低J・順位](figures/offline_learning_curves.png)

![offline Final holdout Q1/Q2](figures/offline_holdout_calibration.png)

## 並列の評価軸：Critic誤差とFinal音響精度

{joint_table}

列は異なる目的で、足し合わせた総合scoreを作らない。online固定99とoffline holdout99は同じ真のr、offline trainは固定12288、online自身の経験は各runで異なる4096である。RMSE/MAE/bias/ρ/誤順位率を全て保存し、lowJ subsetも別列にした。訓練誤差の低下を未知領域の汎化やSACのFinal改善と呼ばない。[並列表CSV](parallel_online_offline.csv)。

onlineは各seed個別に増えていくReplayを使う。offlineは3source seedを混合した12288経験が最初から全て利用可能で、batch列もonlineとは別である。両実験の誤差差を、LRだけの因果差やActorを取り除いた純粋な効果とは扱わない。LRの1要因比較は各実験内で成立する。

lowJ10は各集合内の上位10%を指す。train/holdout/各online訪問経験ではJ境界が異なるため、同じ名前でも同じ点集合ではない。固定99とoffline holdout99のlowJ10だけは同一集合である。

![異なる評価軸を分けた比較](figures/parallel_online_offline_outcomes.png)

今回の3LR・4080更新・3seedは実行前に固定したが、このholdoutの結果から次の候補を選ぶ議論は事後の候補選びであり、新しい独立検証で確定した最適LRではない。

## 局所勾配の参照範囲

新規局所鳴管計算は0。Issue23の既存36音（seed2の2944/3840/4096行動、正規化刻み±.002/.01）をhash固定して参照した。現在のonline/offline Criticをその同じ固定点で微分し、旧真の中心差分と比較した。新条件の訪問点の真の勾配を測ったとは言わない。minQの折れ点、有限差分の刻み・非線形性のため、中心autogradと有限差分は同じ量の厳密測定ではない。[online勾配](online_gradient_comparison.csv) / [offline勾配](offline_gradient_comparison.csv) / [集約](gradient_summary.csv)。

![既存anchorだけの勾配比較](figures/legacy_anchor_gradients.png)

## 波形・重み・音・監査

オンラインは{audit['model_reloads']}保存方策をロードし、異なる行動だけで{audit['generated_sounds']}音を独立再生。行動・20ブロック制御・元float64波形・E/J・alpha/std・Finalの99予測・勾配が一致した。offlineは{oaudit['model_reloads']}個のCritic-onlyをロードし、train/holdout全予測と勾配、実際の更新数・batch/初期hashを照合した。ロギング・独立評価・Q照会・保存前後の学習器/optimizer/RNG/Replay署名も一致。[オンライン監査](audit.json) / [offline監査](offline_audit.json)。

オンライン推論用Actor+online twin Criticと、offline Critic-onlyは別の保存形式。どちらもoptimizer/Replayを含む再開学習用完全checkpointではない。offline CriticをSAC Actorへ接続していない。[オンラインmodel/audio索引](checkpoint_index.json) / [offline model索引](offline_checkpoint_index.json)。

[聴取ZIP](listening.zip)はTargetと全保存endpoint・新規初回成功方策を含み、同一波形を重複除去して{wavs}音。[索引](listening_index.csv)あり。全音に同じTarget peak逆数gainを掛けたfloat32で、個別正規化・クリップ・位相合わせは0。元float64波形/制御は各run NPZ、代表図は事前固定seed1で共通PSD/色/振幅/RMS軸を使う。10MHzへの補間は表示のみで新たな周波数情報は増えない。

![seed1 Final](figures/audio_final.png)

![seed1 Selected/minJ](figures/audio_selected.png)

## 実行会計・保護・再現

オンライン6run壁時間{execution['wall_s']:.1f}s、実学習候補{execution['training_candidates']}、独立評価{execution['independent_policy_evaluations']}、再生{audit['generated_sounds']}、計{sounds}音、完了native step{native:,}。基準の再学習0、追加局所音0。offlineは完了{oexecution['complete_runs']}・失敗{oexecution['failed_runs']}、実更新{oexecution['updates']}、壁時間{oexecution['wall_s']:.1f}s、native/Actor/温度更新0。2実験の更新・計算量を混ぜない。

8種類のテスト成功。nativeテスト171音=145学習候補（うち停止注入run17候補）+14独立評価+12再生。高LR初期nativeテストは3seed各32候補で成功し、停止注入は研究結果ではない。合成終端env179episode、短いofflineテスト3本計96更新はnative0。圧縮ログのreader引数を本計算前に修正し、該当offlineテストを再実行した。[テスト記録](test_execution.json)。

既存{protected}ファイルのhash一致、変更/削除0。root READMEと.gitignoreのみ索引/作業ログ除外を追加。主図10枚、全成果物は専用ディレクトリ、commit/push/Issue更新は未実施。[元成果保護](integrity.json) / [実装hash](implementation_checksums.json) / [成果物hash](artifact_checksums.json) / [再現・オフライン確認](REPRODUCE.md)。
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'README.md').write_text('''# Issue25 — Critic LR：オンラインと固定データを分離

[REPORT](REPORT.md) / [計算前PLAN](PLAN.md) / [再現手順](REPRODUCE.md) / [並列の評価軸](parallel_online_offline.csv) / [WAV ZIP](listening.zip)

Actor/温度LRを固定したSAC6runと、既存データ固定のCritic-only9本。Final音響精度とCritic予測精度を混同しない。基準と旧成果物は読み取り専用。
''')
    (OUT/'REPRODUCE.md').write_text('''# 再現とオフライン確認

リポジトリrootで環境を指定する。新規計算は別checkout/新規OUTと着手前hash snapshotを使い、既存正本のprepareや未完了runの上書きは拒否する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue25-mpl
Deeplearning/.venv-rl/bin/python -m unittest Deeplearning.tests.test_issue25 -v
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_prepare --before /tmp/birdsong-issue25-before.json
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_training --workers 6
# Stage Bが終わってからStage C。offlineは鳴管計算・Actor更新0
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_offline --workers 3
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis offline_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue25_analysis integrity
```

prepareの前に、workspace数・git状態・旧results/reports/追跡コードのpath/bytes/SHA256を記録したJSONが必要。共通learning_rateを変更せず、issue25_agent.make_agentを使う。基準3e-4も同じscheduler/標準SAC.trainで比較検証した。実効param_groups LRは全step/evalで検査する。

帰省中はREPORT・10図・CSV・聴取ZIPを開けば追加計算不要。policy_evaluations.csvは各33点、policy_dynamics.csvは温度/std、actor_updates.csv.gzは1更新の行動/物理差とloss、optimizer_learning_rates.csv.gzは3実効LR（既存基準はprovenanceと区別）。各runのcurve/updates/q_historyには保存後に復元できない中間診断がある。minJ/minE/Final/first_successを混同しない。

offline_data.npzとoffline_batches.npzが固定train/holdoutと全batch列。offline_points.csv.gzが各点の出典/hash、dataset_auditとholdout_nearest_neighborsが重複・近傍・偏りを示す。offlineのseedは初期化とbatch列のseedで、全3source seedのtrainを毎回共有する。holdoutは更新に使わず、全LRとも4080回帰更新。offlineはcritic.ptのみ、Actorを含まない。

オンライン.ptは共用Issue23 reader形式のActor+online twin Criticで、issue23_data.load_checkpoint(path,seed,manifest['settings'])で推論できる。loaderのoptimizerは仮初期化で学習LR/状態の復元ではない。offlineは同じSAC構造のcritic.load_state_dictで読めるが、オンラインActorへ接続する実験は本Issueでしていない。どちらもoptimizer/Replayを含まず再開学習用ではない。

report/tables/critics/offline_auditは学習や鳴管音を生成しない。online auditは保存方策の独立再生を行うので、再実行回数は別に記録する。integrityはhash/link/WAV gainを検査するだけ。WAVは全音共通gain=1/Target peak、元波形はfloat64 NPZ。f0_lはHzでなく開閉外力係数。
''')
    write_json(OUT/'report_execution.json',dict(generated_sounds=0,learning_updates=0,images=10,source_sha256=digest(__file__),findings_sha256=digest(ROOT/'issue25_findings.md'),wall_s=time.perf_counter()-tick))
    print('Online/offline report,ten figures and listening archive complete; freezing integrity and inventory',flush=True);freeze()


if __name__=='__main__':report()


