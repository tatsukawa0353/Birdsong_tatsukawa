"""Report the Actor LR intervention with Final-first conclusions and offline artifacts."""
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

from .issue24_data import ROOT,BASE,OUT,TARGET,RATES,BASE_LR,SEEDS,folder_for,target_data,ready
from .issue24_analysis import csv_rows,tables,integrity
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
            index.append(dict(actor_lr=rate,seed=seed,role=role,archive_entry=entry,shared_gain=gain,sample_sha256=sha,
                raw_audio_path=str(path.relative_to(ROOT)),raw_audio_key=key))
        add(target,'target.wav','','','target',TARGET,'target')
        for rate in RATES:
            for seed in SEEDS:
                for role in ('final','selected','posthoc'):
                    path=folder_for(rate,seed)/f'{role}.npz'
                    with np.load(path) as z:audio=z['audio']
                    add(audio,f'lr_{rate:.0e}_seed_{seed}_{role}.wav',rate,seed,role,path,'audio')
    write_csv(OUT/'listening_index.csv',index);return len(seen)


def freeze():
    checks=OUT/'artifact_checksums.json';write_json(checks,{})
    write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in
        [*sorted(ROOT.glob('issue24*.py')),ROOT/'issue24_findings.md',ROOT/'tests/test_issue24.py']})
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
        block='<!-- ISSUE24_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[
            ['作業ツリー通常ファイル数',before['files'],after['files']],['論理容量（ignored含む）',f"{before['bytes']/1e6:.2f} MB",f"{after['bytes']/1e6:.2f} MB"],
            ['Git新規追加候補','0（クリーン）',f"{record['new_git_candidates']}個 / {record['new_git_candidate_bytes']/1e6:.2f} MB"],
            ['既存追跡ファイルの変更 / 削除','0 / 0',f"{record['modified_tracked_files']} / {record['deleted_tracked_files']}"]])+'\n<!-- ISSUE24_STORAGE_END -->'
        p=OUT/'REPORT.md';body=p.read_text();p.write_text(re.sub(r'<!-- ISSUE24_STORAGE_BEGIN -->.*?<!-- ISSUE24_STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- ISSUE24_STORAGE_BEGIN -->' in body else body+'\n'+block+'\n')
        write_json(checks,{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    record['workspace']=workspace_inventory();write_json(OUT/'storage_after.json',record)


def report():
    m=ready();tick=time.perf_counter();tables()
    endpoints=csv_rows(OUT/'comparison_per_seed.csv');groups=csv_rows(OUT/'comparison_summary.csv');stability=csv_rows(OUT/'stability_and_acquisition.csv')
    motion=csv_rows(OUT/'movement_per_seed.csv');paired=csv_rows(OUT/'paired_differences.csv');critics=csv_rows(OUT/'critic_metrics.csv')
    fmt=lambda r,key:f"{float(r[key+'_mean']):.5f} ± {float(r[key+'_sd']):.5f}"
    main=table(['Actor LR','採用','E_total平均±SD','E_spec平均±SD','E_env平均±SD','J平均±SD','持続/3','失敗/3'],
        [[r['actor_lr'],r['role'],*[fmt(r,key) for key in ('E_total','E_spec','E_env','J')],r['voiced'],r['failures']] for r in groups])
    individual=table(['Actor LR','seed','採用','試行','E_total','E_spec','E_env','J','持続proxy'],
        [[r['actor_lr'],r['seed'],r['role'],r['trial'],*[f"{float(r[k]):.7f}" for k in ('E_total','E_spec','E_env','J')],r['sustained_voiced']] for r in endpoints])
    differences=table(['Actor LR','seed','ΔFinal E_total','ΔE_spec','ΔE_env','ΔJ','LR比','1更新中央値の比','1更新最大の比','累積経路長の比'],
        [[r['actor_lr'],r['seed'],*[f"{float(r[k]):+.6f}" for k in ('delta_E_total','delta_E_spec','delta_E_env','delta_J')],
            *[f"{float(r[k]):.4f}" for k in ('lr_ratio','median_movement_ratio','max_movement_ratio','path_length_ratio')]] for r in paired])
    acquisition=table(['Actor LR','seed','A:学習E≤.10数/4096','A:持続成功数','A:初回持続成功','B:初回E≤.10評価','B:初回持続成功','B:持続成功数/33','C:Final成功','到達後の保持率'],
        [[r['actor_lr'],r['seed'],r['training_threshold_count'],r['voiced_training_threshold_count'],r['first_training_success'] or '未到達',
            r['first_E_le_0p10'] or '未到達',r['first_voiced_E_le_0p10'] or '未到達',r['voiced_policy_threshold_count'],r['final_success'],
            f"{r['retention_successes']}/{r['retention_evaluations']} = {float(r['retention_fraction']):.3f}" if r['retention_fraction'] else '未定義（未到達）'] for r in stability])
    movement=table(['Actor LR','seed','||Δa||中央値','p90','最大','Σ||Δa||','初期→Final正味距離'],
        [[r['actor_lr'],r['seed'],*[f"{float(r[k]):.6f}" for k in ('median_delta_action_l2','p90_delta_action_l2','max_delta_action_l2','total_path_length','net_initial_final')]] for r in motion])
    physical_movement=table(['Actor LR','seed','|Δε_l|中央値 / 最大','|Δp_s|中央値 / 最大','|Δf0_l|中央値 / 最大'],
        [[r['actor_lr'],r['seed'],*[f"{float(r['median_abs_delta_physical_'+name]):.2f} / {float(r['max_abs_delta_physical_'+name]):.2f}" for name in ('epsilon_l','ps','f0_l')]] for r in motion])
    critic_table=table(['Actor LR','seed','集合','minQ RMSE','Q−r bias','Spearmanρ','誤順位率'],
        [[r['actor_lr'],r['seed'],r['subset'],*[f"{float(r[k]):.5f}" if r[k] else 'NA' for k in ('rmse','bias_Q_minus_r','spearman','pairwise_inversion_fraction')]] for r in critics if r['head']=='minQ' and r['subset'] in ('all99','lowJ10')])
    wavs=listening()
    from .issue24_figures import render
    images=render();findings=(ROOT/'issue24_findings.md').read_text()
    audit=json.loads((OUT/'audit.json').read_text());execution=json.loads((OUT/'execution.json').read_text())
    new_summaries=[json.loads((folder_for(r,s)/'summary.json').read_text()) for r in RATES if r!=BASE_LR for s in SEEDS]
    failures=sum(s['training']['failures']+s['evaluation']['failures'] for s in new_summaries)
    native=sum(s['training']['native_steps_completed']+s['evaluation']['native_steps_completed'] for s in new_summaries)+sum(a['native_steps_completed'] for a in audit['accounts'])
    sounds=execution['training_candidates']+execution['independent_policy_evaluations']+audit['generated_sounds'];protected=len(csv_rows(OUT/'protected_before.csv'))
    goals=[r['actor_lr'] for r in groups if r['role']=='final' and r['final_goal_met']=='True']
    goal_text='暫定目標（Final平均E_total≤.10・持続3/3）を満たした条件：'+', '.join(goals)+'。' if goals else '今回の3条件は暫定目標（Final平均E_total≤.10・持続3/3）を満たさなかった。'
    body=f'''# Issue24 — SAC128・λ=.75・Actorだけの学習率介入

## 結果と判断

{goal_text}

{findings}

## Actor以外を変更しない実装と監査

新規研究runはActor LR=1e-4/3e-5×seed0/1/2の6run×4096候補。3e-4は[Issue23の完全再現済み3seed](../phase1_caseA_sac128_stability_diagnostics/REPORT.md)を読み取り専用で参照し、新規学習しなかった。4番目のLRや一括最適化は追加していない。

Actor optimizerは平均出力だけでなく、共通MLPとlog_std headも含む。今回のLR変更はそれら全Actorパラメータに適用され、探索stdの学習速度も変わり得る。平均だけを変更する別の介入は加えていない。

`issue24_agent.ActorRateSAC`はSB3の`SAC.train`を継承したまま、`_update_learning_rate`で標準処理を呼んだ後にActor optimizerのLRだけを指定値にする。共通learning_rateとlr_scheduleは3e-4で固定し、Criticと温度optimizerも3e-4。初期param_groupsの変更だけで終わらず、学習中のschedulerによる上書き後に毎回適用する。標準loggerのtrain/learning_rate=3e-4は共通schedule値であり、Actor実効LRとは区別する。

[実際のSB3実装](sb3_learning_rate_source.txt) / [事前検証](preflight.json) / [manifest](manifest.json)。実効設定はActor optimizerのlrを3e-4に戻してから既存基準と機械比較し、その他全項目が一致。温度optimizerを含むparam_groupsの全オプションも初期化時に一致した。初期policy・Torch/NumPy/Python RNG・初期温度が同一、warmup16の学習行動・報酬も同一だった。Actor介入後の方策、経験分布、Critic重みやalphaの軌跡は変わり得るため、その一致は要求しない。

各新規runの4080更新で3optimizer全ての直前/直後LRを検査し、Actor更新ログに3LRの実測値を保存した。各33評価点にも実効param_groups[].lrを保存。新規6runで各optimizer24480回の検査が成功し、更新回数もActor/Critic/温度それぞれ24480。基準Issue23には3optimizerの実効LRの時系列がないため、[LR CSV](optimizer_learning_rates.csv.gz)の基準行は凍結された設定・標準schedulerによるprovenanceと明示し、新たな実測ログとして捏造しない。

![実効LRと実際のoptimizer loss](figures/learning_rates_and_losses.png)

テストでは、基準3e-4で新実装と標準SACの32step後の重み・optimizer・RNG・Replayが完全一致した。初回更新ではLR変更条件でもCriticと温度重みがbit単位で一致し、Actorだけ差が生じた。schedulerのprogress1/.8/.1/0と実際のoptimizer.stepでも指定LRを検査。独立評価・統計計算・checkpoint保存の前後で学習器/optimizer/RNG/Replay署名が全198点一致した。

## 固定条件と評価の定義

Case A（右の音響寄与α_r=β_r=0）、一定3D・1episode1行動、0.20s/50kHz/10000sample、native dt=1e-7、x0=.02cm/D=.01、初期状態/遅延履歴0。同じ人工Target波形/hash・模型・音響計算を保持した。可変はε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]、右ε_r=3.5e7/f0_r=49000固定。f0_lはHzでなくラビア開閉外力係数。モデル内部量を実測張力やPaと同一視しない。

SAC Actor/Critic[128,128]、Critic/温度LR3e-4、ent_coef=auto_0.01、target_entropy=-3、batch32、buffer50000、warmup16、gamma1、train_freq1、gradient_steps1、tau/target更新・正規化・更新回数・CPU1threadを固定。範囲縮小・変化量上限・ノイズ/entropy目標・報酬・batch・幅の変更はない。報酬はr=−J、J=.75E_spec+.25E_env、2倍補正なし。主評価は共通E_total=E_spec+E_env。

独立評価は0,128,...,4096の33点で、4096は最終更新後。評価経験はReplayに追加しない。Final=最終更新後の決定論的方策、Selected=33点の最初のminJ、posthoc=同33点の最初のmin共通E（学習目標とは別の事後選別）。最良方策を戻す/教師にする学習は行わない。持続発声proxyは末尾1000sampleのAC標準偏差/Target peak>.05かつ数値失敗なしで、生理的な有声判定ではない。失敗・非持続seedを除外しない。

## Finalを最優先した比較

{main}

平均±SDは3seedの標本SD（ddof=1）。暫定目標はFinal平均≤.10かつ持続3/3で、各seed全てのE≤.10とは別。SelectedやposthocをFinalへ差し替えない。[集約CSV](comparison_summary.csv) / [全seed・全endpointの成分と物理量](comparison_per_seed.csv)。

![全seedの4誤差・同じ縦軸](figures/error_histories.png)

![Final/Selected/posthocと安定性](figures/endpoints_and_stability.png)

{individual}

基準3e-4に対する対応seed差。負のΔFinal E_totalは改善を表す。LR比と、実測された行動移動量の比を別列にした。条件変更後は異なる経験分布・勾配・Adam状態を持つため、移動量がLR比に比例するとは仮定しない。

{differences}

## 到達・維持・学習経験を区別する

Aは学習サンプル（Replayに入る経験）、Bは独立評価された決定論的Actor、CはFinalでの成功。Eの閾値だけの数と、持続proxyも満たす数を両方保存した。初回時刻は評価した33点の中での最初であり、128点間の真の到達時刻ではない。未到達は右打切りとして表示し、平均から除外して高速到達と解釈しない。

{acquisition}

保持率=独立評価で初めてE≤.10かつ持続を満たした点からFinalまでの成功点数/同期間の全評価点数（初回を含む）。一度も到達しない場合は未定義で、成功維持率0/1と同列に扱わない。Final成功と、途中何度も成功したことも別である。Selected−Finalは符号付きで、負ならFinalの共通Eが悪い。minJとminEが違う場合、Selected共通EがFinalより高くても保存ミスではない。[到達・保持・経験CSV](stability_and_acquisition.csv)。

![到達時刻と保持率・未到達seedも表示](figures/acquisition_and_retention.png)

## Actorの1更新・累積・正味移動

{movement}

固定観測[-1,0,...,0]に対するActor optimizer直前/直後のtanh(mean)を記録。フック時点では当該stepのCritic/温度は更新済みで、Actor前後のQ評価には同じCriticを使う。log_std・alphaも含めて保存し、ログが乱数や重みに影響しないことを監査した。学習ごとの音響Eは確率的行動の経験であり、Actor更新前後の決定論的行動のEを毎回計算したわけではない。

128試行ごとの経路長Σ||Δa_update||₂と正味距離||a_end−a_start||₂を比較した。更新差の和と独立評価間の差はfloat32の1.2e-7以内で一致。経路長−正味距離や経路長/正味距離は折返しの記述で、失敗率や因果効果ではない。正規化行動は[-1,1]、全域幅に対する差はΔa/2。物理差はΔa×[12500000,750000,375500]で、ε_l/p_s/f0_lを異なる単位のままL2合成しない。

単位を分けた1更新の物理差（各模型単位）：

{physical_movement}

[1更新の分布・物理差の要約](movement_per_seed.csv) / [全36720 Actor更新のCSV](actor_updates.csv.gz) / [128試行ごとの経路・正味・物理差](update_intervals.csv)。

![1更新の実測分布と区間移動](figures/actor_movement.png)

## 温度・座標別探索std・loss

全33点のalpha、tanh後std、pre-tanh log_std、entropyを保存した。固定の局所NumPy RNG seed220026・4096標準正規サンプルでstd/entropyを推定し、グローバルRNGは消費しない。Gaussianとtanh後の微分entropyは別で、負になり得る。alphaは探索stdそのものではなく、増減の方向が一致するとは仮定しない。

Actor/Critic/温度lossは実際のbatch32の更新で計算された値。図は128試行区間内の平均のseed平均±標本SDで、保存された各update値から再計算できる。温度のoptimizer LRを固定しても、alpha自体は変化する。[33評価点](policy_evaluations.csv) / [温度・std・log_std](policy_dynamics.csv) / [実効LR](optimizer_learning_rates.csv.gz)。

![温度と探索分布・3seedを保持](figures/temperature_and_exploration.png)

## Final Criticの固定候補診断

同じIssue23の99保存候補と実測r=−Jを全条件に使用した。候補・基準native波形を再生成せず、各runのFinal Q1/Q2/minQを比較。終端1stepのためCritic targetはrだが、SAC Actorの目的には確率的行動に対する期待minQとentropyが残る。決定論的1点のQと同一の目的ではない。

{critic_table}

all99、同seedの基準33候補、lowJ上位10%でRMSE・バイアス・Spearman・誤順位率を別に算出した。固定候補は旧方策の回顧集合で、一様な探索空間標本や独立な汎化テストではない。低LRでFinalの値や順位が変わっても、Criticの誤勾配を修正したという介入結果ではない。[全Q指標](critic_metrics.csv) / [候補順位](critic_ranks.csv) / [全予測](critic_predictions.json.gz)。

![Final Q1/Q2・全条件同じ軸](figures/final_critic_calibration.png)

Issue23の[局所36音](../phase1_caseA_sac128_stability_diagnostics/local_sensitivity.csv)と[局所勾配](../phase1_caseA_sac128_stability_diagnostics/local_gradients.csv)はhashを固定して参照した。基準seed2の2944/3840/4096近傍、正規化刻み±.002/.01の結果で、今回は新たな局所鳴管計算0。新LRの訪問領域が同じ感度を持つとは言えず、そこでの勾配精度を新たに測ったとは主張しない。

## 音声・方策の保持と独立再生

9cohort×3基準の27モデルをロードし、同じ行動なら再生を共用して{audit['generated_sounds']}個の音を生成した。決定論的行動・20ブロックの制御・元振幅float64波形・E_total/J・温度/stdとFinal固定候補Qが保存値と一致。[監査](audit.json) / [再生CSV](endpoint_replays.csv) / [modelとaudioのhash索引](checkpoint_index.json)。推論用重みはActor+online twin Criticのみで、optimizer/Replay/target Criticを保存しないため再開学習用ではない。

保存したminJ/minE/Finalの方策を保持しても、学習中の更新やFinal精度が自動的に改善したことにはならない。今回の介入の主評価は、あくまで実際のFinalである。

聴取用[WAV ZIP](listening.zip)にはTargetと全27endpointを含め、同一波形を重複除去して{wavs}音を保存した。[索引CSV](listening_index.csv)で全endpointへ対応付く。全音に同じTarget peak逆数gainを掛けたfloat32、個別正規化・クリップ・位相合わせは0。元波形float64は各runのNPZに残す。代表図のseed1は結果を見ず固定し、全27音で共通のPSD基準・色・振幅/RMS軸を使った。PSD表示の10MHz補間は情報を増やさず、完全窓の中心域だけ表示する。

![代表seed1・Final](figures/audio_final.png)

![代表seed1・Selected/minJ](figures/audio_selected.png)

## 計算量・保護・再現

新規6runの壁時間{execution['wall_s']:.1f}s。学習24576音、独立評価198音、独立再生{audit['generated_sounds']}音、計{sounds}音。完了native stepは{native:,}。研究runの数値失敗{failures}件、非有限学習値0。元の3e-4の学習3runは再実行せず、局所感度の追加音も0。Critic集計・CSV/図生成で鳴管計算や更新はしない。[実行数](execution.json) / [run別会計](run_accounting.csv)。

5テスト成功。nativeテストは32候補×2条件=64学習音、6独立評価、6endpoint再生の76音で、研究runと別。scheduler/標準実装比較には鳴管を使わない合成終端envを179episode使い、基準の研究予算を消費していない。[テスト記録](test_execution.json)。

既存{protected}ファイルのhashが一致し、変更・削除0。root READMEと.gitignoreに索引・作業ログ除外だけを追加した。[元成果保護](integrity.json) / [実装hash](implementation_checksums.json) / [成果物hash](artifact_checksums.json)。主図9枚、保存場所は本専用ディレクトリ。commit/push/Issue更新は未実施。[再現・オフライン利用手順](REPRODUCE.md)。
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'README.md').write_text('''# Issue24 — SAC128・λ=.75のActor学習率比較

[REPORT](REPORT.md) / [再現とオフライン確認](REPRODUCE.md) / [全endpoint](comparison_per_seed.csv) / [到達・維持](stability_and_acquisition.csv) / [WAV ZIP](listening.zip)

既存Actor3e-4を読み取り専用で参照し、Actor1e-4/3e-5を各3seed×4096候補比較する。Critic/温度のLRは3e-4。Finalを主評価とし、minJ/minE保存・学習経験・独立方策の到達・維持を区別する。
''')
    (OUT/'REPRODUCE.md').write_text('''# 再現と帰省中のオフライン確認

リポジトリrootで、環境を指定する。既存正本のprepareは拒否し、完了runはhash一致時だけ再利用する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue24-mpl
# 着手前のworkspace/git/既存全path・bytes・sha256 JSONを用意
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_prepare --before /tmp/birdsong-issue24-before.json
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_training --workers 6
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_analysis critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue24_analysis integrity
```

新規再現は別checkoutでissue24_data.OUTを専用の新規先にし、着手前JSONを作る。過去の成果物は上書きしない。Actorだけ独立LRにするためissue24_agent.make_agentを使う。共通learning_rateを変更する方法は禁止。SAC.trainは標準実装、_update_learning_rateだけでActor LRを適用する。optimizer_lrsを各stepで検査し、SB3の共通train/learning_rateログをActor LRと誤読しない。

学習せずに確認する場合、REPORTと9図、CSVとWAV ZIPを開くだけでよい。全33評価はpolicy_evaluations.csvと各runのcurve.json.gz。全1更新の差分/物理量/lossはactor_updates.csv.gz、温度/std/log_stdはpolicy_dynamics.csv、3実効LRはoptimizer_learning_rates.csv.gz。保持率・未到達・Replayの成功はstability_and_acquisition.csv。WAV索引のraw_audio_pathは既存基準も含むため、リポジトリのresultsを一緒に持ち出すかlistening.zipを使う。新たな学習や鳴管計算は不要。

issue24_reportは保存データからCSV/図を再生成するだけで、鳴管計算/学習0。criticsは保存済みQの集計で0音。auditは27checkpointを再ロードし、異なる行動だけを独立再生するため実施回数を別に記録する。integrityは音を生成せず旧成果/hash/link/WAV共通gainを確認する。

checkpoint_index.jsonが全27のpolicy/audio/hashを示す。新規runはfinal.pt/selected.pt/posthoc.ptと対応NPZ、基準はIssue23を参照。issue23_data.load_checkpoint(path,seed,manifest['settings'])で同じMLPを作ってActor+online Critic重みを読み、agent.predict(obs,deterministic=True)で使用できる。loaderのoptimizerは推論用の仮初期化で、学習LR・optimizer状態の復元ではない。optimizer/Replay/target Criticを含まず再開学習はできない。metadataのformatは共用Issue23 reader形式を指し、今回の条件はconfig.json/manifestを正本とする。

WAVはTargetを含む全音共通gain=1/Target peakのfloat32で、クリップや個別正規化はない。音響評価と元波形はfloat64 NPZに保持する。f0_lはHzの基本周波数ではなく開閉外力係数。Selected=minJ、posthoc=min共通E、Final=最終更新後を混同しない。
''')
    write_json(OUT/'report_execution.json',dict(generated_sounds=0,learning_updates=0,images=9,source_sha256=digest(__file__),findings_sha256=digest(ROOT/'issue24_findings.md'),wall_s=time.perf_counter()-tick))
    print('Nine figures, all cohort tables and offline listening archive generated; freezing hashes and inventory',flush=True);freeze()


if __name__=='__main__':report()
