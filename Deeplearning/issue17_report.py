"""Build the canonical 4096-step report and single-condition figure index."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import time
import numpy as np
from .constant020_env import INDICES
from .issue15_analysis import write_csv, write_json
from .issue15_report import table
from .issue17 import DEST,run_specs,weight_label
from .issue17_figures import figures,condition_stem
from .issue17_artifacts import read_record

LABELS={'TD3':'TD3','SAC':'SAC','DE':'DE','random_search':'Random'}


def report(output):
    if (Path(output)/'direct_weight_extension/manifest.json').exists():
        raise RuntimeError('Current72-profile report is preserved; use issue17_redraw for current four-method figures, or issue17_extension render before publication')
    output=Path(output);started=time.perf_counter()
    from .issue17_cleanup import plan_cleanup
    plan_cleanup(output)
    log=output.parent/'training4096.log'
    if log.exists():shutil.copy2(log,output/'execution.log')
    manifest=json.loads((output/'manifest.json').read_text())
    config=manifest['settings']
    execution=json.loads((output/'execution.json').read_text())
    audit=json.loads((output/'audit.json').read_text())
    review=json.loads((output/'diagnostics/review.json').read_text())
    rows=[];points=[];checkpoints=[];runs={}
    for c,m,seed,w in run_specs(config):
        folder=output/'runs'/c/m/weight_label(w)/f'seed_{seed}'
        s=json.loads((folder/'summary.json').read_text())
        curve=read_record(folder/'curve.json')
        final,best=s['final'],s['best'];comp=final['acoustic_components']
        row=dict(condition=c,method=m,envelope_weight=w,seed=seed,
            E_total=final['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],
            E_weighted=final['weighted_loss'],parameter_nrmse=final['parameter_nrmse'],
            initial_error=s['initial']['acoustic_error'],best_error=best['acoustic_error'],
            best_episode=best['episodes'],best_parameter_nrmse=best['parameter_nrmse'],
            training_candidates=s['training']['episodes'],evaluation_sounds=s['evaluation']['episodes'],
            training_native_steps=s['training']['native_steps_attempted'],
            training_completed_native_steps=s['training']['native_steps_completed'],
            evaluation_native_steps=s['evaluation']['native_steps_attempted'],
            training_wall_s=s['training']['wall_s'],evaluation_wall_s=s['evaluation']['wall_s'],
            save_wall_s=s['artifact_wall_s'],process_cpu_s=s['process_cpu_s'],
            gradient_iterations=s['gradient_updates'],
            actor_optimizer_calls=s['optimizer_steps'].get('actor',0),
            critic_optimizer_calls=s['optimizer_steps'].get('critic',0),
            entropy_optimizer_calls=s['optimizer_steps'].get('entropy',0),
            failures=s['training']['failures'],silent=s['training']['silent_episodes'],
            nonvoiced=s['training']['nonvoiced_episodes'],final_sustained_voiced=final['sustained_voiced'],
            epsilon_l=final['physical_controls'][0],ps=final['physical_controls'][2],f0_l=final['physical_controls'][3],
            returned_solution=s['returned_solution'])
        rows.append(row);runs[(c,m,seed,w)]=s
        for p in curve:
            points.append(dict(condition=c,method=m,envelope_weight=w,seed=seed,episodes=p['episodes'],
                E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],
                E_env=p['acoustic_components']['envelope'],E_weighted=p['weighted_loss'],
                parameter_nrmse=p['parameter_nrmse'],best_observed_error=p['best_observed_error'],
                gradient_updates=p['gradient_updates'],evaluation_phase=p['evaluation_phase'],
                role='independent_current_policy' if m in ('TD3','SAC') else 'actual_training_candidate'))
        for n in config['checkpoints']:
            p=next(p for p in curve if p['episodes']==n)
            checkpoints.append(dict(condition=c,method=m,envelope_weight=w,seed=seed,episodes=n,
                E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],
                E_env=p['acoustic_components']['envelope'],E_weighted=p['weighted_loss'],
                parameter_nrmse=p['parameter_nrmse'],best_observed_error=p['best_observed_error'],
                gradient_updates=p['gradient_updates'],evaluation_phase=p['evaluation_phase']))
    aggregates=[]
    fields=('E_total','E_spec','E_env','E_weighted','parameter_nrmse','best_error','best_episode',
            'training_wall_s','evaluation_wall_s','save_wall_s','process_cpu_s')
    groups=sorted({(r['condition'],r['method'],r['envelope_weight']) for r in rows})
    for c,m,w in groups:
        group=[r for r in rows if (r['condition'],r['method'],r['envelope_weight'])==(c,m,w)]
        row=dict(condition=c,method=m,envelope_weight=w,n=3)
        for key in fields:
            vals=np.array([r[key] for r in group]);row[key+'_mean']=float(vals.mean());row[key+'_sample_sd']=float(vals.std(ddof=1))
        aggregates.append(row)
    effects=[]
    for c in config['conditions']:
        for m in ('TD3','SAC'):
            diffs=[runs[(c,m,s,.5)]['final']['acoustic_error']-runs[(c,m,s,1.)]['final']['acoustic_error'] for s in config['seeds']]
            effects.append(dict(condition=c,method=m,delta_definition='lambda0.5 minus lambda1 common final E_total',
                mean_delta=float(np.mean(diffs)),sample_sd_delta=float(np.std(diffs,ddof=1)),
                seed0_delta=diffs[0],seed1_delta=diffs[1],seed2_delta=diffs[2]))
    write_csv(output/'comparison_per_seed.csv',rows)
    write_csv(output/'comparison_aggregate.csv',aggregates)
    write_csv(output/'comparison_curves.csv',points)
    write_csv(output/'checkpoint_comparison.csv',checkpoints)
    write_csv(output/'lambda_effects.csv',effects)
    images=figures(output)
    fmt=lambda r,k:f"{r[k+'_mean']:.6g} ± {r[k+'_sample_sd']:.3g}"
    overview=table(['次元','手法','λ','最終 E_total','E_spec','E_env','E_weighted','parameter NRMSE'],
        [[r['condition'].replace('constant_',''),LABELS[r['method']],r['envelope_weight'],
          *[fmt(r,k) for k in ('E_total','E_spec','E_env','E_weighted','parameter_nrmse')]] for r in aggregates])
    best_table=table(['次元','手法','λ','最終 E_total','観測途中最良 E_total','最良位置'],
        [[r['condition'].replace('constant_',''),LABELS[r['method']],r['envelope_weight'],
          fmt(r,'E_total'),fmt(r,'best_error'),fmt(r,'best_episode')] for r in aggregates])
    individual=table(['次元','手法','λ','seed','最終','途中最良','最良位置','訓練秒','評価秒'],
        [[r['condition'].replace('constant_',''),LABELS[r['method']],r['envelope_weight'],r['seed'],
          f"{r['E_total']:.7g}",f"{r['best_error']:.7g}",r['best_episode'],
          f"{r['training_wall_s']:.1f}",f"{r['evaluation_wall_s']:.1f}"] for r in rows])
    checkpoint_rows=[]
    for c,m,w in groups:
        vals=[np.mean([r['E_total'] for r in checkpoints if
              (r['condition'],r['method'],r['envelope_weight'],r['episodes'])==(c,m,w,n)]) for n in config['checkpoints']]
        checkpoint_rows.append([c.replace('constant_',''),LABELS[m],w,*[f"{v:.5g}" for v in vals]])
    speed_table=table(['次元','手法','λ','256','512','1024','2048','4096'],checkpoint_rows)
    interpretations=[]
    for r in effects:
        interpretations.append(f"- {r['condition']} / {r['method']}: λ=.5−λ=1 の最終共通誤差差は "
            f"{r['mean_delta']:+.6g} ± {r['sample_sd_delta']:.3g}（対応seed差の標本SD）。"
            +('平均で減少した。' if r['mean_delta']<0 else '平均で減少しなかった。'))
    degraded=sum(s['final']['acoustic_error']>s['best']['acoustic_error']+1e-12
                 for (c,m,seed,w),s in runs.items() if m in ('TD3','SAC'))
    optimizer_notes=[]
    for c in config['conditions']:
        de=[runs[(c,'DE',s,1.)]['final']['acoustic_error'] for s in config['seeds']]
        rl=[r['E_total_mean'] for r in aggregates if r['condition']==c and r['method'] in ('TD3','SAC')]
        optimizer_notes.append(f"- {c}: DEの最終3 seedは {', '.join(f'{v:.6g}' for v in de)}。"
            f"RL各群の最終平均は {min(rl):.6g}〜{max(rl):.6g}。"+
            ('DEは全seedで数値精度まで到達した。' if all(v==0 for v in de) else
             'DEでも全seedの高精度到達は安定せず、4096回で十分とは言えない。'))
    train_native=sum(r['training_native_steps'] for r in rows)
    eval_native=sum(r['evaluation_native_steps'] for r in rows)
    report_text=f"""# Issue #17 — 4096回・0.20秒・時間一定制御の正本

**最新実験は各4096候補の54 run。旧256回の36 runを対照群として流用せず、全runをゼロから新規実行した。**
TD3/SAC×λ=1/.5×1D/2D/3D×3 seed=36 run、DE/Random×1D/2D/3D×3 seed=18 run。
最初に [README](README.md) の索引から、1条件・1曲線の図、スペクトログラム、物理制御、CSVを確認できる。

## 新4096回の条件

Case A・モデル内生成目標、0.20秒・50 kHz・10000 samples。
1Dはε_l、2Dはε_l/p_s、3Dはε_l/p_s/f0_lを**1行動で選び、全区間保持**する。
未学習値は目標値へ固定。目標制御は `{manifest['target_controls']}`。
旧256版のtarget.npzとSHA256まで一致し、学習途中に目標・範囲・初期値を変更していない。

範囲はε左右[2.5e7,5e7]、p_s[2.5e6,4e6]、f0左右[49000,1e7]。
f0_lは `-epsilon*x - b*y - c*x*x*y - f0` の外力/開閉操作項であり**基音周波数Hzではない**。
Case Aの右パラメータは出力に寄与しない。対照のない4D/5Dや左右2音源の主張は行わない。
各試行で状態・遅延履歴・因果フィルタをリセットし、20個の10 msブロック内では連続維持する。

共通主評価は全音声 `E_total=E_spec+E_env`。E_specは対数PSD、E_envは5 ms RMS包絡。
RL報酬だけを `-(E_spec+λ E_env)`、λ=1/.5とする。γ=1の1行動1報酬。
**DE/Randomはλ=1のみ**。λ効果は同手法の両群で、直接法との比較は共通E_totalで判断する。
重み付き損失を主評価と取り違えない。失敗は候補数に残して報酬−1e6、短いprefixを有利に評価しない。
有限の大誤差はclipしない。

LR=.0003、net[32,32]、batch32、warmup16、buffer50000、γ=1、train_freq=1、gradient_steps=1。
TD3 noise=.1、SAC entropy=auto_0.01。両λ群の実効設定・初期policy hash・warmup行動/音響値の一致を監査した。
全run CPU・workerごとPyTorch1 thread。並列worker={execution['workers']}。
DEはpopulation8、mutation(.5,1)、recombination=.7、polishなし。
SciPyの世代反復を使い、完全収束後も4096候補まで同じ更新則を継続した。
定数目的関数でも候補数を満たすこと、および通常SciPy solveと収束前の候補列が一致することをテストした。

## Oracle・発声領域・物理的な限界

float64 Oracle最大誤差 {max(r['error'] for r in review['oracles'] if r['action_dtype']=='float64'):.6g}、
float32行動丸めを含む最大 {max(r['error'] for r in review['oracles'] if r['action_dtype']=='float32'):.6g}。
[事前診断](diagnostics/review.json) に全次元の波形誤差・発散・発声と感度を保存した。
Oracleでほぼ0にできるため、探索結果が悪いことをモデルの表現限界と同一視しない。

f0_l=300000は下限付近の発声域（全範囲の{review['f0_fraction_of_range']:.3%}位置）にある。
600000付近等で持続振動が消え、広い範囲を占める非持続領域が探索を難しくし得る。
局所の波形差分Jacobian特異値は `{review['normalized_late_waveform_jacobian_singular_values']}`。
これは局所感度の証拠であり、大域的に音響値から3物理量を一意識別できる証明ではない。
3Dの悪化を単純に「学習回数不足」と断定せず、範囲縮小・再パラメータ化は今回行わない。

全音声のpeakが1e-10未満を無音、終盤20 msのAC RMS/共通target peakが.05以下を非持続発声の補助指標として記録。
これは生理学的な有声判定器ではない。訓練数値失敗 {sum(r['failures'] for r in rows)}件、
無音 {sum(r['silent'] for r in rows)}件、非持続 {sum(r['nonvoiced'] for r in rows)}件。
loss・勾配・更新後parameterのNaN/Infは全更新で検査し0件。失敗/無音/高誤差seedを除外していない。

## 最終音響誤差・内訳・物理量誤差（3 seed平均 ± 標本SD）

{overview}

parameter NRMSEは選択した物理量の誤差を各LOW/HIGH幅で正規化したRMS。
非学習量は目標値に一致。個別の実物理値ε_l/p_s/f0_lは [per-seed CSV](comparison_per_seed.csv) にある。
WAVは両波形を共通target peakで割ったfloat32、raw音声/実制御はnpzのfloat64。
個別のgain合わせ、位相合わせ、時間伸縮はしない。

## λ比較の判断

{chr(10).join(interpretations)}

これは対応する3 seedの観測であり、有意差や別目標への汎化を示さない。
Phase0の10 ms局所報酬での重み効果が、鳴管モデルの全0.20秒報酬にも成立するかを独立に検証した結果である。
λ=.5を最適値と決め付けず、目標・予算・学習率を結果を見て変更していない。
1行動の課題は静的ブラックボックス探索に近く、RLが直接法より有利という前提を置かない。

{chr(10).join(optimizer_notes)}

3DのRL最終誤差は約0.44に残り、Oracleのほぼ0から遠い。局所感度は存在するが、発声領域の狭さ、
非持続領域、限られた初期探索とQ値学習を含む最適化上の問題があり得る。
この結果だけでモデルの表現限界や単純な回数学習不足を断定しない。

## 途中最良と最終

{best_table}

RL36 run中 {degraded} runで最終が観測途中最良より悪化した。
最終値は更新をすべて終えた方策、直接法は**実際の候補4096**。
直接法の採用可能なincumbent/best-so-farは別に保存・独立再生し、最終候補へ置き換えない。
初期はRLの未学習方策（0回）、直接法は予算内の候補1。bestは同一目標の反復評価による補助値である。
初期/best/finalと指定チェックポイントにモデル（RL）・actions・physical_controls・npz/WAVを保存した。
4096のcheckpointは最終artifactを参照し、配列/モデルの複製を避けた。

## 到達速度（指定候補数での共通E_total平均）

{speed_table}

[checkpoint CSV](checkpoint_comparison.csv) に個別値・内訳・重み付き損失・parameter誤差と更新数を保存。
独立方策評価は事前固定の0,128,…,4096、計33回であり、256/512/1024/2048/4096を含む。
中間は直前のgradient更新完了後、最後は最終更新後。独立評価の前後でpolicy・optimizer・replay・entropy・RNGのhash不変を確認した。
曲線はRLの現在方策評価、直接法は各実候補の値で、best-so-farとは別である。

## 個別seed

{individual}

## 実時間・計算量

4096環境step=4096候補音、1音20ブロック×100000 native step=200万step、
1 runは8,192,000,000 native step。環境step、native step、gradient iteration、optimizer呼出しを区別する。
RLは各4080 gradient iterations。TD3 actor2040/critic4080、SAC actor/critic/entropy各4080。

訓練 {execution['training_candidates']:,}候補・native {train_native:,} step、
独立評価 {execution['evaluation_sounds']:,}音・native {eval_native:,} step。
DE/Randomの追加評価は最終とbest各1音、RLは各33音で予算外計上。
並列実験wall **{execution['concurrent_run_wall_s']:.2f}秒**、
訓練process wall合計 {execution['training_process_wall_s']:.2f}秒、
評価合計 {execution['evaluation_process_wall_s']:.2f}秒、
保存合計 {execution['save_process_wall_s']:.2f}秒、
process CPU合計 {execution['process_cpu_s']:.2f}秒。並列wallと合算時間は異なる。

事前物理診断は{review['setup_generated_sounds']}音・{review['setup_native_steps']:,} native step。
短い6 runのsmoke（192候補）と11テストは [smoke.json](smoke.json) / [tests.json](tests.json) に分離。
別プロセス監査は {audit['audit_generated_sounds']}音・{audit['audit_native_steps']:,} native step、
{audit['audit_wall_s']:.2f}秒。全{len(audit['reload_comparisons'])}再生の評価差と波形差を [audit.json](audit.json) に保存した。
描画/レポートは [report_timing.json](report_timing.json) に別計測。attemptedは失敗ブロックの要求全量、
completedは完了ブロックのみ。runごとの双方の値・失敗数を残した。

全候補・評価点のJSONは可逆gzipで保存した（curve.json.gz、RL training_trace.json.gz）。
全値・全点を保持し、元バイト列への復元を検証してから非圧縮版を削除した。
[compression.json](compression.json) に元/圧縮hash、サイズと圧縮時間を記録した。図の点を間引く処理ではない。

## 図・整理・研究上の範囲

1画像=1次元・1手法・1λ。seed0生曲線、3seed平均±標本SD、bestの図は独立ファイル。
中心線1本、SD帯は同条件だけ。軸46pt、目盛30pt、線幅2.8pt、x=Number of trials、y=Acoustic error。
タイトル・上部条件文・凡例・グリッド・閾値線なし。頂点省略/平滑化/間引きなし。
全学習曲線のx/y範囲を揃え、両λは必ず同範囲。
SD帯は算術的mean±sample SDで、負側の帯があっても誤差が負という意味ではない。
スペクトログラムは上段target/下段generated、同runだけの2段、無題、共通0–10kHz表示。
物理制御図は**実際に選択した1物理量の一定値1本**。目標値は上記条件とmanifestで照合する。

旧256回は初期パイロットで、1Dの直接法が強いこと、最良/最終が異なること、3Dの発声域が狭いことが分かった。
λ=1の新規36 runは、256回時点の音声・行動・制御・評価が旧パイロットの最終値と一致した
（[prefix照合](pilot_alignment.json)、追加生成なし）。初期化は全てゼロからで、旧モデルを学習へ読み込んでいない。
今回の54 runと混ぜて集計していない。小さい数表/設定だけ [履歴](history/README.md) に残し、旧モデル・音声・重複図は置換した。
旧0.35秒もモデル・目標・制御課題が異なる歴史情報として区別する。
削除/保持の一覧と根拠、旧監査が歴史的な記録であることは [CLEANUP.md](CLEANUP.md) と [cleanup_plan.json](cleanup_plan.json) を参照。
Phase0・実音声は保護対象で、変更しない。監査 {audit['checks']:,} checks passed。
新正本を対象とする監査へ切り替え、旧監査ログや旧実行条件を新条件のものとして書き換えない。

全次元で同一目標がOracleで表現可能なので、次元増加による表現空間と探索難度を分けて判断する。
4096回で不十分な条件があっても追加予算や目標の無断変更を行わず、時変制御・実録音・左右2音源へ進まない。

## 実行・再検証

```bash
cd /home/tatsukawa/work/Birdsong_tatsukawa
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue17-v2-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 audit
# 同じ凍結条件を新IDへ再計算（54 run、正本内reproductions/へ保存）:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_replay --name replica4096 --all --workers 27
# 単独runの再計算例:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_replay --name td3_2d_lambda0p5 --condition constant_2d --method TD3 --weight 0.5 --seed 0
# 未実行の作業コピー（旧256回の正本がある状態）で新規作成:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 train --workers 27
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 finish
# 保存済みの新CSVから図だけを描画:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_redraw
```

既存4096版へのprepareは上書きを拒否する。再監査は訓練せず、結果を/tmpへ保存する。
今回のcommit/pushは行わない。GitHubの結果パスはユーザーが変更をcommit/pushするまで旧版を示し得る。
"""
    (output/'REPORT.md').write_text(report_text)
    index=["# Phase 1 — 4096回の正本","",
        "最初に [REPORT.md](REPORT.md) を読み、次に以下の曲線→スペクトログラム→物理制御→CSVの順で確認する。",
        "各画像は1条件だけ。seed0と平均±標本SDは別ファイル。DE/Randomはλ=1のみ。","",
        "## 各次元・手法・λの単一曲線","",
        table(['次元','手法','λ','seed0','平均±SD','best seed0','best 平均±SD'],[
            [c.replace('constant_',''),LABELS[m],w,
             f"[図](figures/learning_curves/{condition_stem(c,m,w)}_seed0.png)",
             f"[図](figures/learning_curves/{condition_stem(c,m,w)}_mean_sd.png)",
             f"[図](figures/best_observed/{condition_stem(c,m,w)}_seed0.png)",
             f"[図](figures/best_observed/{condition_stem(c,m,w)}_mean_sd.png)"] for c,m,w in groups]),
        "","## 目標/生成スペクトログラムと物理制御","",
        "スペクトログラムは上段Target・下段Generated。制御図は実際の選択値1本で、目標値はREPORTの条件表を参照する。","",
        table(['次元','手法','λ','スペクトログラム','選択物理量'],[
            [c.replace('constant_',''),LABELS[m],w,
             f"[図](figures/spectrograms/{condition_stem(c,m,w)}_seed0.png)",
             " / ".join(f"[{name}](figures/physical_controls/{condition_stem(c,m,w)}_{name}_seed0.png)"
                for i,name in ((0,'epsilon_l'),(2,'ps'),(3,'f0_l')) if i in INDICES[c])] for c,m,w in groups]),
        "","## 数表・検証・データ","",
        "- [最終値の平均・標本SD](comparison_aggregate.csv) / [個別seed](comparison_per_seed.csv)",
        "- [全曲線の生データ](comparison_curves.csv) / [指定checkpoint](checkpoint_comparison.csv) / [λの対応差](lambda_effects.csv)",
        "- [target診断](diagnostics/review.json) / [設定](config.json) / [manifest](manifest.json)",
        "- [独立監査](audit.json) / [図の仕様監査](figure_audit.json) / [図と数値の対応](figure_sources.json)",
        "- [実行時間](execution.json) / [描画時間](report_timing.json) / [整理一覧](CLEANUP.md) / [歴史資料](history/README.md)",
        "","raw音声・モデルは runs/<次元>/<手法>/<lambda1またはlambda0p5>/seed_<n>/。",
        "initial/、best/、最終（run直下）、checkpoints/<回数>/を区別する。",
        "旧256回の詳細ログ・重複図は新しい正本に混在させていない。"]
    (output/'README.md').write_text("\n".join(index)+"\n")
    from .issue16 import digest
    from datetime import datetime,timezone
    write_json(output/'completion_verification.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,python_executable=sys.executable,platform=platform.platform(),
        cpp_compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        worker_torch_threads=1,device='cpu',gpu_used=False,working_tree_initially_clean=True,
        orchestration_source_sha256={name:digest(Path(__file__).parent/name) for name in
            ('issue17_report.py','issue17_figures.py','issue17_audit.py','issue17_cleanup.py','issue17_artifacts.py')},
        commit_created=False,push_performed=False))
    write_json(output/'report_timing.json',dict(plot_and_report_wall_s=time.perf_counter()-started,
               figures=len(images),smoothing=False,sample_sd_ddof=1))
