"""Comparable figures and measured-cost report from frozen Issue 17 artifacts."""
import json
import time
import numpy as np

from .constant020_env import INDICES,RATE
from .issue15_analysis import write_csv,write_json
from .issue15_report import table
from .model import LOW,HIGH

METHODS=('TD3','SAC','DE','random_search')
COLORS={'TD3':'#0072B2','SAC':'#D55E00','DE':'#009E73','random_search':'#CC79A7'}
LABELS={'TD3':'TD3','SAC':'SAC','DE':'DE','random_search':'Random'}
NAMES={0:r'$\varepsilon_l$',2:r'$p_s$',3:r'$f_{e,l}$ (native f0_l)'}


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots
    from .plotting import comparison
    started=time.perf_counter();figures=output/'figures';figures.mkdir(exist_ok=True)
    manifest=json.loads((output/'manifest.json').read_text())
    review=json.loads((output/'diagnostics/review.json').read_text())
    execution=json.loads((output/'execution.json').read_text())
    audit=json.loads((output/'audit.json').read_text())
    rows=[];runs={};points=[]
    for condition in INDICES:
        for method in METHODS:
            for seed in (0,1,2):
                folder=output/'runs'/condition/method/f'seed_{seed}'
                s=json.loads((folder/'summary.json').read_text())
                curve=json.loads((folder/'curve.json').read_text())
                row=dict(condition=condition,method=method,seed=seed,
                    E_total=s['final']['acoustic_error'],E_spec=s['final']['acoustic_components']['log_spectral'],
                    E_env=s['final']['acoustic_components']['envelope'],initial_error=s['initial']['acoustic_error'],
                    best_error=s['best']['acoustic_error'],best_episode=s['best']['episodes'],
                    training_candidates=s['training']['episodes'],evaluation_sounds=s['evaluation']['episodes'],
                    training_native_steps=s['training']['native_steps_attempted'],
                    evaluation_native_steps=s['evaluation']['native_steps_attempted'],
                    training_wall_s=s['training']['wall_s'],evaluation_wall_s=s['evaluation']['wall_s'],
                    save_wall_s=s['artifact_wall_s'],process_cpu_s=s['process_cpu_s'],gradient_iterations=s['gradient_updates'],
                    actor_optimizer_calls=s['optimizer_steps'].get('actor',0),critic_optimizer_calls=s['optimizer_steps'].get('critic',0),
                    failures=s['training']['failures'],silent=s['training']['silent_episodes'])
                rows.append(row);runs[(condition,method,seed)]=(s,curve,folder)
                points.extend(dict(condition=condition,method=method,seed=seed,episodes=p['episodes'],
                    E_total=p['acoustic_error'],best_observed_error=p['best_observed_error'],
                    role='independent_current_policy' if method in ('TD3','SAC') else 'actual_training_candidate') for p in curve)
    aggregates=[]
    fields=('E_total','E_spec','E_env','best_error','best_episode','training_wall_s','evaluation_wall_s','save_wall_s','process_cpu_s')
    for condition in INDICES:
        for method in METHODS:
            group=[r for r in rows if r['condition']==condition and r['method']==method]
            row=dict(condition=condition,method=method,n=3)
            for key in fields:
                y=np.array([r[key] for r in group]);row[key+'_mean']=float(y.mean());row[key+'_sample_sd']=float(y.std(ddof=1))
            aggregates.append(row)
    write_csv(output/'comparison_per_seed.csv',rows)
    write_csv(output/'comparison_aggregate.csv',aggregates)
    write_csv(output/'comparison_curves.csv',points)
    upper=max(r['E_total'] for r in points)*1.08
    for best in (False,True):
        for mean in (False,True):
            with plt.style.context(['science','ieee','no-latex']),plt.rc_context({'path.simplify':False}):
                fig,axes=plt.subplots(3,1,figsize=(13,12),sharex=True,sharey=True)
                for ax,condition in zip(axes,INDICES):
                    for method in METHODS:
                        curves=[runs[(condition,method,s)][1] for s in (0,1,2)]
                        x=[p['episodes'] for p in curves[0]]
                        field='best_observed_error' if best else 'acoustic_error'
                        y=np.array([[p[field] for p in c] for c in curves]);center=y.mean(axis=0) if mean else y[0]
                        ax.plot(x,center,label=LABELS[method],color=COLORS[method],lw=1.2 if not best else 1.8)
                        if mean:
                            sd=y.std(axis=0,ddof=1);ax.fill_between(x,center-sd,center+sd,color=COLORS[method],alpha=.12)
                    ax.set_ylabel(condition.replace('constant_','')+' acoustic error',fontsize=16)
                    ax.tick_params(labelsize=13);ax.set_ylim(-.05*upper,upper)
                    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=4,frameon=False,fontsize=13)
                axes[-1].set_xlim(0,256);axes[-1].set_xlabel('Training candidates / one-action episodes',fontsize=17)
                fig.tight_layout(h_pad=3)
                fig.savefig(figures/(('best_observed_' if best else 'learning_curves_')+('mean_sd.png' if mean else 'seed0.png')),dpi=140,bbox_inches='tight')
                plt.close(fig)
    for condition,indices in INDICES.items():
        # All controls are constant; show the actual held physical values over
        # precisely 0.20 s, including the full native bounds on every axis.
        with plt.style.context(['science','ieee','no-latex']):
            fig,axes=plt.subplots(len(indices),1,figsize=(13,3.5*len(indices)+1),squeeze=False,sharex=True)
            for ax,index in zip(axes[:,0],indices):
                oracle=manifest['target_controls'][index]
                ax.plot([0,.2],[oracle,oracle],color='black',ls='--',lw=2,label='Target')
                for method in METHODS:
                    s,_,_=runs[(condition,method,0)];value=s['final']['physical_controls'][index]
                    ax.plot([0,.2],[value,value],color=COLORS[method],lw=1.8,label=LABELS[method])
                margin=(HIGH[index]-LOW[index])*.04
                ax.set_ylim(LOW[index]-margin,HIGH[index]+margin);ax.set_ylabel(NAMES[index],fontsize=15)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
                ax.tick_params(labelsize=12)
            axes[0,0].legend(loc='lower center',bbox_to_anchor=(.5,1.08),ncol=5,frameon=False,fontsize=12)
            axes[-1,0].set_xlim(0,.2);axes[-1,0].set_xlabel('Time [s]',fontsize=16)
            fig.tight_layout(h_pad=2);fig.savefig(figures/f'{condition}_seed0_physical_controls.png',dpi=140,bbox_inches='tight');plt.close(fig)
        for method in METHODS:
            _,_,folder=runs[(condition,method,0)]
            with np.load(folder/'rollout.npz') as data:
                comparison(data['target'],data['audio'],RATE,figures/f'{condition}_{method}_seed0_spectrogram.png')
    # Machine-readable file identifies each figure's exact final waveform and
    # controls; no evaluation or training is triggered while drawing.
    from .issue16 import digest
    write_json(output/'figure_sources.json',{str(folder.relative_to(output)):dict(
        final_rollout_sha256=digest(folder/'rollout.npz'),final_error=s['final']['acoustic_error'],
        final_action=s['final']['actions']) for (c,m,seed),(s,curve,folder) in runs.items() if seed==0})
    fmt=lambda r,k:f"{r[k+'_mean']:.6g} ± {r[k+'_sample_sd']:.3g}"
    overview=table(['次元','手法','最終 E_total','E_spec','E_env','途中最良（補助）'],
        [[r['condition'].replace('constant_',''),LABELS[r['method']],*[fmt(r,k) for k in ('E_total','E_spec','E_env','best_error')]] for r in aggregates])
    individual=table(['次元','手法','seed','最終','途中最良','最良候補/episode','学習秒','評価秒'],
        [[r['condition'].replace('constant_',''),LABELS[r['method']],r['seed'],f"{r['E_total']:.7g}",f"{r['best_error']:.7g}",
          r['best_episode'],f"{r['training_wall_s']:.2f}",f"{r['evaluation_wall_s']:.2f}"] for r in rows])
    statements=[]
    for condition in INDICES:
        group=[r for r in aggregates if r['condition']==condition]
        winner=min(group,key=lambda r:r['E_total_mean'])
        degraded=sum(s['final']['acoustic_error']>s['best']['acoustic_error']+1e-12
                     for (c,m,seed),(s,_,_) in runs.items() if c==condition and m in ('TD3','SAC'))
        statements.append(f"- {condition}: 最終平均の最小は {LABELS[winner['method']]} ({winner['E_total_mean']:.6g})。RL 6 run中 {degraded} runで最終方策が観測途中最良より悪化。")
    train_steps=sum(r['training_native_steps'] for r in rows);eval_steps=sum(r['evaluation_native_steps'] for r in rows)
    text=f'''# Issue #17 — Case A・0.20秒・時間一定1D→2D→3D

事前の発声・感度・Oracle gateを通過し、**1D/2Dを全24 run完了した後に3Dの12 runを実施**。TD3/SAC/DE/Random × seed 0/1/2、各256候補で36 runを完了。時変制御・実録音・Case B/C・PPO・追加予算・ハイパーパラメータ探索は実施しない。

## 共通目標と物理的な意味

全36 runで同じCase Aモデル内生成音を使う。制御順序は `(epsilon_l,epsilon_r,ps,f0_l,f0_r)`、目標値は `{manifest['target_controls']}`。既存範囲はε左右[2.5e7,5e7]、p_s[2.5e6,4e6]、f0左右[4.9e4,1e7]。ε_lは左の復元項、p_sは圧力、**f0_lは式 `-epsilon*x - b*y - c*x²*y - f0` の外力（開閉操作項）でありHzの基本周波数ではない**。Case Aでは右のalpha/betaが0で出力に寄与しない。右2変数を変えても波形がbitwise一致することを確認し、4D/5Dは作らない。

旧目標f0_l=49000は探索下限だったため、先に固定の物理surveyを実施。600000付近等では持続振動が消える。今回の共通値300000は範囲内部（全幅の{review['f0_fraction_of_range']:.3%}位置）であり、隣接±9951でも持続発声する。**発声領域は全探索範囲の一部に偏っている**。範囲の縮小や変換の調整は行わない。ε_l/p_s/f0_lの各±0.1%範囲摂動で音響誤差と後半波形が変化し、正規化した後半波形差分Jacobianの特異値は `{review['normalized_late_waveform_jacobian_singular_values']}`。3方向の局所非退化性は確認できたが、大域的な一意識別を証明するものではない。

0.20秒・50 kHz・10000 sample。DT=1e-7、20個の10 msブロック、1候補は200万native step。**行動は1回だけ**で選択値を0.20秒保持する。初期状態・遅延履歴・16次低域フィルタ状態を候補ごとにリセットし、20ブロック間では状態とフィルタを連続維持。旧0.35秒環境は変更していない。

同制御・同リセットで別途0.35秒を生成すると先頭0.20秒はbitwise一致する（因果的生成の当然の性質）。ただし今回の目標は新たに生成し、旧目標とf0が異なり、評価長・STFTフレーム数も異なる。0.35秒結果と同一条件の数値として混在させない。`diagnostics/target_transient.png`とスペクトログラムで初期過渡と終盤の持続音を確認した。

## Oracleと評価・失敗時の扱い

1D=ε_l、2D=ε_l/p_s、3D=ε_l/p_s/f0_l。非学習値は全て同じ目標値へ固定。float64 Oracleの最大誤差は {max(r['error'] for r in review['oracles'] if r['action_dtype']=='float64'):.3g}、float32行動丸めを含む最大は {max(r['error'] for r in review['oracles'] if r['action_dtype']=='float32'):.3g}。波形の絶対差/目標peak比は `diagnostics/review.json` に別記した。

主指標は全0.20秒 `acoustics.distance` の対数PSD＋5 ms RMS包絡。共通target peakで両波形を校正し、個別振幅正規化や位相合わせはしない。学習報酬はその負値、γ=1。Phase0のλ=.5・局所10 ms報酬を持ち込まない。WAVは全run共通のtarget peakで割ったfloat32、npzは未正規化float64。表示スペクトログラムのpeak正規化は従来の描画定義だけに使い、損失やWAVには適用しない。

数値エラー候補は除外せず、診断用音声を10000 sampleへゼロ埋めしreward=-1e6。実際の誤差とfailure flagを別記する。有限の大きな報酬はclipしない。loss/勾配/更新後パラメータのNaN/Infを全更新で検査し、発生時は停止・failure.json保存。無音フラグも全候補に保存。今回の訓練失敗 {sum(r['failures'] for r in rows)} 件、無音 {sum(r['silent'] for r in rows)} 件、nonfinite 0件。候補を再試行・除外していない。

## 学習条件・候補と更新の区別

Issue13の基準を維持: LR=.0003、net[32,32]、batch32、warmup16、buffer50000、train_freq=1、gradient_steps=1、γ=1、TD3 noise=.1、SAC entropy=`auto_0.01`。DE pop8、mutation(.5,1)、recombination=.7、31世代＋初期8候補、polishなし。全手法CPU、workerごとPyTorch 1 thread。詳細版とソース/native binary hashはmanifest。

RLの256エピソード=256方策決定=256候補音=512,000,000 native stepであり、Phase0の102,400行動と異なる単位。各RL runはgradient iterations240、TD3 actor120/critic240、SAC actor/critic/entropy各240。決定論的独立評価は **0,8,…,248（更新前）と256（最終更新後）の33回**。旧Issue13の毎回評価とは費用が異なる。モデル・replay・optimizer・entropy・RNGの評価前後hash不変を確認した。

DE/Randomの初期は最初の**予算内候補1**。全256候補のactual errorとbest-so-farを保存。主評価は**実際の候補256**、補助はbest。DE内部が選んだincumbentは`de_selected_incumbent.json`へ別保存した。候補256をbestに置換しない。最終候補の独立再生を1回だけ追加する。

## 最終の3 seed平均 ± 標本SD

{overview}

{chr(10).join(statements)}

初期・途中最良・最終は各runの別ディレクトリ/summaryとして保存し、RLは各段階のagentも保存。直接法のモデルに相当する解はactions/physical_controlsである。bestは同じ目標への反復評価で選んだ補助値であり、汎化評価ではない。

## 個別値

{individual}

## 計算量と時間

- 訓練: {execution['training_candidates']}候補、native {train_steps:,} step。独立評価: {execution['evaluation_sounds']}音、native {eval_steps:,} step。
- 実験の並列wall {execution['concurrent_run_wall_s']:.2f}秒。学習process wall合計 {execution['training_process_wall_s']:.2f}秒、評価合計 {execution['evaluation_process_wall_s']:.2f}秒、保存合計 {execution['save_process_wall_s']:.2f}秒。process CPU合計 {execution['process_cpu_s']:.2f}秒。並列wallと合算時間は異なる。
- 事前物理診断: {review['setup_generated_sounds']}音、native {review['setup_native_steps']:,} step、{review['setup_wall_s']:.2f}秒。診断描画 {review['diagnostic_plot_wall_s']:.2f}秒。nativeを使う4テスト（別の短いTD3/SAC 24候補を含む）は比較予算外、`tests.json`参照。
- 上記gate前の対話的な物理下調べも別記: 22音・44,000,000 native step（`preliminary_checks.json`）。9音surveyは2.024秒、残る13音probeのwallは未計測。4テスト全体は共通目標生成を含む60音要求（1件の模擬失敗を含む）・実積分118,000,000 step、15.544秒。これらを比較用の256候補へ混ぜていない。
- 保存成果物の別プロセス監査: {audit['audit_generated_sounds']}音、native {audit['audit_native_steps']:,} step、{audit['audit_wall_s']:.2f}秒。全{len(audit['reload_comparisons'])}初期/best/最終再生で保存前後の評価差0・波形bitwise一致。描画/レポートは別途 `report_timing.json`。
- attempted native stepは失敗したブロックも要求全量を数える上限値、completedは完了ブロックのみ。各runに双方のブロック/積分時間と失敗数を保存。

## 図・監査・解釈

`figures/learning_curves_seed0.png`と`learning_curves_mean_sd.png`は次元別4手法、生の観測点を平滑化せず描画。RLは独立した現時点方策、直接法は実候補値であり同義ではない。`best_observed_*`は別図で、RLの評価済み方策の最良と直接法の候補最良を区別して読む。色・全パネルの軸は統一、凡例は図外。標本SD帯は算術的mean±SDをそのまま描き、負側の帯があっても誤差自体が負になった意味ではない。seed0の最終スペクトログラム（同じ0–10 kHz/時間/表示尺度）と物理制御の0–0.20秒一定値も保存した。`figure_sources.json`で最終npz hash・評価値・行動を結び付ける。

独立監査 {audit['checks']} checks passed。既存成果物 {audit['protected_files_verified']}ファイルのhashを確認し、旧0.35秒/Phase0/実音声データを変更しない。新環境の4テストで秒数取り違え、Oracle、状態リセット、失敗予算、SB3自動リセット後の候補記録を検証した。

1行動の問題は静的ブラックボックス探索に近い。RLがDEより有利という前提は置かない。全次元で同じ目標が既に表現可能なので、次元増加による表現空間の拡張と、固定256予算で探索が難しくなることを分けて考える。3Dは発声領域が狭いことも探索難度に寄与する。今回だけで最適化器一般の優劣・大域的パラメータ同定・実録音汎化・左右2音源の効果を結論しない。精度が不十分でも予算を延ばさず、時変制御へは進まない。

## 実行・再検証

```bash
cd /home/tatsukawa/work/Birdsong_tatsukawa
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue17-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.model
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 audit
# 今回の出力が存在しない作業コピーで、既存結果を維持して新規実行:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 train --workers 6
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 finish
```

prepare/train/finishは既存出力を上書きせずエラーにする。全成果は指定の `results/phase1_constant_020s/` に集約。Gitのcommit/pushやIssue状態変更は実施していない。

実際のPythonパス・Python/C++コンパイラ版・CPU機種・CPU affinity・描画コードhashは `completion_verification.json` に保存した。GPUは使用していない。
'''
    (output/'REPORT.md').write_text(text)
    write_json(output/'report_timing.json',dict(plot_and_report_wall_s=time.perf_counter()-started,
        figures=len(list(figures.glob('*.png'))),smoothing=False,sample_sd_ddof=1))
