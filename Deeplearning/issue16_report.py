"""Read-only baseline comparison and final Phase 0 report for Issue 16."""
import json
from pathlib import Path
import time
import numpy as np

from .issue15_analysis import write_csv,write_json
from .issue15_report import _spectrogram,controls_with_clear_legend,table


def report(output, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots
    started=time.perf_counter();rows=[];points=[];runs={}
    for lr,root in ((.0003,baseline),(.0001,output)):
        for name in ('TD3','SAC'):
            for seed in (0,1,2):
                folder=root/f'{name}_{seed}'
                s=json.loads((folder/'summary.json').read_text())
                curve=json.loads((folder/'curve.json').read_text())
                best=json.loads((folder/'best.json').read_text())
                row=dict(algorithm=name,learning_rate=lr,seed=seed,
                    E_spec=s['metrics']['log_spectral'],E_env=s['metrics']['envelope'],E_total=s['error'],
                    frequency_mae_hz=s['tracking']['frequency_mae_hz'],amplitude_mae=s['tracking']['amplitude_mae'],
                    initial_error=s['initial_error'],best_error=best['error'],best_steps=best['steps'],
                    first_recorded_below_008=s['first_threshold_step'],
                    training_wall_s=s['training_wall_s'],evaluation_wall_s=s['evaluation_wall_s'],
                    training_environment_steps=s['training_steps'],evaluation_environment_steps=s['evaluation_steps'],
                    gradient_iterations=s['sb3_n_updates'],actor_optimizer_steps=s['optimizer_steps']['actor'],
                    critic_optimizer_steps=s['optimizer_steps']['critic'],entropy_optimizer_steps=s['optimizer_steps'].get('entropy',0),
                    nonfinite=s['nonfinite_detected'],failures=s['training_numerical_failures'],final_silent=s['metrics']['generated_silent'])
                rows.append(row);runs[(name,lr,seed)]=(s,curve,folder)
                for p in curve:
                    points.append(dict(algorithm=name,learning_rate=lr,seed=seed,steps=p['steps'],
                        E_spec=p['metrics']['log_spectral'],E_env=p['metrics']['envelope'],E_total=p['error'],
                        frequency_mae_hz=p['tracking']['frequency_mae_hz'],amplitude_mae=p['tracking']['amplitude_mae']))
    aggregates=[]
    fields=('E_spec','E_env','E_total','frequency_mae_hz','amplitude_mae','best_error','best_steps',
            'training_wall_s','evaluation_wall_s')
    for name in ('TD3','SAC'):
        for lr in (.0003,.0001):
            group=[r for r in rows if r['algorithm']==name and r['learning_rate']==lr]
            a=dict(algorithm=name,learning_rate=lr,n=3)
            for key in fields:
                vals=[r[key] for r in group];a[key+'_mean']=float(np.mean(vals));a[key+'_sample_sd']=float(np.std(vals,ddof=1))
            aggregates.append(a)
    write_csv(output/'comparison_per_seed.csv',rows)
    write_csv(output/'comparison_aggregate.csv',aggregates)
    write_csv(output/'comparison_curves.csv',points)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    for mean in (False,True):
        with plt.style.context(['science','ieee','no-latex']),plt.rc_context({'path.simplify':False}):
            fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
            for ax,name in zip(axes,('TD3','SAC')):
                for lr,color in ((.0003,'#0072B2'),(.0001,'#D55E00')):
                    curves=[runs[(name,lr,s)][1] for s in (0,1,2)]
                    x=[p['steps'] for p in curves[0]];y=np.array([[p['error'] for p in c] for c in curves])
                    center=y.mean(axis=0) if mean else y[0]
                    ax.plot(x,center,label=f'LR={lr:.4f}',color=color,lw=2)
                    if mean:
                        sd=y.std(axis=0,ddof=1);ax.fill_between(x,center-sd,center+sd,color=color,alpha=.15)
                ax.set_ylabel(f'{name}: full acoustic error',fontsize=15)
                ax.tick_params(labelsize=12);ax.set_ylim(bottom=0)
                ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=13,frameon=False)
            axes[-1].set_xlim(0,102400);axes[-1].set_xlabel('Training environment steps',fontsize=16)
            fig.tight_layout(h_pad=3)
            fig.savefig(figures/('learning_curves_mean_sd.png' if mean else 'learning_curves_seed0.png'),dpi=140,bbox_inches='tight')
            plt.close(fig)
    for name in ('TD3','SAC'):
        for lr in (.0003,.0001):
            s,_,folder=runs[(name,lr,0)]
            with np.load(folder/'rollout.npz') as archive:snapshot={key:archive[key].copy() for key in archive.files}
            render=dict(s,_snapshot=snapshot)
            prefix=f'{name}_lr_{str(lr).replace(".","p")}_seed0'
            _spectrogram(render,figures/f'{prefix}_spectrogram.png')
            controls_with_clear_legend(render,figures/f'{prefix}_controls.png')
    execution=json.loads((output/'execution.json').read_text())
    audit=json.loads((output/'audit.json').read_text())
    fmt=lambda a,k:f"{a[k+'_mean']:.6g} ± {a[k+'_sample_sd']:.3g}"
    overview=table(['手法','LR','E_spec','E_env','E_total','周波数MAE Hz','振幅MAE'],
        [[a['algorithm'],a['learning_rate'],*[fmt(a,k) for k in fields[:5]]] for a in aggregates])
    detail=table(['手法','LR','seed','最終 E_total','途中最良 E_total','最良step','学習秒','評価秒'],
        [[r['algorithm'],r['learning_rate'],r['seed'],f"{r['E_total']:.7g}",f"{r['best_error']:.7g}",
          r['best_steps'],f"{r['training_wall_s']:.2f}",f"{r['evaluation_wall_s']:.2f}"] for r in rows])
    judgments=[]
    for name in ('TD3','SAC'):
        old,new=[a for a in aggregates if a['algorithm']==name]
        change=new['E_total_mean']-old['E_total_mean']
        judgments.append(f"- {name}: LR 0.0001 の平均最終誤差は {new['E_total_mean']:.6g}、既存0.0003との差は {change:+.6g}。"+
                         ('この3 seedでは改善。' if change<0 else 'この3 seedでは改善しなかった。'))
    text=f'''# Issue #16 — Phase 0 最終学習率比較

TD3/SAC・seed 0/1/2・λ=0.5・102,400環境ステップを固定し、LR=0.0001の6 runのみ新規実行。LR=0.0003の6 runは保存結果を読み取り専用で利用した。**本比較でPhase 0を終了する。追加探索・PPO再実験はしない。**

## 条件と整合性

0.35秒、48 kHz、35行動×10 ms、1000→2000 Hz glideと既存振幅系列。報酬は局所 `-(E_spec+0.5 E_env)`、共通評価は全区間 `E_spec+E_env`（λ=1）。CPU・各worker PyTorch 1 thread、MLP [32,32]、γ=.99、batch64、buffer50000、warmup256、train_freq=1、gradient_steps=1。TD3 noise=.1、SAC entropy=auto（初期1）。科学計算4ソースと依存版は既存manifestに一致し、実効設定はoptimizer groupsを含めLR以外一致、初期方策hashと256 warmup行動・報酬も一致した。

各runは102,400環境step（2925完了episode＋25部分step）であり、gradient iterationsは102,144。TD3の実optimizer呼出しはactor51,072/critic102,144、SACはactor/critic/entropy各102,144。評価は0＋2048ごとの51 callback点と独立した最終更新後1回の計52音、1820評価step。提示曲線51点の終点は最終更新後で、モデル・音声・表と一致。`callback_curve.json`に更新直前の元の終点も残した。`audit.json`で両者の差とhashを確認できる。

NaN/Infはloss、勾配、更新後パラメータを全更新で検査し0件、数値失敗0件。無音フラグとloss min/maxは各summary/curveに保持。無音や高誤差のrunを除外していない。Oracle・独立評価の学習状態/RNG不変性・保存モデル再読み込み・波形からの評価再計算を検証した。

## 最終方策の3 seed平均 ± 標本SD

{overview}

{chr(10).join(judgments)}

これは同一の学習目標音に対する小規模比較であり、汎化性能や有意差を示すものではない。途中最良を最終の代わりに採用しない。

## 個別値・途中最良（補助）・所要時間

{detail}

新規6 runの並列wall時間 {execution['concurrent_run_wall_s']:.2f}秒、学習process wall合計 {execution['training_process_wall_s']:.2f}秒、評価合計 {execution['evaluation_process_wall_s']:.2f}秒。保存等は各summaryのtotalと学習/評価との差に含まれる。描画・レポート時間は `report_timing.json` に別記。事前320 step smokeは比較予算外で `smoke.json` に記録。

## 成果物と再検証

- `comparison_per_seed.csv` / `comparison_aggregate.csv` / `comparison_curves.csv`: 全12 run、音響2成分・MAE・best・到達step・費用。
- `figures/learning_curves_seed0.png` / `learning_curves_mean_sd.png`: TD3/SACの2パネル、各パネルは2 LRだけ、生51点、mean±sample SD、平滑化なし。各LR seed0の最終スペクトログラム・物理制御図も保存。
- 新規6 runに最終/bestモデル、初期/最終npz・WAV、実行設定・lossとoptimizer回数・全訓練traceを保存。初期/best/finalの意味は各summary/best.jsonに分離。
- `manifest.json` / `protected_before.json` / `audit.json`: 版・計算ソース・旧成果物全hash。既存変更済みのグラフ2枚も現在のバイト列を維持した。監査 {audit['checks']} checks passed。

リポジトリ親から `.venv-rl` の絶対Pythonパスで実行する。既存結果がある場合、再学習コマンドは上書きを拒否する。

```bash
cd /home/tatsukawa/work/Birdsong_tatsukawa
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue16-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 audit
# 同じ既存baselineを持ち、今回の出力がまだ存在しない作業コピーで:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 train --workers 6
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue16 finish
```

Phase1の0.20秒実験は別Issue #17・別環境・γ=1・LR=.0003・全区間報酬で行う。この比較のLRやλは移植しない。

実際のPythonパス・Python/C++コンパイラ版・CPU機種・CPU affinity・描画コードhashは `completion_verification.json` に保存した。GPUは使用していない。
'''
    (output/'REPORT.md').write_text(text)
    write_json(output/'report_timing.json',dict(plot_and_report_wall_s=time.perf_counter()-started))
