"""Read A/B/C saved records; compare identical full-waveform evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from .issue15_analysis import write_json, write_csv
from .issue14_report import _spectrogram, _control_plot, COLORS as ALGORITHM_COLORS

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"results/phase0_fixed_target_20261108"
CONDITIONS=("combined","spectral_only","envelope_only")
ALGORITHMS=("PPO","TD3","SAC")
COLORS={"combined":"#0072B2","spectral_only":"#009E73","envelope_only":"#D55E00"}
LABELS={"combined":"A: combined","spectral_only":"B: spectral only","envelope_only":"C: envelope only"}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |",
        *("| "+" | ".join(map(str,row))+" |" for row in rows)])


def fmt(mean,sd): return f"{mean:.6f} ± {sd:.6f}"


def controls_with_clear_legend(run,path):
    try:
        _control_plot(run,path)
        return "inside upper right, verified no overlap"
    except RuntimeError as error:
        if "legend overlaps" not in str(error): raise
    snapshot=run["_snapshot"]; times=np.arange(36)*.01
    with plt.style.context(["science","ieee","no-latex"]),plt.rc_context({"path.simplify":False}):
        fig,axes=plt.subplots(2,1,figsize=(14,12),sharex=True)
        try:
            for i,ax in enumerate(axes):
                for key,color,label in (("target_controls","black","Target"),("physical_controls",ALGORITHM_COLORS[run["algorithm"]],"Generated")):
                    values=snapshot[key][:,i]
                    ax.step(times,np.r_[values,values[-1]],where="post",linewidth=2.8,linestyle="-",color=color,label=label)
                ax.set_ylabel("Frequency [Hz]" if i==0 else "Amplitude",fontsize=46)
                ax.tick_params(labelsize=30,width=2,length=10,direction="in",top=False,right=False)
                for spine in ax.spines.values():spine.set_linewidth(2)
                ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=2,fontsize=30,frameon=False)
            axes[0].set_ylim(500,3000);axes[1].set_ylim(-.03,1);axes[1].set_xlim(0,.35)
            axes[1].set_xlabel("Time [s]",fontsize=46)
            fig.tight_layout(h_pad=3)
            fig.savefig(path,dpi=120,bbox_inches="tight",pad_inches=.1)
        finally:plt.close(fig)
    return "above axes to avoid measured data; physical axis limits unchanged"


def generate_report(output,control):
    output,control=Path(output),Path(control)
    started=time.perf_counter(); rows=[]; all_curves=[]; runs={}; source_hashes={}; figures=[]
    for condition in CONDITIONS:
        runs[condition]={}
        for name in ALGORITHMS:
            for seed in (0,1,2):
                directory=(control if condition=="combined" else output/condition)/f"{name}_{seed}"
                summary=json.loads((directory/"summary.json").read_text())
                curve=json.loads((directory/"curve.json").read_text())
                best=json.loads((directory/"best.json").read_text())
                assert [p["steps"] for p in curve]==list(range(0,102401,2048))
                assert curve[-1]["error"]==summary["error"] and curve[-1]["evaluation_phase"]=="post_final_update"
                logical=Path() if condition=="combined" else Path("reward_ablation")/condition
                for filename in ("summary.json","curve.json","best.json","agent.zip","rollout.npz"):
                    source_hashes[str(logical/directory.name/filename)]=digest(directory/filename)
                row=dict(condition=condition,algorithm=name,seed=seed,E_spec=summary["metrics"]["log_spectral"],
                    E_env=summary["metrics"]["envelope"],E_total=summary["error"],
                    frequency_mae_hz=summary["tracking"]["frequency_mae_hz"],amplitude_mae=summary["tracking"]["amplitude_mae"],
                    initial_E_total=summary["initial_error"],initial_frequency_mae_hz=summary["initial_tracking"]["frequency_mae_hz"],
                    initial_amplitude_mae=summary["initial_tracking"]["amplitude_mae"],
                    best_error=best["error"],best_steps=best["steps"],
                    first_recorded_below_008=next((p["steps"] for p in curve if p["error"]<=.08),None),
                    training_steps=summary["training_steps"],evaluation_steps=summary["evaluation_steps"],
                    training_wall_s=summary["training_wall_s"],evaluation_wall_s=summary["evaluation_wall_s"])
                assert abs(row["E_total"]-row["E_spec"]-row["E_env"])<1e-14
                rows.append(row)
                for point in curve:
                    all_curves.append(dict(condition=condition,algorithm=name,seed=seed,steps=point["steps"],
                        E_spec=point["metrics"]["log_spectral"],E_env=point["metrics"]["envelope"],E_total=point["error"],
                        frequency_mae_hz=point["tracking"]["frequency_mae_hz"],amplitude_mae=point["tracking"]["amplitude_mae"]))
                runs[condition][(name,seed)]=(summary,curve,directory)
    aggregates=[]
    for condition in CONDITIONS:
        for name in ALGORITHMS:
            selected=[r for r in rows if r["condition"]==condition and r["algorithm"]==name]
            record=dict(condition=condition,algorithm=name)
            for field in ("E_spec","E_env","E_total","frequency_mae_hz","amplitude_mae","best_error","training_wall_s","evaluation_wall_s"):
                record[field+"_mean"]=float(np.mean([r[field] for r in selected]))
                record[field+"_sample_sd"]=float(np.std([r[field] for r in selected],ddof=1))
            aggregates.append(record)
    write_csv(output/"comparison_per_seed.csv",rows)
    write_csv(output/"comparison_aggregate.csv",aggregates)
    write_csv(output/"comparison_curves.csv",all_curves)
    folder=output/"figures";folder.mkdir(exist_ok=True)
    for mean in (False,True):
        with plt.style.context(["science","ieee","no-latex"]),plt.rc_context({"path.simplify":False}):
            fig,axes=plt.subplots(3,1,figsize=(12,12),sharex=True)
            for ax,name in zip(axes,ALGORITHMS):
                for condition in CONDITIONS:
                    curves=[runs[condition][(name,seed)][1] for seed in (0,1,2)]
                    x=np.array([p["steps"] for p in curves[0]])
                    array=np.array([[p["error"] for p in curve] for curve in curves])
                    values=array.mean(axis=0) if mean else array[0]
                    ax.plot(x,values,label=LABELS[condition],color=COLORS[condition],linewidth=2)
                    if mean:
                        sd=array.std(axis=0,ddof=1)
                        ax.fill_between(x,values-sd,values+sd,color=COLORS[condition],alpha=.15)
                ax.set_ylabel(name+" acoustic error",fontsize=20); ax.tick_params(labelsize=15)
                ax.legend(fontsize=13,loc="upper right")
            axes[-1].set_xlim(0,102400);axes[-1].set_xlabel("Training environment steps",fontsize=20)
            fig.tight_layout()
            name="learning_curves_mean_sd.png" if mean else "learning_curves_seed0.png"
            fig.savefig(folder/name,dpi=140,bbox_inches="tight");plt.close(fig)
            figures.append(dict(path="figures/"+name,role="three algorithms, shared evaluation, raw 51 points",mean=mean,sd_ddof=1,smoothing=False,threshold_line=False))
    for condition in CONDITIONS[1:]:
        for name in ALGORITHMS:
            summary,curve,directory=runs[condition][(name,0)]
            with np.load(directory/"rollout.npz",allow_pickle=False) as archive:
                snapshot={k:archive[k].copy() for k in archive.files}
            render_run=dict(summary,_snapshot=snapshot)
            for kind,render in (("spectrogram",_spectrogram),("controls",controls_with_clear_legend)):
                filename=f"{condition}_{name}_seed0_{kind}.png"
                layout=render(render_run,folder/filename)
                figures.append(dict(path="figures/"+filename,condition=condition,algorithm=name,seed=0,
                    stage="post_final_update",policy_sha256=summary["final_parameter_sha256"],
                    source=f"{condition}/{name}_0/rollout.npz",title=None,legend_layout=layout))
    for figure in figures: figure["sha256"]=digest(output/figure["path"])
    elapsed=time.perf_counter()-started
    write_json(output/"figure_metadata.json",dict(source_sha256_relative_to_parent=source_hashes,figures=figures,
        added_training_steps=0,added_policy_evaluation_steps=0,figure_and_report_generation_wall_s=elapsed,
        common_evaluation="Full 0.35 s combined spectral+envelope error; f/A errors separate"))
    execution=json.loads((output/"execution.json").read_text()); provenance=json.loads((output/"manifest.json").read_text())
    analysis=json.loads((output/"reanalysis/analysis.json").read_text())
    observations=analysis["selected_observations"]
    by_key={(r["condition"],r["algorithm"]):r for r in aggregates}
    combined_best=[name for name in ALGORITHMS if by_key[("combined",name)]["E_total_mean"] < min(by_key[(c,name)]["E_total_mean"] for c in CONDITIONS[1:])]
    findings="今回、合算報酬Aの最終共通誤差平均が両単独条件より小さかった手法: "+", ".join(combined_best)+"。"
    findings+="スペクトル単独では振幅MAEの残存、包絡単独では周波数MAEの増大が見られ、単一成分の一致を模倣全体の成功と同一視できない。"
    import csv
    with (output/"training_reward_distributions.csv").open() as stream:
        distributions=list(csv.DictReader(stream))
    reward_rows=[]
    for condition in CONDITIONS:
        for name in ALGORITHMS:
            selected=[r for r in distributions if r['condition']==condition and r['algorithm']==name and r['quantity']=='reward']
            reward_rows.append([condition,name]+[f"{np.mean([float(r[k]) for r in selected]):.6f}" for k in ('mean','sd','q05','q95')])
    lines=["# Issue #15：音響報酬の分解・単独報酬比較", "",
        f"基準ブランチ: `{provenance['branch']}`、Aのcommit: `{provenance['control_commit']}`。mainの旧4,096-step版ではなく、保存済み102,400-step版を対照に使用。", "",
        "Aは既存の合算報酬、Bはスペクトル単独、Cは包絡単独。B/Cのみ各3手法×3 seed、102,400ステップをゼロから学習した。Aは再学習していない。"
        "目標・音生成・観測・行動・seed管理・ネットワーク・学習率等は共通で、学習スカラーに含める報酬項だけを変更。評価は全条件で同じ合算距離と周波数/振幅MAE。", "",
        "## 最終結果（3 seed平均±標本SD）", "",
        table(["条件","手法","全区間E_spec","全区間E_env","全区間E_total","周波数MAE[Hz]","振幅MAE"],[[r['condition'],r['algorithm']]+[fmt(r[k+'_mean'],r[k+'_sample_sd']) for k in ('E_spec','E_env','E_total','frequency_mae_hz','amplitude_mae')] for r in aggregates]), "",
        "主結果は最終更新後の方策。全27 runの個別値、初期値、途中最良、参考精度への到達点、計算時間は [comparison_per_seed.csv](comparison_per_seed.csv)。"
        "到達点は2,048ステップの評価グリッド上の観測値で、真の初到達stepを特定したものではない。0.08は聴覚的同一性の合格基準ではない。", "",
        findings,"",
        "## 学習推移", "",
        "![seed 0比較](figures/learning_curves_seed0.png)", "",
        "各パネルはPPO・TD3・SAC、各線はA/B/C。同じ51点を平滑化せず描いた。最終点は最終更新後、最後の更新前callbackは各runに別保存。"
        "[3 seed平均±SD](figures/learning_curves_mean_sd.png) と [comparison_curves.csv](comparison_curves.csv) に全seedの推移を示す。", "",
        "## 再学習なしのA成分分解", "",
        "既存curve/best/summary JSONだけを読み、追加学習0・追加方策評価0で再分析した。入力hashは [reanalysis/analysis.json](reanalysis/analysis.json)。", "",
        table(["手法","step","E_spec平均","E_env平均","E_total平均"],[[r['algorithm'],r['steps'],f"{r['E_spec_mean']:.6f}",f"{r['E_env_mean']:.6f}",f"{r['E_total_mean']:.6f}"] for r in observations]), "",
        "SACの30,720→最終の悪化は主にスペクトル項の増大で、包絡項は同程度。この記述だけでは包絡報酬の干渉が原因だとはいえない。"
        "[seed 0成分曲線](reanalysis/components_seed0.png)、[3 seed平均±SD](reanalysis/components_mean_sd.png)、"
        "[初期・20,480・51,200・最終・各seed最良の比較](reanalysis/selected_points.csv) を保存した。", "",
        "E_specは80 dBで正規化した対数パワースペクトル差であり、周波数制御誤差そのものではない。E_envは約5 ms RMS包絡差であり、振幅制御誤差そのものではない。"
        "両者は現行scale=1では無次元でもオーダーや学習寄与は同じとは限らず、周波数MAE[Hz]と無次元振幅MAEを別に比較した。", "",
        f"例えばTD3のBでは周波数MAEがAの{by_key[('combined','TD3')]['frequency_mae_hz_mean']:.2f} Hzから{by_key[('spectral_only','TD3')]['frequency_mae_hz_mean']:.2f} Hzへ小さくなったが、"
        f"E_specは{by_key[('combined','TD3')]['E_spec_mean']:.5f}から{by_key[('spectral_only','TD3')]['E_spec_mean']:.5f}へ大きくなった。スペクトル項を周波数制御誤差と置き換えて説明できない具体例である。", "",
        "## 実際の学習報酬とスケール", "",
        "学習器には毎step一つのスカラーを渡す。Bは−E_spec、Cは−E_envで、除いた項もinfoとtraining_traceへ記録。評価環境では常に合算報酬を使用する。"
        "共通の主評価は完成0.35秒波形の距離であり、35個の10 ms局所報酬の和とは異なる。", "",
        "[training_reward_distributions.csv](training_reward_distributions.csv) に実報酬のmin/mean/SD/分位点/maxとB/C局所成分を記録。"
        "Aは既存training_traceのスカラー分布を利用し、未記録だったAの局所成分は再生成・推定していない。"
        "報酬項の除去は数値スケールやSACのentropy項との相対関係も変えるため、差を項の干渉だけへ帰属しない。", "",
        table(["条件","手法","報酬mean","報酬SD","p05","p95"],reward_rows),"",
        "上表は各seedの全102,400学習stepを混ぜた報酬分布の統計を3 seedで平均した値で、学習終盤だけの分布ではない。SDはstep分布のSDであり、最終性能表のseed間標本SDとは区別する。","",
        "## 代表スペクトログラムと制御", "",
        "全図はseed 0の最終更新後の同じモデル・音声・制御。既存SciencePlots系スタイル、0–2500 Hz、Blackman–Harris窓、表示専用10 MHz、gray_r、−50〜−20 dB、図中タイトルなしを維持。", "",
        *[f"- {condition} / {name}: [スペクトログラム](figures/{condition}_{name}_seed0_spectrogram.png)、[周波数・振幅](figures/{condition}_{name}_seed0_controls.png)" for condition in CONDITIONS[1:] for name in ALGORITHMS], "",
        "Aの図は親のfigures/をそのまま利用する。特にCの周波数追従は、合計誤差や包絡項が小さいことだけから成功と判断せず、上のHz単位のMAEと制御図を確認する。", "",
        "## 重みλ・カリキュラムの判断", "",
        findings,"",
        "この共通評価・予算では両項を維持するAを当面の基準とし、今回の結果から直ちに重みを変更する必要性は認めない。"
        "ただし最適な正のλは同定しておらず、合算報酬の普遍的な必須性やλ=1の最適性を証明したものではない。"
        "Bはλ=0に相当し、Cはスペクトル係数を0にした対照で、正のλ探索とは区別する。"
        "段階的切替・新しい正のλの追加学習は未検証で見送り、将来実施する場合は値・予算・seedを事前固定する。判断は [decision.json](decision.json) に記録。", "",
        "## 計算費用・検証", "",
        f"B/C新規学習: {execution['completed_runs']} run、学習{execution['training_steps']:,}、run内評価{execution['evaluation_steps']:,}環境ステップ。"
        f"実行wall {execution['execution_wall_s']:.3f}秒、CPU各1スレッド・独立{provenance['workers']}プロセス。図・比較表生成は約{elapsed:.3f}秒で別計上。厳密な速度順位は評価しない。", "",
        "[audit.json](audit.json) では全モデルの再読み込み、最終音声/制御/曲線/指標の一致、実更新数、全stepの報酬分解、有限値、評価中に学習器を更新しない署名を検査。"
        "Oracle・同一目標・再現性・共通評価のテストはtest_results.json/txtを参照。Aと同じライブラリ版を使用し、差異一覧はmanifestに記録した。", "",
        "## 保存構造と再実行", "",
        "Phase 0の親はphase0_fixed_target_20261108/の一つ。保存名の日付は歴史的な名前で、実行日ではない。"
        "Aは親直下、B/Cはこのreward_ablation内、旧ランダム目標の全固有データは親のlegacy_random_target/。"
        "移設前後の全hash/サイズ・Git復元経路・参照の扱いはlegacy_inventory.json、reference_inventory.json、integration.jsonを参照。", "",
        "初回実行（既存の同名B/C成果は上書きしない）:", "", "```bash",provenance['reproducible_command'],"```", "",
        "保存済みの値・図だけを再生成:", "", "```bash", "PYTHONDONTWRITEBYTECODE=1 Deeplearning/.venv-rl/bin/python -m Deeplearning.issue15_report", "```", "",
        "本結果は目標f/Aが観測される固定正弦波のin-target追従課題。鳴管の逆推定、実録音、左右相互作用、未知音への汎化は示さない。"
        "3 seedの有限予算・途中最良の選択バイアス・学習報酬スケールの違いを保ったまま解釈する。", "",
    ]
    (output/"REPORT.md").write_text("\n".join(lines))
    spectral_total={r['algorithm']:r['E_total_mean'] for r in aggregates if r['condition']=='spectral_only'}
    combined_total={r['algorithm']:r['E_total_mean'] for r in aggregates if r['condition']=='combined'}
    decision=dict(weight_search_performed=False,curriculum_performed=False,positive_lambda_values_tested=[1],spectral_only_equivalent_lambda=0,
        interpretation="A outperforms either single-term condition on shared final mean total in the algorithms listed; no positive lambda was optimized",
        combined_best_algorithms=combined_best,
        next_action="Retain both terms (lambda=1) as current reference; no immediate weight change justified by this finite ablation. Other positive weights and curricula remain unverified",
        spectral_only_minus_combined_final_mean={k:spectral_total[k]-combined_total[k] for k in ALGORITHMS},
        evidence="comparison_aggregate.csv; common E_spec/E_env/E_total and f/A MAE, three seeds",
        necessity="Not established by single-term ablations alone; use observed accuracy/tradeoffs to preregister a separate weighted study if needed")
    write_json(output/"decision.json",decision)
    return aggregates


def update_index(control,aggregates):
    text=["# Phase 0：固定目標と報酬アブレーション", "",
        "正本ブランチはmain_deeplearning。保存名20261108は歴史的な名前で、実行日は各manifestを参照。", "",
        "- [Issue #15の比較レポート](reward_ablation/REPORT.md)：A/B/C、3手法×3 seed、共通評価。",
        "- [Issue #14のA長期学習レポート（保存時の原文）](CONTROL_REPORT.md)",
        "- [再学習なしの成分曲線](reward_ablation/reanalysis/components_mean_sd.png)",
        "- [A/B/Cの学習曲線](reward_ablation/figures/learning_curves_seed0.png)",
        "- [旧ランダム目標の実データ](legacy_random_target/)：移設前と全bytes一致。", "",
        table(["条件","手法","最終共通誤差 平均±標本SD","周波数MAE[Hz]","振幅MAE"],[[r['condition'],r['algorithm'],fmt(r['E_total_mean'],r['E_total_sample_sd']),f"{r['frequency_mae_hz_mean']:.3f}",f"{r['amplitude_mae_mean']:.6f}"] for r in aggregates]), "",
        "Aは保存済み9 runを再利用。B/Cはそれぞれ9 runをゼロから102,400環境step、2,048step毎に評価。報酬項の選択以外は共通。"
        "主評価は最終方策の全0.35秒距離とf/A MAEで、局所報酬や途中最良とは区別した。", "",
        "スカラーの数値スケールが変わるため、差を項の干渉だけで説明しない。λ調整・カリキュラムは未検証で、追加学習は行っていない。"
        "固定正弦波のin-target成績であり、未知音・鳴管・実録音への一般化は主張しない。", "",
        "親直下のAモデル・音声・曲線・JSONと既存図は保持した。過去のmanifest/CSV/復元記録にある旧results/phase0パスは履歴で、"
        "現在はlegacy_random_targetへ対応する。監査・復元用resolverは旧パスのhash値を変更せず解決する。", "",
    ]
    (Path(control)/"REPORT.md").write_text("\n".join(text))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PARENT/"reward_ablation")
    parser.add_argument("--control",type=Path,default=PARENT)
    args=parser.parse_args()
    values=generate_report(args.output,args.control)
    if args.output.resolve()==(args.control/"reward_ablation").resolve(): update_index(args.control,values)
