"""Current72-profile comparison, with explicit objective/common selection semantics."""
import csv
import json
from pathlib import Path
import time
import numpy as np

from .issue15_analysis import write_json
from .issue15_report import table
from .issue16 import digest
from .issue17 import DEST,weight_label
from .issue17_direct_weights import WORK
from .issue17_extension_data import load_runs,write_csv,METHODS,CONDITIONS
from .issue17_comparison_figures import figures,COLORS


def new_acoustic_figures(stage,runs):
    import matplotlib.pyplot as plt
    import scienceplots
    from .plotting import draw_spectrogram
    from .model import LOW,HIGH
    from .constant020_env import INDICES
    metadata=[]
    for c in CONDITIONS:
        for m in ('DE','random_search'):
            folder=runs[(c,m,0,.5)]['folder'];stem=f"{c.replace('constant_','')}_{m}_lambda0p5"
            with np.load(folder/'rollout.npz') as z:target,audio,controls=z['target'].copy(),z['audio'].copy(),z['physical_controls'].copy()
            destination=stage/'figures/spectrograms'/(stem+'_seed0.png');destination.parent.mkdir(parents=True,exist_ok=True)
            with plt.style.context(['science','ieee','no-latex']):
                fig,axes=plt.subplots(2,1,figsize=(14,12),sharex=True)
                for ax,wave in zip(axes,(target,audio)):draw_spectrogram(ax,wave,50000)
                axes[-1].set_xlabel('Time [s]',fontsize=46);fig.tight_layout(pad=.5)
                fig.savefig(destination,dpi=120,bbox_inches='tight',pad_inches=.08);plt.close(fig)
            metadata.append(dict(path=str(destination.relative_to(stage)),kind='spectrogram',condition=c,method=m,
                envelope_weight=.5,stage='actual_candidate4096',title='',samples=10000,duration_s=.2,
                source_rollout_sha256=digest(folder/'rollout.npz'),image_sha256=digest(destination)))
            for i,name,label in ((0,'epsilon_l',r'$\varepsilon_l$'),(2,'ps',r'$p_s$'),(3,'f0_l',r'$f_{e,l}$')):
                if i not in INDICES[c]:continue
                destination=stage/'figures/physical_controls'/(stem+'_'+name+'_seed0.png');destination.parent.mkdir(parents=True,exist_ok=True)
                with plt.style.context(['science','ieee','no-latex']):
                    fig,ax=plt.subplots(figsize=(14,9));ax.plot([0,.2],[controls[0,i]]*2,color=COLORS[m],lw=2.8)
                    delta=.04*(HIGH[i]-LOW[i]);ax.set_ylim(LOW[i]-delta,HIGH[i]+delta);ax.set_xlim(0,.2)
                    ax.set_xlabel('Time [s]',fontsize=46);ax.set_ylabel(label,fontsize=46);ax.tick_params(labelsize=30,pad=10)
                    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False);ax.yaxis.get_offset_text().set_fontsize(30)
                    fig.tight_layout(pad=.5);fig.savefig(destination,dpi=120,bbox_inches='tight',pad_inches=.08);plt.close(fig)
                metadata.append(dict(path=str(destination.relative_to(stage)),kind='physical_control',condition=c,method=m,
                    envelope_weight=.5,stage='actual_candidate4096',parameter=name,title='',physical_value=float(controls[0,i]),
                    source_rollout_sha256=digest(folder/'rollout.npz'),image_sha256=digest(destination)))
    return metadata


def report(stage=WORK,*,render_images=True):
    if (DEST/'visualization_revision').exists():
        raise RuntimeError('The visualization revision supersedes legacy horizontal-control/audio plots; use issue17_visualization')
    stage=Path(stage);start=time.perf_counter();runs=load_runs(stage)
    base=json.loads((DEST/'manifest.json').read_text());execution=json.loads((stage/'execution.json').read_text())
    audit=json.loads((stage/'audit.json').read_text())
    with (stage/'baseline_comparison_per_seed.csv').open() as stream:
        baseline={(r['condition'],r['method'],int(r['seed']),float(r['envelope_weight'])):r for r in csv.DictReader(stream)}
    with (DEST/'comparison_aggregate.csv').open() as stream:
        baseline_aggregate={(r['condition'],r['method'],float(r['envelope_weight'])):r for r in csv.DictReader(stream)}
    rows=[];points=[];parameters=[];checkpoints=[]
    for key,data in runs.items():
        c,m,seed,w=key;summary=data['summary'];curve=data['curve'];final=data['final'];selected=data['objective_selected'];common=data['common_posthoc']
        old=dict(baseline.get(key,{}));training=summary['training'];evaluation=summary['evaluation']
        source_account=summary.get('source_sampling_account',training)
        row=dict(condition=c,method=m,envelope_weight=w,seed=seed,E_total=final['acoustic_error'],
            E_spec=final['acoustic_components']['log_spectral'],E_env=final['acoustic_components']['envelope'],
            E_weighted=final['weighted_loss'],parameter_nrmse=final['parameter_nrmse'],
            best_error=common['acoustic_error'],best_episode=common['episodes'],training_candidates=4096,
            evaluation_sounds=evaluation['episodes'],training_native_steps=training['native_steps_attempted'],
            training_wall_s=training['wall_s'],evaluation_wall_s=evaluation['wall_s'],
            initial_error=summary['initial']['acoustic_error'],
            best_parameter_nrmse=common['parameter_nrmse'],
            save_wall_s=summary.get('artifact_wall_s',0.),
            process_cpu_s=summary.get('process_cpu_s',''),
            epsilon_l=final['physical_controls'][0],ps=final['physical_controls'][2],f0_l=final['physical_controls'][3],
            returned_solution='final deterministic actor' if m in ('TD3','SAC') else 'actual candidate4096; J-selected archive reported separately',
            gradient_iterations=summary['gradient_updates'],
            objective_selected_E_total=selected['acoustic_error'],objective_selected_J=selected['weighted_loss'],
            objective_selected_E_spec=selected['acoustic_components']['log_spectral'],
            objective_selected_E_env=selected['acoustic_components']['envelope'],
            objective_selected_trial=selected['episodes'],objective_selected_parameter_nrmse=selected['parameter_nrmse'],
            common_posthoc_E_total=common['acoustic_error'],common_posthoc_trial=common['episodes'],
            pool_generation_wall_s=source_account['wall_s'],
            incremental_wall_s=summary['elapsed_s'] if key not in baseline else 0.,
            new_candidate_sounds=summary.get('new_candidate_sounds',0) if key not in baseline else 0,
            pool_reused=bool(summary.get('candidate_pool_reused',False)),
            source_sampling_native_steps=source_account['native_steps_attempted'],
            final_sustained_voiced=final['sustained_voiced'])
        # Existing54 numeric columns are preserved exactly, including their
        # historical best_error=post-hoc minimum common error definition.
        row.update(old)
        rows.append(row)
        for p in curve:
            points.append(dict(condition=c,method=m,envelope_weight=w,seed=seed,episodes=p['episodes'],
                E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],
                E_env=p['acoustic_components']['envelope'],E_weighted=p['weighted_loss'],
                parameter_nrmse=p['parameter_nrmse'],best_observed_error=p['common_posthoc_error'],
                objective_selected_common_error=p['objective_selected_common_error'],
                objective_selected_J=p['objective_selected_loss'],objective_selected_trial=p['objective_selected_trial'],
                common_posthoc_error=p['common_posthoc_error'],common_posthoc_trial=p['common_posthoc_trial'],
                gradient_updates=p['gradient_updates'],evaluation_phase=p['evaluation_phase'],
                role='independent_current_policy' if m in ('TD3','SAC') else 'actual_training_candidate'))
        for n in (256,512,1024,2048,4096):
            p=next(p for p in curve if p['episodes']==n)
            checkpoints.append(dict(condition=c,method=m,envelope_weight=w,seed=seed,trials=n,episodes=n,
                E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],E_env=p['acoustic_components']['envelope'],
                E_weighted=p['weighted_loss'],parameter_nrmse=p['parameter_nrmse'],best_observed_error=p['common_posthoc_error'],
                gradient_updates=p['gradient_updates'],evaluation_phase=p['evaluation_phase'],
                actual_E_total=p['acoustic_error'],objective_selected_common_error=p['objective_selected_common_error'],
                objective_selected_J=p['objective_selected_loss'],common_posthoc_error=p['common_posthoc_error']))
        from .model import LOW,HIGH
        from .constant020_env import INDICES
        for role,record in (('final_policy_or_actual_candidate',final),('objective_selected',selected)):
            for i,name in ((0,'epsilon_l'),(2,'ps'),(3,'f0_l')):
                if i not in INDICES[c]:continue
                value=record['physical_controls'][i];target=base['target_controls'][i]
                parameters.append(dict(condition=c,method=m,envelope_weight=w,seed=seed,selection=role,
                    parameter=name,value=value,target=target,error=value-target,range_normalized_error=(value-target)/(HIGH[i]-LOW[i])))
    write_csv(stage/'comparison_per_seed.csv',rows);write_csv(stage/'comparison_curves.csv',points)
    write_csv(stage/'checkpoint_comparison.csv',checkpoints);write_csv(stage/'parameter_errors.csv',parameters)
    groups=[(c,m,w) for c in CONDITIONS for w in (1.,.5) for m in METHODS];aggregates=[]
    fields=('E_total','E_spec','E_env','E_weighted','parameter_nrmse','objective_selected_E_total','objective_selected_J',
            'objective_selected_E_spec','objective_selected_E_env','objective_selected_parameter_nrmse','common_posthoc_E_total',
            'pool_generation_wall_s','incremental_wall_s','evaluation_wall_s')
    for c,m,w in groups:
        row=dict(baseline_aggregate.get((c,m,w),{}));row.update(condition=c,method=m,envelope_weight=w,n=3)
        for field in fields:
            a=np.array([float(r[field]) for r in rows if (r['condition'],r['method'],float(r['envelope_weight']))==(c,m,w)])
            row.setdefault(field+'_mean',float(a.mean()));row.setdefault(field+'_sample_sd',float(a.std(ddof=1)))
        aggregates.append(row)
    write_csv(stage/'comparison_aggregate.csv',aggregates)
    effects=[]
    for c in CONDITIONS:
        for m in METHODS:
            row=dict(condition=c,method=m)
            for name,field in (('final','E_total'),('objective_selected','objective_selected_E_total'),('common_posthoc','common_posthoc_E_total')):
                delta=[float(next(r[field] for r in rows if (r['condition'],r['method'],int(r['seed']),float(r['envelope_weight']))==(c,m,s,.5)))-
                       float(next(r[field] for r in rows if (r['condition'],r['method'],int(r['seed']),float(r['envelope_weight']))==(c,m,s,1.))) for s in (0,1,2)]
                row[name+'_delta_mean']=float(np.mean(delta));row[name+'_delta_sample_sd']=float(np.std(delta,ddof=1))
                if name=='final':
                    row.update(delta_definition='lambda0.5 minus lambda1 common final E_total',
                        mean_delta=float(np.mean(delta)),sample_sd_delta=float(np.std(delta,ddof=1)),
                        seed0_delta=delta[0],seed1_delta=delta[1],seed2_delta=delta[2])
            effects.append(row)
    write_csv(stage/'lambda_effects.csv',effects)
    if render_images:
        old_photos=[r for r in json.loads((DEST/'figure_sources.json').read_text())['images'] if r.get('kind') in ('spectrogram','physical_control')]
        new_photos=new_acoustic_figures(stage,runs)
        write_json(stage/'figure_sources.json',dict(images=old_photos+new_photos))
        figures(stage)
    else:
        photos=[r for r in json.loads((stage/'figure_sources.json').read_text())['images'] if r.get('kind') in ('spectrogram','physical_control')]
        old_photos=[r for r in photos if (DEST/r['path']).exists()]
        new_photos=[r for r in photos if (stage/r['path']).exists()]
    fmt=lambda r,k:f"{float(r[k+'_mean']):.6g} ± {float(r[k+'_sample_sd']):.3g}"
    final_table=table(['次元','λ','手法','最終方策/実候補4096 E_total','E_spec','E_env','Jλ','物理量NRMSE'],[
        [c.replace('constant_',''),w,m,*[fmt(r,k) for k in ('E_total','E_spec','E_env','E_weighted','parameter_nrmse')]]
        for (c,m,w),r in zip(groups,aggregates)])
    selected_table=table(['次元','λ','手法','Jλ採用解のE_total','採用解Jλ','採用解E_spec','採用解E_env','事後minE_total'],[
        [c.replace('constant_',''),w,m,*[fmt(r,k) for k in ('objective_selected_E_total','objective_selected_J',
            'objective_selected_E_spec','objective_selected_E_env','common_posthoc_E_total')]] for (c,m,w),r in zip(groups,aggregates)])
    effect_table=table(['次元','手法','λ.5−λ1:最終/実候補','λ.5−λ1:J採用解E_total','事後minE_total差'],[
        [r['condition'].replace('constant_',''),r['method'],*[f"{r[k+'_delta_mean']:+.6g} ± {r[k+'_delta_sample_sd']:.3g}"
          for k in ('final','objective_selected','common_posthoc')]] for r in effects])
    individual=table(['次元','手法','λ','seed','最終/実候補','J採用解E_total','J採用位置','事後minE_total'],[
        [r['condition'].replace('constant_',''),r['method'],r['envelope_weight'],r['seed'],f"{float(r['E_total']):.7g}",
         f"{float(r['objective_selected_E_total']):.7g}",r['objective_selected_trial'],f"{float(r['common_posthoc_E_total']):.7g}"] for r in rows])
    mean_sd=lambda values:f"{np.mean(values):.6g} ± {np.std(values,ddof=1):.3g}"
    progress_rows=[];control_rows=[];cost_rows=[]
    for c,m,w in groups:
        selected_rows=[r for r in rows if (r['condition'],r['method'],float(r['envelope_weight']))==(c,m,w)]
        label=[c.replace('constant_',''),w,m]
        progress_rows.append(label+[mean_sd([
            next(p for p in runs[(c,m,s,w)]['curve'] if p['episodes']==n)['objective_selected_common_error']
            for s in (0,1,2)]) for n in (256,1024,4096)])
        cells=[]
        for field,index in (('epsilon_l',0),('ps',2),('f0_l',3)):
            values=[float(r[field]) for r in selected_rows]
            cells.append(mean_sd(values)+' / '+mean_sd([v-base['target_controls'][index] for v in values]))
        control_rows.append(label+cells)
        cost_rows.append(label+[mean_sd([float(r['pool_generation_wall_s']) for r in selected_rows]),
            '/'.join(str(r['evaluation_sounds']) for r in selected_rows),
            '33' if m in ('TD3','SAC') else '4096',
            '既存プール再利用' if m=='random_search' and w==.5 else '既存記録' if (c,m,0,w) in baseline else '今回追加'])
    progress_table=table(['次元','λ','手法','256: J採用E_total','1024: J採用E_total','4096: J採用E_total'],progress_rows)
    control_table=table(['次元','λ','手法','ε_l: 値 / 目標差','p_s: 値 / 目標差','f0_l: 値 / 目標差'],control_rows)
    cost_table=table(['次元','λ','手法','候補生成/学習wall秒: 平均±SD','追加評価音数(seed0/1/2)','曲線の観測数/run','計算の出所'],cost_rows)
    de_notes=[]
    for c in CONDITIONS:
        selected=[r for r in aggregates if r['condition']==c]
        for w in (1.,.5):
            de=next(r for r in selected if r['method']=='DE' and r['envelope_weight']==w)
            winner=min((r for r in selected if r['envelope_weight']==w),key=lambda r:r['objective_selected_E_total_mean'])
            de_notes.append(f"- {c}・λ={w}: DEのJ採用解E_totalは {fmt(de,'objective_selected_E_total')}。"
                f"評価済みJ採用解の平均最小は {winner['method']}。3 seedのSDと個別値も併せて判断する。")
    text=f'''# Issue #17 — 4手法×λ=1/.5、0.20秒・一定制御・4096候補

**TD3/SACは再学習していない。** 既存54 runを凍結し、DE λ=.5を3次元×3 seedの9 runだけ新規探索。
Random Search λ=.5は同じ9候補プールの全4096候補を再評価して選択した。候補の再抽選/再生成は0。
比較表は72条件run相当（4手法×2λ×3次元×3 seed）だが、新規学習/探索はDE9 runだけである。

## 条件・目的関数と共通評価

Case A、0.20秒、50 kHz、10000 sample、1行動で全区間一定。1D=ε_l、2D=ε_l/p_s、3D=ε_l/p_s/f0_l。
目標は `{base['target_controls']}`。f0_lは外力/開閉操作項であり基本周波数Hzではない。
目標、LOW/HIGH、DT=1e-7、20×10 ms、初期状態/遅延履歴/因果フィルタ、音響関数は変更していない。
DEは既存population8、mutation(.5,1)、recombination=.7、初期乱数生成規則とseed0/1/2を維持。
両λの初期8候補が同一であることを検査。差は最小化する `Jλ=E_spec+λE_env` のλだけ。
全手法の比較用音響誤差は **E_total=E_spec+E_env**。重み付きJを共通評価へ混ぜない。
RL既存条件はLR=.0003、net[32,32]、batch32、warmup16、γ=1等をそのまま再利用する。

## 3種類の選択を区別する

1. **最終方策/実際の4096番目候補**: RLは4096環境step終了・最終更新後の決定論的方策。DE/Randomは実候補4096でありincumbentとは別。従来の最終数値を保持する。
2. **Jλ採用解**: 観測済み点のうちJλが最小のもの。同値は最初に観測した点を選ぶ。直接法の採用可能な解はこちら。RLは独立評価された33方策からの補助選択で、最終方策とは置き換えない。
3. **事後minE_total**: 共通評価で最良の観測点を事後的に選んだ補助値。Jλ採用解と同義ではない。

`figures/best_observed/` は **Jλ採用解のE_total** を描く。Jが下がってもE_totalは上がり得るため、
この曲線をE_totalの累積最小で単調化していない。事後minE_totalはCSV/表へ別列で残した。
SciPyが同値候補を返す場合のsolver自体の解も、新DEの `solver_returned/` と `de_selected_incumbent.json` に別記。
RLの元の `best/agent.zip` は旧コードが共通E_totalで選んだモデルであり、今回のJ採用モデルと偽って扱わない。
保存のない中間ネットワークを復元したとは主張せず、J採用時点の保存済み行動・物理量・評価値を使用する。

## 最終方策/実候補4096（3 seed平均 ± 標本SD）

{final_table}

Randomの実候補列・最後の候補はλに依存しないため、両λのこの最終値と生曲線は同一になる。
重みの影響は次の採用解選択に現れる。DE/Randomの実候補だけを採用可能な最良解と誤認しない。

## Jλ採用解と共通評価による事後最良

{selected_table}

{chr(10).join(de_notes)}

DEの高精度到達は、時間一定・少数変数で既知目標を再現する静的最適化に近い課題では不自然ではない。
その一方で、seed差・物理量の同定誤差・3Dの狭い持続発声領域も表とCSVに残し、一部seedを除外しない。
Oracleがほぼ0であることと、どの最適化手法も安定して到達できることは別である。

## λ変更の対応seed差（.5−1、共通E_total）

{effect_table}

負値がλ=.5での減少。最終/実候補・J採用解・事後最良の列を混ぜずに読む。
λ=.5の優位を先取りせず、3 seedの個別値と標本SDを示した小規模な条件内比較である。

## 探索・学習の進行

{progress_table}

1DはDEが早期に低誤差へ到達し、RLは改善後も最終方策で誤差が戻る例がある。
2DはDE λ=1のseed差が平均/SDにも残る一方、λ=.5は3 seedとも低誤差へ到達した。
3DではRLの改善が小さく、DE/Randomの採用解に段階的改善があるが、DEも全seedで目標へ収束したわけではない。
Randomの実候補誤差は試行ごとに変動する。λ変更による採用解の違いを、生曲線の改善と解釈しない。
実候補/現方策とJ採用解の両方を図と [checkpoint_comparison.csv](checkpoint_comparison.csv) で確認する。

## 個別seed

{individual}

最終およびJ採用解の物理制御・目標との差・範囲正規化誤差は [parameter_errors.csv](parameter_errors.csv)。
ε_l/p_s/f0_lの値はモデル係数であり、周波数Hzと誤記しない。
256/512/1024/2048/4096時点の実評価・J採用解・事後最良は [checkpoint_comparison.csv](checkpoint_comparison.csv)。

## 最終方策/実候補4096の制御値と目標差

各セルは値の3 seed平均±標本SD / (値−目標)の平均±標本SD。未制御の成分は目標に固定。
J採用解の制御値と最終値は区別し、前者の全値はparameter_errors.csvのselection=objective_selectedに保存した。

{control_table}

## 予算・再利用・計算時間

各profileの論理的候補予算は4096。RLは4096学習音＋独立方策評価33音、直接法は4096候補に追加検証音を別計上。
RLの現在方策33点とDE/Randomの実候補4096点では観測回数が異なる。学習時のノイズ付き候補と評価済み決定論的方策も同一視しない。
best比較も、RLは33方策の選択、直接法は4096候補の選択という差がある。

{cost_table}

wall秒は各runの学習/候補生成時間で、音声評価・保存を別計上する。Random λ=.5には元プールの生成時間を表示した。
追加評価音数は各条件で記録された実計算数であり、候補プール内4096回の音響誤差計算とは別である。
再利用した初期/最終/採用音は新たな評価音数に含めない。曲線の観測数は最適化の評価機会の差として併記した。

今回の新規DEは合計 **{execution['new_training_candidates']:,}候補**、native73,728,000,000 step。
CPU {execution['workers']} workersで並列wall **{execution['DE_parallel_wall_s']:.2f}秒**、訓練process wall合計{execution['DE_training_process_wall_s']:.2f}秒。
Randomは{execution['reused_random_candidates']:,}候補を再利用し、追加候補シミュレーション0、再スコア計{execution['random_rescoring_wall_s']:.6f}秒。
元の候補生成時間/native量は各source_sampling_accountと比較CSVに保存し、再スコア時間だけを新規4096候補探索の速度と比較しない。
λ=.5の採用音が保存されていない場合のみ保存行動を検証再生した。これは候補再抽選ではなく選択音の出力である。
今回の追加評価/選択音出力は{execution['new_evaluation_sounds']}音。独立監査はさらに{audit['audit_generated_sounds']}音を別計上。
既存の54 runは{json.loads((DEST/'execution.json').read_text())['concurrent_run_wall_s']:.2f}秒・別の27 worker条件で実行済み。
並列度/実行時負荷が異なるため、wall差だけを最適化器固有の速度差としない。
今回追加した直接法のgradient updateは0、既存RLは各4080 gradient iterationsのまま。candidate/native step/updateを区別した。

## 図と監査・整理

次元×λの6条件、各4手法を同じ画像に表示。seed0生曲線/3 seed平均±標本SD、J採用解のseed0/平均±SDを別々に計24枚。
SciencePlots、軸46pt、目盛30pt、凡例29ptを内側、全手法を実線・線幅4.0pt。共通色・両λで同じ軸、無題・グリッド/注釈/閾値線なし。
Phase 0の実線表記を踏襲し、追加の図修正依頼により線幅を2.8ptから4.0ptへ拡大した。
全33または4096点を使用し、平滑化/間引き/描画時の頂点省略を行わない。点間の線は観測点を結ぶもので追加評価ではない。
平均±SD帯は算術的に表示する。下端が0未満でも負の誤差観測があったことを意味しない。
凡例はデータ/SD帯との交差も検査し、目視確認後に公開する。

旧単独手法図は、新図の検証後に `figures/learning_curves/` と `figures/best_observed/` から全72枚を削除し、移動保管しない。
削除したファイルは [一覧](direct_weight_extension/deleted_single_curve_images.txt)。既存スペクトログラム・物理制御図・モデル・音声・CSV・生データは削除していない。
既存54 runの原データと全Phase0等をハッシュ保護し、新規DE/再評価Randomは同じ `runs/<次元>/<手法>/lambda0p5/seed_<n>/` へ追加。
旧CSVの既存54行の数値は維持して行/列を拡張した。[追加監査](direct_weight_extension/audit.json) は{audit['checks']:,} checks passed。
元54 runの [独立監査](audit.json) は当時の結果として保存し、今回のhash不変検査と併用している。

将来は時変制御、複数音節、未知目標、学習済み方策の再利用を公平に比較できる課題として設計する。
これらでRLがDEに勝つという結論を先取りしない。今回は一定制御から条件を広げず、追加予算・報酬再設計・RL再学習は行わない。

## 実行手順

```bash
cd /home/tatsukawa/work/Birdsong_tatsukawa
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue17-extend-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17 audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_redraw
# 拡張が未実行の既存54run版を持つ作業コピーで（同名の追加結果は上書き拒否）:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_extension prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_extension run --workers 9
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_extension audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_extension render
# 生成画像の目視確認をvisual_review.jsonへ記録した後:
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue17_extension publish
```

新しい結果のcommit/pushは今回も行わない。GitHub結果リンクはユーザーのpushまで前版を示し得る。
'''
    (stage/'REPORT.md').write_text(text)
    links=[]
    for c in CONDITIONS:
        for w,label in ((1.,'lambda1'),(.5,'lambda0p5')):
            prefix=c.replace('constant_','')+'_'+label
            links.append([c.replace('constant_',''),w,
                f'[seed0](figures/learning_curves/{prefix}_seed0.png)',f'[平均±SD](figures/learning_curves/{prefix}_mean_sd.png)',
                f'[seed0](figures/best_observed/{prefix}_seed0.png)',f'[平均±SD](figures/best_observed/{prefix}_mean_sd.png)'])
    index=['# Phase1 — 4手法×2λの比較','',
        '最初に [REPORT.md](REPORT.md)。次に下表の6条件を確認する。各画像は同じ次元・同じλの4手法を表示する。',
        '旧単独手法の誤差図は残していない。次元間・λ間を同じ画像には混在させない。','',
        '誤差線図は4手法すべて実線・線幅4.0pt。','',
        table(['次元','λ','実評価:seed0','実評価:平均±SD','J採用解:seed0','J採用解:平均±SD'],links),'',
        '実評価はRLの現方策と直接法の実候補。J採用解図はminJλで選ぶ解の共通E_totalであり、事後minE_totalではない。',
        'RLは33回の独立評価、DE/Randomは4096候補。候補数・評価数の違いはREPORTを参照。','',
        '## 数表・条件・検証','',
        '- [最終/選択解の平均±標本SD](comparison_aggregate.csv) / [各seed](comparison_per_seed.csv)',
        '- [全曲線](comparison_curves.csv) / [指定候補数](checkpoint_comparison.csv) / [λ差](lambda_effects.csv)',
        '- [実物理量と目標との差](parameter_errors.csv)',
        '- [追加条件・データ再利用](direct_weight_extension/manifest.json) / [追加実行時間](direct_weight_extension/execution.json)',
        '- [現在の追加監査](direct_weight_extension/audit.json) / [元54runの監査](audit.json) / [図の検査](figure_audit.json)',
        '- [削除した旧単独手法図72枚](direct_weight_extension/deleted_single_curve_images.txt)',
        '- [元54runの設定](config.json) / [元manifest](manifest.json) / [追加設定](direct_weight_extension/config.json)',
        '','## 音響図・物理量図（seed0、実候補4096／最終方策）','',
        'スペクトログラムは上段Target・下段Generated。既存図は変更せず保持し、新しいDE/Random λ=.5を追加した。']
    gallery=[]
    from .constant020_env import INDICES
    for c,m,w in groups:
        stem=f"{c.replace('constant_','')}_{m}_{weight_label(w)}"
        gallery.append([c.replace('constant_',''),m,w,f'[図](figures/spectrograms/{stem}_seed0.png)',
            ' / '.join(f'[{name}](figures/physical_controls/{stem}_{name}_seed0.png)' for i,name in ((0,'epsilon_l'),(2,'ps'),(3,'f0_l')) if i in INDICES[c])])
    index+=['',table(['次元','手法','λ','スペクトログラム','物理制御'],gallery),'',
        'モデル・音声・rawは runs/<次元>/<手法>/<lambda>/seed_<n>/。新直接法のbest_objectiveとbest_common_posthocを区別する。',
        'Random λ=.5のsource_reuse.jsonに候補プール・音声の共有元と新規選択音を記録した。']
    (stage/'README.md').write_text('\n'.join(index)+'\n')
    write_json(stage/'report_timing.json',dict(latest_report_update_wall_s=time.perf_counter()-start,
        images_rendered_this_update=render_images,
        comparison_render_wall_s=json.loads((stage/'figure_sources.json').read_text())['plot_wall_s'],
        comparison_images=24,new_acoustic_images=len(new_photos),existing_acoustic_images_preserved=len(old_photos)))
