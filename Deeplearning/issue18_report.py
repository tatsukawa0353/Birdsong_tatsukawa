"""Domain diagnostics and paired full/restricted results; no optimizer changes."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots

from .issue18_diagnostics import OUT,BASE
from .issue17_artifacts import read_record
from .issue17_selection import selection_curve
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue15_report import table
from .issue16 import digest
from .issue17_comparison_figures import comparison_curve,METHODS
from .acoustic_views import display_psd,relative_db,rms_envelope,spectrogram_pair,envelope_pair,control_comparison


def read_csv(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def diagnostic_figures():
    out=OUT/'figures/diagnostics';out.mkdir(parents=True,exist_ok=True)
    rows=read_csv(OUT/'diagnostics/domain_grid.csv');metadata=[]
    with np.load(OUT/'diagnostics/target.npz') as z:controls=z['controls']
    line=sorted([r for r in rows if float(r['epsilon_l'])==controls[0] and float(r['ps'])==controls[2]],key=lambda r:float(r['f0_l']))
    with plt.style.context(['science','ieee','no-latex']):
        for fields,name,ylabel in ((('E_total','E_spec','E_env'),'oracle_f0_errors','Acoustic error'),
                                  (('late_ac_rms_over_target_peak',),'oracle_f0_voicing','Late AC RMS / Target peak')):
            fig,ax=plt.subplots(figsize=(14,8),layout='constrained')
            for field in fields:ax.plot([float(r['f0_l']) for r in line],[float(r[field]) for r in line],lw=3,ls='-',label=field)
            if name=='oracle_f0_voicing':ax.axhline(.05,ls='--',lw=1.5,color='.5',label='Proxy threshold')
            ax.set_xscale('log');ax.set_xlim(49000,1e7);ax.set_xlabel(r'$f_{e,l}$ [model units]',fontsize=30);ax.set_ylabel(ylabel,fontsize=30)
            ax.tick_params(labelsize=22);ax.legend(fontsize=20);path=out/(name+'.png');fig.savefig(path,dpi=110);plt.close(fig)
            metadata.append(dict(path=str(path.relative_to(OUT)),image_sha256=digest(path),kind='physical_f0_sweep',observations=len(line)))
        for varying,fixed in (('epsilon_l','ps'),('ps','epsilon_l')):
            fixed_value=controls[2 if fixed=='ps' else 0]
            part=[r for r in rows if float(r[fixed])==fixed_value]
            x=sorted({float(r['f0_l']) for r in part});y=sorted({float(r[varying]) for r in part})
            lookup={(float(r[varying]),float(r['f0_l'])):r for r in part}
            for field in ('E_total','late_ac_rms_over_target_peak'):
                values=np.array([[float(lookup[(v,f)][field]) for f in x] for v in y])
                voiced=np.array([[float(lookup[(v,f)]['late_ac_rms_over_target_peak']) for f in x] for v in y])
                fig,ax=plt.subplots(figsize=(14,8),layout='constrained')
                mesh=ax.pcolormesh(x,y,values,shading='nearest',cmap='magma',vmin=0,vmax=max(float(r[field]) for r in rows))
                ax.contour(x,y,voiced,levels=[.05],colors='white',linewidths=1.5)
                ax.scatter([controls[3]],[controls[0 if varying=='epsilon_l' else 2]],marker='x',s=90,color='cyan')
                ax.set_xscale('log');ax.set_xlim(min(x),max(x));ax.set_ylim(min(y),max(y));ax.set_xlabel(r'$f_{e,l}$ [model units]',fontsize=30)
                ax.set_ylabel(varying+' [model units]',fontsize=30);ax.tick_params(labelsize=22)
                bar=fig.colorbar(mesh,ax=ax);bar.set_label(field,fontsize=24);bar.ax.tick_params(labelsize=20)
                path=out/f'{varying}_f0_{field}.png';fig.savefig(path,dpi=110);plt.close(fig)
                metadata.append(dict(path=str(path.relative_to(OUT)),image_sha256=digest(path),kind='physical_slice',fixed_parameter=fixed,fixed_value=float(fixed_value),
                    white_contour='lateAC RMS /target_peak=.05',cyan_cross='unchanged target controls'))
        # Sample histories are retained separately from independent policy evals.
        points=read_csv(OUT/'diagnostics/existing_history_points.csv')
        for method in ('TD3','SAC'):
            fig,axes=plt.subplots(2,1,figsize=(14,11),sharex=True,layout='constrained')
            for role,ax in zip(('training_samples','independent_policy_evaluations'),axes):
                for seed,color in zip((0,1,2),('#0072B2','#D55E00','#009E73')):
                    subset=[r for r in points if r['method']==method and float(r['envelope_weight'])==1. and int(r['seed'])==seed and r['role']==role]
                    ax.plot([int(r['trial']) for r in subset],[float(r['f0_l']) for r in subset],lw=1.8,ls='-',alpha=.8,label=f'seed {seed}',color=color)
                ax.set_ylabel(r'$f_{e,l}$ [model units]',fontsize=28);ax.tick_params(labelsize=20);ax.legend(fontsize=18)
                ax.text(.02,.05,role,transform=ax.transAxes,fontsize=17)
            axes[-1].set_xlabel('Number of trials',fontsize=30)
            path=out/f'{method}_original_training_vs_policy_f0.png';fig.savefig(path,dpi=110);plt.close(fig)
            metadata.append(dict(path=str(path.relative_to(OUT)),image_sha256=digest(path),kind='original_history',weight=1.))
        waveforms=[]
        for source in sorted((OUT/'diagnostics/representative_audio').glob('*/rollout.npz'),key=lambda p:float(p.parent.name.split('_')[1])):
            with np.load(source) as z:waveforms.append((source.parent.name,z['audio'].copy(),z['target'].copy()))
        peak=max(abs(waveforms[0][2]));bound=max(float(abs(w[1]).max()/peak) for w in waveforms)*1.1
        for mode,extent in (('full',(0,.2)),('late',(.18,.2))):
            with plt.rc_context({'path.simplify':False}):
                fig,axes=plt.subplots(len(waveforms),1,figsize=(15,17),sharex=True,sharey=True,layout='constrained')
                for ax,(label,audio,target) in zip(axes,waveforms):
                    t=np.arange(len(audio))/50000
                    ax.plot(t,target/peak,lw=1,ls='-',color='.6',alpha=.7)
                    ax.plot(t,audio/peak,lw=1,ls='-',color='#0072B2')
                    ax.set(xlim=extent,ylim=(-bound,bound));ax.tick_params(labelsize=17)
                    ax.text(.02,.86,label,transform=ax.transAxes,fontsize=17)
                axes[-1].set_xlabel('Time [s]',fontsize=30);fig.supylabel('Amplitude / Target peak',fontsize=30)
                path=out/f'representative_waveforms_{mode}.png';fig.savefig(path,dpi=110);plt.close(fig)
                metadata.append(dict(path=str(path.relative_to(OUT)),image_sha256=digest(path),kind='physical_waveforms',
                    mode=mode,normalization='One target peak for all waveforms',gray='Target',blue='Generated',decimation=False))
    return metadata


def report():
    start=time.perf_counter();decision=json.loads((OUT/'domain_decision.json').read_text());execution=json.loads((OUT/'execution.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text());rows=[];curves={};parameters=[];sounds=[];metadata=[]
    with np.load(OUT/'diagnostics/target.npz') as z:target,controls=z['target'].copy(),z['controls'].copy()
    for domain,root in (('full',BASE),('restricted',OUT)):
        for method in METHODS:
            for seed in (0,1,2):
                folder=root/'runs/constant_3d'/method/'lambda1'/f'seed_{seed}'
                summary=json.loads((folder/'summary.json').read_text());curve=selection_curve(read_record(folder/'curve.json'),1.)
                curves[(domain,method,seed)]=curve;selected=curve[curve[-1]['objective_selected_index']]
                final=curve[-1]
                row=dict(domain=domain,method=method,seed=seed,envelope_weight=1.,
                    final_E_total=final['acoustic_error'],final_E_spec=final['acoustic_components']['log_spectral'],final_E_env=final['acoustic_components']['envelope'],
                    selected_E_total=selected['acoustic_error'],selected_E_spec=selected['acoustic_components']['log_spectral'],selected_E_env=selected['acoustic_components']['envelope'],
                    final_voiced=final['sustained_voiced'],selected_voiced=selected['sustained_voiced'],selected_trial=selected['episodes'],
                    final_parameter_nrmse=final['parameter_nrmse'],selected_parameter_nrmse=selected['parameter_nrmse'],
                    training_candidates=summary['training']['episodes'],evaluation_sounds=summary['evaluation']['episodes'],
                    training_wall_s=summary['training']['wall_s'],evaluation_wall_s=summary['evaluation']['wall_s'],
                    gradient_iterations=summary['gradient_updates'],training_voiced_fraction=1-summary['training']['nonvoiced_episodes']/4096)
                rows.append(row)
                for role,p in (('final',final),('selected_best',selected)):
                    parameters.append(dict(domain=domain,method=method,seed=seed,role=role,
                        epsilon_l=p['physical_controls'][0],ps=p['physical_controls'][2],f0_l=p['physical_controls'][3],
                        epsilon_l_difference=p['physical_controls'][0]-controls[0],ps_difference=p['physical_controls'][2]-controls[2],
                        f0_l_difference=p['physical_controls'][3]-controls[3]))
                    if domain=='restricted':
                        source=folder/('best' if role=='selected_best' else '')/'rollout.npz'
                        with np.load(source) as z:audio=z['audio'].copy()
                        sounds.append(dict(method=method,seed=seed,role=role,audio=audio,source=str(source.relative_to(OUT))))
    write_csv(OUT/'comparison_per_seed.csv',rows);write_csv(OUT/'parameter_errors.csv',parameters)
    points=[dict(domain=d,method=m,seed=s,trial=p['episodes'],E_total=p['acoustic_error'],
        selected_common=p['objective_selected_common_error'],E_spec=p['acoustic_components']['log_spectral'],E_env=p['acoustic_components']['envelope'])
        for (d,m,s),curve in curves.items() for p in curve]
    write_csv(OUT/'comparison_curves.csv',points)
    fields=('final_E_total','selected_E_total','final_E_spec','final_E_env','selected_E_spec','selected_E_env',
            'final_parameter_nrmse','selected_parameter_nrmse','training_wall_s','training_voiced_fraction')
    groups=[]
    for domain in ('full','restricted'):
        for method in METHODS:
            a=[r for r in rows if r['domain']==domain and r['method']==method];group=dict(domain=domain,method=method,n=3)
            for field in fields:
                values=np.array([r[field] for r in a]);group[field+'_mean']=float(values.mean());group[field+'_sd']=float(values.std(ddof=1))
            group.update(final_voiced_count=sum(r['final_voiced'] for r in a),selected_voiced_count=sum(r['selected_voiced'] for r in a));groups.append(group)
    write_csv(OUT/'comparison_aggregate.csv',groups)
    # Common limits across the paired full/restricted plots, no interpolation.
    extrema=[0.]
    for domain in ('full','restricted'):
        for method in METHODS:
            for field in ('acoustic_error','objective_selected_common_error'):
                values=np.array([[p[field] for p in curves[(domain,method,s)]] for s in (0,1,2)])
                mean,sd=values.mean(axis=0),values.std(axis=0,ddof=1)
                extrema.extend([values.min(),values.max(),(mean-sd).min(),(mean+sd).max()])
    lo,hi=min(extrema),max(extrema);span=hi-lo;limits=(lo-.025*span,hi+.30*span)
    for domain in ('full','restricted'):
        for sub,field in (('learning_curves','acoustic_error'),('best_observed','objective_selected_common_error')):
            for average in (False,True):
                trajectories={m:(np.array([p['episodes'] for p in curves[(domain,m,0)]]),np.array([[p[field] for p in curves[(domain,m,s)]] for s in (0,1,2)])) for m in METHODS}
                path=OUT/'figures'/sub/f"3d_{domain}_lambda1_{'mean_sd' if average else 'seed0'}.png";path.parent.mkdir(parents=True,exist_ok=True)
                geometry=comparison_curve(trajectories,path,limits=limits,average=average)
                metadata.append(dict(path=str(path.relative_to(OUT)),kind='comparison_curve',domain=domain,field=field,average=average,image_sha256=digest(path),**geometry))
    for role in ('final','selected_best'):
        samples=[dict(method=p['method'],seed=p['seed'],column=p['domain'],controls=[p['epsilon_l'],controls[1],p['ps'],p['f0_l'],controls[4]]) for p in parameters if p['role']==role]
        path=OUT/'figures/physical_controls'/f'3d_lambda1_{role}.png';path.parent.mkdir(exist_ok=True)
        meta=control_comparison(samples,controls,path,columns=(('full','Full range'),('restricted','Restricted range')))
        meta.update(path=str(path.relative_to(OUT)),kind='constant_control_comparison',role=role);metadata.append(meta)
    registry=OUT/'diagnostic_figure_sources.json'
    diagnostic_metadata=json.loads(registry.read_text()) if registry.exists() else diagnostic_figures()
    for row in diagnostic_metadata:assert digest(OUT/row['path'])==row['image_sha256']
    metadata+=diagnostic_metadata
    f,t,p,reference=display_psd(target);target_db=relative_db(p,reference);target_env=rms_envelope(target,abs(target).max())
    # Include physical-survey representative sounds without conflating budgets.
    for source in sorted((OUT/'diagnostics/representative_audio').glob('*/rollout.npz')):
        with np.load(source) as z:audio=z['audio'].copy()
        sounds.append(dict(method='physical_survey',seed=None,role='diagnostic',audio=audio,source=str(source.relative_to(OUT)),name=source.parent.name))
    for r in sounds:
        _,_,power,_=display_psd(r['audio']);r['db']=relative_db(power,reference);r['env']=rms_envelope(r['audio'],abs(target).max())
    db_limits=(-70.,max(10.,float(10*np.ceil(max(r['db'].max() for r in sounds)/10))))
    rms_limits=(0.,1.3*max(float(target_env.max()),max(float(r['env'].max()) for r in sounds)))
    for r in sounds:
        stem=r.get('name',f"{r['method']}_lambda1_seed{r['seed']}")
        for kind,sub,draw in (('spectrogram','spectrograms',lambda path:spectrogram_pair((f,t,target_db),(f,t,r['db']),path,limits=db_limits)),
                              ('rms_envelope','amplitude_envelopes',lambda path:envelope_pair(target_env,r['env'],path,limits=rms_limits))):
            path=OUT/'figures'/sub/r['role']/(stem+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            meta=draw(path);meta.update(path=str(path.relative_to(OUT)),kind=kind,role=r['role'],method=r['method'],seed=r['seed'],source=r['source']);metadata.append(meta)
    write_json(OUT/'figure_sources.json',dict(images=metadata,display=dict(cmap='magma',db_limits=db_limits,reference_psd=reference,rms_ylim=rms_limits),
        no_source_data_thinning=True,renderer_sha256=digest(__file__)))
    fmt=lambda r,f:f"{r[f+'_mean']:.6g} ± {r[f+'_sd']:.3g}"
    result_table=table(['範囲','手法','Final E_total','Selected E_total','Final E_spec','Final E_env','Final発声/3','Selected発声/3','学習発声率'],
        [[r['domain'],r['method'],*[fmt(r,f) for f in ('final_E_total','selected_E_total','final_E_spec','final_E_env')],r['final_voiced_count'],r['selected_voiced_count'],fmt(r,'training_voiced_fraction')] for r in groups])
    history=read_csv(OUT/'diagnostics/existing_history_summary.csv')
    history_table=table(['手法','λ','seed','記録','観測数','発声数','minJ時点','その共通誤差','上限5%域の比率'],[
        [r['method'],r['envelope_weight'],r['seed'],r['role'],r['observations'],r['voiced_count'],r['min_J_trial'],f"{float(r['min_J_common_error']):.6g}",f"{float(r['f0_upper_boundary_fraction']):.3f}"] for r in history])
    individual=table(['範囲','手法','seed','Final','Selected','採用試行','Final発声','Selected発声'],[
        [r['domain'],r['method'],r['seed'],f"{r['final_E_total']:.7g}",f"{r['selected_E_total']:.7g}",r['selected_trial'],r['final_voiced'],r['selected_voiced']] for r in rows])
    cost_table=table(['範囲','手法','学習wall秒:平均±SD'],[[r['domain'],r['method'],fmt(r,'training_wall_s')] for r in groups])
    text=f'''# Issue18 — Case A・3Dの発声領域と制限探索の診断

既存Issue17の数値・モデル・音声を対照として読み取り専用で利用。今回の新規学習はλ=1、3D、4手法×seed0/1/2の12 runだけ。全て4096候補。目標・物理モデル・報酬・ネットワーク・optimizer・seed・CPU条件を維持し、f0_lの行動から物理値への範囲だけを変更した。λ=.5は追加せず、範囲効果と重み効果を分離する。

## 事前の物理診断と範囲の決定

Oracleのε_l/psを固定した全域f0_l sweep、下限付近の線形/対数格子、ε_l×f0_lとps×f0_lの断面、全域ε_l/psの3×3 joint anchorを計3388点で評価。新規最適化ではなく、全ての失敗・非発声点を保持した。[事前計画](diagnostics/plan.json)と[全点](diagnostics/domain_grid.csv)。

発声proxyは既存と同じ `std(最後の1000 sample)/Target peak > .05`、数値失敗なし。最後の20 msのAC標準偏差であり、生理学的有声・無声や全区間RMSと同一ではない。
観測した最大発声f0_l={decision['largest_voiced_f0']:.0f}。事前規則 `ceil(1.5*最大発声f0/1e5)*1e5` により上限={decision['restricted_f0_high']:.0f}を追試前に固定。E_totalや目標近傍の誤差を上限決定に使わず、50%の安全余裕と丸めを加えた。[決定記録](domain_decision.json)。

旧f0_l範囲[49000,10000000]→新[49000,800000]。ε_l=[2.5e7,5e7]、ps=[2.5e6,4e6]は維持。目標f0_l=300000は新範囲の約33.4%位置で、目標値中心の狭い窓ではない。反面、既知目標のε_l/psを用いる診断を含むため、未知目標への汎化を証明する設計ではない。断面と9 joint anchorの調査であり、3D全域でこれ以上発声しないことを証明していない。非一様格子の702/3388発声比率を物理体積の確率と誤記しない。

同じ正規化行動幅・TD3 noise=.1でも物理f0_lの探索幅は約13.25倍縮む。これは範囲制約の帰結で、同じ物理探索ノイズを維持した実験ではない。右側係数はCase Aの固定値のまま。f0_lは外力/開閉項でHzではない。元コードを改変せず、spawnした追試worker内だけDomainEnvを供給した。元runのsource hashと追加adapter hashを両方保存。

## 既存履歴からの説明と限界

{history_table}

学習サンプル4096点と独立方策評価33点を分けた。全域TD3の発声サンプルは0–1/4096、SACは0–9/4096で、遭遇が極めて少ない。TD3は学習サンプルの約58–63%、SACは約19–35%がf0_lの上限5%領域に入る。これらは行動分布の記述であり、学習アルゴリズム固有の因果の証明ではない。

高f0_lの非持続領域ではE_totalの下側が約0.44に留まる。Oracle断面では発声境界で損失が上昇する一方、非持続領域に入った後のE_specは約0.21で変化が小さく、E_envは上限側へ緩やかに下がる。狭い発声領域への遭遇不足と、この非持続領域内の誤差地形は境界偏りを説明する仮説に整合する。ただし発声だけで低誤差になるわけではなく、critic近似・entropy・探索機構等の寄与を単独に同定していない。報酬の再設計は行わない。

調べた格子では持続702点のE_total中央値0.336499・最小ほぼ0に対し、非持続2686点は最小0.437212・中央値0.516072だった。非持続でE_total<0.3の点は0。従って「既存報酬が発声を全く区別しない」とは結論しない。一方、持続点の最大誤差は0.727039で分布に重なりがあり、発声proxyだけでは正確な模倣を保証しない。Oracleのε_l/ps断面では非持続のf0≈500725でE_total≈0.532778、f0=1e7で≈0.450586となり、非持続領域内でも上限側へ動くと損失が下がる。[条件付き集計と実例](diagnostics/loss_proxy_summary.json)。この有限・緩勾配の領域への方策集中は仮説として整合するが、criticの内部推定が原因だと直接検証したわけではない。

## 3 seed平均±標本SDと発声状態

{result_table}

FinalはRL最終決定論的方策、直接法は実4096番目候補。SelectedはRL33独立評価／直接法4096候補からminJを最初の同値で選択した解。λ=1なのでJ=共通E_totalだが、候補集合・評価回数は異なる。直接法の到達性能を見る際はSelectedを確認し、Finalと置換しない。学習中の最良ノイズ付きサンプルを方策の到達性能と偽らない。

{individual}

E_spec/E_env、物理量NRMSE、制御値とTargetとの差は[各seed](comparison_per_seed.csv)、[平均SD](comparison_aggregate.csv)、[物理量](parameter_errors.csv)に保存。NRMSEの正規化は両条件とも元の全域物理幅を使用する。n=3の結果から一般的な有意差や優劣を主張しない。

## 計算量と検証

新規学習{execution['training_candidates']:,}候補、native98,304,000,000 step。{execution['workers']} CPU workers、並列wall{execution['wall_s']:.2f}秒。独立訓練時評価{execution['evaluation_sounds']}音。RL各4080 gradient iterations、TD3 actor/critic、SAC actor/critic/entropyの更新数を各summaryに分けて保存。

{cost_table}

元全域実験は別の27 worker条件なので、wall差をアルゴリズム固有の速度差としない。物理診断3388音＋代表音7、範囲マッピングテスト4音、追加監査{audit['audit_generated_sounds']}音は4096予算から別計上。元実験の学習は再実行しない。

監査{audit['checks']:,}件を通過。元RLとの初期ネットワーク・16 warmup正規化行動、effective hyperparametersが一致。Randomの4096正規化候補列、DEの初期8正規化候補も対照と同一。物理f0_lの写像だけが異なる。新モデルは別プロセスで{audit['model_reloads']}回再読み込みし、Final/Best計24音の行動・制御・波形・評価差0を確認。loss/gradient/parameterのNaN/Infで停止する既存検査を維持し、全runが完了した。

## 図の読み方

各誤差図は同一範囲・λ=1の4手法を重ねる。全域と制限域は別画像で共通軸。実線4pt、SciencePlots、軸46pt・目盛30pt・凡例29pt。RL33点と直接法4096点を全て描き、平滑化・頂点省略なし。帯は標本SD。
音響図はmagma、Target最大PSDの共通基準、範囲{db_limits} dB、共通カラーバー。5 ms RMSは双方を同じTargetピークで割り、縦軸{rms_limits}。表示は10 MHzへupsampleしたdensity PSDで、学習の元50 kHz・paddingありspectrum PSDとは別。音圧校正dB SPLではなく、upsampleで情報は増えない。
表示PSDは従来同様にフレーム平均を除去するため、DC/緩い過渡成分は暗くてもRMSに残り得る。RMSとAC発声proxyを区別し、代表波形の全区間と終盤20msを共通振幅で併記した。
パラメータ図はFinal/Selected別、行が係数・列が全域/制限域。λは両列とも1であり、重み比較を装っていない。各seed・平均±標本SD・Target破線を表示する。[図の全索引](FIGURES.md)。

## 今回の結果から言える範囲

SACのFinal発声は0/3→3/3、TD3は0/3→2/3になった。Random Searchの経験的発声率は約2.34%→31.6%で、制限範囲が発声点への遭遇頻度を増やしたことを支持する。この一様乱数探索の実サンプル率と、非一様診断格子の702/3388は区別する。

SACの平均Final E_envは約0.234→0.0127まで低下した一方、E_specは約0.208→0.241へ増加した。発声・振幅包絡の改善と、スペクトルの正確な模倣は同じではない。TD3 seed1は制限後も非持続で、Final E_totalは約0.439→0.521へ悪化している。平均改善だけで全seedの解決としない。

DE/RandomのSelected E_totalも約0.1766→0.02162 / 0.1666→0.04722へ低下した。改善はRL固有ではなく、同じ領域制約が直接法にも有効だった。制限後もDEが最も低い平均誤差で、DEのうち2 seedには約0.03の局所解が残る。制御値の同定と音響誤差の低さも別に判断する。

次の切り分けでは、まず保存済みcriticの予測と保存実測Jの対応を再学習なしで調べることを候補とする。その後、同じ制限範囲・目標・λ=1・seed・4096・ネットワーク・学習率を固定し、探索幅など1要因だけを変える追試を計画する。今回はその追加学習や報酬変更を行っていない。

## 判断と次の課題

制限後の改善は探索制約の効果として扱い、元の広域条件からRL自体の優位を結論しない。改善/非改善・非持続seedも全て表へ含めた。制限しても誤差が残る場合、次はまず探索機構または物理パラメータの尺度を1要因ずつ切り分ける計画を立てる。発声proxyを報酬へ追加する案は別実験であり、この診断で無断導入しない。未知目標・時変制御・複数音節・方策再利用を公平に比較するための準備とし、そこでRLがDEに勝つという結論は先取りしない。

## 再実行

リポジトリ直下、既存`.venv-rl`を使用。新規同名結果は上書き拒否する。
```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-newissues-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_diagnostics survey --workers 9
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_diagnostics reanalyze
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_training prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_training train --workers 12
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue18_report
```
既存結果がある作業コピーではaudit/reportのみ再実行できる。学習再現は空の同名出力を使う別作業コピーで行い、既存正本を削除しない。commit/pushは行っていない。
'''
    interruption=OUT/'interrupted_attempt_1/interruption.json'
    accounting=dict(valid_comparison_training_candidates=execution['training_candidates'],
        valid_training_evaluation_sounds=execution['evaluation_sounds'],physical_survey_sounds=3388,
        physical_representative_sounds=7,range_test_sounds=4,audit_sounds=audit['audit_generated_sounds'])
    if interruption.exists():
        stopped=json.loads(interruption.read_text())
        accounting.update(interrupted_training_candidates_lower_bound=stopped['training_candidates_lower_bound'],
            interrupted_policy_evaluations_lower_bound=156,interrupted_exact_final_count_available=False)
        text+='\n\n## 実行セッション中断と再実行の計上\n\n最初の12 runは実行セッションがSIGTERM（終了143）で終了し、全runの最後の進捗記録は3200候補だった。終了の外的な正確な原因は未確認。損失・NaN/Inf例外として扱わず、完了4096の比較へ混ぜない。少なくとも38,400学習候補と156方策評価が実行され、最後の記録以後の厳密候補数は復元できない。\n\n中断checkpointには完全なreplay buffer/RNG状態がないため、同じseed・条件で12 run全てを最初から実行した。中断時の全記録は [interrupted_attempt_1](interrupted_attempt_1/interruption.json) に保持し、再開を装わない。再実行の48 checkpointで行動・物理値・誤差が中断試行と完全一致することも監査した。本文の49,152候補・wall時間は完了した比較用試行のみで、中断分の計算費用を除外して速さを主張しない。[全計上](total_accounting.json)。\n'
    write_json(OUT/'total_accounting.json',accounting)
    (OUT/'REPORT.md').write_text(text)
    image_rows=[[r['kind'],r.get('domain',r.get('role','diagnostic')),f"[図]({r['path']})"] for r in metadata]
    (OUT/'FIGURES.md').write_text('# 図一覧\n\n'+table(['種別','条件/出所','画像'],image_rows)+'\n')
    (OUT/'README.md').write_text('# Issue18 — 3D発声領域の診断\n\n[REPORT](REPORT.md) → [図一覧](FIGURES.md)。λ=1のみの範囲追試。既存Issue17は読み取り専用。\n\n'
        '- [事前の診断計画](diagnostics/plan.json) / [範囲決定](domain_decision.json) / [manifest](manifest.json)\n'
        '- [Final/Selectedの各seed](comparison_per_seed.csv) / [平均±標本SD](comparison_aggregate.csv) / [制御値と目標差](parameter_errors.csv)\n'
        '- [元の学習サンプルと方策評価](diagnostics/existing_history_summary.csv) / [発声・誤差地形](diagnostics/domain_grid.csv)\n'
        '- [時間と回数](execution.json) / [独立再読み込み監査](audit.json) / [描画条件と出所](figure_sources.json)\n'
        '- [元の全域3Dを含むIssue17](../parameter_dimension/README.md)\n')
    write_json(OUT/'report_timing.json',dict(wall_s=time.perf_counter()-start,images=len(metadata)))
    print('Issue18 report and figures complete',flush=True)


if __name__=='__main__':report()
