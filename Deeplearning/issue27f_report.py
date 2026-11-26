"""Report and final preservation audit from completed saved results; native0."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile
from .issue27f_data import ROOT,OUT,OLD,TARGET,METHODS,SEEDS,folder,ready,hashes
from .issue27f_analysis import read,load,records,summaries,ROLES
from .issue27f_native import PROBE
from .issue19_protection import workspace_inventory
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest
from .issue23_stage_a import table as plain_table

def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def table(headers,data):return plain_table(headers,[[str(x).replace('|','\\|') for x in r] for r in data])
def number(x):return f'{float(x):.6f}'
def figure(name):return f'![{name}](figures/{name}.png)\n'

def report():
    manifest=ready();s=summaries();assert all(x['status']=='complete' for x in s)
    assert read(OUT/'audit/endpoints.json')['passed'] and read(OUT/'physics_audit.json')['passed']
    choice=read(OUT/'preflight/target_choice.json');f=choice['physical_features'];ep=rows('endpoints.csv');means=rows('method_summary.csv');accounts=rows('training_accounting.csv');exe=read(OUT/'execution.json')
    seedtable=table(['手法','seed','役割','試行','E_spec','E_env','E_total','J','持続proxy','ε_l','p_s','f0_l'],
        [[r['method'],r['seed'],r['role'],r['trial'],*[number(r[k]) for k in ('E_spec','E_env','E_total','J')],r['voiced'],*[f'{float(r[k]):.9g}' for k in ('epsilon_l','ps','f0_l')]] for r in ep])
    meantable=table(['手法','役割','E_spec 平均±SD','E_env 平均±SD','E_total 平均±SD','J 平均±SD'],
        [[r['method'],r['role'],*[f"{number(r[k+'_mean'])} ± {number(r[k+'_sample_sd'])}" for k in ('E_spec','E_env','E_total','J')]] for r in means])
    grid=table(['候補','ε_l','p_s','f0_r','末尾AC/自身peak','直線除去後','entropy','peak数','左/右変動proxy','適格'],
        [[r['candidate'],r['epsilon_l'],r['ps'],r['f0_r'],number(r['late_ac_ratio']),number(r['detrended_ac_ratio']),number(r['spectral_entropy']),r['spectral_peak_count'],f"{r['left_motion_periodic_proxy']}/{r['right_motion_periodic_proxy']}",r['eligible']] for r in records()])
    damping=rows('old_damping_readonly.csv')+rows('damping_diagnostics.csv')
    dt=table(['音','側','末尾x平均','末尾x std','c x²平均','速度係数平均','速度係数<0の割合'],
        [[r['name'],r['side'],*[f'{float(r[k]):.8g}' for k in ('x_tail_mean','x_tail_std','nonlinear_damping_mean','effective_velocity_coefficient_mean','negative_fraction')]] for r in damping if r['name'] in ('old_c00','b2_c00','b3_c00')])
    runtimes=table(['手法','seed','学習step','独立評価','学習wall秒','評価wall秒','process CPU秒','非持続学習候補','数値失敗'],
        [[r['method'],r['seed'],r['training_steps'],r['evaluations'],number(r['training_wall_s']),number(r['evaluation_wall_s']),number(r['process_cpu_s']),r['nonvoiced'],r['failures']] for r in accounts])
    geometry=table(['手法','Actor/方策','Critic/価値','target Actor','target twin Critic','policy合計','更新回数/run'],
        [[x['method'],x['network_geometry']['actor_parameters'],x['network_geometry'].get('critic_parameters',x['network_geometry'].get('value_parameters')),x['network_geometry'].get('actor_target_parameters',0),x['network_geometry'].get('critic_target_parameters',0),x['network_geometry']['total_policy_parameters'],x['optimizer_steps']] for x in s if x['seed']==0])
    # Descriptive acquisition/retention, using each run's own initial error.
    from .issue17_artifacts import read_record
    acquisition=[];exploration=[]
    for x in s:
        c=read_record(folder(x['method'],x['seed'])/'curve.json');initial=c[0]['acoustic_error'];hit=next((r['episodes'] for r in c if r['acoustic_error']<=initial/2),None)
        acquisition.append(dict(method=x['method'],seed=x['seed'],initial_E=initial,first_half_initial_E_trial=hit,Final_E=x['final']['acoustic_error'],Selected_E=x['selected']['acoustic_error'],
            Final_minus_Selected_E=x['final']['acoustic_error']-x['selected']['acoustic_error'],training_best_E=x['training_best']['acoustic_error'],Final_minus_training_best_E=x['final']['acoustic_error']-x['training_best']['acoustic_error']))
        trace=read_record(folder(x['method'],x['seed'])/'training_trace.json');control=np.array([r['physical_controls'] for r in trace]);action=np.array([r['action'] for r in trace])
        assert np.all(control[:,1]==3.5e7) and np.all(control[:,4]==49000.) and np.all(np.ptp(control[:,[0,2,3]],axis=0)>0)
        exploration.append(dict(method=x['method'],seed=x['seed'],voiced_training_fraction=float(np.mean([r['sustained_voiced'] for r in trace])),
            epsilon_min=float(control[:,0].min()),epsilon_max=float(control[:,0].max()),ps_min=float(control[:,2].min()),ps_max=float(control[:,2].max()),f0_l_min=float(control[:,3].min()),f0_l_max=float(control[:,3].max()),
            f0_l_le100000_fraction=float(np.mean(control[:,3]<=100000)),f0_l_equal49000_fraction=float(np.mean(control[:,3]==49000)),
            executed_any_boundary_fraction=float(np.mean(np.any(abs(action)==1,axis=1))),PPO_clipped_action_fraction=float(np.mean([r.get('action_was_clipped',False) for r in trace])) if x['method']=='PPO' else None,
            initial_policy_std=c[0]['policy_statistics']['normalized_action_std_mc'],final_policy_std=c[-1]['policy_statistics']['normalized_action_std_mc'],
            initial_alpha=c[0]['policy_statistics'].get('alpha'),final_alpha=c[-1]['policy_statistics'].get('alpha')))
    write_csv(OUT/'acquisition_retention.csv',acquisition)
    write_csv(OUT/'exploration_summary.csv',exploration)
    exploration_table=table(['手法','seed','学習持続率','左f0の実最小–最大','左f0≤100000の率','左f0=49000の率','実行行動の境界率'],[[r['method'],r['seed'],number(r['voiced_training_fraction']),f"{r['f0_l_min']:.2f}–{r['f0_l_max']:.2f}",number(r['f0_l_le100000_fraction']),number(r['f0_l_equal49000_fraction']),number(r['executed_any_boundary_fraction'])] for r in exploration])
    at=table(['手法','seed','初期E','初期E半減の初回評価step','Final−Selected E','Final−training_best E'],[[r['method'],r['seed'],number(r['initial_E']),r['first_half_initial_E_trial'],number(r['Final_minus_Selected_E']),number(r['Final_minus_training_best_E'])] for r in acquisition])
    finals=[r for r in means if r['role']=='final'];ranking='、'.join(f"{r['method']} {float(r['E_total_mean']):.6f} ± {float(r['E_total_sample_sd']):.6f}" for r in sorted(finals,key=lambda r:float(r['E_total_mean'])))
    text=f'''# Issue27再改訂 — Case B左右外力18点・SAC/TD3/PPO

2026-11-26 JST。[今回参照したIssue本文](issue_snapshot.json)（[GitHub #27](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/27)）は左右外力18点へ再改訂されている。前の左49000のみ9点の依頼に続く「再度取り組む」をこの最新版に従って実施した。旧①9音は読み取り専用、新規②③各9音を計算し、**18/18が持続proxyを満たした。Targetを学習前に確定し、3手法×3seedの9runを各4096step、計36864学習候補で完了した。** 失敗runの差し替えや追加探索はない。

Targetは **b3_c00、[ε_l, ε_r, p_s, f0_l, f0_r]=[3.0e7,3.5e7,3.0e6,49000,49000]**。独立B oracleは音声・20block状態・native運動mean/stdが完全一致し、E_spec/E_env/E_total/Jは全て0。実学習Finalの3seed平均±標本SDは **{ranking}**。これは今回の固定Targetと予算での観測であり、一般的優劣やCase A/Bの構造優位を示すものではない。

[Target WAV](audio/Target.wav) / [生音・状態NPZ](preflight/target.npz) / [選定理由](preflight/target_choice.json) / [事前計画](PLAN.md) / [manifest](manifest.json)。

{figure('Target_spectrogram')}
{figure('Target_waveform')}

## Targetの選定と物理モデル

{choice['reason']} 持続proxy={f['late_ac_ratio']:.10f}、直線除去後={f['detrended_ac_ratio']:.10f}。entropy={f['spectral_entropy']:.6f}、peak数={f['spectral_peak_count']}。entropyは末尾0.10sのWelch PSD（100–10000Hz）における正規化Shannon entropy、peak数はコード記載のprominence/強度閾値による記述量で、生理学的複雑性指標とは区別する。候補間の物理表示を確認後、方法別smokeと正式学習の前に選定した。選定後の差し替えはない。

元native/model.cppとmodel.py、既存libbirdsong.soは変更していない。[同一ソース再ビルドのbyte/hash一致](native/build.json)。Case B native condition=1、左右α=1.2e6/β=1500、右ε=3.5e7、x0=.02cm/D=.01、dt=1e-7を実メモリで検査した。元BirdModel('B')のPython制御処理は右f0を1e7へ固定するため、独立ConfiguredCaseB subclassの_controlsのみでIssue指定の右固定値を設定した。元Bのコンストラクタ・native reset/式/固定係数はそのままで、Aへの置換や右αβのゼロ化はない。[初回REPORT](../initial_stop/REPORT.md)と[前回9点REPORT](../high_left_force/REPORT.md)の停止理由と次案を確認して着手した。

各音0.20s、2,000,000 native step、50kHz/10000sample。初期4状態、遅延履歴、16次15kHz lowpass状態を0にし、200分の1にデシメートする。左右f0は外力係数であり音響基本周波数Hzではない。学習では右f0=49000を全run固定する一方、左f0は[49000,800000]を探索させた。

## Stage A：18点と前回9点

①左300000/右1e7（前回保存9音）、②左49000/右1e7（新9音）、③左右49000（新9音）。ε_l×p_sの9点は同一。数値失敗0、generated_silent0、②③とも適格9/9。

{grid}

一次持続proxyは末尾1000sample（20ms）のAC std/自身全期間peak>.05。直線除去後ratio>.05も学習前のドリフト除外確認とした。左右変動proxyは末尾traceの線形傾向除去後std/元std>.5かつstd/全期間state幅>.001をx/y双方で確認した記述的判定であり、周期軌道の数学的証明ではない。末尾native全刻み統計は10ms・100000刻み、traceによる指標は20ms・50kHzで、窓と統計を混同しない。

[全18点指標](candidate_summary.csv) / [末尾左右運動](tail_motion.csv) / [左右音響寄与](source_statistics.csv) / [全時系列CSV.gz](timeseries/) / [生波形・全刻みblock統計](preflight/candidates/) / [全WAV索引](audio_index.csv)。

{figure('group2_spectrograms')}
{figure('group3_spectrograms')}
{figure('group2_waveforms')}
{figure('group3_waveforms')}

②③のPSDは共通Targetの全周波数PSD最大を基準とし同一色軸。①②③の次図は**前回のPSD基準と色軸[-80,40]dBをそのまま保持**し、27音の時間・周波数軸も統一した。新しい強い成分の飽和率は[表示設定](display_settings.json)に保存し、同じ旧PSD基準で上限だけ27音共通にした[補助図](figures/three_groups_joint_color.png)も用意した。個別正規化はしていない。STFTは従来通り10MHzへ表示用resample、Blackman-Harris窓245760/noverlap184320、PSD density、detrend constant、complete windowのみ、0–10kHz。報酬用50kHz STFTと表示用STFTは別処理である。

{figure('three_groups_previous_color')}

## 左右振動と非線形減衰の変化

同じε_l=3e7、p_s=3e6で①→②→③を比較する。元の式をA=1で整理した瞬時速度係数は k_y=b+c x²−D(p_s−π̃)、b=2e4、c=2e8、D=.01。以下は末尾20msの50kHz traceによる値である。

{dt}

①は左右ともドリフト主体で、左の|x|とc x²が大きく、k_yは全サンプルで正だった。①→②では左f0を下げたことで左の変位範囲・速度が変わり、振動中にk_y<0となる区間が現れ、左の持続変動と音圧proxy増大が観測された。右は外力1e7による大きい負変位と強いc x²を保ち、周期的変動proxyは不合格でも右source/outputは非ゼロで圧力結合へ寄与している。

②→③では右外力も下げると右の平均変位の絶対値と非線形減衰が大幅に小さくなり、左右とも持続変動を示した。選定Targetでは左右のεが異なり、複数の周波数成分を持つ。正の平均k_yだけから非発声とは判定できず、振動中の符号変化・連成・遅延と一緒に見る必要がある。この整理は計算された0.20s内の挙動の解釈で、遅延連成全系の安定性解析や全領域の発声保証ではない。

{figure('force_damping_comparison')}
{figure('group2_motion_tail')}
{figure('group3_motion_tail')}

[②全時間の左右状態](figures/group2_motion_full.png) / [③全時間の左右状態](figures/group3_motion_full.png) / [②左右source/output全時間](figures/group2_sources_full.png) / [③左右source/output全時間](figures/group3_sources_full.png) / [②末尾寄与](figures/group2_sources_tail.png) / [③末尾寄与](figures/group3_sources_tail.png)。source_j=α_j(x_j−τ_j y_j)+β_j y_j、output_j=source_j−β_j τ_j dy_j/dt、feedback=−γπ(t−T)。output_l+output_r+feedback=未フィルタ音圧を丸め誤差範囲で検査した。これは連成した同じ状態での代数的分解で、単独発声させた音や因果的寄与率ではない。右の受動応答・ドリフトも保存しclampしていない。

## 学習の凍結条件と事前検査

一定3D：[ε_l,p_s,f0_l]を[2.5e7–5e7,2.5e6–4e6,49000–800000]で探索。1episode=1action=0.20s、観測8D/reset[-1,0,…]。報酬−J、J=.75E_spec+.25E_env、共通E_total=E_spec+E_env。Target共通peakで評価し、ゲイン・位相調整はしない。学習時の持続proxyはTarget peak基準、Stage Aは各候補自身peak基準である。

SAC/TD3は128×128 ReLU、PPOは方策/価値別128×128 Tanh。全てCPU1thread、LR3e-4、gamma1、batch32。SAC/TD3はbuffer50000/warmup16/train_freq1/gradient_steps1/tau.005。SACはauto_0.01/target_entropy−3。TD3は探索Normal(0,.1)、policy_delay2、target noise.2/clip.5。PPOはn_steps128/n_epochs10/clip.2/gae.95/vf_coef.5/ent_coef0/max_grad_norm.5/normalize_advantage=True/target_kl=None。Adam epsはSAC/TD3=1e-8、PPO=1e-5。全実効設定・バージョンは各run configとmanifestに保存した。

{geometry}

SACの温度1parameterはpolicy合計の外。隠れ層幅が同じでもパラメータ数・活性化・更新量・計算時間は異なる。PPOは32rollout×4minibatch×10epoch=1280 optimizer step、内部_n_updatesはepoch数320。SAC actor/critic/entropy各4080、TD3 critic4080/actor2040を実測した。実層構造と各parameter数をassertした。

[事前検査](preflight/tests.json)：標準SB3と監査付き実装を3手法×128stepで比較し、重み・optimizer・Buffer・RNGが完全一致。9初期seedの独立性・構造・制御マッピングも検査。物理smokeは各128学習音+2評価+4保存結果再生、計402音。本学習へ混入しない。最初のsynthetic検査はTD3のfloat32逆変換の1ULP差を監査側が誤検出して停止した。元のaction/Replay値は変えず、標準の順方向unscale一致を検査するよう観測コードを修正し、全照合を再実施した。[最初の停止記録](preflight/synthetic_precision_failure.json)。syntheticは不成功検査・再現確認も含め計1282episodeでnative音生成0。SAC/TD3は実行action・Replay・reward・終端を、PPOは元Gaussian raw行動/rolloutと環境clip行動を区別して全36864行を検査した。

全optimizer更新で実効LR、有限性、勾配/parameter/Adam momentの既存数値停止条件を検査。PPOの標準grad clip.5以外に学習を変えるclampは追加していない。独立評価前後の重み・optimizer・Buffer・RNG・training_bestのsignature不変を297回確認した。PPOも128stepの全更新後に評価し、最終評価はlearn終了後である。

## 学習曲線と評価結果

{figure('learning_curves')}

Final=4096step終了後の決定論的方策。Selected=33点中最初のminJ方策、posthoc=同33点中最初のminE_total方策、training_best=4096学習候補中最初のminJ行動で、保存Actorとは区別する。各役割を独立に36音再生し、波形・状態・制御・音響誤差の一致を確認した。[独立監査](audit/endpoints.json)。

{meantable}

3seed、標本SDはddof=1。[全役割の平均・SD（制御値と持続率を含む）](method_summary.csv)。

SACとTD3のFinal平均差は約0.00151で、各手法のseed標本SD（約0.055、0.048）より小さい。この3seedから両者の明確な優劣は断定できない。PPOのFinalではE_env=0.283650が大きく、SAC/TD3の約0.098/0.100との違いがE_total差の主成分だった。全36役割音の音圧持続proxyは真だが、左右の運動状態は別にendpoint_motion/sourceへ記録した。持続することとTargetを精密に模倣することは別の評価である。

PPOは3seedともFinalの実行行動が[-1,1,1]、物理値[ε_l,p_s,f0_l]=[2.5e7,4e6,8e5]になり、同一の音・誤差になった。3seedの最終重みhashは異なり、同じ学習結果を複製したものではない。方策Gaussian平均は各seedで[-1,1]の範囲外へ移り、決定論的実行時のclipにより同じ境界点へ写った。これは学習を停止する数値異常ではなく、今回の標準PPOで得られた方策の挙動である。

PPOのtraining_best平均E≈0.10778に対しFinal≈0.40382で、良い探索候補が方策Finalへ保持されたとは言えない。SACもtraining_best≈0.10787とFinal≈0.20496に隔たりがあった。PPO seed1/2ではposthocのminEが初期step0、minJのSelectedはそれぞれ640/768だった。J=.75E_spec+.25E_envの改善と、E_total=E_spec+E_envの改善が同義ではないこともこの結果に現れている。

{seedtable}

[全seedの評価CSV](endpoints.csv) / [33点学習曲線CSV](learning_curves.csv) / [学習ログ・3役割モデル・生音](runs/) / [左右内部状態・寄与](endpoint_motion.csv) / [音響項](endpoint_sources.csv)。保存.ptは方策・価値・targetネットとSAC温度の推論用重みで、optimizer/Replayを含む学習再開checkpointではない。Final/Selected/posthocの27モデルとtraining_bestの9行動を保存している。全36864学習候補の行動・物理値・音響指標はtraining_trace.json.gzへ保存した。各候補のWAVや全時刻重みを全て保存したものではなく、WAV/生音は18候補、Target、4役割×9runを保存している。

## 獲得・保持と探索候補の区別

{at}

初期Eの半減時点は33評価点のうち最初の観測点であり、連続時間の到達時刻ではない。SelectedとFinalの差は途中の方策獲得を最終時点で保持したかを見る記述値。training_bestとのギャップは探索で見つけた行動とActorへの取り込みを区別するための値で、異なる目的J/minEを同じ選択と見なさない。E_total≤.10による合否・再試行・追加予算は用いていない。

{exploration_table}

左f0は学習中に固定されず、3制御全ての実行範囲が非ゼロであることを各runで検査した。[探索の実範囲・行動分散・SAC温度](exploration_summary.csv)。Targetの左f0は探索下限にあり、SACのtanh方策、TD3のclip付き探索、PPOのGaussianと環境clipでは境界へ到達する性質が異なる。上表は実行された行動の観測であり、この差だけで性能差の因果を決めるものではない。PPOのraw行動をclip済み行動に置き換えて学習させる変更はしていない。

## Target最上段の音・内部状態比較

全図の列はseed0/1/2、上段Target、その下にSAC/TD3/PPO。4役割で同じ時間・周波数・Target PSD基準・色軸を使用。波形/RMS/WAVは全て同じTarget peakの逆数によるgainで、個別正規化・クリップ・位相合わせはない。float32 WAVには1を超える値もそのまま保持し、聴取環境によるclipと保存波形の一致を混同しない。主観的な聴取評価は実施していない。

代表seed1のFinal音：[SAC](audio/SAC_s1_final.wav) / [TD3](audio/TD3_s1_final.wav) / [PPO](audio/PPO_s1_final.wav)。Selected音：[SAC](audio/SAC_s1_selected.wav) / [TD3](audio/TD3_s1_selected.wav) / [PPO](audio/PPO_s1_selected.wav)。[全55 WAVと対応生音の索引](README.md)に全seed・全役割を掲載した。

{figure('final_spectrograms')}
{figure('selected_spectrograms')}
{figure('final_waveforms')}
{figure('selected_waveforms')}

[training_best PSD](figures/training_best_spectrograms.png) / [training_best 波形](figures/training_best_waveforms.png) / [posthoc PSD](figures/posthoc_spectrograms.png) / [posthoc 波形](figures/posthoc_waveforms.png)。

{figure('control_parameters')}
{figure('final_internal_seed1')}
{figure('selected_internal_seed1')}

内部状態代表図は事前固定seed1を用い、良いseedの事後選択はしない。全seed全役割の詳細traceは[audit NPZ](audit/)と[timeseries CSV.gz](timeseries/)、末尾統計は[endpoint_motion](endpoint_motion.csv)と[endpoint_sources](endpoint_sources.csv)。似た音響誤差が得られても、制御値の一意復元や唯一の振動状態を主張しない。

## 実行会計・時間

{runtimes}

9run並列実行の全体wall={exe['wall_s']:.3f}秒。個別training wallはそのrunのlearn区間から独立評価・保存時間を差し引いた値、CPUはrun process全体であり一致する量ではない。

| 区分 | 実際の音生成回数 | native step |
|---|---:|---:|
| 新規Stage A候補 | 18 | 36000000 |
| 選定後の独立B oracle | 1 | 2000000 |
| 方法別smoke学習 | 384 | 768000000 |
| smoke独立評価 | 6 | 12000000 |
| smoke保存モデル/候補再生 | 12 | 24000000 |
| 正式学習 | 36864 | 73728000000 |
| 正式独立方策評価 | 297 | 594000000 |
| 終了後の独立詳細監査 | 36 | 72000000 |
| 合計 | **37618** | **75236000000** |
| 旧①9点の再生成 / 新規A音 / synthetic音 | 0 / 0 / 0 | 0 |

全新規音のnative積分時間合計7523.6秒。各計算は2,000,000stepで、実時間のwallとは異なる。追加探索・最適化・時間延長・failed run差し替え0。[正式会計](execution.json) / [smoke会計](preflight/smoke_audit.json) / [Stage A ledger](preflight/ledger.json) / [終了後ledger](audit/ledger.json)。

## 保存・監査と限界

旧Issue17〜27の6362保護対象ファイルを開始時にhash化し、終了時に全件不変・削除0を検査する。[旧成果照合](integrity.json)。初回非持続3音と前回非持続9音・0runの報告はそのまま残した。新規実装、PLAN、Target、実library、ソース、全モデル/ログ/CSV/WAV/図のhashを保存する。開始時inventoryには前の中断で作られた2つの新規agentファイルがあり、clean-startへの補正根拠を[storage_before](storage_before.json)に明記した。

今回の目的は**Case B内部・同じ自己生成Targetでの3手法比較**。Case Aは異なるTargetのためA/B誤差から2音源構造の優位を結論しない。3seed・単一Target・固定予算での結果に限られる。TQC、DE/GA/CMA-ES、5D、時変制御、共通実録音比較、追加grid、長時間発声、全遅延系の安定性解析は未実施。commit/push/Issue更新も行っていない。

[再現手順](REPRODUCE.md) / [実装hash](implementation_checksums.json) / [成果物hash](artifact_checksums.json) / [保存容量](storage_after.json)。
'''
    (OUT/'REPORT.md').write_text(text)
    audio='\n'.join(f"| {r['name']} | [WAV]({r['wav']}) | [生音]({r['raw_npz']}) |" for r in rows('audio_index.csv'))
    (OUT/'README.md').write_text('# Issue27再改訂の成果索引\n\n[REPORT](REPORT.md) / [PLAN](PLAN.md) / [manifest](manifest.json) / [再現](REPRODUCE.md) / [監査](integrity.json)。\n\n18点適格、b3_c00 Target、SAC/TD3/PPO×3seed×4096stepを完了。旧成果は保持。\n\n|音|WAV|生音・状態|\n|---|---|---|\n'+audio+'\n')
    (OUT/'REPRODUCE.md').write_text('''# 再現・検査

リポジトリ親から同じ .venv-rl を使う。既存成果先は削除・上書きしない。新規計算を再実行する場合は別の成果先と事前計画が必要。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27f-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27f_report audit
```

これは保存済みデータ・モデル・ソース・WAV・旧成果hashの検査のみで新規音0。事前smokeと36独立詳細再生は既に実行済み。test_issue27fやanalysis endpointsの再実行は新しい音を発生させるため、保存済みのledgerがあれば拒否する。

実行した順序：issue27f_prepare scan → 物理表示の確認 → select --candidate b3_c00（理由指定） → test_issue27f（標準実装照合と3手法128step） → prepare freeze → training --workers9 → analysis physics / figures physics → analysis endpoints（36音） → figures learning → report report / freeze / audit。最初のsynthetic逆変換検査停止と修正後の検査を別ログで保持した。

manifestのcore hashesは学習前に凍結。implementation_checksumsは終了後の分析/描画/報告を含む。artifact_checksumsは自分自身とstorage_afterを除く全成果。推論.ptにはoptimizerやBufferがないため学習再開とは区別する。旧①9音の比較は保存NPZ読込のみ。
''')
    # Keep full stdout/failure evidence, outside-run logs never alter frozen core.
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for name in ('tests','tests-fixed','training','scan'):
        p=Path(f'/tmp/birdsong-issue27f-{name}.log')
        if p.exists():shutil.copyfile(p,logs/p.name)
    from .issue27f_data import CORE
    sources=OUT/'executed_sources';sources.mkdir(exist_ok=True)
    copied={}
    for rel in sorted(set(CORE)|{str(p.relative_to(ROOT)) for p in ROOT.glob('issue27f*.py')}|{'tests/test_issue27f.py'}):
        dest=sources/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(resolve_result_path(ROOT/rel),dest);copied[rel]=digest(dest)
    write_json(OUT/'executed_source_checksums.json',copied)
    # Explicit full-block statistics for candidates and all independently audited endpoints.
    blocks=[]
    from .issue27f_native import CHANNELS
    paths=[(r['candidate'],OUT/r['raw_npz']) for r in records()]+[(p.stem,p) for p in sorted((OUT/'audit').glob('*.npz'))]
    for name,path in paths:
        for i,stat in enumerate(load(path)['diagnostic_moments']):
            for j,ch in enumerate(CHANNELS):blocks.append(dict(name=name,block=i+1,start_s=i*.01,end_s=(i+1)*.01,channel=ch,mean=stat[j],std=stat[12+j],minimum=stat[24+j],maximum=stat[36+j]))
    write_csv(OUT/'native_block_statistics.csv',blocks)
    write_json(OUT/'accounting_complete.json',dict(passed=True,native_sounds=37618,native_steps=75236000000,StageA=19,smoke=402,training=36864,independent_evaluations=297,post_training_audit=36,synthetic_episodes=1282,synthetic_native_sounds=0,
        old_sounds_regenerated=0,research_runs=9,failed_runs=0,retries=0,additional_grid=0))

def audit():
    manifest=ready();assert source_hashes_match(read(OUT/'preflight/tests.json')['source_sha256'], hashes())
    if not (OUT/'integrity.json').exists():write_json(OUT/'integrity.json',dict(status='running'))
    protected=rows('protected_before.csv');changed=[];deleted=[];broken=[];links=0
    for r in protected:
        p=resolve_result_path(ROOT/r['path'])
        if not p.exists():deleted.append(r['path'])
        elif digest(p)!=r['sha256']:changed.append(r['path'])
    for path in (OUT/'REPORT.md',OUT/'README.md',OUT/'PLAN.md',OUT/'REPRODUCE.md'):
        for dest in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            links+=1
            target=path.parent/dest.split('#')[0]
            if not target.exists() and target!=OUT/'integrity.json':broken.append(dict(file=path.name,link=dest))
    for name,root in (('implementation_checksums.json',ROOT),('artifact_checksums.json',OUT)):
        if (OUT/name).exists():
            for path,h in read(OUT/name).items():assert digest(root/path)==h,path
    for path,h in read(OUT/'executed_source_checksums.json').items():assert digest(OUT/'executed_sources'/path)==h and digest(resolve_result_path(ROOT/path))==h
    for r in rows('audio_index.csv'):
        rate,wave=wavfile.read(OUT/r['wav']);assert rate==50000 and wave.shape==(10000,) and wave.dtype==np.float32
        np.testing.assert_array_equal(wave,(load(OUT/r['raw_npz'])['audio']*float(r['gain'])).astype(np.float32))
    assert read(OUT/'audit/endpoints.json')['passed'] and read(OUT/'physics_audit.json')['passed']
    assert read(OUT/'execution.json')['training_candidates']==36864 and read(OUT/'execution.json')['independent_policy_evaluations']==297
    for s in summaries():
        assert s['status']=='complete' and source_hashes_match(s['source_sha256'], hashes())
        for name,h in s['artifact_sha256'].items():assert digest(folder(s['method'],s['seed'])/name)==h
    result=dict(passed=not changed and not deleted and not broken,original_files_checked=len(protected),original_files_changed=changed,original_files_deleted=deleted,
        local_links_checked=links,broken_links=broken,wavs_checked=len(rows('audio_index.csv')),native_sounds_generated=0,stageA_candidates=18,research_runs=9,
        research_training_steps=36864,independent_evaluations=297,independent_endpoint_audits=36,source_sha256=digest(__file__))
    # Stable contents allow a read-only audit after the artifact manifest is frozen.
    if read(OUT/'integrity.json').get('status')=='running':write_json(OUT/'integrity.json',result)
    else:assert read(OUT/'integrity.json')==result
    assert result['passed'],result;return result

def freeze():
    write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in [*sorted(ROOT.glob('issue27f*.py')),ROOT/'tests/test_issue27f.py']})
    write_json(OUT/'artifact_checksums.json',{});write_json(OUT/'storage_after.json',{});audit()
    repository=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    for _ in range(5):
        status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True);new=[s[3:] for s in status.splitlines() if s.startswith('?? ')]
        data=dict(workspace=workspace_inventory(),new_untracked_files=len(new),new_untracked_bytes=sum((repository/p).stat().st_size for p in new),modified_tracked_files=sum(s.startswith(' M ') for s in status.splitlines()),
            deleted_tracked_files=sum('D' in s[:2] for s in status.splitlines()),measure='Logical regular files including ignored; exclude .git/symlinks; no commit/stage')
        write_json(OUT/'storage_after.json',data);before=read(OUT/'storage_before.json')['workspace']['total'];after=data['workspace']['total']
        block='<!-- STORAGE_BEGIN -->\n'+table(['対象','開始前','完了時'],[['ファイル数',before['files'],after['files']],['論理bytes',before['bytes'],after['bytes']],['新規Git候補',0,f"{len(new)} / {data['new_untracked_bytes']} bytes"],['追跡変更/削除','0/0',f"{data['modified_tracked_files']}/{data['deleted_tracked_files']}"]])+'\n<!-- STORAGE_END -->'
        p=OUT/'REPORT.md';body=p.read_text();p.write_text(re.sub(r'<!-- STORAGE_BEGIN -->.*?<!-- STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- STORAGE_BEGIN -->' in body else body+'\n'+block+'\n')
        write_json(OUT/'artifact_checksums.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    data['workspace']=workspace_inventory();write_json(OUT/'storage_after.json',data)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('report','audit','freeze'));a=p.parse_args();report() if a.stage=='report' else print(audit()) if a.stage=='audit' else freeze()
