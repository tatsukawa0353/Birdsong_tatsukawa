"""Integrated report, exact completed-run accounting and preservation audit."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile
from .issue27g_data import ROOT,OUT,CONTROLS,CORE,METHODS,SEEDS,folder,ready,hashes,target_path
from .issue27g_analysis import read,load,rows,summaries
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest
from .issue19_protection import workspace_inventory
from .issue23_stage_a import table as plain_table

def table(headers,data):return plain_table(headers,[[str(x).replace('|','\\|') for x in r] for r in data])
def n(x):return f'{float(x):.6f}'
def image(name):return f'![{name}](figures/{name}.png)\n'
def ms(r,k):return n(r[k+'_mean'])+' ± '+n(r[k+'_sample_sd'])

def report():
    manifest=ready();runs=summaries();assert len(runs)==54 and all(s['status']=='complete' for s in runs);assert read(OUT/'audit/endpoints.json')['passed']
    gates=read(OUT/'preflight/gate.json');summary=rows('method_summary.csv');pairs=rows('paired_summary.csv');endpoints=rows('endpoints.csv');timing=rows('training_accounting.csv');exe=read(OUT/'execution.json');interrupted=read(OUT/'interrupted_attempt/interruption.json')
    targets=table(['Target','[ε_l,ε_r,p_s,f0_l,f0_r]','一次持続proxy','直線除去後','左右変動proxy','oracle E/J'],
        [[t,c.tolist(),n(gates[t]['features']['late_ac_ratio']),n(gates[t]['features']['detrended_ac_ratio']),f"{gates[t]['features']['left_motion_periodic_proxy']}/{gates[t]['features']['right_motion_periodic_proxy']}",'0 / 0'] for t,c in CONTROLS.items()])
    main=table(['Target','次元','手法','Final E_total 平均±SD','Selected E_total 平均±SD','Final持続率'],
        [[t,d,m,ms(next(r for r in summary if (r['target'],int(r['dim']),r['method'],r['role'])==(t,d,m,'final')),'E_total'),
          ms(next(r for r in summary if (r['target'],int(r['dim']),r['method'],r['role'])==(t,d,m,'selected')),'E_total'),
          next(r['voiced_mean'] for r in summary if (r['target'],int(r['dim']),r['method'],r['role'])==(t,d,m,'final'))] for t in CONTROLS for d in (3,4) for m in METHODS])
    paired=table(['Target','手法','役割','ΔE_total 平均±SD','ΔJ 平均±SD','Δ共通3変数誤差 平均±SD'],
        [[r['target'],r['method'],r['role'],ms(r,'E_total_4D_minus_3D'),ms(r,'J_4D_minus_3D'),ms(r,'common3_error_mean_4D_minus_3D')] for r in pairs if r['role'] in ('final','selected')])
    all_metrics=table(['Target','次元','手法','役割','E_spec 平均±SD','E_env 平均±SD','E_total 平均±SD','J 平均±SD'],
        [[r['target'],r['dim'],r['method'],r['role'],*[ms(r,k) for k in ('E_spec','E_env','E_total','J')]] for r in summary])
    parameter=table(['Target','次元','手法','共通3変数の幅正規化誤差 平均±SD','右ε絶対誤差 平均±SD','右ε幅正規化誤差 平均±SD'],
        [[r['target'],r['dim'],r['method'],ms(r,'common3_error_mean'),ms(r,'epsilon_r_abs_error') if r['dim']=='4' else '固定・推定対象外',ms(r,'epsilon_r_width_normalized_error') if r['dim']=='4' else '固定・推定対象外'] for r in summary if r['role']=='final'])
    geometry=table(['次元','手法','Actor/方策','Critic/価値','target Actor','target Critic','policy合計','optimizer step/run'],
        [[s['dim'],s['method'],s['network_geometry']['actor_parameters'],s['network_geometry'].get('critic_parameters',s['network_geometry'].get('value_parameters')),s['network_geometry'].get('actor_target_parameters',0),s['network_geometry'].get('critic_target_parameters',0),s['network_geometry']['total_policy_parameters'],s['optimizer_steps']] for s in runs if s['target']=='T1' and s['seed']==0])
    times=table(['Target','次元','手法','seed','学習wall秒','評価wall秒','process CPU秒','非持続学習音','数値失敗'],[[r['target'],r['dim'],r['method'],r['seed'],n(r['training_wall_s']),n(r['evaluation_wall_s']),n(r['process_cpu_s']),r['nonvoiced'],r['failures']] for r in timing])
    boundaries=[];gaps=[]
    from .issue17_artifacts import read_record
    for s in runs:
        final=s['final'];curve=read_record(folder(s['target'],s['dim'],s['method'],s['seed'])/'curve.json');action=np.asarray(final['actions'][0]);initial=curve[0]['acoustic_error']
        boundaries.append(dict(target=s['target'],dim=s['dim'],method=s['method'],seed=s['seed'],final_action=action.tolist(),boundary_coordinates=int(np.sum(abs(action)==1)),
            final_physical_controls=final['physical_controls'],final_policy_sha256=final['learner_after']['learner']['policy_sha256'],final_policy_statistics=final['policy_statistics']))
        gaps.append(dict(target=s['target'],dim=s['dim'],method=s['method'],seed=s['seed'],initial_E=initial,first_half_initial_E_trial=next((r['episodes'] for r in curve if r['acoustic_error']<=initial/2),None),
            first_voiced_evaluation_trial=next((r['episodes'] for r in curve if r['sustained_voiced']),None),Final_E=final['acoustic_error'],Selected_E=s['selected']['acoustic_error'],training_best_E=s['training_best']['acoustic_error'],
            Final_minus_Selected_E=final['acoustic_error']-s['selected']['acoustic_error'],Final_minus_training_best_E=final['acoustic_error']-s['training_best']['acoustic_error']))
    write_csv(OUT/'boundary_diagnostics.csv',boundaries);write_csv(OUT/'acquisition_retention.csv',gaps)
    common_actions=[]
    for t in CONTROLS:
        for d in (3,4):
            for m in METHODS:
                subset=[r for r in boundaries if (r['target'],r['dim'],r['method'])==(t,d,m)]
                if all(r['final_action']==subset[0]['final_action'] for r in subset):
                    common_actions.append(dict(target=t,dim=d,method=m,action=subset[0]['final_action'],unique_weight_hashes=len({r['final_policy_sha256'] for r in subset})))
    write_json(OUT/'audit/common_Final_actions.json',dict(groups=common_actions,native_sounds=0))
    conclusions=[]
    for t in CONTROLS:
        for m in METHODS:
            r=next(r for r in pairs if (r['target'],r['method'],r['role'])==(t,m,'final'));delta=float(r['E_total_4D_minus_3D_mean']);sd=float(r['E_total_4D_minus_3D_sample_sd'])
            conclusions.append(f"- {t} / {m}：4D−3DのFinal E_totalは{delta:+.6f} ± {sd:.6f}。この3seedの平均では4Dで{'減少' if delta<0 else '増加'}した。")
    known=4+2+2412+221184+1782+216;assert known==225600
    lower=interrupted['completed_training_lower']+interrupted['evaluation_lower'];upper=interrupted['completed_training_upper']+interrupted['evaluation_upper']+interrupted['in_progress_sounds_upper']
    accounting=dict(completed_confirmed_sounds=known,confirmed_native_steps=known*2000000,Target_and_oracle_sounds=4,physical_sensitivity_sounds=2,smoke_sounds=2412,
        formal_training=221184,independent_evaluations=1782,endpoint_replays=216,interrupted_extra_sound_equivalent_lower=lower,interrupted_extra_sound_equivalent_upper=upper,
        total_native_steps_lower=(known+lower)*2000000,total_native_steps_upper=(known+upper)*2000000,exact_overall_count_available=False,synthetic_episodes=1536,synthetic_native_sounds=0,
        same_seed_infrastructure_recalculations=27,changed_or_replaced_seeds=0,extra_parameter_search=0)
    write_json(OUT/'accounting_complete.json',accounting)
    text=f'''# Issue27再改訂 — 3Target×一定3D/4D・SAC/TD3/PPO

2026-11-26 JST。[今回のIssue本文](issue_snapshot.json)（[GitHub #27](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/27)）で事前指定されたT1/T2/T3を確定し、**3Target×2次元×3手法×3seedの54runを各4096step、計221184学習音で完了した。** 独立方策評価は1782回、4役割の独立詳細再生は216回で全て一致した。

実行セッションのSIGTERMによる中断が一度あり、最初の27本の途中成果は保護して同じseed・同じ凍結条件から復旧した。中断分は正式54runと別会計で、終了stepが保存されなかったため全計算の回数は確定値と範囲を併記する。[復旧記録](RECOVERY.md)。完了54runと、中断27本を計算コスト上で混同しない。

[中断前54チェックポイントの照合](audit/recovery_prefix.json)では、復旧後の同じ評価時点の行動とE_total/Jが全件完全一致した。再計算前の誤差を利用した結果選別は行っていない。

## 3つのTarget（学習前に指定・凍結）

{targets}

{image('targets')}

[T1 WAV](audio/T1_Target.wav) / [T2 WAV](audio/T2_Target.wav) / [T3 WAV](audio/T3_Target.wav)。図の3列はT1/T2/T3、上段スペクトログラム、中段生波形、下段時間平均PSD。全3Targetで共通PSD基準と共通gainを用い、絶対強度を比較できる。T1は片側主導・右閉鎖側設定、T2は両側対称、T3は両側非対称の指定であり、解剖学的閉鎖や同期/非同期の証明とは呼ばない。

T1/T2は前回保存b2_c10/b3_c10のNPZをbyte同一で参照し、T3のみ新規1点を生成した。3音とも別Bインスタンスoracleを再計算し、波形、全20block状態、native運動mean/stdが完全一致、E_spec/E_env/E_total/J=0。右source/outputは非ゼロ。旧18点/9点の再探索やTarget差し替えはない。[発声ゲートと出典hash](preflight/gate.json) / [事前計画](PLAN.md) / [manifest](manifest.json)。

[初回停止REPORT](../phase1_caseB_self_target_020s_sac128/REPORT.md)、[旧9点REPORT](../phase1_caseB_self_target_020s_methods/REPORT.md)、[前回18点・9run REPORT](../phase1_caseB_self_target_020s_methods_f0lr_scan/REPORT.md)を参照し、今回を別実験として保存している。

発声ゲートは末尾1000sample（20ms）のAC std/自身の全期間peak>.05、線形傾向除去後ratio>.05、finite、generated_silent=False。左右変動proxyは同じ末尾traceでde-trended std/元std>.5かつde-trended std/全期間state幅>.001をx/y双方に適用した記述量。T1は左、T2/T3は左右の条件を確認した。音声そのものをde-trendしたり、周期性の数学的証明と呼んだりはしない。学習・独立方策評価の持続proxyは従来どおりTarget peak基準で、Stage Aの自身peak基準と区別する。

## 同じTarget内の3D→4D比較

T1とT2では全3手法のFinal平均E_totalが4Dで減少した。T1はSAC/TD3のseed差が大きく、Finalの音圧持続proxyは3Dが1/9、4Dが3/9にとどまった。Target自体の発声ゲート・oracleは合格しており、非持続のFinal方策を計算失敗や物理モデルの発声不能と混同しない。

T2はFinal18/18が持続proxyを満たしたが、Target模倣誤差は残った。TD3のFinal平均E_totalは3Dの0.361573から4Dの0.227417へ減少した。T3もFinal18/18が持続し、SAC/TD3は3D→4Dで平均E_totalがそれぞれ約0.01828/0.00827増加、PPOは約0.01488減少した。異なるTarget間の誤差の大小から、片側/対称/非対称構造の一般的優劣を結論しない。

{main}

平均±標本SD（ddof=1）、各3seed。音響精度は各Target自身を基準としている。以下のΔは同じTarget・手法・seedについて4D−3Dを計算し、その3seed平均と標本SDを求めた値。負ならその指標は4Dで小さい。異なるTargetの絶対誤差を構造効果として比較しない。

{paired}

{chr(10).join(conclusions)}

上の記述は単一予算・3seedの結果に限られ、一般的優劣や統計的有意性を断定しない。右εを追加すると探索域・ネット出力・SAC target_entropy・ノイズ次元が指定どおり変わるため、次元の影響を単一要因の厳密因果効果として切り分けたものではない。

## Final/Selectedの視覚比較

次の各図は4行Target/SAC/TD3/PPO、6列3Dのseed0/1/2、4Dのseed0/1/2。同じTargetの3D/4D、全手法、全seed、Final/Selectedで時間・周波数・PSD基準と色軸[-40,0]dBを統一した。Targetのfull-band表示PSD最大が基準で、生成音ごとの正規化はしない。飽和率は[display_settings](display_settings.json)に全108生成パネル分を保存し、広域表示は補助1枚だけとした。

{image('T1_final_spectrograms')}
{image('T1_selected_spectrograms')}
{image('T2_final_spectrograms')}
{image('T2_selected_spectrograms')}
{image('T3_final_spectrograms')}
{image('T3_selected_spectrograms')}

[広域補助：代表seed1のFinal](figures/wide_psd_reference.png)。波形の個別gain調整、位相合わせ、ドリフト除去による音の加工はしない。狭い色軸で見えない成分を報酬や保存音から除去したものでもない。

## 学習曲線・探索候補と方策の獲得

{image('learning_curves_all_seeds')}
{image('learning_curves_mean_sd')}

全seed図は列T1/T2/T3、行3D seed0/1/2、4D seed0/1/2。各パネルに3手法の33評価点をすべて表示した。平均図は各指標E_spec/E_env/E_total/Jについて3D/4Dを隣接行、列T1/T2/T3、中心は3seed平均・帯は標本SD。平滑化・間引き・頂点省略は行わない。

Final=4096step全更新完了後の決定論的方策。Selected=33点中最初のminJ方策。posthoc=同じ33点中最初のminE_total方策。training_best=4096学習候補中最初のminJ行動で、保存Actorと呼ばない。J=.75E_spec+.25E_envとE_total=E_spec+E_envは選択基準が異なる。

[獲得・保持・training_bestとの差](acquisition_retention.csv)には初期E半減の初回評価時点、持続proxyの初回観測点、Final−Selected/Final−training_bestを保存。[境界への張り付き](boundary_diagnostics.csv)と[探索範囲・方策分散](exploration.csv)、[3seed共通Final行動の検査](audit/common_Final_actions.json)も保存した。同じ出力になった場合もseed別重みhashと実行時clipを区別し、同一結果を複製したとみなさない。E_total≤.10を合否・延長・再試行の条件には用いていない。

3Targetとも左f0の正解49000は探索下限。SACのtanh付きGaussian、TD3のtanh Actorと学習時Normalノイズ・clip、PPOのGaussianと環境clipでは境界へ到達する性質が異なる。TD3の決定論的評価には探索ノイズがなく、学習候補で境界に達したことをActorの復元成功と混同しない。T3の右ε正解21Mも4D下限20Mに近いが、3Dでは正解値に固定した既知条件である。

## 制御値と同定誤差

{image('final_controls')}

列はT1/T2/T3。各パネルの横配置はTarget（黒印）、3DのSAC/TD3/PPO、4DのSAC/TD3/PPO（S/T/P）。seed点と平均±標本SDを表示。3Dの右εは各Target自身の正解値に固定されており、復元の成功例には数えない。

{parameter}

各パラメータの絶対誤差と探索幅正規化誤差を[endpoints.csv](endpoints.csv)へ全216結果保存した。共通3変数誤差は|Δε_l|/2.5e7、|Δp_s|/1.5e6、|Δf0_l|/751000の平均。4D右εは|Δε_r|/3e7として別列、3Dでは推定対象外の空欄。共通3変数と4D右εを混ぜた平均で次元間の精度を主張しない。音響模倣とパラメータ同定は別で、対称条件・閉鎖側条件でも制御値が一意に決まるとは仮定しない。

## 波形・平均PSDと音源内部状態

{image('final_waveform_mean_psd')}

列T1の3D/4D、T2の3D/4D、T3の3D/4D。代表seed1を学習前に固定。行は全0.20sの5ms RMS、末尾5msの生波形、全表示窓の時間平均PSD。全波形は3Target共通peakの逆数によるgain、PSDは同じTarget内の共通基準。全seed・4役割のWAVと生波形は[音声索引](AUDIO.md)に保存し、training_best/posthocの網羅的PNGは作成していない。聴取による主観評価は未実施。

{image('target_internal')}

T1は左の持続変動に対して右はドリフト主体だが、右source/outputは非ゼロで圧力結合へ寄与する。T2の同一係数・零初期条件での左右trace一致を、摂動下の同期安定性の証明にはしない。T3の指定非対称εでは左右traceと周波数構造に違いが観測されたが、非同期の一般的証明とも呼ばない。右の微小応答をゼロへclampしない。

T1の右ε=2e7/5e7の事前検査では実メモリ値と出力の変化を確認した。[保存データによる差の数値](preflight/right_sensitivity_saved_data.json)。これは広域2点間の音圧差であり、局所感度・一意な同定可能性を証明するものではない。追加の感度gridは実行していない。

左右のsource_j=α_j(x_j−τ_j y_j)+β_j y_j、output_j=source_j−β_j τ_j dy_j/dt、feedback=−γπ(t−T)を保存し、output_l+output_r+feedbackと未フィルタ音圧の一致を丸め誤差範囲で監査した。連成状態での代数分解で、独立に発声させた音や因果的寄与率ではない。[全役割の左右運動](endpoint_motion.csv) / [左右音響寄与](endpoint_sources.csv) / [全native block統計](endpoint_native_blocks.csv) / [全時系列NPZ](audit/) / [Targetと代表結果のCSV.gz](timeseries/)。

## 物理・学習条件と実ネット構造

元native/model.cpp、model.py、既存libbirdsong.soは変更していない。[同一ソース再ビルドのbyte/hash一致](native/build.json)。独立ConfiguredCaseB wrapperで入力範囲を右ε2e7まで拡げ、右f0をTargetごとに保持するだけで、native condition1と左右α1.2e6/β1500を維持。元の右εの下限2.5e7チェックや以前の35M固定を今回へ持ち込まない。全5制御の実効値・固定物理係数をreset/step/評価/再生で実メモリ検査した。

0.20s一定、2M native step/音、dt1e-7、50kHz10000sample。初期4状態・遅延・filter0、16次15kHz lowpass→200分の1デシメーション。x0=.02cm/D=.01等も不変。3D可変[ε_l,p_s,f0_l]、4D可変[ε_l,ε_r,p_s,f0_l]。ε_l25–50M、右ε20–50M、p_s2.5–4M、左f049000–800000。右f0はT1=1e7、T2/T3=49000で固定。f0は外力項でHzではない。観測は全ケース8Dの同じreset[-1,0,…]で正解制御値を含まない。

全手法LR3e-4/gamma1/batch32/CPU1thread。隠れ層128×128、SAC/TD3 ReLU、PPOはpi/vf別128×128 Tanh。SAC/TD3 buffer50000/warmup16/train_freq1/gradient_steps1/tau.005。SAC auto_0.01/target_entropy=−3または−4。TD3次元別Normal(0,.1)、delay2/targetnoise.2/clip.5。PPO n_steps128/n_epochs10/clip.2/gae.95/vf_coef.5/ent0/max_grad_norm.5/normalize_advantage=True/target_kl=None。Adam epsはPPO1e-5、他1e-8。[全実効設定とpackage版](manifest.json)、各run configに保存した。

{geometry}

SAC温度1parameterはpolicy合計の外。隠れ層幅が同じでもパラメータ数や更新数が同じとは言わない。SAC actor/critic/entropy各4080、TD3 critic4080/actor2040、PPO32rollout×4minibatch×10epoch=1280 optimizer step（内部epoch counter320）をrunごとに実測検査した。

## 事前検査・独立監査

6組のマッピング、上下限、正解値の逆変換、実メモリ、全54初期seed設定、実ネット構造とparameter数を検査。2次元×3手法の標準SB3と監査付き実装をsynthetic128stepで照合し、重み・optimizer・Buffer・RNGが完全一致した（計1536synthetic episode、新規音0）。方法別smokeは18本×128学習=2304音、独立評価36音、保存3方策とtraining_best再生72音、計2412音。[事前検査](preflight/tests.json)。最初の保存モデル読込試験で引数指定の不一致があり修正した。完了済み18smoke学習を再利用し、未実施だった再生検査だけを実行した。再学習による差し替えはない。

SAC/TD3はReplayに保存する正規化actionと標準unscaleによる環境実行actionの整合、PPOはraw Gaussian actionを保持するrolloutと環境clip actionの整合を全221184行で検査した。全optimizer更新のLR・有限性・既存の勾配/parameter/Adam moment停止条件を検査。PPO既定grad clip.5を保持し、追加clampはない。

独立評価は0,128,…,4096の33点/run、対応する全更新後。評価前後の学習器・optimizer・Buffer・RNG・training_best signatureが1782回すべて不変。Final/Selected/posthocの162方策とtraining_best54行動を別Bで216音再生し、波形・状態・制御・音響誤差が全て一致。[独立再生監査](audit/endpoints.json)。保存.ptは推論用方策/価値/targetネットとSAC温度で、optimizer/Replayを含む学習再開checkpointではない。

## 全指標の平均・標本SD

{all_metrics}

[全seed全216結果](endpoints.csv) / [72群の平均・標本SDとparameter誤差](method_summary.csv) / [1782評価点CSV](learning_curves.csv) / [対応するseedの3D→4D差](paired_differences.csv) / [学習ログ・モデル・生音](runs/)。全221184候補の行動・5物理値・音響指標をtraining_trace.json.gzに保存した。全候補WAVや全更新重みを保存したものではない。

## 実行時間・回数と中断分

{times}

復旧後54runの並列wallは{exe['wall_s']:.3f}秒、27worker、各Torch1thread。学習wallは各runのlearn区間から独立評価・保存時間を差し引き、CPUはrun process全体。並列負荷下の時間で、専有CPUでの速度benchmarkではない。[群別時間平均±SD](timing_summary.csv)。

| 確認済み区分 | 完了した音生成回数 |
|---|---:|
| 新規T3と独立oracle3音 | 4 |
| T1右εの事前物理検査 | 2 |
| 方法別smoke学習/評価/再生 | 2304 / 36 / 72 |
| 正式54run学習 | 221184 |
| 正式独立方策評価 | 1782 |
| 終了後独立再生 | 216 |
| 確認済み完了音の小計 | **225600** |
| 別途発生した中断計算 | 下記の範囲で記録 |

中断27runには、完了学習候補{interrupted['completed_training_lower']}〜{interrupted['completed_training_upper']}、独立評価{interrupted['evaluation_lower']}〜{interrupted['evaluation_upper']}、加えて最大27音の実行途中があり得る。これはログからの保守的な範囲で、正確な終了回数は不明。**中断分を含む全native stepは{accounting['total_native_steps_lower']}〜{accounting['total_native_steps_upper']}の範囲**であり、225600音だけを全実計算と表示しない。中断27本は同じseed/条件で初期状態から再計算し、残り27本は最初の実行。異なるseedへの差し替え、精度を見た追加学習やTarget変更はない。[全会計](accounting_complete.json) / [中断記録](interrupted_attempt/interruption.json) / [逐次counter](progress/) / [正式会計](execution.json)。

## 図の書式・画像整理・旧成果保護

SciencePlotsの従来実コードissue17_comparison_figures.pyのscience/ieee/no-latexを読み、実効font.family=serif/font.serif=STIXGeneral、mathtext=cm、TeX=Falseを踏襲した。代替フォントへの変更はない。軸46pt/目盛30pt/凡例29pt/カラーバー46・30pt、実線4pt、共通手法色（SAC橙/TD3青/PPO緑）、薄いSD帯。タイトル・grid・閾値線はなく、パネルIDのみ最小表示。共通設定はissue27g_style.pyの1か所へ集約し、保存時に実artistの書式をassertした。

過去の単独図15×10inch/120DPIを基準に、集約図は物理寸法を拡大して文字サイズを維持した。各図の実寸・解決font名/path・フォントサイズ・線幅・DPI・全線の頂点数/hash・色軸は[target図metadata](target_figures.json)、[学習図metadata](learning_figures.json)、[表示設定](display_settings.json)に保存した。表示STFTは従来通り10MHzへresample、Blackman-Harris245760/noverlap184320、PSD density、detrend constant、complete windowのみ、0–10kHz。報酬用50kHz STFTは変更していない。

Final/SelectedのTarget別集約図、全seed/平均学習曲線、制御、波形/平均PSD、3Target比較、必要な内部状態図、広域補助1枚だけを保存した。training_best/posthocの大量個別PNGや重複試作画像は作成していない。[画像の採否・削除記録](image_cleanup.json)。旧画像は整理対象にせず、旧Issue17〜26とIssue27の初回/9点/18点・9runの成果を全件hash照合し保護した。[最終監査](integrity.json)。追跡ソースの保護hashは開始時clean HEADを根拠に補完し、元の一覧も保持した。[補完根拠](protection_supplement.json)。

今回の目的はCase B内部で同一Targetに対する次元・手法の比較。Case Aとは異なる自己Targetのため、A/B誤差から2音源構造の優位を結論しない。TQC、DE/GA/CMA-ES、5D、時変制御、共通実録音、追加Target探索、全遅延系の安定性解析、主観聴取評価は未実施。commit/push/Issue更新も行っていない。

[再現手順](REPRODUCE.md) / [実装hash](implementation_checksums.json) / [成果物hash](artifact_checksums.json) / [保存容量](storage_after.json)。
'''
    (OUT/'REPORT.md').write_text(text)
    (OUT/'README.md').write_text('# 3Target×3D/4Dの成果\n\n[REPORT](REPORT.md) / [PLAN](PLAN.md) / [manifest](manifest.json) / [音声](AUDIO.md) / [復旧記録](RECOVERY.md) / [監査](integrity.json)。\n\n54runの完了結果と、中断27本の追加計算を区別している。\n')
    (OUT/'AUDIO.md').write_text('# 全Targetと全seed・4役割の音声\n\n全219 WAV、同じ3Target共通gain。生NPZには左右状態・寄与を含む。\n\n'+table(['音','WAV','生音/左右状態'],[[r['name'],f"[WAV]({r['wav']})",f"[NPZ]({r['raw_npz']})"] for r in rows('audio_index.csv')]))
    (OUT/'REPRODUCE.md').write_text('''# 保存済み結果の検証

リポジトリ親から実行する。既存成果の削除・上書きや再学習はしない。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27g-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27g_report audit
```

このauditは保存WAV・モデル・全source/artifact/Target hash・旧成果・counterを検査するだけで新規音0。事前smokeやanalysis endpointsは音を生成するので、完了済みのledgerがある場合は再実行しない。モデルは推論用で学習再開Buffer/optimizerを含まない。

実行順：prepare stage_a（4音）→3Target図→tests（mapping/標準実装照合/右ε2音/18smoke）→保存モデル読込の引数修正後replay-only（学習再実行なし）→prepare freeze→training（SIGTERM中断）→recovery（同じ54設定・seed、途中成果保護・逐次counter）→analysis physics（保存データのみ）/endpoints（216音）→figures learning→report/freeze/audit。中断の詳細と不確定な追加回数はRECOVERYとinterruption.jsonを参照。
''')
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for name in ('stage_a','tests','tests-replay','training','recovery'):
        p=Path(f'/tmp/birdsong-issue27g-{name}.log')
        if p.exists():shutil.copyfile(p,logs/p.name)
    files=set(CORE)|{str(p.relative_to(ROOT)) for p in ROOT.glob('issue27g*.py')}|{'tests/test_issue27g.py'};copied={}
    for name in sorted(files):
        dest=OUT/'executed_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);copied[name]=digest(dest)
    write_json(OUT/'executed_source_checksums.json',copied)

def audit():
    ready();assert read(OUT/'preflight/tests.json')['source_sha256']==hashes();assert read(OUT/'recovery_manifest.json')['source_sha256']==digest(ROOT/'issue27g_recovery.py')
    protected=rows('protected_before.csv');changed=[];deleted=[];broken=[];links=0
    if not (OUT/'integrity.json').exists():write_json(OUT/'integrity.json',dict(status='running'))
    for r in protected:
        p=ROOT/r['path']
        if not p.exists():deleted.append(r['path'])
        elif digest(p)!=r['sha256']:changed.append(r['path'])
    for p in (OUT/'REPORT.md',OUT/'README.md',OUT/'AUDIO.md',OUT/'PLAN.md',OUT/'RECOVERY.md',OUT/'REPRODUCE.md'):
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            links+=1
            if not (p.parent/dest.split('#')[0]).exists():broken.append(dict(file=p.name,link=dest))
    for name,base in (('implementation_checksums.json',ROOT),('artifact_checksums.json',OUT)):
        if (OUT/name).exists():
            for p,h in read(OUT/name).items():assert digest(base/p)==h,p
    for p,h in read(OUT/'executed_source_checksums.json').items():assert digest(ROOT/p)==h and digest(OUT/'executed_sources'/p)==h
    for r in rows('audio_index.csv'):
        rate,wave=wavfile.read(OUT/r['wav']);assert rate==50000 and wave.shape==(10000,) and wave.dtype==np.float32
        np.testing.assert_array_equal(wave,(load(OUT/r['raw_npz'])['audio']*float(r['gain'])).astype(np.float32))
    for s in summaries():
        assert s['status']=='complete' and s['source_sha256']==hashes()
        for p,h in s['artifact_sha256'].items():assert digest(folder(s['target'],s['dim'],s['method'],s['seed'])/p)==h
        for kind,num in (('training',4096),('evaluation',33)):
            counter=read(OUT/f"progress/{s['target']}_{s['dim']}D_{s['method']}_s{s['seed']}/{kind}.json")
            assert counter['completed_episodes']==num and counter['native_blocks_completed']==num*20 and counter['failures']==0
    assert read(OUT/'audit/endpoints.json')['passed'] and read(OUT/'execution.json')['complete_runs']==54
    figures=read(OUT/'target_figures.json')+read(OUT/'learning_figures.json');referenced=set()
    for item in figures:
        assert digest(OUT/item['path'])==item['sha256'];assert (item['label_pt'],item['tick_pt'],item['legend_pt'],item['dpi'])==(46,30,29,120)
        assert item['mathtext_fontset']=='cm' and item['text_usetex']==False and item['font_serif']==['STIXGeneral'] and item['line_widths_pt'] in ([],[4.])
        referenced.add(item['path'])
        if item['path'].endswith('learning_curves_all_seeds.png'):assert len(item['line_vertices'])==54 and all(x['points']==33 for x in item['line_vertices'])
        if item['path'].endswith('learning_curves_mean_sd.png'):assert len(item['line_vertices'])==72 and all(x['points']==33 for x in item['line_vertices'])
    actual={str(p.relative_to(OUT)) for p in (OUT/'figures').glob('*.png')};assert actual==referenced
    assert all(p in (OUT/'REPORT.md').read_text() for p in actual)
    result=dict(passed=not changed and not deleted and not broken,original_files_checked=len(protected),original_files_changed=changed,original_files_deleted=deleted,
        local_links_checked=links,broken_links=broken,wavs_checked=len(rows('audio_index.csv')),figures_checked=len(figures),research_runs=54,training_steps=221184,independent_evaluations=1782,independent_replays=216,
        interrupted_runs_separately_preserved=27,exact_overall_count_available=False,additional_native_sounds=0,source_sha256=digest(__file__))
    if read(OUT/'integrity.json').get('status')=='running':write_json(OUT/'integrity.json',result)
    else:assert read(OUT/'integrity.json')==result
    assert result['passed'],result;return result

def freeze():
    images={str(p.relative_to(OUT)) for p in (OUT/'figures').glob('*.png')};body=(OUT/'REPORT.md').read_text();assert all(p in body for p in images)
    write_json(OUT/'image_cleanup.json',dict(kept=sorted(images),deleted=[],deletion_reasons=[],reason='Only REPORT-referenced aggregate figures retained; no individual debug/role copies were produced. Old images unchanged.'))
    write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in [*sorted(ROOT.glob('issue27g*.py')),ROOT/'tests/test_issue27g.py']})
    write_json(OUT/'artifact_checksums.json',{});write_json(OUT/'storage_after.json',{});audit()
    repo=ROOT.parent
    for _ in range(5):
        status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True);new=[s[3:] for s in status.splitlines() if s.startswith('?? ')]
        record=dict(workspace=workspace_inventory(),new_untracked_files=len(new),new_untracked_bytes=sum((repo/p).stat().st_size for p in new),modified_tracked_files=sum(s.startswith(' M ') for s in status.splitlines()),deleted_tracked_files=sum('D' in s[:2] for s in status.splitlines()),measure='Logical regular-file bytes incl ignored; exclude .git/symlinks')
        write_json(OUT/'storage_after.json',record);before=read(OUT/'storage_before.json')['workspace']['total'];after=record['workspace']['total']
        block='<!-- STORAGE_BEGIN -->\n'+table(['対象','開始前','完了時'],[['ファイル数',before['files'],after['files']],['論理bytes',before['bytes'],after['bytes']],['新規Git候補',0,f"{len(new)} / {record['new_untracked_bytes']} bytes"],['追跡変更/削除','0/0',f"{record['modified_tracked_files']}/{record['deleted_tracked_files']}"]])+'\n<!-- STORAGE_END -->'
        p=OUT/'REPORT.md';text=p.read_text();p.write_text(re.sub(r'<!-- STORAGE_BEGIN -->.*?<!-- STORAGE_END -->',lambda _:block,text,flags=re.S) if '<!-- STORAGE_BEGIN -->' in text else text+'\n'+block+'\n')
        write_json(OUT/'artifact_checksums.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    record['workspace']=workspace_inventory();write_json(OUT/'storage_after.json',record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('report','freeze','audit'));a=p.parse_args();report() if a.stage=='report' else freeze() if a.stage=='freeze' else print(audit())
