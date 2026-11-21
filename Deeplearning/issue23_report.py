"""Evidence-based stability diagnosis; artifact selection is not learning improvement."""
import csv
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

from .issue23_data import (ROOT,BASE,OUT,SEEDS,NAMES,HALF_WIDTH,ready,folder_for,curve_for,
    chosen,target_data)
from .issue23_stage_a import table
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest
from .issue19_protection import workspace_inventory


def csv_rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def listening():
    target,_=target_data();gain=1/float(np.max(abs(target)));seen={};index=[]
    with zipfile.ZipFile(OUT/'listening.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for seed in SEEDS:
            for role in ('final','selected','posthoc'):
                with np.load(folder_for(seed)/f'{role}.npz') as z:audio=z['audio']
                wave=(audio*gain).astype(np.float32);key=hashlib.sha256(wave.tobytes()).hexdigest()
                name=seen.get(key)
                if name is None:
                    name=f'seed_{seed}_{role}.wav';buffer=io.BytesIO();wavfile.write(buffer,50000,wave)
                    archive.writestr(name,buffer.getvalue());seen[key]=name
                index.append(dict(seed=seed,role=role,archive_entry=name,shared_gain=gain,sample_sha256=key))
    write_csv(OUT/'listening_index.csv',index)
    return len(seen)


def freeze():
    from .issue23_analysis import integrity
    # Report generation changes its own outputs; discard no research files and
    # replace the new checksum manifest only after the current integrity check.
    checks=OUT/'artifact_checksums.json'
    write_json(checks,{})
    write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in
        [*sorted(ROOT.glob('issue23*.py')),ROOT/'issue23_findings.md',ROOT/'tests/test_issue23.py']})
    integrity()
    write_json(OUT/'storage_after.json',{})
    repository=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    for _ in range(5):
        status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True)
        new=[line[3:] for line in status.splitlines() if line.startswith('?? ')]
        record=dict(workspace=workspace_inventory(),new_git_candidates=len(new),
            new_git_candidate_bytes=sum((repository/p).stat().st_size for p in new if (repository/p).is_file()),
            modified_tracked_files=sum(not line.startswith('?? ') for line in status.splitlines()),
            deleted_tracked_files=sum('D' in line[:2] for line in status.splitlines()),
            measurement='Logical regular files including ignored; .git/symlinks excluded; nothing staged/committed')
        write_json(OUT/'storage_after.json',record)
        before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total'];after=record['workspace']['total']
        block='<!-- ISSUE23_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[
            ['作業ツリー通常ファイル数',before['files'],after['files']],
            ['論理容量（ignored含む）',f"{before['bytes']/1e6:.2f} MB",f"{after['bytes']/1e6:.2f} MB"],
            ['Git新規追加候補','0（クリーン）',f"{record['new_git_candidates']}個 / {record['new_git_candidate_bytes']/1e6:.2f} MB"],
            ['既存追跡ファイルの変更 / 削除','0 / 0',f"{record['modified_tracked_files']} / {record['deleted_tracked_files']}"]])+'\n<!-- ISSUE23_STORAGE_END -->'
        p=OUT/'REPORT.md';body=p.read_text();p.write_text(re.sub(r'<!-- ISSUE23_STORAGE_BEGIN -->.*?<!-- ISSUE23_STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- ISSUE23_STORAGE_BEGIN -->' in body else body+'\n'+block+'\n')
        write_json(checks,{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    # Refresh after the checksum manifest and report table have converged.
    final=workspace_inventory();record['workspace']=final;write_json(OUT/'storage_after.json',record)


def report():
    ready();tick=time.perf_counter();endpoints=[];classifications=[];events=[];event_q=[];sampling=[]
    predictions=read_record(OUT/'critic_predictions.json');qp={(r['seed'],r['trial'],r['candidate_id']):r for r in predictions}
    curves={s:curve_for(s) for s in SEEDS}
    for seed in SEEDS:
        c=curves[seed];selection=chosen(c);arrivals=[p for p in c if p['acoustic_error']<=.10]
        trace=read_record(folder_for(seed)/'training_trace.json');good=[p for p in trace if p['acoustic_error']<=.10]
        best_sample=min(trace,key=lambda p:p['acoustic_error'])
        sampling.append(dict(seed=seed,training_samples=4096,training_E_le_0p10=len(good),
            first_training_E_le_0p10=good[0]['episodes'] if good else None,best_training_trial=best_sample['episodes'],
            best_training_E=best_sample['acoustic_error'],best_training_J=best_sample['weighted_loss'],
            independent_policy_E_le_0p10=len(arrivals)))
        classifications.append(dict(seed=seed,first_observed_E_le_0p10=arrivals[0]['episodes'] if arrivals else None,
            evaluations_E_le_0p10=len(arrivals),minimum_E=selection['posthoc']['acoustic_error'],minimum_trial=selection['posthoc']['episodes'],
            selected_trial=selection['selected']['episodes'],selected_E=selection['selected']['acoustic_error'],final_E=selection['final']['acoustic_error'],
            selected_minus_final=selection['selected']['acoustic_error']-selection['final']['acoustic_error'],
            posthoc_minus_final=selection['posthoc']['acoustic_error']-selection['final']['acoustic_error']))
        for role,p in selection.items():
            endpoints.append(dict(seed=seed,role=role,trial=p['episodes'],E_total=p['acoustic_error'],E_spec=p['acoustic_components']['log_spectral'],
                E_env=p['acoustic_components']['envelope'],J=p['weighted_loss'],sustained_voiced=p['sustained_voiced'],
                alpha=p['policy_statistics']['alpha'],entropy=p['policy_statistics']['squashed_entropy_mc'],
                **{name:p['physical_controls'][i] for name,i in zip(NAMES,(0,2,3))}))
        pairs=[(selection['selected']['episodes'],4096)] if seed!=2 else [(2944,3072),(3072,3840),(3840,4096)]
        by_n={p['episodes']:p for p in c}
        for start,end in pairs:
            a,b=by_n[start],by_n[end];da=np.array(b['actions'][0])-a['actions'][0]
            row=dict(seed=seed,start=start,end=end,E_start=a['acoustic_error'],E_end=b['acoustic_error'],
                delta_E=b['acoustic_error']-a['acoustic_error'],J_start=a['weighted_loss'],J_end=b['weighted_loss'],
                delta_J=b['weighted_loss']-a['weighted_loss'],delta_action_l2=float(np.linalg.norm(da)),
                alpha_start=a['policy_statistics']['alpha'],alpha_end=b['policy_statistics']['alpha'],
                entropy_start=a['policy_statistics']['squashed_entropy_mc'],entropy_end=b['policy_statistics']['squashed_entropy_mc'])
            for i,name in enumerate(NAMES):
                row.update({f'delta_action_{name}':float(da[i]),f'delta_physical_{name}':float(da[i]*HALF_WIDTH[i]),
                    f'std_start_{name}':a['policy_statistics']['normalized_action_std_mc'][i],f'std_end_{name}':b['policy_statistics']['normalized_action_std_mc'][i]})
            events.append(row)
            for critic_time in (start,end):
                qa=qp[seed,critic_time,f's{seed}_t{start}'];qb=qp[seed,critic_time,f's{seed}_t{end}']
                event_q.append(dict(seed=seed,start=start,end=end,critic_time=critic_time,
                    actual_reward_start=-a['weighted_loss'],actual_reward_end=-b['weighted_loss'],
                    actual_delta_reward=a['weighted_loss']-b['weighted_loss'],
                    **{f'{head}_{which}':q[head] for head in ('Q1','Q2','minQ') for which,q in (('start',qa),('end',qb))},
                    predicted_delta_reward=qb['minQ']-qa['minQ'],
                    ranking_correct=(qb['minQ']-qa['minQ'])*(a['weighted_loss']-b['weighted_loss'])>0))
    write_csv(OUT/'comparison_per_seed.csv',endpoints);write_csv(OUT/'seed_classification.csv',classifications)
    write_csv(OUT/'replay_success.csv',sampling)
    write_csv(OUT/'event_changes.csv',events);write_csv(OUT/'event_q_comparisons.csv',event_q)
    groups=[]
    for role in ('final','selected','posthoc'):
        values=[r for r in endpoints if r['role']==role];g=dict(role=role,voiced=sum(p['sustained_voiced'] for p in values))
        for key in ('E_total','E_spec','E_env','J'):
            data=np.array([p[key] for p in values]);g.update({key+'_mean':float(data.mean()),key+'_sd':float(data.std(ddof=1))})
        groups.append(g)
    write_csv(OUT/'comparison_summary.csv',groups)
    wavs=listening()
    from .issue23_figures import render
    images=render()
    # Evidence-specific interpretation is a separately reviewable new text file,
    # written after the numerical diagnostics, never a training configuration.
    findings=(ROOT/'issue23_findings.md').read_text()
    ep_table=table(['seed','採用','試行','E_total','E_spec','E_env','J','持続proxy'],
        [[r['seed'],r['role'],r['trial'],*[f"{r[k]:.7f}" for k in ('E_total','E_spec','E_env','J')],r['sustained_voiced']] for r in endpoints])
    group_table=table(['採用','E_total平均±SD','E_spec平均±SD','E_env平均±SD','J平均±SD','持続/3'],
        [[r['role'],*[f"{r[k+'_mean']:.5f} ± {r[k+'_sd']:.5f}" for k in ('E_total','E_spec','E_env','J')],r['voiced']] for r in groups])
    event_table=table(['seed','区間','ΔE_total','ΔJ','||Δa||₂','Δε_l','Δp_s','Δf0_l','α開始→終了'],
        [[r['seed'],f"{r['start']}→{r['end']}",f"{r['delta_E']:+.6f}",f"{r['delta_J']:+.6f}",f"{r['delta_action_l2']:.6f}",
            *[f"{r['delta_physical_'+k]:+.2f}" for k in NAMES],f"{r['alpha_start']:.6f}→{r['alpha_end']:.6f}"] for r in events])
    q_table=table(['seed','行動の区間','Critic時点','実測Δr','予測ΔminQ','順位の符号一致'],
        [[r['seed'],f"{r['start']}→{r['end']}",r['critic_time'],f"{r['actual_delta_reward']:+.7f}",f"{r['predicted_delta_reward']:+.7f}",r['ranking_correct']] for r in event_q])
    corr=csv_rows('stage_a_correlations.csv');ct=[]
    for seed in SEEDS:
        for field in ('alpha','std_epsilon_l','std_ps','std_f0_l','entropy'):
            rows=[r for r in corr if int(r['seed'])==seed and r['window']=='late' and r['field']==field]
            ct.append([seed,field,*[f"{float(next(r['spearman'] for r in rows if r['alignment']==kind)):+.3f}" for kind in ('level_same_time','difference_same_interval','prior_delta_predicts_next_delta')]])
    local=csv_rows('local_gradients.csv')
    local_table=table(['seed2時点','変数','刻みh','最大|ΔE|','最大|ΔJ|','実測∂r/∂a','予測差分∂minQ/∂a'],
        [[r['anchor'],r['variable'],r['step'],*[f"{float(r[k]):.6f}" for k in ('max_abs_delta_E','max_abs_delta_J','true_reward_gradient','predicted_minQ_gradient')]] for r in local])
    execution=json.loads((OUT/'execution.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    protected=len(csv_rows('protected_before.csv'));final=groups[0]
    body=f'''# Issue23 — SAC128・λ=.75のSelected→Final診断と方策保持

## 結果

設定を変えずに3seed×4096候補を再現し、旧Issue22の全12288学習サンプルと99独立評価が一致した。Final平均E_totalは{final['E_total_mean']:.5f}±{final['E_total_sd']:.5f}、持続proxy3/3で、暫定目標の平均≤.10は未達。Selectedやposthocを保持してもFinalの学習精度が改善したことにはならない。

{findings}

## Stage Aを先行した判断と固定条件

[新規学習前に保存したSTAGE_A.md](STAGE_A.md)は変更せず、必要な中間Criticが既存成果物にないこと、追加記録、選択基準、局所36音の上限を先に固定した。Stage Aは既存CSV/JSON99行を照合し、旧Final/Selected6モデル・6音を監査。中間重みを復元できたとは主張していない。

Case A（右α_r=β_r=0）、x0=.02cm/D=.01、0.20s・50kHz・10000sample、native dt=1e-7、状態/遅延履歴0。人工TargetはIssue22と同じhash。一定3D：ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]、右ε_r=3.5e7/f0_r=49000固定。f0_lはHzでなく開閉外力係数。モデル量を実測張力やPaと直接同一視しない。

SAC Actor/Critic[128,128]、lr3e-4、batch32、gamma1、buffer50000、warmup16、train_freq1、gradient_steps1、ent_coef=auto_0.01、target_entropy=-3、CPU1thread/worker。報酬はr=−J、J=.75E_spec+.25E_env、余分な2倍はない。共通評価E_total=E_spec+E_env。範囲・学習率・幅・分散/entropy目標・batch・報酬重み・1試行の変化量上限は変更しなかった。RLの学習候補は各4096、4080 Actor/Critic/温度更新。独立評価は0,128,...,4096の33回で、最後は最終更新後。

研究対象の新規学習はSAC3runのみ。TD3の過去Selected/Final差は[補助事実](other_methods.csv)として参照し、SAC固有の現象とは断定しない。Case B・時変・実録音・優先Replay・最良方策を教師にする変更は対象外。

## 全seedの到達・維持と採用規則

Final=4096候補の最終更新後、Selected=33独立評価の最初のminJ、posthoc=同33点の最初のmin共通E。posthocは学習目標とは別の事後選別で、今回はその方策も保持した。いずれも音を再生し、行動・20ブロックの物理制御・元振幅float64波形・成分別損失を照合した。

{ep_table}

{group_table}

標本SDはddof=1、seedは3個。持続proxyは末尾1000sampleのAC標準偏差/Target peak>.05かつ数値失敗なしで、生理的な発声判定ではない。平均目標はFinal≤.10かつ持続3/3であり、保存した途中解へ差し替えて達成としない。seed分類は評価された33点に限る。[分類CSV](seed_classification.csv) / [全成分と制御](comparison_per_seed.csv)。

![全seed・全33点の音響損失](figures/loss_histories.png)

## 行動の変化・更新・温度の時間関係

{event_table}

Δaは正規化[-1,1]座標の差、全探索幅に対する比はΔa/2。物理量の差はΔa×[12500000,750000,375500]。物理単位の違う差をそのままL2で合成しない。最終2点だけでなく128候補間の変化と4080個のActor更新前後出力を保存し、更新差の和と評価間の正味変化がfloat32換算の1.2e-7以内で一致した。折返しのため経路長Σ||Δa_update||と正味||Δa||は異なる。各更新の音響値は未評価なので捏造しない。

更新前後のフックはActor optimizerの直前/直後で、当該stepのCritic・温度optimizerはすでに更新済み。2点では同じCriticを使って出力行動のQを照会する。logged actor_lossはSACが実際のbatch32の確率的行動で計算した値で、決定論的出力1点のQや実測音響Jと同じ値ではない。

![行動と物理差](figures/action_changes.png)

![実際のActor更新とloss](figures/actor_updates.png)

alpha、tanh後の座標別std、pre-tanh log_std、entropyを同じ評価時刻で記録した。std/entropyは局所NumPy seed220026の共通4096サンプルによる推定で、グローバルRNGを消費しない。連続分布のentropyは負になり得る。alphaはstd自体ではなく、上昇したから全座標の探索幅が増えると仮定しない。

後半17点（≥2048）でのSpearman相関。最後の列は前区間Δx→次区間ΔE（15組）であり、時間順序の記述にとどまる。水準と差分は標本数17/16。自己相関・小標本・多重比較・共通トレンドを考慮した因果検定ではない。全区間/全指標は[相関CSV](stage_a_correlations.csv)に保存した。

{table(['seed','x','同時水準ρ(x,E)','同区間ρ(Δx,ΔE)','前区間Δx→次区間ΔE'],ct)}

![温度・分散・entropy](figures/policy_histories.png)

## 中間Criticの実測と限界

固定候補は旧33評価×3seedの99行動。新しい各時点のQ1/Q2/minQを全て記録し、保存したActorの出力へのQも別に保存した。1step終端のためCritic targetは実測r=−Jで、次状態のbootstrap項は0。SAC Actorの最適化には現在方策の期待minQとentropyが残る。決定論的出力でminQが低くても、それだけでSACの更新則違反とは言えない。

以下は開始/終了の同じ2行動を、それぞれ開始/終了時点のCriticで評価した回顧比較。実測Δrと予測ΔminQが同符号なら、その2行動の順位は合う。正のΔrは真の改善、負は悪化。結果を見て候補を学習器に提示したものではなく、過去の99行動を学習前に固定して照会した。Q1/Q2の値は[イベントCSV](event_q_comparisons.csv)、全候補は[圧縮予測](critic_predictions.json.gz)、RMSE/バイアス/順位は[指標CSV](critic_metrics.csv)。

{q_table}

![Criticの時間変化](figures/critic_histories.png)

![seed2の4時点の同一候補](figures/seed2_q_snapshots.png)

固定候補は良い途中解を含む回顧集合で、探索領域の一様標本や独立な検証データではない。Qの局所誤順位やバイアスは観測できるが、それを取り除く介入は今回していない。Q1/Q2の差が小さいことも正確さの保証ではない。

## 独立評価とReplay経験・保持

独立評価33回/seedは学習用env/Replayへ追加しない。保存はActorとonline twin Criticの推論用state_dictと時刻/温度の小さなmetadataのみで、optimizer/Replay/target Criticは保存しない。再開学習用checkpointとは呼ばない。minJ/minE/Finalを別名で保持し、seed2の2944/3072/3840だけ追加診断重みを残した。結果は[checkpoint索引](checkpoint_index.json)、保持基準ごとの波形と重みは[runs](runs/)。

{table(['seed','学習4096サンプル中E≤.10','最初の試行','学習サンプル最小E','その試行','独立評価33点中E≤.10'],[[r['seed'],r['training_E_le_0p10'],r['first_training_E_le_0p10'],f"{r['best_training_E']:.7f}",r['best_training_trial'],r['independent_policy_E_le_0p10']] for r in sampling])}

seed0でも2290試行の学習サンプルにE=.05935の良い行動が1個あり、Replayに入った。従って「良い制御値を一度も試せなかった」とは言えない。一方、決定論的評価方策の良い解への到達は33点中0回で、sampling成功とActorへの定着を分ける。seed1/2にも独立評価での発見より前に良い学習サンプルがあり、Replayに含まれるだけで確実に採用されるわけではない。[学習サンプルと方策評価の区別](replay_success.csv)。

評価された良い行動と学習サンプルの完全一致数、最寄りの正規化距離/全域幅、半径.005/.01/.05以内の経験数は[coverage CSV](candidate_coverage.csv)に記録した。近い経験があることと良い決定論的評価の経験がReplayに入ったことは別で、完全一致0も近傍の学習信号がないことを意味しない。保持された方策へ戻す、教師にする、優先する処理は0。

## 限定した局所感度

事前に決めたseed2の3行動・3座標・±(.002,.01)の36音だけを追加評価した。境界clipは0、失敗は0。物理刻みはε_l±(25000,125000)、p_s±(1500,7500)、f0_l±(751,3755)。同じ保存Criticで近傍のQも予測した。中心差分∂r/∂aと∂minQ/∂aは近傍スケール依存で、非線形性やtwin-minの折れ点のため、中心のautogradと必ずしも等しくない。

{local_table}

![局所感度36音](figures/local_sensitivity.png)

[全36測定](local_sensitivity.csv) / [中心差分・中心勾配](local_gradients.csv) / [元振幅float64波形](local_audio.npz)。3近傍だけの結果を空間全体の地形や大域的な最適探索幅へ一般化しない。探索・介入の性能改善をここで評価したものではない。

## 再現・監査・計算量

Stage AのREPORTを保存してから再現を開始した（STAGE_Aのhashをmanifestと各prefixへ記録）。初期policy/実効設定/評価が一致し、最初の128候補で一致を確認後、全4096候補まで照合した。学習前だけ未使用のOS初期化training_env_rngは旧runとの比較から除外し、不一致を明記した。SB3の学習開始resetでseedが適用され、128以降は除外せず全署名が一致。初期点も含む評価/ロギング/保存の前後署名は全キーで一致した。

3runの壁時間は{execution['wall_s']:.1f}s。研究計算は学習12288音、独立評価99音、Stage Aの旧モデル6音、Stage B再生{audit['generated_sounds']}音、局所36音、計12437音。各0.20sのnative2,000,000stepで計24,874,000,000完了step。研究上の失敗・非有限値は0、学習のActor/Critic/温度更新は各12240。Q推論と描画に鳴管計算はない。主図は{len(images)}枚。

新規4テストと共有11テストの計15テストが成功。テスト時のモデル評価は135回（うち注入失敗1件）で、研究runとは別。初回の短いテストで初期未使用環境RNGの署名差を検出し、比較方法を修正したため、32候補のSACテストを2回実行した（64候補）。共有テストには既存24候補×TD3/SACの接続検査も含むが、TD3の研究実験・条件比較は追加していない。[テスト記録](test_execution.json)。

既存{protected}ファイルはhash一致で変更・削除0。[Stage A監査](stage_a_audit.json) / [Stage B監査](audit.json) / [実行数](execution.json) / [元ファイル保護](integrity.json) / [全成果物hash](artifact_checksums.json) / [実装hash](implementation_checksums.json)。root READMEと.gitignoreだけに索引/作業ログ除外を追加した。commit/push/Issue更新は行っていない。

聴取用は[WAV ZIP](listening.zip)の{wavs}音（同一音は重複除去、9endpointへの[索引](listening_index.csv)あり）。全音にTarget peakの逆数という共通gainを掛けたfloat32、個別正規化やクリップは0。float64元波形は各run NPZで保持する。再現手順は[REPRODUCE.md](REPRODUCE.md)。
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'README.md').write_text('''# Issue23 — SAC128・λ=.75の安定性診断

[REPORT](REPORT.md) / [学習前のStage A](STAGE_A.md) / [再現手順](REPRODUCE.md) / [方策索引](checkpoint_index.json)

旧Issue22の3seedを設定変更なしで再現し、中間Critic・Actor更新を診断した。Final、minJのSelected、minE_totalのposthocを別々に保持する。保存による利得と学習改善を区別する。過去成果物は読み取り専用。
''')
    (OUT/'REPRODUCE.md').write_text('''# 再現と保存成果の利用

リポジトリrootで環境を指定する。元成果を保持するため同名のprepareは拒否し、完了runはhashが同じ場合だけ再利用する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue23-mpl
# 着手前のworkspace/git/既存全path・bytes・sha256 JSONを--beforeへ渡す
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_stage_a --before /tmp/birdsong-issue23-before.json
# REPORTのStage A・必要理由・事前固定した追加記録を確認した後のみ
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_training --workers 3
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis local
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue23_analysis integrity
```

新規出力を持つ再現は別checkoutでissue23_data.OUTを専用の新規先へ設定し、着手前JSONを作る。既存正本を上書きしない。学習source/native/target/Stage A/候補hashを固定し、条件変更時は新規studyにする。

既存結果の図/表はissue23_reportで再生成する（鳴管計算/学習0）。criticsは保存Qの集計だけ。localは完了hashがあれば36音を再計算しない。auditは保存重みの再ロードと8個の異なる行動の音を再生するため、実施回数を別に記録する。integrityは音を生成せず旧ファイル/hash/linkを検証する。

保存した.ptはactorとonline twin criticだけの推論用重みで、optimizer/Replay/target Criticを含まず、再開学習はできない。issue23_data.load_checkpoint(path,seed,manifest['settings'])で同じ構造を作って重みをロードし、agent.predict(obs,deterministic=True)で実行できる。posthoc.ptは共通Eによる事後選別で、selected.ptのminJとは別。

全33評価はcurve.json.gz、全4096学習サンプルはtraining_trace.json.gz、実際の4080 Actor更新の前後出力とlossはupdates.json.gz、99固定候補への33時点のQはq_history.json.gz。1更新ごとの音響損失を新たに計算したログではない。actionは正規化[-1,1]、f0_lはHzでなく開閉外力係数。

聴取WAVは全音共通のTarget peak逆数gain、元振幅float64はNPZで保持する。固定MC seed220026・4096標準正規サンプルでstd/entropyを求め、学習RNGには触れない。
''')
    write_json(OUT/'report_execution.json',dict(generated_sounds=0,learning_updates=0,images=7,
        source_sha256=digest(__file__),findings_sha256=digest(ROOT/'issue23_findings.md'),wall_s=time.perf_counter()-tick))
    print('Seven figures and diagnostic report generated; freezing hashes and final inventory',flush=True)
    freeze()


if __name__=='__main__':report()
