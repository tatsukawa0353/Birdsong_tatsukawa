"""Render Issue26 and freeze local provenance, protected files and storage."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from .issue26_data import ROOT,OUT,ready,target_data
from .issue26_analysis import aggregate,listening,csv_rows
from .issue23_stage_a import table as plain_table
from .issue19_protection import workspace_inventory
from .issue15_analysis import write_json
from .issue16 import digest


def table(headers,rows):
    esc=lambda v:str(v).replace('|','\\|')
    return plain_table([esc(h) for h in headers],[[esc(v) for v in row] for row in rows])


def integrity():
    ready();path=OUT/'integrity.json'
    if not path.exists():write_json(path,dict(status='running'))
    protected=csv_rows('protected_before.csv');changed=[];deleted=[];broken=[];links=0
    for r in protected:
        p=ROOT/r['path']
        if not p.exists():deleted.append(r['path'])
        elif digest(p)!=r['sha256']:changed.append(r['path'])
    for p in (OUT/'README.md',OUT/'REPORT.md',OUT/'PLAN.md',OUT/'STAGE_B.md',OUT/'REPRODUCE.md'):
        if not p.exists():continue
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            links+=1
            if not (p.parent/dest.split('#')[0]).exists():broken.append(dict(file=p.name,link=dest))
    for name,root in (('artifact_checksums.json',OUT),('implementation_checksums.json',ROOT)):
        if (OUT/name).exists():
            checks=json.loads((OUT/name).read_text())
            for p,h in checks.items():assert digest(root/p)==h,p
            if name=='implementation_checksums.json':
                expected={str(p.relative_to(ROOT)) for p in [*ROOT.glob('issue26*.py'),ROOT/'issue26_findings.md',ROOT/'tests/test_issue26.py']}
                assert set(checks)==expected,'Incomplete implementation manifest'
    wavs=0
    if (OUT/'listening_wav.zip').exists():
        gain=1/float(np.max(abs(target_data()[0])))
        with zipfile.ZipFile(OUT/'listening_wav.zip') as archive:
            for r in csv_rows('wav_index.csv'):
                rate,wave=wavfile.read(io.BytesIO(archive.read(r['wav'])))
                assert rate==50000 and wave.dtype==np.float32 and wave.shape==(10000,) and float(r['gain'])==gain
                with np.load(ROOT/r['source']) as z:np.testing.assert_array_equal(wave,(z[r['raw_key']]*gain).astype(np.float32))
                wavs+=1
    assert json.loads((OUT/'audit.json').read_text())['passed']
    assert json.loads((OUT/'aggregate_execution.json').read_text())['source_sha256']==digest(ROOT/'issue26_analysis.py')
    audited=json.loads((OUT/'audit.json').read_text())
    if audited['source_sha256']!=digest(ROOT/'issue26_analysis.py'):
        import ast
        original=OUT/audited['executed_source_snapshot'];assert digest(original)==audited['source_sha256']
        def without_aggregate(path):
            tree=ast.parse(path.read_text());tree.body=[node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name!='aggregate']
            return ast.dump(tree,include_attributes=False)
        assert without_aggregate(original)==without_aggregate(ROOT/'issue26_analysis.py'),'Audit implementation changed'
    if (OUT/'report_execution.json').exists():assert json.loads((OUT/'report_execution.json').read_text())['source_sha256']==digest(__file__)
    result=dict(passed=not changed and not deleted and not broken,original_files_checked=len(protected),original_files_changed=changed,
        original_files_deleted=deleted,local_links_checked=links,broken_links=broken,wavs_checked=wavs,source_sha256=digest(__file__))
    write_json(path,result);assert result['passed'],result;return result


def freeze():
    checks=OUT/'artifact_checksums.json';write_json(checks,{})
    write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in
        [*sorted(ROOT.glob('issue26*.py')),ROOT/'issue26_findings.md',ROOT/'tests/test_issue26.py']})
    integrity();write_json(OUT/'storage_after.json',{})
    repository=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    for _ in range(5):
        status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True)
        new=[line[3:] for line in status.splitlines() if line.startswith('?? ')]
        record=dict(workspace=workspace_inventory(),new_untracked_files=len(new),new_untracked_bytes=sum((repository/p).stat().st_size for p in new),
            modified_tracked_files=sum(line.startswith(' M ') for line in status.splitlines()),deleted_tracked_files=sum('D' in line[:2] for line in status.splitlines()),
            measure='Logical regular-file bytes, ignored environments/logs included; .git/symlinks excluded; nothing staged/committed')
        write_json(OUT/'storage_after.json',record)
        before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total'];after=record['workspace']['total']
        block='<!-- ISSUE26_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[
            ['通常ファイル数',before['files'],after['files']],['論理容量（ignored含む）',f"{before['bytes']/1e6:.2f} MB",f"{after['bytes']/1e6:.2f} MB"],
            ['Git新規追加候補','0（クリーン）',f"{record['new_untracked_files']}個 / {record['new_untracked_bytes']/1e6:.2f} MB"],
            ['既存追跡ファイル変更 / 削除','0 / 0',f"{record['modified_tracked_files']} / {record['deleted_tracked_files']}"]])+'\n<!-- ISSUE26_STORAGE_END -->'
        p=OUT/'REPORT.md';body=p.read_text();p.write_text(re.sub(r'<!-- ISSUE26_STORAGE_BEGIN -->.*?<!-- ISSUE26_STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- ISSUE26_STORAGE_BEGIN -->' in body else body+'\n'+block+'\n')
        write_json(checks,{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    record['workspace']=workspace_inventory();write_json(OUT/'storage_after.json',record)


def report():
    ready();tick=time.perf_counter();aggregate();wavs=listening()
    from .issue26_figures import render
    images=render();groups=csv_rows('summary_by_condition.csv');ends=csv_rows('endpoints.csv');arr=csv_rows('arrivals_and_retention.csv')
    dist=csv_rows('candidate_distribution.csv');sampling=csv_rows('local_sampling_summary.csv')
    execution=json.loads((OUT/'execution.json').read_text());audit=json.loads((OUT/'audit.json').read_text());tests=json.loads((OUT/'test_execution.json').read_text())
    fmt=lambda r,k:f"{float(r[k+'_mean']):.5f} ± {float(r[k+'_sd']):.5f}" if r.get(k+'_mean') else '未定義（失敗seed含む）'
    gt=table(['条件','採用','E_total平均±SD','E_spec平均±SD','E_env平均±SD','J平均±SD','持続/3'],
        [[r['condition'],r['role'],*[fmt(r,k) for k in ('E_total','E_spec','E_env','J')],r.get('voiced_seeds','欠測')] for r in groups])
    et=table(['条件','seed','採用','試行','E_total','J','持続','ε_l','p_s','f0_l'],
        [[r['condition'],r['seed'],r['role'],r['trial'],f"{float(r['E_total']):.6f}",f"{float(r['J']):.6f}",r['voiced'],*[f"{float(r[k]):.0f}" for k in ('epsilon_l','ps','f0_l')]] for r in ends])
    at=table(['条件','seed','prefix256 minJ','学習J半減の初回','Actor E初期比半減の初回','Actor J≤prefix×1.25初回','Final J / prefix','Selected−Final E'],
        [[r['condition'],r['seed'],f"{float(r['prefix256_best_J']):.6f}",r['first_training_half_prefix_J'] or '未到達',r['first_actor_half_initial_E'] or '未到達',
            r['first_actor_near_prefix_J'] or '未到達',f"{float(r['final_J_to_prefix_J']):.3f}",f"{-float(r['final_minus_selected_E']):.6f}"] for r in arr])
    dt=table(['条件','seed','発声数/4096','f0中央値','>800k 訪問数/発声数','>2.5M 訪問数/発声数','>5M 訪問数/発声数'],
        [[r['condition'],r['seed'],r['voiced'],f"{float(r['f0_q50']):.0f}",*[f"{r['above_'+str(x)+'_count']}/{r['above_'+str(x)+'_voiced']}" for x in (800000,2500000,5000000)]] for r in dist])
    lt=table(['条件','seed','local件数','257以降のlocal率','再抽選数','連続f0差>局所半径','最長中心停滞'],
        [[r['condition'],r['seed'],r['local_count'],f"{float(r['local_fraction_after256']):.3f}",r['rounding_redraws'],r['consecutive_f0_steps_over_radius'],r['longest_incumbent_plateau']] for r in sampling])
    sounds=execution['training_candidates']+execution['independent_policy_evaluations'];total=sounds+audit['generated_sounds']+tests['native_sound_evaluations']
    findings=(ROOT/'issue26_findings.md').read_text()
    body=f'''# Issue26 — Case A：f0広域探索と連続局所候補

## 結果と判断

{findings}

新規9runは完了{execution['complete_runs']}・失敗{execution['failed_runs']}。[通常SACを先に実行した記録](STAGE_B.md)の後、C/D全6runを実施した。E_total≤.10は目標・合否・追加計算のゲートに使っていない。3seed・同一Target・固定予算の範囲での比較である。

## 比較条件と固定要素

| 条件 | f0_lの全体範囲 | 257試行以降 | f0局所半径 |
|---|---|---|---|
| A | [49000,800000] | 通常SAC、Issue23読み取り専用・新規0本 | — |
| B | [49000,10000000] | 通常SAC | — |
| C | Bと同じ | local80% / 通常SAC20% | ±50000 |
| D | Bと同じ | local80% / 通常SAC20% | ±250000 |

A/Bは範囲、B対C/Dはoff-policy行動データ生成アルゴリズム、C/Dはf0局所半径だけの比較。localのε_l半径±625000、p_s半径±37500は共通。通常SAC20%は全域の候補生成可能性を残すが、一様被覆は保証しない。局所幅は格子刻み・隣接試行の変化上限・波形内の変化速度ではない。float32正規化に伴う有限精度はある。

Case A（右音響寄与α_r=β_r=0）、一定3D、1episode1行動、0.20s/50kHz/10000sample、native dt1e-7、x0=.02cm/D=.01、初期状態/遅延0。同一保存Target、ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、右ε_r3.5e7/f0_r49000を固定。f0_lはHzではなく開閉外力係数。p_s/εを実測Pa/張力と同一視しない。

r=−J、J=.75E_spec+.25E_env、係数和1・2倍補正なし。共通E_total=E_spec+E_env。Actor/Critic[128,128]、3optimizer LR3e-4、auto_0.01/target_entropy−3、batch32/buffer50000/warmup16/gamma1/train_freq1/gradient_steps1、tau/target更新/CPU1threadを維持。[凍結計画](PLAN.md) / [manifest](manifest.json) / [実効設定](runs/B/seed_0/config.json)。

## 候補・Replay・乱数の監査

最初256試行は全B/C/Dで標準SAC。warmupは16のまま、17試行目から通常更新する。事前の3seed×標準SAC/B/C/Dで行動・報酬・全重み・optimizer・Replay・RNGが一致。正式C/DもBに対して最初256件と128/256時点の全学習器署名が一致した。[事前試験](test_execution.json) / [正式prefix照合](prefix_checks.csv)。

毎試行、標準SAC提案を先に生成し、local時だけ実行行動とReplay行動の両方を置換。未実行SAC提案は別欄に記録する。中心は過去の実学習候補の最初minJで、失敗/非持続も除外せず同点は先着。独立Actor評価とTargetの既知制御は中心に使わない。各座標の[中心±半径]∩全体範囲で連続一様抽選し、クリップしない。正規化float32の丸めで区間を出たときだけ物理計算前に再抽選する。

選択PCG64 seed260026+seed、候補PCG64 seed260126+seedを分離し、学習Python/NumPy/Torch RNGを消費しない。最初256は両RNG未消費、後半C/Dは同じ選択系列。33独立評価・ログ・保存の前後で学習器/Replay/両local RNG/中心が不変。実行行動・適用物理値・Replay行動/報酬を新規36864件で検査し、全local提案と中心更新を保存ログから再構成した。[sampler監査](sampler_checks.csv) / [run会計](run_accounting.csv) / [完全ログ例](runs/C/seed_0/training_trace.json.gz)。

全更新前後でLR・勾配・重み・Adam状態を観測。非有限、負の二次moment、勾配L2>1e6、パラメータ絶対値>1e6、moment絶対値>1e12で停止し、失敗runを保持する。クリップや収束判定ではない。停滞時の再始動、幅調整、oracle初期化、追加最適化は行わない。

## Final・保存方策・学習候補

Final=4096終了後のActor。Selected=0,128,...,4096の33独立評価の最初minJ。posthoc=同33点の最初min共通E。training_best=4096実学習候補の最初minJで、Actor checkpointではない。保存方策へ復帰しない。3seed平均±標本SD（ddof=1）を示す。失敗seedを含む3seed Final平均は未定義とし、完了seedだけの平均へ置換しない。

{gt}

![4誤差・全seed・同じ軸](figures/error_histories.png)

![4種類のendpointを分離](figures/endpoints.png)

持続proxyは末尾1000sampleのAC標準偏差/Target peak>.05かつ数値失敗なしで、生理的有声判定ではない。[全成分・全制御](endpoints.csv) / [対応seed差](paired_differences.csv)。

{et}

## 発見・Actor獲得・Final保持

結果前に相対到達基準を固定した。(1)257以降の学習J≤prefix256 minJ×.5、(2)Actor E≤自身の初期Actor E×.5、(3)256以降のActor J≤自身のprefix256 minJ×1.25。Aは自身の既存prefix、B/C/Dはseed内で同一prefix。いずれも記述指標で合否・追加計算のゲートではない。「未到達」は4096候補/33観測の範囲で未観測。

prefix自体が悪いseedでは基準(3)も緩いため、これを満たすことと高精度模倣の獲得は同じではない。A/Bでは初期Eとprefixも異なる。C/D/Bは同一seedで同じprefixを使うが、絶対的な音響誤差と発声を併記して判断する。

{at}

![学習候補の発見とActorの獲得](figures/discovery_and_acquisition.png)

Selected−Final Eが負ならminJ保存ActorよりFinalの共通Eが悪い。J/Eの基準差を残すためJ比も[到達・保持CSV](arrivals_and_retention.csv)へ保存。良い学習候補からActorの獲得を推論せず、33点の間の未観測方策に到達時刻を割り当てない。

## 訪問領域と局所候補

{dt}

f0の固定区間[49k,800k]、(800k,2.5M]、(2.5M,5M]、(5M,10M]で訪問数と条件付き発声率を集計。ε_l/p_sとの同時訪問を示す。方策に偏った標本であり一様な物理領域調査ではない。高f0を一括して無音/不要と断定しない。[分布・分位点](candidate_distribution.csv) / [区間集計](f0_bins.csv) / [全候補](training_candidates.csv.gz)。

![f0とε・p_sの同時訪問](figures/visited_joint_controls.png)

![訪問分布・条件付き発声・候補源別改善率](figures/coverage_voice_and_hits.png)

{lt}

localの期待率は257以降80%・全4096で75%。hitは直前incumbent Jを厳密に改善した件数で、異なる候補源の分布を調整した因果効果ではない。半径に対する実偏差、境界区間の短縮、丸め、連続試行差、中心停滞を保持する。[候補源別](behavior_sources.csv) / [局所要約](local_sampling_summary.csv) / [全局所偏差](local_displacements.csv.gz) / [中心履歴](center_history.csv)。

![局所実偏差と中心履歴](figures/local_displacement_and_centers.png)

## Actorと探索尺度

![Actorの物理制御値](figures/actor_controls.png)

![温度・正規化std・物理std](figures/policy_exploration.png)

33点でalpha、潜在mean/std、tanh後std、entropyを保存。tanh後統計は同じlocal seed220026・MC4096で推定し、学習RNGを使わない。alphaとstdは別量で、微分entropyは負にもなる。f0正規化半幅はA375500・B/C/D4975500なので、同じ正規化stdでも物理stdは異なる。local候補分布とActor stdも別量。[方策推移](policy_dynamics.csv)。

全4080 Actor更新/runの前後の正規化/物理移動量と実際のbatch lossを保存し、128試行区間の経路長と正味移動を分離した。異なる単位の物理差を直接L2合成しない。[全更新](actor_updates.csv.gz) / [区間統計](update_intervals.csv)。

## 音・独立再生・計算量

seed1は[新規結果前の表示計画](display_plan.json)で代表例に固定。FinalとSelectedを別図にし、全seedのfloat64生波形/制御は[sound index](sound_index.csv)で参照する。共通gain=1/Target peakのfloat32 [WAV一式](listening_wav.zip)と[index](wav_index.csv)を保存。個別正規化・クリップ・位相合わせはない。振幅1超もfloat値を保持する。

![Finalの共通基準表示](figures/audio_final.png)

![Selectedの共通基準表示](figures/audio_selected.png)

STFTは表示用10MHz再サンプリング、Blackman-Harris245760/overlap184320、完全窓のみ、0–10kHz。Target全帯域PSDを唯一の色基準にし、全endpoint共通の色/波形/RMS軸を使用。表示再サンプリングによる新情報はない。波形のみ40–60ms、5ms RMSは全0.20秒。[表示設定](display_settings.json)。

保存Actor39本（4条件×3seed×Final/Selected/posthoc＋B prefix256の3本）と学習best行動12件を独立再生し、行動/物理制御/波形/Jが完全一致。Aの学習best音は本成果物内へ新規保存した。[再生チェック](model_replay_checks.csv) / [監査](audit.json)。

| 計算区分 | 新規件数 |
|---|---:|
| フル学習 | 9×4096 = {execution['training_candidates']}候補 |
| 独立Actor評価 | 9×33 = {execution['independent_policy_evaluations']}音 |
| optimizer更新 | 各Actor/Critic/温度 {execution['gradient_updates']}回 |
| 研究runの生成音 | {sounds}音 |
| 独立再生監査 | {audit['generated_sounds']}音 |
| 事前native試験・限定再検査 | {tests['native_sound_evaluations']}音 |
| 今回native合計 | {total}音、{total*2000000:,} steps（全件正常時） |
| synthetic試験 | {tests['synthetic_terminal_episodes']}候補、native0 |
| 純局所抽選試験 | {tests['pure_local_candidate_draws']}件、native0 |
| Target再生成・追加グリッド・追加最適化 | 0 |

native block数・積分時間・Replay件数は各summaryと[会計](run_accounting.csv)に保存。[wide](wide_execution.json) / [local](local_execution.json) / [全実行](execution.json)。事前試験のNumPy許容誤差表示の不備はテスト内で修正し、限定再検査分を上の件数に含めた。研究設定は変えていない。

## 来歴・保存と次の作業

Issue18/22/23/25のTargetと物理/音響コードを検査。Issue18の当時のlibrary hashはx0格納値.04→.02の変更前なので現行と異なる。既存Issue19 gateで格納のみ・保存波形一致が確認されており、旧manifest更新や旧音再生成はしていない。[参照監査](reference_audit.json)。

再生監査の完了後、CSV集計の列参照名だけを修正した。監査時の[実行ソース](sources/issue26_analysis_at_audit.py)を保持し、集計関数以外の構文木が現行と同一であることを保護検査で確認する。学習・再生監査を再実行せず、追加native音は0件。

既存6044ファイルはhash一致・変更/削除0。root READMEの索引と.gitignoreの作業ログ除外のみ変更した。[保護検査](integrity.json) / [実装hash](implementation_checksums.json) / [成果物hash](artifact_checksums.json) / [再現手順](REPRODUCE.md)。commit/push、Issueコメント/close、教員への送信はしていない。

Case A全体の総括・教員共有、Case B自身のTarget生成とIssue20旧共通Case A Target仕様の改訂は次の作業。今回はCase B、時変、実録音、別reward/LR/entropy、追加の中心救済を実行していない。
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'README.md').write_text('''# Issue26 — 広域SACと連続局所探索

[結果・判断と全比較](REPORT.md) / [凍結計画](PLAN.md) / [先行Bの結果](STAGE_B.md) / [再現](REPRODUCE.md)

A=既存の制限域、B=広域通常SAC、C/D=広域local80%・SAC20%（f0半径±50000/±250000）。新規9run×4096、33独立評価、最初256一致。発見・Actor獲得・Final保持を分け、E≤.10のゲートを使わない。

[集計](summary_by_condition.csv) / [全endpoint](endpoints.csv) / [相対到達と保持](arrivals_and_retention.csv) / [訪問・発声](candidate_distribution.csv) / [局所候補](local_sampling_summary.csv) / [方策推移](policy_dynamics.csv)

[生波形索引](sound_index.csv) / [共通gain WAV](listening_wav.zip) / [再生監査](audit.json) / [run会計](run_accounting.csv) / [元成果保護](integrity.json) / [成果物hash](artifact_checksums.json)
''')
    (OUT/'REPRODUCE.md').write_text('''# Issue26の再現

リポジトリrootで実行する。新規研究は新しいcheckoutでissue26_data.OUTを未使用の専用pathに設定し、着手前snapshotを新規保存する。prepareは既存OUTを拒否し、完了runは同じソース/hashのときだけ再利用する。途中失敗/既存成果を上書きしない。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue26-mpl
# 着手前snapshot。既存snapshot/既存OUTを拒否
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_inventory
# 6試験とrecord生成。native119音を伴う
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue26
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_prepare
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_training wide --workers 3
# STAGE_B.md確認後、Bの結果にかかわらず全C/Dを実行
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_training local --workers 6
# 51音を独立再生。再実行した場合は追加回数を別記
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_analysis audit
# 保存データから図/表/WAV/報告を更新。native0
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_report
# hash/保護/link/WAV検査のみ。native0
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue26_report --integrity
```

manifestに環境/実効設定/source/native/Target/PLANのhashを保存。checkpointはActor+online Critic推論用でoptimizer/Replay/local RNGを含む再開用ではない。全候補・33評価点・4080更新をlossless gzip保存。凍結済み成果を検証するときは、図や表を再生成する前にintegrityを実行する。
''')
    write_json(OUT/'report_execution.json',dict(generated_sounds=0,source_sha256=digest(__file__),figures=len(images),wav_entries=len(wavs),wall_s=time.perf_counter()-tick))
    print('Report, ten figures, tables and WAV complete; freezing integrity and storage',flush=True);freeze()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--integrity',action='store_true');a=p.parse_args();integrity() if a.integrity else report()
