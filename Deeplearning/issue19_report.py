"""Per-seed/paired results, eight figures and a decision for the next phase."""
import csv
import gzip
import json
import time

import numpy as np

from .issue19_data import (ROOT, BASE, OUT, METHODS, WEIGHTS, folder_for, curve_for,
                          endpoints, target_data)
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def report():
    tick=time.perf_counter();target,controls=target_data();rows=[];groups=[];deltas=[]
    audit_total=json.loads((OUT/'audit.json').read_text()).get('cumulative_audit_generated_sounds',48)
    for weight in WEIGHTS:
        for method in METHODS:
            for seed in (0,1,2):
                for role,p in endpoints(method,seed,weight).items():
                    comp=p['acoustic_components'];c=np.array(p['physical_controls']);diff=c-controls
                    row=dict(method=method,seed=seed,weight=weight,role=role,trial=p['episodes'],
                        E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],
                        sustained_voiced=p['sustained_voiced'],failed=p['failed'],
                        epsilon_l=c[0],ps=c[2],f0_l=c[3],epsilon_l_error=diff[0],ps_error=diff[2],f0_l_error=diff[3],
                        parameter_nrmse_original=p['parameter_nrmse'],
                        parameter_nrmse_restricted=float(np.sqrt(np.mean((diff[[0,2,3]]/[2.5e7,1.5e6,751000.])**2))))
                    rows.append(row)
    fields=('E_total','E_spec','E_env','J','parameter_nrmse_restricted')
    for method in METHODS:
        for weight in WEIGHTS:
            for role in ('final','selected'):
                subset=[r for r in rows if r['method']==method and r['weight']==weight and r['role']==role]
                group=dict(method=method,weight=weight,role=role,n=3,voiced=sum(r['sustained_voiced'] for r in subset))
                for field in fields:
                    values=np.array([r[field] for r in subset]);group[field+'_mean']=float(values.mean());group[field+'_sd']=float(values.std(ddof=1))
                groups.append(group)
        for role in ('final','selected'):
            for seed in (0,1,2):
                a=next(r for r in rows if r['method']==method and r['role']==role and r['seed']==seed and r['weight']==1.)
                b=next(r for r in rows if r['method']==method and r['role']==role and r['seed']==seed and r['weight']==.5)
                deltas.append(dict(method=method,role=role,seed=seed,**{'delta_'+f:b[f]-a[f] for f in fields},
                    voiced_lambda1=a['sustained_voiced'],voiced_lambda0p5=b['sustained_voiced']))
    write_csv(OUT/'comparison_per_seed.csv',rows);write_csv(OUT/'comparison_aggregate.csv',groups)
    write_csv(OUT/'paired_lambda_deltas.csv',deltas)
    # CSV access to every new seed's samples; JSON archives retain exact nested
    # metrics and independent-evaluation state signatures as the canonical log.
    columns=['method','seed','trial','role','E_total','E_spec','E_env','J','sustained_voiced','action','physical_controls']
    with gzip.open(OUT/'new_candidate_metrics.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for method in ('TD3','SAC','DE'):
            for seed in (0,1,2):
                folder=folder_for(method,seed,.5)
                for p in read_record(folder/('training_trace.json' if method!='DE' else 'curve.json')):
                    comp=p['acoustic_components'];action=p['action'] if 'action' in p else p['actions'][0]
                    writer.writerow(dict(method=method,seed=seed,trial=p['episodes'],role='training_sample',
                        E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],
                        sustained_voiced=p['sustained_voiced'],action=json.dumps(action),physical_controls=json.dumps(p['physical_controls'])))
    reuse=[]
    for seed in (0,1,2):
        source=folder_for('random_search',seed,1.)
        reuse.append(dict(seed=seed,source_curve=str((source/'curve.json.gz').relative_to(ROOT)),
            source_sha256=digest(source/'curve.json.gz'),candidates_reused=4096,candidates_drawn=0,candidate_sounds_generated=0,
            selected_trial=endpoints('random_search',seed,.5)['selected']['episodes']))
    write_json(OUT/'random_reuse.json',reuse)
    from .issue19_figures import render_figures
    images=render_figures(rows,deltas)
    from .issue19_storage import review,after
    cleanup=review()
    with (OUT/'critic_metrics_all.csv').open() as stream:cm=list(csv.DictReader(stream))
    with (OUT/'actor_diagnostics_all.csv').open() as stream:actors=list(csv.DictReader(stream))
    critics=[]
    for method in ('TD3','SAC'):
        for weight in WEIGHTS:
            for name in ('Q1','Q2','minQ'):
                part=[r for r in cm if r['method']==method and float(r['weight'])==weight and r['critic']==name and r['subset']=='fixed_pool']
                low=[r for r in cm if r['method']==method and float(r['weight'])==weight and r['critic']==name and r['subset']=='low_J_decile']
                fmt=lambda field:f"{np.mean([float(r[field]) for r in part]):.4f}"
                critics.append([method,weight,name,fmt('rmse'),fmt('spearman'),fmt('pairwise_inversion_fraction'),
                                f"{np.mean([float(r['bias_Q_minus_r']) for r in low]):.4f}"])
    fmt=lambda r,f:f"{r[f+'_mean']:.5f} ± {r[f+'_sd']:.5f}"
    aggregate=table(['手法','λ','採用','E_total 平均±SD','E_spec 平均±SD','E_env 平均±SD','持続/3'],
        [[r['method'],r['weight'],r['role'],*[fmt(r,f) for f in ('E_total','E_spec','E_env')],r['voiced']] for r in groups])
    delta_summary=[]
    for method in METHODS:
        for role in ('final','selected'):
            part=[r for r in deltas if r['method']==method and r['role']==role]
            values=[np.array([r['delta_'+f] for r in part]) for f in ('E_total','E_spec','E_env')]
            delta_summary.append([method,role,*[f'{v.mean():+.5f} ± {v.std(ddof=1):.5f}' for v in values]])
    individual=table(['手法','λ','seed','採用','試行','E_total','E_spec','E_env','Jλ','持続'],
        [[r['method'],r['weight'],r['seed'],r['role'],r['trial'],*[f'{r[f]:.7g}' for f in ('E_total','E_spec','E_env','J')],r['sustained_voiced']] for r in rows])
    actor_table=table(['手法','λ','seed','Actor J','候補内best J (oracle除外)','そのQ順位/1005','pool max Q − Actor Q'],
        [[r['method'],r['weight'],r['seed'],f"{float(r['actor_J']):.5f}",f"{float(r['best_nonoracle_J']):.5g}",r['best_nonoracle_rank_by_actor_Q'],f"{float(r['pool_max_actor_objective_Q_gap']):+.5f}"] for r in actors])
    text=f'''# Issue19 — 制限3Dの重み比較とActor/Critic診断

Case A・0.20秒・一定3D。新規学習はλ=.5のTD3/SAC/DE×3 seed、各4096候補だけ。λ=1のIssue18全12 runは読み取り専用。Randomは既存4096候補/seedを再スコアしてminJを再選択した。報酬重み以外の物理条件・学習設定を維持し、初期方策・warmup16行動・DE初期人口8も対応seedで一致した。

## 固定条件とモデル整合

目標制御は `[4.1e7,3.5e7,3.4e6,3e5,4.9e4]`、初期状態・遅延履歴は0、Case Aは右側音響出力を無効化。可制御値は `ε_l,ps,f0_l`、範囲は `[2.5e7,5e7]`, `[2.5e6,4e6]`, `[49000,800000]`。f0_lは開閉外力係数でありHzではない。この制限は持続発声模倣の診断用で、将来のCase Bの閉鎖側 `f_e=1e7` を排除しない。

研究設定 `x0=.02 cm,D=.01` をC++の実際のreset代入で検証。x0は構造体に保存されるだけで、展開済みDが運動式に入る。修正前 `.04 cm` / 現行 `.02 cm` の別ビルドと実際の実行ビルドで目標および4手法の保存Final計15音を比較し、全サンプル完全一致。旧manifestのnative source/library hashが現行と異なる理由を[モデル監査](model_gate.json)に残し、旧manifestや旧監査を更新していない。旧CLIの厳密ハッシュ検査を迂回して旧結果を書き直すこともしない。

学習率3e-4、MLP[32,32]、batch32、warmup16、gamma1、buffer50000、毎step1更新、CPU1thread/worker。TD3 noise=.1、policy_delay2、SAC auto_0.01。DE人口8、mutation[.5,1]、recombination.7。4096後に4080回のCritic更新、TD3 Actor2040回、SAC Actor4080回。途中checkpointの保存を省いたことは数値条件の変更ではない。[全条件とコードhash](manifest.json)。

## 主結果

主指標は全λ共通の `E_total=E_spec+E_env`。`Jλ=E_spec+λ E_env` は同一λ内の学習と選択に用いる。異なるλのJを直接比較して改善としない。平均±SDは3seedの標本SD（ddof=1）。

{aggregate}

FinalはRLの4096候補（4080回のCritic更新）終了後の決定論的方策、直接法は実際の候補4096。SelectedはRLの33回（0,128,...,4096）の独立評価または直接法4096候補の最初のminJ。直接法の最適化成果はSelectedで判断する。観測機会は33対4096で異なる。評価は探索ノイズなし、学習/RNGを変えないことを評価前後の署名で確認。SACの決定論的Finalと学習時の確率的行動は異なる。

同じseedのλ=.5−1（負なら改善）。λ間で軌跡が変わるため、共通の初期条件と乱数seedによる対応比較であり、全学習候補が同じという意味ではない。Randomだけは全4096候補が同一。

{table(['手法','採用','ΔE_total 平均±SD','ΔE_spec 平均±SD','ΔE_env 平均±SD'],delta_summary)}

[全seed](comparison_per_seed.csv) / [対応seed差](paired_lambda_deltas.csv) / [全新規学習候補のCSV](new_candidate_metrics.csv.gz)。物理値とその誤差も別列に保存。`parameter_nrmse_original` は元の全域幅を使う従来値、`parameter_nrmse_restricted` は今回の3D制限幅で再計算した値。音響誤差の小ささから物理制御値の一意復元は主張できない。

![4手法比較曲線](figures/learning.png)

![誤差内訳](figures/errors.png)

![対応seed差](figures/lambda_deltas.png)

![全物理制御](figures/controls.png)

## CriticとActor

固定初期観測は `[-1,0,0,0,0,0,0,0]`。状態のVecNormalizeはない。行動は[-1,1]の3Dで、policy.scale_actionを適用（数式上は恒等、float32の丸めは最大1.2e-7）。物理写像は今回の制限幅。既存候補の実測損失は元のfloat64行動のもので、Critic推論はfloat32。この精度差を新しい測定として扱っていない。

本課題は1行動でterminated=Trueのため、SB3実装のTD targetでは `(1-done)=0`。TD3/SACのQ1/Q2は共に `r=-Jλ` を回帰する。TD3のbootstrapはtwin targetのminだがこの課題では消え、ActorはQ1のみを最大化する。SACも次状態のminQ/entropy項は終端で消える一方、Actor学習には **現在行動のminQとentropy** が残り、確率的方策を最適化する。従って決定論的Actorが診断候補のminQ最大点と一致しないことだけではActor学習の欠陥を証明しない。使用中のSB3 train実装を[実装記録](critic_execution_all.json)に保存。

診断プールは**既存の1005点**で固定。制限内の発声/非持続格子を層別抽出し、各層の低誤差点、既存Random候補の等間隔抽出、DE採用候補を含む。目標そのものに相当する既知oracle格子点も含むため、Actor表ではoracle（E_total≤1e-8）を除くbestも示す。各モデルの学習点128点は別集計。exact-training判定は保存されたfloat32行動の完全一致であり、未一致点を独立な統計的テスト集合と主張しない。診断の追加シミュレーションは0音。λ=1の保存モデルで先に推論を試し、同じ候補プールを新規λ=.5にも使用した。

以下は各seedで計算した指標の平均。順位反転は実測と予測の順位が逆の候補対の割合（同値対を除く）。低J biasは固定プール内Jの下位10%で `Q−r`、負なら本当に良い行動を過小評価。

{table(['手法','λ','Critic','RMSE','Spearman','反転率','低J bias'],critics)}

{actor_table}

![Critic実測比較](figures/critic_scatter.png)

![低誤差領域のbias](figures/critic_bias.png)

全候補のQ1/Q2/minQ・実測r・発声・学習点一致判定は[圧縮CSV](critic_predictions_all.csv.gz)、各seedの精度と反転率は[診断指標](critic_metrics_all.csv)、Actorの制御と位置比較は[Actor診断](actor_diagnostics_all.csv)。全体相関が正でも、低誤差領域の順位が十分とは限らない。最終モデルの回顧診断だけで、学習途中の因果関係を特定できない。

## 代表音と図

同じseed1を両λ・全4手法・Final/Selectedで固定して比較。λ=1のTD3非持続例を含む。成功例だけをλごとに選び直さない。スペクトログラムは10MHz表示、245760窓、184320 overlap、Blackman-Harris、Target最大PSDに対する共通dB範囲と共通カラーバー。RMSは両音をTarget peakで除した5ms窓の共通振幅軸。制御は1枚に集約。描画設定と音声の出典hashは[図の記録](figure_sources.json) / [元波形の参照](endpoint_sources.json)。

![Final代表音](figures/representative_final.png)

![Selected代表音](figures/representative_selected.png)

## 保護・計算量・整理

[変更前棚卸し](artifact_inventory.json)で旧研究ファイル5053個をhash化。[監査](audit.json)で全48 endpointの保存モデル/行動を再評価し、波形・指標・発声の完全一致を確認。新規9runの学習候補36864音、独立方策評価198音、学習後replay18音。別途モデル整合15音、監査は1回48音・実行累計{audit_total}音（各実行をauditのpassesへ記録）。Critic推論0音。Random再スコア自体は0音で、λ=.5 Selectedの3音は各48監査音に含む。Random Selectedは全seedとも旧Selectedと同一だったため、音声ファイルは[元データを参照](random_audio_deduplication.json)し重複保存していない。[回数・時間](execution.json) / [各runの計算量](run_accounting.csv) / [endpoint再生](endpoint_replays.csv)。関連unittest21件も通過し、テスト用の64回のモデル評価（うち注入失敗1回）は研究runの計算量に含めず[テスト記録](tests.json)へ別記した。

Issue17のPNG {cleanup['issue17_pngs']}枚、Issue18のPNG {cleanup['issue18_pngs']}枚を確認。全研究成果中、同一hash PNGは{cleanup['duplicate_png_groups']}群、重複分{cleanup['duplicate_png_extra_files']}枚・{cleanup['duplicate_png_extra_bytes']/1e6:.2f} MB。[重複の保持理由](duplicate_png_review.csv)と[保持/整理一覧](artifact_retention_review.csv)を残した。参照される旧PNGはREPORT・画像registry・過去監査のexact hashを維持するため保留。Phase0の図、中断計算に残る重み・履歴、研究資料も保持した。重要な旧科学データの削除は0。

整理は無参照のignored一時stdoutログ3本だけをbyte単位一致で1つの[圧縮記録](consolidated_temporary_logs.json.gz)に統合し、元3本を除去。[削除一覧](cleanup_changes.json)。図の削除は0なので旧リンクの読み替えは不要。旧図の再生成先は保持一覧に記録した（旧学習用の厳密hash gateを再実行する際は、その実験当時のソース/buildが必要）。新規は主要図8枚のみ、全seedのPNG/途中checkpointを生成しない。全新規seedの数値・設定・Final/Selected重み・波形は保持。ignoreはこの追試の途中出力に限定。commit/pushや過去履歴の書換えは行っていない。

[変更前のファイル数/容量](storage_before.json) / [変更後とGit追加候補数/容量](storage_after.json) / [旧hash・リンク最終照合](final_integrity.json)。容量は.gitとsymlinkを除く論理bytes、ignored環境も含む。Git追加候補は未追跡・非ignoreファイルで、stage/commit済みという意味ではない。

## 全seedの評価値

{individual}

## Phase 1の判断

結論は[判断メモ](DECISION.md)。3seedの探索的比較であり、統計的一般化やRLのDE超えは主張しない。時変制御・Case Bをこの追試に追加せず、次の課題として分離する。
'''
    (OUT/'REPORT.md').write_text(text)
    (OUT/'README.md').write_text('''# Issue19 — 制限3Dの報酬重み・Actor/Critic診断

[結果レポート](REPORT.md) / [次段階の判断](DECISION.md)。旧Issue17/18は読み取り専用、λ=.5の9 runのみ新規。

- [条件・コードhash](manifest.json)、[モデル設定と数値一致](model_gate.json)
- [全seed](comparison_per_seed.csv)、[平均±標本SD](comparison_aggregate.csv)、[対応seed差](paired_lambda_deltas.csv)
- [Critic](critic_metrics_all.csv)、[Actor](actor_diagnostics_all.csv)、[既存候補再利用](random_reuse.json)
- [監査](audit.json)、[最終整合](final_integrity.json)、[図8枚の出所](figure_sources.json)、[成果物hash](artifact_checksums.json)
- [棚卸し](artifact_inventory.json)、[保持・整理一覧](artifact_retention_review.csv)、[整理記録](cleanup_changes.json)
- [変更前](storage_before.json)、[変更後](storage_after.json)
- [再現・監査手順](REPRODUCE.md)
''')
    (OUT/'REPRODUCE.md').write_text('''# 再現・監査

リポジトリ直下で実行。出力先はこの正本に固定し、inventoryは既存ディレクトリを拒否、未完了runは上書きしない。新規計算を作り直す場合は、別checkoutで新しい出力パスを設定すること。旧成果は変更しない。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue19-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_protection inventory
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_protection model
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_critics --baseline-only
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_training --workers 9
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_report
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue19_audit --finalize-only
```

保存済み成果の検証だけなら `issue19_audit`、描き直しは `issue19_report`。いずれも再学習しない。旧モデルはnative/model.cppのx0修正により歴史的なsource hashが異なるため、旧の厳密なload_readyでは拒否される。今回のmanifestとmodel_gateで変更を追跡し、旧manifestを改変しない。

各runの `curve.json.gz` は全独立評価/直接法候補、RLの `training_trace.json.gz` は全4096学習サンプル。 `agent.zip` / `selected_agent.zip` と `final.npz` / `selected.npz` を保持。共有の目標音・初期条件・全hyperparameterはmanifestと旧targetへのリンクを参照。NPZはfloat64の元振幅、WAVを重複保存しない。聴取時のみ全音を同じTarget peakで割ってWAVへ出力する。

旧PNGのレンダラは保持一覧に列挙。必要なexact source revisionは旧manifestのgit_head/source_sha256を参照し、旧の数値runを新設定で再学習しない。今回の8図は保存済み波形だけから再生成できる。
''')
    write_json(OUT/'report_execution.json',dict(wall_s=time.perf_counter()-tick,generated_sounds=0,image_count=len(images),source_sha256=digest(__file__)))
    from .issue19_storage import freeze_checksums
    freeze_checksums()
    print('Report complete: 48 endpoints, paired seed deltas, eight figures',flush=True)


if __name__=='__main__':report()
