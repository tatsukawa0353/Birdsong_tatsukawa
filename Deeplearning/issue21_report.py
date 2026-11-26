"""Report the width experiment without retraining or changing any old artifact."""
from .result_paths import resolve as resolve_result_path
import csv
import gzip
import io
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from .issue21_data import (ROOT,BASE,OUT,METHODS,WIDTHS,SEEDS,folder_for,curve_for,endpoints,target_data,ready)
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue15_analysis import write_json
from .issue16 import digest
from .issue19_protection import workspace_inventory


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def statistics(rows):
    groups=[]
    for method in METHODS:
        for width in WIDTHS:
            for role in ('final','selected'):
                a=[r for r in rows if r['method']==method and r['width']==width and r['role']==role]
                g=dict(method=method,width=width,role=role,n=3,voiced=sum(r['sustained_voiced'] for r in a),
                    seeds_under_0p10=sum(r['E_total']<=.1 for r in a))
                for field in ('E_total','E_spec','E_env','parameter_nrmse_restricted'):
                    values=np.array([r[field] for r in a]);g[field+'_mean']=float(values.mean());g[field+'_sd']=float(values.std(ddof=1))
                g['provisional_goal_met']=role=='final' and g['E_total_mean']<=.10 and g['voiced']==3
                groups.append(g)
    return groups


def listening_archive():
    target,_=target_data();gain=1/abs(target).max();aliases=[];stored={}
    registry=json.loads((OUT/'endpoint_sources.json').read_text())
    with zipfile.ZipFile(OUT/'new_listening_audio.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for r in registry:
            if r['width']==32:continue
            with np.load(resolve_result_path(ROOT/r['audio_path'])) as z:audio=z['audio'].copy()
            data=io.BytesIO();wavfile.write(data,50000,(audio*gain).astype(np.float32));payload=data.getvalue()
            import hashlib
            h=hashlib.sha256(payload).hexdigest();name=f'{r["method"]}_w{r["width"]}_seed{r["seed"]}_{r["role"]}.wav'
            if h not in stored:
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,payload);stored[h]=name
            aliases.append(dict(method=r['method'],width=r['width'],seed=r['seed'],role=r['role'],member=stored[h],sha256=h))
    write_json(OUT/'listening_audio.json',dict(sample_rate=50000,gain=gain,per_sound_normalization=False,
        reference_target_wav=str((BASE/'runs/constant_3d/TD3/lambda1/seed_0/target.wav').relative_to(ROOT)),
        unmodified_raw_audio='endpoint_sources.json references original-amplitude float64 NPZ',
        unique_members=len(stored),endpoints=aliases,generated_sounds=0))


def freeze():
    """Verify first, then measure final file counts and freeze a portable hash list."""
    from .issue21_audit import integrity
    for name in ('artifact_checksums.json','storage_after.json'):
        if not (OUT/name).exists():(OUT/name).write_text('{}\n')
    integrity()
    new=[resolve_result_path(ROOT/p) for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines() if (resolve_result_path(ROOT/p)).is_file()]
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True).splitlines()
    before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total']
    for _ in range(4):
        workspace=workspace_inventory();size=sum(p.stat().st_size for p in new)
        record=dict(workspace=workspace,new_git_candidate_files=len(new),new_git_candidate_bytes=size,
            modified_tracked_files=sum(r[:2].strip() not in ('??','D') for r in status),deleted_tracked_files=sum('D' in r[:2] for r in status),
            status=status,measurement='Logical regular-file bytes, including ignored files; .git/symlinks excluded; new candidates are not staged/committed')
        write_json(OUT/'storage_after.json',record)
        block='<!-- ISSUE21_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[
            ['作業ツリー通常ファイル数',before['files'],workspace['total']['files']],
            ['論理容量（ignored含む）',f"{before['bytes']/1e6:.2f} MB",f"{workspace['total']['bytes']/1e6:.2f} MB"],
            ['Git新規追加候補','0（クリーン）',f'{len(new)}個 / {size/1e6:.2f} MB'],
            ['既存追跡ファイルの変更 / 削除','0 / 0',f"{record['modified_tracked_files']} / {record['deleted_tracked_files']}"]])+ '\n<!-- ISSUE21_STORAGE_END -->'
        p=OUT/'REPORT.md';text=p.read_text()
        p.write_text(re.sub(r'<!-- ISSUE21_STORAGE_BEGIN -->.*?<!-- ISSUE21_STORAGE_END -->',lambda _:block,text,flags=re.S) if '<!-- ISSUE21_STORAGE_BEGIN -->' in text else text+'\n'+block+'\n')
        write_json(OUT/'artifact_checksums.json',dict(excludes=['artifact_checksums.json','storage_after.json'],
            implementation_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.glob('issue21*.py'))},
            artifacts={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')}))


def report():
    ready();tick=time.perf_counter();target,controls=target_data();rows=[];policy=[];deltas=[]
    for method in METHODS:
        for width in WIDTHS:
            for seed in SEEDS:
                for role,p in endpoints(method,width,seed).items():
                    c=np.array(p['physical_controls']);error=c-controls;comp=p['acoustic_components']
                    rows.append(dict(method=method,width=width,seed=seed,role=role,trial=p['episodes'],
                        E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],
                        sustained_voiced=p['sustained_voiced'],failed=p['failed'],epsilon_l=c[0],ps=c[2],f0_l=c[3],
                        epsilon_l_error=error[0],ps_error=error[2],f0_l_error=error[3],
                        parameter_nrmse_original=p['parameter_nrmse'],
                        parameter_nrmse_restricted=float(np.sqrt(np.mean((error[[0,2,3]]/[2.5e7,1.5e6,751000.])**2)))))
                for p in curve_for(method,width,seed):
                    comp=p['acoustic_components'];policy.append(dict(method=method,width=width,seed=seed,trial=p['episodes'],
                        E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],J=p['weighted_loss'],
                        sustained_voiced=p['sustained_voiced'],gradient_updates=p['gradient_updates']))
    groups=statistics(rows)
    for method in METHODS:
        for width in WIDTHS[1:]:
            for seed in SEEDS:
                for role in ('final','selected'):
                    a=next(r for r in rows if (r['method'],r['width'],r['seed'],r['role'])==(method,32,seed,role))
                    b=next(r for r in rows if (r['method'],r['width'],r['seed'],r['role'])==(method,width,seed,role))
                    deltas.append(dict(method=method,width=width,seed=seed,role=role,
                        **{'delta_'+f:b[f]-a[f] for f in ('E_total','E_spec','E_env','parameter_nrmse_restricted')},
                        voiced_baseline=a['sustained_voiced'],voiced_wide=b['sustained_voiced']))
    write_csv(OUT/'comparison_per_seed.csv',rows);write_csv(OUT/'comparison_aggregate.csv',groups)
    write_csv(OUT/'paired_width_deltas.csv',deltas);write_csv(OUT/'policy_evaluations.csv',policy)
    with gzip.open(OUT/'new_training_samples.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','width','seed','trial','E_total','E_spec','E_env','reward_float32','sustained_voiced','action','physical_controls'],lineterminator='\n');writer.writeheader()
        for method in METHODS:
            for width in WIDTHS[1:]:
                for seed in SEEDS:
                    for p in read_record(folder_for(method,width,seed)/'training_trace.json'):
                        comp=p['acoustic_components'];writer.writerow(dict(method=method,width=width,seed=seed,trial=p['episodes'],
                            E_total=p['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],reward_float32=p['reward'],
                            sustained_voiced=p['sustained_voiced'],action=json.dumps(p['action']),physical_controls=json.dumps(p['physical_controls'])))
    lower=[]
    for dimension in (1,2):
        for method in METHODS:
            for seed in SEEDS:
                p=ROOT/f'results/Phase1/CaseA/parameter_dimension/runs/constant_{dimension}d/{method}/lambda1/seed_{seed}/summary.json'
                s=json.loads(p.read_text());lower.append(dict(dimension=dimension,method=method,seed=seed,E_total=s['final']['acoustic_error'],source=str(p.relative_to(ROOT)),source_sha256=digest(p)))
    write_csv(OUT/'lower_dimensional_reference.csv',lower)
    listening_archive()
    from .issue21_figures import render
    images=render(rows,deltas)
    with (OUT/'critic_metrics.csv').open() as stream:critics=list(csv.DictReader(stream))
    with (OUT/'actor_diagnostics.csv').open() as stream:actors=list(csv.DictReader(stream))
    with (OUT/'run_accounting.csv').open() as stream:accounting=list(csv.DictReader(stream))
    fmt=lambda r,f:f"{r[f+'_mean']:.5f} ± {r[f+'_sd']:.5f}"
    main=table(['手法','幅','採用','E_total 平均±SD','E_spec 平均±SD','E_env 平均±SD','持続/3','暫定目標'],
        [[r['method'],r['width'],r['role'],*[fmt(r,f) for f in ('E_total','E_spec','E_env')],r['voiced'],'PASS' if r['provisional_goal_met'] else ('未達' if r['role']=='final' else '副評価')] for r in groups])
    differences=[]
    for method in METHODS:
        for width in WIDTHS[1:]:
            for role in ('final','selected'):
                p=[r for r in deltas if r['method']==method and r['width']==width and r['role']==role]
                vals=[np.array([r['delta_'+f] for r in p]) for f in ('E_total','E_spec','E_env')]
                differences.append([method,width,role,*[f'{v.mean():+.5f} ± {v.std(ddof=1):.5f}' for v in vals]])
    individual=[]
    for method in METHODS:
        for width in WIDTHS:
            for seed in SEEDS:
                a=next(r for r in rows if (r['method'],r['width'],r['seed'],r['role'])==(method,width,seed,'final'))
                b=next(r for r in rows if (r['method'],r['width'],r['seed'],r['role'])==(method,width,seed,'selected'))
                individual.append([method,width,seed,f"{a['E_total']:.7g}",f"{b['E_total']:.7g}",b['trial'],a['sustained_voiced'],b['sustained_voiced']])
    critic_table=[]
    def metric_mean(part,field):
        values=np.array([float(r[field]) for r in part if r[field]])
        return f'{values.mean():.4f} ± {values.std(ddof=1):.4f}' if len(values)>1 else 'NA'
    for method in METHODS:
        for width in WIDTHS:
            for critic in ('Q1','Q2','minQ'):
                p=[r for r in critics if r['method']==method and int(r['width'])==width and r['critic']==critic and r['subset']=='fixed_pool']
                low=[r for r in critics if r['method']==method and int(r['width'])==width and r['critic']==critic and r['subset']=='low_J_decile']
                critic_table.append([method,width,critic,metric_mean(p,'rmse'),metric_mean(p,'spearman'),metric_mean(p,'pairwise_inversion_fraction'),metric_mean(low,'bias_Q_minus_r')])
    passed=[g for g in groups if g['provisional_goal_met']]
    goal=('暫定目標を満たした条件：'+', '.join(f"{g['method']} [{g['width']},{g['width']}]" for g in passed)+'。') if passed else '今回の幅変更では、Final平均E_total≤0.10かつ持続発声3/3の暫定目標を満たす条件はなかった。'
    conclusions=[]
    for method in METHODS:
        final=[g for g in groups if g['method']==method and g['role']=='final'];base=next(g for g in final if g['width']==32);best=min(final,key=lambda g:g['E_total_mean'])
        conclusions.append(f"{method}：Final平均が最小だった幅は{best['width']}（{fmt(best,'E_total')}、持続{best['voiced']}/3）。32幅の{fmt(base,'E_total')}に対する平均差は{best['E_total_mean']-base['E_total_mean']:+.5f}。")
    next_step=('暫定目標を満たす条件は得られたが、3seedだけの安定性・一般化は未確定。次Issueでは達成条件の再現性を確認し、探索ノイズ変更を必須とはせず独立した仮説として検証する。' if passed else
        'ネット幅だけでは暫定目標を達成できなかったため、次Issueで探索幅を1要因として比較する価値がある。TD3の物理量ごとのσは事前固定し、ネット幅・LR・報酬は同時変更しない。SACの分散/entropyは別の設計として扱う。ただし今回の結果だけから探索ノイズが唯一の原因とは断定できない。')
    timing=[]
    for method in METHODS:
        for width in WIDTHS:
            p=[r for r in accounting if r['method']==method and int(r['width'])==width]
            v=np.array([float(r['source_training_wall_s']) for r in p]);timing.append([method,width,f'{v.mean():.2f} ± {v.std(ddof=1):.2f}'])
    execution=json.loads((OUT/'execution.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    architectures=json.loads((OUT/'preflight.json').read_text())['architecture_checks']
    architecture_table=table(['手法','幅（2層）','Actorパラメータ数','Critic Q1+Q2パラメータ数'],
        [[r['method'],r['width'],r['network_geometry']['actor_parameters'],r['network_geometry']['critic_parameters']] for r in architectures])
    decision='\n\n'.join([goal,*conclusions,next_step])
    text=f'''# Issue21 — Case A・一定3Dのネットワーク幅比較

## 結果と判断

{decision}

今回のTD3では、拡幅した全条件で持続発声3/3となったが、32→256幅の平均E_specは0.22094→0.23157で、平均E_envの0.11936→0.01097が共通誤差の改善を支えている。発声回復とスペクトル模倣の改善を同一視しない。128/256幅のFinal平均差は約0.000028と小さく、256幅が明確に優れるとは判断できない。

SACは128幅でFinal平均E_specが0.24065→0.17329へ下がり、共通誤差も3seedすべてで32幅より改善した。しかし256幅のFinal平均は0.24870に戻り、幅を増やすほど良くなる結果ではなかった。128幅のseed1/2ではSelectedが0.04017/0.09013まで下がった一方、Finalは0.18355/0.14250だった。途中の低誤差解を最終方策が保持したとは言えず、暫定目標をSelectedで代用しない。

幅以外の実効設定が一致することを全runで確認した。ActorとCriticを同時に拡幅し、それに伴って学習データの分布も変わるため、性能差を「Criticの表現力不足だけが原因だった」と因果的に断定しない。3seedの探索的比較であり、有意差や唯一の最適幅を主張しない。Selectedの改善をFinalの改善と混同しない。

固定観測・1行動の課題なので、Actorは同じ初期観測に対して3値を返す。拡幅しても可制御次元や物理範囲は増えず、近似・最適化・探索データの変化を含む実験として解釈する。

## 固定条件・保護

Case Aのみ、0.20秒・50kHz・10000sample、native dt=1e-7、初期状態/遅延履歴0、右側α_r=β_r=0。目標は既存波形/hashを参照し、制御は `[4.1e7,3.5e7,3.4e6,3e5,4.9e4]`。可制御はε_l∈[2.5e7,5e7]、p_s∈[2.5e6,4e6]、f0_l∈[49000,800000]。右のε_r=3.5e7、f0_r=4.9e4固定。f0_lはHzの基本周波数ではなく開閉外力係数。ε_lとp_sも直接測定された張力やPaと同一とはせず、図はモデル内部単位とした。

研究設定x0=.02 cm,D=.01とCase Aの右結合ゼロを実際のC++ resetで検証。Issue19で数値一致を確認したnative library/hashを保持し、目標と基準4波形を再生して全サンプル一致した。[事前検証](preflight.json)。実効設定の初回照合ではJSON listとPython tupleの表記差で停止し、JSON表現に揃えて再確認した。学習前の2回・計10音を[事前検証回数](preflight_attempts.json)に記録している。

新規は[64,64]/[128,128]/[256,256]×TD3/SAC×seed0/1/2の18run。32幅はIssue18のrestricted3D・λ=1を読み取り専用利用し、再学習・再描画・旧ファイル更新を行わない。Issue20案、Case B、時変制御、4D/5D、実録音、DEの新規計算およびDEの比較図は本Issueの対象外。[条件・依存版・source hash](manifest.json) / [着手前の旧ファイルhash](protected_before.csv)。着手時はクリーン、未pushコミットなし。

## 幅以外の学習設定

lr=3e-4、batch32、gamma1、buffer50000、warmup16、train_freq1、gradient_steps1、CPU・worker内1thread。TD3は正規化σ=.1の既存探索ノイズ、policy_delay2、SACはent_coef=auto_0.01。報酬は `r=−(E_spec+E_env)`。先生の別提案 `−{{λ E_spec+(1−λ)E_env}}` やIssue19のλ=.5は導入していない。

ActorとQ1/Q2双方、target networkの実際のLinear層を検査し、baselineとの実効設定差がnet_archだけであることを確認した。NN形状の変更では初期重み・乱数消費・学習軌跡は変わる。同じseedは同じ候補列の意味ではないが、学習前warmup16行動はbaselineと完全一致した。[設定と層の事前確認](preflight.json) / [各runの実効設定](runs/) / [監査](audit.json)。

{architecture_table}

パラメータ数はonline networkのみで、target/optimizerの保存容量を含まない。

各runは4096学習候補・Critic更新4080回。TD3のActor更新2040回、SACのActor更新4080回。独立評価は0,128,...,4096の33回で、128〜3968は直前の勾配更新後、4096は最終更新後。評価前後のモデル・optimizer・buffer・RNG署名が同じであることを検査した。

## 共通評価

主指標はFinalの `E_total=E_spec+E_env`、平均±標本SD（ddof=1、3seed）。Selectedは33回の独立評価の最初のminJ。今回はλ=1なのでJ=E_totalだが、Finalと途中Selectedを分離する。失敗/非持続seedを除かない。暫定目標はFinal平均≤.10かつ持続3/3であり、全seedのE_total≤.10や聴覚的同一性を保証する条件ではない。持続proxyは末尾1000sampleのAC標準偏差/Target peak>.05かつ数値失敗なし。

{main}

同じseedの拡幅−32幅の差。負なら誤差改善。平均差と対応差のSDを分け、seed別の符号は[CSV](paired_width_deltas.csv)に保持した。

{table(['手法','幅','採用','ΔE_total 平均±SD','ΔE_spec 平均±SD','ΔE_env 平均±SD'],differences)}

![独立評価の学習曲線](figures/learning_curves.png)

![誤差内訳](figures/endpoint_errors.png)

![対応seed差](figures/paired_differences.png)

学習中のfloat32報酬と独立方策評価を混ぜない。全73728学習サンプルは[圧縮CSV](new_training_samples.csv.gz)、独立評価は[方策評価CSV](policy_evaluations.csv)、Final/Selectedの成分・制御・物理誤差は[各seed CSV](comparison_per_seed.csv)。物理NRMSEは従来の全域幅と今回の制限幅の両方を明示し、音響的に近い制御が唯一の物理的正解とは主張しない。

![物理制御の集約](figures/physical_controls.png)

1D/2Dの[既存参照値](lower_dimensional_reference.csv)も保存JSONから抽出した。1D/2Dでは非学習変数、特にf0_lを正解に固定しているため、3Dとの差を次元数だけの効果と解釈しない。これらの新規学習は行っていない。

## Critic/Actor診断

Issue19と同一hashの1005候補プールを参照。新しい候補抽選や鳴管計算は0。発声/非持続格子・保存Random・既存DEの高精度3点を含むが、DEを新規学習したり比較手法の曲線/図へ掲載したりしていない。既知oracle点もあるため、Actor比較の最良候補はoracle除外（E_total>1e-8）も別記した。各モデルの学習点128点は別集計。完全一致しない行動点を統計的に独立とみなさず、低誤差を厚く選ぶ層別抽出の反転率を物理空間の発生確率とは解釈しない。

観測は固定 `[-1,0,0,0,0,0,0,0]`、VecNormalizeなし。行動は同じ[-1,1]の3Dから制限範囲へ写像。Criticはfloat32のpolicy.scale_action、実測は保存された元のfloat64物理行動のもの（丸め差最大1.2e-7）。32幅で既存診断のRMSE・相関・反転率・biasが再現されることも確認した。[推論記録](critic_execution.json)。

終端1stepなのでTD targetの(1−done)=0で、TD3/SACの両Criticの教師はr。TD3のtarget minQとSACの次状態entropy項は消える。TD3 ActorはQ1を最大化し、SAC Actorは現在行動のminQとentropyを使う確率的方策を学習する。SACの決定論的Finalが候補内最大minQとずれることだけでActorの欠陥とはしない。

指標は各seedで計算して平均±標本SD。順位反転は実測/予測の順位が逆の候補対（真値または予測の同値対を除く）。低J biasは固定プールの下位10%でのQ−rで、負なら良い候補を過小評価。表と図ではQ1/Q2/minQを区別する。全幅で同じ報酬尺度なのでRMSEを同じ単位で比較できるが、Actorと学習サンプル分布も変わるため容量だけの因果評価ではない。

{table(['手法','幅','Critic','RMSE','Spearman','固定pool反転率','低J bias'],critic_table)}

本プールではTD3 Q1の平均RMSEが32幅0.10711→256幅0.06543、低J biasが−0.22734→−0.10174へ改善したが、低J順位反転率は256幅でも0.459。SACのminQは平均RMSEが32幅0.06690→128幅0.05987となった一方、256幅は0.06406で、低J biasも128幅−0.08956から256幅−0.10784へ戻った。近似指標の一部は改善したが、低誤差領域の順位付けを解決したとは言えない。

TD3 256幅のseed1/2では、実測が良い非oracle候補（共通誤差約3.7e-5）がQ1で3位/8位まで上がったのに、Actorの共通誤差は0.24072/0.24358。候補内最大Q1−Actor Q1は+0.00293/+0.02938で、Actorの選択・局所的な勾配・探索被覆も検討する必要がある。一方seed0は同候補97位で、Actor Q1が候補内最大Q1より高く、Critic側の課題も残る。固定候補内の比較から連続空間のActor最適化の成否を断定はできない。

![Critic実測比較](figures/critic_scatter.png)

![Critic指標比較](figures/critic_metrics.png)

[全候補Qの圧縮CSV](critic_predictions.csv.gz) / [seed・subset別の診断](critic_metrics.csv) / [Actor制御と良い候補の順位・Q gap](actor_diagnostics.csv)。有限の診断候補内最大Qは連続空間の大域最大ではない。最終モデルの回顧診断から学習途中の因果関係は特定できない。

## 代表音

全幅・両手法でseed1を固定し、Final/Selectedを分けた。Target最大full-band PSDを共通参照として、全48endpointを覆う同じdB範囲・同じカラーバー。RMSは共通Target peak基準、5ms窓。波形は位相合わせせず40–60msだけを表示し、これは描画範囲の切出しで報酬や保存音の短縮ではない。表示PSDは10MHzへ再サンプリング、Blackman-Harris窓245760・overlap184320、有効窓のみ、0–10kHz。[表示条件と図hash](figure_sources.json)。

![Finalの代表音](figures/audio_final.png)

![Selectedの代表音](figures/audio_selected.png)

全新規runのFinal/Selectedは元振幅float64 NPZとモデルを保持。聴取用の[新規音WAV ZIP](new_listening_audio.zip)は全音に同じTarget由来gainを使い、完全同一WAVは[対応表](listening_audio.json)で共用する。旧32幅音声は[既存の元データ参照](endpoint_sources.json)を使い、コピーしない。ZIP化と描画に追加シミュレーションはない。

## 全seed

{table(['手法','幅','seed','Final E_total','Selected E_total','採用試行','Final持続','Selected持続'],individual)}

## 回数・時間・再現性

18runすべて4096試行、計73728学習音。独立評価594音、学習後replay36音。[実行記録](execution.json)のwall時間は{execution['wall_s']:.2f}秒、worker数{execution['workers']}。事前整合は計10音、テストのモデル評価140回（注入失敗1回を含む）は研究runと分離。追加監査は1回48音、実行累計{audit['cumulative_audit_generated_sounds']}音。Critic診断は0音。全48endpointを保存モデルから再生し、行動・物理制御・波形・成分・発声判定が完全一致した。[再生監査](endpoint_replays.csv) / [監査結果](audit.json) / [14件のテスト記録](tests.json)。

{table(['手法','幅','学習wall秒：平均±標本SD'],timing)}

旧32幅は歴史的な12worker実行、新規は18worker実行で競合条件が異なる。上表は実測時間の記録であり、NN幅のみの純粋な速度比較とはしない。全runはCPU1thread、同じ予算・学習設定。処理CPU時間・評価音数は[run別記録](run_accounting.csv)に保持した。

着手前の全旧成果・既存コードはhash照合し、変更/削除0。root READMEの索引とignoreだけを更新した。不要な途中checkpointやseedごとのPNGを出力せず、主要図8枚に集約した。過去の履歴やバイナリの書換え、commit/push、Issueのクローズは行っていない。[最終整合](integrity.json) / [成果物hash](artifact_checksums.json) / [変更前容量](storage_before.json) / [変更後容量](storage_after.json)。

## 次Issueへの判断

{next_step}

正規化TD3σ=.1の物理換算は `[1.25e6,7.5e4,3.755e4]`。これは最大変化量や離散刻みではなく標準偏差で、境界ではクリップされる。ノイズの調整例や報酬式の変更は未検証の別仮説とし、この追試に追加しない。Case Aの人工持続音用f0_l≤800000を実音声・無音/閉鎖・Case Bへ無条件に適用しない。今回の指示は[Issue21](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/21)、研究管理メモは[Notion](https://app.notion.com/p/41f6906b3ff081c3ba69e92cedfafb95)（参照のみ）。
'''
    (OUT/'REPORT.md').write_text(text)
    (OUT/'README.md').write_text('''# Issue21 — Case A・一定3Dのネットワーク幅比較

[結果レポート](REPORT.md)。λ=1、Actor/Critic幅64/128/256の18run。32幅はIssue18をreadonly参照。

- [条件・source hash](manifest.json)、[事前検証](preflight.json)
- [全seed](comparison_per_seed.csv)、[平均±標本SD](comparison_aggregate.csv)、[対応seed差](paired_width_deltas.csv)
- [独立方策評価](policy_evaluations.csv)、[新規学習サンプル](new_training_samples.csv.gz)
- [Critic指標](critic_metrics.csv)、[Actor比較](actor_diagnostics.csv)、[推論記録](critic_execution.json)
- [再生監査](audit.json)、[旧成果の最終照合](integrity.json)、[ファイルhash](artifact_checksums.json)
- [図8枚の出所](figure_sources.json)、[元波形/モデル参照](endpoint_sources.json)
- [聴取用WAV ZIP](new_listening_audio.zip)、[gainと対応表](listening_audio.json)
- [再現と監査](REPRODUCE.md)、[変更前](storage_before.json)、[変更後](storage_after.json)
''')
    (OUT/'REPRODUCE.md').write_text('''# 再現・検証

リポジトリ直下、既存のDeeplearning/.venv-rlを使用。専用出力先を新規作成して18runのみを学習し、Issue17–19へ書き込まない。prepareは既存manifest/runを拒否する。未完了runも上書きせず、完了済みrunはsourceと全成果物hashが一致するときのみ再利用。全計算完了後のtrainはexecution.jsonを保持して終了する。

```bash
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue21-mpl
# 新規実験のための着手前snapshotが別途必要。既存正本でprepare/trainを繰り返さない。
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_training --workers 18
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_critics
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_audit
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue21_report
```

保存済み成果の再生検証はissue21_audit（48音を別途計数）、シミュレーションなしの検証は `issue21_audit --integrity-only`。保存音・CSVのみからの描画はissue21_report。critic診断は学習済み重みとIssue19の既存1005候補だけを読み、再学習・鳴管計算を行わない。数値履歴はcurve.json.gzの33独立評価とtraining_trace.json.gzの全4096学習行動を保持。

新しい別出力先の本計算を行う場合は、元ファイルのhash snapshot（protected_before.csv/storage_before.jsonのschema）を用意し、issue21_prepareの--before/--outputおよびissue21_trainingの--outputを同じ新規先へ指定する。解析CLIはこの正本に固定されるため、別出力の解析にはissue21_data.OUTをその別checkoutで変更してからprepareする。元の正本は変更しない。

各runのconfig.jsonに実効設定・Actor/Q1/Q2/targetの層構造、summary.jsonに予算・評価回数・wall/CPU時間・モデル/波形/履歴hashを保存。Final/Selectedを別のagent.zip/selected_agent.zipとNPZで保持。旧32幅のmodel/source hashは当時のものを維持し、今回のnative一致はpreflightとmanifestで追跡する。
''')
    write_json(OUT/'report_execution.json',dict(wall_s=time.perf_counter()-tick,generated_sounds=0,images=len(images),source_sha256=digest(__file__)))
    print('Report generated: validating all widths/seeds, eight figures, listening audio and final hashes',flush=True)
    freeze()


if __name__=='__main__':report()
