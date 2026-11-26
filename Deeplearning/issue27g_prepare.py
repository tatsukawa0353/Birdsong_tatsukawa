"""Only T3 new generation;3 independent oracles, no parameter search."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import numpy as np
from .issue27g_data import ROOT,OUT,OLD,CONTROLS,CONFIG,target_path,hashes
from .issue27g_native import render,build_probe,PROBE
from .issue27f_prepare import features
from .model import LIBRARY
from .acoustics import distance
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest

PLAN='''# Issue27：3Target×一定3D/4Dの事前計画

2026-11-26。最新版Issue27、初回/旧9点/18点と9runのREPORTを参照。旧成果は保護し新成果先を使用。native/model.cpp/model.py/libbirdsong.soは不変。独立B subclassのみで右f0固定と右εの入力範囲2e7–5e7を扱う。native condition1、左右α1.2e6/β1500、x0.02cm/D.01/dt1e-7と元の全係数・遅延・式を保持。右寄与をclampしない。

TargetはIssue指定の3音で事前確定：T1=[35e6,35e6,3e6,49000,1e7]、T2=[35e6,35e6,3e6,49000,49000]は旧b2_c10/b3_c10をbyte同一で読み取り参照。T3=[35e6,21e6,3e6,49000,49000]だけ新規生成。各Target別インスタンスoracleを1音ずつ再計算。計4新規音。0.20s/50kHz/10000sample、2M native step、4状態/履歴/filter0、元16次15kHz lowpass/200デシメーション。一次持続proxy末尾1000sample AC std/自身peak>.05、直線除去後も>.05、有限、silent偽。前回と同じ左右変動proxyを使用しT1は左、T2/T3は左右の持続変動をgateとする。対称/非対称設定は同期/非同期の証明と呼ばない。独立oracleの波形・状態・native mean/std一致とE_spec/E_env/E/J=0、右source/output非ゼロを確認。非適格Targetは両次元未開始、差し替え/grid/時間延長なし。他適格Targetは続行。

全Targetに3D[ε_l,ps,f0_l]と4D[ε_l,ε_r,ps,f0_l]。3D右εは各Target自身の35M/35M/21M、右f0は1e7/49000/49000で両次元固定。ε_l25–50M、ε_r20–50M、ps2.5–4M、左f049000–800000。1episode1action0.20s。観測8Dは従来と同じ定数reset[-1,0,...]でTarget/正解値を漏らさない。r=−(.75Espec+.25Eenv)、共通E=Espec+Eenv、Target peak共通gain、位相合わせなし。

SAC/TD3/PPO×seed0/1/2×最大6組＝54run、各厳密4096環境step、最大221184学習音。独立方策評価0,128,...,4096の33点/run＝最大1782音は学習予算外。128×128、CPU1thread、LR3e-4/gamma1/batch32。SAC/TD3:buffer50000,warmup16,train_freq1,gradient_steps1,tau.005。SACauto_0.01,target_entropy=−行動次元（−3/−4）。TD3Normal(0,.1)も次元一致、delay2/targetnoise.2/clip.5。PPO方策/価値別128×128,nsteps128/epochs10/clip.2、他前回既定（gae.95,vf.5,ent0,gradclip.5,Adam eps1e-5）。他Adam eps1e-8。実構造/parameter数/実効値をassertする。

標準SB3と監査付き実装を2次元×3手法×128step×2のsynthetic1536episodeで照合（native0）。全6組の制御上下限・正解逆変換・実メモリ・全5制御・初期状態・9seed初期重み独立性を確認（native0）。T1の4D右ε下限/上限を2音、実メモリと出力差を確認するpreflightとしTarget探索には使用しない。方法別smokeは6組×3手法、各128学習+2評価+4保存再生＝2412音（全Target適格時）。smokeは合格した組のみ本学習へ進み、正式runを差し替えない。

正式SAC actor/critic/entropy各4080、TD3critic4080actor2040、PPO32rollout×4batch×10epoch＝1280 optimizer step（内部epoch320）。各更新LRと前回数値停止条件を観測し学習を変更しない。PPOraw Gaussianをrolloutへ、環境clip行動を別記。SAC/TD3Replay・実行action・reward・終端を観測。独立評価は対応する全更新後、最終はlearn終了後。重み/optimizer/Buffer/RNG/bestの前後signature一致で評価非混入を確認。Final4096、Selected33点の最初minJ、posthoc最初minE、training_best4096学習候補最初minJ（Actorではない）。E≤.10ゲートなし。追加予算/seed差し替え/TQC/直接最適化/5D/時変/実録音/commit/push/Issue送信をしない。

全4役割×全runを終了後に独立詳細再生（最大216音）、保存Actor27×6と学習best54を区別。全seed E_spec/E_env/E/J、持続/左右状態/source、探索値、絶対/探索幅正規化誤差、時間/更新/回数をCSV保存。3D固定右εの誤差を学習復元へ数えず共通3変数と4D追加右εを分ける。同Target内の3D→4D差と3seed標本SDを報告する。異なる自己TargetやCase A/Bの誤差を構造優位にしない。

図はIssue17 SciencePlots ['science','ieee','no-latex']の実フォントを確認して継承。軸46pt/目盛30pt/凡例29pt/線幅4pt、全曲線実線、全33点表示、SD薄帯、無題・grid/閾値線なし。共通設定を1か所へ集約、実効font/mathtext/TeX/DPI/寸法を保存。単独15×10inch/120DPIを基準に集約図の物理寸法を拡大し文字を縮小しない。従来Blackman-Harris24.576ms/75%overlap/0–10kHz、報酬STFT不変。各Target3D/4D全seed同PSD基準・色軸、狭い[-40,0]dBで飽和率を記録し必要な広域図だけ補助。冒頭3Target横並び絶対強度共通図、TargetごとFinal/Selected各1枚（4行Target/SAC/TD3/PPO×6列3Dseed0–2/4Dseed0–2）。学習曲線、制御、波形/平均PSD、代表物理状態を集約。training_best/posthoc個別画像やデバッグ/重複画像は作成しない。旧画像を削除せず、新規図の採否・削除一覧を明記して最終hash。
'''

def stage_a():
    if OUT.exists():raise FileExistsError('Preserve previous output')
    before=json.loads(Path('/tmp/birdsong-issue27g-before.json').read_text())
    assert all(digest(resolve_result_path(ROOT/r['path']))==r['sha256'] for r in before['files'])
    OUT.mkdir(parents=True);write_csv(OUT/'protected_before.csv',before.pop('files'));write_json(OUT/'storage_before.json',before);(OUT/'PLAN.md').write_text(PLAN)
    build=build_probe();assert build['unchanged_source_build_binary_exact'];ledger=[];gates={}
    for t,c in CONTROLS.items():
        p=target_path(t);p.parent.mkdir(parents=True)
        if t in ('T1','T2'):
            source=OLD/f"preflight/candidates/{'b2_c10' if t=='T1' else 'b3_c10'}.npz";shutil.copyfile(source,p)
            origin=dict(kind='read_only_previous',path=str(source.relative_to(ROOT)),sha256=digest(source));assert digest(p)==digest(source)
        else:
            ledger.append(dict(purpose='T3_target',status='started'));write_json(OUT/'preflight/ledger.json',ledger)
            v,coeff=render(c,detailed=True);np.savez_compressed(p,**v,sample_rate=50000,duration_s=.2,model_condition='B')
            ledger[-1].update(status='complete',native_steps=2000000);origin=dict(kind='new_prespecified_one_point')
        with np.load(p) as z:v={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(c,v['controls']);f=features(v);passed=bool(f['eligible'] and f['left_motion_periodic_proxy'] and (t=='T1' or f['right_motion_periodic_proxy']))
        ledger.append(dict(purpose=t+'_independent_oracle',status='started'));write_json(OUT/'preflight/ledger.json',ledger)
        oracle,coeff=render(c,detailed=False);np.savez_compressed(p.parent/'oracle.npz',**oracle)
        for k in ('audio','state_end_blocks','motion_mean','motion_std'):np.testing.assert_array_equal(v[k],oracle[k])
        comp=distance(v['audio'],oracle['audio'],50000,scale=f['peak']);assert comp['total']==0
        passed=passed and not comp['generated_silent'];assert abs(v['diagnostic_moments'][-1,5])>0 and abs(v['diagnostic_moments'][-1,9])>0
        gates[t]=dict(passed=passed,features=f,controls=c.tolist(),target_sha256=digest(p),origin=origin,oracle_components=comp,oracle_J=0.,native_coefficients=coeff)
        ledger[-1].update(status='complete',native_steps=2000000);write_json(OUT/'preflight/ledger.json',ledger);write_json(OUT/'preflight/gate.json',gates)
        print(json.dumps(dict(target=t,passed=passed,features=f)),flush=True)

def freeze():
    from .issue24_data import sb3_hashes
    tests=json.loads((OUT/'preflight/tests.json').read_text());assert tests['passed'] and source_hashes_match(tests['source_sha256'], hashes())
    gates=json.loads((OUT/'preflight/gate.json').read_text());cases=[(t,d) for t,g in gates.items() if g['passed'] for d in (3,4)]
    assert len(tests['smoke_runs'])==len(cases)*3
    if (OUT/'manifest.json').exists():raise FileExistsError('Frozen protocol preserved')
    previous=json.loads((OLD/'manifest.json').read_text());versions={p:importlib.metadata.version(p) for p in previous['versions']};assert versions==previous['versions']
    write_json(OUT/'manifest.json',dict(issue=27,revision='3Targets_3D4D',created_utc=datetime.now(timezone.utc).isoformat(),settings=CONFIG,source_sha256=hashes(),
        native_library_sha256=digest(LIBRARY),probe_sha256=digest(PROBE),plan_sha256=digest(OUT/'PLAN.md'),target_sha256={t:digest(target_path(t)) for t in CONTROLS},
        preflight_passed=True,eligible_cases=cases,skipped_targets=[t for t,g in gates.items() if not g['passed']],research_runs=len(cases)*9,planned_maximum_runs=54,
        versions=versions,sb3_implementation_sha256=sb3_hashes(),observation='8D same constant reset[-1,0,...], no Target information',representative_seed=1))
    print('Official protocol frozen',cases,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('stage_a','freeze'));a=p.parse_args();stage_a() if a.stage=='stage_a' else freeze()
