"""Case B physical gate; preserve a diagnostic study if its Target fails."""
from .result_paths import resolve as resolve_result_path
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np

from .issue27_data import ROOT,OUT,BASE,TARGET,CONTROLS,SEEDS,source_hashes
from .issue27_native import build_probe,render,PROBE,CHANNELS
from .model import LIBRARY
from .model_env import AudioDecimator
from .acoustics import distance
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest


def physics(before=Path('/tmp/birdsong-issue27-before.json')):
    if OUT.exists():raise FileExistsError('Existing study preserved')
    before=json.loads(Path(before).read_text())
    for r in before['files']:assert digest(resolve_result_path(ROOT/r['path']))==r['sha256'],r['path']
    OUT.mkdir(parents=True);(OUT/'preflight').mkdir();write_csv(OUT/'protected_before.csv',before.pop('files'));write_json(OUT/'storage_before.json',before)
    (OUT/'PLAN.md').write_text('''# Issue27 — 計算前の計画

Case B自身のTarget [4.1e7,3.5e7,3.4e6,3e5,1e7]を新規生成する。左右α=1.2e6/β1500を保持し、右f0=1e7を強制する。実ロードlibraryのメモリをread-only probeで読み、元bird_advanceの関数ポインタのみで積分する。元native/model.cppとlibraryを変更しない。x0=.02cm/D=.01、0.20s/50kHz/10000sample、dt1e-7、初期4状態/遅延/filter=0を保持。

先にCase B Target、同B oracle、同じ5制御でCase A対照を各1音生成する。ソース再ビルドが実libraryとbinary一致しなければ、同じTarget制御でreference libraryを1音追加し波形/状態の一致を検証する。Targetの末尾1000sample AC std/Target peak>.05、有限値、Bの左右係数非ゼロ、右寄与非ゼロ、oracle誤差≤1e-12を物理ゲートとする。不合格では学習せず診断報告に進む。変位だけから開閉を断定せず、右の微小運動も保存する。

物理ゲートと短いReplay/実効LR/seed/正規化/再生試験を通過後、SAC128・3seed(0,1,2)を各4096候補、独立Actor評価0,128,...,4096の33点で実行する。ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]、右ε_r3.5e7/f0_r1e7固定。通常SACのみ。Actor/Critic[128,128]、3LR3e-4、auto_0.01/target_entropy−3、batch32/buffer50000/warmup16/gamma1/train_freq1/gradient_steps1、その他Issue23設定と同一。

r=−J、J=.75E_spec+.25E_env（係数和1、倍補正なし）、共通E=E_spec+E_env。Finalは4096終了後、Selectedは33点の最初minJ、posthocは最初minE。training_bestは実学習4096候補の最初minJで、Actor保存方策と区別。独立評価はReplay・学習RNG・最良学習候補に入れない。全更新のLR・有限性・Adam状態、実行行動/物理値/Replayを検査。数値停止基準は既存Issue25と同じで、失敗runを保存し再始動・救済・条件置換しない。

獲得は初期Actor Eから半減する最初の独立評価点と発声proxyを記述し、Selected→FinalのJ/Eと持続状態で保持を別記する。E≤.10は合否・延長・追加探索のゲートにしない。3seed平均±標本SD、全seed/非持続/失敗を表示する。

Final/Selected/posthoc/training_bestをCase Bで独立再生し、左右末尾状態・native全刻みの振動統計、π̃の左右source項と出力項/遅延feedbackを計測する。分解は現在状態の代数的寄与で、片側除去時の因果寄与ではない。WAVはB Target peak由来の共通gain、PSD色基準/表示範囲も全音で共通。Case A Issue23は自身の別Targetに対する参考値のみで、誤差だけによる分岐優劣/学習難易度の結論を出さない。

既存Issue17〜26を上書き/再生成/削除せず専用出力に保存する。Case A旧Targetの流用、旧Issue20の実験、局所探索、左右切替、5D、時変、実録音、別reward、Critic高LRは実施しない。commit/push/Issue更新・外部送信は行わない。
''')
    ledger=[];build=build_probe();failure=None;started=time.perf_counter()
    def generate(name,condition,detailed,library=None):
        entry=dict(purpose=name,condition=condition,detailed=detailed,status='started',requested_native_steps=2000000);ledger.append(entry)
        write_json(OUT/'preflight/ledger.json',ledger)
        values,initial=render(CONTROLS,condition,detailed=detailed,library=library)
        entry.update(status='complete',native_steps=2000000,episodes=1)
        np.savez_compressed(OUT/f'preflight/{name}.npz',**values,sample_rate=50000,duration_s=.2,model_condition=condition)
        write_json(OUT/'preflight/ledger.json',ledger);return values,initial
    try:
        target,coeff=generate('target','B',False)
        oracle,oc=generate('oracle','B',True)
        contrast,ac=generate('same_controls_caseA','A',True)
        peak=float(abs(target['audio']).max());error=distance(target['audio'],oracle['audio'],50000,scale=peak)
        np.testing.assert_array_equal(target['audio'],oracle['audio']);np.testing.assert_array_equal(target['state_end_blocks'],oracle['state_end_blocks'])
        assert max(error[k] for k in ('log_spectral','envelope','total'))<=1e-12
        assert not np.array_equal(target['audio'],contrast['audio'])
        assert digest(TARGET)!=digest(ROOT/'results/Phase1/CaseA/domain_diagnostics/diagnostics/target.npz')
        if not build['unchanged_source_build_binary_exact']:
            reference,_=generate('source_reference','B',False,PROBE.parent/'libsource_reference.so')
            for k in ('audio','state_end_blocks','motion_mean','motion_std'):np.testing.assert_array_equal(target[k],reference[k])
        proxy=float(np.std(target['audio'][-1000:])/peak);assert proxy>.05,'Case B Target is not sustained: stop before learning'
        detail=oracle['diagnostic_moments'];assert np.max(abs(detail[:,5]))>0
        assert detail[-1,14]>0 and detail[-1,15]>0,'Right motion unexpectedly hard zero'
        assert np.max(abs(contrast['diagnostic_moments'][:,5]))==0
        decomp_error=float(detail[:,48].max());assert decomp_error/max(1.,float(abs(oracle['diagnostic_samples'][:,-1]).max()))<1e-10
        write_json(OUT/'preflight/physics.json',dict(passed=True,case='B',target_coefficients=coeff,oracle_coefficients=oc,caseA_coefficients=ac,
            target_peak=peak,target_late_ac_ratio=proxy,oracle_components=error,oracle_J=.75*error['log_spectral']+.25*error['envelope'],
            source_r_tail_mean=float(detail[-1,5]),source_r_tail_std=float(detail[-1,17]),right_x_tail_std=float(detail[-1,14]),right_y_tail_std=float(detail[-1,15]),
            output_decomposition_max_abs_error=decomp_error,caseA_B_audio_RMSE=float(np.sqrt(np.mean((target['audio']-contrast['audio'])**2))),
            native_library_sha256=digest(LIBRARY),native_source_sha256=digest(ROOT/'native/model.cpp'),source_sha256=digest(__file__),
            generated_sounds=len(ledger),native_steps=sum(x['native_steps'] for x in ledger),wall_s=time.perf_counter()-started))
        np.savez_compressed(OUT/'preflight/filter.npz',sos=AudioDecimator().sos,initial_state=np.zeros((8,2)),factor=200,native_rate=10000000,sample_rate=50000)
        print(f'Physics gate passed:Case B coupled Target sustained,oracle exact,{len(ledger)} sounds;right coefficients nonzero',flush=True)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc));write_json(OUT/'preflight/physics.json',dict(passed=False,failure=failure,ledger=ledger,source_sha256=digest(__file__)))
        print('Physics gate failed; no learning:'+str(failure),flush=True)
    return failure is None


def seal():
    """Preserve a failed physical gate as a diagnostic study, never a SAC run."""
    assert not json.loads((OUT/'preflight/physics.json').read_text())['passed']
    tests=json.loads((OUT/'preflight/tests.json').read_text());assert tests['passed'] and tests['source_sha256']==digest(ROOT/'tests/test_issue27.py')
    if (OUT/'manifest.json').exists():raise FileExistsError('Frozen study preserved')
    old=json.loads((BASE/'manifest.json').read_text());config=dict(old['settings'],experiment_id='issue27_caseB_self_target')
    versions={p:importlib.metadata.version(p) for p in old['versions']};assert versions==old['versions']
    write_json(OUT/'manifest.json',dict(issue=27,created_utc=datetime.now(timezone.utc).isoformat(),status='stopped_preflight',planned_settings=config,versions=versions,
        source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),probe_sha256=digest(PROBE),target_sha256=digest(TARGET),
        plan_sha256=digest(OUT/'PLAN.md'),preflight_passed=False,initial_checks=tests['initialization_checks'],baseline_manifest_sha256=digest(BASE/'manifest.json'),
        conditions=dict(model='B',target='B',alpha_l=1.2e6,beta_l=1500,alpha_r=1.2e6,beta_r=1500,epsilon_r=3.5e7,f0_r=1e7),
        planned_runs=3,planned_training_candidates=12288,actual_training_runs=0,actual_training_candidates=0,actual_gradient_updates=0,
        no_0p10_gate=True,source_link='https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/27'))
    print('Diagnostic study sealed:preflight failed,0 training runs,0 optimizer updates',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('physics','seal'));p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue27-before.json'));a=p.parse_args()
    if a.stage=='physics':
        if not physics(a.before):raise SystemExit(2)
    else:seal()
