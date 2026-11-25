"""Freeze wide-domain behavior protocol and reference provenance before research runs."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import re
import time

import numpy as np

from .issue26_data import (ROOT,BASE,OUT,TARGET,SEEDS,PREFIX,LOCAL_PROBABILITY,RADII,LOW,HIGH,F0_BINS,
    environment,source_hashes,sb3_hashes,folder_for)
from .issue26_agent import make_agent,observer_signature
from .issue14 import effective_hyperparameters
from .issue25_agent import LIMITS
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest


def prepare(before_path=Path('/tmp/birdsong-issue26-before.json'),output=OUT):
    import torch
    from .model import LIBRARY
    torch.set_num_threads(1);output=Path(output);tick=time.perf_counter()
    if output.exists():raise FileExistsError('Existing study preserved')
    before=json.loads(Path(before_path).read_text())
    for r in before['files']:assert digest(ROOT/r['path'])==r['sha256'],r['path']
    references={};names={18:'phase1_3d_domain_diagnostics',22:'phase1_caseA_sac128_reward_weights',
        23:'phase1_caseA_sac128_stability_diagnostics',25:'phase1_caseA_sac128_critic_lr'}
    for issue,name in names.items():
        path=ROOT/'results'/name/'manifest.json';m=json.loads(path.read_text());assert m['target_sha256']==digest(TARGET)
        if issue!=18:
            assert m['native_library_sha256']==digest(LIBRARY)
            for p,h in m['source_sha256'].items():assert digest(ROOT/p)==h,p
        references[issue]=dict(manifest=str(path.relative_to(ROOT)),sha256=digest(path),target_sha256=m['target_sha256'],
            historical_library_sha256=m['native_library_sha256'],library_matches_current=m['native_library_sha256']==digest(LIBRARY))
    gate_path=ROOT/'results/phase1_restricted3d_weights/model_gate.json';gate=json.loads(gate_path.read_text())
    assert gate['passed'] and gate['x0_storage_only'] and all(p['exact_equal'] for p in gate['comparisons'])
    assert len(re.findall(r'\bx0\b',(ROOT/'native/model.cpp').read_text()))==1
    old=json.loads((BASE/'manifest.json').read_text());versions={p:importlib.metadata.version(p) for p in old['versions']};assert versions==old['versions']
    tests=json.loads(Path('/tmp/birdsong-issue26-preflight-tests.json').read_text());assert tests['passed']
    assert tests['test_source_sha256']==digest(ROOT/'tests/test_issue26.py')
    output.mkdir(parents=True);write_csv(output/'protected_before.csv',before.pop('files'));write_json(output/'storage_before.json',before)
    config=dict(old['settings']);config.update(experiment_id='issue26_wide_local_behavior',local_start=PREFIX,local_probability=LOCAL_PROBABILITY)
    checks=[]
    for seed in SEEDS:
        prior=json.loads((folder_for('A',seed)/'summary.json').read_text())
        for p,h in prior['artifact_sha256'].items():assert digest(folder_for('A',seed)/p)==h,p
        signatures=[]
        for condition in ('B','C','D'):
            env=environment(condition)
            try:
                agent=make_agent(condition,seed,env,config);effective=json.loads(json.dumps(effective_hyperparameters(agent)));assert effective==prior['effective_hyperparameters']
                signature=observer_signature(agent);signatures.append(signature)
                assert signature['learner']['policy_sha256']==prior['initial']['learner_before']['policy_sha256']
                assert agent.incumbent is None and agent.candidate_log==[]
                assert env.decode([-1,-1,-1])[[0,2,3]].tolist()==LOW.tolist() and env.decode([1,1,1])[[0,2,3]].tolist()==HIGH.tolist()
                checks.append(dict(seed=seed,condition=condition,effective_settings_exact=True,initial_policy_exact=True,no_initial_center=True))
            finally:env.close()
        for s in signatures[1:]:
            for k in ('policy_sha256','rng_sha256','optimizer_sha256','entropy_sha256'):assert s['learner'][k]==signatures[0]['learner'][k]
            assert s['choice_rng']==signatures[0]['choice_rng'] and s['candidate_rng']==signatures[0]['candidate_rng']
    write_json(output/'reference_audit.json',dict(references=references,current_library_sha256=digest(LIBRARY),
        issue18_difference='Historical library predates x0 .04→.02 assignment; existing Issue19 model gate shows identical saved waves and x0 storage-only. Do not rewrite old manifest.',
        existing_gate=str(gate_path.relative_to(ROOT)),existing_gate_sha256=digest(gate_path),same_target_all=True,generated_sounds=0))
    write_json(output/'test_execution.json',tests)
    plan=f'''# Issue26 — 計算前の凍結計画

AはIssue23の制限域[49000,800000]・SAC128・λ=.75・3LR=3e-4の3seedを読み取り専用参照。B/C/Dは全てf0_l=[49000,10000000]。Stage BのB3seed×4096候補を先に実行・記録し、結果が良くても悪くてもStage CのC/D×3seed×4096候補を省略しない。新規合計9run、warmup16・評価0,128,...,4096の33点・4080更新/optimizer/run。

最初256試行は全広域条件で標準SACの同じ生成方法。256の後はC/Dのみ、choice RNGでlocal80%/SAC20%。choice seed260026+seed、candidate seed260126+seedの独立PCG64を使い、学習のPython/NumPy/Torch RNGは使わない。条件C/Dでchoice系列を共通にする。毎試行、標準SAC提案を先に生成し、local試行だけ実行行動とReplay行動の双方を置き換える。local試行の未実行SAC提案は別欄に記録する。

中心は過去の実学習候補の最初のminJ（同点は先着）。独立評価・Target oracleは使わず、非持続も除外しない。Cの物理半径[625000,37500,50000]、Dは[625000,37500,250000]。各次元の[center−radius,center+radius]とglobal boundsの共通区間で連続一様サンプリング。格子の刻みではない。float32写像の端点超過だけ最大16回まで再抽選し、クリップせず、attempt数と丸め誤差を記録する。実行はfloat32正規化の丸めを伴い、無限精度の連続値そのものではない。

中心は実行された制御値で更新し、実行行動・環境の物理値・Replay stored action/rewardを全試行で一致検査する。隣接試行差が半径を超えることを許す。通常SAC20%は全域の可能性を残すが、一様な全域被覆を保証しない。初めの256はlocal0%なので、全4096での期待local率は75%、257以降の期待率80%。

数値停止は既存と同じ：非有限、負のAdam二次moment、勾配L2>{LIMITS['gradient_l2']:g}、重み絶対値>{LIMITS['parameter_abs']:g}、moment絶対値>{LIMITS['optimizer_moment_abs']:g}で停止して失敗保存。条件削除・再始動・幅調整・oracleによる中心初期化はしない。

事前テストは3seed×標準SAC/B/C/Dで256試行の行動・報酬・ネットワーク/optimizer/RNG/Replayが一致。正式B/C/Dの各seedでも最初256件と128/256時点の全学習器署名を照合する。独立評価・ログ・保存では学習RNG、local RNG、中心が不変であることを検査。

発見=学習候補のminJとその試行・共通E/発声。獲得=決定論的Actor33点。保持=Selected→Final。Finalは4096終了後、Selectedは33点の最初minJ、posthocは同点の最初min共通E。学習bestをActorの出力と同一視しない。Bのprefix256モデル/波形だけ追加保持する。

相対的な到達基準を事前固定：(1)学習試行257以降でJ≤0.5×そのseedの最初256学習候補minJ、(2)独立ActorのE≤0.5×同条件/seedの初期Actor E、(3)256以降のActor J≤1.25×その条件/seedのprefix256 minJ。Aは自身の既存prefixで、B/C/Dは同一prefix。これらは記述指標で合否・追加計算のゲートではない。E_total≤.10の達成を今回の目標として最適化しない。

f0分布の固定区間は{list(F0_BINS)}、高域割合は>800000、>2500000、>5000000。各区間の発声・ε_l/p_sの組合せも示し、高f0を一律に無音・不要と断定しない。local hit=直前incumbentのJを厳密に改善した件数。local/SACを分け、非持続/失敗を除外せず、実偏差・境界共通区間・中心履歴・実割合を残す。

Case A、0.20s/50kHz/10000sample、dt1e-7、x0=.02cm/D=.01、初期状態/遅延0、同一Target/hash、ε_l[2.5e7,5e7]・p_s[2.5e6,4e6]・右ε_r3.5e7/f0_r49000を固定。r=−(.75E_spec+.25E_env)、共通E=E_spec+E_env、2倍補正なし。f0_lはHzでなく開閉外力係数。Actor/Critic[128,128]、全LR3e-4、auto_0.01/target_entropy−3、batch32/buffer50000/gamma1/train_freq1/gradient_steps1/CPU1threadを保持する。

A/Bは物理範囲、B対C/Dはoff-policy行動データ生成アルゴリズム、C/Dはf0局所半径の差と解釈する。正規化行動やstdはA/Bで物理尺度が違う。元音・共通gain/色基準の表示を保存。追加の探索グリッド・Case B・時変・実録音・学習率/entropy調整・最良方策復帰は行わない。Case A全体の総括と教員共有、Case B自身のTarget生成・Issue20旧仕様の改訂は次の作業で、今回外部へ送信しない。
'''
    (output/'PLAN.md').write_text(plan)
    write_json(output/'preflight.json',dict(passed=True,generated_sounds=0,initial_checks=checks,source_sha256=digest(__file__),wall_s=time.perf_counter()-tick))
    write_json(output/'manifest.json',dict(issue=26,created_utc=datetime.now(timezone.utc).isoformat(),git_head=before['git_head'],settings=config,
        source_sha256=source_hashes(),native_library_sha256=digest(LIBRARY),target_sha256=digest(TARGET),versions=versions,
        baseline_manifest_sha256=digest(BASE/'manifest.json'),sb3_implementation_sha256=sb3_hashes(),plan_sha256=digest(output/'PLAN.md'),
        conditions=dict(A='existing restricted SAC',B='wide standard SAC',C='wide local80% radius50000',D='wide local80% radius250000'),
        prefix=PREFIX,local_probability=LOCAL_PROBABILITY,local_radii={k:v.tolist() for k,v in RADII.items()},global_low=LOW.tolist(),global_high=HIGH.tolist(),
        local_choice_seed=260026,local_candidate_seed=260126,numerical_limits=LIMITS,research_runs=9,
        relative_criteria=dict(training_J_ratio_to_prefix=.5,actor_E_ratio_to_initial=.5,actor_J_ratio_to_prefix=1.25),
        f0_bins=list(F0_BINS),no_0p10_gate=True,source_link='https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/26'))
    print('Stage A frozen:wide bounds,continuous local proposals,exact Replay,256 prefix,relative criteria;no .10 gate',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue26-before.json'));a=p.parse_args();prepare(a.before)
