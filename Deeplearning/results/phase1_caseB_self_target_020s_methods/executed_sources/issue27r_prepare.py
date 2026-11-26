"""Nine predefined Case B sounds, physical Target selection, then protocol freeze."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .issue27r_data import ROOT,OUT,BASE,TARGET,OLD,GRID_E,GRID_P,hashes
from .issue27r_native import build_probe,render,PROBE
from .model import LIBRARY
from .acoustics import distance
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest

def scan(before=Path('/tmp/birdsong-issue27r-before.json')):
    if OUT.exists():raise FileExistsError('Existing revised study preserved')
    r=json.loads(Path(before).read_text())
    for item in r['files']:assert digest(ROOT/item['path'])==item['sha256'],item['path']
    OUT.mkdir(parents=True);(OUT/'preflight/candidates').mkdir(parents=True)
    write_csv(OUT/'protected_before.csv',r.pop('files'));write_json(OUT/'storage_before.json',r)
    (OUT/'PLAN.md').write_text('''# Issue27改訂 — 計算前の凍結計画

旧Issue27停止記録、旧Case A17〜26、native/model.cppと実libraryを保持する。新規成果はこの独立フォルダへ保存。Case Bの左右α=1.2e6/β1500、右ε_r3.5e7/f0_r1e7、x0=.02cm/D=.01、dt1e-7を維持する。全0.20s一定、4状態・遅延・16次15kHz filter=0から50kHz/10000sampleを生成。

Stage Aはε_l[3e7,3.5e7,4.1e7]×p_s[3e6,3.4e6,3.8e6]の9条件のみ。左f0_l=3e5、右固定。各点の末尾20ms AC std/自身peak>.05、finite、generated_silent偽を基本候補とし、直線除去後std/自身peak>.05も要求してドリフトだけの候補を除く。波形・左右動態を確認し、p_sは3.0→3.4→3.8e6の順を優先、同じp_sでは直線除去後ratioが高い順（同点はε_l昇順）にTargetを選ぶ。これは物理指標であり学習誤差を見ない。候補0なら本学習へ進まず追加探索しない。

選定後、同じBの別インスタンスでoracleを1音、同じ5制御でA対照を1音生成する。B oracleの波形/状態/運動統計とE_spec/E_env/E/Jが0誤差、B右source/output非ゼロ、A右source0を確認。Target byte列/hash・選定根拠を9run共通で固定する。

Stage BはSAC/TD3/PPO×seed0/1/2の9run、各厳密4096環境step/音。独立評価は0,128,...,4096の33点で予算と別会計。f0_l範囲[49000,800000]、ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、右固定。観測8D reset[-1,0,...]、1episode1行動。r=−J、J=.75E_spec+.25E_env、共通E=E_spec+E_env。

全ネットの隠れ層128×128、LR3e-4、gamma1、batch32、CPU1thread。SACはbuffer50000/warmup16/train_freq1/gradient_steps1/auto_.01/target_entropy−3。TD3は同buffer/warmup/train_freq/gradient_steps、NormalActionNoise(0,.1)、その他標準policy_delay2等。PPOはpi/vfを別の128×128、n_steps128/n_epochs10/clip_range.2、その他既定。PPOは128rollout×32回で4096、40minibatch/rollout×32=1280 optimizer step、内部epoch counter320。SACはActor/Critic/温度各4080、TD3はCritic4080/Actor2040で、更新数は同一とは呼ばない。

正式run前に各手法128stepのsmokeをseed0で1本ずつ、2独立評価/本と保存モデル/学習best4音再生/本を別会計。標準実装と計測付き実装の128step synthetic試験を3手法で照合（native0）。各seedの初期化と実ネット層/総パラメータ数もassert。smokeは本番に混ぜない。

SAC/TD3は実行正規化行動とReplay action/reward/終端を照合。PPOの標準Gaussianは環境へclipして実行するが、rolloutには元の非clip行動を保存してlog_probと一致させる。この標準PPOの仕様を変えず、raw/実行行動を別欄に記録する。全optimizerのLR・有限性・Adam状態を観測し既存Issue25の数値停止基準を適用、失敗は置換しない。PPOの既定max_grad_norm=.5は保持する。

評価は各128環境step後の全更新を終えてから実行し、PPOのrollout終了/更新前との混同を避ける。Finalは4096終了後、Selectedは33点の最初minJ、posthocは最初minE、training_bestは実行学習候補の最初minJ。評価・保存では学習器/optimizer/Buffer/RNG/最良学習候補を変えず、最良方策へ復帰しない。E≤.10は目標/合否/延長/追加計算のゲートにしない。

Final/Selected/posthoc/training_bestの36音を独立に詳細再生し、波形・J・制御一致、左右末尾運動とsource/output寄与を監査する。全音のWAVは選定Target peak由来の共通gain、表示PSDは同Target基準で共通色軸。代表音はseed1、全seedの数値を保存。3seed平均±標本SD、未完走/非持続も表示し、制御値の一意復元や一般的手法優位を主張しない。

今回の比較はCase B内部の3手法。別人工TargetのCase A誤差から構造優位を結論しない。共通実録音Targetでの2音源構造比較は将来。TQC、直接最適化、時変、5D、左右切替、実録音、局所探索、物理式/固定係数変更、許可外の追加探索は行わない。commit/push/Issue更新/外部送信なし。
''')
    build=build_probe();assert build['unchanged_source_build_binary_exact'],'Investigate source/binary mismatch before physics'
    records=[];ledger=[];t0=time.perf_counter()
    for i,e in enumerate(GRID_E):
        for j,p in enumerate(GRID_P):
            name=f'c{i}{j}';controls=np.array([e,3.5e7,p,3e5,1e7]);entry=dict(purpose='grid',candidate=name,condition='B',status='started',requested_steps=2000000);ledger.append(entry)
            write_json(OUT/'preflight/ledger.json',ledger)
            try:
                values,coeff=render(controls,'B',detailed=True);audio=values['audio'];peak=float(abs(audio).max());tail=audio[-1000:];x=np.arange(1000)/50000
                residual=tail-np.polyval(np.polyfit(x,tail,1),x);comp=distance(audio,audio,50000,scale=max(peak,1e-8));ratio=float(tail.std()/max(peak,1e-8));detrended=float(residual.std()/max(peak,1e-8))
                np.savez_compressed(OUT/f'preflight/candidates/{name}.npz',**values,sample_rate=50000,duration_s=.2,model_condition='B')
                q=values['diagnostic_moments'][-1];eligible=ratio>.05 and detrended>.05 and not comp['generated_silent']
                records.append(dict(candidate=name,row=i,column=j,epsilon_l=e,ps=p,f0_l=3e5,failed=False,generated_silent=comp['generated_silent'],
                    sustained_proxy=ratio>.05,eligible=eligible,peak=peak,late_ac_ratio=ratio,detrended_ac_ratio=detrended,
                    motion_tail_mean=q[:4].tolist(),motion_tail_std=q[12:16].tolist(),source_l_tail_mean=float(q[4]),source_r_tail_mean=float(q[5]),
                    source_r_tail_std=float(q[17]),output_r_tail_mean=float(q[9]),native_coefficients=coeff,raw_npz=f'preflight/candidates/{name}.npz'))
                entry.update(status='complete',native_steps=2000000,episodes=1)
                print(json.dumps({k:records[-1][k] for k in ('candidate','epsilon_l','ps','late_ac_ratio','detrended_ac_ratio','eligible')}),flush=True)
            except Exception as exc:
                entry.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
                records.append(dict(candidate=name,row=i,column=j,epsilon_l=e,ps=p,failed=True,eligible=False,error=str(exc)))
            write_json(OUT/'preflight/ledger.json',ledger);write_json(OUT/'preflight/candidates.json',records)
    write_csv(OUT/'preflight/candidates.csv',[{k:v for k,v in r.items() if k!='native_coefficients'} for r in records])
    ranked=sorted([r for r in records if r['eligible']],key=lambda r:(r['ps'],-r['detrended_ac_ratio'],r['epsilon_l']))
    write_json(OUT/'preflight/scan.json',dict(complete=True,candidates=9,eligible=len(ranked),ranked=[r['candidate'] for r in ranked],generated_sounds=len(ledger),wall_s=time.perf_counter()-t0))

def select():
    scan=json.loads((OUT/'preflight/scan.json').read_text());assert scan['complete']
    if not scan['ranked']:
        write_json(OUT/'preflight/gate.json',dict(passed=False,reason='All nine fixed candidates non-sustained/inappropriate; no additional search or learning'))
        print('All nine candidates fail; stop before Stage B',flush=True);return
    if TARGET.exists():raise FileExistsError('Existing Target selection preserved')
    rows=json.loads((OUT/'preflight/candidates.json').read_text());r=next(r for r in rows if r['candidate']==scan['ranked'][0]);shutil.copyfile(OUT/r['raw_npz'],TARGET)
    with np.load(TARGET) as z:target={k:z[k].copy() for k in z.files}
    controls=target['controls'];oracle,coeff=render(controls,'B',detailed=False);contrast,ac=render(controls,'A',detailed=True)
    ledger=json.loads((OUT/'preflight/ledger.json').read_text())
    for name,case,values in (('oracle','B',oracle),('selected_controls_caseA','A',contrast)):
        np.savez_compressed(OUT/f'preflight/{name}.npz',**values,sample_rate=50000,duration_s=.2,model_condition=case)
        ledger.append(dict(purpose=name,condition=case,status='complete',native_steps=2000000,episodes=1))
    write_json(OUT/'preflight/ledger.json',ledger)
    for key in ('audio','state_end_blocks','motion_mean','motion_std'):np.testing.assert_array_equal(target[key],oracle[key])
    error=distance(target['audio'],oracle['audio'],50000,scale=float(abs(target['audio']).max()));assert error['total']==0
    assert r['source_r_tail_mean']!=0 and np.max(abs(contrast['diagnostic_moments'][:,5]))==0
    choice=dict(candidate=r['candidate'],controls=controls.tolist(),reason='Predefined ps priority, then highest detrended sustained ratio; waveform/internal-state review before learning',
        target_sha256=digest(TARGET),source_candidate_sha256=digest(OUT/r['raw_npz']),late_ac_ratio=r['late_ac_ratio'],detrended_ac_ratio=r['detrended_ac_ratio'])
    assert choice['target_sha256']==choice['source_candidate_sha256'];write_json(OUT/'preflight/target_choice.json',choice)
    write_json(OUT/'preflight/gate.json',dict(passed=True,choice=choice,oracle_components=error,oracle_J=0.,native_coefficients=coeff,caseA_coefficients=ac,
        old_failed_experiment_manifest_sha256=digest(OLD/'manifest.json'),new_sounds=11,old_outputs_modified=0))
    print('Selected '+r['candidate']+'; B oracle exact; Case B coupled;11 Stage A sounds',flush=True)

def freeze():
    from .issue24_data import sb3_hashes
    gate=json.loads((OUT/'preflight/gate.json').read_text());assert gate['passed']
    smoke=json.loads((OUT/'preflight/tests.json').read_text());assert smoke['passed'] and smoke['source_sha256']==hashes()
    if (OUT/'manifest.json').exists():raise FileExistsError('Official manifest already frozen')
    config=dict(episodes=4096,eval_every_episodes=128,learning_rate=3e-4,gamma=1.,batch_size=32,buffer_size=50000,warmup_episodes=16,
        train_freq=1,gradient_steps=1,sac_ent_coef='auto_0.01',td3_noise_std=.1,ppo_n_steps=128,ppo_n_epochs=10,ppo_clip_range=.2)
    old=json.loads((BASE/'manifest.json').read_text());versions={p:importlib.metadata.version(p) for p in old['versions']};assert versions==old['versions']
    write_json(OUT/'manifest.json',dict(issue=27,revision='2026-11-26',created_utc=datetime.now(timezone.utc).isoformat(),settings=config,
        preflight_passed=True,source_sha256=hashes(),native_library_sha256=digest(LIBRARY),probe_sha256=digest(PROBE),target_sha256=digest(TARGET),
        plan_sha256=digest(OUT/'PLAN.md'),sb3_implementation_sha256=sb3_hashes(),versions=versions,Target_choice=gate['choice'],
        methods=['SAC','TD3','PPO'],seeds=[0,1,2],research_runs=9,no_0p10_gate=True,representative_audio_seed=1))
    print('Frozen:SAC/TD3/PPO×3 seeds×4096 steps',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('scan','select','freeze'));p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue27r-before.json'));a=p.parse_args()
    scan(a.before) if a.stage=='scan' else select() if a.stage=='select' else freeze()
