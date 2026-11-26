"""Eighteen fixed physical candidates; select before algorithm smoke/full runs."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.signal import find_peaks,welch
from .issue27f_data import ROOT,OUT,OLD,BASE,TARGET,GRID_E,GRID_P,RIGHT_GROUPS,hashes
from .issue27f_native import build_probe,render,PROBE
from .model import LIBRARY
from .acoustics import distance
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest

CONFIG=dict(episodes=4096,eval_every_episodes=128,learning_rate=3e-4,gamma=1.,batch_size=32,buffer_size=50000,warmup_episodes=16,
    train_freq=1,gradient_steps=1,sac_ent_coef='auto_0.01',td3_noise_std=.1,ppo_n_steps=128,ppo_n_epochs=10,ppo_clip_range=.2)

def features(values):
    a=values['audio'];peak=max(float(abs(a).max()),1e-8);tail=a[-1000:];tx=np.arange(1000)/50000
    residual=tail-np.polyval(np.polyfit(tx,tail,1),tx);ratio=float(tail.std()/peak);dr=float(residual.std()/peak)
    f,p=welch(a[-5000:],fs=50000,nperseg=2048,noverlap=1024,detrend='constant');keep=(f>=100)&(f<=10000);p=p[keep];f=f[keep]
    prob=p/max(float(p.sum()),1e-30);entropy=float(-np.sum(prob*np.log(np.maximum(prob,1e-30)))/np.log(len(p)))
    peaks,_=find_peaks(p,prominence=float(p.max())*.005,distance=2);significant=peaks[p[peaks]>=p.max()*.01]
    points=values['diagnostic_samples'][-1000:];side=[];dstd=[]
    for j in range(4):
        v=points[:,j];d=v-np.polyval(np.polyfit(tx,v,1),tx);s=float(d.std());dstd.append(s)
        side.append(s/max(float(v.std()),1e-30)>.5 and s/max(float(np.ptp(values['diagnostic_samples'][:,j])),1e-30)>.001)
    return dict(peak=peak,late_ac_ratio=ratio,detrended_ac_ratio=dr,sustained_proxy=ratio>.05,eligible=ratio>.05 and dr>.05,
        spectral_entropy=entropy,spectral_peak_count=len(significant),dominant_frequency_Hz=float(f[np.argmax(p)]),
        left_motion_periodic_proxy=bool(side[0] and side[1]),right_motion_periodic_proxy=bool(side[2] and side[3]),motion_detrended_std=dstd)

def scan(before=Path('/tmp/birdsong-issue27f-before.json')):
    if OUT.exists():raise FileExistsError('Existing experiment preserved')
    r=json.loads(Path(before).read_text())
    for x in r['files']:assert digest(resolve_result_path(ROOT/x['path']))==x['sha256'],x['path']
    OUT.mkdir(parents=True);(OUT/'preflight/candidates').mkdir(parents=True)
    write_csv(OUT/'protected_before.csv',r.pop('files'));write_json(OUT/'storage_before.json',r)
    (OUT/'PLAN.md').write_text('''# Issue27再改訂 — 左右外力18点と3手法の事前計画

以前のIssue17〜27の全成果を保持し、新しい独立フォルダへ保存する。native/model.cpp/元model.py/実libraryの式・係数は不変。Case Bのnative condition1と左右α1.2e6/β1500を維持、ε_r3.5e7固定、x0=.02cm/D=.01/dt1e-7、0.20s/50kHz/10000sample、初期4状態/遅延/16次15kHz lowpass状態0を保持。

元PythonのBirdModel(B)は右f0を1e7へ固定するため、別のConfiguredCaseB subclassの_controlsだけで今回の固定右f0を設定する。native resetはB/1を呼び、右係数を0にするAや別条件へ置換しない。実メモリの右f0と左右αβを全経路で検査する。

Stage Aは①左3e5/右1e7の旧9音を読み取り専用参照。②左49000/右1e7、③左49000/右49000をε_l[3e7,3.5e7,4.1e7]×p_s[3e6,3.4e6,3.8e6]で各9点、新規18点のみ。各2,000,000 native step、計36,000,000。全点の波形/state/source/output/全刻み統計を保存。末尾1000sample AC std/自身peak>.05、finite、generated_silent偽を候補とし、直線除去後ratio>.05もドリフト除外に確認する。左右の周期的変動proxyは末尾traceのde-trended stdとdrift割合で別に記述し、生理的発声とは呼ばない。

選定はp_s3.0e6を優先し、持続性・波形・周波数構造・spectral entropy/peak数を物理診断で確認。左右の持続変動と相互作用が確認された③も積極的に検討。RL誤差を一切見ず候補1つと理由を先に凍結する。18点不適格なら追加探索/時間延長せず停止。選定音は別Bインスタンスでoracleを1音再生し、波形・状態・native運動統計の一致とE_spec/E_env/E/J=0、右source/output非ゼロを検査。右f0はTargetに対応する値を9run全てで固定し、行動へ含めない。

Stage BはSAC/TD3/PPO×seed0/1/2の9run、各厳密4096環境step＝音、合計36864学習音。独立評価0,128,...,4096の33点/runは別会計。一定3D、観測8D reset[-1,0,...]、1episode1行動。ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]を探索し、Target生成の左f0固定を学習へ引き継がない。r=−(.75E_spec+.25E_env)、共通E=E_spec+E_env。

CPU1thread、全隠れ層128×128/LR3e-4/gamma1/batch32。SAC:buffer50000,warmup16,train_freq1,gradient_steps1,auto_0.01,target_entropy−3。TD3:同buffer/warmup等、NormalActionNoise(0,.1)、既定policy_delay2等。PPO:pi/vf別128×128,n_steps128,n_epochs10,clip.2、他既定（max_grad_norm.5等）。全層/実パラメータ数をassert。SAC optimizer各4080、TD3 Critic4080/Actor2040、PPO32rollout×40minibatch=1280 optimizer step（内部epoch counter320）を予定し、更新数一致とは呼ばない。

各手法128step・seed0のsmokeを別計上、評価0/128と保存3方策+training_bestの4再生/本を検査。標準と計測付き実装を3手法×128step×2のsynthetic環境で照合し、9初期seedと構造も確認（native0）。SAC/TD3の実行行動/Replay/報酬/終端を検査。PPOは元Gaussian raw行動をrolloutへ保存し、環境実行clip行動と分ける標準仕様を保持する。全optimizerの実効LR・有限性・Adam状態を観測、既存Issue25の数値停止基準を適用。PPO既定gradient clipは変更しない。

評価は128step後の全更新完了後（PPOの次rollout開始時、最終はlearn終了後）とし、学習器/optimizer/Buffer/RNG/学習bestを変えず評価を学習へ混入しない。Final=4096終了後、Selected=33点の最初minJ、posthoc=最初minE、training_best=実学習の最初minJを別記する。最良方策へ復帰しない。E≤.10は合否/追加計算/やり直しのゲートにしない。failed runは保持し置換/追加学習しない。

完了後は3役割Actorと学習bestの計36音をBで詳細独立再生し、波形/制御/Jと右sourceを確認。全seedの生音と共通gain WAV、共通Target PSD色軸のTarget最上段→SAC/TD3/PPO比較図、包絡/左右状態を保存する。代表seed1を事前固定。①②③の27音は旧①PSD基準・色軸を維持した図も保存する。異なる人工TargetのA/B誤差から構造優位を主張せず、将来の共通実録音比較と分ける。3seedで一般的手法優位・制御の一意復元を主張しない。TQC、直接最適化、時変、5D、実録音、追加grid、固定物理係数変更、commit/push/Issue更新/外部送信はしない。
''')
    build=build_probe();assert build['unchanged_source_build_binary_exact']
    data=[];ledger=[];started=time.perf_counter()
    for group,right in RIGHT_GROUPS:
        for i,e in enumerate(GRID_E):
            for j,p in enumerate(GRID_P):
                name=f'b{group}_c{i}{j}';controls=np.array([e,3.5e7,p,49000.,right]);entry=dict(candidate=name,condition='B',status='started',requested_native_steps=2000000);ledger.append(entry)
                write_json(OUT/'preflight/ledger.json',ledger)
                try:
                    values,coeff=render(controls,'B',detailed=True);f=features(values)
                    comp=distance(values['audio'],values['audio'],50000,scale=f['peak']);f['eligible']=f['eligible'] and not comp['generated_silent']
                    np.savez_compressed(OUT/f'preflight/candidates/{name}.npz',**values,sample_rate=50000,duration_s=.2,model_condition='B')
                    q=values['diagnostic_moments'][-1]
                    row=dict(candidate=name,group=group,row=i,column=j,epsilon_l=e,ps=p,f0_l=49000.,f0_r=right,failed=False,generated_silent=comp['generated_silent'],**f,
                        source_r_tail_mean=float(q[5]),source_r_tail_std=float(q[17]),output_r_tail_mean=float(q[9]),native_coefficients=coeff,raw_npz=f'preflight/candidates/{name}.npz')
                    entry.update(status='complete',native_steps=2000000,episodes=1);data.append(row)
                    print(json.dumps({k:row[k] for k in ('candidate','late_ac_ratio','eligible','spectral_entropy','spectral_peak_count','left_motion_periodic_proxy','right_motion_periodic_proxy')}),flush=True)
                except Exception as exc:
                    entry.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)));data.append(dict(candidate=name,group=group,epsilon_l=e,ps=p,failed=True,eligible=False,error=str(exc)))
                write_json(OUT/'preflight/ledger.json',ledger);write_json(OUT/'preflight/candidates.json',data)
    write_csv(OUT/'preflight/candidates.csv',[{k:v for k,v in r.items() if k!='native_coefficients'} for r in data])
    write_json(OUT/'preflight/scan.json',dict(complete=True,candidates=18,eligible=sum(r['eligible'] for r in data),generated_sounds=len(ledger),wall_s=time.perf_counter()-started))

def select(candidate,reason):
    records=json.loads((OUT/'preflight/candidates.json').read_text());eligible=[r for r in records if r['eligible']]
    if not eligible:raise RuntimeError('All18 candidates failed; do not start learning or additional search')
    row=next(r for r in eligible if r['candidate']==candidate)
    if TARGET.exists():raise FileExistsError('Frozen Target preserved')
    shutil.copyfile(OUT/row['raw_npz'],TARGET)
    with np.load(TARGET) as z:target={k:z[k].copy() for k in z.files}
    ledger=json.loads((OUT/'preflight/ledger.json').read_text());ledger.append(dict(purpose='independent_B_oracle',status='started',requested_native_steps=2000000));write_json(OUT/'preflight/ledger.json',ledger)
    oracle,coeff=render(target['controls'],'B',detailed=False)
    np.savez_compressed(OUT/'preflight/oracle.npz',**oracle,sample_rate=50000,duration_s=.2,model_condition='B')
    for k in ('audio','state_end_blocks','motion_mean','motion_std'):np.testing.assert_array_equal(target[k],oracle[k])
    errors=distance(target['audio'],oracle['audio'],50000,scale=float(abs(target['audio']).max()));assert errors['total']==0
    assert abs(row['source_r_tail_mean'])+row['source_r_tail_std']>0
    ledger[-1].update(status='complete',native_steps=2000000,episodes=1);write_json(OUT/'preflight/ledger.json',ledger)
    choice=dict(candidate=candidate,controls=target['controls'].tolist(),reason=reason,target_sha256=digest(TARGET),candidate_sha256=digest(OUT/row['raw_npz']),
        physical_features={k:row[k] for k in ('late_ac_ratio','detrended_ac_ratio','spectral_entropy','spectral_peak_count','left_motion_periodic_proxy','right_motion_periodic_proxy')})
    assert choice['target_sha256']==choice['candidate_sha256'];write_json(OUT/'preflight/target_choice.json',choice)
    write_json(OUT/'preflight/gate.json',dict(passed=True,choice=choice,oracle_components=errors,oracle_J=0.,native_coefficients=coeff,generated_sounds=19))
    print('Target fixed before learning:'+candidate+'; independent B oracle exact',flush=True)

def freeze():
    from .issue24_data import sb3_hashes
    gate=json.loads((OUT/'preflight/gate.json').read_text());tests=json.loads((OUT/'preflight/tests.json').read_text());assert gate['passed'] and tests['passed'] and source_hashes_match(tests['source_sha256'], hashes())
    if (OUT/'manifest.json').exists():raise FileExistsError('Existing official manifest preserved')
    old=json.loads((BASE/'manifest.json').read_text());versions={p:importlib.metadata.version(p) for p in old['versions']};assert versions==old['versions']
    write_json(OUT/'manifest.json',dict(issue=27,revision='2026-11-26_18points',created_utc=datetime.now(timezone.utc).isoformat(),settings=CONFIG,
        source_sha256=hashes(),native_library_sha256=digest(LIBRARY),probe_sha256=digest(PROBE),target_sha256=digest(TARGET),plan_sha256=digest(OUT/'PLAN.md'),
        preflight_passed=True,sb3_implementation_sha256=sb3_hashes(),versions=versions,Target_choice=gate['choice'],methods=['SAC','TD3','PPO'],seeds=[0,1,2],research_runs=9,
        fixed_right_f0=gate['choice']['controls'][4],no_0p10_gate=True,representative_audio_seed=1))
    print('Official9run protocol frozen',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('scan','select','freeze'));p.add_argument('--candidate');p.add_argument('--reason');p.add_argument('--before',type=Path,default=Path('/tmp/birdsong-issue27f-before.json'));a=p.parse_args()
    scan(a.before) if a.stage=='scan' else select(a.candidate,a.reason) if a.stage=='select' else freeze()
