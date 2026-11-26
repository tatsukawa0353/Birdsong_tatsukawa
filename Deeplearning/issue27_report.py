"""Scientific diagnostic report after the mandatory Case B Target gate stops SAC."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from .issue27_data import ROOT,OUT,BASE,TARGET,source_hashes
from .issue27_native import PROBE
from .issue27_diagnostics import audit,load,rows,SOUNDS
from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue19_protection import workspace_inventory
from .issue23_stage_a import table as plain_table
from .issue15_analysis import write_json
from .issue16 import digest
from .model import LIBRARY


def table(headers,data):
    escape=lambda v:str(v).replace('|','\\|')
    return plain_table([escape(h) for h in headers],[[escape(v) for v in row] for row in data])


def figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
    (OUT/'figures').mkdir(exist_ok=True);images=[];data={s:load(s) for s in SOUNDS}
    target=data['target']['audio'];peak=float(abs(target).max());f,t,p,reference=display_psd(target)
    psds={s:display_psd(v['audio'])[2] for s,v in data.items()};maximum=max(float(relative_db(p,reference).max()) for p in psds.values())
    clim=(-80.,max(0.,float(np.ceil(maximum/5)*5)));wave_lim=1.05*max(float(abs(v['audio']/peak).max()) for v in data.values())
    rms_lim=1.05*max(float(rms_envelope(v['audio'],peak).max()) for v in data.values())
    def save(fig,name,description):
        path=OUT/f'figures/{name}.png';fig.savefig(path,dpi=130);plt.close(fig)
        images.append(dict(path=str(path.relative_to(OUT)),sha256=digest(path),description=description))
    labels=dict(target='Case B Target: specified controls',oracle='Case B independent oracle: same controls',same_controls_caseA='Case A control contrast: same five controls')
    fig,axes=plt.subplots(3,3,figsize=(16,10));mesh=None
    for i,name in enumerate(SOUNDS):
        audio=data[name]['audio'];times=np.arange(len(audio))/50000
        mesh=axes[i,0].pcolormesh(t,f/1000,relative_db(psds[name],reference),cmap='magma',shading='auto',vmin=clim[0],vmax=clim[1])
        axes[i,0].set(title=labels[name],xlabel='Time [s]',ylabel='Frequency [kHz]',xlim=(t[0],t[-1]),ylim=(0,10))
        axes[i,1].plot(times,audio/peak,lw=.7,color='black');axes[i,1].set(title='Full raw waveform / B Target peak',xlabel='Time [s]',ylabel='Amplitude / B peak',xlim=(0,.2),ylim=(-wave_lim,wave_lim))
        env=rms_envelope(audio,peak);axes[i,2].plot(times,env,color='black');axes[i,2].axvspan(.18,.2,color='#0072B2',alpha=.12)
        axes[i,2].set(title='5ms RMS, shared scale; shaded=last20ms',xlabel='Time [s]',ylabel='RMS / B peak',xlim=(0,.2),ylim=(0,rms_lim))
    fig.colorbar(mesh,ax=axes[:,0],label='dB re B Target full-band PSD peak')
    save(fig,'audio_contrast','All three newly generated physical diagnostics; one B Target PSD/gain, full waveforms/RMS, no per-sound normalization')
    colors={'oracle':'#0072B2','same_controls_caseA':'#D55E00'};fig,axes=plt.subplots(4,2,figsize=(15,12))
    for j,name in enumerate(('x_l','y_l','x_r','y_r')):
        for col,(lo,hi) in enumerate(((0.,.2),(.18,.2))):
            ax=axes[j,col]
            for sound,color in colors.items():
                points=data[sound]['diagnostic_samples'];time=np.arange(1,len(points)+1)/50000;mask=(time>=lo)&(time<=hi)
                ax.plot(time[mask],points[mask,j],color=color,lw=.8,label='Case B coupled' if sound=='oracle' else 'Case A: right acoustic coefficients0')
            ax.set(title=name+(': full0.20s' if col==0 else ': last20ms, zoomed axis'),xlabel='Time [s]',ylabel='Model internal '+name,xlim=(lo,hi));ax.legend(fontsize=8)
    save(fig,'motion_transients','State traces sampled every200 actual native steps; tail axes zoomed; statistics use every native step and include drift')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for i,sound in enumerate(('oracle','same_controls_caseA')):
        points=data[sound]['diagnostic_samples'];time=np.arange(1,len(points)+1)/50000
        for col,(lo,hi) in enumerate(((0.,.2),(.18,.2))):
            ax=axes[i,col];mask=(time>=lo)&(time<=hi)
            for index,label,color in ((4,'Left source L','#009E73'),(5,'Right source R','#D55E00'),(6,'Delayed feedback','#777777'),(7,'Common pi_tilde','#0072B2')):
                ax.plot(time[mask],points[mask,index],color=color,lw=.8,label=label)
            ax.set(title=('Case B coupled' if i==0 else 'Case A acoustic contrast')+(': full' if col==0 else ': tail zoom'),xlabel='Time [s]',ylabel='Native model terms',xlim=(lo,hi));ax.legend(fontsize=8)
    save(fig,'acoustic_contributions','pi_tilde=L+R−gamma*delayed_pi; right B term is nonzero, A term0; additive current-state decomposition, not isolated causal sound')
    write_json(OUT/'figures.json',images)
    write_json(OUT/'display_settings.json',dict(target_peak=peak,target_psd_reference=reference,color_limits_db=clim,shared_waveform_limits=[-wave_lim,wave_lim],
        shared_RMS_limits=[0,rms_lim],sample_rate=50000,display_resampling_rate=10000000,nperseg=245760,noverlap=184320,window='blackmanharris',
        frequency_limits=[0,10000],complete_windows_only=True,individual_normalization=False,phase_alignment=False,
        state_trace_stride=200,state_trace_rate=50000,state_trace_filter='none; visualization only',statistics_rate=10000000,new_information_from_resampling=False,generated_sounds=0))
    return images


def integrity():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['source_sha256']==source_hashes()
    assert manifest['native_library_sha256']==digest(LIBRARY) and manifest['target_sha256']==digest(TARGET) and manifest['probe_sha256']==digest(PROBE)
    assert manifest['status']=='stopped_preflight' and not manifest['preflight_passed'] and manifest['plan_sha256']==digest(OUT/'PLAN.md')
    if not (OUT/'integrity.json').exists():write_json(OUT/'integrity.json',dict(status='running'))
    old=rows('protected_before.csv');changed=[];deleted=[];broken=[];links=0
    for row in old:
        p=ROOT/row['path']
        if not p.exists():deleted.append(row['path'])
        elif digest(p)!=row['sha256']:changed.append(row['path'])
    for name in ('README.md','REPORT.md','REPRODUCE.md','PLAN.md'):
        p=OUT/name
        if not p.exists():continue
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            links+=1
            if not (p.parent/dest.split('#')[0]).exists():broken.append(dict(file=name,link=dest))
    for name,root in (('implementation_checksums.json',ROOT),('artifact_checksums.json',OUT)):
        if (OUT/name).exists():
            values=json.loads((OUT/name).read_text())
            for p,h in values.items():assert digest(root/p)==h,p
            if name=='implementation_checksums.json':
                assert set(values)=={str(p.relative_to(ROOT)) for p in [*ROOT.glob('issue27*.py'),ROOT/'issue27_probe.cpp',ROOT/'tests/test_issue27.py']}
    gain=1/float(abs(load('target')['audio']).max())
    for r in rows('sound_metrics.csv'):
        rate,wave=wavfile.read(OUT/r['wav']);assert rate==50000 and wave.shape==(10000,) and wave.dtype==np.float32
        with np.load(OUT/r['raw_npz']) as z:np.testing.assert_array_equal(wave,(z['audio']*gain).astype(np.float32))
    result=dict(passed=not changed and not deleted and not broken,original_files_checked=len(old),original_files_changed=changed,original_files_deleted=deleted,
        links_checked=links,broken_links=broken,wavs_checked=3,physical_Target_gate_passed=False,actual_training_runs=0,source_sha256=digest(__file__))
    write_json(OUT/'integrity.json',result);assert result['passed'],result;return result


def freeze():
    write_json(OUT/'artifact_checksums.json',{})
    write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in
        [*sorted(ROOT.glob('issue27*.py')),ROOT/'issue27_probe.cpp',ROOT/'tests/test_issue27.py']})
    integrity();write_json(OUT/'storage_after.json',{})
    repository=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    for _ in range(5):
        status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True);new=[s[3:] for s in status.splitlines() if s.startswith('?? ')]
        record=dict(workspace=workspace_inventory(),new_untracked_files=len(new),new_untracked_bytes=sum((repository/p).stat().st_size for p in new),
            modified_tracked_files=sum(s.startswith(' M ') for s in status.splitlines()),deleted_tracked_files=sum('D' in s[:2] for s in status.splitlines()),
            measurement='Logical regular-file bytes, ignored files included; .git/symlinks excluded; no staging/commit')
        write_json(OUT/'storage_after.json',record);before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total'];after=record['workspace']['total']
        block='<!-- ISSUE27_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[['通常ファイル数',before['files'],after['files']],['論理容量',before['bytes'],after['bytes']],
            ['Git新規候補','0（クリーン）',f"{len(new)}個 / {record['new_untracked_bytes']/1e6:.2f}MB"],['既存追跡の変更/削除','0/0',f"{record['modified_tracked_files']}/{record['deleted_tracked_files']}"]])+'\n<!-- ISSUE27_STORAGE_END -->'
        p=OUT/'REPORT.md';text=p.read_text();p.write_text(re.sub(r'<!-- ISSUE27_STORAGE_BEGIN -->.*?<!-- ISSUE27_STORAGE_END -->',lambda _:block,text,flags=re.S) if '<!-- ISSUE27_STORAGE_BEGIN -->' in text else text+'\n'+block+'\n')
        write_json(OUT/'artifact_checksums.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    record['workspace']=workspace_inventory();write_json(OUT/'storage_after.json',record)


def report():
    evidence=audit();images=figures();metrics=rows('sound_metrics.csv');motions=rows('tail_motion.csv');eq=rows('equilibrium_reference.csv');a=rows('caseA_reference.csv')
    mt=table(['音','末尾20ms AC std','AC/B Target peak','AC/自身peak','直線除去後std','B Targetに対するE/J'],
        [[r['sound'],f"{float(r['tail_std']):.6f}",f"{float(r['tail_ac_ratio_B_Target_peak']):.9f}",f"{float(r['tail_ac_ratio_self_peak']):.9f}",f"{float(r['tail_detrended_std']):.6f}",f"{float(r['E_total_to_B_Target']):.8f} / {float(r['J_to_B_Target']):.8f}"] for r in metrics])
    motion_table=table(['音','状態','末尾10ms平均','native全刻みの標本内std','0.20s最終値'],
        [[r['sound'],r['channel'],*[f"{float(r[k]):.9g}" for k in ('mean','std','final_value')]] for r in motions if r['sound']!='target'])
    st=table(['音','左source平均','右source平均','右source std','遅延feedback平均','π̃平均','右output項平均'],
        [[r['sound'],*[f"{float(r[k]):.6f}" for k in ('source_l_mean_last10ms','source_r_mean_last10ms','source_r_std_last10ms','feedback_mean_last10ms','pi_tilde_mean_last10ms','output_r_mean_last10ms')]] for r in metrics if r['sound']!='target'])
    et=table(['分岐','定常πの解析値','定常x_l','定常x_r','局所速度係数k_y（π̃固定）'],
        [[r['condition'],*[f"{float(r[k]):.9g}" for k in ('predicted_pi','predicted_x_l','predicted_x_r','local_velocity_coefficient_fixed_pi')]] for r in eq])
    reference=float(np.mean([float(r['E_total']) for r in a]));reference_sd=float(np.std([float(r['E_total']) for r in a],ddof=1))
    body=f'''# Issue27 — Case B自己Target：事前診断で停止

## 結論と実行状態

指定されたCase B Targetは、0.20秒内で持続発声proxyの基準を満たさなかった。[Issue27](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/27)の「Targetの明らかな非持続発声があれば学習を強行せず事前診断を報告」に従い、**SAC学習は0run・0候補・0更新**で停止した。予定の3seed×4096候補と33評価点/seedは未実施で、Final/Selected/posthoc/training_bestの方策・誤差は存在しない。[seed別の未開始会計](training_accounting.csv)。

モデルのA/B分岐、右係数、右f0、Target生成とoracleの一致は確認できた。停止理由は実装の右係数欠落やoracle不一致ではなく、指定Targetの持続性である。E_total≤.10の合否判定や延長ゲートは使用していない。

| 検査 | 結果 |
|---|---|
| 実libraryと現行nativeソースの対応 | 同じflagsで再ビルドした.soがbyte/hash完全一致 |
| Case Bのα_l/β_l・α_r/β_r | 1.2e6/1500・1.2e6/1500、全て非ゼロ |
| 右ε_r/f0_r | 3.5e7 / 1e7、実メモリで確認 |
| x0 / D / dt | .02cm / .01 / 1e-7、実メモリで確認 |
| 初期4状態・遅延・filter | 全て0 |
| B Targetと独立B oracle | 波形・全20block状態・native運動mean/stdが完全一致、E_spec/E_env/E_total/J=0 |
| 同じ5制御のA対照 | 右source=0、Bとは出力が異なる |
| B Target末尾AC std/peak | **{evidence['B_Target_late_ac_ratio']:.10f} < .05：不合格** |
| 短い学習・Replay更新試験／3本の本学習 | Targetゲート不合格により実施しない |
| 0更新での3seed初期設定・正規化・停止guard | 7テスト合格、新規音0 |

[凍結計画](PLAN.md) / [初回preflight記録](preflight/physics.json) / [完全診断](diagnostic_audit.json) / [manifest](manifest.json) / [テスト](preflight/tests.json)。

## TargetとA対照

Targetは新規のBirdModel('B')で、全期間一定の[ε_l,ε_r,p_s,f0_l,f0_r]=[4.1e7,3.5e7,3.4e6,3e5,1e7]から生成した。0.20s/50kHz/10000sample、native dt1e-7、初期状態/遅延0、16次15kHz lowpass→200分の1の50kHz出力、filter初期状態0。旧Case A Targetは入力に使用していない。[Target生波形・制御](preflight/target.npz) / [filter係数](preflight/filter.npz)。

対照Aにも**同じ5制御（右f0も1e7）**を与え、Aだけ右α/βが0になる既存分岐を使った。この1音は今回の物理対照であり、TargetをA音へ置き換えていない。B oracleは別のインスタンスをゼロ状態から再計算した。読み取りプローブから実ロードlibraryのbird_advance関数ポインタを呼び、元の物理積分を使ったまま全刻みの状態・音響項を観測した。元native/model.cppとlibbirdsong.soは変更していない。[ビルド対応](native/build.json)。

{mt}

持続proxyは末尾20ms（1000sample）の平均を除いたstd/Target peak>.05。Bの値は約0.0003395で、基準の約1/147だった。A対照は自身peakで0.32840、共通B peakでも0.16770で基準を超える。Bの末尾音は非ゼロのゆっくりした変動であり、generated_silent判定では無音になっていない。末尾から直線傾向を除いたstdはB約12.84、A約436654で、BのAC stdの大半もドリフトによるものと整合する。直線除去は診断だけで、Target・報酬・保存音は変更しない。

[B Target WAV](audio/target.wav) / [B oracle WAV](audio/oracle.wav) / [同一制御A対照 WAV](audio/same_controls_caseA.wav)。全音のgainは1/B Target peakの共通値で、個別正規化・クリップ・位相合わせはない。

![共通基準による3音の音響表示](figures/audio_contrast.png)

## 左右の運動と非ゼロの右音響寄与

{motion_table}

stdは末尾10ms・100000 native stepの母標準偏差で、ゆっくりしたドリフトも含む。変位stdをそのまま周期振動の振幅と呼ばない。Bの右側は末尾でもx/yが変化し、完全固定や0へclampしていない。右f0が高い既存の閉鎖側設定であることと、微小な受動応答・音響寄与が残ることを区別する。物理的な開閉を変位の符号だけでは決めない。

![左右状態の過渡と末尾拡大](figures/motion_transients.png)

source項を L=α_l(x_l−τ_l y_l)+β_l y_l、R=α_r(x_r−τ_r y_r)+β_r y_r、F=−γπ(t−T) と定義するとπ̃=L+R+F。最終出力はO_l+O_r+F、O_j=source_j−β_jτ_j dy_j/dt と分解できる。以下は末尾10msのnative全刻みで測った値。

{st}

Bの右source平均は約−268453、stdは約803.66で非ゼロだった。Aでは右の機械状態が動いていてもα_r/β_r=0のため右sourceは0。BがAへ退化していないことを、係数と波形の両方から確認できた。出力の分解再構成誤差の最大値は{evidence['output_decomposition_max_error']:.3g}。この分解は現在の連成状態での代数的寄与であり、右側を取り除いて再計算した因果的な独立音ではない。[全block・全項の統計](native_block_statistics.csv)。

![左右sourceと遅延feedback](figures/acoustic_contributions.png)

## 減衰と整合するモデル内の説明

同じ入力制御のAでは左の持続振動が見られ、Bでは右の負のsource項を残すことで共通π̃の平均が大きく変わった。Bの左右状態は0.20秒でもゆっくり動いており、定常に到達したとは扱わない。

参考として、現行nativeのA係数=1、y=dy/dt=0、一定の遅延πという条件で定常点を解くと、π*=−(α_l f0_l/ε_l+α_r f0_r/ε_r)/(1+γ−α_l/ε_l−α_r/ε_r)、x_j*=(π*−f0_j)/ε_j となる。左右の結合を残すと左の平衡変位も変わる。π̃を固定した局所の速度係数k_y=b+c x_l*²−D(p_s−π*)は次の通り。

{et}

Bの平衡変位では非線形減衰c x²が大きくなり、局所速度係数の符号変化は観測された減衰と整合する。これは現行モデルの定常点に基づく説明であり、遅延・左右連成を含む全系の安定性の証明や、全Case B制御域が非発声であることの証明ではない。今回の0.20秒を延長したり別制御を試す追加計算はしていない。x/y、音響項、p_s/ε/f0はいずれもモデル内部量で、実測Pa・張力・生理的発声判定と同一視しない。f0はHzではない。

## 学習設定とCase A参考値

予定は通常SAC128（Actor/Critic[128,128]）、r=−(.75E_spec+.25E_env)、共通E=E_spec+E_env、3optimizer LR3e-4、auto_0.01/target_entropy−3、batch32/buffer50000/warmup16/gamma1/train_freq1/gradient_steps1。ε_l[2.5e7,5e7]・p_s[2.5e6,4e6]・f0_l[49000,800000]、右ε_r3.5e7/f0_r1e7固定。3seedの初期設定・重みは既存A同seedと一致し、seed間の重みは異なることを0更新で確認した。

独立したCaseBEnvを追加し、旧Constant020EnvのA固定経路を通らずBを構築・resetする。右係数/実効f0を実メモリで検査する。Targetゲート不合格ならstepを積分前に拒否し、学習readyも拒否する。Replay行動・報酬と学習中LRの検証は未実施であり、学習済みモデルやFinalとして未学習重みを保存していない。

旧Issue23のA Final共通Eは3seedで{reference:.5f}±{reference_sd:.5f}（標本SD）、持続proxy3/3だった。[読み取り専用の参考値](caseA_reference.csv)。Aは自身の別Target、Bは今回は学習未開始であり、分岐の優劣・学習難易度の結論には使わない。

## 会計・保存・次の判断

| 区分 | 実行量 |
|---|---:|
| B Target新規生成 | 1音・2,000,000 native step |
| 独立B oracle再計算 | 1音・2,000,000 native step |
| 同じ制御のA対照 | 1音・2,000,000 native step |
| 合計 | **3音・6,000,000 step・積分0.60秒** |
| 追加生成・Target/条件変更・追加探索 | 0 |
| SAC本学習／学習候補／optimizer更新 | 0 / 0 / 0 |
| 独立Actor評価・保存方策の再生 | 0 |
| 7テスト・初期化確認・保存データの再集計/図 | 新規native音0 |

[物理生成ledger](preflight/ledger.json) / [実行会計](execution.json) / [seed別の未開始理由](training_accounting.csv)。初回テストのLR検査呼出しにrate引数の不足があり修正後に7件合格した。両テスト実行とも積分・学習は0。初回生成に使ったソース4件を[実行時ソース](executed_sources/issue27_prepare.py)と[hash](executed_source_checksums.json)で保持し、physics関数とnative呼出し・プローブが現行と同じことを監査した。診断のために元3音を再生成していない。

既存6241ファイルをhash照合し、変更/削除0。元Case A環境・Target・モデル・グラフは不変。root READMEには索引だけ追加した。[元成果保護](integrity.json) / [全成果物hash](artifact_checksums.json) / [実装hash](implementation_checksums.json) / [再現](REPRODUCE.md)。commit/push・Issueコメント/close・外部送信はしていない。

次に必要なのは、**持続発声するCase B Targetの仕様を確定するか、この非持続Targetを模倣する課題として許容するか**の方針確認である。右α/βを0へ変えて解決したことにはしない。新しい仕様が確定してから物理ゲートを再確認し、通常SAC3seedの実装・短いReplay/更新検査・本学習へ進む。旧Issue20のA Target共用仕様は今回実行せず、別途整合を取る。
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'README.md').write_text('''# Issue27 — Case B自己Target：preflight停止

指定Targetは持続発声proxy0.0003395（基準>.05）で、Issueの指示に従いSACは0run。左右α/β非ゼロ・右f0=1e7・oracle一致・右音響寄与は確認済み。

[診断と次の判断](REPORT.md) / [事前計画](PLAN.md) / [再現](REPRODUCE.md) / [manifest](manifest.json)

[Target B WAV](audio/target.wav) / [B oracle WAV](audio/oracle.wav) / [同一制御A対照 WAV](audio/same_controls_caseA.wav) / [音響指標](sound_metrics.csv) / [左右末尾状態](tail_motion.csv) / [source項](native_block_statistics.csv)

[新規B生波形・制御](preflight/target.npz) / [native詳細B](preflight/oracle.npz) / [A対照詳細](preflight/same_controls_caseA.npz) / [検証](diagnostic_audit.json) / [未開始会計](training_accounting.csv) / [保存検査](integrity.json)
''')
    (OUT/'REPRODUCE.md').write_text('''# Issue27の検証・再現

既存3音を保存したまま、以下は新たな音や学習を生成せず検証できる。リポジトリrootから実行する。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_report --integrity
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27
# 保存済み3音・native統計からCSV/WAV/図/報告を更新。元Targetは再生成しない
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_report
```

再生成には別checkoutでissue27_data.OUTを未使用pathへ設定し、issue26_inventory.inventoryの保護集合と同じ全tracked/既存results/reportsをhash化した開始snapshotを/tmp/birdsong-issue27-before.jsonへ用意する（root README/.gitignoreは索引編集用に除く）。

```bash
# 事前3音。ゲート不合格は終了code2となり、本学習へ進めない
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_prepare physics
# 不合格の診断を確定。現行仕様ではSAC学習コマンドを実装・実行していない
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_prepare seal
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27_report
```

native/build.jsonにcompiler/flags/ソースと全libraryのhashを保存。元build/libbirdsong.soは変更せず、現行ソースの別.soを生成してbyte一致を確認する。診断probeは既存libraryの実Modelを読み、渡された元bird_advance関数ポインタで積分する。native/*.soは既存Git ignoreの対象なので別環境では同じcompiler/flagsで再ビルドし、保存hashと一致するものを使用する。既存ビルドを上書きして検証しない。

表示STFTは10MHzへの表示用再サンプリング、Blackman-Harris245760/overlap184320、完全窓、0–10kHz。新情報は増えない。波形/WAV/5ms RMSは全てB Target peakの共通gain、PSDもB Target全帯域最大を基準とする。状態traceはnativeを200刻み抽出した可視化用で、統計は10MHz全刻み。変位/速度のstdはドリフトを含む。直線除去は診断統計にのみ使い、音や報酬を変更しない。
''')
    write_json(OUT/'report_execution.json',dict(generated_sounds=0,actual_training_runs=0,figures=len(images),source_sha256=digest(__file__)))
    print('Failed-Target diagnostic report, three figures and WAVs complete; freezing preservation/storage',flush=True);freeze()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--integrity',action='store_true');args=p.parse_args();integrity() if args.integrity else report()
