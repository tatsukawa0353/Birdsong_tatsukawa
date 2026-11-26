"""Report revised Issue27 Stage A; all9 candidates failed to voice, learning0."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from .issue27r_data import ROOT,OUT,OLD,TARGET,GRID_E,GRID_P,hashes
from .issue27r_native import PROBE
from .issue27r_diagnostics import audit,records,load,rows
from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue19_protection import workspace_inventory
from .issue23_stage_a import table as plain_table
from .issue15_analysis import write_json
from .issue16 import digest
from .model import LIBRARY

def table(headers,data):
    esc=lambda v:str(v).replace('|','\\|')
    return plain_table([esc(v) for v in headers],[[esc(v) for v in r] for r in data])

def figures():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
    (OUT/'figures').mkdir(exist_ok=True);items=[];data={r['candidate']:load(r['candidate']) for r in records()}
    peak=max(float(abs(v['audio']).max()) for v in data.values());f,t,_,reference=display_psd(data['c00']['audio'])
    powers={name:display_psd(v['audio'])[2] for name,v in data.items()};maxdb=max(float(relative_db(p,reference).max()) for p in powers.values())
    clim=(-80.,max(0.,float(np.ceil(maxdb/5)*5)));rmsmax=max(float(rms_envelope(v['audio'],peak).max()) for v in data.values())*1.05
    def save(fig,name,text):
        p=OUT/f'figures/{name}.png';fig.savefig(p,dpi=130);plt.close(fig);items.append(dict(path=str(p.relative_to(OUT)),sha256=digest(p),description=text))
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharex=True,sharey=True);mesh=None
    for r in records():
        name=r['candidate'];ax=axes[r['row'],r['column']]
        mesh=ax.pcolormesh(t,f/1000,relative_db(powers[name],reference),shading='auto',cmap='magma',vmin=clim[0],vmax=clim[1])
        ax.set(title=f"{name}:epsilon={r['epsilon_l']/1e7:g}e7, ps={r['ps']/1e6:g}e6\nlate AC/own peak={r['late_ac_ratio']:.6f}",xlabel='Time [s]',ylabel='Frequency [kHz]',xlim=(t[0],t[-1]),ylim=(0,10))
    fig.colorbar(mesh,ax=axes.ravel().tolist(),label='dB re c00 full-band PSD reference (display only, no selected Target)')
    save(fig,'candidate_spectrograms','All9 physical candidates; c00 is first fixed grid point, not selected Target; same PSD reference/color limits across grid')
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,sharey=True)
    for r in records():
        a=data[r['candidate']]['audio'];time=np.arange(len(a))/50000;ax=axes[r['row'],r['column']]
        ax.plot(time,a/peak,color='#222222',lw=.7,label='Raw waveform');ax.plot(time,rms_envelope(a,peak),color='#D55E00',lw=1,label='5ms RMS')
        ax.axvspan(.18,.2,color='#0072B2',alpha=.12);ax.set(title=f"{r['candidate']}: own-peak proxy {r['late_ac_ratio']:.6f}, non-sustained",xlabel='Time [s]',ylabel='Amplitude / common9-sound peak',xlim=(0,.2),ylim=(-1.05,1.05));ax.legend(fontsize=7)
    save(fig,'candidate_waveforms_rms','Raw full0.20s waves and5ms RMS, common gain1/max9peak; shaded last20ms; no individual audio normalization')
    colors=('#0072B2','#009E73','#D55E00')
    for title,lo,hi in (('motion_full',0.,.2),('motion_tail',.18,.2)):
        fig,axes=plt.subplots(4,3,figsize=(15,13),sharex=True,sharey='row')
        for j,channel in enumerate(('x_l','y_l','x_r','y_r')):
            for pindex,p in enumerate(GRID_P):
                ax=axes[j,pindex]
                for eindex,e in enumerate(GRID_E):
                    points=data[f'c{eindex}{pindex}']['diagnostic_samples'];time=np.arange(1,len(points)+1)/50000;mask=(time>=lo)&(time<=hi)
                    ax.plot(time[mask],points[mask,j],color=colors[eindex],lw=.8,label=f'epsilon={e/1e7:g}e7')
                ax.set(title=f'{channel}, ps={p/1e6:g}e6'+(' (tail zoom)' if lo else ''),xlabel='Time [s]',ylabel='Model internal '+channel,xlim=(lo,hi));ax.legend(fontsize=7)
        save(fig,title,'All9 state traces; full transient or20ms zoom, no state clamping;50kHz sampled traces, native10MHz statistics saved separately')
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharex=True,sharey=True)
    for r in records():
        p=data[r['candidate']]['diagnostic_samples'];time=np.arange(1,len(p)+1)/50000;mask=time>=.18;ax=axes[r['row'],r['column']]
        for index,label,color,style in ((5,'R source','#D55E00','-'),(9,'O_R output','#777777','--'),(6,'Delayed feedback','#009E73','-'),(7,'Common pi_tilde','#0072B2','-')):
            ax.plot(time[mask],p[mask,index],color=color,ls=style,lw=1,label=label)
        ax.set(title=r['candidate']+': right contribution remains nonzero',xlabel='Time [s]',ylabel='Native model terms',xlim=(.18,.2));ax.legend(fontsize=7)
    save(fig,'right_contributions_tail','All9 retain nonzero right source/output contribution and feedback; additive coupled-state decomposition, not isolated source sound')
    write_json(OUT/'figures.json',items);write_json(OUT/'display_settings.json',dict(selected_Target=False,display_reference_candidate='c00',reference_reason='First predefined grid point; display only',
        common_audio_peak=peak,gain=1/peak,target_individual_normalization=False,PSD_reference=reference,color_limits_db=clim,frequency_limits=[0,10000],
        native_rate=10000000,audio_rate=50000,display_resampling_rate=10000000,nperseg=245760,noverlap=184320,window='blackmanharris',complete_windows_only=True,
        state_trace_stride=200,state_trace_filter='none, visualization only',statistics_rate=10000000,phase_alignment=False,generated_sounds=0))
    return items

def integrity():
    m=json.loads((OUT/'manifest.json').read_text());assert source_hashes_match(m['source_sha256'], hashes()) and not m['preflight_passed']
    assert m['native_library_sha256']==digest(LIBRARY) and m['probe_sha256']==digest(PROBE) and m['plan_sha256']==digest(OUT/'PLAN.md')
    assert not TARGET.exists() and not (OUT/'runs').exists() and m['actual_training_runs']==0
    for p,h in m['candidate_sha256'].items():assert digest(OUT/'preflight/candidates'/p)==h
    if not (OUT/'integrity.json').exists():write_json(OUT/'integrity.json',dict(status='running'))
    protected=rows('protected_before.csv');changed=[];deleted=[];broken=[];nlinks=0
    for row in protected:
        path=resolve_result_path(ROOT/row['path'])
        if not path.exists():deleted.append(row['path'])
        elif digest(path)!=row['sha256']:changed.append(row['path'])
    for name in ('README.md','REPORT.md','PLAN.md','REPRODUCE.md'):
        p=OUT/name
        if not p.exists():continue
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'):continue
            nlinks+=1
            if not (p.parent/dest.split('#')[0]).exists():broken.append(dict(file=name,link=dest))
    for name,root in (('implementation_checksums.json',ROOT),('artifact_checksums.json',OUT)):
        if (OUT/name).exists():
            checks=json.loads((OUT/name).read_text())
            for p,h in checks.items():assert digest(root/p)==h,p
            if name=='implementation_checksums.json':assert set(checks)=={str(p.relative_to(ROOT)) for p in [*ROOT.glob('issue27r*.py'),ROOT/'tests/test_issue27r.py']}
    for r in rows('audio_index.csv'):
        rate,wave=wavfile.read(OUT/r['wav']);assert rate==50000 and wave.shape==(10000,) and wave.dtype==np.float32
        with np.load(OUT/r['raw_npz']) as z:np.testing.assert_array_equal(wave,(z['audio']*float(r['gain'])).astype(np.float32))
    d=json.loads((OUT/'diagnostic_audit.json').read_text());assert d['verification_passed'] and not d['Target_selection_gate_passed'] and d['source_sha256']==digest(ROOT/'issue27r_diagnostics.py')
    result=dict(passed=not changed and not deleted and not broken,original_files_checked=len(protected),original_files_changed=changed,original_files_deleted=deleted,
        local_links_checked=nlinks,broken_links=broken,wavs_checked=9,stageA_candidates=9,eligible=0,training_runs=0,source_sha256=digest(__file__))
    write_json(OUT/'integrity.json',result);assert result['passed'],result;return result

def freeze():
    write_json(OUT/'artifact_checksums.json',{});write_json(OUT/'implementation_checksums.json',{str(p.relative_to(ROOT)):digest(p) for p in [*sorted(ROOT.glob('issue27r*.py')),ROOT/'tests/test_issue27r.py']})
    integrity();write_json(OUT/'storage_after.json',{})
    repository=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    for _ in range(5):
        status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True);new=[s[3:] for s in status.splitlines() if s.startswith('?? ')]
        r=dict(workspace=workspace_inventory(),new_untracked_files=len(new),new_untracked_bytes=sum((repository/p).stat().st_size for p in new),
            modified_tracked_files=sum(s.startswith(' M ') for s in status.splitlines()),deleted_tracked_files=sum('D' in s[:2] for s in status.splitlines()),
            measure='Logical regular-file bytes; ignored files included, .git/symlinks excluded; no stage/commit')
        write_json(OUT/'storage_after.json',r);before=json.loads((OUT/'storage_before.json').read_text())['workspace']['total'];after=r['workspace']['total']
        block='<!-- ISSUE27R_STORAGE_BEGIN -->\n'+table(['対象','着手前','完了時'],[['通常ファイル数',before['files'],after['files']],['論理容量bytes',before['bytes'],after['bytes']],
            ['Git新規候補','0（クリーン）',f"{len(new)}個 / {r['new_untracked_bytes']/1e6:.2f}MB"],['既存追跡変更/削除','0/0',f"{r['modified_tracked_files']}/{r['deleted_tracked_files']}"]])+'\n<!-- ISSUE27R_STORAGE_END -->'
        p=OUT/'REPORT.md';body=p.read_text();p.write_text(re.sub(r'<!-- ISSUE27R_STORAGE_BEGIN -->.*?<!-- ISSUE27R_STORAGE_END -->',lambda _:block,body,flags=re.S) if '<!-- ISSUE27R_STORAGE_BEGIN -->' in body else body+'\n'+block+'\n')
        write_json(OUT/'artifact_checksums.json',{str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('artifact_checksums.json','storage_after.json')})
    r['workspace']=workspace_inventory();write_json(OUT/'storage_after.json',r)

def report():
    check=audit();images=figures();summary=rows('candidate_summary.csv');motions=rows('tail_motion.csv');eq=rows('equilibrium_reference.csv')
    grid=table(['候補','ε_l','p_s','末尾AC std/自身peak','直線除去後std/peak','持続proxy','silent','数値失敗','選定可'],
        [[r['candidate'],r['epsilon_l'],r['ps'],f"{float(r['late_ac_ratio']):.10f}",f"{float(r['detrended_ac_ratio']):.10f}",r['sustained_proxy'],r['generated_silent'],r['failed'],r['eligible']] for r in summary])
    side=table(['候補','右source平均','右source std','右output平均','右output std','feedback平均','π̃平均'],
        [[r['candidate'],*[f"{float(r[k]):.6f}" for k in ('source_r_tail_mean','source_r_tail_std','output_r_tail_mean','output_r_tail_std','feedback_tail_mean','pi_tilde_tail_mean')]] for r in summary])
    mt=table(['候補','状態','native末尾10ms平均','native末尾10ms std','最終値','直線除去後20ms std（50kHz trace）'],
        [[r['candidate'],r['channel'],*[f"{float(r[k]):.8g}" for k in ('mean','std_native','end_value','detrended_std_sampled_last20ms')]] for r in motions])
    et=table(['候補','定常π参考値','定常x_l参考値','局所速度係数k_y（π̃固定）'],
        [[r['candidate'],*[f"{float(r[k]):.9g}" for k in ('steady_pi_reference','steady_x_l_reference','local_velocity_coefficient_fixed_pi')]] for r in eq])
    body=f'''# Issue27改訂 — Case Bの9条件を診断、全条件非持続のためStage B停止

## 結果と現在の状態

**指定のε_l×p_sの9条件をすべて計算したが、持続発声候補は0/9だった。** 末尾AC std/各音peakは{check['ratio_min']:.10f}〜{check['ratio_max']:.10f}で、全条件が事前の基準>.05を満たさない。[改訂Issue27](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/27)の「9条件すべて非持続ならStage Bを開始しない」に従い、Targetを選ばず停止した。

SAC/TD3/PPOの手法別smoke test・本番9runは未開始。学習候補・optimizer更新・独立Actor評価はいずれも0で、Final/Selected/posthoc/training_bestや手法間の精度差は存在しない。学習器の失敗として9runを計上したり、非持続Targetを成功扱いして学習を強行していない。E_total≤.10は合否・追加計算のゲートに使っていない。

物理式・固定係数・右音響寄与は指定どおりだった。左右α=1.2e6/β1500、右ε_r3.5e7/f0_r1e7、x0=.02cm/D=.01、dt1e-7を実メモリで確認。数値失敗0、generated_silent0で、今回の結果は「無音」判定ではなく、持続振動が非常に小さくドリフト主体という診断である。

[9点の全指標](candidate_summary.csv) / [事前計画](PLAN.md) / [初回scan結果](preflight/scan.json) / [選定ゲート](preflight/gate.json) / [manifest](manifest.json)。

## Stage Aの固定条件と9点

各音はCase B、0.20秒一定、50kHz/10000sample、native 2,000,000 step。初期4状態・遅延・16次15kHz lowpass filter状態は0、200分の1に出力する。[filter係数](preflight/filter.npz)。全9点でf0_l=300000・f0_r=10000000、ε_r=35000000固定。f0は音響基本周波数Hzではない。p_s=3.0e6を優先し、その中で物理的に持続した候補を選ぶ計画だった。

{grid}

持続proxyは末尾1000sample（20ms）のAC std/その音自身の全期間peak>.05。生理学的発声判定やHNRではない。事前に、直線傾向を除いたratio>.05もドリフト除外の確認として定めたが、今回は全点が一次proxy自体を満たさず、この追加条件によって候補を落としたわけではない。p_s=3.0e6の3点も約.000408〜.000409で基準未達だった。

c22（ε_l4.1e7、p_s3.8e6）には約0.04秒まで大きい過渡振動が見える。10ms窓のAC std/自身peakは0.03–0.04秒で約.15049、0.04–0.05秒で.00949へ下がり、終盤20msは.00029114だった。これは「全期間まったく振動しない」という結果ではなく、過渡的に鳴っても0.20秒の持続Targetには至らないという区別である。音を短縮して合格扱いにする変更はしていない。

![全9条件の共通基準スペクトログラム](figures/candidate_spectrograms.png)

![全9音の生波形とRMS](figures/candidate_waveforms_rms.png)

**Targetは未選定。** 図のPSD基準c00は事前の9点表の先頭を表示参照にしただけでTargetではない。全音を同じPSD基準/色軸で表示し、WAV・生波形・RMSは9音中の最大peakによる共通gainを使った。個別正規化、時間合わせ、音の直線除去やclipはしていない。[表示設定](display_settings.json) / [WAV索引](audio_index.csv)。

## 左右の過渡・末尾運動とドリフト

![左右4状態の全0.20秒](figures/motion_full.png)

![左右4状態の末尾20ms](figures/motion_tail.png)

以下のnative stdは最終10msの100000積分刻みをすべて使った区間内標準偏差（ddof=0）。ドリフトも含む。直線除去後の別指標は50kHzで記録した末尾20ms traceに対するもので、窓とサンプリングが異なる。stdをそのまま周期振動の振幅と呼ばない。右の微小な応答・ドリフトを0へclampせず、そのまま保存した。変位の符号だけで物理的開閉を断定しない。

{mt}

[末尾状態CSV](tail_motion.csv) / [全20block・全12項のnative統計](native_block_statistics.csv)。各NPZには50kHzの内部state/source/output traceとnative全刻みのmean/std/min/maxも保存している。

## 右音響source/outputは全9条件で非ゼロ

R=α_r(x_r−τ_r y_r)+β_r y_r は共通π̃への右source項。O_r=R−β_rτ_r dy_r/dt は右output項で、π̃=L+R−γπ(t−T)、最終出力π=O_l+O_r−γπ(t−T)となる。末尾10msのnative全刻みの計測は次の通り。

{side}

![右source/outputと声道feedback](figures/right_contributions_tail.png)

右source/outputとも全点で非ゼロだった。Bの右係数を削除したA経路や、右側の完全固定へ退化させた結果ではない。分解の最大再構成誤差は{check['max_output_decomposition_error']:.3g}。これは連成状態における代数的な寄与で、片側を取り除いて独立に鳴らした音の寄与率ではない。全てモデル内部量で、圧力や変位を実測Pa・張力などへ対応づけていない。

## 物理実装・前回結果との照合

実ロードlibbirdsong.soと、変更していないnative/model.cppを同じcompiler/flagsで別にビルドした.soのbyte/hashが完全一致した。読み取りprobeも前回と同じ保護対象のC++ソースから新成果フォルダへビルドした。積分は実libraryのbird_advance関数ポインタを使い、元運動方程式・圧力結合・遅延・音響式・固定係数を変更していない。[ビルド監査](native/build.json)。

c21（ε_l4.1e7、p_s3.4e6）は前回停止条件と同じ。今回の波形・物理制御・20block状態・native運動mean/std・全内部trace/統計は、**保存済みの前回B oracleと完全一致**した。これは既存ファイルの読み取り照合であり、比較のための追加生成はしていない。前回の3音・停止レポート・未実行記録は変更せず保持した。[診断監査](diagnostic_audit.json) / [前回の停止記録](../initial_stop/REPORT.md)。

選定候補がないため、「選定Targetの独立oracle」は未実施である。c21の過去データ一致を、今回の新Target選定やSAC等の学習成功と混同しない。

## 原因と次に検討する方向（追加計算なし）

9点とも、初期過渡の後は左右状態のゆっくりした移動が主体で、末尾の周期的成分が非常に小さい。過去のACIマップと今回のf0条件は異なるため、低いεなら必ず発声するという仮定は置けない。

参考に、現行式のA係数=1、y=dy/dt=0、遅延πが一定の定常点を解くと、π*=−(α_l f0_l/ε_l+α_r f0_r/ε_r)/(1+γ−α_l/ε_l−α_r/ε_r)、x_j*=(π*−f0_j)/ε_j。左の局所速度係数（π̃を固定した項）k_y=b+c x_l*²−D(p_s−π*)は次のようになる。

{et}

この参考式ではp_sは定常変位から消え、今回の全点でk_yは正となる。右の非ゼロ寄与で変化する左の平衡変位と非線形減衰c x²が、観測された減衰と整合する。これは0.20秒で既に定常に達したことや、遅延・連成全系の安定性、Case B全制御域での非発声を証明するものではない。

**次の案は、左f0_lをより小さくする方向を含む固定候補を別途定めて物理診断すること。** 定常式では左変位の絶対値と非線形減衰を減らす方向になるという仮説で、発声を保証しない。今回の許可は9点までなので、新しいf0、長時間化、追加grid、直接最適化、Target差し替えは実行していない。右α/βを0にしたり固定物理係数を変えて解決したことにはしない。

## Stage Bの予定と未実施項目

予定は同じ選定B TargetでSAC/TD3/PPO×3seed、各4096環境step＝36864学習音、独立Actor評価各33点（計297点）。全隠れ層128×128、LR3e-4、gamma1、batch32。SACはbuffer50000/warmup16/train_freq1/gradient_steps1/auto_0.01/target_entropy−3、TD3は同buffer/warmup等とNormalActionNoise(0,.1)、PPOはpi/vfをそれぞれ128×128・n_steps128/n_epochs10/clip.2を予定した。

PPOの予定は128step rollout×32回、40minibatch×32=1280 optimizer stepで、内部epoch counter320。SACのActor/Critic/温度各4080、TD3のCritic4080/Actor2040とは更新回数が異なる。**これらは予定値で、実更新は全て0。** 実ネットの層/総パラメータassert、Replay/rollout、checkpoint再生、手法別smokeと本学習は、Stage A合格後の工程なので今回は未着手。未学習モデルをFinalとして保存していない。[9runの未開始会計](training_accounting.csv) / [計画と停止status](manifest.json)。

Case Aは別の自己生成Targetに対する参考であり、今回の未実施結果と誤差を比較して構造優位や学習難易度を判断しない。最終的な2音源構造の効果は、将来同じ実録音Targetで検証する段階に分ける。TQC・DE/GA/CMA-ES・時変・5D・左右切替・実録音は今回0件。

## 会計・保存・再現

| 区分 | 実行量 |
|---|---:|
| Stage AのB物理候補 | 9条件・9音 |
| native積分 | **18,000,000 step、1.80秒** |
| 数値失敗 / generated_silent / 持続候補 | 0 / 0 / 0 |
| Target選定 / 選定後oracle / 新規A対照 | 未実施 / 0 / 0 |
| 手法別smoke / 学習run / 学習候補 / 更新 | 0 / 0 / 0 / 0 |
| 新規音を増やさない保存データ検査 | 7テスト合格 |
| 追加探索・追加物理生成 | 0 |

[生成ledger](preflight/ledger.json) / [会計](execution.json) / [テスト](preflight/tests.json)。初回scan実行時のソースは[hash](executed_source_checksums.json)と[原本](executed_sources/issue27r_prepare.py)を保存し、scan関数とnative wrapperの不変を確認。後から加えたのは停止guard・診断・報告で、9音を再生成していない。

旧Issue17〜26と初回Issue27を含む6294ファイルはhash一致・変更/削除0。新成果は本フォルダへ独立保存し、root READMEには改訂の索引だけ追加。[保護検査](integrity.json) / [実装hash](implementation_checksums.json) / [成果物hash](artifact_checksums.json) / [再現手順](REPRODUCE.md)。commit/push、Issueコメント/close、外部送信はしていない。
'''
    (OUT/'REPORT.md').write_text(body)
    (OUT/'README.md').write_text('''# Issue27改訂 — Stage A全9点が非持続

指定9条件を計算し、持続proxy>.05を満たした候補は0。改訂Issueの停止条件に従い、Target未選定、SAC/TD3/PPOのsmoke・本学習は0。

[結果と次の検討案](REPORT.md) / [事前計画](PLAN.md) / [9点CSV](candidate_summary.csv) / [左右末尾運動](tail_motion.csv) / [右音響寄与](native_block_statistics.csv) / [検証](diagnostic_audit.json) / [再現](REPRODUCE.md)

[c00 WAV](audio/c00.wav) / [c01 WAV](audio/c01.wav) / [c02 WAV](audio/c02.wav) / [c10 WAV](audio/c10.wav) / [c11 WAV](audio/c11.wav) / [c12 WAV](audio/c12.wav) / [c20 WAV](audio/c20.wav) / [c21 WAV](audio/c21.wav) / [c22 WAV](audio/c22.wav)

[WAV・生波形索引](audio_index.csv) / [未開始9run](training_accounting.csv) / [manifest](manifest.json) / [旧成果保護](integrity.json) / [成果物hash](artifact_checksums.json)
''')
    (OUT/'REPRODUCE.md').write_text('''# Issue27改訂Stage Aの再現

リポジトリrootから、まず保存済み結果を検査する。下記の検査/再集計は新しい鳴管音や学習を生成しない。実環境は.venv-rl、各BLAS/CPU thread1。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-issue27r-mpl
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_report --integrity
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27r
# 保存9音だけから表/図/WAV/報告を更新
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_report
```

物理生成を再現する場合は別checkoutでissue27r_data.OUTを未使用pathに変更し、着手前の全tracked（root README/.gitignore除く）＋results/reportsをpath/bytes/sha256でsnapshot化する。JSON形式はstorage_before.jsonにprotected_before.csvの全行をfilesキーとして加えたものと同じ。元成果のhash不一致・既存OUTではscanを拒否する。

```bash
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_prepare scan --before /tmp/birdsong-issue27r-before.json
# 全点非持続なら終了code2。Targetも学習も作らない
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_prepare select
Deeplearning/.venv-rl/bin/python -m Deeplearning.tests.test_issue27r
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_prepare seal
Deeplearning/.venv-rl/bin/python -m Deeplearning.issue27r_report
```

native/build.jsonにcompiler/flagsと全hashを記録。元native/model.cpp/libbirdsong.soは変更せず、同じソースを別.soへビルドしてbyte一致を検証する。計測は実libraryのbird_advanceへの関数ポインタで、probeは読み取りと統計のみ。native/*.soは既存Git ignore対象のため、別環境では同じcompiler/flagsで再ビルドし保存hashと照合する。

初回scanの実行ソースはexecuted_sourcesに保持。後の停止guard・報告追加と区別し、scan関数の構文木、native wrapper、現行モデルのhashを検査する。今回の9点として新規生成したc21は、前回の保存B oracleと読み取り比較する。旧ファイルの再作成・書換えや、照合目的の追加生成はしない。

全9点のPSD表示はc00の全帯域PSD最大を共通基準とする。Target未選定でありc00は表示用参照。表示用10MHz再サンプリング、Blackman-Harris245760/overlap184320、完全窓、0–10kHz。WAV/波形/RMSは全9音最大peakによる共通gain。state traceはnativeを200刻み抽出した可視化用で、統計はnative10MHz全刻み。直線除去はドリフト診断のみに使う。
''')
    write_json(OUT/'report_execution.json',dict(generated_sounds=0,training_runs=0,figures=len(images),source_sha256=digest(__file__)))
    print('Revised Stage A report,5 figures,9 WAVs complete; freezing preservation/storage',flush=True);freeze()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--integrity',action='store_true');a=p.parse_args();integrity() if a.integrity else report()
