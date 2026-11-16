"""Issue17 display-only revision; reuse all Final/Selected waveforms and numbers."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np

from .issue17 import DEST,ROOT,weight_label,load_ready
from .issue17_extension_data import load_runs,write_csv
from .issue15_analysis import write_json
from .issue15_report import table
from .issue16 import digest
from .issue17_cleanup import safe_unlink
from .acoustics import distance
from .acoustic_views import display_psd,relative_db,rms_envelope,spectrogram_pair,envelope_pair,control_comparison

WORK=ROOT/'.issue17-work/audio_parameter_revision'
EVIDENCE=DEST/'visualization_revision'
MUTABLE={'README.md','REPORT.md','figure_sources.json','visual_review.json','checksums.json','storage_summary.json','issue_update_proposal.md','issue_update_status.json'}


def render():
    if WORK.exists() or EVIDENCE.exists():raise FileExistsError('Previous visualization output preserved')
    WORK.mkdir(parents=True);base,target,controls=load_ready(DEST);runs=load_runs(DEST);start=time.perf_counter()
    old=[dict(path=str(p.relative_to(DEST)),sha256=digest(p),bytes=p.stat().st_size)
         for sub in ('spectrograms','physical_controls') for p in (DEST/'figures'/sub).glob('*.png')]
    assert len(old)==72
    excluded={r['path'] for r in old}|MUTABLE
    protected={str(p.relative_to(DEST)):digest(p) for p in DEST.rglob('*') if p.is_file() and str(p.relative_to(DEST)) not in excluded}
    write_json(WORK/'protected_before.json',protected);write_json(WORK/'retired_images.json',old)
    f,t,p,reference=display_psd(target);target_db=relative_db(p,reference);target_env=rms_envelope(target,abs(target).max())
    records=[];catalog=[]
    for (c,m,seed,w),data in runs.items():
        for role,key in (('final','final'),('selected_best','objective_selected')):
            record=data[key];source=None
            for rel in (('',) if role=='final' else ('best_objective','best','best_common_posthoc','initial','')):
                candidate=data['folder']/rel/'rollout.npz'
                if not candidate.exists():continue
                with np.load(candidate) as z:
                    if np.array_equal(z['physical_controls'][0],record['physical_controls']):source=candidate;audio=z['audio'].copy();break
            if source is None:raise RuntimeError('Selected audio missing; explicit verified regeneration required')
            comp=distance(target,audio,50000)
            assert abs(comp['total']-record['acoustic_error'])<1e-12
            assert abs(comp['log_spectral']-record['acoustic_components']['log_spectral'])<1e-12
            _,_,power,_=display_psd(audio);db=relative_db(power,reference);env=rms_envelope(audio,abs(target).max())
            assert abs(np.mean(abs(env-target_env))-comp['envelope'])<1e-12
            stem=f"{c.replace('constant_','')}_{m}_{weight_label(w)}_seed{seed}"
            row=dict(condition=c,method=m,seed=seed,envelope_weight=w,role=role,trial=record['episodes'],
                source_rollout=str(source.relative_to(DEST)),source_sha256=digest(source),
                controls=record['physical_controls'],E_total=record['acoustic_error'],E_spec=comp['log_spectral'],E_env=comp['envelope'],
                definition='final actor / actual4096th candidate' if role=='final' else 'first minJ among33 RL policy evaluations /4096 direct candidates',
                stem=stem)
            catalog.append(row);records.append(dict(row,db=db,envelope=env))
    maxima=max(float(r['db'].max()) for r in records);upper=max(10.,float(10*np.ceil(maxima/10)))
    limits=(-70.,upper);rms_limits=(0.,1.3*max(float(r['envelope'].max()) for r in records))
    all_db=np.concatenate([r['db'].ravel() for r in records])
    config=dict(cmap='magma',db_limits=limits,target_reference_psd=reference,
        target_reference='maximum full-band display PSD of this single unchanged Target',
        displayed_db_quantiles=np.quantile(all_db,[0,.01,.5,.99,1]).tolist(),
        below_display_floor_fraction=float(np.mean(all_db<limits[0])),above_display_ceiling_count=int(np.sum(all_db>limits[1])),
        rate=50000,display_resample_rate=10000000,nperseg=245760,noverlap=184320,window='blackmanharris',
        detrend='constant',scaling='density',max_hz=10000,complete_windows_only=True,
        rms_window_ms=5,rms_ylim=rms_limits,rms_reference='same raw Target peak for both sounds',
        calibration='Relative numerical power; not calibrated dB SPL; display resampling adds no high frequency information',
        frozen_before_render_utc=datetime.now(timezone.utc).isoformat())
    write_json(WORK/'display_config.json',config);metadata=[]
    for i,r in enumerate(records):
        source=dict(condition=r['condition'],method=r['method'],seed=r['seed'],envelope_weight=r['envelope_weight'],
                    role=r['role'],trial=r['trial'],source_rollout=r['source_rollout'],source_sha256=r['source_sha256'])
        for kind,folder,renderer in (
            ('spectrogram','spectrograms',lambda p:spectrogram_pair((f,t,target_db),(f,t,r['db']),p,limits=limits)),
            ('rms_envelope','amplitude_envelopes',lambda p:envelope_pair(target_env,r['envelope'],p,limits=rms_limits))):
            path=WORK/'figures'/folder/r['role']/(r['stem']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            meta=renderer(path);meta.update(source,kind=kind,path=str(path.relative_to(WORK)));metadata.append(meta)
        if (i+1)%24==0:print(f'Audio views:{i+1}/144 profiles, two images each',flush=True)
    for c in ('constant_1d','constant_2d','constant_3d'):
        indices={'constant_1d':(0,),'constant_2d':(0,2),'constant_3d':(0,2,3)}[c]
        for role in ('final','selected_best'):
            points=[dict(method=r['method'],seed=r['seed'],column=r['envelope_weight'],controls=r['controls']) for r in records if r['condition']==c and r['role']==role]
            path=WORK/'figures/physical_controls'/f"{c.replace('constant_','')}_{role}.png";path.parent.mkdir(exist_ok=True)
            meta=control_comparison(points,controls,path,indices=indices)
            meta.update(path=str(path.relative_to(WORK)),kind='constant_control_comparison',condition=c,role=role);metadata.append(meta)
    write_json(WORK/'catalog.json',catalog);write_json(WORK/'new_figure_sources.json',metadata)
    write_json(WORK/'audit.json',dict(passed=True,saved_waveforms_reused=144,regenerated_sounds=0,
        spectrograms=144,rms_figures=144,parameter_comparisons=6,common_db_limits=limits,
        common_rms_ylim=rms_limits,saved_metrics_verified=144,render_wall_s=time.perf_counter()-start,
        source_csv_unchanged=digest(DEST/'comparison_per_seed.csv')==protected['comparison_per_seed.csv']))
    rows=[]
    for c,m,s,w in sorted(runs):
        stem=f"{c.replace('constant_','')}_{m}_{weight_label(w)}_seed{s}"
        rows.append([c.replace('constant_',''),m,w,s,*[f'[図](../figures/{folder}/{role}/{stem}.png)' for folder,role in
            (('spectrograms','final'),('spectrograms','selected_best'),('amplitude_envelopes','final'),('amplitude_envelopes','selected_best'))]])
    (WORK/'FIGURES.md').write_text('# 全seedの音響図\n\n'+table(['次元','手法','λ','seed','Final PSD','Selected PSD','Final RMS','Selected RMS'],rows)+'\n')
    index=(DEST/'README.md').read_text().split('## 音響図・物理量図')[0]
    controls_table=table(['次元','Final','Selected Best'],[[d,*[f'[集約図](figures/physical_controls/{d}_{role}.png)' for role in ('final','selected_best')]] for d in ('1d','2d','3d')])
    index+='## 一定制御パラメータの集約比較\n\n'+controls_table+'\n\n各図はλの2列、可変係数の行。seed0/1/2と平均±標本SD、目標の破線を表示。f0_lはHzではない。\n\n'
    index+='## Final / Selected BestのスペクトログラムとRMS包絡\n\n[全次元・手法・λ・seedの図一覧](visualization_revision/FIGURES.md)。Final PSD、Selected PSD、Final RMS、Selected RMSを別リンクで案内する。\n\n'
    index+='全図でTarget最大PSDを0 dBとする共通基準・共通範囲・magmaを使用。RMSは双方を同じTargetピークで正規化。校正済みdB SPLではない。\n\n[描画条件](visualization_revision/display_config.json) / [出所一覧](visualization_revision/catalog.json) / [監査](visualization_revision/audit.json) / [削除一覧](visualization_revision/deleted_images.txt)。\n'
    (WORK/'README.md').write_text(index)
    report=(DEST/'REPORT.md').read_text()+f'''\n\n## Issue17追加：共通振幅基準と一定制御図の集約\n\n学習・探索・既存数値結果・誤差線図24枚は変更しない。FinalとminJλ Selected Bestの各72波形を全て既存NPZから再利用し、音声再生成は0。RL Selected Bestは33方策評価からの補助解で、未保存ネットワークの復元ではない。\n\nTargetの全帯域最大表示PSD={reference:.12g}を全Generatedにも適用。magma、表示範囲{limits} dBを全手法・λ・seedで固定し、各図に共通カラーバーを置いた。全144波形のdB分布を調べてから範囲を固定し、上端超過は0。下端未満は表示床であり音声や損失を変更していない。0 dBはTargetのピークPSD基準で、校正済みdB SPLではない。\n\n表示は10 MHzへのリサンプリング、Blackman-Harris窓245760点（24.576 ms）、overlap184320点、mean detrend、density PSD、境界のない完全窓、0–10 kHz。リサンプリングは高周波情報を増やさない。既存E_specは元50 kHzでの窓丸め・境界padding・detrend=False・spectrum scaling等が異なる。表示PSDを学習損失と同一視しない。\n\n5 ms RMSはacoustics.distanceと同じ250点kernel・same convolution、共通Targetピーク正規化。両包絡の平均絶対差が既存E_envと一致することを全144例で確認。終盤AC標準偏差の発声proxyとは別の量である。RMSの縦軸は全例で{rms_limits}に固定。振幅差を個別ピーク正規化で隠さない。\n\n[全seed図一覧](visualization_revision/FIGURES.md)はFinal/Selected Bestを区別する。Selected BestでDE/Randomの採用可能解を見る一方、Finalの実4096番目候補も残す。3Dの非持続Finalでは終盤AC振動が弱く、RMS包絡がTargetと不一致である。DC・低周波過渡成分によりRMS自体は残る例があり、発声proxyとRMS包絡を混同しない。初期過渡のPSDも選択解と区別できる。\n\n物理図は次元ごとにFinal/Selected Bestの計6枚へ集約。可変係数を行、λを列、4手法を横軸に配置。各seed点と平均±標本SDを表示し、目標は破線。同じ係数のλ間で縦軸を共有し、未制御値は表の固定値のまま。係数はモデル内部の単位であり、特にf0_lを基本周波数Hzと解釈しない。n=3から統計的確実性を主張しない。\n\n新図検証後に旧白黒スペクトログラム24枚と単独水平線48枚を削除。モデル・音声・CSV・run・manifest・Phase0・誤差線図24枚は保持する。[削除一覧](visualization_revision/deleted_images.txt)。新図は144 PSD＋144 RMS＋6パラメータ比較。[監査](visualization_revision/audit.json)に共通基準・保存値照合・保護を記録する。\n'''
    (WORK/'REPORT.md').write_text(report)


def publish():
    if EVIDENCE.exists():raise FileExistsError('Published visualization preserved')
    if not json.loads((WORK/'visual_review.json').read_text())['passed']:raise RuntimeError('Visual review required')
    protected=json.loads((WORK/'protected_before.json').read_text())
    for p,h in protected.items():assert digest(DEST/p)==h,p
    metadata=json.loads((WORK/'new_figure_sources.json').read_text())
    assert len(metadata)==294
    for r in metadata:assert digest(WORK/r['path'])==r['image_sha256']
    old=json.loads((WORK/'retired_images.json').read_text())
    for r in old:
        p=DEST/r['path'];assert not p.is_symlink() and digest(p)==r['sha256']
    for r in metadata:
        source=WORK/r['path'];target=DEST/r['path'];assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
    for r in old:safe_unlink(DEST/r['path'],DEST,r['sha256'])
    prior=json.loads((DEST/'figure_sources.json').read_text())
    prior['images']=[r for r in prior['images'] if 'field' in r]+metadata
    prior['audio_view_renderer_sha256']={p:digest(ROOT/p) for p in ('acoustic_views.py','issue17_visualization.py')}
    write_json(DEST/'figure_sources.json',prior)
    for name in ('README.md','REPORT.md'):(WORK/name).replace(DEST/name)
    (WORK/'deleted_images.txt').write_text('\n'.join(r['path'] for r in old)+'\n')
    audit=json.loads((WORK/'audit.json').read_text());audit.update(protected_original_files=len(protected),
        preserved_error_curve_images=24,deleted_old_spectrograms=24,deleted_old_horizontal_control_figures=48)
    write_json(WORK/'audit.json',audit)
    EVIDENCE.mkdir()
    for p in list(WORK.iterdir()):
        if p.is_file():p.rename(EVIDENCE/p.name)
    for p,h in protected.items():assert digest(DEST/p)==h,p
    print('Published common-reference audio views and six control comparisons; retired72 old images',flush=True)


def audit():
    """Verify saved evidence without generating any native sound or learning."""
    load_ready(DEST)
    protected=json.loads((EVIDENCE/'protected_before.json').read_text())
    for path,expected in protected.items():assert digest(DEST/path)==expected,path
    config=json.loads((EVIDENCE/'display_config.json').read_text())
    metadata=json.loads((DEST/'figure_sources.json').read_text())['images']
    assert len(metadata)==318
    for row in metadata:assert digest(DEST/row['path'])==row['image_sha256']
    for row in json.loads((EVIDENCE/'retired_images.json').read_text()):assert not (DEST/row['path']).exists()
    assert len([r for r in metadata if 'field' in r])==24
    assert len([r for r in metadata if r.get('kind')=='spectrogram'])==144
    assert len([r for r in metadata if r.get('kind')=='rms_envelope'])==144
    assert config['above_display_ceiling_count']==0
    result=dict(passed=True,protected_original_files=len(protected),current_images=318,preserved_error_images=24,
        retired_images_absent=72,existing_waveforms_reused=144,new_native_sounds=0,
        prior_numeric_audit='Historical 54/72-profile audits plus unchanged raw-data hashes; no new native replay required')
    write_json(EVIDENCE/'current_validation.json',result);print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('render','publish','audit'));a=p.parse_args()
    {'render':render,'publish':publish,'audit':audit}[a.action]()
