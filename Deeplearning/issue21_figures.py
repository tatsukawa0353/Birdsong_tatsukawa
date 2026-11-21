"""Eight integrated TD3/SAC-only figures from saved observations and raw audio."""
import csv
import gzip
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue21_data import ROOT,OUT,METHODS,WIDTHS,SEEDS,curve_for,target_data
from .issue15_analysis import write_json
from .issue16 import digest

COLORS={32:'#777777',64:'#0072B2',128:'#D55E00',256:'#009E73'}


def render(rows,deltas):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'path.simplify':False,'figure.constrained_layout.use':True})
    (OUT/'figures').mkdir(exist_ok=True);images=[]
    def save(fig,name,**extra):
        path=OUT/'figures'/f'{name}.png';fig.savefig(path,dpi=120);plt.close(fig)
        images.append(dict(path=str(path.relative_to(OUT)),sha256=digest(path),**extra))
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,sharey=True)
    for mi,method in enumerate(METHODS):
        for ci,role in enumerate(('final','selected')):
            ax=axes[mi,ci]
            for width in WIDTHS:
                curves=[curve_for(method,width,s) for s in SEEDS]
                x=[p['episodes'] for p in curves[0]];values=[]
                for curve in curves:
                    best=float('inf');series=[]
                    for p in curve:
                        best=min(best,p['weighted_loss']);series.append(p['acoustic_error'] if role=='final' else best)
                    values.append(series)
                y=np.array(values);mean,sd=y.mean(0),y.std(0,ddof=1)
                ax.plot(x,mean,lw=1.7,color=COLORS[width],label=f'[{width},{width}]')
                ax.fill_between(x,mean-sd,mean+sd,color=COLORS[width],alpha=.12)
            ax.set(title=method+(': independent policy evaluation' if role=='final' else ': Selected min J archive'),
                xlabel='Training candidates (33 separate evaluations)',ylabel='Common E_total',xlim=(0,4096))
            ax.legend(fontsize=9,ncol=2)
    save(fig,'learning_curves',description='All 33 observations retained; mean ± sample SD; no training rewards mixed into evaluation curves')

    fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
    for mi,method in enumerate(METHODS):
        for ci,role in enumerate(('final','selected')):
            ax=axes[mi,ci]
            for wi,width in enumerate(WIDTHS):
                subset=[r for r in rows if r['method']==method and r['width']==width and r['role']==role]
                spec=np.array([r['E_spec'] for r in subset]);env=np.array([r['E_env'] for r in subset]);total=spec+env
                ax.bar(wi,spec.mean(),color=COLORS[width]);ax.bar(wi,env.mean(),bottom=spec.mean(),color=COLORS[width],alpha=.35,hatch='//')
                ax.errorbar(wi,total.mean(),yerr=total.std(ddof=1),fmt='k_',capsize=4)
                ax.scatter(wi+np.array([-.12,0,.12]),total,color='black',s=20,zorder=5)
            ax.axhline(.1,color='.3',ls=':',lw=1,label='Provisional mean target .10')
            ax.set(title=f'{method} {role.title()}',xticks=range(4),xticklabels=WIDTHS,ylabel='Common E_spec + E_env',xlabel='Hidden width (two layers)')
            ax.legend(fontsize=8)
    save(fig,'endpoint_errors',description='Solid spectral + hatched envelope; individual seed totals and sample SD; goal applies to Final mean and all-seed voicing')

    fig,axes=plt.subplots(2,3,figsize=(16,8),sharey='col')
    for mi,method in enumerate(METHODS):
        for ci,field in enumerate(('E_total','E_spec','E_env')):
            ax=axes[mi,ci];ax.axhline(0,color='.5',lw=1)
            for wi,width in enumerate(WIDTHS[1:]):
                for offset,role,marker in ((-.12,'final','o'),(.12,'selected','^')):
                    y=np.array([r['delta_'+field] for r in deltas if r['method']==method and r['width']==width and r['role']==role])
                    ax.scatter(wi+offset+np.array([-.035,0,.035]),y,color=COLORS[width],marker=marker,s=35)
                    ax.errorbar(wi+offset,y.mean(),yerr=y.std(ddof=1),fmt='k_',capsize=4)
            ax.set(title=f'{method}: paired Δ {field}',xticks=range(3),xticklabels=WIDTHS[1:],xlabel='Width; circle Final, triangle Selected',ylabel='Wider − width32 (negative: better)')
    save(fig,'paired_differences',description='Same-seed differences against width32, dots all seeds and sample SD; no assumption of identical trajectories')

    target,controls=target_data();fig,axes=plt.subplots(3,2,figsize=(14,11),sharey='row')
    for ri,(field,index,name) in enumerate((('epsilon_l',0,'ε_l [model units]'),('ps',2,'p_s [model units]'),('f0_l',3,'f_e,l [model units; not Hz]'))):
        for mi,method in enumerate(METHODS):
            ax=axes[ri,mi];ax.axhline(controls[index],color='.45',lw=1,label='Target')
            for wi,width in enumerate(WIDTHS):
                for offset,role,marker in ((-.12,'final','o'),(.12,'selected','^')):
                    values=[r[field] for r in rows if r['method']==method and r['width']==width and r['role']==role]
                    ax.scatter(wi+offset+np.array([-.035,0,.035]),values,color=COLORS[width],marker=marker,s=30,label=role if wi==0 else None)
            ax.set(title=method if ri==0 else '',xticks=range(4),xticklabels=WIDTHS,xlabel='Hidden width',ylabel=name)
            if ri==0:ax.legend(fontsize=8)
    save(fig,'physical_controls',description='One combined figure; both endpoints, all seeds, model-internal values and shared target references')

    with gzip.open(OUT/'critic_predictions.csv.gz','rt') as stream:
        predictions=[r for r in csv.DictReader(stream) if r['source'] not in ('own_training','actor')]
    limits=(min(float(r[k]) for r in predictions for k in ('actual_reward','Q1','Q2'))-.02,
            max(.02,max(float(r[k]) for r in predictions for k in ('actual_reward','Q1','Q2'))+.02))
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    for mi,method in enumerate(METHODS):
        for wi,width in enumerate(WIDTHS):
            ax=axes[mi,wi];part=[r for r in predictions if r['method']==method and int(r['width'])==width]
            for head,color,marker in (('Q1','#0072B2','o'),('Q2','#D55E00','^')):
                ax.scatter([float(r['actual_reward']) for r in part],[float(r[head]) for r in part],color=color,marker=marker,s=5,alpha=.12,rasterized=True,label=head)
            ax.plot(limits,limits,color='black',lw=1)
            ax.set(title=f'{method} [{width},{width}]',xlabel='Measured r = −(E_spec + E_env)',ylabel='Predicted Q',xlim=limits,ylim=limits)
            if mi==0 and wi==0:ax.legend(fontsize=9)
    save(fig,'critic_scatter',description='Only TD3/SAC compared; Q1 and Q2 separate colors/markers; same 1005 saved candidates x3 seeds')

    with (OUT/'critic_metrics.csv').open() as stream:metrics=list(csv.DictReader(stream))
    fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True,sharey='col')
    descriptors=(('fixed_pool','rmse','Fixed-pool RMSE'),('fixed_pool','spearman','Fixed-pool Spearman'),
        ('low_J_decile','pairwise_inversion_fraction','Low-J ranking inversions'),('low_J_decile','bias_Q_minus_r','Low-J bias Q − r'))
    for mi,method in enumerate(METHODS):
        for ci,(subset,field,label) in enumerate(descriptors):
            ax=axes[mi,ci]
            for offset,critic,color,marker in ((-.08,'Q1','#0072B2','o'),(0,'Q2','#D55E00','^'),(.08,'minQ','#009E73','s')):
                mean=[];sd=[]
                for width in WIDTHS:
                    values=np.array([float(r[field]) if r[field] else np.nan for r in metrics if r['method']==method and int(r['width'])==width and r['critic']==critic and r['subset']==subset])
                    mean.append(np.nanmean(values));sd.append(np.nanstd(values,ddof=1))
                ax.errorbar(np.arange(4)+offset,mean,yerr=sd,fmt=marker+'-',color=color,lw=1,capsize=3,label=critic)
            ax.set(title=f'{method}: {label}',xticks=range(4),xticklabels=WIDTHS,xlabel='Hidden width');ax.legend(fontsize=8)
    save(fig,'critic_metrics',description='Same reward scale and pool for every width; per-seed metrics averaged with sample SD; ties excluded from inversion denominator')

    sources=json.loads((OUT/'endpoint_sources.json').read_text());f,t,psd,reference=display_psd(target)
    peak=abs(target).max();target_db=relative_db(psd,reference);target_env=rms_envelope(target,peak)
    sounds={};views={};max_db=0.;max_rms=float(target_env.max())
    for r in sources:
        with np.load(ROOT/r['audio_path']) as z:audio=z['audio'].copy()
        db=relative_db(display_psd(audio)[2],reference);env=rms_envelope(audio,peak)
        max_db=max(max_db,float(db.max()));max_rms=max(max_rms,float(env.max()))
        if r['seed']==1:
            key=(r['method'],r['width'],r['role']);sounds[key]=audio/peak;views[key]=(db,env)
    db_limits=(-70.,max(10.,float(10*np.ceil(max_db/10))));rms_limits=(0.,max_rms*1.15)
    time=np.arange(len(target))/50000;crop=(time>=.04)&(time<=.06)
    wave_limit=max(float(abs(target[crop]/peak).max()),max(float(abs(a[crop]).max()) for a in sounds.values()))*1.05
    for role in ('final','selected'):
        fig,axes=plt.subplots(7,4,figsize=(18,20))
        for ax in axes[0]:ax.axis('off')
        axes[0,0].axis('on');mesh=axes[0,0].pcolormesh(t,f/1000,target_db,shading='gouraud',cmap='magma',vmin=db_limits[0],vmax=db_limits[1])
        axes[0,0].set(title='Shared Target PSD',xlabel='Time [s]',ylabel='Frequency [kHz]',xlim=(t[0],t[-1]),ylim=(0,10))
        axes[0,1].text(.02,.5,'Fixed seed1 across all widths / methods\nRaw waveform display crop: 40–60 ms\nNo phase alignment or per-sound gain\nRMS: full 0.20 s, Target peak normalization',transform=axes[0,1].transAxes)
        for mi,method in enumerate(METHODS):
            for wi,width in enumerate(WIDTHS):
                key=(method,width,role);db,env=views[key];a=sounds[key];r=1+mi*3
                mesh=axes[r,wi].pcolormesh(t,f/1000,db,shading='gouraud',cmap='magma',vmin=db_limits[0],vmax=db_limits[1],rasterized=True)
                axes[r,wi].set(title=f'{method} [{width},{width}] {role}',xlabel='Time [s]',ylabel='Frequency [kHz]',xlim=(t[0],t[-1]),ylim=(0,10))
                axes[r+1,wi].plot(time[crop],target[crop]/peak,color='.6',lw=.8,label='Target')
                axes[r+1,wi].plot(time[crop],a[crop],color=COLORS[width],lw=.8,label='Generated')
                axes[r+1,wi].set(xlabel='Time [s]; display crop',ylabel='Raw audio / Target peak',xlim=(.04,.06),ylim=(-wave_limit,wave_limit))
                axes[r+2,wi].plot(time,target_env,color='.6',lw=.8,label='Target');axes[r+2,wi].plot(time,env,color=COLORS[width],lw=.8,label='Generated')
                axes[r+2,wi].set(xlabel='Time [s]',ylabel='RMS / Target peak',xlim=(0,.2),ylim=rms_limits)
                if mi==0 and wi==0:axes[r+1,wi].legend(fontsize=8)
        fig.colorbar(mesh,ax=axes.ravel().tolist(),shrink=.4,pad=.01,label='Power [dB relative to shared Target maximum PSD]')
        save(fig,f'audio_{role}',description='Fixed seed1; PSD + display-cropped raw waveform + full RMS; no per-seed image generation',
            db_limits=db_limits,rms_ylim=rms_limits,waveform_crop_s=[.04,.06],target_reference_psd=reference)
    write_json(OUT/'figure_sources.json',dict(images=images,source_sha256=digest(__file__),
        comparison_methods=list(METHODS),widths=list(WIDTHS),baseline='Issue18 lambda1 read only',
        target_reference_psd=reference,db_limits=db_limits,rms_ylim=rms_limits,shared_limits_cover_all_48_endpoints=True,
        display=dict(sample_rate=10000000,nperseg=245760,noverlap=184320,window='blackmanharris',frequency_hz=[0,10000],cmap='magma'),
        sources=['comparison_per_seed.csv','paired_width_deltas.csv','critic_predictions.csv.gz','endpoint_sources.json']))
    return images
