"""Ten aggregate figures; common Target-derived audio/display scales."""
from .result_paths import resolve as resolve_result_path
import gzip
import csv
import io
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .issue26_data import ROOT,OUT,CONDITIONS,SEEDS,RADII,folder_for,curve_for,chosen,target_data
from .issue26_analysis import csv_rows
from .issue17_artifacts import read_record
from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue15_analysis import write_json
from .issue16 import digest

COLORS=dict(A='#666666',B='#0072B2',C='#009E73',D='#D55E00')
LABELS=dict(A='A restricted (existing)',B='B wide SAC',C='C local f0 ±50k',D='D local f0 ±250k')
VARIABLES=('epsilon_l','ps','f0_l')


def gz_rows(name):return list(csv.DictReader(io.StringIO(gzip.decompress((OUT/name).read_bytes()).decode())))


def render():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
    (OUT/'figures').mkdir(exist_ok=True);images=[]
    def save(fig,name,description):
        p=OUT/'figures'/f'{name}.png';fig.savefig(p,dpi=120);plt.close(fig)
        images.append(dict(path=str(p.relative_to(OUT)),sha256=digest(p),description=description))
    curves={(c,s):curve_for(c,s) for c in CONDITIONS for s in SEEDS}
    traces={(c,s):read_record(folder_for(c,s)/'training_trace.json') for c in CONDITIONS for s in SEEDS}
    fig,axes=plt.subplots(4,3,figsize=(16,13),sharex=True,sharey='row')
    for seed in SEEDS:
        for row,field in enumerate(('E_total','E_spec','E_env','J')):
            ax=axes[row,seed]
            for c in CONDITIONS:
                getter=lambda p:p['acoustic_error'] if field=='E_total' else p['weighted_loss'] if field=='J' else p['acoustic_components']['log_spectral' if field=='E_spec' else 'envelope']
                curve=curves[c,seed];ax.plot([p['episodes'] for p in curve],[getter(p) for p in curve],'.-',ms=3,lw=1.2,color=COLORS[c],label=LABELS[c])
                for role,marker in (('selected','D'),('posthoc','*')):
                    p=chosen(curve)[role];ax.scatter(p['episodes'],getter(p),s=35,marker=marker,color=COLORS[c],edgecolor='black',linewidth=.4,zorder=5)
            ax.axvline(256,color='.7',ls=':',lw=.8);ax.set(title=f'Seed {seed}: {field}',ylabel=field,xlim=(0,4096));ax.legend(fontsize=7)
            if row==3:ax.set_xlabel('Training trials; diamonds=minJ, stars=min common E')
    save(fig,'error_histories','All33 Actor observations; four errors, same row scales across3 seeds; no .10 target line')

    rows=csv_rows('endpoints.csv');fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,role in zip(axes.ravel(),('final','selected','posthoc','training_best')):
        for i,c in enumerate(CONDITIONS):
            rs=[r for r in rows if r['condition']==c and r['role']==role];spec=np.array([float(r['E_spec']) for r in rs]);env=np.array([float(r['E_env']) for r in rs]);total=spec+env
            ax.bar(i,spec.mean(),color=COLORS[c]);ax.bar(i,env.mean(),bottom=spec.mean(),color=COLORS[c],alpha=.35,hatch='//')
            ax.errorbar(i,total.mean(),yerr=total.std(ddof=1),fmt='k_',capsize=4);ax.scatter(i+np.array([-.12,0,.12]),total,c='black',s=23,zorder=5)
        ax.set(title={'final':'Final Actor after4096','selected':'Selected Actor: minJ /33','posthoc':'Posthoc Actor: minE /33','training_best':'Best training candidate: minJ /4096'}[role],
            ylabel='E_spec + E_env',xticks=range(4),xticklabels=list(CONDITIONS))
    save(fig,'endpoints','Four distinct endpoints, components and all3 seeds; bars=means, whiskers=sample SD; training best is action-only')

    fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey='row')
    for seed in SEEDS:
        for c in CONDITIONS:
            t=traces[c,seed];running=np.minimum.accumulate([p['weighted_loss'] for p in t]);p=curves[c,seed]
            axes[0,seed].plot(np.arange(1,4097),running,color=COLORS[c],label=LABELS[c]);axes[1,seed].plot([r['episodes'] for r in p],[r['weighted_loss'] for r in p],'.-',color=COLORS[c],label=LABELS[c])
        for i,ax in enumerate(axes[:,seed]):
            ax.axvline(256,color='.7',ls=':');ax.set(title=f'Seed {seed}: '+('training incumbent J' if i==0 else 'deterministic Actor J'),ylabel='J (log scale)',xlabel='Training trials',yscale='log',xlim=(0,4096));ax.legend(fontsize=7)
    save(fig,'discovery_and_acquisition','Separate axes for actual candidate discovery and independent Actor acquisition; all trials retained, no oracle center')

    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,sharey='row')
    for j,variable in enumerate(VARIABLES):
        for seed in SEEDS:
            ax=axes[j,seed]
            for c in CONDITIONS:
                p=curves[c,seed];ax.plot([r['episodes'] for r in p],[r['physical_controls'][[0,2,3][j]] for r in p],'.-',color=COLORS[c],ms=3,label=LABELS[c])
            ax.axvline(256,color='.7',ls=':');ax.set(title=f'Seed {seed}: {variable}',ylabel='Physical '+variable,xlabel='Training trials');ax.legend(fontsize=7)
    save(fig,'actor_controls','Actual deterministic Actor physical controls; f0 is force coefficient, not measured frequency; common scales across all seeds')

    fig,axes=plt.subplots(4,2,figsize=(14,14),sharex=True,sharey='col')
    for i,c in enumerate(CONDITIONS):
        t=[p for seed in SEEDS for p in traces[c,seed]];f0=np.array([p['physical_controls'][3] for p in t]);voice=np.array([p['sustained_voiced'] for p in t])
        for j,index in enumerate((0,2)):
            ax=axes[i,j];physical=np.array([p['physical_controls'][index] for p in t])
            for v,label,color in ((False,'non-sustained proxy','#B0B0B0'),(True,'sustained proxy',COLORS[c])):
                mask=voice==v;ax.scatter(f0[mask],physical[mask],s=4,alpha=.25,color=color,label=label,rasterized=True)
            ax.axvline(800000,color='black',ls=':',lw=.8);ax.set(title=LABELS[c]+': all3 seeds /12288 candidates',ylabel=('epsilon_l' if j==0 else 'ps'),xlabel='Physical f0_l (not Hz)',xlim=(49000,1e7));ax.legend(fontsize=7)
    save(fig,'visited_joint_controls','Visited ε/f0 and ps/f0 combinations, with voiced proxy; density is not a physical-volume survey; A naturally occupies a smaller range')

    bins=csv_rows('f0_bins.csv');sources=csv_rows('behavior_sources.csv');fig,axes=plt.subplots(2,2,figsize=(14,9))
    labels=['49k–800k','800k–2.5M','2.5M–5M','5M–10M']
    for k,c in enumerate(CONDITIONS):
        rs=[r for r in bins if r['condition']==c];bounds=sorted(set(float(r['lower']) for r in rs));counts=np.array([sum(int(r['count']) for r in rs if float(r['lower'])==b) for b in bounds]);voiced=np.array([sum(int(r['voiced']) for r in rs if float(r['lower'])==b) for b in bounds])
        x=np.arange(4)+(k-1.5)*.18;axes[0,0].bar(x,counts/12288,width=.17,color=COLORS[c],label=LABELS[c]);axes[0,1].bar(x,np.divide(voiced,counts,out=np.full(4,np.nan),where=counts>0),width=.17,color=COLORS[c])
        for seed in SEEDS:
            ts=traces[c,seed];f=np.array([p['physical_controls'][3] for p in ts]);axes[1,0].plot(np.sort(f),np.arange(1,4097)/4096,color=COLORS[c],alpha=.7,lw=.8)
        for source,marker,offset in (('sac','o',-.08),('local','D',.08)):
            rs=[r for r in sources if r['condition']==c and r['source']==source]
            if rs:axes[1,1].scatter(k+offset+np.array([-.03,0,.03]),[float(r['hit_rate']) for r in rs],color=COLORS[c],marker=marker,s=30)
    for ax in axes[0]:ax.set(xticks=range(4),xticklabels=labels,ylim=(0,1.05))
    axes[0,0].legend(fontsize=7)
    axes[0,0].set(title='Fraction of actual training visits',ylabel='Visits /12288');axes[0,1].set(title='Sustained proxy conditional on visited bin',ylabel='Voiced / bin visits; absent bins blank')
    axes[1,0].axvline(800000,color='black',ls=':');axes[1,0].set(title='f0 empirical CDF: individual seeds',xlabel='Physical f0_l',ylabel='Cumulative visit fraction',xlim=(49000,1e7))
    axes[1,1].set(title='Strict incumbent improvement after256',xticks=range(4),xticklabels=list(CONDITIONS),ylabel='Improvements / source trials (o=SAC, diamond=local)')
    save(fig,'coverage_voice_and_hits','Fixed physical f0 bins; all visits including nonvoiced; conditional voice and source hit rates are descriptive, not efficacy-adjusted comparisons')

    local=gz_rows('local_displacements.csv.gz');centers=csv_rows('center_history.csv');fig,axes=plt.subplots(2,3,figsize=(16,9))
    for j,variable in enumerate(VARIABLES):
        for c in ('C','D'):
            vals=[float(r['fraction_radius_'+variable]) for r in local if r['condition']==c]
            axes[0,j].hist(vals,bins=np.linspace(-1,1,51),density=True,histtype='step',color=COLORS[c],label=LABELS[c])
        axes[0,j].set(title=variable+': actual local displacement / radius',xlabel='From previous training incumbent',ylabel='Density',xlim=(-1,1));axes[0,j].legend(fontsize=7)
        seed=SEEDS[j]
        for c in CONDITIONS:
            rs=[r for r in centers if r['condition']==c and int(r['seed'])==seed];axes[1,j].step([*[int(r['trial']) for r in rs],4096],[*[float(r['f0_l']) for r in rs],float(rs[-1]['f0_l'])],where='post',color=COLORS[c],label=LABELS[c])
        axes[1,j].set(title=f'Seed {seed}: incumbent center f0',xlabel='Training trials',ylabel='Physical f0_l',xlim=(0,4096),ylim=(49000,1e7));axes[1,j].legend(fontsize=7)
    save(fig,'local_displacement_and_centers','Intersection-uniform proposals may be asymmetric near bounds; actual reference deviations and first-minJ center histories, no clipping/grid/restarts')

    policies=csv_rows('policy_dynamics.csv');fig,axes=plt.subplots(2,3,figsize=(16,9))
    fields=('alpha','normalized_std_f0_l','physical_std_f0_l','physical_std_epsilon_l','physical_std_ps','squashed_entropy_mc')
    for ax,field in zip(axes.ravel(),fields):
        for c in CONDITIONS:
            a=np.array([[float(r[field]) for r in policies if r['condition']==c and int(r['seed'])==seed] for seed in SEEDS]);x=[p['episodes'] for p in curves[c,0]]
            for y in a:ax.plot(x,y,color=COLORS[c],alpha=.2,lw=.6)
            ax.plot(x,a.mean(0),color=COLORS[c],label=LABELS[c]);ax.fill_between(x,a.min(0),a.max(0),color=COLORS[c],alpha=.07)
        ax.set(title=field,xlabel='Training trials',ylabel='Mean + seed range');ax.legend(fontsize=7)
        if field=='alpha':ax.set_yscale('log')
    save(fig,'policy_exploration','Identical MC4096 samples at33 points; alpha is not std; wide f0 physical std uses half-range4,975,500 versus375,500 in A')

    audio_figures(save)
    assert len(images)==10;write_json(OUT/'figures.json',images);return images


def audio_figures(save):
    target,_=target_data();peak=float(abs(target).max());f,t,power,reference=display_psd(target)
    sounds=csv_rows('sound_index.csv');displays={'Target':(target,power)};maximum=float(relative_db(power,reference).max());max_wave=float(abs(target/peak).max());max_rms=float(rms_envelope(target,peak).max())
    for r in sounds:
        with np.load(resolve_result_path(ROOT/r['raw_npz'])) as z:audio=z['audio']
        _,_,p,_=display_psd(audio);key=(r['condition'],int(r['seed']),r['role']);displays[key]=(audio,p)
        maximum=max(maximum,float(relative_db(p,reference).max()));max_wave=max(max_wave,float(abs(audio/peak).max()));max_rms=max(max_rms,float(rms_envelope(audio,peak).max()))
    color_limits=(-80.,max(0.,float(np.ceil(maximum/5)*5)));seed=1
    for role in ('final','selected'):
        fig,axes=plt.subplots(5,3,figsize=(15,14));mesh=None
        for i,c in enumerate(('Target',*CONDITIONS)):
            audio,p=displays['Target' if c=='Target' else (c,seed,role)];db=relative_db(p,reference);mesh=axes[i,0].pcolormesh(t,f/1000,db,shading='auto',cmap='magma',vmin=color_limits[0],vmax=color_limits[1])
            axes[i,0].set(title=c+': '+('Target' if c=='Target' else LABELS[c]+f' seed{seed} {role}'),ylabel='kHz',xlabel='Time (s)',ylim=(0,10),xlim=(t[0],t[-1]))
            times=np.arange(len(audio))/50000;axes[i,1].plot(times,audio/peak,lw=.7,color='black');axes[i,1].set(title='Waveform:40–60ms crop, no alignment',xlabel='Time (s)',ylabel='Amplitude /Target peak',xlim=(.04,.06),ylim=(-1.05*max_wave,1.05*max_wave))
            env=rms_envelope(audio,peak);axes[i,2].plot(np.arange(len(env))/50000,env,color='black');axes[i,2].set(title='5ms RMS: full0.20s',xlabel='Time (s)',ylabel='RMS /Target peak',xlim=(0,.2),ylim=(0,1.05*max_rms))
        fig.colorbar(mesh,ax=axes[:,0],label='dB /Target full-band PSD reference')
        save(fig,'audio_'+role,'Seed1 fixed before new results; Target+four conditions; full-window STFT, common color/gain/limits across every endpoint; no time alignment')
    write_json(OUT/'display_settings.json',dict(representative_seed=seed,target_psd_reference=reference,target_peak=peak,color_limits_db=color_limits,
        shared_waveform_limits=[-1.05*max_wave,1.05*max_wave],shared_rms_limits=[0,1.05*max_rms],sample_rate=50000,display_resampling_rate=10000000,
        nperseg=245760,noverlap=184320,window='blackmanharris',frequency_limits=[0,10000],complete_windows_only=True,waveform_crop_s=[.04,.06],
        rms_s=.005,individual_normalization=False,extra_information_from_resampling=False,generated_sounds=0))


if __name__=='__main__':render()
