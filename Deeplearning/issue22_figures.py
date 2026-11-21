"""Eight compact figures, keeping common error, own objectives and policy state apart."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue22_data import ROOT,OUT,CONDITIONS,SEEDS,label,curve_for,folder_for,target_data
from .issue17_artifacts import read_record
from .issue15_analysis import write_json
from .issue16 import digest

COLORS={'legacy_sum':'#555555',.5:'#0072B2',.75:'#D55E00',.9:'#009E73'}


def title(c):return 'Old: 1+1' if c=='legacy_sum' else f'λ={c:.2f} (sum1)'


def render(rows,stability):
    plt.rcParams.update({'font.size':10,'path.simplify':False,'axes.spines.top':False,
        'axes.spines.right':False,'figure.constrained_layout.use':True})
    (OUT/'figures').mkdir(exist_ok=True);images=[]
    def save(fig,name,**meta):
        p=OUT/'figures'/f'{name}.png';fig.savefig(p,dpi=120);plt.close(fig)
        images.append(dict(path=str(p.relative_to(OUT)),sha256=digest(p),**meta))
    curves={(c,s):curve_for(c,s) for c in CONDITIONS for s in SEEDS}
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    for ax,field in zip(axes.ravel(),('E_total','E_spec','E_env','J')):
        for c in CONDITIONS:
            values=[]
            for s in SEEDS:
                values.append([p['acoustic_error'] if field=='E_total' else p['weighted_loss'] if field=='J' else p['acoustic_components']['log_spectral' if field=='E_spec' else 'envelope'] for p in curves[c,s]])
            y=np.array(values);x=[p['episodes'] for p in curves[c,0]];mean,sd=y.mean(0),y.std(0,ddof=1)
            ax.plot(x,mean,color=COLORS[c],label=title(c),lw=1.7);ax.fill_between(x,mean-sd,mean+sd,color=COLORS[c],alpha=.12)
        ax.set(title=field+(' (own objective; scales differ)' if field=='J' else ' (common scale)'),xlabel='Training candidates; 33 independent evaluations',ylabel=field,xlim=(0,4096));ax.legend(fontsize=8)
    save(fig,'evaluation_curves',description='33 independent policy evaluations; seed mean ± sample SD; own J is separated from common errors')

    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    for ax,role in zip(axes.ravel()[:3],('final','selected','posthoc')):
        for ci,c in enumerate(CONDITIONS):
            a=[r for r in rows if r['condition']==label(c) and r['role']==role]
            spec=np.array([r['E_spec'] for r in a]);env=np.array([r['E_env'] for r in a]);total=spec+env
            ax.bar(ci,spec.mean(),color=COLORS[c]);ax.bar(ci,env.mean(),bottom=spec.mean(),color=COLORS[c],alpha=.35,hatch='//')
            ax.errorbar(ci,total.mean(),yerr=total.std(ddof=1),fmt='k_',capsize=4);ax.scatter(ci+np.array([-.12,0,.12]),total,color='black',s=20,zorder=5)
        ax.set(title={'final':'Final deterministic policy','selected':'Selected: first min own J','posthoc':'Posthoc: min common E (diagnostic)'}[role],ylabel='Common E_spec + E_env')
        ax.axhline(.1,color='.4',ls=':',lw=1)
    ax=axes[1,1];ax.axhline(0,color='.4',lw=1)
    for ci,c in enumerate(CONDITIONS):
        y=np.array([r['selected_minus_final_E_total'] for r in stability if r['condition']==label(c)])
        ax.scatter(ci+np.array([-.12,0,.12]),y,color=COLORS[c],s=35);ax.errorbar(ci,y.mean(),yerr=y.std(ddof=1),fmt='k_',capsize=4)
    ax.set(title='Selected − Final (common error)',ylabel='Negative: Final worse than Selected')
    for ax in axes.ravel():ax.set(xticks=range(4),xticklabels=[title(c) for c in CONDITIONS])
    save(fig,'endpoints_and_stability',description='Solid spectral / hatched envelope; seed dots and sample SD; posthoc not substituted for Selected/Final')

    legacy=json.loads((OUT/'legacy_policy_statistics.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True)
    fields=('alpha','squashed_entropy_mc','gaussian_entropy',0,1,2)
    names=('Temperature α','Squashed entropy estimate [nats]','Pre-tanh Gaussian entropy [nats]',
        'Action std: ε_l (normalized)','Action std: p_s (normalized)','Action std: f_e,l (normalized; not Hz)')
    for ax,field,name in zip(axes.ravel(),fields,names):
        getter=lambda p:p['normalized_action_std_mc'][field] if isinstance(field,int) else p[field]
        for c in CONDITIONS[1:]:
            x=[p['episodes'] for p in curves[c,0]];y=np.array([[getter(p['policy_statistics']) for p in curves[c,s]] for s in SEEDS]);mean,sd=y.mean(0),y.std(0,ddof=1)
            ax.plot(x,mean,color=COLORS[c],label=title(c));ax.fill_between(x,mean-sd,mean+sd,color=COLORS[c],alpha=.12)
        for role,marker in (('final','x'),('selected','D')):
            a=[p for p in legacy if p['role']==role]
            ax.scatter([p['trial'] for p in a],[getter(p) for p in a],color=COLORS['legacy_sum'],marker=marker,s=30,label='Old '+role+' only',zorder=6)
        ax.set(title=name,xlabel='Training candidates',xlim=(0,4096))
        if field=='alpha':ax.set_yscale('log')
        ax.legend(fontsize=7)
    save(fig,'policy_dynamics',description='No inferred legacy time series; fixed local MC4096 for squashed entropy/std; differential entropy may be negative')

    fig,axes=plt.subplots(1,2,figsize=(14,5),sharex=True)
    for c in CONDITIONS:
        loss=[];ratio=[]
        for s in SEEDS:
            trace=read_record(folder_for(c,s)/'training_trace.json')
            j=np.abs(np.array([p['reward'] for p in trace])).reshape(32,128).mean(1);loss.append(j)
            if c!='legacy_sum':ratio.append(np.array([p['alpha_before_update'] for p in trace]).reshape(32,128).mean(1)/np.maximum(j,1e-12))
        x=np.arange(32)*128+64.5;y=np.array(loss);axes[0].plot(x,y.mean(0),color=COLORS[c],label=title(c));axes[0].fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=COLORS[c],alpha=.12)
        if ratio:
            y=np.array(ratio);axes[1].plot(x,y.mean(0),color=COLORS[c],label=title(c));axes[1].fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=COLORS[c],alpha=.12)
    axes[0].set(title='Sampled reward magnitude: mean |r|',ylabel='Own objective; scales/ratios differ')
    axes[1].set(title='Temperature / sampled reward magnitude',ylabel='Mean α before update / mean |r|')
    for ax in axes:ax.set(xlabel='Actual training candidates (128-sample bins)',xlim=(0,4096));ax.legend(fontsize=8)
    save(fig,'reward_temperature_scale',description='Training samples only, not policy evaluations; ratio is descriptive, not a causal entropy-dominance measure; old alpha trajectory unavailable')

    _,controls=target_data();fig,axes=plt.subplots(3,2,figsize=(13,11),sharey='row')
    for ri,(field,index,name) in enumerate((('epsilon_l',0,'ε_l [model units]'),('ps',2,'p_s [model units]'),('f0_l',3,'f_e,l [model units; not Hz]'))):
        for ci,role in enumerate(('final','selected')):
            ax=axes[ri,ci];ax.axhline(controls[index],color='.5',lw=1)
            for wi,c in enumerate(CONDITIONS):
                y=[r[field] for r in rows if r['condition']==label(c) and r['role']==role]
                ax.scatter(wi+np.array([-.12,0,.12]),y,color=COLORS[c],s=35)
            ax.set(title=role.title() if ri==0 else '',xticks=range(4),xticklabels=[title(c) for c in CONDITIONS],ylabel=name)
    save(fig,'physical_controls',description='All seeds, Final/Selected, one combined control figure; target reference, not constant control-vs-time plots')

    import csv,gzip
    with gzip.open(OUT/'critic_predictions.csv.gz','rt') as stream:predictions=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,4,figsize=(18,5))
    for ax,c in zip(axes,CONDITIONS):
        a=[r for r in predictions if r['condition']==label(c)];extrema=[float(r[k]) for r in a for k in ('actual_reward','Q1','Q2')];limits=(min(extrema)-.02,max(.02,max(extrema)+.02))
        for head,color,marker in (('Q1','#0072B2','o'),('Q2','#D55E00','^')):ax.scatter([float(r['actual_reward']) for r in a],[float(r[head]) for r in a],s=5,alpha=.12,color=color,marker=marker,rasterized=True,label=head)
        ax.plot(limits,limits,color='black',lw=1);ax.set(title=title(c),xlabel='Measured own r = −J',ylabel='Predicted Q',xlim=limits,ylim=limits);ax.legend(fontsize=8)
    save(fig,'critic_calibration',description='Each panel uses its own reward/axis scale; raw RMSE across panels is not a scale-free comparison; same1005 saved candidates')

    target,_=target_data();f,t,psd,reference=display_psd(target);peak=abs(target).max();target_env=rms_envelope(target,peak)
    sources=json.loads((OUT/'endpoint_sources.json').read_text());sounds={};views={};max_db=0.;max_env=float(target_env.max())
    for p in sources:
        if p['role']=='posthoc':continue
        with np.load(ROOT/p['audio_path']) as z:audio=z['audio'].copy()
        db=relative_db(display_psd(audio)[2],reference);env=rms_envelope(audio,peak);max_db=max(max_db,float(db.max()));max_env=max(max_env,float(env.max()))
        if p['seed']==1:sounds[p['condition'],p['role']]=audio/peak;views[p['condition'],p['role']]=(db,env)
    limits=(-70.,max(10.,float(10*np.ceil(max_db/10))));rms_limits=(0.,max_env*1.15);time=np.arange(len(target))/50000;crop=(time>=.04)&(time<=.06)
    amp=max(float(abs(target[crop]/peak).max()),max(float(abs(a[crop]).max()) for a in sounds.values()))*1.05
    for role in ('final','selected'):
        fig,axes=plt.subplots(4,4,figsize=(18,13))
        for ax in axes[0]:ax.axis('off')
        axes[0,0].axis('on');axes[0,0].pcolormesh(t,f/1000,relative_db(psd,reference),shading='gouraud',cmap='magma',vmin=limits[0],vmax=limits[1]);axes[0,0].set(title='Shared Target PSD',xlabel='Time [s]',ylabel='Frequency [kHz]',xlim=(t[0],t[-1]),ylim=(0,10))
        axes[0,1].text(.02,.45,'Fixed seed1 across all conditions\nOld: coefficients1+1; new: sum1\nSelected = first min own J, not min common E\nWaveform crop: 40–60 ms, no phase alignment',transform=axes[0,1].transAxes)
        for ci,c in enumerate(CONDITIONS):
            key=(label(c),role);db,env=views[key];a=sounds[key]
            mesh=axes[1,ci].pcolormesh(t,f/1000,db,shading='gouraud',cmap='magma',vmin=limits[0],vmax=limits[1],rasterized=True)
            axes[1,ci].set(title=title(c)+' '+role,xlabel='Time [s]',ylabel='Frequency [kHz]',xlim=(t[0],t[-1]),ylim=(0,10))
            axes[2,ci].plot(time[crop],target[crop]/peak,color='.65',lw=.8,label='Target');axes[2,ci].plot(time[crop],a[crop],color=COLORS[c],lw=.8,label='Generated')
            axes[2,ci].set(xlabel='Time [s]; display crop',ylabel='Raw audio / Target peak',xlim=(.04,.06),ylim=(-amp,amp))
            axes[3,ci].plot(time,target_env,color='.65',lw=.8);axes[3,ci].plot(time,env,color=COLORS[c],lw=.8)
            axes[3,ci].set(xlabel='Time [s]',ylabel='RMS / Target peak',xlim=(0,.2),ylim=rms_limits)
            if ci==0:axes[2,ci].legend(fontsize=8)
        fig.colorbar(mesh,ax=axes.ravel().tolist(),pad=.01,shrink=.45,label='Power [dB relative to shared Target maximum PSD]')
        save(fig,f'audio_{role}',description='Fixed seed1; shared Target PSD and gain, full RMS and explicitly display-cropped raw waveform',db_limits=limits,rms_ylim=rms_limits)
    write_json(OUT/'figure_sources.json',dict(images=images,source_sha256=digest(__file__),conditions=[label(c) for c in CONDITIONS],
        display=dict(target_reference_psd=reference,db_limits=limits,rms_ylim=rms_limits,cmap='magma',display_rate=10000000,
            window='blackmanharris',nperseg=245760,noverlap=184320,frequency_hz=[0,10000]),
        common_limits_cover_all_24_saved_final_selected_sounds=True,no_simulation=True,source_data=['comparison_per_seed.csv','policy_dynamics.csv','training_samples.csv.gz','endpoint_sources.json']))
    return images
