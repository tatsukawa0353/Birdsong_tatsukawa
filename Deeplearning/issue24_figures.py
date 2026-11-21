"""Nine aggregate comparisons with common error axes and shared acoustic gain."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .issue24_data import ROOT,BASE,OUT,RATES,BASE_LR,SEEDS,NAMES,folder_for,curve_for,chosen,target_data
from .issue17_artifacts import read_record
from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue15_analysis import write_json
from .issue16 import digest

COLORS={3e-4:'#555555',1e-4:'#0072B2',3e-5:'#D55E00'}


def rate_name(rate):return {3e-4:'Actor LR3e-4 (existing)',1e-4:'Actor LR1e-4',3e-5:'Actor LR3e-5'}[rate]


def csv_rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def render():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
    (OUT/'figures').mkdir(exist_ok=True);images=[]
    def save(fig,name,description):
        p=OUT/'figures'/f'{name}.png';fig.savefig(p,dpi=120);plt.close(fig)
        images.append(dict(path=str(p.relative_to(OUT)),sha256=digest(p),description=description))
    curves={(r,s):curve_for(r,s) for r in RATES for s in SEEDS}
    fig,axes=plt.subplots(4,3,figsize=(16,13),sharex=True,sharey='row')
    for seed in SEEDS:
        for row,field in enumerate(('E_total','E_spec','E_env','J')):
            ax=axes[row,seed]
            for rate in RATES:
                c=curves[rate,seed];getter=lambda p:p['acoustic_error'] if field=='E_total' else p['weighted_loss'] if field=='J' else p['acoustic_components']['log_spectral' if field=='E_spec' else 'envelope']
                ax.plot([p['episodes'] for p in c],[getter(p) for p in c],'.-',ms=3,color=COLORS[rate],label=rate_name(rate),lw=1.3)
                if field in ('E_total','J'):
                    for role,marker in (('selected','D'),('posthoc','*')):
                        p=chosen(c)[role];ax.scatter(p['episodes'],getter(p),s=40,marker=marker,color=COLORS[rate],edgecolor='black',linewidth=.5,zorder=5)
            if field=='E_total':ax.axhline(.1,color='.5',ls=':',label='E=.10 reference')
            ax.set(title=f'Seed {seed}: {field}',ylabel=field,xlim=(0,4096));ax.legend(fontsize=7)
            if row==3:ax.set_xlabel('Training candidates; diamonds=minJ, stars=minE')
    save(fig,'error_histories','Four errors with identical axes across3 seeds; all33 observations,3 LRs; Selected and posthoc marked separately')

    endpoints=csv_rows('comparison_per_seed.csv');stability=csv_rows('stability_and_acquisition.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,role in zip(axes.ravel()[:3],('final','selected','posthoc')):
        for i,rate in enumerate(RATES):
            rows=[r for r in endpoints if float(r['actor_lr'])==rate and r['role']==role];spec=np.array([float(r['E_spec']) for r in rows]);env=np.array([float(r['E_env']) for r in rows]);total=spec+env
            ax.bar(i,spec.mean(),color=COLORS[rate]);ax.bar(i,env.mean(),bottom=spec.mean(),color=COLORS[rate],alpha=.35,hatch='//')
            ax.errorbar(i,total.mean(),yerr=total.std(ddof=1),fmt='k_',capsize=4);ax.scatter(i+np.array([-.12,0,.12]),total,c='black',s=22,zorder=5)
        ax.set(title={'final':'Final: post-final update','selected':'Selected: minJ','posthoc':'Posthoc: min common E'}[role],ylabel='Common E_spec + E_env');ax.axhline(.1,color='.5',ls=':')
    ax=axes[1,1];ax.axhline(0,color='.5')
    for i,rate in enumerate(RATES):
        rows=[r for r in stability if float(r['actor_lr'])==rate];y=np.array([float(r['selected_minus_final_E']) for r in rows])
        ax.scatter(i+np.array([-.12,0,.12]),y,color=COLORS[rate]);ax.errorbar(i,y.mean(),yerr=y.std(ddof=1),fmt='k_',capsize=4)
    ax.set(title='Selected − Final E_total',ylabel='Negative: Final worse than minJ-selected policy')
    for ax in axes.ravel():ax.set(xticks=range(3),xticklabels=['3e-4 existing','1e-4','3e-5'])
    save(fig,'endpoints_and_stability','All3 seeds as dots; sample SD; separate Final/minJ/minE; spectral/envelope components stacked')

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for i,rate in enumerate(RATES):
        rows=[r for r in stability if float(r['actor_lr'])==rate]
        x=i+np.array([-.16,0,.16]);arrival=[float(r['first_voiced_E_le_0p10']) if r['first_voiced_E_le_0p10'] else 4500 for r in rows]
        axes[0].scatter(x,arrival,color=COLORS[rate],s=40)
        for xx,yy,row in zip(x,arrival,rows):axes[0].annotate('never' if yy==4500 else f"s{row['seed']}",(xx,yy),xytext=(0,5),textcoords='offset points',fontsize=7,ha='center')
        axes[1].scatter(x,[int(r['voiced_policy_threshold_count']) for r in rows],color=COLORS[rate],s=40)
        axes[2].scatter(x,[float(r['retention_fraction']) if r['retention_fraction'] else np.nan for r in rows],color=COLORS[rate],s=40)
    axes[0].axhline(4096,color='.6',ls=':');axes[0].set(title='First E≤.10 + sustained proxy',ylabel='Observed independent-evaluation trial',ylim=(-100,4900))
    axes[1].set(title='Successful independent evaluations',ylabel='Count among33',ylim=(-1,34))
    axes[2].set(title='Retention after first observed success',ylabel='Successful points / all subsequent evaluation points',ylim=(-.05,1.05))
    for ax in axes:ax.set(xticks=range(3),xticklabels=['3e-4 existing','1e-4','3e-5'])
    fig.supxlabel('Left / center / right offsets = seed0 / seed1 / seed2; no arrival is shown above budget')
    save(fig,'acquisition_and_retention','Non-arrivals explicitly shown; retention absent when never reached; three seed dots retained')

    intervals=csv_rows('update_intervals.csv');fig,axes=plt.subplots(2,3,figsize=(16,9))
    for seed in SEEDS:
        for rate in RATES:
            updates=read_record(folder_for(rate,seed)/'updates.json');x=np.sort([u['delta_action_l2'] for u in updates]);y=np.arange(1,len(x)+1)/len(x)
            axes[0,seed].plot(x,y,color=COLORS[rate],label=rate_name(rate))
            rows=[r for r in intervals if float(r['actor_lr'])==rate and int(r['seed'])==seed]
            n=[int(r['end']) for r in rows]
            axes[1,seed].plot(n,[float(r['path_length']) for r in rows],color=COLORS[rate],label=rate_name(rate)+' path')
            axes[1,seed].plot(n,[float(r['net_distance']) for r in rows],color=COLORS[rate],ls='--',label=rate_name(rate)+' net')
        axes[0,seed].set(title=f'Seed {seed}:4080 Actor updates',xlabel='One-update ||Δa||₂',ylabel='Empirical cumulative fraction',xscale='symlog');axes[0,seed].set_xscale('symlog',linthresh=1e-5)
        axes[1,seed].set(title=f'Seed {seed}:128-candidate intervals',xlabel='Training candidates',ylabel='Σ||Δa_update||₂ versus ||a_end−a_start||₂')
        for ax in axes[:,seed]:ax.legend(fontsize=7)
    save(fig,'actor_movement','All update magnitudes, not LR-proportional predictions; per-interval path and net distinguish reversals')

    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for j,name in enumerate(('actor','critic','entropy')):
        for rate in RATES:
            # New lines are measured; baseline explicitly comes from frozen config/scheduler.
            c=curves[rate,0];x=[p['episodes'] for p in c];y=[p['optimizer_lrs'][name][0] if 'optimizer_lrs' in p else 3e-4 for p in c]
            axes[0,j].plot(x,y,color=COLORS[rate],ls='--' if rate==BASE_LR else '-',label=rate_name(rate))
            loss={'actor':'actor_loss','critic':'critic_loss','entropy':'ent_coef_loss'}[name]
            values=[]
            for seed in SEEDS:
                rows=[r for r in intervals if float(r['actor_lr'])==rate and int(r['seed'])==seed];values.append([float(r[loss]) for r in rows])
            a=np.array(values);n=[int(r['end']) for r in rows];axes[1,j].plot(n,a.mean(0),color=COLORS[rate],label=rate_name(rate));axes[1,j].fill_between(n,a.mean(0)-a.std(0,ddof=1),a.mean(0)+a.std(0,ddof=1),color=COLORS[rate],alpha=.1)
        axes[0,j].set(title=name+' optimizer effective LR',xlabel='Training candidates',ylabel='param_groups[0].lr',yscale='log')
        axes[1,j].set(title=name+' loss:interval means',xlabel='Training candidates',ylabel='Seed mean ± sample SD')
        for ax in axes[:,j]:ax.legend(fontsize=7)
    save(fig,'learning_rates_and_losses','New rates measured before/after all3 optimizers and33 evals; baseline dashed provenance; losses actual batch32 optimizer records')

    fig,axes=plt.subplots(2,3,figsize=(16,9));fields=('alpha','squashed_entropy_mc','gaussian_entropy',0,1,2)
    for ax,field in zip(axes.ravel(),fields):
        for rate in RATES:
            getter=lambda p:p['normalized_action_std_mc'][field] if isinstance(field,int) else p[field]
            y=np.array([[getter(p['policy_statistics']) for p in curves[rate,s]] for s in SEEDS]);x=[p['episodes'] for p in curves[rate,0]]
            for yi in y:ax.plot(x,yi,color=COLORS[rate],alpha=.25,lw=.6)
            ax.plot(x,y.mean(0),color=COLORS[rate],label=rate_name(rate));ax.fill_between(x,y.min(0),y.max(0),color=COLORS[rate],alpha=.08)
        ax.set(title=(f'Post-tanh std: {NAMES[field]}' if isinstance(field,int) else field),xlabel='Training candidates',ylabel='Seed mean, thin lines/range:3 seeds');ax.legend(fontsize=7)
        if field=='alpha':ax.set_yscale('log')
    save(fig,'temperature_and_exploration','MC4096, same local RNG samples; seed means + individual/range, no alpha/std equivalence assumed')

    predictions=read_record(OUT/'critic_predictions.json');fig,axes=plt.subplots(3,3,figsize=(14,13),sharex=True,sharey=True)
    bound=[r['reward'] for r in predictions]+[r['Q1'] for r in predictions]+[r['Q2'] for r in predictions];low,high=min(bound)-.01,max(bound)+.01
    for i,rate in enumerate(RATES):
        for seed in SEEDS:
            rows=[r for r in predictions if r['actor_lr']==rate and r['seed']==seed];ax=axes[i,seed]
            ax.scatter([r['reward'] for r in rows],[r['Q1'] for r in rows],s=12,color='#0072B2',alpha=.5,label='Q1')
            ax.scatter([r['reward'] for r in rows],[r['Q2'] for r in rows],s=12,color='#D55E00',alpha=.5,label='Q2')
            ax.plot([low,high],[low,high],color='black',ls=':',lw=1);ax.set(title=rate_name(rate)+f',seed{seed}',xlabel='Measured r=−J (same99 candidates)',ylabel='Final Critic prediction',xlim=(low,high),ylim=(low,high));ax.legend(fontsize=7)
    save(fig,'final_critic_calibration','Identical99 retrospective actions/rewards, same axes, both Q heads; no new native calls or claim of corrected local gradients')

    target,_=target_data();peak=float(np.max(abs(target)));f,t,target_power,reference=display_psd(target)
    audios={};displays={};maximum=float(relative_db(target_power,reference).max())
    # Shared display limits across all saved endpoints and all seeds.
    for rate in RATES:
        for seed in SEEDS:
            for role in ('final','selected','posthoc'):
                with np.load(folder_for(rate,seed)/f'{role}.npz') as z:a=z['audio'].copy()
                audios[rate,seed,role]=a;ff,tt,power,_=display_psd(a);db=relative_db(power,reference)
                if seed==1 and role in ('final','selected'):displays[rate,role]=(ff,tt,db)
                maximum=max(maximum,float(db.max()))
    clim=(-70,max(10.,float(np.ceil(maximum/10)*10)))
    waveform_scale=max(float(np.max(abs(a/peak))) for a in (target,*audios.values()))*1.05
    envelope_scale=max(float(rms_envelope(a,peak).max()) for a in (target,*audios.values()))*1.05
    for role in ('final','selected'):
        fig,axes=plt.subplots(4,3,figsize=(16,12));waves=[('Target',target,(f,t,relative_db(target_power,reference)))]
        waves.extend((rate_name(rate)+':seed1 '+role,audios[rate,1,role],displays[rate,role]) for rate in RATES)
        for i,(name,a,(ff,tt,db)) in enumerate(waves):
            times=np.arange(len(a))/50000;mask=(times>=.04)&(times<=.06)
            axes[i,0].plot(times[mask],a[mask]/peak,color='black',lw=.7);axes[i,0].set(title=name+';40–60ms',xlabel='Time [s]',ylabel='Amplitude / Target peak',ylim=(-waveform_scale,waveform_scale))
            mesh=axes[i,1].pcolormesh(tt,ff,db,shading='auto',cmap='magma',vmin=clim[0],vmax=clim[1]);axes[i,1].set(title='PSD / one Target reference',xlabel='Time [s]',ylabel='Frequency [Hz]',xlim=(tt[0],tt[-1]),ylim=(0,10000));fig.colorbar(mesh,ax=axes[i,1],label='dB re Target PSD max')
            e=rms_envelope(a,peak);axes[i,2].plot(np.arange(len(e))/50000,e,color='black');axes[i,2].set(title='5ms RMS envelope',xlabel='Time [s]',ylabel='RMS / Target peak',xlim=(0,.2),ylim=(0,envelope_scale))
        save(fig,'audio_'+role,'Seed1 fixed in advance; common Target PSD/peak and color limits across all27 endpoints; no phase alignment or per-sound normalization')
    assert len(images)==9
    write_json(OUT/'figures.json',images)
    write_json(OUT/'display_settings.json',dict(target_psd_reference=reference,target_peak=peak,color_limits_db=clim,representative_seed=1,
        sample_rate=50000,display_resampling_rate=10000000,nperseg=245760,noverlap=184320,window='blackmanharris',frequency_limits=[0,10000],
        complete_windows_only=True,waveform_crop_s=[.04,.06],rms_s=.005,individual_normalization=False,
        shared_waveform_limits=[-waveform_scale,waveform_scale],shared_rms_limits=[0,envelope_scale]))
    return images


if __name__=='__main__':render()
