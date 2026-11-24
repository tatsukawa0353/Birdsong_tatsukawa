"""Ten figures separating online imitation from fixed-data Critic regression."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .issue25_data import ROOT,BASE,OUT,RATES,BASE_LR,SEEDS,NAMES,folder_for,curve_for,chosen,target_data
from .issue17_artifacts import read_record
from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue15_analysis import write_json
from .issue16 import digest

COLORS={3e-4:'#555555',1e-4:'#0072B2',1e-3:'#D55E00'}


def rate_name(rate):return {3e-4:'Critic LR3e-4 (existing)',1e-4:'Critic LR1e-4',1e-3:'Critic LR1e-3'}[rate]


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
            rows=[r for r in endpoints if float(r['critic_lr'])==rate and r['role']==role];spec=np.array([float(r['E_spec']) for r in rows]);env=np.array([float(r['E_env']) for r in rows]);total=spec+env
            ax.bar(i,spec.mean(),color=COLORS[rate]);ax.bar(i,env.mean(),bottom=spec.mean(),color=COLORS[rate],alpha=.35,hatch='//')
            ax.errorbar(i,total.mean(),yerr=total.std(ddof=1),fmt='k_',capsize=4);ax.scatter(i+np.array([-.12,0,.12]),total,c='black',s=22,zorder=5)
        ax.set(title={'final':'Final: post-final update','selected':'Selected: minJ','posthoc':'Posthoc: min common E'}[role],ylabel='Common E_spec + E_env');ax.axhline(.1,color='.5',ls=':')
    ax=axes[1,1];ax.axhline(0,color='.5')
    for i,rate in enumerate(RATES):
        rows=[r for r in stability if float(r['critic_lr'])==rate];y=np.array([float(r['selected_minus_final_E']) for r in rows])
        ax.scatter(i+np.array([-.12,0,.12]),y,color=COLORS[rate]);ax.errorbar(i,y.mean(),yerr=y.std(ddof=1),fmt='k_',capsize=4)
    ax.set(title='Selected − Final E_total',ylabel='Negative: Final worse than minJ-selected policy')
    for ax in axes.ravel():ax.set(xticks=range(3),xticklabels=['3e-4 existing','1e-4','1e-3'])
    save(fig,'endpoints_and_stability','All3 seeds as dots; sample SD; separate Final/minJ/minE; spectral/envelope components stacked')

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
            rows=[r for r in predictions if r['critic_lr']==rate and r['seed']==seed];ax=axes[i,seed]
            ax.scatter([r['reward'] for r in rows],[r['Q1'] for r in rows],s=12,color='#0072B2',alpha=.5,label='Q1')
            ax.scatter([r['reward'] for r in rows],[r['Q2'] for r in rows],s=12,color='#D55E00',alpha=.5,label='Q2')
            ax.plot([low,high],[low,high],color='black',ls=':',lw=1);ax.set(title=rate_name(rate)+f',seed{seed}',xlabel='Measured r=−J (same99 candidates)',ylabel='Final Critic prediction',xlim=(low,high),ylim=(low,high));ax.legend(fontsize=7)
    save(fig,'final_critic_calibration','Identical99 retrospective actions/rewards, same axes, both Q heads; no new native calls or claim of corrected local gradients')

    target,_=target_data();peak=float(np.max(abs(target)));f,t,target_power,reference=display_psd(target)
    audios={};displays={};maximum=float(relative_db(target_power,reference).max())
    # Shared display limits across all saved endpoints and all seeds.
    for rate in RATES:
        for seed in SEEDS:
            for role in ('final','selected','posthoc','first_success'):
                path=folder_for(rate,seed)/f'{role}.npz'
                if not path.exists():continue
                with np.load(path) as z:a=z['audio'].copy()
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
    extra_figures(save,curves)
    assert len(images)==10
    write_json(OUT/'figures.json',images)
    write_json(OUT/'display_settings.json',dict(target_psd_reference=reference,target_peak=peak,color_limits_db=clim,representative_seed=1,
        sample_rate=50000,display_resampling_rate=10000000,nperseg=245760,noverlap=184320,window='blackmanharris',frequency_limits=[0,10000],
        complete_windows_only=True,waveform_crop_s=[.04,.06],rms_s=.005,individual_normalization=False,
        shared_waveform_limits=[-waveform_scale,waveform_scale],shared_rms_limits=[0,envelope_scale]))
    return images


def extra_figures(save,curves):
    offline_name=lambda rate:rate_name(rate).replace(' (existing)',' (new regression baseline)')
    history=csv_rows('offline_history.csv');final=csv_rows('offline_final_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    specifications=(('train','all','rmse','Train RMSE:12288 fixed experiences'),('holdout99','all','rmse','Holdout RMSE:99 isolated actions'),
        ('holdout99','lowJ_decile','bias_Q_minus_r','Holdout lowJ10:bias Q-r'),('holdout99','all','spearman','Holdout rank correlation'))
    for ax,(split,subset,key,title) in zip(axes.ravel(),specifications):
        for rate in RATES:
            values=[]
            for seed in SEEDS:
                rows=[r for r in history if float(r['critic_lr'])==rate and int(r['seed'])==seed and r['head']=='minQ' and r['split']==split and r['subset']==subset]
                x=[int(r['update']) for r in rows];y=np.array([float(r[key]) if r[key] else np.nan for r in rows]);values.append(y);ax.plot(x,y,color=COLORS[rate],alpha=.25,lw=.7)
            a=np.array(values);ax.plot(x,a.mean(0),color=COLORS[rate],label=offline_name(rate));ax.fill_between(x,a.mean(0)-a.std(0,ddof=1),a.mean(0)+a.std(0,ddof=1),color=COLORS[rate],alpha=.10)
        ax.set(title=title,xlabel='Offline Critic updates (not SAC candidates)',ylabel='minQ:seed mean ± sample SD');ax.legend(fontsize=7)
        if key=='rmse':ax.set_yscale('log')
        if key=='spearman':ax.set_ylim(-1.05,1.05)
    save(fig,'offline_learning_curves','Fixed4080 updates, same batches/initial weights per seed; no holdout early stopping; no native or Actor learning')

    pred=read_record(OUT/'offline_holdout_predictions.json');fig,axes=plt.subplots(3,3,figsize=(14,13),sharex=True,sharey=True)
    values=[r['reward'] for r in pred]+[r['Q1'] for r in pred]+[r['Q2'] for r in pred];low,high=min(values)-.01,max(values)+.01
    for i,rate in enumerate(RATES):
        for seed in SEEDS:
            rows=[r for r in pred if r['critic_lr']==rate and r['seed']==seed];ax=axes[i,seed]
            ax.scatter([r['reward'] for r in rows],[r['Q1'] for r in rows],s=12,c='#0072B2',alpha=.5,label='Q1')
            ax.scatter([r['reward'] for r in rows],[r['Q2'] for r in rows],s=12,c='#D55E00',alpha=.5,label='Q2')
            ax.plot([low,high],[low,high],c='black',ls=':',lw=1);ax.set(title='Offline '+offline_name(rate)+f',seed{seed}',xlabel='True r=-J (holdout99)',ylabel='Critic-only final prediction',xlim=(low,high),ylim=(low,high));ax.legend(fontsize=7)
    save(fig,'offline_holdout_calibration','All9 offline Critic-only fits on the same99 holdout actions; no Actor attached and no global generalization claim')

    grad=csv_rows('online_gradient_comparison.csv')+csv_rows('offline_gradient_comparison.csv');fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
    selected=[r for r in grad if float(r['step'])==.002]
    bound=[float(r[k]) for r in selected for k in ('true_centered_reward_gradient','analytic_minQ_gradient')];low,high=min(bound)-.2,max(bound)+.2
    for i,kind in enumerate(('online','offline')):
        for j,rate in enumerate(RATES):
            ax=axes[i,j]
            for name,col in zip(NAMES,('#0072B2','#D55E00','#009E73')):
                rows=[r for r in selected if r['kind']==kind and float(r['critic_lr'])==rate and r['variable']==name]
                ax.scatter([float(r['true_centered_reward_gradient']) for r in rows],[float(r['analytic_minQ_gradient']) for r in rows],s=26,color=col,label=name)
            ax.plot([low,high],[low,high],color='black',ls=':');ax.axhline(0,c='.8',lw=.6);ax.axvline(0,c='.8',lw=.6)
            ax.set(title=kind+':'+(offline_name(rate) if kind=='offline' else rate_name(rate)),xlabel='Legacy true reward centered difference (h=.002)',ylabel='Current minQ analytic action gradient',xlim=(low,high),ylim=(low,high));ax.legend(fontsize=7)
    save(fig,'legacy_anchor_gradients','Only3 pre-existing seed2 anchors; all3 model seeds; analytical gradient versus existing finite differences, not new visited-point measurements')

    online=csv_rows('comparison_per_seed.csv');qmetrics=csv_rows('critic_metrics.csv');fig,axes=plt.subplots(2,2,figsize=(13,9))
    for j,rate in enumerate(RATES):
        ys=[np.array([float(r['E_total']) for r in online if float(r['critic_lr'])==rate and r['role']=='final' and r['status']=='complete']),
            np.array([float(r['rmse']) for r in qmetrics if float(r['critic_lr'])==rate and r['subset']=='all99' and r['head']=='minQ']),
            np.array([float(r['rmse']) for r in final if float(r['critic_lr'])==rate and r['split']=='holdout99' and r['subset']=='all' and r['head']=='minQ']),
            np.array([float(r['spearman']) if r['spearman'] else np.nan for r in final if float(r['critic_lr'])==rate and r['split']=='holdout99' and r['subset']=='all' and r['head']=='minQ'])]
        for ax,y in zip(axes.ravel(),ys):
            if len(y)==3:
                ax.bar(j,y.mean(),color=COLORS[rate],alpha=.65);ax.errorbar(j,y.mean(),yerr=y.std(ddof=1),fmt='k_',capsize=4);ax.scatter(j+np.array([-.12,0,.12]),y,c='black',s=20,zorder=5)
            else:ax.text(j,.1,'failed/missing',ha='center')
    titles=('Online SAC Final E_total','Online Final minQ RMSE (fixed99)','Offline minQ RMSE (holdout99)','Offline minQ Spearman (holdout99)')
    for ax,title in zip(axes.ravel(),titles):ax.set(title=title,xticks=range(3),xticklabels=['3e-4 baseline','1e-4','1e-3']);ax.axhline(0,c='.7',lw=.6)
    axes[0,0].axhline(.1,c='.5',ls=':');axes[1,1].set_ylim(-1.05,1.05)
    save(fig,'parallel_online_offline_outcomes','Different evaluation axes remain separate; all seeds and sample SD, no combined score or offline-to-Actor transfer')


if __name__=='__main__':render()
