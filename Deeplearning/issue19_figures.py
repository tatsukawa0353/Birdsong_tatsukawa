"""Eight registered summary figures; no per-run image explosion."""
from .result_paths import resolve as resolve_result_path
import csv
import gzip
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .acoustic_views import display_psd, relative_db, rms_envelope
from .issue19_data import ROOT, OUT, METHODS, WEIGHTS, curve_for, target_data
from .issue16 import digest
from .issue15_analysis import write_json

COLORS = dict(TD3='#0072B2', SAC='#D55E00', DE='#009E73', random_search='#CC79A7')
LABELS = dict(TD3='TD3', SAC='SAC', DE='DE', random_search='Random')


def render_figures(rows, deltas):
    folder = OUT/'figures'; folder.mkdir(exist_ok=True); images = []
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                         'path.simplify':False, 'figure.constrained_layout.use':True})
    def save(fig, name, **meta):
        path = folder/f'{name}.png'; fig.savefig(path, dpi=120); plt.close(fig)
        images.append(dict(path=str(path.relative_to(OUT)), sha256=digest(path), **meta))
    curves = {(m,s,w):curve_for(m,s,w) for m in METHODS for s in (0,1,2) for w in WEIGHTS}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for col, weight in enumerate(WEIGHTS):
        for row, field in enumerate(('acoustic_error', 'objective_selected_common_error')):
            ax = axes[row,col]
            for method in METHODS:
                points = [curves[method,s,weight] for s in (0,1,2)]
                x = np.array([p['episodes'] for p in points[0]])
                y = np.array([[p[field] for p in series] for series in points])
                mean, sd = y.mean(0), y.std(0, ddof=1)
                ax.plot(x, mean, color=COLORS[method], lw=1.5, label=LABELS[method])
                ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.12)
            ax.set_title(f'λ = {weight:g}; '+('current evaluation' if row==0 else 'min J archive (common error)'))
            ax.set(xlim=(0,4096), ylabel='Common E_total', xlabel='Training candidates (RL: 33 evaluations)')
            ax.legend(ncol=2, fontsize=9)
    save(fig, 'learning', description='All vertices retained; mean ± sample SD; RL 33 independent evaluations, direct 4096 candidates')

    fig, axes = plt.subplots(2, 2, figsize=(14,9), sharey=True)
    for ri, role in enumerate(('final','selected')):
        for ci, weight in enumerate(WEIGHTS):
            ax=axes[ri,ci]
            for mi, method in enumerate(METHODS):
                subset=[r for r in rows if r['role']==role and r['weight']==weight and r['method']==method]
                spec=np.array([r['E_spec'] for r in subset]); env=np.array([r['E_env'] for r in subset]); total=spec+env
                ax.bar(mi, spec.mean(), color=COLORS[method], alpha=.9)
                ax.bar(mi, env.mean(), bottom=spec.mean(), color=COLORS[method], alpha=.35, hatch='//')
                ax.errorbar(mi, total.mean(), yerr=total.std(ddof=1), color='black', capsize=4)
                ax.scatter(mi+np.array([-.13,0,.13]),total,color='black',s=16,zorder=4)
            ax.set(title=f'{role.title()}, λ={weight:g}', xticks=range(4), xticklabels=[LABELS[m] for m in METHODS], ylabel='Common E_spec + E_env')
    save(fig,'errors',description='Solid spectral + hatched envelope; dots individual seeds, error bars sample SD of E_total')

    fig, axes=plt.subplots(1,3,figsize=(16,5))
    for ax,field in zip(axes,('E_total','E_spec','E_env')):
        ax.axhline(0,color='.5',lw=1)
        for mi,method in enumerate(METHODS):
            for offset,role,marker in ((-.13,'final','o'),(.13,'selected','s')):
                values=np.array([r['delta_'+field] for r in deltas if r['method']==method and r['role']==role])
                ax.scatter(mi+offset+np.array([-.04,0,.04]),values,color=COLORS[method],marker=marker,s=35)
                ax.errorbar(mi+offset,values.mean(),yerr=values.std(ddof=1),fmt='k_',capsize=4)
        ax.set(title=f'Paired Δ {field}: λ .5 − 1',xticks=range(4),xticklabels=[LABELS[m] for m in METHODS],ylabel='Negative: lower error')
    save(fig,'lambda_deltas',description='Paired seeds; circles Final, squares Selected; sample SD')

    target,controls=target_data(); spans=np.array([2.5e7,1.5e6,751000.])
    fig,axes=plt.subplots(3,2,figsize=(15,11),sharey='row')
    for ri,(index,name,span) in enumerate(zip((0,2,3),('ε_l [CGS]','p_s [CGS]','f_e,l [CGS; not Hz]'),spans)):
        for ci,role in enumerate(('final','selected')):
            ax=axes[ri,ci]
            ax.axhline(controls[index],color='.4',lw=1,label='Target')
            for mi,method in enumerate(METHODS):
                for offset,w,marker in ((-.16,1.,'o'),(.16,.5,'^')):
                    values=[r[('epsilon_l','ps','f0_l')[ri]] for r in rows if r['role']==role and r['weight']==w and r['method']==method]
                    ax.scatter(mi+offset+np.array([-.04,0,.04]),values,marker=marker,color=COLORS[method],s=45,label=f'λ={w:g}' if mi==0 else None)
            ax.set(title=role.title() if ri==0 else '',ylabel=name,xticks=range(4),xticklabels=[LABELS[m] for m in METHODS])
            if ri==0:ax.legend(fontsize=9)
    save(fig,'controls',description='One combined physical-control figure; Final/Selected, both weights, all seeds; target line is reference only')

    with gzip.open(OUT/'critic_predictions_all.csv.gz','rt') as stream:predictions=list(csv.DictReader(stream))
    fixed=[r for r in predictions if r['source'] not in ('own_training','actor')]
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
    extrema=[float(r[k]) for r in fixed for k in ('actual_reward','Q1','Q2')]
    limits=(min(extrema)-.02,max(.02,max(extrema)+.02))
    for ri,method in enumerate(('TD3','SAC')):
        for wi,weight in enumerate(WEIGHTS):
            for qi,name in enumerate(('Q1','Q2')):
                ax=axes[ri,wi*2+qi];subset=[r for r in fixed if r['method']==method and float(r['weight'])==weight]
                for voiced,color in (('True','#0072B2'),('False','#999999')):
                    part=[r for r in subset if r['sustained_voiced']==voiced]
                    ax.scatter([float(r['actual_reward']) for r in part],[float(r[name]) for r in part],s=5,alpha=.15,color=color,rasterized=True,label='sustained' if voiced=='True' else 'not sustained')
                ax.plot(limits,limits,color='black',lw=1)
                ax.set(title=f'{method} λ={weight:g} {name}',xlabel='Measured r = −Jλ',ylabel='Predicted Q',xlim=limits,ylim=limits)
                if ri==0 and wi==0 and qi==0:ax.legend(fontsize=9)
    save(fig,'critic_scatter',description='Q1 and Q2 separated; fixed 1005 candidate pool x3 seeds; no synthetic rewards or new sounds')

    with (OUT/'critic_metrics_all.csv').open() as stream:metrics=list(csv.DictReader(stream))
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for ri,method in enumerate(('TD3','SAC')):
        for ci,weight in enumerate(WEIGHTS):
            ax=axes[ri,ci];ax.axhline(0,color='.5',lw=1)
            for qi,name in enumerate(('Q1','Q2','minQ')):
                for offset,subset,color in ((-.1,'low_J_decile','#D55E00'),(.1,'own_training','#0072B2')):
                    values=np.array([float(r['bias_Q_minus_r']) for r in metrics if r['method']==method and float(r['weight'])==weight and r['subset']==subset and r['critic']==name])
                    ax.scatter(qi+offset+np.array([-.02,0,.02]),values,color=color,s=25)
                    ax.errorbar(qi+offset,values.mean(),yerr=values.std(ddof=1),fmt='_',color=color,capsize=4,label=subset if qi==0 else None)
            ax.set(title=f'{method} λ={weight:g}',xticks=range(3),xticklabels=('Q1','Q2','minQ'),ylabel='Bias Q − measured r');ax.legend(fontsize=9)
    save(fig,'critic_bias',description='Low-J fixed-pool decile vs own training points, seed dots and sample SD; negative means pessimism')

    registry=json.loads((OUT/'endpoint_sources.json').read_text());sounds={}
    # Fixed seed 1 is chosen for both weights and both roles, not cherry-picked
    # separately for a successful result. It contains the baseline TD3 failure.
    for entry in registry:
        if entry['seed']==1:
            with np.load(resolve_result_path(ROOT/entry['audio_path'])) as z:sounds[(entry['method'],entry['weight'],entry['role'])]=z['audio'].copy()
    f,t,p,reference=display_psd(target);target_db=relative_db(p,reference);peak=abs(target).max()
    target_env=rms_envelope(target,peak)
    views={key:(relative_db(display_psd(audio)[2],reference),rms_envelope(audio,peak)) for key,audio in sounds.items()}
    all_db_max=max(float(x[0].max()) for x in views.values())
    all_rms_max=max(float(target_env.max()),max(float(x[1].max()) for x in views.values()))
    for entry in registry:
        if entry['seed']==1:continue
        with np.load(resolve_result_path(ROOT/entry['audio_path'])) as z:audio=z['audio'].copy()
        all_db_max=max(all_db_max,float(relative_db(display_psd(audio)[2],reference).max()))
        all_rms_max=max(all_rms_max,float(rms_envelope(audio,peak).max()))
    db_limits=(-70.,max(10.,float(10*np.ceil(all_db_max/10))))
    rms_limits=(0.,1.15*all_rms_max)
    time=np.arange(len(target))/50000
    for role in ('final','selected'):
        fig,axes=plt.subplots(5,4,figsize=(18,17))
        axes[0,2].axis('off');axes[0,3].axis('off')
        panels=[(axes[0,0],axes[0,1],target_db,target_env,'Target (one shared reference)')]
        for ri,method in enumerate(METHODS,1):
            for ci,weight in enumerate(WEIGHTS):
                if method=='random_search' and weight==.5:
                    axes[ri,ci*2].axis('off');axes[ri,ci*2+1].axis('off')
                    axes[ri,ci*2].text(.05,.5,'Random λ=0.5: same candidate / waveform\nas λ=1; shown once at left.',transform=axes[ri,ci*2].transAxes)
                    continue
                db,env=views[method,weight,role]
                label=f'{LABELS[method]} λ={weight:g} seed1 {role}'
                if method=='random_search':label=f'Random λ=1 / 0.5 seed1 {role} (shared)'
                panels.append((axes[ri,ci*2],axes[ri,ci*2+1],db,env,label))
        for spec_ax,rms_ax,db,env,label in panels:
            mesh=spec_ax.pcolormesh(t,f/1000,db,shading='gouraud',cmap='magma',vmin=db_limits[0],vmax=db_limits[1],rasterized=True)
            spec_ax.set(title=label,xlabel='Time [s]',ylabel='Frequency [kHz]',xlim=(t[0],t[-1]),ylim=(0,10))
            rms_ax.plot(time,target_env,color='.6',lw=1,label='Target')
            if label.startswith('Target') is False:rms_ax.plot(time,env,color='#0072B2',lw=1,label='Generated')
            rms_ax.set(xlabel='Time [s]',ylabel='RMS / Target peak',xlim=(0,.2),ylim=rms_limits)
            if spec_ax is axes[1,0]:rms_ax.legend(fontsize=8)
        fig.colorbar(mesh,ax=axes.ravel().tolist(),shrink=.45,pad=.01,label='Power [dB relative to Target maximum PSD]')
        save(fig,f'representative_{role}',description='Fixed seed1 across methods/weights; shared Target PSD and RMS normalization',
             db_limits=db_limits,rms_ylim=rms_limits,target_reference_psd=reference,seed=1,role=role)
    write_json(OUT/'figure_sources.json',dict(images=images,source_sha256=digest(__file__),
        no_training_or_reward_changes=True,no_per_seed_image_grid=True,
        reference='Full-band target maximum display PSD, 10 MHz / Blackman-Harris / 245760 / overlap184320',
        common_display_limits_cover_all_48_endpoints=True,
        source_records=['comparison_per_seed.csv','paired_lambda_deltas.csv','critic_predictions_all.csv.gz','endpoint_sources.json']))
    return images
