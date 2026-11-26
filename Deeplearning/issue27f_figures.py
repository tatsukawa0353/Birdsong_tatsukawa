"""Common physical PSD references and Target-first method comparisons; native0."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .issue27f_data import OUT,OLD,GRID_E,GRID_P,METHODS,SEEDS,TARGET,folder,target_data
from .issue27f_analysis import read,load,records,summaries,ROLES
from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue15_analysis import write_json
from .issue16 import digest

COLORS=dict(SAC='#0072B2',TD3='#D55E00',PPO='#009E73')
ITEMS=[]
def save(fig,name,description):
    path=OUT/f'figures/{name}.png';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig)
    ITEMS.append(dict(path=str(path.relative_to(OUT)),sha256=digest(path),description=description))
def psd_panel(ax,z,reference,limits,label):
    f,t,p,_=display_psd(z['audio'])
    mesh=ax.pcolormesh(t,f/1000,relative_db(p,reference),shading='auto',cmap='magma',vmin=limits[0],vmax=limits[1],rasterized=True)
    ax.set(title=label,xlim=(t[0],t[-1]),ylim=(0,10),xlabel='Time [s]',ylabel='Frequency [kHz]');return mesh
def wave_panel(ax,z,scale,label):
    t=np.arange(10000)/50000;ax.plot(t,z['audio']/scale,lw=.55,color='.25',label='Waveform')
    ax.plot(t,rms_envelope(z['audio'],scale),lw=1,color='#D55E00',label='5ms RMS');ax.set(title=label,xlim=(0,.2),xlabel='Time [s]',ylabel='Amplitude / Target peak')

def physics():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
    data={r['candidate']:load(OUT/r['raw_npz']) for r in records()};old={r['candidate']:load(OLD/r['raw_npz']) for r in read(OLD/'preflight/candidates.json')}
    target=load(TARGET);scale=float(abs(target['audio']).max());reference=display_psd(target['audio'])[3];previous=read(OLD/'display_settings.json')
    # Store all display matrices, making the common numerical normalization inspectable.
    matrices={};maxdb=-np.inf
    for name,z in {**data,**{'old_'+k:v for k,v in old.items()}}.items():
        f,t,p,_=display_psd(z['audio']);matrices[name]=p;maxdb=max(maxdb,float(relative_db(p,previous['PSD_reference']).max()))
    np.savez_compressed(OUT/'display_psd.npz',frequency_Hz=f,time_s=t,**matrices)
    joint_limits=[-80.,max(0.,float(np.ceil(maxdb/5)*5))]
    newmax=max(float(relative_db(matrices[k],reference).max()) for k in data);limits=[-80.,max(0.,float(np.ceil(newmax/5)*5))]
    fig,ax=plt.subplots(figsize=(10,5));mesh=psd_panel(ax,target,reference,limits,'Frozen Case B Target: b3_c00');fig.colorbar(mesh,ax=ax,label='dB re Target full-band PSD max');save(fig,'Target_spectrogram','Standalone Target; same shared18-candidate PSD reference')
    fig,ax=plt.subplots(figsize=(11,4));wave_panel(ax,target,scale,'Frozen Target');ax.legend();save(fig,'Target_waveform','Standalone raw Target and loss-compatible5ms RMS')
    for group in (2,3):
        for kind in ('spectrograms','waveforms'):
            fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,sharey=True)
            for r in [r for r in records() if r['group']==group]:
                ax=axes[r['row'],r['column']];z=data[r['candidate']];label=f"{r['candidate']}, epsilon={r['epsilon_l']/1e7:g}e7, ps={r['ps']/1e6:g}e6\nAC/own peak={r['late_ac_ratio']:.4f}"
                if kind=='spectrograms':mesh=psd_panel(ax,z,reference,limits,label)
                else:wave_panel(ax,z,scale,label)
            if kind=='spectrograms':fig.colorbar(mesh,ax=axes.ravel().tolist(),label='dB re shared Target PSD max')
            save(fig,f'group{group}_{kind}','All9 within group; references/gain/color shared across both new groups')
        for tail in (False,True):
            lo=.18 if tail else 0.;tag='tail' if tail else 'full';t=np.arange(1,10001)/50000;mask=t>=lo
            fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,sharey='row')
            for j,ch in enumerate(('x_l','y_l','x_r','y_r')):
                for col,p in enumerate(GRID_P):
                    ax=axes[j,col]
                    for row,e in enumerate(GRID_E):ax.plot(t[mask],data[f'b{group}_c{row}{col}']['diagnostic_samples'][mask,j],lw=.7,label=f'epsilon={e/1e7:g}e7')
                    ax.set(title=f'{ch}, ps={p/1e6:g}e6',xlim=(lo,.2),xlabel='Time [s]',ylabel=ch+' [model units]');ax.legend(fontsize=7)
            save(fig,f'group{group}_motion_{tag}','Left/right displacement and velocity; sampled at50kHz, no clamping/filtering')
            fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True,sharey=True)
            for r in [r for r in records() if r['group']==group]:
                ax=axes[r['row'],r['column']];v=data[r['candidate']]['diagnostic_samples']
                for j,label in ((4,'Left source'),(5,'Right source'),(6,'Feedback'),(8,'Left output'),(9,'Right output')):ax.plot(t[mask],v[mask,j],lw=.7,label=label)
                ax.set(title=r['candidate'],xlim=(lo,.2),xlabel='Time [s]',ylabel='Native acoustic terms');ax.legend(fontsize=6)
            save(fig,f'group{group}_sources_{tag}','Coupled-state source/output terms and feedback; not isolated-source sounds')
    for name,clim in (('previous_color',previous['color_limits_db']),('joint_color',joint_limits)):
        fig,axes=plt.subplots(9,3,figsize=(15,25),sharex=True,sharey=True)
        for i in range(3):
            for j in range(3):
                row=i*3+j
                for col,key in enumerate((f'c{i}{j}',f'b2_c{i}{j}',f'b3_c{i}{j}')):
                    z=old[key] if col==0 else data[key];mesh=psd_panel(axes[row,col],z,previous['PSD_reference'],clim,f'Group {col+1}: {key} (e={GRID_E[i]/1e7:g}, p={GRID_P[j]/1e6:g})')
        fig.colorbar(mesh,ax=axes.ravel().tolist(),label='dB re previous c00 full-band PSD reference')
        save(fig,'three_groups_'+name,'27 saved sounds; same reference, time/frequency axes and color; old9 read only')
    fig,axes=plt.subplots(4,3,figsize=(15,11),sharex=True,sharey='row');t=np.arange(1,10001)/50000;mask=t>=.18
    for col,(label,z) in enumerate((('1: left300000/right1e7',old['c00']),('2: left49000/right1e7',data['b2_c00']),('3: both49000',target))):
        v=z['diagnostic_samples'];axes[0,col].plot(t[mask],v[mask,0],lw=.7,label='left');axes[0,col].plot(t[mask],v[mask,2],lw=.7,label='right')
        axes[1,col].plot(t[mask],v[mask,1],lw=.7,label='left');axes[1,col].plot(t[mask],v[mask,3],lw=.7,label='right')
        for side,index in (('left',0),('right',2)):
            nl=2e8*v[mask,index]**2;axes[2,col].plot(t[mask],nl,lw=.7,label=side)
            axes[3,col].plot(t[mask],2e4+nl-.01*(z['controls'][2]-v[mask,7]),lw=.7,label=side)
        for row,ylabel in enumerate(('Displacement','Velocity','c*x^2','b+c*x^2-D*(ps-pi_tilde)')):
            axes[row,col].set(title=label if row==0 else '',xlabel='Time [s]',ylabel=ylabel,xlim=(.18,.2));axes[row,col].legend(fontsize=7)
        axes[2,col].set_yscale('log')
        axes[3,col].set_yscale('symlog',linthresh=10000)
        axes[3,col].axhline(0,color='.5',ls='--',lw=.7)
        axes[2,col].set_yscale('log')
        axes[3,col].set_yscale('symlog',linthresh=1e4)
        axes[3,col].axhline(0,color='.4',lw=.5,ls='--')
    save(fig,'force_damping_comparison','Matched epsilon30M/ps3M; late motion and instantaneous velocity coefficients, not a full stability proof')
    settings=dict(PSD_reference=reference,candidate_color_limits_db=limits,previous_PSD_reference=previous['PSD_reference'],previous_color_limits_db=previous['color_limits_db'],joint_color_limits_db=joint_limits,
        previous_color_upper_clipped_fraction={k:float(np.mean(relative_db(v,previous['PSD_reference'])>previous['color_limits_db'][1])) for k,v in matrices.items()},
        frequency_limits_Hz=[0,10000],window='blackmanharris',nperseg=245760,noverlap=184320,display_resampling_rate=10000000,complete_windows_only=True,
        gain=1/scale,no_individual_normalization=True,phase_alignment=False,sound_generation_count=0)
    write_json(OUT/'display_settings.json',settings);write_json(OUT/'physics_figures.json',ITEMS)

def learning():
    plt.rcParams.update({'font.size':10,'figure.constrained_layout.use':True});target=load(TARGET);scale=float(abs(target['audio']).max());reference=display_psd(target['audio'])[3]
    settings=read(OUT/'display_settings.json');runs=summaries();data={(s['method'],s['seed'],r):load(folder(s['method'],s['seed'])/f'{r}.npz') for s in runs for r in ROLES}
    maxdb=max(float(relative_db(display_psd(z['audio'])[2],reference).max()) for z in data.values());limits=[-80.,max(0.,float(np.ceil(maxdb/5)*5))]
    fig,axes=plt.subplots(4,3,figsize=(15,13),sharex=True,sharey='row')
    for col,m in enumerate(METHODS):
        for s in [s for s in runs if s['method']==m]:
            from .issue17_artifacts import read_record
            c=read_record(folder(m,s['seed'])/'curve.json');t=[r['episodes'] for r in c]
            for row,key in enumerate(('log_spectral','envelope','total','J')):
                y=[r['weighted_loss'] if key=='J' else r['acoustic_components'][key] for r in c];axes[row,col].plot(t,y,marker='.',lw=1,label=f"seed{s['seed']}")
                axes[row,col].set(title=m if row==0 else '',xlabel='Training candidates',ylabel=('E_spec','E_env','E_total','J')[row],xlim=(0,4096));axes[row,col].legend(fontsize=8)
    save(fig,'learning_curves','All9 runs;33 deterministic evaluations after corresponding updates')
    for role in ROLES:
        # All9 seeds are shown, each column starts with the identical Target.
        fig,axes=plt.subplots(4,3,figsize=(15,13),sharex=True,sharey=True)
        for seed in SEEDS:
            mesh=psd_panel(axes[0,seed],target,reference,limits,f'Target (column seed{seed})')
            for row,m in enumerate(METHODS,1):mesh=psd_panel(axes[row,seed],data[m,seed,role],reference,limits,f'{m}, seed{seed}, {role}')
        fig.colorbar(mesh,ax=axes.ravel().tolist(),label='dB re frozen Target full-band PSD max');save(fig,f'{role}_spectrograms','Target top then SAC/TD3/PPO; all seeds, same axes, reference and color across all four roles')
        fig,axes=plt.subplots(4,3,figsize=(15,11),sharex=True,sharey=True)
        for seed in SEEDS:
            wave_panel(axes[0,seed],target,scale,f'Target (column seed{seed})')
            for row,m in enumerate(METHODS,1):
                z=data[m,seed,role];wave_panel(axes[row,seed],z,scale,f'{m}, seed{seed}, {role}')
                axes[row,seed].plot(np.arange(10000)/50000,rms_envelope(target['audio'],scale),color='#0072B2',lw=.8,label='Target RMS')
        axes[1,0].legend(fontsize=7);save(fig,f'{role}_waveforms','All raw waveforms and generated/Target RMS; common Target peak gain, no phase alignment')
    fig,axes=plt.subplots(3,4,figsize=(16,11),sharey='row')
    controls=target_data()[1]
    for col,role in enumerate(ROLES):
        for row,index in enumerate((0,2,3)):
            ax=axes[row,col]
            for mi,m in enumerate(METHODS):
                v=np.array([data[m,s,role]['physical_controls'][0,index] for s in SEEDS])
                for seed,offset in zip(SEEDS,(-.15,0,.15)):ax.scatter(mi+offset,v[seed],color=COLORS[m],marker=('o','s','^')[seed],s=28)
                ax.errorbar(mi+.25,v.mean(),yerr=v.std(ddof=1),fmt='D',color='black',capsize=3,ms=4)
            ax.axhline(controls[index],color='.4',ls='--');ax.set_xticks(range(3),METHODS);ax.set(title=role if row==0 else '',ylabel=('epsilon_l','ps','f0_l (force, not Hz)')[row]);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    save(fig,'control_parameters','All endpoint physical controls, per-seed points and mean/sample SD; dashed Target')
    for role in ('final','selected'):
        fig,axes=plt.subplots(6,4,figsize=(16,15),sharex=True,sharey='row');t=np.arange(1,10001)/50000;mask=t>=.18
        for col,(label,z) in enumerate([('Target',target)]+[(m,load(OUT/f'audit/{m}_s1_{role}.npz')) for m in METHODS]):
            v=z['diagnostic_samples']
            for row,ch in enumerate(('x_l','y_l','x_r','y_r')):axes[row,col].plot(t[mask],v[mask,row],lw=.7);axes[row,col].set(ylabel=ch)
            for row,indices in ((4,((4,'Left source'),(5,'Right source'))),(5,((8,'Left output'),(9,'Right output'),(6,'Feedback')))):
                for j,ch in indices:axes[row,col].plot(t[mask],v[mask,j],lw=.7,label=ch)
                axes[row,col].legend(fontsize=7)
            axes[0,col].set_title(label+' / seed1 '+role)
            for ax in axes[:,col]:ax.set(xlim=(.18,.2),xlabel='Time [s]')
        save(fig,f'{role}_internal_seed1','Preregistered representative seed1, no best-seed selection; all seeds in endpoint CSV/NPZ')
    settings.update(RL_color_limits_db=limits,RL_reference=reference,representative_seed=1);write_json(OUT/'display_settings.json',settings);write_json(OUT/'learning_figures.json',ITEMS)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('physics','learning'));a=p.parse_args();physics() if a.stage=='physics' else learning()
