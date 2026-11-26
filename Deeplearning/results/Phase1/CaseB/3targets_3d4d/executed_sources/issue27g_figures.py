"""Minimal aggregate scientific figures; common axes/PSD/typography, native0."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .issue27g_data import OUT,CONTROLS,METHODS,SEEDS,target_path,folder
from .issue27g_style import context,save,bar,identify,metadata,COLORS
from .acoustic_views import display_psd,relative_db,rms_envelope
from .issue15_analysis import write_json

def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def read(path):return json.loads(Path(path).read_text())
def panel(ax,z,ref,label,limits=(-40,0)):
    f,t,p,_=display_psd(z['audio']);mesh=ax.pcolormesh(t,f/1000,relative_db(p,ref),shading='auto',cmap='magma',vmin=limits[0],vmax=limits[1],rasterized=True)
    ax.set(xlim=(t[0],t[-1]),ylim=(0,10));ax.set_xticks([.05,.1,.15]);ax.set_yticks([0,5,10]);identify(ax,label);return mesh

def targets():
    data={t:load(target_path(t)) for t in CONTROLS};refs={t:display_psd(z['audio'])[3] for t,z in data.items()};absolute=max(refs.values());peak=max(float(abs(z['audio']).max()) for z in data.values())
    stats={};time=np.arange(10000)/50000
    with context():
        fig,axes=plt.subplots(3,3,figsize=(45,29),sharex='row',sharey='row')
        for col,(t,z) in enumerate(data.items()):
            mesh=panel(axes[0,col],z,absolute,t)
            axes[0,col].set_ylabel('Frequency [kHz]' if col==0 else '')
            axes[1,col].plot(time,z['audio']/peak,color=COLORS['Target']);axes[1,col].set(xlim=(0,.2),ylabel='Pressure / shared peak' if col==0 else '');identify(axes[1,col],t)
            f,tt,p,ref=display_psd(z['audio']);axes[2,col].plot(f/1000,relative_db(p.mean(1),absolute),color=COLORS['Target']);axes[2,col].set(xlim=(0,10),ylim=(-80,5),xlabel='Frequency [kHz]',ylabel='Mean PSD [dB]' if col==0 else '');identify(axes[2,col],t)
            axes[1,col].set_xlabel('Time [s]');db=relative_db(p,absolute);stats[t]=dict(above=float(np.mean(db>0)),below=float(np.mean(db< -40)))
        bar(fig,mesh,axes[0]);save(fig,'targets','Three prespecified Targets side by side; absolute intensity uses one common PSD reference/gain')
        fig,axes=plt.subplots(6,3,figsize=(45,48),sharex=True,sharey='row');time=np.arange(1,10001)/50000;mask=time>=.18
        for col,(t,z) in enumerate(data.items()):
            v=z['diagnostic_samples']
            for row,label in enumerate((r'$x_l$',r'$y_l$',r'$x_r$',r'$y_r$')):
                ax=axes[row,col];ax.plot(time[mask],v[mask,row],color=COLORS['Target']);ax.set_ylabel(label if col==0 else '');identify(ax,t)
            for row,terms in ((4,((4,'Left source'),(5,'Right source'))),(5,((8,'Left output'),(9,'Right output'),(6,'Feedback')))):
                ax=axes[row,col]
                for j,label in terms:ax.plot(time[mask],v[mask,j],label=label)
                ax.legend(loc='upper right');ax.set_ylabel('Acoustic terms' if col==0 else '');identify(ax,t)
            axes[-1,col].set_xlabel('Time [s]')
        for ax in axes.ravel():ax.set_xlim(.18,.2);ax.set_xticks([.18,.19,.2])
        for row in range(6):
            low,high=axes[row,0].get_ylim();axes[row,0].set_ylim(low,high+(high-low)*(.6 if row>=4 else .2))
        save(fig,'target_internal','Representative physical interpretation: all3 Target late20ms motion and coupled source/output terms')
    write_json(OUT/'display_settings.json',dict(styles=['science','ieee','no-latex'],inherited_from='issue17_comparison_figures.py',font_family=['serif'],font_serif=['STIXGeneral'],mathtext_fontset='cm',text_usetex=False,
        label_pt=46,tick_pt=30,legend_pt=29,line_width_pt=4.,dpi=120,base_figure_inches=[15,10],aggregate_dimension_reason='Expand physical size to preserve required46/30/29pt; never shrink type',
        reference_per_target=refs,absolute_target_reference=absolute,absolute_target_gain=1/peak,Target_comparison_color_limits_db=[-40,0],Target_clipping_fractions=stats,
        stft=dict(window='blackmanharris',window_s=.024576,overlap=.75,nperseg=245760,noverlap=184320,display_resampling_rate=10000000,complete_windows_only=True,frequency_limits_Hz=[0,10000]),
        reward_stft_unchanged=True,individual_normalization=False,phase_alignment=False))
    metadata('target')

def learning():
    from .issue27g_analysis import summaries
    from .issue17_artifacts import read_record
    from .issue27g_data import CONTROLS
    settings=read(OUT/'display_settings.json');refs=settings['reference_per_target'];target={t:load(target_path(t)) for t in CONTROLS};runs=summaries()
    assert len(runs)==54 and all(s['status']=='complete' for s in runs)
    data={(s['target'],s['dim'],s['method'],s['seed'],r):load(folder(s['target'],s['dim'],s['method'],s['seed'])/f'{r}.npz') for s in runs for r in ('final','selected')}
    curves={(s['target'],s['dim'],s['method'],s['seed']):read_record(folder(s['target'],s['dim'],s['method'],s['seed'])/'curve.json') for s in runs}
    saturation={};maximum=0.
    for key,z in data.items():
        db=relative_db(display_psd(z['audio'])[2],refs[key[0]]);saturation['_'.join(map(str,key))]=dict(above=float(np.mean(db>0)),below=float(np.mean(db< -40)));maximum=max(maximum,float(db.max()))
    with context():
        for t in CONTROLS:
            for role in ('final','selected'):
                fig,axes=plt.subplots(4,6,figsize=(84,36),sharex=True,sharey=True)
                for col,(dim,seed) in enumerate((d,s) for d in (3,4) for s in SEEDS):
                    mesh=panel(axes[0,col],target[t],refs[t],f'{t} / {dim}D / s{seed} / Target')
                    for row,m in enumerate(METHODS,1):mesh=panel(axes[row,col],data[t,dim,m,seed,role],refs[t],f'{m} / {dim}D / s{seed}')
                    axes[-1,col].set_xlabel('Time [s]')
                for row in range(4):axes[row,0].set_ylabel('Frequency [kHz]')
                bar(fig,mesh,axes);save(fig,f'{t}_{role}_spectrograms','Rows Target/SAC/TD3/PPO; columns3D seeds0–2,4D seeds0–2; same Target PSD reference and[-40,0]dB')
        wide=[-80.,max(0.,float(np.ceil(maximum/5)*5))]
        fig,axes=plt.subplots(3,7,figsize=(91,27),sharex=True,sharey=True)
        for row,t in enumerate(CONTROLS):
            mesh=panel(axes[row,0],target[t],refs[t],t+' / Target',wide)
            for col,(d,m) in enumerate(((d,m) for d in (3,4) for m in METHODS),1):
                mesh=panel(axes[row,col],data[t,d,m,1,'final'],refs[t],f'{t} / {d}D / {m} / s1',wide)
            axes[row,0].set_ylabel('Frequency [kHz]')
        for ax in axes[-1]:ax.set_xlabel('Time [s]')
        bar(fig,mesh,axes);save(fig,'wide_psd_reference','Single supplementary wide-range image, representative seed1 Final; same wide color scale across panels')

        fig,axes=plt.subplots(6,3,figsize=(45,54),sharex=True,sharey='col')
        for col,t in enumerate(CONTROLS):
            ymax=max(r['acoustic_error'] for (tt,d,m,s),c in curves.items() if tt==t for r in c)*1.5
            for row,(d,seed) in enumerate((d,s) for d in (3,4) for s in SEEDS):
                ax=axes[row,col]
                for m in METHODS:
                    c=curves[t,d,m,seed];ax.plot([r['episodes'] for r in c],[r['acoustic_error'] for r in c],color=COLORS[m],label=m)
                ax.set(xlim=(0,4096),ylim=(0,ymax));ax.set_xticks([0,2048,4096]);identify(ax,f'{t} / {d}D / s{seed}');ax.legend(loc='upper right')
                if col==0:ax.set_ylabel(r'$E_{total}$')
                if row==5:ax.set_xlabel('Training candidates')
        save(fig,'learning_curves_all_seeds','Every one of33 independent evaluation points of all54 runs; no smoothing, thinning or omitted vertices')

        fig,axes=plt.subplots(8,3,figsize=(45,72),sharex=True)
        for col,t in enumerate(CONTROLS):
            for metric_i,key in enumerate(('log_spectral','envelope','total','J')):
                ymax=max((r['weighted_loss'] if key=='J' else r['acoustic_components'][key]) for (tt,d,m,s),c in curves.items() if tt==t for r in c)*1.5
                for di,d in enumerate((3,4)):
                    ax=axes[metric_i*2+di,col]
                    for m in METHODS:
                        c=[curves[t,d,m,s] for s in SEEDS];x=np.array([r['episodes'] for r in c[0]])
                        y=np.array([[r['weighted_loss'] if key=='J' else r['acoustic_components'][key] for r in z] for z in c]);mean=y.mean(0);sd=y.std(0,ddof=1)
                        ax.plot(x,mean,color=COLORS[m],label=m);ax.fill_between(x,mean-sd,mean+sd,color=COLORS[m],alpha=.12)
                    ax.set(xlim=(0,4096),ylim=(0,ymax));ax.set_xticks([0,2048,4096]);identify(ax,f'{t} / {d}D');ax.legend(loc='upper right')
                    if col==0:ax.set_ylabel((r'$E_{spec}$',r'$E_{env}$',r'$E_{total}$',r'$J$')[metric_i])
                    if metric_i==3 and di==1:ax.set_xlabel('Training candidates')
        save(fig,'learning_curves_mean_sd','All33 points,3seed means and pale sample SD bands; four acoustic metrics, paired dimension axes')

        fig,axes=plt.subplots(4,3,figsize=(45,36),sharex=True,sharey='row')
        for col,t in enumerate(CONTROLS):
            for row,index in enumerate((0,1,2,3)):
                ax=axes[row,col];ax.scatter([-1],[CONTROLS[t][index]],s=180,marker='*',color=COLORS['Target'],label='Target')
                for di,d in enumerate((3,4)):
                    for mi,m in enumerate(METHODS):
                        x=di*4+mi;v=np.array([data[t,d,m,s,'final']['physical_controls'][0,index] for s in SEEDS])
                        for seed,offset in zip(SEEDS,(-.18,0,.18)):ax.scatter([x+offset],[v[seed]],marker=('o','s','^')[seed],s=110,color=COLORS[m],label=f'seed{seed}' if (di,mi)==(0,0) else None)
                        ax.plot([x+.35,x+.35],[v.mean()-v.std(ddof=1),v.mean()+v.std(ddof=1)],color=COLORS[m]);ax.scatter([x+.35],[v.mean()],s=110,marker='D',color=COLORS[m])
                ax.set_xticks([-1,0,1,2,4,5,6],['T','S','T','P','S','T','P']);identify(ax,t+' / 3D | 4D')
                if col==0:ax.set_ylabel((r'$\varepsilon_l$',r'$\varepsilon_r$',r'$p_s$',r'$f_{0,l}$')[row])
                if row==0:ax.legend(loc='upper right')
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False);ax.yaxis.get_offset_text().set_fontsize(30)
        for row in range(4):
            lo,hi=axes[row,0].get_ylim();span=hi-lo;axes[row,0].set_ylim(lo-.02*span,hi+.45*span)
        save(fig,'final_controls','Final physical controls, every seed and mean/sample SD;3D right epsilon is fixed, not learned; S/T/P=SAC/TD3/PPO')

        draw_waveform(data,target,settings,refs,wide)
    settings.update(comparison_color_limits_db=[-40,0],wide_color_limits_db=wide,comparison_clipping_fractions=saturation,representative_seed=1,all_curve_points=33,
        plot_selection='Final/Selected only; training_best/posthoc CSV/WAV/raw retained without redundant images')
    write_json(OUT/'display_settings.json',settings);metadata('learning')

def draw_waveform(data,target,settings,refs,wide):
    fig,axes=plt.subplots(3,6,figsize=(84,29),sharey='row');time=np.arange(10000)/50000;mask=time>=.195;global_peak=1/settings['absolute_target_gain']
    for col,(t,d) in enumerate((t,d) for t in CONTROLS for d in (3,4)):
        for label,z in [('Target',target[t])]+[(m,data[t,d,m,1,'final']) for m in METHODS]:
            color=COLORS[label];axes[0,col].plot(time,rms_envelope(z['audio'],global_peak),color=color,label=label)
            axes[1,col].plot(time[mask],z['audio'][mask]/global_peak,color=color,label=label)
            f,tt,p,_=display_psd(z['audio']);axes[2,col].plot(f/1000,relative_db(p.mean(1),refs[t]),color=color,label=label)
        axes[0,col].set(xlim=(0,.2));axes[1,col].set(xlim=(.195,.2));axes[2,col].set(xlim=(0,10),ylim=(-80,max(10,wide[1])))
        for row in range(3):identify(axes[row,col],f'{t} / {d}D / s1')
        axes[0,col].set_xticks([0,.1,.2]);axes[1,col].set_xticks([.195,.2]);axes[2,col].set_xticks([0,5,10]);axes[2,col].set_xlabel('Frequency [kHz]');axes[0,col].set_xlabel('Time [s]');axes[1,col].set_xlabel('Time [s]')
        axes[2,col].legend(loc='upper right')
    for row,label in enumerate(('RMS / shared peak','Pressure / shared peak','Mean PSD [dB]')):axes[row,0].set_ylabel(label)
    save(fig,'final_waveform_mean_psd','Preregistered seed1, full5ms RMS, late5ms raw wave without alignment, and time-mean PSD; common references and axes')

def waveform_only():
    settings=read(OUT/'display_settings.json');refs=settings['reference_per_target'];wide=settings['wide_color_limits_db']
    target={t:load(target_path(t)) for t in CONTROLS}
    data={(t,d,m,s,'final'):load(folder(t,d,m,s)/'final.npz') for t in CONTROLS for d in (3,4) for m in METHODS for s in SEEDS}
    from .issue27g_style import ITEMS
    with context():draw_waveform(data,target,settings,refs,wide)
    previous=read(OUT/'learning_figures.json');new=ITEMS[-1]
    write_json(OUT/'learning_figures.json',[item for item in previous if item['path']!=new['path']]+[new])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('targets','learning','waveform'));a=p.parse_args();targets() if a.stage=='targets' else learning() if a.stage=='learning' else waveform_only()
