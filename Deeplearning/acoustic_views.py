"""Display-only shared-target PSD, exact loss-compatible RMS, constant controls."""
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
from .acoustics import resample
from .plotting import TARGET_FS,NPERSEG,NOVERLAP,WINDOW_TYPE
from .issue17_comparison_figures import COLORS,METHODS,LABELS
from .issue16 import digest


def display_psd(wave,rate=50000):
    x=resample(wave,rate,TARGET_FS)
    f,t,p=signal.spectrogram(x,fs=TARGET_FS,window=WINDOW_TYPE,nperseg=NPERSEG,noverlap=NOVERLAP,
        detrend='constant',scaling='density',mode='psd')
    reference=float(p.max());keep=f<=10000
    return f[keep],t,p[keep],reference


def relative_db(psd,target_reference):
    if not np.isfinite(target_reference) or target_reference<=0:raise ValueError('Positive target PSD reference required')
    return 10*np.log10(np.maximum(np.asarray(psd)/target_reference,1e-12))


def rms_envelope(wave,target_peak,rate=50000):
    x=np.asarray(wave,dtype=np.float64)/target_peak
    width=min(len(x),max(1,round(.005*rate)))
    return np.sqrt(np.maximum(np.convolve(x*x,np.ones(width)/width,mode='same'),0))


def spectrogram_pair(target_psd,generated_psd,path,*,limits):
    f,t,a=target_psd;_,_,b=generated_psd
    with plt.style.context(['science','ieee','no-latex']):
        fig,axes=plt.subplots(2,1,figsize=(14,11),sharex=True,sharey=True,layout='constrained')
        for ax,z,label in zip(axes,(a,b),('Target','Generated')):
            mesh=ax.pcolormesh(t,f/1000,z,shading='gouraud',cmap='magma',vmin=limits[0],vmax=limits[1],rasterized=True)
            ax.set(xlim=(t[0],t[-1]),ylim=(0,10));ax.set_ylabel('Frequency [kHz]',fontsize=32)
            ax.tick_params(labelsize=24,top=False,right=False)
            ax.text(.02,.94,label,transform=ax.transAxes,ha='left',va='top',fontsize=20,
                bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
        axes[-1].set_xlabel('Time [s]',fontsize=36)
        bar=fig.colorbar(mesh,ax=axes,shrink=.85,pad=.02)
        bar.set_label('Relative power [dB]',fontsize=28);bar.ax.tick_params(labelsize=22)
        fig.savefig(path,dpi=100);plt.close(fig)
    return dict(path=str(path),image_sha256=digest(path),cmap='magma',db_limits=list(limits),
        reference='Target maximum full-band display PSD',shared_colorbar=True,title='')


def envelope_pair(target,generated,path,*,limits):
    time=np.arange(len(target))/50000
    with plt.style.context(['science','ieee','no-latex']),plt.rc_context({'path.simplify':False}):
        fig,ax=plt.subplots(figsize=(14,8),layout='constrained')
        ax.plot(time,target,lw=3,ls='-',color='#222222',label='Target')
        ax.plot(time,generated,lw=3,ls='-',color='#0072B2',label='Generated')
        ax.set(xlim=(0,.2),ylim=limits);ax.set_xlabel('Time [s]',fontsize=36)
        ax.set_ylabel('RMS / Target peak',fontsize=32);ax.tick_params(labelsize=24,pad=10,top=False,right=False)
        ax.set_xticks([0,.05,.1,.15,.2],['0.00','0.05','0.10','0.15','0.20'])
        ax.legend(fontsize=24,loc='upper right');fig.savefig(path,dpi=100);plt.close(fig)
    return dict(path=str(path),image_sha256=digest(path),rms_window_ms=5,ylim=list(limits),
        normalization='Both raw waveforms divided by target peak',title='')


def control_comparison(rows,target,path,*,columns=((1.,'λ = 1.0'),(.5,'λ = 0.5')),indices=(0,2,3)):
    """rows: method, seed, column, controls; one Final or Selected role per image."""
    names={0:r'$\varepsilon_l$',2:r'$p_s$',3:r'$f_{e,l}$'}
    with plt.style.context(['science','ieee','no-latex']):
        fig,axes=plt.subplots(len(indices),len(columns),figsize=(18,5*len(indices)),squeeze=False,sharey='row')
        for ri,index in enumerate(indices):
            values=np.array([r['controls'][index] for r in rows]);span=max(float(np.ptp(np.r_[values,target[index]])),abs(target[index])*.02)
            all_bounds=[target[index]]
            for ci,(column,label) in enumerate(columns):
                ax=axes[ri,ci]
                for mi,method in enumerate(METHODS):
                    v=np.array([next(r['controls'][index] for r in rows if r['method']==method and r['seed']==s and r['column']==column) for s in (0,1,2)])
                    for si,(offset,marker) in enumerate(zip((-.16,-.05,.06),('o','s','^'))):
                        ax.scatter(mi+offset,v[si],marker=marker,s=65,color=COLORS[method],zorder=4,label=f'seed {si}' if mi==0 else None)
                    ax.errorbar(mi+.20,v.mean(),yerr=v.std(ddof=1),fmt='D',ms=6,color='black',capsize=5,lw=2,zorder=5,label='mean ± SD' if mi==0 else None)
                    all_bounds.extend((float(v.mean()-v.std(ddof=1)),float(v.mean()+v.std(ddof=1)),*v.tolist()))
                ax.axhline(target[index],ls='--',color='.3',lw=2,label='Target')
                ax.set_xticks(range(4),[LABELS[m].replace('Random Search','Random\nSearch') for m in METHODS],fontsize=20)
                ax.set_xlim(-.5,3.5);ax.tick_params(axis='y',labelsize=22,top=False,right=False)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False);ax.yaxis.get_offset_text().set_fontsize(20)
                ax.set_xlabel(label,fontsize=24)
                if ci==0:ax.set_ylabel(names[index]+' [model units]',fontsize=26)
            low=min(all_bounds);high=max(all_bounds);span=max(high-low,span)
            for ax in axes[ri]:ax.set_ylim(low-.15*span,high+.15*span)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=5,fontsize=18,frameon=False)
        # Reserve a physical margin so the one-row layout's lambda labels do
        # not collide with the common seed/mean/target legend.
        fig.tight_layout(rect=(0,.9/(5*len(indices)),1,1),h_pad=2,w_pad=2);fig.savefig(path,dpi=100);plt.close(fig)
    return dict(path=str(path),image_sha256=digest(path),rows=len(indices),columns=len(columns),
        seeds=[0,1,2],sample_sd_ddof=1,target_line='dashed',f0_l_is_hz=False,title='')
