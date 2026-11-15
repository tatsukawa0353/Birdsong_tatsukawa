"""One condition and one curve per research figure; numerical artifacts are read-only."""
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from .constant020_env import INDICES,RATE
from .issue15_analysis import write_json
from .issue16 import digest
from .issue17 import weight_label
from .model import LOW,HIGH
from .plotting import draw_spectrogram

COLORS={'TD3':'#0072B2','SAC':'#D55E00','DE':'#009E73','random_search':'#CC79A7'}
PARAMETERS={0:('epsilon_l',r'$\varepsilon_l$'),2:('ps',r'$p_s$'),3:('f0_l',r'$f_{e,l}$')}
STYLE={'path.simplify':False,'agg.path.chunksize':0,'axes.grid':False,
       'font.size':30,'axes.labelsize':46,'xtick.labelsize':30,'ytick.labelsize':30}


def condition_stem(condition,method,weight):
    return f"{condition.replace('constant_','')}_{method}_{weight_label(weight)}"


def single_curve(x,y,path,*,band=None,limits):
    """Return measured geometry so style/vertices/axis equality can be audited."""
    with plt.style.context(['science','ieee','no-latex']),plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(14,9))
        ax.plot(x,y,linewidth=2.8,color='#0072B2')
        if band is not None:
            ax.fill_between(x,band[0],band[1],color='#0072B2',alpha=.16,linewidth=0)
        ax.set_xlim(0,4096);ax.set_ylim(*limits)
        ax.set_xticks([0,1024,2048,3072,4096])
        ax.set_xlabel('Number of trials',fontsize=46,labelpad=12)
        ax.set_ylabel('Acoustic error',fontsize=46,labelpad=12)
        ax.tick_params(labelsize=30,pad=10,width=2,length=9,top=False,right=False)
        for spine in ax.spines.values():spine.set_linewidth(2)
        fig.tight_layout(pad=.5)
        fig.canvas.draw()
        extent=fig.get_tightbbox(fig.canvas.get_renderer())
        metadata=dict(line_count=len(ax.lines),band_count=len(ax.collections),
            x_values_sha256=hashlib.sha256(np.asarray(ax.lines[0].get_xdata(),dtype=np.float64).tobytes()).hexdigest(),
            y_values_sha256=hashlib.sha256(np.asarray(ax.lines[0].get_ydata(),dtype=np.float64).tobytes()).hexdigest(),
            line_width=ax.lines[0].get_linewidth(),vertex_count=len(ax.lines[0].get_xydata()),
            xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),xlabel_fontsize=ax.xaxis.label.get_fontsize(),
            ylabel_fontsize=ax.yaxis.label.get_fontsize(),tick_fontsizes=sorted({t.get_fontsize() for t in ax.get_xticklabels()+ax.get_yticklabels()}),
            title=ax.get_title(),legend=ax.get_legend() is not None,
            xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),path_simplify=plt.rcParams['path.simplify'],
            tight_bounds_inches=extent.bounds,
            x_grid_visible=any(line.get_visible() for line in ax.get_xgridlines()),
            y_grid_visible=any(line.get_visible() for line in ax.get_ygridlines()))
        fig.savefig(path,dpi=120,bbox_inches='tight',pad_inches=.08)
        plt.close(fig)
    return metadata


def figures(output):
    output=Path(output);started=time.perf_counter()
    with (output/'comparison_curves.csv').open() as stream:
        points=list(csv.DictReader(stream))
    grouped={}
    for p in points:
        key=(p['condition'],p['method'],float(p['envelope_weight']),int(p['seed']))
        grouped.setdefault(key,[]).append(p)
    conditions=sorted({key[:3] for key in grouped})
    # One common range for every image, including both lambda groups and SD.
    ymin,ymax=0.,0.
    for key in conditions:
        for field in ('E_total','best_observed_error'):
            array=np.array([[float(p[field]) for p in grouped[key+(s,)]] for s in (0,1,2)])
            center,sd=array.mean(axis=0),array.std(axis=0,ddof=1)
            ymin=min(ymin,float((center-sd).min()),float(array.min()))
            ymax=max(ymax,float((center+sd).max()),float(array.max()))
    margin=.035*max(ymax-ymin,.01);limits=(ymin-margin,ymax+margin)
    metadata=[]
    for subfolder in ('learning_curves','best_observed','spectrograms','physical_controls'):
        (output/'figures'/subfolder).mkdir(parents=True,exist_ok=True)
    for condition,method,weight in conditions:
        stem=condition_stem(condition,method,weight)
        by_seed=[grouped[(condition,method,weight,s)] for s in (0,1,2)]
        x=np.array([int(p['episodes']) for p in by_seed[0]])
        expected=np.arange(0,4097,128) if method in ('TD3','SAC') else np.arange(1,4097)
        if not np.array_equal(x,expected):raise ValueError(f'Wrong evaluation positions: {stem}')
        for directory,field in (('learning_curves','E_total'),('best_observed','best_observed_error')):
            array=np.array([[float(p[field]) for p in run] for run in by_seed])
            for mean in (False,True):
                y=array.mean(axis=0) if mean else array[0]
                sd=array.std(axis=0,ddof=1) if mean else None
                destination=output/'figures'/directory/(stem+('_mean_sd.png' if mean else '_seed0.png'))
                geometry=single_curve(x,y,destination,band=(y-sd,y+sd) if mean else None,limits=limits)
                metadata.append(dict(path=str(destination.relative_to(output)),condition=condition,method=method,
                    envelope_weight=weight,mean=mean,field=field,source_csv_sha256=digest(output/'comparison_curves.csv'),
                    source_point_count=len(x),source_seed_count=3 if mean else 1,sd_ddof=1 if mean else None,**geometry))
        folder=output/'runs'/condition/method/weight_label(weight)/'seed_0'
        with np.load(folder/'rollout.npz') as data:
            target,audio,controls=data['target'].copy(),data['audio'].copy(),data['physical_controls'].copy()
        destination=output/'figures/spectrograms'/(stem+'_seed0.png')
        with plt.style.context(['science','ieee','no-latex']),plt.rc_context(STYLE):
            fig,axes=plt.subplots(2,1,figsize=(14,12),sharex=True)
            for ax,wave in zip(axes,(target,audio)):
                draw_spectrogram(ax,wave,RATE,max_hz=10000)
                ax.set_title('')
            axes[-1].set_xlabel('Time [s]',fontsize=46)
            fig.tight_layout(pad=.5,h_pad=1)
            fig.savefig(destination,dpi=120,bbox_inches='tight',pad_inches=.08);plt.close(fig)
        metadata.append(dict(path=str(destination.relative_to(output)),kind='spectrogram',condition=condition,
            method=method,envelope_weight=weight,title='',upper='target',lower='generated',max_hz=10000,
            rate=RATE,duration_s=.2,samples=len(audio),source_rollout_sha256=digest(folder/'rollout.npz')))
        for index in INDICES[condition]:
            name,label=PARAMETERS[index]
            destination=output/'figures/physical_controls'/(stem+'_'+name+'_seed0.png')
            with plt.style.context(['science','ieee','no-latex']),plt.rc_context(STYLE):
                fig,ax=plt.subplots(figsize=(14,9))
                ax.plot([0,.2],[controls[0,index]]*2,lw=2.8,color=COLORS[method])
                delta=(HIGH[index]-LOW[index])*.04
                ax.set_xlim(0,.2);ax.set_ylim(LOW[index]-delta,HIGH[index]+delta)
                ax.set_xlabel('Time [s]',fontsize=46);ax.set_ylabel(label,fontsize=46)
                ax.tick_params(labelsize=30,pad=10,top=False,right=False)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
                ax.yaxis.get_offset_text().set_fontsize(30)
                fig.tight_layout(pad=.5);fig.savefig(destination,dpi=120,bbox_inches='tight',pad_inches=.08);plt.close(fig)
            metadata.append(dict(path=str(destination.relative_to(output)),kind='physical_control',condition=condition,
                method=method,envelope_weight=weight,parameter=name,physical_value=float(controls[0,index]),
                unit='native model coefficient; f0_l is external force, not frequency in Hz',
                line_count=1,title='',source_rollout_sha256=digest(folder/'rollout.npz')))
    for item in metadata:item['image_sha256']=digest(output/item['path'])
    write_json(output/'figure_sources.json',dict(renderer_sha256=digest(__file__),images=metadata,
        shared_learning_curve_ylim=limits,plot_wall_s=time.perf_counter()-started,
        smoothing=False,decimation=False,mean_sd_band='exact mean +/- sample SD, untruncated data',
        titles=False,one_condition_per_image=True))
    figure_audit(output)
    return metadata


def figure_audit(output):
    output=Path(output)
    data=json.loads((output/'figure_sources.json').read_text());errors=[]
    curves=[r for r in data['images'] if 'field' in r]
    with (output/'comparison_curves.csv').open() as stream:
        points=list(csv.DictReader(stream))
    if len(curves)!=72:errors.append('Expected 72 single-curve images')
    for r in data['images']:
        if digest(output/r['path'])!=r['image_sha256']:errors.append(r['path']+': hash')
        if r.get('title'):errors.append(r['path']+': title')
        if 'field' not in r:continue
        selected=[[p for p in points if p['condition']==r['condition'] and p['method']==r['method'] and
                   float(p['envelope_weight'])==r['envelope_weight'] and int(p['seed'])==seed] for seed in (0,1,2)]
        values=np.array([[float(p[r['field']]) for p in run] for run in selected])
        y=values.mean(axis=0) if r['mean'] else values[0]
        x=np.array([float(p['episodes']) for p in selected[0]])
        if hashlib.sha256(y.astype(np.float64).tobytes()).hexdigest()!=r['y_values_sha256']:
            errors.append(r['path']+': plotted values do not match CSV')
        if hashlib.sha256(x.astype(np.float64).tobytes()).hexdigest()!=r['x_values_sha256']:
            errors.append(r['path']+': plotted positions do not match CSV')
        good=(r['line_count']==1 and r['line_width']>=2.8 and r['band_count']==int(r['mean']) and
              r['xlabel']=='Number of trials' and r['ylabel']=='Acoustic error' and
              r['xlabel_fontsize']==r['ylabel_fontsize']==46 and r['tick_fontsizes']==[30] and
              not r['legend'] and not r['path_simplify'] and not r['x_grid_visible'] and not r['y_grid_visible'] and
              r['vertex_count']==r['source_point_count'] and r['xlim']==[0,4096] and
              r['ylim']==list(data['shared_learning_curve_ylim']))
        if not good:errors.append(r['path']+': geometry')
    result=dict(passed=not errors,errors=errors,curve_images=len(curves),total_images=len(data['images']),
                one_condition_one_curve=True,all_lambda_axis_ranges_equal=True)
    write_json(output/'figure_audit.json',result)
    if errors:raise RuntimeError(f'Figure audit failed: {errors[:3]}')
    return result
