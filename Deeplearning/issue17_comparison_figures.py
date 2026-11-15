"""Four methods per dimension/lambda; preserve every original observation."""
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

from .issue15_analysis import write_json
from .issue16 import digest

METHODS=('TD3','SAC','DE','random_search')
LABELS={'TD3':'TD3','SAC':'SAC','DE':'DE','random_search':'Random Search'}
COLORS={'TD3':'#0072B2','SAC':'#D55E00','DE':'#009E73','random_search':'#CC79A7'}
STYLES={'TD3':'-','SAC':'--','DE':':','random_search':'-.'}
STYLE={'path.simplify':False,'agg.path.chunksize':0,'axes.grid':False,
       'axes.labelsize':46,'xtick.labelsize':30,'ytick.labelsize':30}


def array_hash(values):
    return hashlib.sha256(np.asarray(values,dtype=np.float64).tobytes()).hexdigest()


def comparison_curve(trajectories,path,*,limits,average=False):
    if set(trajectories)!=set(METHODS):raise ValueError('Exactly four methods required')
    with plt.style.context(['science','ieee','no-latex']),plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(15,10))
        lines={};metadata={};bands=[]
        # Draw dense candidate trajectories first so deterministic policy
        # evaluations remain visible. No candidate/vertex is discarded.
        for method in ('random_search','DE','SAC','TD3'):
            x,values=trajectories[method]
            values=np.asarray(values)
            if values.shape!=(3,len(x)):raise ValueError('Three aligned seeds required')
            center=values.mean(axis=0) if average else values[0]
            line,=ax.plot(x,center,color=COLORS[method],ls=STYLES[method],lw=2.8,
                          alpha=.65 if method=='random_search' else 1.)
            lines[method]=line
            sd=values.std(axis=0,ddof=1)
            if average:
                bands.append(ax.fill_between(x,center-sd,center+sd,color=COLORS[method],alpha=.10,linewidth=0))
            metadata[method]=dict(vertices=len(line.get_xydata()),x_sha256=array_hash(line.get_xdata()),
                y_sha256=array_hash(line.get_ydata()),sd_sha256=array_hash(sd) if average else None,
                color=line.get_color(),linewidth=line.get_linewidth())
        ax.set_xlim(0,4096);ax.set_ylim(*limits);ax.set_xticks([0,1024,2048,3072,4096])
        ax.set_xlabel('Number of trials',fontsize=46,labelpad=12)
        ax.set_ylabel('Acoustic error',fontsize=46,labelpad=12)
        ax.tick_params(labelsize=30,pad=10,width=2,length=9,top=False,right=False)
        for spine in ax.spines.values():spine.set_linewidth(2)
        legend=ax.legend([lines[m] for m in METHODS],[LABELS[m] for m in METHODS],loc='upper right',
                         ncol=2,fontsize=29,frameon=True,framealpha=.95,facecolor='white',edgecolor='.7',
                         borderpad=.35,columnspacing=.9,handlelength=1.7)
        fig.tight_layout(pad=.5);fig.canvas.draw()
        renderer=fig.canvas.get_renderer();bbox=legend.get_window_extent(renderer)
        overlaps=[method for method,line in lines.items()
                  if line.get_path().transformed(ax.transData).intersects_bbox(bbox,filled=False)]
        band_overlap=any(path.transformed(band.get_transform()).intersects_bbox(bbox,filled=True)
                         for band in bands for path in band.get_paths())
        result=dict(methods=metadata,line_count=len(ax.lines),band_count=len(ax.collections),
            xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),label_fontsize=46,
            tick_fontsizes=sorted({t.get_fontsize() for t in ax.get_xticklabels()+ax.get_yticklabels()}),
            legend_labels=[t.get_text() for t in legend.get_texts()],
            legend_fontsizes=sorted({t.get_fontsize() for t in legend.get_texts()}),
            legend_inside=bool(ax.get_window_extent(renderer).contains(*bbox.p0) and ax.get_window_extent(renderer).contains(*bbox.p1)),
            legend_line_overlaps=overlaps,legend_band_overlap=bool(band_overlap),
            title=ax.get_title(),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
            path_simplify=bool(plt.rcParams['path.simplify']),
            grid=any(l.get_visible() for l in ax.get_xgridlines()+ax.get_ygridlines()))
        if overlaps or band_overlap:
            plt.close(fig);raise RuntimeError('Legend overlaps data; adjust one shared axis range before publishing')
        fig.savefig(path,dpi=120,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    return result


def grouped_points(output):
    grouped={}
    with (Path(output)/'comparison_curves.csv').open() as stream:
        for row in csv.DictReader(stream):
            key=(row['condition'],float(row['envelope_weight']),row['method'],int(row['seed']))
            grouped.setdefault(key,[]).append(row)
    return grouped


def figures(output):
    output=Path(output);start=time.perf_counter();grouped=grouped_points(output)
    if len(grouped)!=72:raise ValueError('Expected 72 method/weight/seed records')
    extrema=[0.]
    for key in grouped:
        if key[3]!=0:continue
        for field in ('E_total','objective_selected_common_error'):
            a=np.array([[float(p[field]) for p in grouped[key[:3]+(s,)]] for s in (0,1,2)])
            mean,sd=a.mean(axis=0),a.std(axis=0,ddof=1)
            extrema.extend((float(a.min()),float(a.max()),float((mean-sd).min()),float((mean+sd).max())))
    low,high=min(extrema),max(extrema);span=max(high-low,.1)
    # Space inside the axes for the required 29 pt, two-row legend.
    limits=(low-.025*span,high+.30*span)
    old_meta=output/'figure_sources.json'
    metadata=[]
    if old_meta.exists():
        metadata=[p for p in json.loads(old_meta.read_text())['images'] if p.get('kind') in ('spectrogram','physical_control')]
    source_hash=digest(output/'comparison_curves.csv')
    for condition in ('constant_1d','constant_2d','constant_3d'):
        for weight,label in ((1.,'lambda1'),(.5,'lambda0p5')):
            for subfolder,field in (('learning_curves','E_total'),('best_observed','objective_selected_common_error')):
                (output/'figures'/subfolder).mkdir(parents=True,exist_ok=True)
                trajectories={}
                for method in METHODS:
                    records=[grouped[(condition,weight,method,s)] for s in (0,1,2)]
                    x=np.array([int(r['episodes']) for r in records[0]])
                    expected=np.arange(0,4097,128) if method in ('TD3','SAC') else np.arange(1,4097)
                    if not np.array_equal(x,expected):raise ValueError('Unexpected raw observation positions')
                    trajectories[method]=(x,np.array([[float(r[field]) for r in run] for run in records]))
                for mean in (False,True):
                    filename=f"{condition.replace('constant_','')}_{label}_{'mean_sd' if mean else 'seed0'}.png"
                    destination=output/'figures'/subfolder/filename
                    geometry=comparison_curve(trajectories,destination,limits=limits,average=mean)
                    metadata.append(dict(path=str(destination.relative_to(output)),condition=condition,envelope_weight=weight,
                        field=field,mean=mean,sd_ddof=1 if mean else None,source_csv_sha256=source_hash,
                        image_sha256=digest(destination),**geometry))
    write_json(output/'figure_sources.json',dict(images=metadata,renderer_sha256=digest(__file__),
        shared_ylim=limits,smoothing=False,decimation=False,grouping='one dimension and one lambda; all four methods',
        best_definition='Common error of min-J_lambda archive incumbent; ties first; not post-hoc min common error',
        plot_wall_s=time.perf_counter()-start))
    figure_audit(output)
    return metadata


def figure_audit(output):
    output=Path(output);data=json.loads((output/'figure_sources.json').read_text());grouped=grouped_points(output)
    errors=[];curves=[p for p in data['images'] if 'field' in p]
    if len(curves)!=24:errors.append('Expected exactly24 four-method comparison images')
    for image in curves:
        if digest(output/image['path'])!=image['image_sha256']:errors.append(image['path']+':image hash')
        valid=(image['line_count']==4 and image['band_count']==(4 if image['mean'] else 0) and
            image['xlabel']=='Number of trials' and image['ylabel']=='Acoustic error' and image['label_fontsize']==46 and
            image['tick_fontsizes']==[30] and image['legend_fontsizes']==[29] and
            image['legend_labels']==[LABELS[m] for m in METHODS] and image['legend_inside'] and
            not image['legend_line_overlaps'] and not image['legend_band_overlap'] and
            not image['title'] and not image['grid'] and not image['path_simplify'] and
            image['xlim']==[0,4096] and image['ylim']==list(data['shared_ylim']))
        if not valid:errors.append(image['path']+':format')
        for method in METHODS:
            rows=[grouped[(image['condition'],image['envelope_weight'],method,s)] for s in (0,1,2)]
            x=np.array([float(r['episodes']) for r in rows[0]])
            a=np.array([[float(r[image['field']]) for r in run] for run in rows])
            y=a.mean(axis=0) if image['mean'] else a[0];geometry=image['methods'][method]
            if (geometry['x_sha256']!=array_hash(x) or geometry['y_sha256']!=array_hash(y) or
                geometry['vertices']!=len(x) or geometry['color']!=COLORS[method] or geometry['linewidth']<2.8):
                errors.append(image['path']+':'+method+' data')
            if image['mean'] and geometry['sd_sha256']!=array_hash(a.std(axis=0,ddof=1)):
                errors.append(image['path']+':'+method+' SD')
    result=dict(passed=not errors,errors=errors,comparison_images=len(curves),
        groups=6,methods_per_image=4,shared_axes_between_lambdas=True,all_observations_preserved=True)
    write_json(output/'figure_audit.json',result)
    if errors:raise RuntimeError(f'Comparison figure audit failed:{errors[:3]}')
    return result
