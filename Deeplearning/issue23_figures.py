"""Seven aggregate scientific figures, preserving all three seed trajectories."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .issue23_data import OUT,SEEDS,NAMES,folder_for,curve_for,chosen
from .issue17_artifacts import read_record
from .issue15_analysis import write_json
from .issue16 import digest

COLORS=('#0072B2','#D55E00','#009E73')
LABELS=('ε_l','p_s','f_e,l (not Hz)')


def csv_rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def render():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'figure.constrained_layout.use':True})
    (OUT/'figures').mkdir(exist_ok=True);images=[]
    def save(fig,name,description):
        p=OUT/'figures'/f'{name}.png';fig.savefig(p,dpi=125);plt.close(fig)
        images.append(dict(path=str(p.relative_to(OUT)),sha256=digest(p),description=description))
    stage=csv_rows('stage_a_evaluations.csv');curves={s:curve_for(s) for s in SEEDS}
    def markers(ax,seed):
        p=chosen(curves[seed]);ax.axvline(p['selected']['episodes'],color='#AA3377',ls='--',lw=1,label='min J')
        ax.axvline(p['posthoc']['episodes'],color='#228833',ls=':',lw=1.5,label='min E (posthoc)')
    fig,axes=plt.subplots(4,3,figsize=(16,13),sharex=True)
    for s in SEEDS:
        rows=[r for r in stage if int(r['seed'])==s];x=[int(r['trial']) for r in rows]
        for i,key in enumerate(('E_total','E_spec','E_env','J')):
            ax=axes[i,s];ax.plot(x,[float(r[key]) for r in rows],'.-',color=COLORS[s],ms=4);markers(ax,s)
            if key=='E_total':ax.axhline(.1,color='.5',ls=':',label='E=.10 reference')
            ax.set(title=f'Seed {s}: {key}',ylabel=key,xlim=(0,4096));ax.legend(fontsize=7)
            if i==3:ax.set_xlabel('Training candidates; 33 independent evaluations')
    save(fig,'loss_histories','Existing Issue22 logs, all seeds and all33 points; lines connect observations, no unseen states')

    fig,axes=plt.subplots(4,3,figsize=(16,13),sharex=True)
    for s in SEEDS:
        points=curves[s];x=[p['episodes'] for p in points]
        ax=axes[0,s];ax.plot(x,[p['policy_statistics']['alpha'] for p in points],'.-',color=COLORS[s]);ax.set(ylabel='Temperature α',yscale='log',title=f'Seed {s}')
        for j in range(3):
            axes[1,s].plot(x,[p['policy_statistics']['normalized_action_std_mc'][j] for p in points],color=COLORS[j],label=LABELS[j])
            axes[2,s].plot(x,[p['policy_statistics']['latent_log_std'][j] for p in points],color=COLORS[j],label=LABELS[j])
        axes[1,s].set_ylabel('Post-tanh action std (MC4096)');axes[2,s].set_ylabel('Pre-tanh log_std')
        axes[3,s].plot(x,[p['policy_statistics']['squashed_entropy_mc'] for p in points],label='Squashed (MC4096)',color='black')
        axes[3,s].plot(x,[p['policy_statistics']['gaussian_entropy'] for p in points],label='Gaussian',color='.6',ls='--')
        axes[3,s].set(ylabel='Differential entropy [nats]',xlabel='Training candidates')
        for ax in axes[:,s]:markers(ax,s);ax.legend(fontsize=7);ax.set_xlim(0,4096)
    save(fig,'policy_histories','Same time axis as acoustic errors; alpha, coordinate std, log_std and both entropies; no causal inference')

    fig,axes=plt.subplots(5,3,figsize=(16,15),sharex=True)
    for s in SEEDS:
        rows=[r for r in stage if int(r['seed'])==s];x=[int(r['trial']) for r in rows]
        for j,name in enumerate(NAMES):
            axes[0,s].plot(x,[float(r[f'action_{name}']) for r in rows],color=COLORS[j],label=LABELS[j])
            axes[1,s].plot(x[1:],[float(r[f'delta_action_{name}']) for r in rows[1:]],color=COLORS[j],label='Δ'+LABELS[j])
            axes[j+2,s].plot(x[1:],[float(r[f'delta_physical_{name}']) for r in rows[1:]],'.-',color=COLORS[j])
            axes[j+2,s].set_ylabel('Δ'+LABELS[j]+' [model units]');axes[j+2,s].ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        axes[0,s].set(title=f'Seed {s}: deterministic action',ylabel='Normalized action [-1,1]')
        axes[1,s].plot(x[1:],[float(r['delta_action_l2']) for r in rows[1:]],color='black',label='||Δa||₂')
        axes[1,s].set_ylabel('Δa per128 candidates');axes[0,s].legend(fontsize=7);axes[1,s].legend(fontsize=7)
        for ax in axes[:,s]:markers(ax,s);ax.axhline(0,color='.8',lw=.5)
        axes[-1,s].set_xlabel('Training candidates; physical units kept separate')
    save(fig,'action_changes','128-candidate observation differences, normalized coordinates and physical increments on separate axes')

    ranks=csv_rows('candidate_ranks.csv');metrics=csv_rows('critic_metrics.csv')
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True)
    for s in SEEDS:
        p=curves[s];x=[r['episodes'] for r in p]
        axes[0,s].plot(x,[-r['weighted_loss'] for r in p],color='black',label='Actor measured r=-J')
        axes[0,s].plot(x,[r['actor_minQ'] for r in p],color=COLORS[s],label='Critic minQ(actor)')
        for role,ls,col in (('selected','--','#AA3377'),('posthoc',':','#228833')):
            rows=[r for r in ranks if int(r['seed'])==s and r['reference_role']==role]
            axes[1,s].plot([int(r['trial']) for r in rows],[float(r['reference_bias']) for r in rows],ls=ls,color=col,label=role+' Q-r')
        rows=[r for r in metrics if int(r['seed'])==s and r['head']=='minQ' and r['subset']=='own_late17']
        axes[2,s].plot([int(r['trial']) for r in rows],[float(r['spearman']) if r['spearman'] else np.nan for r in rows],color=COLORS[s],label='Own late17: ρ(minQ,r)')
        rows=[r for r in metrics if int(r['seed'])==s and r['head']=='minQ' and r['subset']=='all99']
        axes[2,s].plot([int(r['trial']) for r in rows],[float(r['spearman']) if r['spearman'] else np.nan for r in rows],color='.4',ls='--',label='All99: ρ(minQ,r)')
        axes[0,s].set(title=f'Seed {s}',ylabel='Q and measured terminal reward');axes[1,s].set_ylabel('Saved best candidate Q bias');axes[2,s].set(ylabel='Spearman ρ',ylim=(-1.05,1.05),xlabel='Critic training candidates')
        for ax in axes[:,s]:ax.axhline(0,color='.7',lw=.7);markers(ax,s);ax.legend(fontsize=7)
    save(fig,'critic_histories','Q from newly recorded intermediate Critics, fixed retrospective candidates; terminal target=r, deterministic Q excludes SAC entropy')

    predictions=read_record(OUT/'critic_predictions.json')
    candidates=json.loads((OUT/'fixed_candidates.json').read_text());truth={c['candidate_id']:c for c in candidates}
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,n in zip(axes.ravel(),(2944,3072,3840,4096)):
        rows=[r for r in predictions if r['seed']==2 and r['trial']==n]
        ax.scatter([r['reward'] for r in rows],[r['minQ'] for r in rows],s=14,c='.75',label='99 fixed candidates')
        own=[r for r in rows if r['source_seed']==2]
        ax.scatter([r['reward'] for r in own],[r['minQ'] for r in own],s=18,c='#0072B2',label='Own33')
        for trial,col,marker in ((2944,'#228833','D'),(3072,'#CCBB44','v'),(3840,'#AA3377','s'),(4096,'#EE6677','X')):
            r=next(r for r in rows if r['candidate_id']==f's2_t{trial}')
            ax.scatter(r['reward'],r['minQ'],c=col,marker=marker,s=85,edgecolor='black',linewidth=.5,label=f'Actor@{trial}')
        v=[r['reward'] for r in rows]+[r['minQ'] for r in rows];low=min(v)-.01;high=max(v)+.01
        ax.plot([low,high],[low,high],color='black',ls=':',lw=1,label='Q=r')
        ax.set(title=f'Seed2 Critic@{n}',xlabel='Measured r = −(.75 E_spec+.25 E_env)',ylabel='min(Q1,Q2)',xlim=(low,high),ylim=(low,high));ax.legend(fontsize=7)
    save(fig,'seed2_q_snapshots','Same actions at all four Critics; all99 common axes per panel, actual reward independently generated before reproduction')

    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True)
    for s in SEEDS:
        updates=read_record(folder_for(s)/'updates.json');x=np.array([r['trial'] for r in updates])
        axes[0,s].plot(x,[u['delta_action_l2'] for u in updates],color=COLORS[s],lw=.5,label='Actual one-update ||Δa||₂')
        for key,color in (('actor_loss','#0072B2'),('critic_loss','#D55E00')):
            y=np.array([u[key] for u in updates]);centers=[];means=[]
            for end in range(128,4097,128):
                mask=(x>end-128)&(x<=end)
                centers.append(end);means.append(y[mask].mean())
            axes[1,s].plot(centers,means,color=color,label=key+' (interval mean)')
        p=curves[s];axes[2,s].plot([r['episodes'] for r in p],[r['previous_interval']['actor_path_length'] for r in p],color=COLORS[s],label='Path length Σ||Δa_update||₂')
        rows=[r for r in stage if int(r['seed'])==s and r['delta_action_l2']]
        axes[2,s].plot([int(r['trial']) for r in rows],[float(r['delta_action_l2']) for r in rows],color='black',label='Net ||a_end−a_start||₂')
        axes[0,s].set(title=f'Seed {s}',ylabel='Per-update change');axes[1,s].set_ylabel('Logged actual optimizer losses');axes[2,s].set(ylabel='Per128-candidate path/net change',xlabel='Training candidates')
        for ax in axes[:,s]:markers(ax,s);ax.legend(fontsize=7)
    save(fig,'actor_updates','All4080 real Actor updates per seed; no interpolated acoustic values; interval loss means and telescoping path/net increments')

    rows=csv_rows('local_sensitivity.csv');fig,axes=plt.subplots(2,3,figsize=(16,9))
    for j,n in enumerate((2944,3840,4096)):
        for i,name in enumerate(NAMES):
            subset=sorted([r for r in rows if int(r['anchor'])==n and r['variable']==name],key=lambda r:float(r['step'])*int(r['direction']))
            x=[float(r['step'])*int(r['direction']) for r in subset];mid=2
            for k,field in enumerate(('delta_E_total','delta_J')):
                y=[float(r[field]) for r in subset];axes[k,j].plot(x[:mid]+[0]+x[mid:],y[:mid]+[0]+y[mid:],'.-',color=COLORS[i],label=LABELS[i])
                axes[k,j].set(title=f'Seed2 anchor@{n}',xlabel='Perturbation in one normalized action coordinate',ylabel=field);axes[k,j].axhline(0,color='.7',ls=':');axes[k,j].legend(fontsize=8)
    save(fig,'local_sensitivity','Predeclared36 simulations:3 anchors ×3 coordinates ×±(.002,.01); local observations only, no optimization')
    assert len(images)==7;write_json(OUT/'figures.json',images)
    return images


if __name__=='__main__':render()
