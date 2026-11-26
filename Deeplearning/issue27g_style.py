"""One effective SciencePlots configuration, inherited from Issue17 code."""
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from .issue27g_data import OUT
from .issue16 import digest
from .issue15_analysis import write_json

from .plot_style import COLORS, STYLE, context, colorbar

ITEMS=[]
def identify(ax,label):
    ax.text(.025,.965,label,transform=ax.transAxes,ha='left',va='top',fontsize=29,bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=3))
def bar(fig,mesh,axes):
    return colorbar(fig,mesh,axes)
def save(fig,name,description):
    # Assert effective properties on actual artists, not only rcParams.
    fig.canvas.draw();fonts=set();line_widths=set();legends=[];vertices=[]
    for ax in fig.axes:
        assert not ax.get_title() and not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
        for label in (ax.xaxis.label,ax.yaxis.label):
            if label.get_text():assert label.get_fontsize()==46;fonts.add(fm.findfont(label.get_fontproperties()))
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            if tick.get_visible():assert tick.get_fontsize()==30
        if ax.get_legend():
            assert all(t.get_fontsize()==29 for t in ax.get_legend().get_texts());legends.append(ax.get_legend()._loc)
        for line in ax.lines:
            assert line.get_linestyle()=='-' and line.get_linewidth()==4.,(name,line.get_linestyle(),line.get_linewidth());line_widths.add(line.get_linewidth())
            values=np.column_stack((np.asarray(line.get_xdata(),float),np.asarray(line.get_ydata(),float)))
            vertices.append(dict(points=len(values),data_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    path=OUT/f'figures/{name}.png';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=120,bbox_inches='tight',pad_inches=.08)
    item=dict(path=str(path.relative_to(OUT)),sha256=digest(path),description=description,figure_inches=fig.get_size_inches().tolist(),dpi=120,
        label_pt=46,tick_pt=30,legend_pt=29,line_widths_pt=sorted(line_widths),fonts=sorted(fonts),mathtext_fontset=plt.rcParams['mathtext.fontset'],text_usetex=plt.rcParams['text.usetex'],
        font_family=plt.rcParams['font.family'],font_serif=plt.rcParams['font.serif'],resolved_font_names=sorted({fm.FontProperties(fname=p).get_name() for p in fonts}),
        legend_locations=legends,line_vertices=vertices,all_observed_points=True)
    ITEMS.append(item);plt.close(fig);return item
def metadata(name):
    write_json(OUT/f'{name}_figures.json',ITEMS)
