"""Shared publication style inherited from Issues 17 and 27.

Use ``with context():`` for every new research figure. Plot all observations;
show sample standard deviation with a translucent fill_between band.
"""
from contextlib import contextmanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # registers SciencePlots styles
from cycler import cycler

COLORS = {'TD3': '#0072B2', 'SAC': '#D55E00', 'PPO': '#009E73', 'Target': '#222222'}
STYLE = {
    'font.family': 'serif', 'font.serif': ['STIXGeneral'],
    'mathtext.fontset': 'cm', 'text.usetex': False,
    'font.size': 30, 'axes.labelsize': 46, 'xtick.labelsize': 30,
    'ytick.labelsize': 30, 'legend.fontsize': 29,
    'lines.linewidth': 4., 'lines.linestyle': '-', 'axes.grid': False,
    'axes.titlesize': 30, 'path.simplify': False, 'agg.path.chunksize': 0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.constrained_layout.use': True,
    'xtick.top': False, 'ytick.right': False,
    'xtick.minor.top': False, 'ytick.minor.right': False,
    'axes.prop_cycle': cycler(color=['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#222222']),
}
COLORBAR_LABEL_PT = 46
COLORBAR_TICK_PT = 30
STD_ALPHA = .15


@contextmanager
def context():
    with plt.style.context(['science', 'ieee', 'no-latex']), plt.rc_context(STYLE):
        yield


def colorbar(fig, mesh, axes, label='Relative PSD [dB]'):
    import numpy as np
    bar = fig.colorbar(mesh, ax=np.asarray(axes).ravel().tolist(), pad=.025, shrink=.95)
    bar.set_label(label, fontsize=COLORBAR_LABEL_PT)
    bar.ax.tick_params(labelsize=COLORBAR_TICK_PT)
    return bar
