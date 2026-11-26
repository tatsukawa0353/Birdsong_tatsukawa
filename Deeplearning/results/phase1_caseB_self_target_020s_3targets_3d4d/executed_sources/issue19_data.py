"""Immutable inputs and compact records for the restricted-3D weight study."""
import gzip
import json
from pathlib import Path

import numpy as np

from .issue16 import digest
from .issue17 import ROOT, weight_label
from .issue17_artifacts import read_record
from .issue17_extension_data import write_csv
from .issue17_selection import selection_curve
from .issue15_analysis import write_json
from .issue18_domain_env import DomainEnv

BASE = ROOT / 'results/phase1_3d_domain_diagnostics'
OUT = ROOT / 'results/phase1_restricted3d_weights'
METHODS = ('TD3', 'SAC', 'DE', 'random_search')
WEIGHTS = (1., .5)
INDICES = (0, 2, 3)
SOURCES = ('issue19_data.py', 'issue19_training.py', 'constant020_env.py',
           'issue18_domain_env.py', 'model.py', 'model_env.py', 'native/model.cpp',
           'acoustics.py', 'fixed_budget_de.py', 'issue14.py')


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCES}


def target_data():
    with np.load(BASE/'diagnostics/target.npz') as z:
        return z['target'].copy(), z['controls'].copy()


def environment(weight=1.):
    target, controls = target_data()
    return DomainEnv(INDICES, target, controls, envelope_weight=weight, f0_high=800000.)


def folder_for(method, seed, weight):
    root = BASE if weight == 1. else OUT
    return root/'runs/constant_3d'/method/weight_label(weight)/f'seed_{seed}'


def curve_for(method, seed, weight):
    # Random's complete pool is referenced, never copied or re-rendered.
    source_weight = 1. if method == 'random_search' else weight
    return selection_curve(read_record(folder_for(method, seed, source_weight)/'curve.json'), weight)


def endpoints(method, seed, weight):
    curve = curve_for(method, seed, weight)
    return {'final': curve[-1], 'selected': curve[curve[-1]['objective_selected_index']]}


def pack(path, records):
    data = json.dumps(records, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    compressed = gzip.compress(data, mtime=0)
    assert gzip.decompress(compressed) == data
    Path(path).write_bytes(compressed)


def metric_row(metrics, values, n, **extra):
    return dict(episodes=n, **metrics, actions=values['actions'].tolist(),
                physical_controls=values['physical_controls'][0].tolist(), **extra)


def save_audio(path, values):
    # The target and listening gain live in the shared manifest / baseline.
    np.savez_compressed(path, **values, rate=50000, duration_s=.2)


def ready():
    from .model import LIBRARY
    m = json.loads((OUT/'manifest.json').read_text())
    if m['source_sha256'] != source_hashes() or m['native_library_sha256'] != digest(LIBRARY):
        raise RuntimeError('Frozen study sources or native library changed')
    if digest(BASE/'diagnostics/target.npz') != m['target_sha256']:
        raise RuntimeError('Original target changed')
    return m
