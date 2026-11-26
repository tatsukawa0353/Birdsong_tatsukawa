"""Read-only baselines and fixed conditions for the Case A network-width study."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import json
from pathlib import Path

import numpy as np

from .issue17 import ROOT
from .issue17_artifacts import read_record
from .issue19_data import pack, metric_row, save_audio
from .issue18_domain_env import DomainEnv
from .issue16 import digest

BASE = ROOT/'results/Phase1/CaseA/domain_diagnostics'
PRIOR = ROOT/'results/Phase1/CaseA/reward_weights/restricted_domain'
OUT = ROOT/'results/Phase1/CaseA/network_width'
METHODS = ('TD3', 'SAC')
WIDTHS = (32, 64, 128, 256)
SEEDS = (0, 1, 2)
SOURCES = ('issue21_data.py', 'issue21_training.py', 'constant020_env.py',
    'issue18_domain_env.py', 'issue19_data.py', 'issue14.py', 'issue17_artifacts.py',
    'model.py', 'model_env.py', 'native/model.cpp', 'acoustics.py')


def source_hashes():
    return {p:digest(resolve_result_path(ROOT/p)) for p in SOURCES}


def target_data():
    with np.load(BASE/'diagnostics/target.npz') as z:
        return z['target'].copy(), z['controls'].copy()


def environment():
    target, controls = target_data()
    return DomainEnv((0, 2, 3), target, controls, envelope_weight=1., f0_high=800000.)


def folder_for(method, width, seed, output=OUT):
    if method not in METHODS or width not in WIDTHS or seed not in SEEDS:
        raise ValueError('Only registered TD3/SAC widths and seeds are supported')
    if width == 32:
        return BASE/'runs/constant_3d'/method/'lambda1'/f'seed_{seed}'
    return Path(output)/'runs'/method/f'width_{width}'/f'seed_{seed}'


def curve_for(method, width, seed, output=OUT):
    return read_record(folder_for(method, width, seed, output)/'curve.json')


def endpoints(method, width, seed, output=OUT):
    points = curve_for(method, width, seed, output)
    # Strict minimum: first tied point, without post-hoc metric tie breaking.
    selected = min(points, key=lambda p:p['weighted_loss'])
    return dict(final=points[-1], selected=selected)


def ready(output=OUT):
    from .model import LIBRARY
    m = json.loads((Path(output)/'manifest.json').read_text())
    if not source_hashes_match(m['source_sha256'], source_hashes()) or m['native_library_sha256'] != digest(LIBRARY):
        raise RuntimeError('Frozen numerical implementation or native library changed')
    if m['target_sha256'] != digest(BASE/'diagnostics/target.npz'):
        raise RuntimeError('Original target changed')
    if m['critic_pool_sha256'] != digest(PRIOR/'critic_pool.csv'):
        raise RuntimeError('Original fixed critic pool changed')
    return m
