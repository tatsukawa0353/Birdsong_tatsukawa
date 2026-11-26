"""Inventory first; verify physics and preserve all pre-existing research files."""
from .result_paths import resolve as resolve_result_path
import argparse
from collections import defaultdict
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

from . import model
from .constant020_env import render, episode
from .issue19_data import BASE, OUT, ROOT, environment, source_hashes, target_data
from .issue15_analysis import write_json
from .issue16 import digest
from .issue17_extension_data import write_csv


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def workspace_inventory():
    """Logical regular-file bytes, including ignored environments/work folders."""
    groups = defaultdict(lambda: dict(files=0, bytes=0))
    total = dict(files=0, bytes=0)
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in names:
            path = Path(directory)/name
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            group = '/'.join(rel.parts[:2]) if rel.parts[0] == 'results' else rel.parts[0]
            size = path.stat().st_size
            groups[group]['files'] += 1; groups[group]['bytes'] += size
            total['files'] += 1; total['bytes'] += size
    return dict(total=total, groups=dict(sorted(groups.items())), excludes='.git and symlinks; logical bytes')


def protected_files():
    for prefix in ('results', 'reports'):
        for path in sorted((resolve_result_path(ROOT/prefix)).rglob('*')):
            if path.is_file() and not path.is_symlink() and not path.is_relative_to(OUT):
                yield path


def inventory():
    if OUT.exists():
        raise FileExistsError('Existing Issue19 output preserved')
    before = workspace_inventory()
    status = git('status', '--porcelain=v1', '--untracked-files=all')
    OUT.mkdir(parents=True)
    write_json(OUT/'storage_before.json', dict(workspace=before, git_status=status,
        head=git('rev-parse', 'HEAD').strip(), unpushed=git('log', '--oneline', '@{upstream}..HEAD'),
        created_utc=datetime.now(timezone.utc).isoformat()))
    rows = []
    for path in protected_files():
        rows.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest(path)))
    write_csv(OUT/'protected_before.csv', rows)
    groups = defaultdict(lambda: dict(files=0, bytes=0))
    for r in rows:
        p = Path(r['path']); category = p.suffix or '(no extension)'
        group = f'{p.parts[1]}/{category}'
        groups[group]['files'] += 1; groups[group]['bytes'] += r['bytes']
    write_json(OUT/'artifact_inventory.json', dict(groups=dict(sorted(groups.items())),
        protected_files=len(rows), protected_bytes=sum(r['bytes'] for r in rows),
        policy='All original research files hashed; only explicitly listed exact temporary duplicates may be removed.'))
    print(f'Inventoried {len(rows)} original research files', flush=True)


def model_gate():
    """Compile both revisions, inspect assigned constants, compare saved full sounds."""
    if not (OUT/'protected_before.csv').exists():
        raise RuntimeError('Inventory must precede model check')
    source = (ROOT/'native/model.cpp').read_text()
    prior = git('show', '0909f87^:Deeplearning/native/model.cpp')
    assert source.replace('controls[2],0.02,0.1', 'controls[2],0.04,0.1') == prior
    assert len(re.findall(r'\bx0\b', source)) == 1, 'x0 must only be a stored struct member'
    original_library = model.LIBRARY
    target, controls = target_data()
    examples = [('target', controls, target)]
    for method, seed in (('TD3', 0), ('SAC', 1), ('DE', 2), ('random_search', 0)):
        with np.load(BASE/f'runs/constant_3d/{method}/lambda1/seed_{seed}/rollout.npz') as z:
            examples.append((f'{method}_seed{seed}_final', z['physical_controls'][0].copy(), z['audio'].copy()))
    records = []; assigned = []
    tick = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='issue19-physics-') as directory:
            temp = Path(directory)
            for revision, cpp in (('before', prior), ('current', source)):
                path = temp/f'{revision}.cpp'; path.write_text(cpp + '''
extern "C" double issue19_constant(int which) {
    Model m; double c[5]={4.1e7,3.5e7,3.4e6,3e5,4.9e4};
    double s[4]={0,0,0,0}; std::vector<double> h(m.history_size,0);
    bird_reset(&m,c,s,h.data(),m.history_size,0);
    return which==0 ? m.left.params.x0 : m.left.params.D;
}
''')
                library = temp/f'{revision}.so'
                subprocess.run(['g++', '-O3', '-std=c++11', '-shared', '-fPIC', str(path), '-o', str(library)], check=True)
                probe = ctypes.CDLL(str(library)); probe.issue19_constant.restype = ctypes.c_double
                constants = [probe.issue19_constant(i) for i in (0, 1)]
                assert constants == ([.02, .01] if revision == 'current' else [.04, .01])
                assigned.append(dict(revision=revision, x0_cm=constants[0], D=constants[1]))
                model.LIBRARY = library
                for name, c, expected in examples:
                    audio = render(c)
                    difference = float(np.max(np.abs(audio-expected)))
                    records.append(dict(revision=revision, example=name, waveform_max_difference=difference,
                                        exact_equal=bool(np.array_equal(audio, expected))))
                    if difference != 0:
                        write_json(OUT/'model_gate.json', dict(passed=False, comparisons=records))
                        raise RuntimeError('Physics changed; stop without training or baseline mutation')
    finally:
        model.LIBRARY = original_library
    model.build()
    for name, c, expected in examples:
        assert np.array_equal(render(c), expected), f'Actual installed build changed {name}'
    import importlib.metadata
    base = json.loads((BASE/'manifest.json').read_text())
    settings = dict(base['settings'], rl_envelope_weights=[.5], methods=['TD3', 'SAC', 'DE'],
                    checkpoints=[], experiment_id='issue19_restricted3d_lambda0p5')
    result = dict(passed=True, assigned_constants=assigned, x0_storage_only=True,
                  comparisons=records, installed_build_exact_matches=5, generated_sounds=15,
                  source_change='Only stored x0 .04 -> .02; expanded D=.01 used by equations',
                  wall_s=time.perf_counter()-tick)
    write_json(OUT/'model_gate.json', result)
    write_json(OUT/'manifest.json', dict(issue=19, created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=git('rev-parse', 'HEAD').strip(), settings=settings,
        source_sha256=source_hashes(), native_library_sha256=digest(model.LIBRARY),
        baseline_manifest_sha256=digest(BASE/'manifest.json'), target_sha256=digest(BASE/'diagnostics/target.npz'),
        target_path='../phase1_3d_domain_diagnostics/diagnostics/target.npz', target_controls=controls.tolist(),
        model_constants=dict(x0_cm=.02, D=.01), domain_low=[2.5e7, 2.5e6, 49000.],
        domain_high=[5e7, 4e6, 800000.], versions={n:importlib.metadata.version(n) for n in base['versions']},
        device='cpu', torch_threads=1, initial_state_and_delay='all zero; original causal decimator reset',
        reward='-(E_spec + lambda E_env); one action then terminated=True, truncated=False',
        selection='first minimum J within 33 independent RL evaluations or 4096 direct candidates',
        storage='Final/Selected weights and raw audio only; all 4096 sampled points retained losslessly; no checkpoint/seed PNG',
        random_reuse='All saved lambda1 restricted-domain candidates re-scored; no pool copying or resampling',
        old_manifest_compatibility='Historical native/source hashes intentionally differ; full before/after bitwise gate in model_gate.json, original manifests unchanged'))
    print('Model gate passed: 15 full sounds exact; x0=.02 cm, D=.01 verified in actual C++ reset', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('inventory', 'model'))
    a = p.parse_args(); inventory() if a.action == 'inventory' else model_gate()
