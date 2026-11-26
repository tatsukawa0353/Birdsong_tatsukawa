"""Revised Issue27, independent of the preserved failed-Target experiment."""
from .result_paths import resolve as resolve_result_path, source_hashes_match
import json
from pathlib import Path
import numpy as np
from .issue16 import digest

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/Phase1/CaseB/target_diagnostics/high_left_force'
BASE=ROOT/'results/Phase1/CaseA/stability_diagnostics'
OLD=ROOT/'results/Phase1/CaseB/target_diagnostics/initial_stop'
TARGET=OUT/'preflight/target.npz'
METHODS=('SAC','TD3','PPO');SEEDS=(0,1,2)
GRID_E=(3e7,3.5e7,4.1e7);GRID_P=(3e6,3.4e6,3.8e6)
HALF_WIDTH=np.array([12500000.,750000.,375500.])
CORE=('issue27r_data.py','issue27r_native.py','issue27r_prepare.py',
    'issue27_probe.cpp','model.py','native/model.cpp','model_env.py','acoustics.py','constant020_env.py','issue18_domain_env.py',
    'issue14.py','issue19_data.py','issue23_data.py','issue22_data.py','issue25_agent.py')
def hashes():return {name:digest(resolve_result_path(ROOT/name)) for name in CORE}
def target_data():
    require_target()
    with np.load(TARGET) as z:return z['audio'].copy(),z['controls'].copy()
def require_target():
    scan=json.loads((OUT/'preflight/scan.json').read_text())
    if not scan['ranked']:raise RuntimeError('All nine Case B candidates are non-sustained; no Target or Stage B is available')
    if not TARGET.exists():raise RuntimeError('Target not yet physically selected/frozen')
def ready(output=OUT):
    require_target()
    from .model import LIBRARY
    from .issue27r_native import PROBE
    p=Path(output);m=json.loads((p/'manifest.json').read_text())
    assert m['preflight_passed'] and source_hashes_match(m['source_sha256'], hashes())
    assert m['native_library_sha256']==digest(LIBRARY) and m['probe_sha256']==digest(PROBE)
    assert m['target_sha256']==digest(TARGET) and m['plan_sha256']==digest(OUT/'PLAN.md')
    return m
