"""Observe the original loaded native model without modifying its source/library."""
import ctypes as ct
from pathlib import Path
import subprocess
import numpy as np
from .model import BirdModel,LIBRARY,DT
from .model_env import AudioDecimator
from .issue27r_data import ROOT,OUT
from .issue16 import digest
from .issue15_analysis import write_json

PROBE=OUT/'native/libissue27r_probe.so'
COEFFICIENTS=('alpha_l','beta_l','alpha_r','beta_r','f0_l','f0_r','epsilon_l','epsilon_r','ps_l','ps_r',
    'tau_l','tau_r','x0_l','x0_r','D_l','D_r','dt','gamma','history_size','history_max_abs')
CHANNELS=('x_l','y_l','x_r','y_r','source_l','source_r','feedback','pi_tilde','output_l','output_r','delayed_pi','pi')
_probe=None

def build_probe():
    PROBE.parent.mkdir(parents=True,exist_ok=True)
    if PROBE.exists():raise FileExistsError('Existing diagnostic build preserved')
    commands=[['g++','-O3','-std=c++11','-fPIC','-shared',str(ROOT/'issue27_probe.cpp'),'-o',str(PROBE)],
        ['g++','-O3','-std=c++11','-fPIC','-shared',str(ROOT/'native/model.cpp'),'-o',str(PROBE.parent/'libsource_reference.so')]]
    for command in commands:subprocess.run(command,check=True)
    result=dict(commands=commands,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        native_source_sha256=digest(ROOT/'native/model.cpp'),loaded_library_sha256=digest(LIBRARY),probe_source_sha256=digest(ROOT/'issue27_probe.cpp'),
        probe_sha256=digest(PROBE),reference_library_sha256=digest(PROBE.parent/'libsource_reference.so'),
        unchanged_source_build_binary_exact=digest(LIBRARY)==digest(PROBE.parent/'libsource_reference.so'),
        actual_advance='Function pointer to BirdModel.lib.bird_advance; actual model memory, read-only observations')
    write_json(OUT/'native/build.json',result);return result

def probe():
    global _probe
    if _probe is None:
        _probe=ct.CDLL(str(PROBE));pointer=ct.POINTER(ct.c_double)
        _probe.issue27_read.argtypes=[ct.c_void_p,pointer]
        _probe.issue27_diagnostic_advance.argtypes=[ct.c_void_p,ct.c_int,ct.c_void_p,pointer,pointer,pointer,ct.c_int]
        _probe.issue27_diagnostic_advance.restype=ct.c_int
    return _probe

def snapshot(model,*,expect_B=True,reset=False):
    values=np.empty(30);probe().issue27_read(model.ptr,model._pointer(values))
    assert np.isfinite(values).all();np.testing.assert_array_equal(values[20:26],model.state)
    result={name:float(value) for name,value in zip(COEFFICIENTS,values[:20])}
    result.update(state=values[20:26].tolist(),source_l=float(values[26]),source_r=float(values[27]),next_delayed_pi=float(values[28]))
    expected=(1.2e6,1500.,1.2e6 if expect_B else 0.,1500. if expect_B else 0.)
    np.testing.assert_array_equal(values[:4],expected)
    assert model.condition==('B' if expect_B else 'A')
    np.testing.assert_array_equal(values[[4,5,6,7,8]],[model.controls[3],model.controls[4],model.controls[0],model.controls[1],model.controls[2]])
    assert values[8]==values[9] and values[12]==values[13]==.02 and values[14]==values[15]==.01 and values[16]==DT
    if expect_B:assert values[5]==1e7
    if reset:assert np.max(abs(values[20:26]))==0 and values[19]==0 and values[28]==0
    return result

def render(controls,condition='B',*,detailed=False,library=None):
    # library override is used only for an unchanged-source parity diagnostic.
    import Deeplearning.model as bridge
    original=bridge.LIBRARY
    if library is not None:bridge.LIBRARY=Path(library)
    try:model=BirdModel(condition)
    finally:bridge.LIBRARY=original
    try:
        model.reset(controls);initial=snapshot(model,expect_B=condition=='B',reset=True);decimator=AudioDecimator();audio=[];means=[];stds=[];states=[];contributions=[];samples=[];moments=[]
        for _ in range(20):
            if detailed:
                raw=np.empty(100000);stats=np.zeros(50);points=np.empty((500,12))
                rc=probe().issue27_diagnostic_advance(model.ptr,100000,ct.cast(model.lib.bird_advance,ct.c_void_p),model._pointer(raw),model._pointer(stats),model._pointer(points),200)
                if rc:raise FloatingPointError(f'Diagnostic native failure:{rc},completed={stats[49]}')
                moments.append(stats.copy());samples.append(points.copy());means.append(stats[:4].copy());stds.append(stats[12:16].copy())
            else:
                raw,stats=model.advance(.01,return_moments=True);means.append(stats['mean']);stds.append(stats['std'])
            audio.append(decimator.process(raw));s=snapshot(model,expect_B=condition=='B');states.append(s['state']);contributions.append([s['source_l'],s['source_r']])
        values=dict(audio=np.concatenate(audio),controls=model.controls.copy(),physical_controls=np.tile(model.controls,(20,1)),
            state_end_blocks=np.array(states),motion_mean=np.array(means),motion_std=np.array(stds),source_end_blocks=np.array(contributions))
        if detailed:values.update(diagnostic_samples=np.concatenate(samples),diagnostic_moments=np.array(moments),diagnostic_channels=np.array(CHANNELS),diagnostic_sample_rate=50000)
        return values,initial
    finally:model.close()
