"""Saved-data audit for the nine failed-to-voice Stage A points; native0."""
from .result_paths import source_hashes_match
import ast
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from .issue27r_data import ROOT,OUT,OLD,TARGET,METHODS,SEEDS,hashes
from .issue27r_native import CHANNELS,PROBE
from .model import LIBRARY
from .model_env import AudioDecimator
from .model_env import AudioDecimator
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest

def records():return json.loads((OUT/'preflight/candidates.json').read_text())
def load(name):
    with np.load(OUT/f'preflight/candidates/{name}.npz') as z:return {k:z[k].copy() for k in z.files}
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def audit():
    m=json.loads((OUT/'manifest.json').read_text());assert m['status']=='stopped_stageA_no_voiced_Target' and not m['preflight_passed']
    assert source_hashes_match(m['source_sha256'], hashes()) and m['native_library_sha256']==digest(LIBRARY) and m['probe_sha256']==digest(PROBE)
    assert m['target_sha256'] is None and not TARGET.exists() and not (OUT/'runs').exists()
    executed=json.loads((OUT/'executed_source_checksums.json').read_text())
    for name,h in executed.items():assert digest(OUT/'executed_sources'/name)==h
    assert digest(ROOT/'issue27r_native.py')==executed['issue27r_native.py']
    def scan_ast(path):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='scan'),include_attributes=False)
    assert scan_ast(ROOT/'issue27r_prepare.py')==scan_ast(OUT/'executed_sources/issue27r_prepare.py')
    data={r['candidate']:load(r['candidate']) for r in records()};common_peak=max(float(abs(z['audio']).max()) for z in data.values());gain=1/common_peak
    candidate_rows=[];motions=[];blocks=[];equilibria=[];audio_index=[];(OUT/'audio').mkdir(exist_ok=True)
    for r in records():
        name=r['candidate'];z=data[name];a=z['audio'];stats=z['diagnostic_moments'];points=z['diagnostic_samples'];last=stats[-1]
        assert np.isfinite(a).all() and a.shape==(10000,);np.testing.assert_array_equal(stats[:,49],np.full(20,100000))
        ratio=float(a[-1000:].std()/abs(a).max());x=np.arange(1000)/50000;detrended=a[-1000:]-np.polyval(np.polyfit(x,a[-1000:],1),x)
        assert ratio==r['late_ac_ratio'] and ratio<.05 and not r['eligible']
        assert abs(last[5])>0 and abs(last[9])>0 and stats[:,48].max()<1e-7
        np.testing.assert_array_equal(z['motion_mean'],stats[:,:4]);np.testing.assert_array_equal(z['motion_std'],stats[:,12:16])
        np.testing.assert_allclose(points[:,7],points[:,4]+points[:,5]+points[:,6],rtol=0,atol=1e-7)
        native_tail=points[-1000:,:4];tx=np.arange(1000)/50000
        state_detrended=[float(np.std(native_tail[:,j]-np.polyval(np.polyfit(tx,native_tail[:,j],1),tx))) for j in range(4)]
        candidate_rows.append(dict(candidate=name,epsilon_l=r['epsilon_l'],ps=r['ps'],f0_l=3e5,f0_r=1e7,peak=r['peak'],
            late_ac_std=float(a[-1000:].std()),late_mean=float(a[-1000:].mean()),late_ac_ratio=ratio,detrended_ac_std=float(detrended.std()),detrended_ac_ratio=r['detrended_ac_ratio'],
            sustained_proxy=r['sustained_proxy'],eligible=False,failed=False,generated_silent=r['generated_silent'],
            source_l_tail_mean=float(last[4]),source_r_tail_mean=float(last[5]),source_r_tail_std=float(last[17]),
            output_l_tail_mean=float(last[8]),output_r_tail_mean=float(last[9]),output_r_tail_std=float(last[21]),feedback_tail_mean=float(last[6]),pi_tilde_tail_mean=float(last[7]),
            decomp_max_abs_error=float(stats[:,48].max()),native_steps=2000000,raw_npz=r['raw_npz']))
        for j,channel in enumerate(CHANNELS[:4]):
            motions.append(dict(candidate=name,channel=channel,tail_window_s=.01,mean=float(last[j]),std_native=float(last[12+j]),
                minimum=float(last[24+j]),maximum=float(last[36+j]),end_value=float(z['state_end_blocks'][-1,j+2]),
                detrended_std_sampled_last20ms=state_detrended[j],note='native std includes drift; sampled detrending is a separate20ms diagnostic'))
        for block,stat in enumerate(stats):
            for j,channel in enumerate(CHANNELS):
                blocks.append(dict(candidate=name,block=block+1,start_s=block*.01,end_s=(block+1)*.01,channel=channel,
                    mean=float(stat[j]),std=float(stat[12+j]),minimum=float(stat[24+j]),maximum=float(stat[36+j])))
        c=r['native_coefficients'];al=c['alpha_l'];ar=c['alpha_r'];el=c['epsilon_l'];er=c['epsilon_r'];fl=c['f0_l'];fr=c['f0_r'];gamma=c['gamma']
        pi=-(al*fl/el+ar*fr/er)/(1+gamma-al/el-ar/er);xl=(pi-fl)/el;xr=(pi-fr)/er;k=2e4+2e8*xl*xl-c['D_l']*(c['ps_l']-pi)
        equilibria.append(dict(candidate=name,epsilon_l=el,ps=c['ps_l'],steady_pi_reference=pi,steady_x_l_reference=xl,steady_x_r_reference=xr,
            local_velocity_coefficient_fixed_pi=k,note='Static reference, not reached by .20s; not full delayed-system stability proof'))
        path=OUT/f'audio/{name}.wav';wave=(a*gain).astype(np.float32);wavfile.write(path,50000,wave)
        audio_index.append(dict(candidate=name,wav=str(path.relative_to(OUT)),raw_npz=r['raw_npz'],gain=gain,peak=float(abs(wave).max()),sample_rate=50000,samples=10000))
    old_npz=OLD/'preflight/oracle.npz'
    with np.load(old_npz) as old:
        for key in ('audio','controls','state_end_blocks','motion_mean','motion_std','diagnostic_samples','diagnostic_moments'):np.testing.assert_array_equal(data['c21'][key],old[key])
    write_csv(OUT/'candidate_summary.csv',candidate_rows);write_csv(OUT/'tail_motion.csv',motions);write_csv(OUT/'native_block_statistics.csv',blocks)
    write_csv(OUT/'equilibrium_reference.csv',equilibria);write_csv(OUT/'audio_index.csv',audio_index)
    filt=AudioDecimator();np.savez_compressed(OUT/'preflight/filter.npz',sos=filt.sos,initial_state=filt.state,order=16,cutoff_Hz=15000,factor=200,native_rate=10000000,sample_rate=50000)
    write_csv(OUT/'training_accounting.csv',[dict(method=method,seed=seed,status='not_started_no_voiced_Target',planned_steps=4096,actual_steps=0,
        planned_Actor_evaluations=33,actual_Actor_evaluations=0,optimizer_steps=0,Final_E=None,Selected_E=None,posthoc_E=None,training_best_E=None) for method in METHODS for seed in SEEDS])
    filt=AudioDecimator();np.savez_compressed(OUT/'preflight/filter.npz',sos=filt.sos,initial_state=filt.state,order=16,cutoff_Hz=15000,native_rate=10000000,decimation=200,sample_rate=50000)
    ledger=json.loads((OUT/'preflight/ledger.json').read_text());assert len(ledger)==9 and all(r['status']=='complete' for r in ledger)
    write_json(OUT/'execution.json',dict(status='stopped_stageA_no_voiced_Target',StageA_candidates=9,eligible=0,generated_sounds=9,native_steps=18000000,
        integrated_s=1.8,additional_sounds_after_grid=0,Target_selected=False,oracle_for_selected_Target='not started:no selection',
        planned_training_runs=9,actual_training_runs=0,actual_training_steps=0,actual_gradient_updates=0,method_smoke_tests=0,independent_Actor_evaluations=0,
        additional_optimization=0,no_0p10_gate=True))
    result=dict(verification_passed=True,Target_selection_gate_passed=False,candidates_checked=9,numerical_failures=0,generated_silent=sum(r['generated_silent'] for r in candidate_rows),
        sustained_candidates=0,ratio_min=min(r['late_ac_ratio'] for r in candidate_rows),ratio_max=max(r['late_ac_ratio'] for r in candidate_rows),
        all_right_source_and_output_nonzero=True,c21_exactly_matches_previous_saved_B_oracle=True,previous_result_read_only=str(old_npz.relative_to(ROOT)),
        previous_result_sha256=digest(old_npz),max_output_decomposition_error=max(r['decomp_max_abs_error'] for r in candidate_rows),
        actual_source_binary_exact=True,additional_native_sounds=0,source_sha256=digest(__file__))
    write_json(OUT/'diagnostic_audit.json',result);return result

if __name__=='__main__':audit()
