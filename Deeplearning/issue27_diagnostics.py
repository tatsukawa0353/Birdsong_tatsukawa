"""Audit the three saved physical sounds; no new integration or learning."""
import ast
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from .issue27_data import ROOT,OUT,BASE,TARGET,CONTROLS,SEEDS,source_hashes
from .issue27_native import PROBE,CHANNELS,snapshot
from .model import BirdModel,LIBRARY
from .model_env import AudioDecimator
from .acoustics import distance
from .issue15_analysis import write_json
from .issue17_extension_data import write_csv
from .issue16 import digest

SOUNDS=('target','oracle','same_controls_caseA')

def load(name):
    with np.load(OUT/f'preflight/{name}.npz') as z:return {k:z[k].copy() for k in z.files}
def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def audit():
    m=json.loads((OUT/'manifest.json').read_text());assert m['status']=='stopped_preflight' and not m['preflight_passed']
    assert m['source_sha256']==source_hashes() and m['native_library_sha256']==digest(LIBRARY) and m['target_sha256']==digest(TARGET) and m['probe_sha256']==digest(PROBE)
    executed=json.loads((OUT/'executed_source_checksums.json').read_text())
    for name,h in executed.items():assert digest(OUT/'executed_sources'/name)==h
    for name in ('issue27_native.py','issue27_probe.cpp'):assert digest(ROOT/name)==executed[name]
    def function(path,name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    assert function(ROOT/'issue27_prepare.py','physics')==function(OUT/'executed_sources/issue27_prepare.py','physics')
    assert json.loads((OUT/'preflight/physics.json').read_text())['source_sha256']==executed['issue27_prepare.py']
    data={s:load(s) for s in SOUNDS};target=data['target'];oracle=data['oracle'];contrast=data['same_controls_caseA']
    for key in ('audio','controls','physical_controls','motion_mean','motion_std','state_end_blocks'):
        np.testing.assert_array_equal(target[key],oracle[key])
    for name,value in data.items():
        assert value['audio'].dtype==np.float64 and value['audio'].shape==(10000,)
        assert np.isfinite(value['audio']).all();np.testing.assert_array_equal(value['controls'],CONTROLS)
        np.testing.assert_array_equal(value['physical_controls'],np.tile(CONTROLS,(20,1)))
    peak=float(abs(target['audio']).max());coefficients={}
    for case in ('B','A'):
        model=BirdModel(case)
        try:model.reset(CONTROLS);coefficients[case]=snapshot(model,expect_B=case=='B',reset=True);assert model.state[0]==0
        finally:model.close()
    sound_rows=[];motion_rows=[];block_rows=[];equilibria=[]
    (OUT/'audio').mkdir(exist_ok=True)
    for name,value in data.items():
        detail=oracle if name=='target' else value;audio=value['audio'];tail=audio[-1000:];times=np.arange(1000)/50000
        residual=tail-np.polyval(np.polyfit(times,tail,1),times);stats=detail['diagnostic_moments'];last=stats[-1]
        comp=distance(target['audio'],audio,50000,scale=peak)
        own_peak=float(abs(audio).max());r=dict(sound=name,condition='A' if name=='same_controls_caseA' else 'B',
            peak=own_peak,tail_mean=float(tail.mean()),tail_std=float(tail.std()),tail_ac_ratio_self_peak=float(tail.std()/own_peak),
            tail_ac_ratio_B_Target_peak=float(tail.std()/peak),tail_detrended_std=float(residual.std()),tail_detrended_ratio_B_Target_peak=float(residual.std()/peak),
            sustained_proxy_B_Target_scale=bool(tail.std()/peak>.05),generated_silent=comp['generated_silent'],
            E_spec_to_B_Target=comp['log_spectral'],E_env_to_B_Target=comp['envelope'],E_total_to_B_Target=comp['total'],J_to_B_Target=.75*comp['log_spectral']+.25*comp['envelope'],
            source_l_mean_last10ms=float(last[4]),source_r_mean_last10ms=float(last[5]),source_l_std_last10ms=float(last[16]),source_r_std_last10ms=float(last[17]),
            feedback_mean_last10ms=float(last[6]),pi_tilde_mean_last10ms=float(last[7]),output_l_mean_last10ms=float(last[8]),output_r_mean_last10ms=float(last[9]),
            source_stats_provenance='independent identical B oracle' if name=='target' else name,
            raw_npz=f'preflight/{name}.npz',wav=f'audio/{name}.wav',shared_gain=1/peak)
        sound_rows.append(r);wave=(audio*r['shared_gain']).astype(np.float32);wavfile.write(OUT/r['wav'],50000,wave)
        for j,channel in enumerate(CHANNELS[:4]):
            motion_rows.append(dict(sound=name,channel=channel,window_s=.01,end_time_s=.2,mean=float(last[j]),std=float(last[12+j]),minimum=float(last[24+j]),maximum=float(last[36+j]),
                final_value=float(value['state_end_blocks'][-1,j+2]),note='Population std at every native step; includes drift'))
        for block,stat in enumerate(stats):
            for j,channel in enumerate(CHANNELS):
                block_rows.append(dict(sound=name,block=block+1,start_s=block*.01,end_s=(block+1)*.01,channel=channel,
                    mean=float(stat[j]),std=float(stat[12+j]),minimum=float(stat[24+j]),maximum=float(stat[36+j])))
    assert sound_rows[0]['E_total_to_B_Target']==sound_rows[1]['E_total_to_B_Target']==0
    assert not sound_rows[0]['sustained_proxy_B_Target_scale'] and sound_rows[2]['sustained_proxy_B_Target_scale']
    assert sound_rows[0]['source_r_mean_last10ms']!=0 and sound_rows[2]['source_r_mean_last10ms']==0
    assert np.max(abs(oracle['diagnostic_moments'][:,48]))<1e-7
    for name in ('oracle','same_controls_caseA'):np.testing.assert_array_equal(data[name]['diagnostic_moments'][:,49],np.full(20,100000))
    for case in ('A','B'):
        c=coefficients[case];al=c['alpha_l'];ar=c['alpha_r'];el=c['epsilon_l'];er=c['epsilon_r'];fl=c['f0_l'];fr=c['f0_r'];gamma=c['gamma']
        p=-(al*fl/el+ar*fr/er)/(1+gamma-al/el-ar/er);xl=(p-fl)/el;xr=(p-fr)/er
        k=2e4+2e8*xl*xl-c['D_l']*(c['ps_l']-p)
        equilibria.append(dict(condition=case,predicted_pi=p,predicted_x_l=xl,predicted_x_r=xr,local_velocity_coefficient_fixed_pi=k,
            source_r_at_equilibrium=ar*xr,assumptions='y=0,dy=0,constant delayed pi,A=1; not proof of delayed-system stability'))
    a_reference=[]
    for seed in SEEDS:
        s=json.loads((BASE/f'runs/seed_{seed}/summary.json').read_text());r=s['final']
        a_reference.append(dict(seed=seed,model='A',Target='A separate Target',E_total=r['acoustic_error'],E_spec=r['acoustic_components']['log_spectral'],
            E_env=r['acoustic_components']['envelope'],J=r['weighted_loss'],voiced=r['sustained_voiced'],source=str((BASE/f'runs/seed_{seed}/summary.json').relative_to(ROOT))))
    write_csv(OUT/'sound_metrics.csv',sound_rows);write_csv(OUT/'tail_motion.csv',motion_rows);write_csv(OUT/'native_block_statistics.csv',block_rows)
    write_csv(OUT/'equilibrium_reference.csv',equilibria);write_csv(OUT/'caseA_reference.csv',a_reference)
    write_csv(OUT/'training_accounting.csv',[dict(seed=s,planned_candidates=4096,actual_candidates=0,planned_evaluations=33,actual_evaluations=0,gradient_updates=0,
        status='not_started_target_gate_failed',Final_E=None,Selected_E=None,posthoc_E=None,training_best_E=None) for s in SEEDS])
    filt=AudioDecimator();np.savez_compressed(OUT/'preflight/filter.npz',sos=filt.sos,initial_state=filt.state,factor=200,native_rate=10000000,sample_rate=50000,cutoff_Hz=15000,order=16)
    ledger=json.loads((OUT/'preflight/ledger.json').read_text());assert len(ledger)==3 and all(x['status']=='complete' for x in ledger)
    write_json(OUT/'execution.json',dict(status='stopped_preflight',planned_training_runs=3,actual_training_runs=0,not_started_runs=3,
        training_candidates=0,independent_Actor_evaluations=0,gradient_updates=0,preflight_sounds=3,preflight_native_steps=6000000,preflight_integrated_s=.6,
        additional_diagnostic_sounds=0,additional_audit_sounds=0,Target_generated_by='BirdModel(B)',no_0p10_gate=True))
    record=dict(verification_passed=True,physical_Target_gate_passed=False,actual_coefficients=coefficients,
        oracle_audio_exact=True,oracle_states_and_native_moments_exact=True,oracle_E_total=0.,oracle_J=0.,same_controls_A_B_different=True,
        source_right_nonzero_B=True,source_right_zero_A=True,actual_library_binary_matches_current_source=True,
        right_x_std_last10ms=float(oracle['diagnostic_moments'][-1,14]),right_y_std_last10ms=float(oracle['diagnostic_moments'][-1,15]),
        B_Target_late_ac_ratio=sound_rows[0]['tail_ac_ratio_self_peak'],A_contrast_late_ac_ratio_self=sound_rows[2]['tail_ac_ratio_self_peak'],
        A_contrast_late_ac_ratio_shared=sound_rows[2]['tail_ac_ratio_B_Target_peak'],required_ratio=.05,
        output_decomposition_max_error=float(oracle['diagnostic_moments'][:,48].max()),
        source_sha256=digest(__file__),new_sounds=0,training_started=False)
    write_json(OUT/'diagnostic_audit.json',record);return record

if __name__=='__main__':audit()
