"""Pretraining target, voicing, sensitivity and Oracle gates for Issue 17."""
import time
import numpy as np
from scipy.io import wavfile

from .acoustics import distance
from .constant020_env import Constant020Env, INDICES, RATE, episode, render
from .model import LOW, HIGH
from .issue15_analysis import write_json


def prepare_diagnostics(folder):
    folder.mkdir()
    started = time.perf_counter()
    controls = np.array([4.1e7, 3.5e7, 3.4e6, 300000., 49000.])
    scans = []
    render_count, render_blocks = 0, 0

    def generate(c, blocks=20):
        nonlocal render_count, render_blocks
        render_count += 1
        render_blocks += blocks
        return render(c, blocks)

    # Fixed physical survey; no optimizers/learning or performance-based tuning.
    for f0 in (49000., 100000., 300000., 600000., 1000000., 2000000., 3000000., 5024500., 8000000.):
        c = controls.copy(); c[3] = f0
        y = generate(c)
        scans.append(dict(f0_l=f0, peak=float(abs(y).max()),
                          late_ac_rms=float(np.std(y[5000:])), final_20ms_ac_rms=float(np.std(y[-1000:]))))
    target = generate(controls)
    longer = generate(controls,35)
    scale = float(abs(target).max())
    sensitivity, columns = [], []
    for i in (0,2,3):
        h = .001*(HIGH[i]-LOW[i])
        plus, minus = controls.copy(), controls.copy()
        plus[i] += h; minus[i] -= h
        a, b = generate(plus), generate(minus)
        column = (a-b)[5000:]
        columns.append(column/np.linalg.norm(column))
        sensitivity.append(dict(index=i, physical_delta=float(h),
            plus_error=distance(target,a,RATE)['total'], minus_error=distance(target,b,RATE)['total'],
            late_normalized_waveform_difference=float(np.linalg.norm(column)/np.sqrt(len(column))/scale),
            plus_late_ac_rms=float(np.std(a[-1000:])), minus_late_ac_rms=float(np.std(b[-1000:]))))
    singular = np.linalg.svd(np.asarray(columns).T, compute_uv=False)
    right = []
    for i in (1,4):
        c = controls.copy(); c[i] = HIGH[i]
        y = generate(c)
        right.append(dict(index=i, waveform_bitwise_equal=bool(np.array_equal(y,target)),
                          acoustic_error=distance(target,y,RATE)['total']))
    oracles = []
    for condition, indices in INDICES.items():
        env = Constant020Env(indices, target, controls)
        try:
            for dtype in (np.float64,np.float32):
                metrics, values = episode(env, action=env.encode(controls,dtype))
                oracles.append(dict(condition=condition, action_dtype=np.dtype(dtype).name,
                    error=metrics['acoustic_error'], max_waveform_abs_error=float(np.max(abs(values['audio']-target))),
                    max_waveform_relative_error=float(np.max(abs(values['audio']-target))/scale),
                    physical_controls=values['physical_controls'][0].tolist()))
            # A prior different trial must leave no state/filter/delay behind.
            episode(env, action=np.zeros(len(indices)))
            metrics, values = episode(env, action=env.encode(controls,np.float64))
            assert metrics['acoustic_error'] < 1e-9
            render_count += env.completed_episodes
            render_blocks += env.native_blocks
        finally:
            env.close()
    voiced = np.std(target[-1000:])/scale > .05
    interior = bool(LOW[3] < controls[3] < HIGH[3])
    sensitive = all(r['plus_error'] > 1e-5 and r['minus_error'] > 1e-5 for r in sensitivity)
    nondegenerate = singular[-1] > .01
    oracle_ok = all(o['error'] < (1e-9 if o['action_dtype']=='float64' else 1e-4) for o in oracles)
    passed = bool(voiced and interior and sensitive and nondegenerate and oracle_ok and
                  all(r['waveform_bitwise_equal'] for r in right) and np.array_equal(longer[:10000],target))
    review = dict(passed=passed, proceed_3d=passed, controls=controls.tolist(),
        target_samples=len(target), target_rate=RATE, target_duration_s=.2,
        f0_selection='300000: strictly interior, sustained voicing at neighboring values; fixed before optimizer runs',
        f0_fraction_of_range=float((controls[3]-LOW[3])/(HIGH[3]-LOW[3])),
        survey=scans, sensitivity=sensitivity, normalized_late_waveform_jacobian_singular_values=singular.tolist(),
        interpretation='Local output sensitivity/rank only; does not prove unique global acoustic identification',
        inactive_right_controls=right, oracles=oracles, sustained_voicing=bool(voiced),
        target_peak=scale, listening_gain=1/scale, final_20ms_ac_rms=float(np.std(target[-1000:])),
        first_20ms_ac_rms=float(np.std(target[:1000])),
        same_control_035_prefix_bitwise_equal=bool(np.array_equal(longer[:10000],target)),
        duration_note='Fresh 0.20 s generation; same-control 0.35 s prefix is identical by causality. Objective/frame count differs; historical target f0_l=49000 is different.',
        setup_generated_sounds=render_count, setup_native_blocks=render_blocks,
        setup_native_steps=render_blocks*100000, setup_integrated_s=render_blocks*.01,
        setup_wall_s=time.perf_counter()-started)
    np.savez_compressed(folder/'target.npz', target=target, controls=controls, rate=RATE, duration_s=.2, listening_gain=1/scale)
    wavfile.write(folder/'target.wav', RATE, (target/scale).astype(np.float32))
    write_json(folder/'review.json',review)
    # Rendering is separate from the measured physical diagnostics.
    tick = time.perf_counter()
    from .plotting import single_spectrogram
    single_spectrogram(target,RATE,folder/'target_spectrogram.png')
    import matplotlib.pyplot as plt
    ac=np.std(target.reshape(20,500)/scale,axis=1)
    for name,x,y,label in (('target_waveform',np.arange(10000)/RATE,target/scale,'Waveform / target peak'),
                           ('target_transient',(np.arange(20)+.5)*.01,ac,'AC RMS / target peak')):
        fig,ax=plt.subplots(figsize=(14,9))
        ax.plot(x,y,lw=2.8)
        ax.set_xlim(0,.2);ax.set_xlabel('Time [s]',fontsize=46);ax.set_ylabel(label,fontsize=46)
        ax.tick_params(labelsize=30)
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=120,bbox_inches='tight',pad_inches=.05);plt.close(fig)
    review['diagnostic_plot_wall_s']=time.perf_counter()-tick
    write_json(folder/'review.json',review)
    if not passed:
        raise RuntimeError('Target/sensitivity/Oracle gate failed; retain diagnostics and do not train')
    return review
