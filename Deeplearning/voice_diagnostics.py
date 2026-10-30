"""Separate audio amplitude, source motion, persistence, and display visibility.

Thresholds are operational diagnostics relative to a known voiced reference,
not perceptual silence or proof of a physical open/closed labial configuration.
"""
import numpy as np
from scipy import signal

from .acoustics import waveform
from .evaluation import descriptors
from .model import BirdModel
from .model_env import AudioDecimator

RATE = 50000
BAND_LOW = 250
BAND_HIGH = 10000
RELATIVE_THRESHOLD_DB = -60.0
VELOCITY_STD_THRESHOLD = 1.0


def band_audio(audio, rate=RATE):
    sos = signal.butter(6, [BAND_LOW, BAND_HIGH], btype="bandpass", fs=rate, output="sos")
    return signal.sosfilt(sos, waveform(audio))


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x*x))) if x.size else 0.0


def replay(controls, condition, *, output_gain=1.0, control_s=.01):
    """Full-rate moments; do not infer motion from 100 Hz endpoint samples."""
    audio, means, stds, applied, states, raw_stats = [], [], [], [], [], []
    decimator = AudioDecimator()
    with BirdModel(condition) as model:
        for control in controls:
            model.set_controls(control)
            raw, moments = model.advance(control_s, return_moments=True)
            audio.append(decimator.process(raw)/output_gain)
            means.append(moments["mean"])
            stds.append(moments["std"])
            applied.append(model.controls.copy())
            states.append(model.state.copy())
            raw_stats.append([float(raw.mean()), float(raw.std()), rms(raw), float(np.max(abs(raw)))])
    return dict(audio=np.concatenate(audio), motion_mean=np.array(means), motion_std=np.array(stds),
                physical_controls=np.array(applied), states=np.array(states), raw_pressure_stats=np.array(raw_stats))


def pooled_motion(means, stds):
    """Equal-length integration blocks; include between-block displacement."""
    means, stds = np.asarray(means), np.asarray(stds)
    average = means.mean(axis=0)
    variance = np.mean(stds**2+(means-average)**2, axis=0)
    return average, np.sqrt(np.maximum(variance, 0))


def summarize(audio, *, reference_band_rms, condition, motion_mean=None, motion_std=None,
              rate=RATE, control_s=.01, error=None, inspect_display=False, numerical_failure=False):
    audio = waveform(audio)
    filtered = band_audio(audio, rate)
    threshold = reference_band_rms*10**(RELATIVE_THRESHOLD_DB/20)
    if not np.isfinite(reference_band_rms) or reference_band_rms <= 0:
        raise ValueError("Reference band RMS must be finite and positive")
    intervals = {}
    for name, start, end in [("initial_0_05", 0, .05), ("transition_05_10", .05, .1),
                             ("middle_10_20", .1, .2), ("late_20_35", .2, .35),
                             ("steady_10_35", .1, .35)]:
        a, b = round(start*rate), min(len(audio), round(end*rate))
        if b <= a:
            continue
        x, band = audio[a:b], filtered[a:b]
        intervals[name] = dict(rms=rms(x), ac_rms=float(np.std(x)), mean=float(np.mean(x)),
                               peak=float(np.max(abs(x))), band_rms=rms(band))
    tail = filtered[min(round(.1*rate), len(audio)-1):]
    n = max(1, round(.02*rate))
    blocks = [tail[i:i+n] for i in range(0, len(tail), n) if len(tail[i:i+n]) >= n//2]
    duty = float(np.mean([rms(b) >= threshold for b in blocks])) if blocks else 0.0
    band_rms = rms(tail)
    sustained = bool(band_rms >= threshold and duty >= .8 and not numerical_failure)
    features = descriptors(audio[min(round(.1*rate), len(audio)-1):], rate)
    if numerical_failure:
        state = "numerical_failure"
    elif sustained:
        state = "sustained_periodic_like" if features["f0_hz"] is not None else "sustained_noise_like"
    else:
        state = "transient_or_intermittent" if band_rms >= threshold else "below_reference_threshold"
    result = dict(rms=rms(audio), peak=float(np.max(abs(audio))), exactly_zero=bool(np.all(audio == 0)),
                  near_zero_absolute=bool(np.max(abs(audio)) < 1e-10), intervals=intervals,
                  late_band_relative_db=float(20*np.log10(max(band_rms, 1e-30)/reference_band_rms)),
                  active_20ms_fraction=duty, sustained_acoustic=sustained, acoustic_state=state,
                  descriptors=features, numerical_failure=bool(numerical_failure))
    if error is not None:
        result["acoustic_error"] = float(error)
        result["above_operational_error_threshold"] = bool(error > .08)
    if motion_mean is not None and len(motion_mean):
        first = min(round(.1/control_s), len(motion_mean)-1)
        average, std = pooled_motion(np.asarray(motion_mean)[first:], np.asarray(motion_std)[first:])
        result["motion"] = dict(mean=average.tolist(), std=std.tolist(),
                                rms=np.hypot(average, std).tolist(),
                                left_velocity_active=bool(std[1] >= VELOCITY_STD_THRESHOLD),
                                right_velocity_active=bool(std[3] >= VELOCITY_STD_THRESHOLD))
        contributing = std[[1]] if condition == "A" else std[[1, 3]]
        result["contributing_source_motion"] = bool(np.any(contributing >= VELOCITY_STD_THRESHOLD))
    if inspect_display:
        from .plotting import display_spectrum, VMIN
        f, t, db = display_spectrum(audio, rate)
        late = t >= .1
        band = f >= BAND_LOW
        visible = np.any(db[band][:, late] >= VMIN, axis=0)
        result["late_visible_frame_fraction"] = float(np.mean(visible)) if visible.size else 0.0
        result["display_mostly_white"] = bool(result["late_visible_frame_fraction"] < .1)
    return result
