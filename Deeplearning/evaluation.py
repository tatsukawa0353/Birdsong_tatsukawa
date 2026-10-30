"""Post-training descriptors, not reward terms or automatic noise rejection.

SCI follows the repository's f_MSF/f_AFF definition, not a certified SAP export.
HNR here is an autocorrelation diagnostic; it is not validated for noise-like calls.
"""
import numpy as np
from scipy import signal

from .acoustics import AcousticConfig, spectrum, waveform


def descriptors(x, rate):
    x = waveform(x)
    f, _, p = spectrum(x-np.mean(x), rate, AcousticConfig(max_hz=12000))
    band = (f >= 250) & (f <= 12000)
    p, f = p[band], f[band]
    mass = p.mean(axis=1)
    power = float(mass.sum())
    if power <= 1e-30:
        return dict(se=None, repository_sci=None, f0_hz=None, centroid_hz=None,
                    aci=None, autocorrelation_hnr_db=None, periodic_frames=0,
                    status="silent")
    q = mass/power
    centroid = float(np.sum(q*f))
    entropy = float(-np.sum(q*np.log(q+1e-30))/np.log(len(q)))
    sums = p.sum(axis=1)
    active = sums > float(sums.max())*.01
    aci = float(np.sum(np.abs(np.diff(p, axis=1)).sum(axis=1)[active]/sums[active]))
    n, hop = round(.02*rate), round(.01*rate)
    minimum, maximum = max(1, int(rate/3500)), int(rate/250)
    candidates, correlations = [], []
    for i in range(0, len(x)-n+1, hop):
        w = x[i:i+n]-np.mean(x[i:i+n])
        ac = signal.correlate(w, w, mode="full", method="fft")[n-1:]
        if ac[0] <= 1e-30 or maximum >= len(ac):
            continue
        peaks, _ = signal.find_peaks(ac[minimum:maximum])
        if not len(peaks):
            continue
        lag = minimum+int(peaks[np.argmax(ac[minimum+peaks])])
        r = float(np.clip(ac[lag]/ac[0], 0, 1-1e-12))
        correlations.append(r)
        if r >= .6:
            candidates.append(rate/lag)
    f0 = float(np.median(candidates)) if candidates else None
    r = float(np.median(correlations)) if correlations else None
    hnr = float(10*np.log10(max(r, 1e-12)/max(1-r, 1e-12))) if r is not None else None
    return dict(se=entropy, repository_sci=centroid/f0 if f0 else None,
                f0_hz=f0, centroid_hz=centroid, aci=aci,
                autocorrelation_hnr_db=hnr, periodic_frames=len(candidates),
                status="periodic_estimate" if f0 else "no_reliable_f0")
