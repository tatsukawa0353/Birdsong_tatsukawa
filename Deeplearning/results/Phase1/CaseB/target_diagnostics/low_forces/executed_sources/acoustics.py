"""Time-aligned acoustic losses; display normalization is deliberately separate."""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.io import wavfile


def waveform(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite mono waveform")
    return x


@dataclass(frozen=True)
class AcousticConfig:
    window_s: float = 0.024576
    overlap: float = 0.75
    max_hz: float = 10000.0
    floor_db: float = -80.0
    envelope_weight: float = 1.0

    def __post_init__(self):
        if not (self.window_s > 0 and 0 <= self.overlap < 1 and
                self.max_hz > 0 and self.floor_db < 0 and self.envelope_weight >= 0):
            raise ValueError("Invalid acoustic settings")


def spectrum(x, sample_rate, config=AcousticConfig()):
    """Blackman-Harris PSD; short clips are zero padded, never time stretched."""
    x = waveform(x)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive Hz")
    n = max(4, round(config.window_s * sample_rate))
    hop = max(1, round(n * (1 - config.overlap)))
    # Centered frames provide matching frame times even for short step rewards.
    x = np.pad(x, (n // 2, n))
    f, t, p = signal.spectrogram(x, fs=sample_rate, window="blackmanharris",
                               nperseg=n, noverlap=n-hop, detrend=False,
                               scaling="spectrum", mode="psd")
    t -= n / (2 * sample_rate)
    keep = f <= min(config.max_hz, sample_rate / 2)
    return f[keep], t, p[keep]


def distance(target, generated, sample_rate, config=AcousticConfig(), *, scale=None):
    """No independent gain/phase alignment. Both inputs share one physical scale."""
    target, generated = waveform(target), waveform(generated)
    if target.shape != generated.shape:
        raise ValueError("Waveforms must have equal length and sample rate")
    if scale is None:
        scale = max(float(np.max(np.abs(target))), 1e-8)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    a, b = target / scale, generated / scale
    _, _, pa = spectrum(a, sample_rate, config)
    _, _, pb = spectrum(b, sample_rate, config)
    floor = 10 ** (config.floor_db / 10)
    da, db = 10*np.log10(pa + floor), 10*np.log10(pb + floor)
    log_error = float(np.mean(np.abs(da-db)) / abs(config.floor_db))
    width = min(len(a), max(1, round(0.005 * sample_rate)))
    kernel = np.ones(width) / width
    ea = np.sqrt(np.maximum(np.convolve(a*a, kernel, mode="same"), 0))
    eb = np.sqrt(np.maximum(np.convolve(b*b, kernel, mode="same"), 0))
    envelope_error = float(np.mean(np.abs(ea-eb)))
    return dict(log_spectral=log_error, envelope=envelope_error,
                total=log_error + config.envelope_weight*envelope_error,
                target_silent=bool(np.max(np.abs(target)) < 1e-10),
                generated_silent=bool(np.max(np.abs(generated)) < 1e-10))


def resample(x, source_rate, target_rate):
    x = waveform(x)
    if min(source_rate, target_rate) <= 0:
        raise ValueError("Sample rates must be positive")
    ratio = Fraction(float(target_rate) / float(source_rate)).limit_denominator(100000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def load_local_wav(path, *, expected_duration=0.35, target_rate=48000, return_metadata=False):
    """Explicit local input only. Never copies input or includes its path in logs."""
    rate, data = wavfile.read(path)
    if data.ndim != 1:
        raise ValueError("Select a mono channel explicitly before loading")
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype == np.uint8:
            data = (data.astype(float) - 128) / 128
        else:
            data = data.astype(float) / (np.iinfo(data.dtype).max + 1)
    data = waveform(data)
    selected = round(expected_duration*rate)
    if len(data) < selected or abs(len(data)/rate - expected_duration) > .001:
        raise ValueError("Input must already contain the selected duration")
    # Audio editors can leave sub-ms excess. Keep the first exact interval; no stretching.
    metadata = dict(source_rate=int(rate), source_samples=len(data), selected_samples=selected,
                    start_sample=0, trimmed_end_samples=len(data)-selected,
                    selected_duration_s=selected/rate, output_rate=target_rate)
    result = resample(data[:selected], rate, target_rate)
    return (result, metadata) if return_metadata else result


def evaluation_features(x, sample_rate):
    """Descriptive SE/centroid only; neither is labelled complexity or SAP SCI."""
    f, _, p = spectrum(x, sample_rate)
    mass = np.mean(p, axis=1)
    if mass.sum() <= 1e-30:
        return dict(spectral_entropy=None, centroid_hz=None, rms=0.0)
    q = mass / mass.sum()
    return dict(spectral_entropy=float(-np.sum(q*np.log(q+1e-30))/np.log(len(q))),
                centroid_hz=float(np.sum(f*q)), rms=float(np.sqrt(np.mean(waveform(x)**2))))
