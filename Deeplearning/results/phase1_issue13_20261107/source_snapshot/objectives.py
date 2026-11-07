"""Target-conditioned acoustic loss; the original acoustics.distance is unchanged.

All features are causal. STFT frames overlap physically but each frame is assigned
to one reward update. Frame weights are their newly covered sample durations.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .acoustics import waveform


@dataclass(frozen=True)
class BalancedConfig:
    window_s: float = .024576
    overlap: float = .75
    envelope_s: float = .005
    min_hz: float = 250.0
    max_hz: float = 10000.0
    floor_db: float = -80.0
    voiced_ratio: float = .01
    transport_weight: float = 1.0
    log_weight: float = .25
    envelope_weight: float = 1.0
    missing_weight: float = 2.0
    dc_weight: float = .1

    def __post_init__(self):
        values = list(self.__dict__.values())
        if not np.isfinite(values).all():
            raise ValueError("Loss settings must be finite")
        if not (self.window_s > 0 and 0 <= self.overlap < 1 and self.envelope_s > 0
                and 0 < self.min_hz < self.max_hz and self.floor_db < 0
                and 0 < self.voiced_ratio < 1):
            raise ValueError("Invalid temporal or spectral settings")
        if min(self.transport_weight, self.log_weight, self.envelope_weight,
               self.missing_weight, self.dc_weight) < 0:
            raise ValueError("Loss weights cannot be negative")


class BalancedObjective:
    names = ("transport", "target_weighted_log", "band_envelope", "missing", "dc", "total")

    def __init__(self, target, rate, *, scale=None, config=BalancedConfig()):
        self.target = waveform(target).copy()
        self.rate, self.config = rate, config
        if not np.isfinite(rate) or rate <= 2*config.max_hz:
            raise ValueError("Sample rate must exceed twice the maximum analysis frequency")
        peak = float(np.max(abs(self.target)))
        self.scale = (peak if peak > 1e-8 else 1.0) if scale is None else float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("A single finite positive shared scale is required")
        self.n = max(4, round(config.window_s*rate))
        self.hop = max(1, round(self.n*(1-config.overlap)))
        self.ends = np.arange(self.hop, len(self.target)+1, self.hop, dtype=int)
        if not len(self.ends) or self.ends[-1] != len(self.target):
            self.ends = np.r_[self.ends, len(self.target)]
        self.starts = np.r_[0, self.ends[:-1]]
        self.weights = (self.ends-self.starts)/len(self.target)
        self.window = signal.get_window("blackmanharris", self.n, fftbins=True)
        freq = np.fft.rfftfreq(self.n, 1/rate)
        self.keep = (freq >= config.min_hz) & (freq <= config.max_hz)
        self.sos = signal.butter(6, [config.min_hz, config.max_hz], btype="bandpass", fs=rate, output="sos")
        self.zi = signal.sosfilt_zi(self.sos)
        self.target_features = self._features(self.target, len(self.ends))
        self.energy_reference = max(float(np.max(self.target_features["envelope"])), .05)
        maximum = float(np.max(self.target_features["envelope"]))
        self.voiced = ((self.target_features["envelope"] >= config.voiced_ratio*maximum)
                       & (maximum > 1e-8))
        self.reset()

    def reset(self):
        self.cursor = 0
        self.parts = np.zeros((len(self.ends), len(self.names)), dtype=float)

    def _features(self, audio, count):
        normalized = waveform(audio)/self.scale
        band, _ = signal.sosfilt(self.sos, normalized, zi=self.zi*normalized[0])
        padded = np.pad(band, (self.n-1, 0))
        frames = np.lib.stride_tricks.sliding_window_view(padded, self.n)[self.ends[:count]-1]
        transform = np.fft.rfft(frames*self.window, axis=1)/self.window.sum()
        power = (2*abs(transform)**2)[:, self.keep]
        width = max(1, round(self.config.envelope_s*self.rate))
        accum = np.r_[0.0, np.cumsum(band*band)]
        indices = np.arange(1, len(band)+1)
        envelope = np.sqrt(np.maximum((accum[indices]-accum[np.maximum(indices-width, 0)])/width, 0))
        env_sum = np.r_[0.0, np.cumsum(envelope)]
        raw_sum = np.r_[0.0, np.cumsum(normalized)]
        ends, starts = self.ends[:count], self.starts[:count]
        return dict(power=power,
                    envelope=(env_sum[ends]-env_sum[starts])/(ends-starts),
                    dc=(raw_sum[ends]-raw_sum[starts])/(ends-starts))

    def frame_parts(self, generated):
        generated = waveform(generated)
        if len(generated) > len(self.target):
            raise ValueError("Generated prefix is longer than the target")
        count = int(np.searchsorted(self.ends, len(generated), side="right"))
        if not count:
            return np.zeros((0, len(self.names)))
        features = self._features(generated, count)
        pa = self.target_features["power"][:count]
        pb = features["power"]
        ma, mb = pa.sum(axis=1, keepdims=True), pb.sum(axis=1, keepdims=True)
        qa = np.divide(pa, ma, out=np.zeros_like(pa), where=ma > 1e-20)
        qb = np.divide(pb, mb, out=np.zeros_like(pb), where=mb > 1e-20)
        voiced = self.voiced[:count].astype(float)
        transport = np.mean(abs(np.cumsum(qa-qb, axis=1)), axis=1)*voiced
        floor = 10**(self.config.floor_db/10)
        log_distance = np.sum(qa*abs(10*np.log10(pa+floor)-10*np.log10(pb+floor)), axis=1)
        log_distance = log_distance/abs(self.config.floor_db)*voiced
        et, eg = self.target_features["envelope"][:count], features["envelope"]
        envelope = abs(et-eg)/self.energy_reference
        missing = voiced*np.maximum(1-eg/np.maximum(et, 1e-12), 0)**2
        dc = abs(self.target_features["dc"][:count]-features["dc"])
        c = self.config
        total = (c.transport_weight*transport+c.log_weight*log_distance+c.envelope_weight*envelope
                 +c.missing_weight*missing+c.dc_weight*dc)
        return np.column_stack((transport, log_distance, envelope, missing, dc, total))

    def score(self, generated):
        if len(generated) != len(self.target):
            raise ValueError("Whole-episode scoring requires equal waveform lengths")
        parts = self.frame_parts(generated)
        values = np.sum(parts*self.weights[:, None], axis=0)
        return {name: float(v) for name, v in zip(self.names, values)}

    def update(self, generated, *, context_s=.03):
        """Assign each newly completed frame once; filtering always uses the prefix."""
        parts = self.frame_parts(generated)
        count = len(parts)
        if count < self.cursor:
            raise ValueError("Prefix shortened; reset the objective before a new episode")
        previous = self.cursor
        self.parts[previous:count] = parts[previous:count]
        self.cursor = count
        contribution = np.sum(self.parts[previous:count]*self.weights[previous:count, None], axis=0)
        current = np.sum(self.parts[:count]*self.weights[:count, None], axis=0)
        recent = self.ends[:count] > len(generated)-round(context_s*self.rate)
        w = self.weights[:count][recent]
        rolling = (np.sum(self.parts[:count][recent]*w[:, None], axis=0)/w.sum()
                   if w.size else np.zeros(len(self.names)))
        return dict(contribution=dict(zip(self.names, map(float, contribution))),
                    cumulative=dict(zip(self.names, map(float, current))),
                    rolling=dict(zip(self.names, map(float, rolling))),
                    frames_added=count-previous, frames_seen=count)
