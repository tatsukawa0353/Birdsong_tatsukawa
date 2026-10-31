"""Recording-independent analysis of fixed waveforms; no learning reward API.

All variants are post-hoc scores. The original objective, coefficients and
duration weights are preserved and are never patched or installed in an env.
"""
import numpy as np
from scipy import signal
from scipy.stats import rankdata

from .acoustics import waveform
from .model import LOW, HIGH
from .objectives import BalancedObjective

TERMS = BalancedObjective.names[:-1]


def coefficients(config):
    return np.array([config.transport_weight, config.log_weight, config.envelope_weight,
                     config.missing_weight, config.dc_weight])


def target_mask(obj, ratio):
    if not np.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("Activity ratio must be in (0,1)")
    energy = obj.target_features["envelope"]
    maximum = float(energy.max())
    return (energy >= ratio*maximum) & (maximum > 1e-8)


def temporal_regions(obj, *, ratio=.1, boundary_radius_s=.012288):
    """Disjoint diagnostic body/weak/boundary regions, never ground-truth labels."""
    if not np.isfinite(boundary_radius_s) or boundary_radius_s < 0:
        raise ValueError("Boundary radius must be nonnegative seconds")
    active = target_mask(obj, ratio)
    times = obj.ends/obj.rate
    changes = times[1:][active[1:] != active[:-1]]
    boundary = (np.any(abs(times[:, None]-changes[None, :]) <= boundary_radius_s, axis=1)
                if len(changes) else np.zeros(len(times), dtype=bool))
    labels = np.where(boundary, "boundary", np.where(active, "body", "weak"))
    return labels, changes


def shape_distribution(power):
    power = np.asarray(power, dtype=float)
    mass = power.sum(axis=1, keepdims=True)
    return np.divide(power, mass, out=np.zeros_like(power), where=mass > 1e-20)


def variant_parts(obj, generated, variants):
    """Narrow the existing mask, then optionally weight specified spectral terms.

    At ratios >= the original mask, the original frame_parts already contains
    all necessary terms. No duplicate implementation of its acoustic formula is
    needed: unmasked envelope/DC stay intact, gated terms are only restricted.
    """
    generated = waveform(generated)
    if len(generated) != len(obj.target):
        raise ValueError("Offline scoring requires the complete equal-length waveform")
    original = obj.frame_parts(generated)
    target_energy = obj.target_features["envelope"]
    maximum = float(target_energy.max())
    energy_weight = (target_energy/maximum)**2 if maximum > 0 else np.zeros_like(target_energy)
    result = {}
    for name, settings in variants.items():
        ratio = settings["ratio"]
        if ratio < obj.config.voiced_ratio:
            raise ValueError("This diagnostic only narrows the original target mask")
        selected = settings["energy_terms"]
        if any(t not in ("transport", "target_weighted_log") for t in selected):
            raise ValueError("Temporal energy weighting is limited to spectral terms")
        parts = original.copy()
        active = target_mask(obj, ratio)
        parts[:, [0, 1, 3]] *= active[:, None]
        for term in selected:
            parts[:, TERMS.index(term)] *= energy_weight
        parts[:, -1] = parts[:, :-1]@coefficients(obj.config)
        contributions = parts[:, :-1]*coefficients(obj.config)[None, :]*obj.weights[:, None]
        result[name] = dict(parts=parts, contributions=contributions,
                            total=float(np.sum(parts[:, -1]*obj.weights)), mask=active)
    return result


def activity_metrics(obj, generated_envelope, ratio):
    target = target_mask(obj, ratio)
    maximum = float(obj.target_features["envelope"].max())
    generated = (np.asarray(generated_envelope) >= ratio*maximum) & (maximum > 1e-8)
    w = obj.weights

    def fraction(where, denominator):
        duration = float(w[denominator].sum())
        return float(w[where & denominator].sum()/duration) if duration > 0 else None

    return dict(ratio=ratio, active_fraction=float(w[target].sum()),
                low_energy_fraction=float(w[~target].sum()),
                missing_fraction=fraction(~generated, target),
                false_voice_fraction=fraction(generated, ~target),
                disagreement_fraction=float(w[target != generated].sum()))


def component_regions(obj, contributions, labels):
    rows = []
    for region in ("body", "weak", "boundary"):
        selected = labels == region
        values = contributions[selected].sum(axis=0)
        rows.append(dict(region=region, duration_fraction=float(obj.weights[selected].sum()),
                         **dict(zip(TERMS, map(float, values))), total=float(values.sum())))
    return rows


def ranking_changes(scores, reference="default", tolerance=1e-12):
    """Compare candidate order only within each fixed score definition."""
    candidates = list(scores)
    variants = list(scores[candidates[0]])
    output, reversals = [], []
    for variant in variants:
        values = np.array([scores[c][variant] for c in candidates])
        ranks = rankdata(values, method="average")
        for name, value, rank in zip(candidates, values, ranks):
            output.append(dict(candidate=name, variant=variant, score=float(value), rank=float(rank)))
        if variant == reference:
            continue
        for i, a in enumerate(candidates):
            for b in candidates[i+1:]:
                old = scores[a][reference]-scores[b][reference]
                new = scores[a][variant]-scores[b][variant]
                if abs(old) > tolerance and abs(new) > tolerance and old*new < 0:
                    reversals.append(dict(variant=variant, first=a, second=b))
    return output, reversals


def waveform_response(obj, generated, block=500):
    """Fixed-waveform audio summaries on the same 10ms grid as saved controls."""
    generated = waveform(generated)
    if len(generated) != len(obj.target) or len(generated) % block:
        raise ValueError("Expected a complete waveform on the control grid")
    normalized = generated/obj.scale
    filtered, _ = signal.sosfilt(obj.sos, normalized, zi=obj.zi*normalized[0])
    raw_blocks, band_blocks = normalized.reshape(-1, block), filtered.reshape(-1, block)
    features = obj._features(generated, len(obj.ends))
    q = shape_distribution(features["power"])
    frequencies = np.fft.rfftfreq(obj.n, 1/obj.rate)[obj.keep]
    frame_centroid = (q*frequencies[None, :]).sum(axis=1)
    assignment = np.minimum((obj.ends-1)//block, len(raw_blocks)-1)
    centroid = np.zeros(len(raw_blocks))
    for i in range(len(raw_blocks)):
        selected = assignment == i
        if selected.any():
            centroid[i] = np.average(frame_centroid[selected], weights=obj.weights[selected])
    return dict(time_s=(np.arange(len(raw_blocks))+.5)*block/obj.rate,
                ac_rms=np.std(raw_blocks, axis=1), band_rms=np.sqrt(np.mean(band_blocks**2, axis=1)),
                centroid_hz=centroid, envelope=features["envelope"], q=q)


def safe_correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3 or not np.isfinite([a, b]).all():
        raise ValueError("Correlation requires equally sized finite time series")
    if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def control_diagnostics(controls, actions, indices, response, *, lags=(0, 1, 2, 3), boundary_fraction=.01):
    controls, actions = np.asarray(controls, float), np.asarray(actions, float)
    if controls.ndim != 2 or controls.shape[1] != 5 or actions.shape != (len(controls), len(indices)):
        raise ValueError("Applied controls and actions must use the same step grid")
    if not np.isfinite(controls).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite saved trajectory")
    stats, correlations = [], []
    names = ("epsilon_l", "epsilon_r", "p_s", "f_e_l", "f_e_r")
    for index, name in enumerate(names):
        values = controls[:, index]
        span = HIGH[index]-LOW[index]
        normalized = (values-LOW[index])/span
        changing = np.diff(values)
        fixed = index not in indices
        row = dict(control=name, structural_fixed=fixed, minimum=float(values.min()), maximum=float(values.max()),
                   range=float(np.ptp(values)), std=float(np.std(values)),
                   mean_abs_difference=float(np.mean(abs(changing))), total_variation=float(np.sum(abs(changing))),
                   span_fraction=float(np.ptp(values)/span), std_fraction=float(np.std(values)/span),
                   lower_boundary_fraction=float(np.mean(normalized <= boundary_fraction)),
                   upper_boundary_fraction=float(np.mean(normalized >= 1-boundary_fraction)),
                   action_span=None, action_std=None, action_mean_abs_difference=None, action_boundary_fraction=None)
        if not fixed:
            action = actions[:, list(indices).index(index)]
            row.update(action_span=float(np.ptp(action)), action_std=float(np.std(action)),
                       action_mean_abs_difference=float(np.mean(abs(np.diff(action)))),
                       action_boundary_fraction=float(np.mean(abs(action) >= 1-boundary_fraction)))
            for metric in ("band_rms", "ac_rms", "centroid_hz"):
                for lag in lags:
                    if not isinstance(lag, int) or lag < 0 or len(values)-lag < 3:
                        raise ValueError("Invalid exploratory lag")
                    a, b = (values, response[metric]) if lag == 0 else (values[:-lag], response[metric][lag:])
                    correlations.append(dict(control=name, response=metric, lag_steps=lag,
                                             correlation=safe_correlation(a, b)))
        stats.append(row)
    return stats, correlations
