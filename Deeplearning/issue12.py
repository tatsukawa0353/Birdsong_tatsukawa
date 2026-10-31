"""Read-only offline diagnosis of Issue 11; recording results stay private."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .acoustics import distance
from .model import ROOT
from .objectives import BalancedConfig, BalancedObjective
from .offline_diagnostics import (TERMS, target_mask, temporal_regions, shape_distribution,
                                  variant_parts, activity_metrics, component_regions, ranking_changes,
                                  waveform_response, control_diagnostics)
from .phase2 import private_output
from .plotting import plt, CONTROL_LABELS
from .voice_diagnostics import pooled_motion

CORE = ("objectives.py", "acoustics.py", "model.py", "model_env.py", "native/model.cpp",
        "native/reference.cpp", "experiments.py", "phase2.py", "plotting.py", "issue11.py")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def table(path, rows):
    if rows:
        with Path(path).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def guarded_paths(source, output):
    source, output = private_output(source), private_output(output)
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output must be separate non-overlapping private directories")
    return source, output


def inside(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes the private experiment directory")
    return path


def manifest(source):
    result = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            inside(source, path.relative_to(source))
        if path.is_file():
            result[str(path.relative_to(source))] = sha_file(path)
    return result


def prepare(source, output, config):
    source, output = guarded_paths(source, output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if config["ratios"] != [.01, .03, .05, .1]:
        raise ValueError("Issue 12 diagnoses the declared four target thresholds")
    before = manifest(source)
    original = output/"source_manifest_before.json"
    if original.exists() and read(original) != before:
        raise ValueError("Issue 11 source differs from the original offline snapshot")
    write(original, before)
    core = {name: sha_file(ROOT/name) for name in CORE}
    if (output/"config.json").exists() and read(output/"config.json") != config:
        raise ValueError("Diagnostic settings changed; use a fresh private output")
    write(output/"config.json", config)
    if (output/"core_before.json").exists() and read(output/"core_before.json") != core:
        raise ValueError("Original model/reward/display source changed")
    write(output/"core_before.json", core)
    write(output/"provenance.json", dict(source=str(source), output=str(output),
                                         base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                         read_only_source=True, new_training_runs=0,
                                         saved_loss_config=read(source/"protocol.json")["balanced_config"]))
    return source, output


def verify_unchanged(source, output):
    before = read(output/"source_manifest_before.json")
    after = manifest(source)
    if before != after:
        raise AssertionError("Issue 11 input files changed during offline diagnosis")
    core = {name: sha_file(ROOT/name) for name in CORE}
    if core != read(output/"core_before.json"):
        raise AssertionError("Original model/reward/display code changed")
    write(output/"source_integrity.json", dict(content_unchanged=True, files=len(after), core_unchanged=True,
                                             new_training_runs=0, verification="SHA256 and file-set equality before/after"))


def load_target(source):
    protocol = read(source/"protocol.json")
    with np.load(source/"target.npz", allow_pickle=False) as data:
        target = data["audio"].copy()
    if hashlib.sha256(target.tobytes()).hexdigest() != protocol["target_sha256"]:
        raise ValueError("Cached source target hash mismatch")
    if protocol["balanced_config"] != asdict(BalancedConfig()):
        raise ValueError("Saved and current default balanced definitions differ")
    obj = BalancedObjective(target, 50000, scale=protocol["shared_loss_scale"])
    with np.load(source/"target_frames.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(obj.ends, saved["ends"])
        np.testing.assert_array_equal(obj.voiced, saved["voiced"])
        np.testing.assert_allclose(obj.weights, saved["weights"], rtol=0, atol=0)
        np.testing.assert_allclose(obj.target_features["envelope"], saved["envelope"], rtol=0, atol=0)
        np.testing.assert_allclose(obj.target_features["power"], saved["power"], rtol=0, atol=0)
    return target, obj, protocol


def spans(obj, selected):
    output = []
    first = None
    for i, flag in enumerate(np.r_[selected, False]):
        if flag and first is None:
            first = i
        elif not flag and first is not None:
            output.append((float(obj.starts[first]/obj.rate), float(obj.ends[i-1]/obj.rate)))
            first = None
    return output


def mask_plot(obj, config, path):
    time, energy = obj.ends/obj.rate, obj.target_features["envelope"]
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(time, energy, color="black", label="Target causal band envelope")
        colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
        for i, (ratio, color) in enumerate(zip(config["ratios"], colors)):
            axes[0].axhline(ratio*energy.max(), color=color, linestyle="--", label=f"{ratio:.0%} target maximum")
            axes[1].stairs(target_mask(obj, ratio).astype(int)+1.3*i, np.r_[0, time],
                           color=color, baseline=None, label=f"{ratio:.0%}")
        axes[0].set_ylabel("Envelope / common target scale")
        axes[1].set(xlabel="Frame end time [s]", ylabel="Target-only masks (offset)")
        for ax in axes:
            ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def loss_plot(obj, comparisons, labels, path):
    time = obj.ends/obj.rate
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(3, len(comparisons), figsize=(8*len(comparisons), 12), sharex=True, squeeze=False)
        for col, (name, audio, analysis, features) in enumerate(comparisons):
            axes[0, col].plot(time, obj.target_features["envelope"], color="black", label="Target")
            axes[0, col].plot(time, features["envelope"], label="Generated")
            axes[0, col].set_title(name)
            weighted = analysis["default"]["contributions"]
            colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
            for i, (term, color) in enumerate(zip(TERMS, colors)):
                axes[1, col].plot(time, weighted[:, i]/obj.weights, color=color, label=term)
                axes[2, col].plot(time, np.cumsum(weighted[:, i]), color=color, label=term)
            for ax in axes[:, col]:
                for start, end in spans(obj, labels == "weak"):
                    ax.axvspan(start, end, color="#999999", alpha=.15)
                for start, end in spans(obj, labels == "boundary"):
                    ax.axvspan(start, end, color="#E69F00", alpha=.12)
                ax.legend(fontsize=9)
            axes[0, col].set_ylabel("Shared-scale band envelope")
            axes[1, col].set_ylabel("Coefficient-weighted frame loss")
            axes[2, col].set(xlabel="Frame end time [s]", ylabel="Cumulative duration-weighted cost")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)


def normalization_plot(obj, generated, labels, path):
    target = obj.target_features
    features = obj._features(generated, len(obj.ends))
    time = obj.ends/obj.rate
    freq = np.fft.rfftfreq(obj.n, 1/obj.rate)[obj.keep]
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for f, name in ((target, "Target"), (features, "Generated")):
            axes[0, 0].semilogy(time, np.maximum(f["power"].sum(axis=1), 1e-20), label=f"{name} band PSD mass")
            axes[0, 1].plot(time, f["envelope"], label=f"{name} absolute envelope")
        maximum = target["envelope"].max()
        axes[0, 1].plot(time, target["envelope"]/maximum, "--", label="Target relative envelope")
        axes[0, 1].plot(time, (target["envelope"]/maximum)**2, ":", label="Target relative energy g")
        for region in ("body", "weak"):
            available = np.flatnonzero(labels == region)
            if not len(available):
                continue
            ordered = available[np.argsort(target["envelope"][available])]
            index = ordered[len(ordered)//2]
            for f, name in ((target, "Target"), (features, "Generated")):
                p = f["power"][index]
                axes[1, 0].semilogy(freq, np.maximum(p, 1e-20), label=f"{name}/{region}, frame {index+1}")
                axes[1, 1].plot(freq, shape_distribution(p[None, :])[0], label=f"{name}/{region}, frame {index+1}")
        axes[0, 0].set(xlabel="Frame end time [s]", ylabel="Power under common amplitude scale")
        axes[0, 1].set(xlabel="Frame end time [s]", ylabel="Absolute vs target-relative energy")
        axes[1, 0].set(xlabel="Frequency [Hz]", ylabel="Raw feature power P")
        axes[1, 1].set(xlabel="Frequency [Hz]", ylabel="Unit-mass shape q=P/sum(P)")
        for ax in axes.flat:
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def controls_for_target(target):
    rng = np.random.default_rng(12)
    noise = rng.normal(size=len(target))
    noise *= np.sqrt(np.mean(target**2))/np.sqrt(np.mean(noise**2))
    return dict(oracle=target.copy(), silence=np.zeros_like(target), dc_only=np.full_like(target, target.mean()),
                half_gain=target*.5, delayed_20ms=np.r_[np.zeros(1000), target[:-1000]], matched_rms_noise=noise)


def audit_ab(source, output, config):
    target, obj, protocol = load_target(source)
    labels, changes = temporal_regions(obj, ratio=config["region_ratio"], boundary_radius_s=config["boundary_radius_s"])
    time = obj.ends/obj.rate
    np.savez_compressed(output/"target_analysis.npz", frame_end_s=time, starts=obj.starts, weights=obj.weights,
                        envelope=obj.target_features["envelope"], power=obj.target_features["power"],
                        q=shape_distribution(obj.target_features["power"]), regions=labels,
                        transitions_s=changes, masks=np.array([target_mask(obj, r) for r in config["ratios"]]))
    masks = [dict(ratio=r, active_fraction=float(obj.weights[target_mask(obj, r)].sum()),
                  low_energy_duration_s=float(obj.weights[~target_mask(obj, r)].sum()*len(target)/obj.rate)) for r in config["ratios"]]
    write(output/"masks.json", masks)
    table(output/"masks.csv", masks)
    mask_plot(obj, config, output/"activity_masks.png")
    review = ["# 手動確認欄（閾値は真のラベルではない）", "",
              "| 10%候補の区間 [s] | 境界の妥当性 | 背景/残響/弱い声/雑音様音節 | 根拠・確度 |", "|---|---|---|---|"]
    review += [f"| {a:.6f}–{b:.6f} | 要確認 | 未確定 | 低エネルギーだけでは除外しない |" for a, b in spans(obj, ~target_mask(obj, .1))]
    if not (output/"manual_review.md").exists():
        (output/"manual_review.md").write_text("\n".join(review)+"\n")
    rows = read(source/"summary.json")
    run_scores, activity, regions, verified, pools, representative = [], [], [], [], {}, {}
    for number, row in enumerate(rows):
        relative = row["relative_path"]
        path = inside(source, relative)
        destination = inside(output, relative)
        destination.mkdir(parents=True, exist_ok=True)
        with np.load(path/"rollout.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(target, data["target"])
            audio = data["audio"].copy()
        if row["result"]["failed"] or len(audio) != len(target):
            raise ValueError("Failed or incomplete saved waveform requires a separate diagnostic cohort")
        features = obj._features(audio, len(obj.ends))
        analysis = variant_parts(obj, audio, config["variants"])
        original = obj.score(audio)
        old = distance(target, audio, 50000, scale=obj.scale)
        residual = max(abs(old["total"]-row["result"]["error"]),
                       abs(original["total"]-row["result"]["objective_error"]),
                       abs(analysis["default"]["total"]-original["total"]))
        with np.load(source/"target_frames.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(obj.target_features["envelope"], saved["envelope"])
        saved_frames = read(path/"activity_frames.json")
        np.testing.assert_allclose(obj.frame_parts(audio), saved_frames["loss_parts"], rtol=0, atol=0)
        if residual > 1e-11:
            raise AssertionError("Saved default scores no longer reproduce")
        verified.append(dict(run=relative, maximum_score_residual=residual))
        base = {k: row[k] for k in ("algorithm", "budget", "condition", "profile", "seed")}
        scores = {name: value["total"] for name, value in analysis.items()}
        run_scores.append(dict(**base, run=relative, legacy_original=row["result"]["error"], **scores))
        key = f"{row['algorithm']}_{row['budget']}/{row['condition']}"
        pools.setdefault(key, {})[f"{row['profile']}_seed{row['seed']}"] = scores
        frame_rows = []
        for i, end in enumerate(time):
            raw = analysis["default"]["parts"][i]
            weighted = analysis["default"]["contributions"][i]
            frame_rows.append(dict(frame=i+1, end_s=float(end), weight=float(obj.weights[i]), region=str(labels[i]),
                                   target_envelope=float(obj.target_features["envelope"][i]), generated_envelope=float(features["envelope"][i]),
                                   target_power_mass=float(obj.target_features["power"][i].sum()), generated_power_mass=float(features["power"][i].sum()),
                                   **{f"raw_{term}": float(raw[j]) for j, term in enumerate(TERMS)},
                                   **{f"weighted_{term}": float(weighted[j]) for j, term in enumerate(TERMS)},
                                   weighted_total=float(weighted.sum())))
        table(destination/"frame_components.csv", frame_rows)
        np.savez_compressed(destination/"features.npz", target_power=obj.target_features["power"], generated_power=features["power"],
                            target_q=shape_distribution(obj.target_features["power"]), generated_q=shape_distribution(features["power"]),
                            target_envelope=obj.target_features["envelope"], generated_envelope=features["envelope"],
                            **{f"parts_{v}": a["parts"] for v, a in analysis.items()},
                            **{f"contributions_{v}": a["contributions"] for v, a in analysis.items()})
        for ratio in config["ratios"]:
            activity.append(dict(**base, **activity_metrics(obj, features["envelope"], ratio)))
        for variant, values in analysis.items():
            for result in component_regions(obj, values["contributions"], labels):
                regions.append(dict(**base, variant=variant, **result))
        selection = config["representatives"]
        if (row["algorithm"] == selection["algorithm"] and row["budget"] == selection["budget"]
                and row["seed"] == selection["seed"] and row["condition"] in selection["conditions"]):
            representative.setdefault(row["condition"], []).append((row["profile"], audio, analysis, features))
        if (number+1) % 12 == 0:
            print(f"Issue 12 fixed-waveform A/B diagnosis: {number+1}/{len(rows)}", flush=True)
    for case, examples in representative.items():
        loss_plot(obj, examples, labels, output/f"loss_components_{case}.png")
        normalization_plot(obj, examples[-1][1], labels, output/f"spectral_normalization_{case}.png")
    controls = controls_for_target(target)
    control_scores = {name: {variant: result["total"] for variant, result in variant_parts(obj, audio, config["variants"]).items()}
                      for name, audio in controls.items()}
    ranks, reversals = [], []
    for key, candidates in pools.items():
        ranking, changes = ranking_changes(dict(candidates, **{f"control/{k}": v for k, v in control_scores.items()}))
        ranks.extend(dict(pool=key, **r) for r in ranking)
        reversals.extend(dict(pool=key, **r) for r in changes)
    table(output/"fixed_scores.csv", run_scores)
    table(output/"activity_comparison.csv", activity)
    table(output/"regional_components.csv", regions)
    table(output/"rankings.csv", ranks)
    write(output/"rank_reversals.json", reversals)
    write(output/"control_scores.json", control_scores)
    write(output/"score_verification.json", dict(runs=len(rows), maximum_residual=max(r["maximum_score_residual"] for r in verified), runs_verified=verified))
    # Written before C so a short A/B-only run is useful for seminar preparation.
    from .issue12_report import seminar_memo
    seminar_memo(output)


def control_plot(obj, row, data, response, labels, path):
    controls = data["physical_controls"]
    times = np.arange(len(controls)+1)*.01
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(8, 1, figsize=(16, 36), sharex=True)
        for i in range(5):
            axes[i].step(times, np.r_[controls[:, i], controls[-1, i]], where="post", linewidth=2)
            axes[i].set_ylabel(CONTROL_LABELS[i], fontsize=46)
            axes[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
            axes[i].yaxis.get_offset_text().set_fontsize(24)
        axes[5].plot(obj.ends/obj.rate, obj.target_features["envelope"], color="black", label="Target band envelope")
        axes[5].plot(obj.ends/obj.rate, response["envelope"], label="Generated band envelope")
        axes[6].plot(response["time_s"], response["band_rms"], label="Generated 10ms band RMS")
        axes[6].plot(response["time_s"], response["ac_rms"], label="Generated 10ms AC RMS")
        axes[7].plot(np.arange(35)*.01+.005, data["motion_std"][:, 1], label="Left velocity std")
        axes[7].plot(np.arange(35)*.01+.005, data["motion_std"][:, 3], label="Right velocity std")
        for i, label in ((5, "Envelope"), (6, "RMS"), (7, "Velocity std")):
            axes[i].set_ylabel(label, fontsize=35)
            axes[i].legend(fontsize=18)
        for ax in axes:
            for a, b in spans(obj, labels == "weak"):
                ax.axvspan(a, b, color="gray", alpha=.15)
            for a, b in spans(obj, labels == "boundary"):
                ax.axvspan(a, b, color="#E69F00", alpha=.10)
            ax.tick_params(labelsize=30, direction="in", length=10, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        axes[-1].set(xlim=(0, .35))
        axes[-1].set_xlabel("Time [s]", fontsize=46)
        fig.suptitle(f"{row['algorithm']} / {row['budget']} / Case {row['condition']} / {row['profile']} / seed {row['seed']}", fontsize=24)
        fig.tight_layout(rect=(0, 0, 1, .99))
        fig.savefig(path, dpi=90)
        plt.close(fig)


def audit_controls(source, output, config):
    target, obj, _ = load_target(source)
    target_response = waveform_response(obj, target)
    write(output/"target_response_summary.json", dict(
        band_rms_cv=float(np.std(target_response["band_rms"])/max(np.mean(target_response["band_rms"]), 1e-12)),
        centroid_std_hz=float(np.std(target_response["centroid_hz"])),
        centroid_range_hz=float(np.ptp(target_response["centroid_hz"])),
        shape_tv=float(np.mean(np.sum(abs(np.diff(target_response["q"], axis=0)), axis=1)/2))))
    labels, _ = temporal_regions(obj, ratio=config["region_ratio"], boundary_radius_s=config["boundary_radius_s"])
    rows = read(source/"summary.json")
    all_stats, all_correlations, responses = [], [], []
    for number, row in enumerate(rows):
        path = inside(source, row["relative_path"])
        destination = inside(output, row["relative_path"])
        destination.mkdir(parents=True, exist_ok=True)
        cfg = read(path/"config.json")
        with np.load(path/"rollout.npz", allow_pickle=False) as saved:
            data = {k: saved[k].copy() for k in ("audio", "physical_controls", "requested_controls", "actions", "states", "motion_mean", "motion_std")}
        if row["condition"] == "B":
            np.testing.assert_array_equal(data["physical_controls"][:, 4], np.full(35, 1e7))
        response = waveform_response(obj, data["audio"])
        stats, correlations = control_diagnostics(data["physical_controls"], data["actions"], cfg["model"]["control_indices"], response,
                                                  lags=config["control_lags_steps"], boundary_fraction=config["boundary_tolerance_fraction"])
        base = {k: row[k] for k in ("algorithm", "budget", "condition", "profile", "seed")}
        all_stats.extend(dict(**base, **r) for r in stats)
        all_correlations.extend(dict(**base, **r) for r in correlations)
        free = [r for r in stats if not r["structural_fixed"]]
        near_constant = all(r["action_span"] <= config["near_constant_action_span"]
                            and r["span_fraction"] <= config["near_constant_physical_span_fraction"] for r in free)
        rms_values = response["band_rms"]
        centroid = response["centroid_hz"]
        q = response["q"]
        summary = dict(**base, near_constant_controls=near_constant,
                       maximum_free_action_span=max(r["action_span"] for r in free),
                       maximum_physical_span_fraction=max(r["span_fraction"] for r in free),
                       band_rms_mean=float(np.mean(rms_values)), band_rms_cv=float(np.std(rms_values)/max(np.mean(rms_values), 1e-12)),
                       band_rms_total_variation=float(np.sum(abs(np.diff(rms_values)))),
                       centroid_std_hz=float(np.std(centroid)), centroid_range_hz=float(np.ptp(centroid)),
                       shape_temporal_variation=float(np.mean(np.sum(abs(np.diff(q, axis=0)), axis=1)/2)),
                       source_sustained=row["diagnosis"]["sustained_acoustic"],
                       source_left_velocity_std=row["diagnosis"]["motion"]["std"][1],
                       source_right_velocity_std=row["diagnosis"]["motion"]["std"][3])
        responses.append(summary)
        table(destination/"control_statistics.csv", stats)
        table(destination/"control_correlations.csv", correlations)
        table(destination/"audio_response.csv", [dict(time_s=float(t), band_rms=float(b), ac_rms=float(a), centroid_hz=float(c))
                                                 for t, b, a, c in zip(response["time_s"], response["band_rms"], response["ac_rms"], centroid)])
        control_plot(obj, row, data, response, labels, destination/"controls_and_audio.png")
        if (number+1) % 12 == 0:
            print(f"Issue 12 saved-control diagnosis: {number+1}/{len(rows)}", flush=True)
    table(output/"control_statistics.csv", all_stats)
    table(output/"control_correlations.csv", all_correlations)
    table(output/"response_summary.csv", responses)
    audit_states(source, output)


def audit_states(source, output):
    """Compare saved endpoint states with saved full-integration moments."""
    rows = []
    for run in read(source/"summary.json"):
        path = inside(source, run["relative_path"])
        with np.load(path/"rollout.npz", allow_pickle=False) as data:
            means, stds = pooled_motion(data["motion_mean"][10:], data["motion_std"][10:])
            np.testing.assert_allclose(means, run["diagnosis"]["motion"]["mean"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(stds, run["diagnosis"]["motion"]["std"], rtol=1e-12, atol=1e-12)
            endpoints = data["states"][:, 2:]
            for i, name in enumerate(("x_l", "y_l", "x_r", "y_r")):
                rows.append(dict(**{k: run[k] for k in ("algorithm", "budget", "condition", "profile", "seed")},
                                 state=name, endpoint_min=float(endpoints[:, i].min()), endpoint_max=float(endpoints[:, i].max()),
                                 endpoint_std=float(endpoints[:, i].std()), late_full_rate_mean=float(means[i]),
                                 late_full_rate_std=float(stds[i]), late_full_rate_rms=float(np.hypot(means[i], stds[i]))))
    table(output/"state_statistics.csv", rows)


def synthetic_fixtures():
    rate = 50000
    t = np.arange(17500)/rate
    strong = (t >= .03)*(t < .12)*.6*np.sin(2*np.pi*1300*t)
    weak = (t >= .20)*(t < .27)*.02*np.sin(2*np.pi*1900*t)
    target = strong+weak
    noise = np.random.default_rng(12).normal(size=len(t))*.1
    return dict(oracle=(target, target), silence=(target, target*0), missing_weak=(target, strong),
                delayed=(target, np.r_[np.zeros(1000), target[:-1000]]),
                wrong_voice=(target, (abs(target) > 0)*.6*np.sin(2*np.pi*2300*t)),
                always_voice=(target, .6*np.sin(2*np.pi*1300*t)), loud=(target, target*3),
                quiet_correct=(target*0, target*0), quiet_false_voice=(target*0, strong),
                noise_oracle=(noise, noise), noise_missing=(noise, noise*0), noise_wrong=(noise, np.roll(noise, 1000)))


def synthetic_audit(config):
    # This public export has no user-supplied waveform argument and creates all
    # inputs mathematically. Recording data cannot flow into this branch.
    out = ROOT/"results/issue12_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    scores, families = [], {}
    for name, (target, generated) in synthetic_fixtures().items():
        obj = BalancedObjective(target, 50000, scale=1)
        analysis = variant_parts(obj, generated, config["variants"])
        family = "quiet" if name.startswith("quiet_") else "noise" if name.startswith("noise_") else "tonal"
        families.setdefault(family, {})[name] = {variant: value["total"] for variant, value in analysis.items()}
        np.savez_compressed(out/f"{name}.npz", target=target, audio=generated, sample_rate=50000)
        for variant, result in analysis.items():
            scores.append(dict(case=name, variant=variant, score=result["total"],
                               **dict(zip(TERMS, map(float, result["contributions"].sum(axis=0))))))
    table(out/"scores.csv", scores)
    ranks, changes = [], []
    for family, values in families.items():
        ranking, reversals = ranking_changes(values)
        ranks.extend(dict(target_family=family, **r) for r in ranking)
        changes.extend(dict(target_family=family, **r) for r in reversals)
    table(out/"rankings.csv", ranks)
    write(out/"rank_reversals.json", changes)
    write(out/"config.json", config)
    print("Completed recording-independent synthetic checks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ab", "controls", "synthetic", "report", "all"))
    parser.add_argument("--source", type=Path, default=ROOT/"private_runs/issue11")
    parser.add_argument("--output", type=Path, default=ROOT/"private_runs/issue12")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/issue12.json")
    args = parser.parse_args()
    config = read(args.config)
    if args.stage == "synthetic":
        synthetic_audit(config)
        return
    previous = os.umask(0o077)
    try:
        source, output = prepare(args.source, args.output, config)
        if args.stage in ("ab", "all"):
            audit_ab(source, output, config)
            verify_unchanged(source, output)
        if args.stage == "all":
            synthetic_audit(config)
        if args.stage in ("controls", "all"):
            audit_controls(source, output, config)
        if args.stage in ("report", "all"):
            from .issue12_report import main as report
            report(source, output)
        verify_unchanged(source, output)
    finally:
        os.umask(previous)


if __name__ == "__main__":
    main()
