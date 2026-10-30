"""Synthetic-only reward diagnosis and matched Phase 1 learning experiments."""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.io import wavfile

from .acoustics import AcousticConfig, distance, spectrum
from .experiments import rollout, train_run, write_json
from .model import ROOT, DEFAULT_CONTROLS, build
from .model_env import ModelConfig, ModelEnv
from .objectives import BalancedConfig, BalancedObjective
from .plotting import comparison, control_trajectories, learning_curves, plt
from .voice_diagnostics import replay, summarize, band_audio, rms

OUTPUT = ROOT/"results/issue10"
PROFILES = ("baseline", "balanced_context", "balanced_aligned")


def model_config(profile, target="constant", gamma=.99):
    if profile not in PROFILES:
        raise ValueError("Unknown experimental profile")
    return ModelConfig(condition="A", target=target, control_indices=(0, 2), diagnostics=True,
                       loss_kind="baseline" if profile == "baseline" else "balanced",
                       reward_timing="aligned" if profile == "balanced_aligned" else "context",
                       reward_discount=gamma, audit_objective=True)


def synthetic_target(kind):
    env = ModelEnv(ModelConfig(target=kind))
    try:
        return env.target_audio.copy(), env.target_controls.copy()
    finally:
        env.close()


def factory(profile, kind, target, controls, gamma):
    cfg = model_config(profile, kind, gamma)

    def create():
        env = ModelEnv(cfg, external_target=target)
        env.target_controls = controls.copy()
        return env
    return create


def fixed_scores(target, generated, profile, *, gamma=.99):
    objective = BalancedObjective(target, 50000)
    scale = objective.scale
    history, rewards = [], []
    for i in range(35):
        end, start = (i+1)*500, max(0, i-2)*500
        legacy = distance(target[start:end], generated[start:end], 50000, scale=scale)
        audit = objective.update(generated[:end])
        if profile == "baseline":
            cost = legacy["total"]
        elif profile == "balanced_context":
            cost = audit["rolling"]["total"]
        else:
            cost = 35*audit["contribution"]["total"]/gamma**i
        rewards.append(-cost)
        history.append(dict(step=i+1, baseline=legacy, objective=audit, reward=-cost))
    return dict(legacy=distance(target, generated, 50000, scale=scale),
                balanced=objective.score(generated), mean_cost=-float(np.mean(rewards)),
                discounted_cost=-float(np.dot(gamma**np.arange(35), rewards))/35,
                step_metrics=history)


def legacy_padding_counts(samples, rate=50000):
    """Count actual zero-padded frames, including the odd-window alignment."""
    _, times, _ = spectrum(np.zeros(samples), rate)
    n = round(.024576*rate)
    starts = np.rint(times*rate).astype(int)-n//2
    return dict(frames=len(times), padded_frames=int(np.count_nonzero(
        (starts < 0) | (starts+n > samples))))


def audit(config):
    out = OUTPUT/"audit"
    out.mkdir(parents=True, exist_ok=True)
    target, controls = synthetic_target("constant")
    wrong = DEFAULT_CONTROLS.copy()
    wrong[0], wrong[2] = 3e7, 3.4e6
    wrong_audio = replay(np.tile(wrong, (35, 1)), "A")["audio"]
    far = wrong.copy()
    far[0], far[2] = 5e7, 4e6
    far_audio = replay(np.tile(far, (35, 1)), "A")["audio"]
    suppressed = controls.copy()
    suppressed[:, 3] = 5e5
    suppressed_audio = replay(suppressed, "A")["audio"]
    cases = dict(oracle=target, wrong_voiced=wrong_audio, far_voiced=far_audio,
                 wrong_matched_rms=wrong_audio*rms(target)/rms(wrong_audio),
                 far_matched_rms=far_audio*rms(target)/rms(far_audio),
                 silence=np.zeros_like(target), near_silent=target*1e-6,
                 dc_only=np.full_like(target, target.mean()), half_gain=target*.5,
                 double_gain=target*2, loud_gain=target*10,
                 delayed=np.r_[np.zeros(1000), target[:-1000]], suppressed_model=suppressed_audio)
    rows, trajectories = [], {}
    for name, generated in cases.items():
        np.savez_compressed(out/f"{name}.npz", target=target, audio=generated, sample_rate=50000)
        for profile in config["profiles"]:
            score = fixed_scores(target, generated, profile, gamma=config["gamma"])
            trajectories[f"{name}/{profile}"] = score.pop("step_metrics")
            rows.append(dict(case=name, profile=profile, **score))
    write_json(out/"scores.json", rows)
    write_json(out/"step_metrics.json", trajectories)
    objective = BalancedObjective(target, 50000)
    write_json(out/"target_frames.json", dict(frame_ends=objective.ends.tolist(), frame_weights=objective.weights.tolist(),
                                              voiced_mask=objective.voiced.tolist(), shared_scale=objective.scale,
                                              energy_reference=objective.energy_reference,
                                              config=asdict(objective.config)))
    # Target-only activity masks, including correct silence and a noise-like target.
    t = np.arange(len(target))/50000
    gated = target*(t >= .1)*(t < .25)
    noise = np.random.default_rng(123).normal(size=len(target))*rms(target)
    special = dict(gated_oracle=(gated, gated), gated_delayed=(gated, np.r_[np.zeros(1000), gated[:-1000]]),
                   gated_always_voiced=(gated, target), quiet_oracle=(target*0, target*0),
                   quiet_wrong_voice=(target*0, target), noise_oracle=(noise, noise), noise_missing=(noise, noise*0))
    special_rows = []
    for name, (a, b) in special.items():
        obj = BalancedObjective(a, 50000, scale=objective.scale)
        special_rows.append(dict(case=name, voiced_frame_fraction=float(obj.voiced.mean()),
                                 balanced=obj.score(b), baseline=distance(a, b, 50000, scale=objective.scale)))
        np.savez_compressed(out/f"{name}.npz", target=a, audio=b, sample_rate=50000)
    write_json(out/"activity_cases.json", special_rows)
    # One-factor-at-a-time sensitivity; no learning-based tuning of these settings.
    sensitivities = []
    changes = [("default", AcousticConfig(), 1.0)]
    changes += [(f"floor_{v}", replace(AcousticConfig(), floor_db=v), 1.0) for v in (-60, -100)]
    changes += [(f"window_{v}", replace(AcousticConfig(), window_s=v), 1.0) for v in (.012288, .049152)]
    changes += [("max_5000", replace(AcousticConfig(), max_hz=5000), 1.0)]
    changes += [(f"envelope_weight_{v}", replace(AcousticConfig(), envelope_weight=v), 1.0) for v in (.5, 2)]
    changes += [(f"shared_scale_{v}", AcousticConfig(), v) for v in (.5, 2)]
    for label, cfg, gain in changes:
        for name in ("oracle", "wrong_voiced", "far_voiced", "far_matched_rms", "silence", "dc_only", "half_gain", "delayed"):
            sensitivities.append(dict(setting=label, case=name, config=asdict(cfg), scale_multiplier=gain,
                                      metrics=distance(target, cases[name], 50000, cfg, scale=objective.scale*gain)))
    for missing in (0, 1, 2, 4):
        obj = BalancedObjective(target, 50000, config=replace(BalancedConfig(), missing_weight=missing))
        for name in ("wrong_voiced", "silence", "dc_only", "half_gain", "loud_gain"):
            sensitivities.append(dict(setting=f"balanced_missing_{missing}", case=name, balanced=obj.score(cases[name])))
    write_json(out/"sensitivity.json", sensitivities)
    pairs = []
    baseline = {r["case"]: r for r in rows if r["profile"] == "baseline"}
    for a, b in itertools.combinations(baseline, 2):
        terminal = baseline[a]["legacy"]["total"]-baseline[b]["legacy"]["total"]
        instant = baseline[a]["discounted_cost"]-baseline[b]["discounted_cost"]
        if terminal*instant < 0:
            pairs.append([a, b])
    padded = []
    for i in range(35):
        start, end = max(0, i-2)*500, (i+1)*500
        padded.append(dict(step=i+1, **legacy_padding_counts(end-start)))
    write_json(out/"temporal_audit.json", dict(baseline_rank_reversals=pairs, baseline_context_frames=padded,
                                               balanced_unique_frames=len(objective.ends),
                                               balanced_weights_sum=float(objective.weights.sum()),
                                               gamma=config["gamma"]))
    with plt.style.context(["science", "ieee", "no-latex"]):
        labels = list(cases)
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        x = np.arange(len(labels))
        axes[0].bar(x, [baseline[n]["legacy"]["total"] for n in labels], label="Legacy whole episode", alpha=.7)
        axes[0].plot(x, [baseline[n]["discounted_cost"] for n in labels], "o-", label="Legacy discounted context")
        axes[0].legend()
        axes[0].set_ylabel("Legacy cost")
        axes[1].bar(x, [baseline[n]["balanced"]["total"] for n in labels])
        axes[1].set_ylabel("Balanced whole episode")
        axes[1].set_xticks(x, labels, rotation=50, ha="right")
        fig.tight_layout()
        fig.savefig(out/"reward_order.png", dpi=150)
        plt.close(fig)
    comparison(target, wrong_audio, 50000, out/"wrong_voiced.png")
    comparison(gated, special["gated_delayed"][1], 50000, out/"gated_timing.png")
    print("Completed synthetic reward and temporal audits", flush=True)


def decorate(out, result, reference):
    with np.load(out/"rollout.npz", allow_pickle=False) as data:
        audio, controls = data["audio"], data["physical_controls"]
        voice = summarize(audio, reference_band_rms=reference, condition="A", motion_mean=data["motion_mean"],
                          motion_std=data["motion_std"], error=result["error"], numerical_failure=result["failed"])
        control_trajectories(controls, out/"controls.png")
        wavfile.write(out/"generated.wav", 50000, (audio/max(1, float(np.max(abs(audio))))).astype(np.float32))
    write_json(out/"voice.json", voice)
    return voice


def train_cohort(config, *, kind="constant", algorithms=None, steps=None):
    target, controls = synthetic_target(kind)
    target_hash = hashlib.sha256(target.tobytes()).hexdigest()
    reference = rms(band_audio(target)[5000:])
    steps = steps or config["steps"]
    settings = dict(config, steps=steps, eval_interval=config["eval_interval"])
    directory = OUTPUT/f"{kind}_{steps}"
    directory.mkdir(parents=True, exist_ok=True)
    rows, curves, objective_curves = [], {}, {}
    for name, profile in itertools.product(algorithms or config["algorithms"], config["profiles"]):
        create = factory(profile, kind, target, controls, config["gamma"])
        key = f"{name}/{profile}"
        curves[key], objective_curves[key] = [], []
        for seed in config["seeds"]:
            out = directory/f"{profile}/{name}_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            expected = json.loads(json.dumps(dict(settings=settings, model=asdict(model_config(profile, kind, config["gamma"])),
                                                  algorithm=name, seed=seed, target_sha256=target_hash)))
            if (out/"config.json").exists() and json.loads((out/"config.json").read_text()) != expected:
                raise ValueError("Existing experiment settings differ; use a new result directory")
            write_json(out/"config.json", expected)
            if (out/"summary.json").exists() and (out/"curve.json").exists():
                result = json.loads((out/"summary.json").read_text())
                entries = json.loads((out/"curve.json").read_text())
            else:
                print(f"Issue 10: {kind}, {steps} steps, {name}, {profile}, seed {seed}", flush=True)
                result, entries = train_run(name, seed, create, create, settings, out)
            voice = decorate(out, result, reference)
            rows.append(dict(algorithm=name, profile=profile, seed=seed, target=kind, budget=steps,
                             result=result, voice=voice))
            write_json(directory/"summary.json", rows)
            curves[key].append(([e["steps"] for e in entries], [e["error"] for e in entries]))
            objective_curves[key].append(([e["steps"] for e in entries], [e["objective_error"] for e in entries]))
    learning_curves(curves, directory/"legacy_learning_curves.png")
    learning_curves(objective_curves, directory/"balanced_learning_curves.png")
    return rows


def baselines(config, kind="constant"):
    target, controls = synthetic_target(kind)
    reference = rms(band_audio(target)[5000:])
    out = OUTPUT/f"{kind}_baselines"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for profile in config["profiles"]:
        create = factory(profile, kind, target, controls, config["gamma"])
        for seed in config["seeds"]:
            env = create()
            rng = np.random.default_rng(seed)
            try:
                policies = dict(oracle=lambda obs, e: e.encode(e.target_controls[e.index]),
                                constant=lambda obs, e: np.zeros(2), random=lambda obs, e: rng.uniform(-1, 1, 2))
                for policy, choose in policies.items():
                    result, audio, actions = rollout(env, choose, seed=seed)
                    rows.append(dict(profile=profile, seed=seed, method=policy, **result))
                    if seed == config["seeds"][0]:
                        np.savez_compressed(out/f"{profile}_{policy}.npz", target=target, audio=audio, actions=actions, sample_rate=50000)
                best, trace = None, []

                def objective(action):
                    nonlocal best
                    result, audio, _ = rollout(env, lambda obs, e: action, seed=seed)
                    cost = result["discounted_reward_cost"] if not result["failed"] else 1e6
                    if best is None or cost < best["optimized_cost"]:
                        best = dict(optimized_cost=cost, result=result, action=np.array(action), audio=audio.copy())
                    trace.append(dict(steps=(len(trace)+1)*35, optimized_cost=cost,
                                      error=result["error"], objective_error=result["objective_error"]))
                    return cost

                print(f"Issue 10 DE: {kind}, {profile}, seed {seed}", flush=True)
                differential_evolution(objective, [(-1, 1)]*2, seed=seed, popsize=3,
                                       maxiter=config["direct_episodes"]//6-1, polish=False, tol=0, atol=0)
                rows.append(dict(profile=profile, seed=seed, method="DE", search_steps=len(trace)*35,
                                 optimized_cost=best["optimized_cost"], **best["result"]))
                write_json(out/f"{profile}_DE_{seed}_curve.json", trace)
                np.savez_compressed(out/f"{profile}_DE_{seed}.npz", target=target, audio=best["audio"],
                                    action=best["action"], sample_rate=50000)
                # Voice check of the selected constant search candidate (no extra training).
                replayed = replay(np.tile(env.decode(best["action"]), (35, 1)), "A")
                diagnosis = summarize(best["audio"], reference_band_rms=reference, condition="A",
                                      motion_mean=replayed["motion_mean"], motion_std=replayed["motion_std"])
                write_json(out/f"{profile}_DE_{seed}_voice.json", diagnosis)
            finally:
                env.close()
            write_json(out/"summary.json", rows)


def extension(config):
    initial = json.loads((OUTPUT/f"constant_{config['steps']}/summary.json").read_text())
    selected = []
    for algorithm in config["algorithms"]:
        aligned = [r for r in initial if r["algorithm"] == algorithm and r["profile"] == "balanced_aligned"]
        baseline = [r for r in initial if r["algorithm"] == algorithm and r["profile"] == "baseline"]
        improvement = 0
        for r in aligned:
            curve = json.loads((OUTPUT/f"constant_{config['steps']}/balanced_aligned/{algorithm}_{r['seed']}/curve.json").read_text())
            improvement += r["result"]["objective_error"] < curve[0]["objective_error"]
        if improvement >= 2 and np.mean([r["result"]["objective_error"] for r in aligned]) < np.mean([r["result"]["objective_error"] for r in baseline]):
            selected.append(algorithm)
    write_json(OUTPUT/"extension_decision.json", dict(rule=config["extension_rule"], selected_algorithms=selected,
                                                    budget_per_run=config["extension_steps"], exploratory_selection=True))
    if selected:
        train_cohort(config, algorithms=selected, steps=config["extension_steps"])
    return selected


def glide(config):
    selected = []
    for algorithm in config["algorithms"]:
        for directory in [OUTPUT/f"constant_{config['steps']}", OUTPUT/f"constant_{config['extension_steps']}"]:
            if not (directory/"summary.json").exists():
                continue
            group = [r for r in json.loads((directory/"summary.json").read_text())
                     if r["algorithm"] == algorithm and r["profile"] == "balanced_aligned"]
            if sum(r["result"]["error"] <= config["threshold"] for r in group) >= 2:
                selected.append(algorithm)
                break
    write_json(OUTPUT/"glide_decision.json", dict(rule=config["glide_rule"], selected_algorithms=selected,
                                                claim="fixed-target trajectory imitation only; no goal features are added"))
    if selected:
        train_cohort(config, kind="glide", algorithms=selected)
        baselines(config, kind="glide")
    else:
        # Generate the harder target and verify representability without claiming RL success.
        target, controls = synthetic_target("glide")
        env = factory("balanced_aligned", "glide", target, controls, config["gamma"])()
        try:
            result, audio, _ = rollout(env, lambda obs, e: e.encode(e.target_controls[e.index]))
            out = OUTPUT/"glide_oracle"
            out.mkdir(exist_ok=True)
            write_json(out/"summary.json", result)
            np.savez_compressed(out/"rollout.npz", target=target, audio=audio, sample_rate=50000,
                                physical_controls=np.array(env.applied_control_history))
            comparison(target, audio, 50000, out/"spectrogram.png")
            control_trajectories(env.applied_control_history, out/"controls.png")
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "train", "baselines", "extend", "glide", "report", "all"])
    parser.add_argument("--config", type=Path, default=ROOT/"configs/issue10.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage != "report":
        build()
        write_json(OUTPUT/"config.json", config)
    if args.stage in ("audit", "all"):
        audit(config)
    if args.stage in ("train", "all"):
        train_cohort(config)
    if args.stage in ("baselines", "all"):
        baselines(config)
    if args.stage in ("extend", "all"):
        extension(config)
    if args.stage in ("glide", "all"):
        glide(config)
    if args.stage in ("report", "all"):
        from .issue10_report import main as report
        report()


if __name__ == "__main__":
    main()
