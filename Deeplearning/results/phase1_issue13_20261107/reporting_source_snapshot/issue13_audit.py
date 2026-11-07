"""Independent, simulation-free audit of Issue #13 constant-control results.

This checker reads the new experiment and its explicitly recorded source/input
hashes. It never trains a policy or integrates the physical model. Acoustic
metrics are recomputed only for the saved initial/final/best audio archives;
unsaved curve audio cannot be independently verified from JSON alone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile

from .acoustics import distance
from .model import HIGH, LIBRARY, LOW, ROOT
from .objectives import BalancedObjective
from .phase1_validation_audit import close_scalar, parameter_nrmse


CONDITIONS = ("constant_1d", "constant_2d")
METHODS = ("TD3", "SAC", "random_search", "DE")
SEEDS = (0, 1, 2)
BUDGET = 256
TARGET_CONTROLS = np.asarray([4.1e7, 3.5e7, 3.4e6, 49000., 49000.])


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Auditor:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.errors = []
        self.checks = 0
        self.historical_comparisons = []
        self.counts = dict(runs_expected=24, runs_found=0, curves_checked=0,
                           curve_points=0, rollouts_checked=0,
                           acoustic_recomputations=0, wav_files_checked=0,
                           balanced_recomputations=0, oracles_checked=0,
                           policy_predictions_checked=0, protected_files_checked=0,
                           source_hashes_checked=0, training_episodes=0,
                           evaluation_episodes=0, training_integration_s=0.,
                           evaluation_integration_s=0.)

    def check(self, ok, code, path, expected=None, actual=None):
        self.checks += 1
        if not ok:
            path = Path(path)
            label = str(path.relative_to(self.output)) if path.is_relative_to(self.output) else str(path)
            self.errors.append(_plain(dict(code=code, path=label, expected=expected, actual=actual)))
        return bool(ok)

    def scalar(self, actual, expected, path, code="numeric_mismatch"):
        return self.check(close_scalar(actual, expected), code, path, expected, actual)

    def read_json(self, path):
        try:
            value = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            self.check(False, "json_read", path, "valid readable JSON", str(exc))
            return None
        self.check(True, "json_read", path)
        return value

    def array(self, actual, expected, path, *, exact=True):
        try:
            value = np.asarray(actual, dtype=float)
            expected = np.asarray(expected, dtype=float)
            same = value.shape == expected.shape and bool(np.isfinite(value).all())
            same = same and (np.array_equal(value, expected) if exact else np.allclose(value, expected, atol=1e-11, rtol=1e-9))
        except (ValueError, TypeError):
            same = False
        return self.check(same, "array_mismatch", path, expected, actual)

    def numeric_dict(self, actual, expected, path):
        if not self.check(isinstance(actual, dict), "metric_schema", path, "dict", type(actual).__name__):
            return
        for name, value in expected.items():
            if isinstance(value, dict):
                self.numeric_dict(actual.get(name), value, Path(path) / name)
            elif isinstance(value, (list, tuple, np.ndarray)):
                self.array(actual.get(name), value, Path(path) / name)
            elif isinstance(value, bool):
                self.check(actual.get(name) is value, "metric_bool", Path(path) / name, value, actual.get(name))
            elif isinstance(value, (int, float, np.number)):
                self.scalar(actual.get(name), value, Path(path) / name)

    def accounting(self, record, episodes, path):
        if not self.check(isinstance(record, dict), "account_schema", path, "dict", type(record).__name__):
            return
        wanted = dict(episodes=episodes, decision_steps=episodes,
                      integration_s=episodes*.35, integration_completed_s=episodes*.35,
                      native_blocks=episodes*35, native_steps=episodes*3500000,
                      failures=0)
        self.numeric_dict(record, wanted, path)
        wall = record.get("wall_s")
        self.check(close_scalar(wall, wall) and wall >= 0, "account_wall", path, "finite nonnegative wall_s", wall)

    def control_record(self, record, condition, manifest, path):
        indices = [0] if condition == "constant_1d" else [0, 2]
        actions = np.asarray(record.get("actions"), dtype=float)
        controls = np.asarray(record.get("physical_controls"), dtype=float)
        if not self.check(actions.shape == (1, len(indices)), "action_shape", path, (1, len(indices)), actions.shape):
            return
        if not self.check(np.isfinite(actions).all() and np.all(abs(actions) <= 1), "action_bounds", path, "finite [-1,1]", actions):
            return
        low, high = map(np.asarray, (manifest["physical_low"], manifest["physical_high"]))
        decoded = np.tile(TARGET_CONTROLS, (35, 1))
        decoded[:, indices] = low[indices] + (actions+1)/2*(high[indices]-low[indices])
        self.array(controls, decoded, Path(path) / "physical_controls")
        if "parameter_nrmse" in record:
            self.scalar(record["parameter_nrmse"], parameter_nrmse(decoded, TARGET_CONTROLS, low, high, indices), Path(path) / "parameter_nrmse")

    def rollout(self, directory, point, condition, target, manifest):
        directory = Path(directory)
        path = directory / "rollout.npz"
        try:
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: stored[name].copy() for name in ("target", "audio", "controls", "actions")}
                rate = int(stored["rate"])
        except (OSError, ValueError, KeyError) as exc:
            self.check(False, "rollout_read", path, "complete NPZ", str(exc))
            return
        self.counts["rollouts_checked"] += 1
        expected_shapes = dict(target=(17500,), audio=(17500,), controls=(35, 5), actions=(1, 1 if condition == "constant_1d" else 2))
        valid = True
        for name, value in arrays.items():
            valid &= self.check(value.shape == expected_shapes[name] and value.dtype == np.float64 and np.isfinite(value).all(),
                                "rollout_array", path / name,
                                dict(shape=expected_shapes[name], dtype="float64", finite=True),
                                dict(shape=value.shape, dtype=str(value.dtype), finite=bool(np.isfinite(value).all())))
        self.array(arrays["target"], target, path / "target")
        self.scalar(rate, 50000, path / "rate")
        self.array(arrays["controls"], point.get("physical_controls"), path / "controls")
        self.array(arrays["actions"], point.get("actions"), path / "actions")
        format_path = directory / "audio_format.json"
        audio_format = self.read_json(format_path)
        if isinstance(audio_format, dict):
            self.check(audio_format.get("independent_gain_normalization") is False,
                       "audio_gain_policy", format_path, False, audio_format.get("independent_gain_normalization"))
            self.check(audio_format.get("sap_evaluated") is False,
                       "sap_not_computed", format_path, False, audio_format.get("sap_evaluated"))
            self.scalar(audio_format.get("sample_rate"), rate, format_path / "sample_rate")
            self.scalar(audio_format.get("target_peak"), float(np.max(abs(arrays["target"]))), format_path / "target_peak")
            self.scalar(audio_format.get("generated_peak"), float(np.max(abs(arrays["audio"]))), format_path / "generated_peak")
        for name, key in (("target", "target"), ("generated", "audio")):
            wav_path = directory / f"{name}_raw.wav"
            try:
                wav_rate, wave = wavfile.read(wav_path)
                same = wave.dtype == np.float32 and wav_rate == rate and np.array_equal(wave, arrays[key].astype(np.float32))
                self.check(same, "wav_floatcast", wav_path, "unaltered float32 cast of NPZ", dict(dtype=str(wave.dtype), rate=wav_rate, exact=bool(same)))
                self.counts["wav_files_checked"] += 1
            except (OSError, ValueError) as exc:
                self.check(False, "wav_read", wav_path, "float32 WAV", str(exc))
        if valid:
            recomputed = distance(target, arrays["audio"], rate)
            self.scalar(point.get("acoustic_error"), recomputed["total"], path / "acoustic_error")
            self.numeric_dict(point.get("acoustic_components"), recomputed, path / "acoustic_components")
            self.scalar(point.get("negative_return"), recomputed["total"], path / "negative_return")
            self.counts["acoustic_recomputations"] += 1
            balanced = self.objective.score(arrays["audio"])
            self.scalar(point.get("balanced_error"), balanced["total"], path / "balanced_error")
            self.numeric_dict(point.get("balanced_components"), balanced, path / "balanced_components")
            self.counts["balanced_recomputations"] += 1

    def run(self):
        start = time.perf_counter()
        try:
            self._run()
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            self.check(False, "schema_or_io_error", self.output, "complete valid experiment", f"{type(exc).__name__}: {exc}")
        return dict(passed=not self.errors, errors=self.errors, checks=self.checks,
                    counts=self.counts, wall_s=time.perf_counter()-start,
                    historical_1d_common_checkpoint_comparisons=self.historical_comparisons,
                    audited_utc=datetime.now(timezone.utc).isoformat(),
                    tolerances=dict(scalar_atol=1e-11, scalar_rtol=1e-9,
                                    waveform_and_wav="exact equality"),
                    limitations=["No training or model integration was performed by this audit.",
                                 "Only saved initial/final/best_observed audio has independently recomputed acoustic metrics; unsaved curve audio is unavailable."])

    def _run(self):
        manifest_path = self.output / "manifest.json"
        manifest = self.read_json(manifest_path)
        if not isinstance(manifest, dict):
            self.check(False, "manifest_schema", manifest_path, "JSON object", type(manifest).__name__)
            return
        if not self.check(manifest.get("workflow") == "issue13_constant_control_reproduction",
                          "workflow_identity", manifest_path,
                          "issue13_constant_control_reproduction", manifest.get("workflow")):
            return
        for name, expected in (("duration_s", .35), ("sample_rate", 50000),
                               ("native_dt", 1e-7), ("control_interval_s", .01)):
            self.scalar(manifest.get(name), expected, manifest_path / name)
        self.array(manifest.get("target_controls"), TARGET_CONTROLS, manifest_path / "target_controls")
        self.array(manifest.get("fixed_controls"), TARGET_CONTROLS, manifest_path / "fixed_controls")
        self.array(manifest.get("physical_low"), LOW, manifest_path / "physical_low")
        self.array(manifest.get("physical_high"), HIGH, manifest_path / "physical_high")
        self.array(manifest.get("initial_state"), np.zeros(4), manifest_path / "initial_state")
        self.check(manifest.get("condition") == "A", "model_case", manifest_path, "A", manifest.get("condition"))
        self.check(manifest.get("delay_history") == "all zeros" and manifest.get("decimator_state") == "all zeros",
                   "initial_history", manifest_path, "zero delay and filter histories",
                   dict(delay=manifest.get("delay_history"), filter=manifest.get("decimator_state")))
        self.hashes(manifest.get("source_hashes"), ROOT, manifest_path)
        self.hashes(manifest.get("source_hashes"), self.output / "source_snapshot", manifest_path / "source_snapshot")
        self.check(_digest(LIBRARY) == manifest.get("native_library_sha256"), "native_library_hash", LIBRARY,
                   manifest.get("native_library_sha256"), _digest(LIBRARY))
        settings = manifest["settings"]
        self.check(settings == self.read_json(ROOT / "configs/issue13_reproduction.json"),
                   "frozen_settings", manifest_path, "identical frozen source configuration", settings)
        self.check(settings == self.read_json(self.output / "config.json"),
                   "copied_settings", self.output / "config.json", settings, None)
        historical_manifest = self.read_json(ROOT / "results/phase1_validation_20261031/manifest.json")
        unchanged = ("warmup_episodes", "learning_rate", "batch_size", "net_arch", "gamma",
                     "sac_ent_coef", "td3_noise_std", "de_population", "de_mutation", "de_recombination")
        for key in unchanged:
            self.check(settings.get(key) == historical_manifest["settings"].get(key),
                       "historical_hyperparameter", manifest_path / key,
                       historical_manifest["settings"].get(key), settings.get(key))
        self.check(settings.get("seeds") == list(SEEDS), "seed_set", manifest_path, SEEDS, settings.get("seeds"))
        self.check(set(settings.get("methods", [])) == set(METHODS), "method_set", manifest_path, METHODS, settings.get("methods"))
        self.check(set(settings.get("conditions", [])) == set(CONDITIONS), "condition_set", manifest_path, CONDITIONS, settings.get("conditions"))
        self.scalar(settings.get("episodes"), BUDGET, manifest_path / "episodes")
        self.scalar(settings.get("gamma"), 1., manifest_path / "gamma")
        target_path = self.output / "diagnostics/target.npz"
        with np.load(target_path, allow_pickle=False) as data:
            target, controls = data["target"].copy(), data["controls"].copy()
            rate = int(data["rate"])
        self.check(target.shape == (17500,) and target.dtype == np.float64 and np.isfinite(target).all(),
                   "target_schema", target_path, "finite float64 array,17500 samples", dict(shape=target.shape, dtype=str(target.dtype)))
        self.array(controls, np.tile(TARGET_CONTROLS, (35, 1)), target_path / "controls")
        self.scalar(rate, 50000, target_path / "rate")
        self.objective = BalancedObjective(target, rate, scale=max(float(abs(target).max()), 1e-8))
        historical_target = ROOT / "results/phase1_validation_20261031/diagnostics/target.npz"
        self.check(_digest(target_path) == _digest(historical_target), "historical_target_hash", target_path,
                   _digest(historical_target), _digest(target_path))
        before_path = self.output / "protected_before.json"
        protected = self.read_json(before_path)
        self.hashes(protected, ROOT.parent, before_path, protected=True)
        expected = {self.output / "runs" / condition / method / f"seed_{seed}" / "summary.json": (condition, method, seed)
                    for condition in CONDITIONS for method in METHODS for seed in SEEDS}
        found = set((self.output / "runs").glob("*/*/seed_*/summary.json"))
        self.counts["runs_found"] = len(found)
        self.check(found == set(expected), "run_set", self.output / "runs", 24, len(found))
        for path, identity in expected.items():
            try:
                self.audit_run(path, *identity, target, manifest)
            except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                self.check(False, "run_schema_or_io_error", path, "complete run", f"{type(exc).__name__}: {exc}")
        self.oracles()

    def oracles(self):
        path = self.output / "diagnostics/oracles.json"
        records = self.read_json(path)
        if not self.check(isinstance(records, list), "oracle_schema", path, "four records", type(records).__name__):
            return
        expected = {(condition, dtype) for condition in CONDITIONS for dtype in ("float64", "float32")}
        found = {(record.get("condition"), record.get("dtype")) for record in records}
        self.check(len(records) == 4 and found == expected, "oracle_set", path, sorted(expected), sorted(found))
        for record in records:
            dtype = record.get("dtype")
            tolerance = 1e-10 if dtype == "float64" else 1e-5
            error = record.get("acoustic_error")
            self.check(close_scalar(error, error) and 0 <= error <= tolerance, "oracle_tolerance", path,
                       dict(dtype=dtype, max_error=tolerance), error)
            if dtype == "float64":
                self.scalar(record.get("max_waveform_abs_error"), 0., path / "max_waveform_abs_error")
                self.check(record.get("bitwise_equal") is True, "oracle_bitwise", path, True, record.get("bitwise_equal"))
                directory = self.output / "diagnostics" / f"oracle_{record['condition']}"
                with np.load(directory / "rollout.npz", allow_pickle=False) as archive:
                    point = {**record, "actions": archive["actions"].tolist(), "physical_controls": archive["controls"].tolist()}
                    target = archive["target"].copy()
                    audio = archive["audio"].copy()
                self.check(np.array_equal(audio, target), "oracle_waveform_exact", directory,
                           "bitwise-equal audio and target", bool(np.array_equal(audio, target)))
                self.scalar(float(np.max(abs(audio-target))), 0., directory / "actual_max_waveform_abs_error")
                manifest = self.read_json(self.output / "manifest.json")
                with np.load(self.output / "diagnostics/target.npz", allow_pickle=False) as archive:
                    self.array(target, archive["target"], directory / "target")
                self.control_record(point, record["condition"], manifest, directory)
                self.rollout(directory, point, record["condition"], target, manifest)
            self.accounting(record.get("compute"), 1, path / record.get("condition", "unknown") / str(dtype))
            self.counts["oracles_checked"] += 1

    def hashes(self, mapping, base, path, *, protected=False):
        if not self.check(isinstance(mapping, dict) and bool(mapping), "hash_schema", path, "nonempty SHA256 mapping", type(mapping).__name__):
            return
        base = Path(base).resolve()
        for relative, expected in mapping.items():
            destination = (base / relative).resolve()
            if not self.check(not Path(relative).is_absolute() and destination.is_relative_to(base), "hash_scope", path, "relative path within repository", relative):
                continue
            try:
                actual = _digest(destination)
                self.check(actual == expected, "protected_changed" if protected else "source_changed", destination, expected, actual)
                self.counts["protected_files_checked" if protected else "source_hashes_checked"] += 1
            except OSError as exc:
                self.check(False, "hash_read", destination, expected, str(exc))

    def audit_run(self, path, condition, method, seed, target, manifest):
        summary = self.read_json(path)
        curve_path = path.parent / "curve.json"
        curve = self.read_json(curve_path)
        if not isinstance(summary, dict) or not isinstance(curve, list) or not curve:
            self.check(False, "run_schema", path, "summary dict and nonempty curve list", None)
            return
        identity = (summary.get("condition"), summary.get("method"), summary.get("seed"))
        self.check(identity == (condition, method, seed), "run_identity", path, (condition, method, seed), identity)
        self.check(summary.get("source_hashes") == manifest["source_hashes"], "run_source_hashes", path, manifest["source_hashes"], summary.get("source_hashes"))
        self.scalar(summary.get("requested_training_episodes"), BUDGET, path / "requested_training_episodes")
        self.scalar(summary.get("optimizer_dimension"), 1 if condition == "constant_1d" else 2, path / "optimizer_dimension")
        self.check([point.get("episodes") for point in curve] == list(range(BUDGET+1)), "curve_schedule", curve_path, "episodes 0..256", [p.get("episodes") for p in curve])
        self.counts["curves_checked"] += 1
        self.counts["curve_points"] += len(curve)
        updates = []
        best_point = None
        for point in curve:
            n = point["episodes"]
            label = curve_path / str(n)
            phase = ("initial" if n == 0 else
                     "post_final_update" if method in ("TD3", "SAC") and n == BUDGET else
                     "pre_gradient_update" if method in ("TD3", "SAC") else
                     "candidate_best_so_far")
            self.check(point.get("evaluation_phase") == phase, "evaluation_phase", label, phase, point.get("evaluation_phase"))
            self.scalar(point.get("decision_steps"), n, label / "decision_steps")
            self.scalar(point.get("integration_s"), n*.35, label / "integration_s")
            self.scalar(point.get("negative_return"), point.get("acoustic_error"), label / "negative_return", "return_objective_identity")
            self.scalar(point.get("acoustic_components", {}).get("total"), point.get("acoustic_error"), label / "acoustic_components" / "total")
            self.check(close_scalar(point.get("acoustic_error"), point.get("acoustic_error")) and point["acoustic_error"] >= 0, "finite_nonnegative_error", label, "finite >=0", point.get("acoustic_error"))
            self.control_record(point, condition, manifest, label)
            update = point.get("gradient_updates")
            self.check(isinstance(update, int) and update >= 0, "gradient_update_count", label, "nonnegative integer", update)
            expected_update = (max(0, n-manifest["settings"]["warmup_episodes"]-(n != BUDGET))
                               if method in ("TD3", "SAC") else 0)
            self.scalar(update, expected_update, label / "gradient_updates", "gradient_update_schedule")
            updates.append(update)
            if best_point is None or point["acoustic_error"] < best_point["acoustic_error"]:
                best_point = point
            self.scalar(point.get("best_observed_error"), best_point["acoustic_error"], label / "best_observed_error", "observed_prefix_minimum")
            if method in ("random_search", "DE") and n > 0:
                candidate = dict(actions=point.get("candidate_actions"), physical_controls=point.get("candidate_physical_controls"))
                self.control_record(candidate, condition, manifest, label / "candidate")
                self.scalar(point.get("candidate_components", {}).get("total"), point.get("candidate_error"), label / "candidate_components" / "total")
                if n == 1:
                    self.array(point["candidate_actions"], curve[0]["actions"], label / "counted_initial_candidate")
                    self.scalar(point["candidate_error"], curve[0]["acoustic_error"], label / "initial_candidate_error")
                running_min = min(curve[n-1]["acoustic_error"], point["candidate_error"])
                self.scalar(point["acoustic_error"], running_min, label / "acoustic_error", "best_so_far_identity")
                if point["candidate_error"] < curve[n-1]["acoustic_error"]:
                    self.array(point["actions"], point["candidate_actions"], label / "best_actions")
                else:
                    self.array(point["actions"], curve[n-1]["actions"], label / "unchanged_best_actions")
        self.check(all(isinstance(n, int) for n in updates) and updates == sorted(updates), "gradient_updates_monotonic", curve_path, "nondecreasing integers", updates)
        self.scalar(summary.get("gradient_updates"), updates[-1], path / "gradient_updates")
        self.scalar(summary.get("optimizer_nfev"), 0 if method in ("TD3", "SAC") else BUDGET, path / "optimizer_nfev")
        for stage, point, directory in (("initial", curve[0], path.parent / "initial"),
                                         ("final", curve[-1], path.parent),
                                         ("best_observed", best_point, path.parent / "best_observed")):
            self.numeric_dict(summary.get(stage), point, path / stage)
            self.rollout(directory, point, condition, target, manifest)
        self.scalar(summary.get("best_observed_episode"), best_point["episodes"], path / "best_observed_episode")
        for stage, episodes in (("training", BUDGET), ("evaluation", BUDGET+1 if method in ("TD3", "SAC") else 2)):
            self.accounting(summary.get(stage), episodes, path / stage)
            self.counts[f"{stage}_episodes"] += summary[stage]["episodes"]
            self.counts[f"{stage}_integration_s"] += summary[stage]["integration_s"]
        if method in ("TD3", "SAC"):
            self.policy_prediction(path.parent / "agent.zip", curve[-1], condition, method)
            self.policy_prediction(path.parent / "best_observed/agent.zip", best_point, condition, method)
        if condition == "constant_1d":
            reference_path = ROOT / "results/phase1_validation_20261031/runs/A" / method / f"seed_{seed}" / "curve.json"
            reference = self.read_json(reference_path)
            lookup = {point["episodes"]: point for point in curve}
            if isinstance(reference, list) and reference:
                differences = [dict(episodes=point["episodes"],
                                    absolute_acoustic_error_difference=abs(lookup[point["episodes"]]["acoustic_error"]-point["acoustic_error"]))
                               for point in reference]
                self.historical_comparisons.append(dict(method=method, seed=seed,
                    reference=str(reference_path.relative_to(ROOT.parent)),
                    common_evaluation_points=len(differences),
                    max_absolute_acoustic_error_difference=max(point["absolute_acoustic_error_difference"] for point in differences),
                    exactly_equal_all_common_acoustic_errors=all(point["absolute_acoustic_error_difference"] == 0 for point in differences),
                    role="descriptive reproduction diagnostic; not an audit pass criterion",
                    differences=differences))

    def policy_prediction(self, path, point, condition, method):
        # The reset observation is known analytically: reset time=-1, previous
        # action and the four labial state values all zero. Loading a saved SB3
        # policy with no environment does not start a native model.
        try:
            import torch
            from stable_baselines3 import SAC, TD3
            torch.set_num_threads(1)
            dimensions = 1 if condition == "constant_1d" else 2
            observation = np.r_[-1., np.zeros(dimensions+4)].astype(np.float32)
            policy = {"TD3": TD3, "SAC": SAC}[method].load(path, device="cpu")
            action, _ = policy.predict(observation, deterministic=True)
            self.array(np.asarray(action, dtype=np.float64).reshape(1, -1), point["actions"], path / "predicted_action")
            self.counts["policy_predictions_checked"] += 1
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.check(False, "policy_predict", path, "loadable deterministic policy matching saved action", str(exc))


def audit(output):
    """Write only the new Issue #13 directory's independent audit result."""
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("output must lie within Deeplearning/results")
    manifest_path = output / "manifest.json"
    if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("workflow") != "issue13_constant_control_reproduction":
        raise ValueError("only an Issue #13 experiment may receive final_audit.json")
    result = Auditor(output).run()
    (output / "final_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({name: result[name] for name in ("passed", "checks", "counts", "wall_s")}))
    if not result["passed"]:
        raise RuntimeError(f"Issue #13 independent audit failed with {len(result['errors'])} errors")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        parser.error("output must lie within Deeplearning/results")
    manifest_path = output / "manifest.json"
    if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("workflow") != "issue13_constant_control_reproduction":
        parser.error("only an Issue #13 experiment may receive final_audit.json")
    audit(output)


if __name__ == "__main__":
    main()
