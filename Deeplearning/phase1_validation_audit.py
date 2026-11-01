"""Independently audit a completed synthetic Phase 1 result directory.

Usage: python -m Deeplearning.phase1_validation_audit RESULTS_DIRECTORY

Only the supplied experiment's artifacts and the four manifest-listed source
files are read. Historical results are not rescanned: their preservation is
checked using the preservation_check.json produced by the experiment runner.
No simulation, training, recording access or existing artifact rewrite occurs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import wavfile

from .acoustics import distance
from .model import ROOT
from .objectives import BalancedObjective


TASKS = ("A", "B", "C")
METHODS = ("TD3", "SAC", "random_search", "DE")
SOURCE_NAMES = ("phase1_validation.py", "phase1_validation_env.py",
                "phase1_validation_report.py", "configs/phase1_validation.json")
METRICS = ("acoustic_error", "balanced_error", "parameter_nrmse")


def close_scalar(actual, expected, *, atol=1e-11, rtol=1e-9):
    """Finite scalar comparison; NaN, infinity and nonnumeric data fail."""
    try:
        actual, expected = float(actual), float(expected)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(actual) and np.isfinite(expected)
                and abs(actual-expected) <= atol+rtol*abs(expected))


def expected_schedule(budget, interval):
    """Initial evaluation and declared checkpoints, with one final endpoint."""
    if not isinstance(budget, int) or not isinstance(interval, int) or min(budget, interval) < 1:
        raise ValueError("Budget and interval must be positive integers")
    return [0, *range(interval, budget, interval), budget]


def parameter_nrmse(controls, target, low, high, indices):
    controls = np.asarray(controls, dtype=float)
    target, low, high = map(lambda x: np.asarray(x, dtype=float), (target, low, high))
    indices = list(indices)
    if controls.ndim != 2 or controls.shape[1] != 5 or not len(controls):
        raise ValueError("Expected nonempty N-by-5 physical controls")
    if any(x.shape != (5,) for x in (target, low, high)) or not indices:
        raise ValueError("Expected five physical bounds and selected dimensions")
    if (not np.isfinite(controls).all() or not np.isfinite(target).all()
            or not np.isfinite(low).all() or not np.isfinite(high).all()
            or np.any(high <= low) or any(i not in range(5) for i in indices)):
        raise ValueError("Expected finite controls and increasing physical bounds")
    return float(np.sqrt(np.mean(((controls[:, indices]-target[indices]) /
                                  (high[indices]-low[indices]))**2)))


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class Auditor:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.errors = []
        self.checked = 0
        self.counts = dict(runs_expected=0, runs_found=0, curves_checked=0,
                           curve_points=0, rollouts_checked=0, metric_recomputations=0,
                           wav_files_checked=0, baselines_checked=0, oracles_checked=0,
                           source_hashes_checked=0, training_episodes=0,
                           evaluation_episodes=0, training_integration_s=0.0,
                           evaluation_integration_s=0.0, landscape_candidates=0,
                           cross_task_replays_checked=0, supplemental_evaluation_episodes=0,
                           supplemental_evaluation_integration_s=0.0, input_hashes_checked=0)

    def check(self, condition, code, path, expected=None, actual=None):
        self.checked += 1
        if not condition:
            path = Path(path)
            label = str(path.relative_to(self.output)) if path.is_relative_to(self.output) else str(path)
            self.errors.append(_plain(dict(code=code, path=label, expected=expected, actual=actual)))
        return bool(condition)

    def read_json(self, path):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            self.check(False, "json_read", path, "readable valid JSON", str(exc))
            return None
        self.check(True, "json_read", path)
        return data

    def scalar(self, actual, expected, path, code="numeric_mismatch"):
        return self.check(close_scalar(actual, expected), code, path, expected, actual)

    def input_hashes(self, hashes, path, expected_keys=None):
        """Check recorded input hashes, allowing reads only inside this experiment."""
        if not self.check(isinstance(hashes, dict) and bool(hashes), "input_hash_schema", path,
                          "nonempty relative input-to-SHA256 mapping", type(hashes).__name__):
            return
        if expected_keys is not None:
            self.check(set(hashes) == set(expected_keys), "input_hash_set", path,
                       sorted(expected_keys), sorted(hashes))
        for relative, expected in hashes.items():
            destination = (self.output / relative).resolve()
            if not self.check(not Path(relative).is_absolute() and destination.is_relative_to(self.output),
                              "input_hash_scope", path, "input inside this experiment", relative):
                continue
            try:
                actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            except OSError as exc:
                self.check(False, "input_hash_read", destination, expected, str(exc))
                continue
            self.counts["input_hashes_checked"] += 1
            self.check(actual == expected, "input_hash_changed", destination, expected, actual)

    def metric_dict(self, actual, expected, path):
        if not self.check(isinstance(actual, dict), "metrics_schema", path, "dict", type(actual).__name__):
            return
        for key, value in expected.items():
            if isinstance(value, dict):
                self.metric_dict(actual.get(key), value, Path(path) / key)
            elif isinstance(value, bool):
                self.check(actual.get(key) is value, "metric_flag", Path(path) / key, value, actual.get(key))
            else:
                self.scalar(actual.get(key), value, Path(path) / key)

    def accounting(self, record, episodes, horizon, path, duration, control_s, dt):
        if not self.check(isinstance(record, dict), "accounting_schema", path, "dict", type(record).__name__):
            return
        wanted = dict(episodes=episodes, decision_steps=episodes*horizon,
                      integration_s=episodes*duration, integration_completed_s=episodes*duration,
                      native_blocks=round(episodes*duration/control_s),
                      native_steps=round(episodes*duration/dt), failures=0)
        for name, value in wanted.items():
            self.scalar(record.get(name), value, Path(path) / name, "accounting_mismatch")
        wall = record.get("wall_s")
        self.check(close_scalar(wall, wall) and wall >= 0, "wall_time", Path(path) / "wall_s", ">=0 finite", wall)

    def rollout(self, directory, saved_metrics, task, target, objective, manifest):
        path = Path(directory) / "rollout.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                stored_target, audio, controls, actions = (data[k].copy() for k in ("target", "audio", "controls", "actions"))
                rate = int(data["rate"])
        except (OSError, ValueError, KeyError) as exc:
            self.check(False, "npz_read", path, "complete rollout archive", str(exc))
            return
        self.counts["rollouts_checked"] += 1
        rate_expected = manifest["sample_rate"]
        horizon, indices = (1 if task == "A" else 35), ([0, 2] if task == "C" else [0])
        shapes_valid = True
        for name, array, shape in (("target", stored_target, target.shape), ("audio", audio, target.shape),
                                   ("controls", controls, (35, 5)), ("actions", actions, (horizon, len(indices)))):
            shapes_valid &= self.check(array.shape == shape, "array_shape", path / name, shape, array.shape)
            self.check(np.isfinite(array).all(), "nonfinite_array", path / name, "all finite", bool(np.isfinite(array).all()))
            self.check(array.dtype == np.float64, "array_dtype", path / name, "float64", str(array.dtype))
        self.check(np.array_equal(stored_target, target), "target_changed", path / "target", "exact diagnostics target", "equal" if np.array_equal(stored_target, target) else "different")
        self.check(rate == rate_expected, "sample_rate", path / "rate", rate_expected, rate)
        for name, values in (("target", stored_target), ("generated", audio)):
            wav_path = Path(directory) / f"{name}_raw.wav"
            try:
                wav_rate, wave = wavfile.read(wav_path)
            except (OSError, ValueError) as exc:
                self.check(False, "wav_read", wav_path, "IEEE float32 WAV", str(exc))
                continue
            self.counts["wav_files_checked"] += 1
            self.check(wave.dtype == np.float32, "wav_dtype", wav_path, "float32", str(wave.dtype))
            self.check(wav_rate == rate_expected, "wav_rate", wav_path, rate_expected, wav_rate)
            same = np.array_equal(wave, values.astype(np.float32))
            self.check(same, "wav_gain_or_clipping", wav_path, "exact npz waveform cast to float32", "equal" if same else "different")
        if not shapes_valid or not all(np.isfinite(x).all() for x in (audio, controls, actions)):
            return
        low, high, true = map(np.asarray, (manifest["physical_low"], manifest["physical_high"], manifest["target_controls"]))
        self.check(bool(np.all((controls >= low) & (controls <= high))), "physical_bounds", path / "controls", "within physical bounds", "valid" if np.all((controls >= low) & (controls <= high)) else "outside")
        fixed = [i for i in range(5) if i not in indices]
        self.check(np.array_equal(controls[:, fixed], np.tile(true[fixed], (35, 1))), "fixed_controls", path / "controls", "nonlearned controls equal target", "equal" if np.array_equal(controls[:, fixed], np.tile(true[fixed], (35, 1))) else "different")
        self.check(bool(np.all((actions >= -1) & (actions <= 1))), "action_bounds", path / "actions", "[-1,1]", "valid" if np.all((actions >= -1) & (actions <= 1)) else "outside")
        physical = np.tile(true, (35, 1))
        action_blocks = np.repeat(actions, 35, axis=0) if task == "A" else actions
        physical[:, indices] = low[indices]+(action_blocks+1)/2*(high[indices]-low[indices])
        self.check(np.array_equal(physical, controls), "action_decode", path / "controls", "exact decoded actions with target fixed values", "equal" if np.array_equal(physical, controls) else "different")
        acoustic = distance(target, audio, rate_expected)
        balanced = objective.score(audio)
        recomputed = dict(acoustic_error=acoustic["total"], balanced_error=balanced["total"],
                          parameter_nrmse=parameter_nrmse(controls, true, low, high, indices),
                          acoustic_components=acoustic, balanced_components=balanced, failed=False)
        self.metric_dict(saved_metrics, recomputed, directory)
        self.scalar(saved_metrics.get("negative_return"), acoustic["total"], Path(directory) / "negative_return", "return_objective_identity")
        self.counts["metric_recomputations"] += 1

    def run(self):
        start = time.perf_counter()
        try:
            return self._run()
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            self.check(False, "audit_schema_or_io_error", self.output,
                       "complete valid experiment schema", f"{type(exc).__name__}: {exc}")
            return self.finish(start)

    def _run(self):
        start = time.perf_counter()
        manifest = self.read_json(self.output / "manifest.json")
        if manifest is None:
            return self.finish(start)
        if not self.check(set(manifest.get("source_hashes", {})) == set(SOURCE_NAMES), "workflow_identity", self.output / "manifest.json", SOURCE_NAMES, list(manifest.get("source_hashes", {}))):
            return self.finish(start)
        settings = manifest["settings"]
        self.check(len(settings["seeds"]) >= 3 and len(set(settings["seeds"])) == len(settings["seeds"]),
                   "three_seeds", self.output / "manifest.json", "at least 3 distinct seeds", settings["seeds"])
        self.check(set(settings["methods"]) == set(METHODS), "four_methods", self.output / "manifest.json", METHODS, settings["methods"])
        self.check(set(settings["tasks"]) == set(TASKS), "three_tasks", self.output / "manifest.json", TASKS, list(settings["tasks"]))
        self.scalar(settings.get("gamma"), 1, self.output / "manifest.json" / "gamma")
        for name in SOURCE_NAMES:
            source = ROOT / name
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            self.check(actual == manifest["source_hashes"][name], "source_hash_changed", source, manifest["source_hashes"][name], actual)
            self.counts["source_hashes_checked"] += 1
        gate_path = self.output / "diagnostics_review.json"
        gate = self.read_json(gate_path)
        reviewed = None
        if gate:
            self.check(gate.get("passed") is True and bool(gate.get("checks")) and all(gate["checks"].values()), "diagnostic_gate", gate_path, "all checks passed", gate.get("passed"))
            try:
                created = datetime.fromisoformat(manifest["created_utc"])
                reviewed = datetime.fromisoformat(gate["reviewed_utc"])
                self.check(created.tzinfo is not None and reviewed.tzinfo is not None and created <= reviewed, "gate_timestamp", gate_path, "created <= reviewed, timezone-aware", dict(created=created.isoformat(), reviewed=reviewed.isoformat()))
            except (KeyError, TypeError, ValueError) as exc:
                self.check(False, "gate_timestamp", gate_path, "valid timestamp", str(exc))
        preservation_path = self.output / "preservation_check.json"
        preservation = self.read_json(preservation_path)
        if preservation:
            self.check(preservation.get("passed") is True and preservation.get("changed") == [] and preservation.get("files_checked", 0) > 0,
                       "protected_results", preservation_path, "PASS, changed=[], files_checked>0", preservation)
            before = self.read_json(self.output / "protected_before.json")
            if isinstance(before, dict):
                self.scalar(preservation.get("files_checked"), len(before), preservation_path / "files_checked")
        target_path = self.output / "diagnostics/target.npz"
        try:
            with np.load(target_path, allow_pickle=False) as data:
                target = data["target"].copy()
                target_controls = data["controls"].copy()
                target_rate = int(data["rate"])
            wanted_shape = (round(manifest["duration_s"]*manifest["sample_rate"]),)
            self.check(target.shape == wanted_shape and target.dtype == np.float64 and np.isfinite(target).all(), "diagnostics_target", target_path, dict(shape=wanted_shape, dtype="float64", finite=True), dict(shape=target.shape, dtype=str(target.dtype), finite=bool(np.isfinite(target).all())))
            self.check(target_rate == manifest["sample_rate"], "diagnostics_target_rate", target_path, manifest["sample_rate"], target_rate)
            self.check(np.array_equal(target_controls, np.tile(manifest["target_controls"], (35, 1))), "diagnostics_controls", target_path, "35 exact target controls", "equal" if np.array_equal(target_controls, np.tile(manifest["target_controls"], (35, 1))) else "different")
            objective = BalancedObjective(target, manifest["sample_rate"], scale=max(float(abs(target).max()), 1e-8))
        except (OSError, ValueError, KeyError) as exc:
            self.check(False, "diagnostics_target_read", target_path, "valid synthetic target", str(exc))
            return self.finish(start)
        if gate:
            for name in ("1d", "2d"):
                map_path = self.output / "diagnostics" / f"landscape_{name}.csv"
                try:
                    with map_path.open(newline="") as stream:
                        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
                    finite = bool(rows) and all(np.isfinite(list(row.values())).all() for row in rows)
                    self.check(finite, "landscape_finite", map_path, "all finite, nonempty", finite)
                    self.counts["landscape_candidates"] += len(rows)
                    declared = gate["maps"][name]
                    self.scalar(declared.get("candidates"), len(rows), map_path / "candidates")
                    self.scalar(declared.get("integration_s"), len(rows)*manifest["duration_s"], map_path / "integration_s")
                    self.scalar(declared.get("native_steps"), round(len(rows)*manifest["duration_s"]/manifest["native_dt"]), map_path / "native_steps")
                    if rows:
                        minimum = min(row["acoustic_error"] for row in rows)
                        self.scalar(declared.get("minimum_acoustic_error"), minimum, map_path / "minimum")
                        probes = [r for r in rows if r["epsilon_l"] == manifest["target_controls"][0] and r["ps"] == manifest["target_controls"][2]]
                        self.check(len(probes) == 1 and probes[0]["acoustic_error"] <= settings["oracle_tolerance_float64"], "landscape_known_target", map_path, "one zero-error target probe", probes)
                except (OSError, ValueError, KeyError) as exc:
                    self.check(False, "landscape_read", map_path, "complete landscape CSV", str(exc))
        expected_paths = {self.output / "runs" / task / method / f"seed_{seed}" / "summary.json": (task, method, seed)
                          for task in TASKS for method in METHODS for seed in settings["seeds"]}
        found = set((self.output / "runs").glob("*/*/seed_*/summary.json"))
        self.counts.update(runs_expected=len(expected_paths), runs_found=len(found))
        self.check(found == set(expected_paths), "complete_run_set", self.output / "runs", sorted(str(p.relative_to(self.output)) for p in expected_paths), sorted(str(p.relative_to(self.output)) for p in found))
        for path, (task, method, seed) in expected_paths.items():
            summary = self.read_json(path)
            if summary is None:
                continue
            self.check((summary.get("task"), summary.get("method"), summary.get("seed")) == (task, method, seed), "run_identity", path, (task, method, seed), (summary.get("task"), summary.get("method"), summary.get("seed")))
            self.check(summary.get("source_hashes") == manifest["source_hashes"], "run_source_hashes", path, manifest["source_hashes"], summary.get("source_hashes"))
            self.check(summary.get("optimizer_objective") == "acoustic_error" and summary.get("balanced_role") == "independent audit" and summary.get("reward_profile") == manifest["reward"], "objective_roles", path, "primary acoustic; balanced audit; declared reward", {k: summary.get(k) for k in ("optimizer_objective", "balanced_role", "reward_profile")})
            if reviewed is not None:
                self.check(path.stat().st_mtime >= reviewed.timestamp(), "run_after_gate", path, f">={reviewed.isoformat()}", datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
            budget = settings["tasks"][task]["episodes"]
            interval = settings["tasks"][task]["eval_every_episodes"]
            schedule = expected_schedule(budget, interval)
            horizon = 1 if task == "A" else 35
            self.scalar(summary.get("requested_training_episodes"), budget, path / "requested_training_episodes")
            self.scalar(summary.get("optimizer_dimension"), horizon*(2 if task == "C" else 1), path / "optimizer_dimension")
            self.accounting(summary.get("training"), budget, horizon, path / "training", manifest["duration_s"], manifest["control_interval_s"], manifest["native_dt"])
            self.accounting(summary.get("evaluation"), len(schedule), horizon, path / "evaluation", manifest["duration_s"], manifest["control_interval_s"], manifest["native_dt"])
            for stage in ("training", "evaluation"):
                self.counts[f"{stage}_episodes"] += summary[stage]["episodes"]
                self.counts[f"{stage}_integration_s"] += summary[stage]["integration_s"]
            self.scalar(summary.get("optimizer_nfev"), budget if method in ("DE", "random_search") else 0, path / "optimizer_nfev")
            self.scalar(summary.get("elapsed_s"), summary["training"]["wall_s"]+summary["evaluation"]["wall_s"], path / "elapsed_s")
            curve_path = path.parent / "curve.json"
            curve = self.read_json(curve_path)
            if isinstance(curve, list) and curve:
                self.counts["curves_checked"] += 1
                self.counts["curve_points"] += len(curve)
                self.check([p.get("episodes") for p in curve] == schedule, "checkpoint_schedule", curve_path, schedule, [p.get("episodes") for p in curve])
                for point in curve:
                    n = point["episodes"]
                    self.scalar(point.get("decision_steps"), n*horizon, curve_path / str(n) / "decision_steps")
                    self.scalar(point.get("integration_s"), n*manifest["duration_s"], curve_path / str(n) / "integration_s")
                    self.scalar(point.get("negative_return"), point.get("acoustic_error"), curve_path / str(n), "return_objective_identity")
                for stage, point in (("initial", curve[0]), ("final", curve[-1])):
                    expected = {k: v for k, v in point.items() if k not in ("episodes", "decision_steps", "integration_s")}
                    self.metric_dict(summary.get(f"{stage}_metrics"), expected, path / stage)
            else:
                self.check(False, "curve_schema", curve_path, "nonempty list", type(curve).__name__)
            for stage, directory in (("initial", path.parent / "initial"), ("final", path.parent)):
                self.rollout(directory, summary.get(f"{stage}_metrics", {}), task, target, objective, manifest)
        baseline_path = self.output / "baselines/summary.json"
        baselines = self.read_json(baseline_path)
        if isinstance(baselines, list):
            identities = {(x.get("task"), x.get("seed"), x.get("name")) for x in baselines}
            expected = {(task, seed, name) for task in TASKS for seed in settings["seeds"] for name in ("neutral", "random_control")}
            self.check(identities == expected and len(baselines) == len(expected), "baseline_set", baseline_path, len(expected), len(baselines))
            for record in baselines:
                task, seed, name = record["task"], record["seed"], record["name"]
                directory = self.output / "baselines" / task / name / f"seed_{seed}"
                self.rollout(directory, record, task, target, objective, manifest)
                self.accounting(record.get("compute"), 1, 1 if task == "A" else 35, directory, manifest["duration_s"], manifest["control_interval_s"], manifest["native_dt"])
                self.counts["baselines_checked"] += 1
        oracles = self.read_json(self.output / "diagnostics/oracles.json")
        if isinstance(oracles, list):
            for task in TASKS:
                match = [r for r in oracles if r.get("name") == f"{task}_oracle_float64"]
                if self.check(len(match) == 1, "oracle_record", self.output / "diagnostics/oracles.json", 1, len(match)):
                    self.rollout(self.output / "diagnostics" / f"oracle_{task}", match[0], task, target, objective, manifest)
                    self.counts["oracles_checked"] += 1
        self.cross_task_replay(target, objective, manifest)
        diagnosis_path = self.output / "A_diagnosis.json"
        if diagnosis_path.exists():
            diagnosis = self.read_json(diagnosis_path)
            if isinstance(diagnosis, dict) and "inputs_sha256" in diagnosis:
                self.input_hashes(diagnosis["inputs_sha256"], diagnosis_path)
                self.check(diagnosis.get("inputs_unchanged") is True, "diagnosis_inputs_unchanged", diagnosis_path, True, diagnosis.get("inputs_unchanged"))
                self.scalar(diagnosis.get("integration_s"), 0, diagnosis_path / "integration_s")
                self.scalar(diagnosis.get("training_steps"), 0, diagnosis_path / "training_steps")
        return self.finish(start)

    def cross_task_replay(self, target, objective, manifest):
        """Keep A-to-B hold replays separate from all main training/evaluation."""
        directory = self.output / "cross_task_replay"
        summary_path = directory / "summary.json"
        records = self.read_json(summary_path)
        if not self.check(isinstance(records, list), "replay_schema", summary_path,
                          "3 replay records", type(records).__name__):
            return
        seeds = manifest["settings"]["seeds"]
        expected = {("B", "A_DE", "A_DE_constant_hold", seed) for seed in seeds}
        identities = {(x.get("task"), x.get("source"), x.get("name"), x.get("seed")) for x in records}
        self.check(identities == expected and len(records) == len(expected), "replay_set", summary_path,
                   sorted(expected), sorted(identities))
        for record in records:
            seed = record["seed"]
            replay = directory / f"seed_{seed}"
            source = self.output / "runs/A/DE" / f"seed_{seed}"
            self.rollout(replay, record.get("metrics", {}), "B", target, objective, manifest)
            self.accounting(record.get("compute"), 1, 35, replay / "compute", manifest["duration_s"],
                            manifest["control_interval_s"], manifest["native_dt"])
            wanted_inputs = [str((source / name).relative_to(self.output)) for name in ("rollout.npz", "summary.json")]
            self.input_hashes(record.get("source_inputs_sha256"), replay, wanted_inputs)
            self.check(record.get("source_inputs_unchanged") is True, "replay_inputs_unchanged", replay,
                       True, record.get("source_inputs_unchanged"))
            source_summary = self.read_json(source / "summary.json")
            training = record.get("source_training", {})
            self.check(training.get("task") == "A" and training.get("method") == "DE"
                       and training.get("already_counted_in_main_A_training") is True,
                       "replay_source_training_role", replay, "A DE training already counted", training)
            for quantity in ("episodes", "integration_s"):
                if source_summary:
                    self.scalar(training.get(quantity), source_summary["training"][quantity], replay / "source_training" / quantity)
            self.scalar(training.get("additional_training_episodes"), 0, replay / "additional_training_episodes")
            self.scalar(training.get("additional_training_integration_s"), 0, replay / "additional_training_integration_s")
            self.scalar(record.get("held_control_blocks"), 35, replay / "held_control_blocks")
            try:
                with np.load(source / "rollout.npz", allow_pickle=False) as original, np.load(replay / "rollout.npz", allow_pickle=False) as generated:
                    for key, flag in (("audio", "bitwise_audio_equal_source_A"),
                                      ("controls", "bitwise_controls_equal_source_A")):
                        a, b = original[key], generated[key]
                        equal = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
                        self.check(equal and record.get(flag) is True, "replay_source_equality", replay / key,
                                   "bitwise identical to source A", dict(actual_equal=equal, declared=record.get(flag)))
                    original_action = original["actions"][0]
                    self.check(np.array_equal(np.asarray(record.get("source_action")), original_action),
                               "replay_source_action", replay / "source_action", original_action,
                               record.get("source_action"))
                    self.check(np.array_equal(generated["actions"], np.repeat(original_action[None, :], 35, axis=0)),
                               "replay_constant_hold", replay / "actions", "35 exact copies of source A action", "checked")
                    difference = float(np.max(abs(original["audio"]-generated["audio"])))
                    self.scalar(record.get("max_absolute_audio_difference"), difference, replay / "max_absolute_audio_difference")
            except (OSError, ValueError, KeyError) as exc:
                self.check(False, "replay_source_read", replay, "both source and replay archives", str(exc))
            self.counts["cross_task_replays_checked"] += 1
            self.counts["supplemental_evaluation_episodes"] += record["compute"]["episodes"]
            self.counts["supplemental_evaluation_integration_s"] += record["compute"]["integration_s"]
        metadata_path = directory / "metadata.json"
        metadata = self.read_json(metadata_path)
        if isinstance(metadata, dict):
            self.scalar(metadata.get("additional_training_episodes"), 0, metadata_path / "additional_training_episodes")
            self.scalar(metadata.get("additional_training_integration_s"), 0, metadata_path / "additional_training_integration_s")
            self.scalar(metadata.get("additional_evaluation_episodes"), len(seeds), metadata_path / "additional_evaluation_episodes")
            self.scalar(metadata.get("additional_evaluation_integration_s"), len(seeds)*manifest["duration_s"], metadata_path / "additional_evaluation_integration_s")
            self.scalar(metadata.get("evaluation_wall_s"), sum(r["compute"]["wall_s"] for r in records), metadata_path / "evaluation_wall_s")

    def finish(self, start):
        return dict(passed=not self.errors, errors=self.errors, checks=self.checked,
                    counts=self.counts, wall_s=time.perf_counter()-start,
                    audited_utc=datetime.now(timezone.utc).isoformat(),
                    tolerances=dict(scalar_atol=1e-11, scalar_rtol=1e-9,
                                    waveform_and_wav="bitwise equality"),
                    scope="Only this experiment, plus four frozen source hashes. Historical preservation status is read from runner evidence; no old results or recordings are opened.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        parser.error("output must be the new synthetic experiment below Deeplearning/results")
    result = Auditor(output).run()
    destination = output / "final_audit.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("passed", "checks", "counts", "wall_s")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
