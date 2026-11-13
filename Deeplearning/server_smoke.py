"""One PPO seed, exactly 4096 training transitions, with artifact/reload checks.

Uses the existing make_agent, SineEnv and acoustic reward without changes.
Run from the repository root with ``python -m Deeplearning.server_smoke``.
"""
import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def device_argument(value):
    if value != "cpu" and re.fullmatch(r"cuda:\d+", value) is None:
        raise argparse.ArgumentTypeError("Use cpu or an explicit CUDA device such as cuda:0")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", type=device_argument, default="cpu",
                        help="cpu (default) or cuda:N; CUDA failure never falls back to CPU")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Process-local settings only; keep caches out of shared home directories.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["MPLCONFIGDIR"] = str(out / "mpl-cache")
    os.environ["PYTHONPYCACHEPREFIX"] = str(out / "pycache")
    os.environ["MPLBACKEND"] = "Agg"
    marker = out / ("reload_started.json" if args.evaluate_only else "run_started.json")
    with marker.open("x") as stream:
        json.dump({"started_at": now(), "pid": os.getpid(), "python": sys.executable,
                   "requested_device": args.device}, stream)

    import numpy as np
    import torch
    from torch.overrides import TorchFunctionMode
    from scipy.io import wavfile
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger, make_output_format
    from .experiments import make_agent, phase0_factories, rollout
    from .plotting import comparison, PHASE0_MAX_HZ
    from .sine_env import SineConfig

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    nvidia = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader"], capture_output=True, text=True)
    cpu_model = next((line.split(":", 1)[1].strip() for line in
                      Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), "unknown")
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "stable-baselines3", "gymnasium", "numpy", "scipy", "matplotlib", "SciencePlots")}
    environment = dict(hostname=socket.gethostname(), platform=platform.platform(),
                       python=sys.version, executable=sys.executable, prefix=sys.prefix,
                       base_prefix=sys.base_prefix, cwd=os.getcwd(), pid=os.getpid(), packages=packages,
                       torch_path=torch.__file__, cpu_model=cpu_model,
                       cpu_affinity=sorted(os.sched_getaffinity(0)),
                       torch_threads=torch.get_num_threads(),
                       torch_interop_threads=torch.get_num_interop_threads(),
                       requested_device=args.device,
                       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       nvidia_visible_devices=os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                       nvidia_device_nodes=sorted(str(p) for p in Path("/dev").glob("nvidia*")),
                       torch_cuda_build=torch.version.cuda,
                       cuda_available=torch.cuda.is_available(),
                       cuda_device_count=torch.cuda.device_count(),
                       nvidia_smi_returncode=nvidia.returncode,
                       nvidia_smi_output=nvidia.stdout+nvidia.stderr,
                       matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    environment_path = out / ("reload_environment.json" if args.evaluate_only else "environment.json")
    write_json(environment_path, environment)
    if device.type == "cuda":
        if not environment["cuda_available"] or device.index >= environment["cuda_device_count"]:
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable or index is invalid; CPU fallback is disabled")
        torch.cuda.set_device(device)
        probe = torch.ones(1, device=device)
        assert probe.device == device
        torch.cuda.synchronize(device)
        del probe
        properties = torch.cuda.get_device_properties(device)
        environment.update(gpu_logical_index=device.index, gpu_name=properties.name,
                           gpu_uuid=str(getattr(properties, "uuid", "unavailable")),
                           gpu_total_memory_bytes=properties.total_memory)
        write_json(environment_path, environment)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def verify_placement(agent):
        placements = {name: str(param.device) for name, param in agent.policy.named_parameters()}
        if agent.device != device or any(value != str(device) for value in placements.values()):
            raise RuntimeError(f"Device mismatch: requested {device}, actual {placements}")
        return placements

    def finite(name, value):
        if isinstance(value, torch.Tensor):
            ok = bool(torch.isfinite(value).all())
        else:
            ok = bool(np.isfinite(np.asarray(value)).all())
        if not ok:
            raise FloatingPointError(f"Non-finite {name}")

    def finite_tree(name, value):
        if isinstance(value, dict):
            for key, item in value.items():
                finite_tree(f"{name}/{key}", item)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                finite_tree(f"{name}/{i}", item)
        elif isinstance(value, (int, float, np.number, np.ndarray, torch.Tensor)):
            finite(name, value)

    def policy_hash(agent):
        h = hashlib.sha256()
        for key, value in agent.policy.state_dict().items():
            finite(key, value)
            h.update(key.encode())
            h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def evaluate(agent, factory):
        env = factory()
        try:
            result, audio, actions = rollout(
                env, lambda obs, _: agent.predict(obs, deterministic=True)[0], seed=0)
            finite_tree("evaluation", result)
            for name, value in (("audio", audio), ("actions", actions),
                                ("target", env.target_audio), ("rewards", env.rewards)):
                finite(name, value)
            assert not result["failed"]
            assert result["env_steps"] == env.horizon == 35
            assert len(audio) == len(env.target_audio) == 16800
            controls = np.array([env.decode(a) for a in actions])
            data = dict(target=env.target_audio, audio=audio, actions=actions,
                        generated_controls=controls,
                        target_controls=np.column_stack((env.frequencies, env.amplitudes)),
                        rewards=np.array(env.rewards), sample_rate=env.config.sample_rate)
            return result, data
        finally:
            env.close()

    if args.evaluate_only:
        start = time.perf_counter()
        config = json.loads((out / "config.json").read_text())
        _, eval_factory = phase0_factories(config)
        saved_hash = digest(out / "agent.zip")
        assert config.get("requested_device", "cpu") == args.device
        loaded = PPO.load(out / "agent.zip", device=device)
        placements = verify_placement(loaded)
        environment["parameter_devices"] = placements
        write_json(environment_path, environment)
        result, data = evaluate(loaded, eval_factory)
        with np.load(out / "before_reload.npz") as expected:
            matches = {key: bool(np.array_equal(expected[key], value))
                       for key, value in data.items()}
            differences = {}
            for key, value in data.items():
                reference = expected[key].astype(np.float64)
                delta = np.asarray(value, dtype=np.float64)-reference
                finite(f"reload_difference/{key}", delta)
                differences[key] = dict(max_abs=float(np.max(np.abs(delta))),
                                        mean_abs=float(np.mean(np.abs(delta))),
                                        rmse=float(np.sqrt(np.mean(delta**2))),
                                        allclose=bool(np.allclose(reference, value, rtol=1e-5, atol=1e-6)))
        expected_result = json.loads((out / "before_reload.json").read_text())
        error_difference = abs(result["error"]-expected_result["error"])
        if device.type == "cpu":
            assert all(matches.values()), matches
            assert error_difference == 0
        write_json(out / "reload_comparison.json", dict(
            exact_array_matches=matches, array_differences=differences,
            acoustic_error_before=expected_result["error"], acoustic_error_after=result["error"],
            acoustic_error_abs_difference=error_difference,
            diagnostic_tolerances=dict(rtol=1e-5, atol=1e-6)))
        assert loaded.num_timesteps == 4096
        assert loaded._n_updates == config["expected_epoch_updates"]
        assert policy_hash(loaded) == config["trained_policy_sha256"]
        assert digest(out / "agent.zip") == saved_hash
        np.savez_compressed(out / "evaluation.npz", **data)
        for name in ("target", "audio"):
            filename = "target.wav" if name == "target" else "generated.wav"
            wavfile.write(out / filename, int(data["sample_rate"]), data[name].astype(np.float32))
            rate, read_back = wavfile.read(out / filename)
            assert rate == int(data["sample_rate"])
            np.testing.assert_array_equal(read_back, data[name].astype(np.float32))
        import matplotlib.pyplot as plt
        times = np.arange(36) * .01
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, ("Frequency [Hz]", "Amplitude"))):
            for key, label_curve in (("target_controls", "Target"),
                                     ("generated_controls", "Generated (reloaded PPO)")):
                values = data[key][:, i]
                ax.step(times, np.r_[values, values[-1]], where="post", label=label_curve)
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
            ax.legend()
        axes[0].set_ylim(500, 3000)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlim(0, .35)
        axes[1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(out / "frequency_amplitude.png", dpi=150)
        plt.close(fig)
        comparison(data["target"], data["audio"], int(data["sample_rate"]),
                   out / "spectrogram.png", max_hz=PHASE0_MAX_HZ)
        result.update(reload_passed=True, exact_array_matches=matches,
                      array_differences=differences, acoustic_error_abs_difference=error_difference,
                      policy_sha256=policy_hash(loaded), model_sha256=saved_hash,
                      training_steps_in_saved_model=loaded.num_timesteps,
                      epoch_updates_in_saved_model=loaded._n_updates,
                      policy_devices=sorted({str(p.device) for p in loaded.policy.parameters()}),
                      parameter_devices=placements,
                      pid=os.getpid(), finished_at=now(),
                      reload_evaluation_and_artifacts_seconds=time.perf_counter()-start)
        write_json(out / "evaluation.json", result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return

    start = time.perf_counter()
    config = json.loads((ROOT / "configs" / "phase0.json").read_text())
    config.update(algorithms=["PPO"], seeds=[0], steps=4096,
                  purpose="server functional smoke test, no method comparison",
                  evaluation_seed=0, evaluation_deterministic=True,
                  expected_epoch_updates=80,
                  reason="Existing PPO with 256-step rollouts exactly divides 4096.",
                  requested_device=args.device)
    write_json(out / "config.json", config)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for filename in ("__init__.py", "server_smoke.py", "experiments.py", "sine_env.py",
                     "acoustics.py", "plotting.py", "configs/phase0.json"):
        dest = snapshot / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, dest)
    source_hashes = {str(p.relative_to(snapshot)): digest(p)
                     for p in snapshot.rglob("*") if p.is_file()}
    write_json(out / "source_sha256.json", source_hashes)
    train_factory, eval_factory = phase0_factories(config)
    train_env = train_factory()
    agent = make_agent("PPO", train_env, 0, config, device=device)
    placements = verify_placement(agent)
    write_json(out / "parameter_devices_before_training.json", placements)
    write_json(out / "sine_config.json", dict(training=asdict(train_env.config),
               evaluation=asdict(SineConfig(randomize=False, target=config["target"]))))
    config.update(n_envs=agent.n_envs, n_steps=agent.n_steps, batch_size=agent.batch_size,
                  n_epochs=agent.n_epochs, gamma=agent.gamma, gae_lambda=agent.gae_lambda,
                  ent_coef=agent.ent_coef, vf_coef=agent.vf_coef,
                  max_grad_norm=agent.max_grad_norm, target_kl=agent.target_kl,
                  clip_range=float(agent.clip_range(1)),
                  normalize_advantage=agent.normalize_advantage,
                  policy=str(agent.policy), optimizer=str(agent.policy.optimizer),
                  policy_kwargs=agent.policy_kwargs, device=str(agent.device))
    write_json(out / "config.json", config)
    environment.update(parameter_devices=placements,
                       policy_devices=sorted({str(p.device) for p in agent.policy.parameters()}))
    write_json(environment_path, environment)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (out / "environment-freeze.txt").write_text(freeze.stdout)

    metrics = []
    class AuditLogger(Logger):
        def record(self, key, value, exclude=None):
            finite_tree(key, value)
            if key.startswith("train/"):
                row = dict(environment_steps=agent.num_timesteps, key=key,
                           value=float(value))
                metrics.append(row)
                with (out / "training_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
            super().record(key, value, exclude)

    agent.set_logger(AuditLogger(str(out), [make_output_format(fmt, str(out))
                                           for fmt in ("stdout", "log", "csv", "json")]))
    observed_cpu_ids = set()
    getcpu = ctypes.CDLL(None).sched_getcpu
    class AuditCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rollouts = 0
            self.completed_episodes = 0
            self.transitions = []

        def _on_step(self):
            observed_cpu_ids.add(getcpu())
            for key in ("new_obs", "rewards", "actions", "values", "log_probs"):
                finite(key, self.locals[key])
            finite_tree("infos", self.locals["infos"])
            self.completed_episodes += int(np.count_nonzero(self.locals["dones"]))
            self.transitions.append(dict(environment_step=self.num_timesteps,
                                         reward=float(self.locals["rewards"][0]),
                                         done=bool(self.locals["dones"][0])))
            return True

        def _on_rollout_end(self):
            self.rollouts += 1
            for name in ("observations", "actions", "rewards", "values", "log_probs", "advantages", "returns"):
                finite(f"rollout/{name}", getattr(self.model.rollout_buffer, name))

    callback = AuditCallback()
    optimizer_steps = []
    gradient_devices, loss_devices = set(), set()
    def before_step(optimizer, _args, _kwargs):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    finite("gradient", param.grad)
                    gradient_devices.add(str(param.grad.device))
                    assert param.grad.device == device

    def after_step(optimizer, _args, _kwargs):
        for group in optimizer.param_groups:
            for param in group["params"]:
                finite("parameter", param)
                assert param.device == device
        for state in optimizer.state.values():
            finite_tree("optimizer_state", state)
        observed_cpu_ids.add(getcpu())
        optimizer_steps.append(dict(optimizer_step=len(optimizer_steps)+1,
                                    environment_steps=agent.num_timesteps))

    hooks = [agent.policy.optimizer.register_step_pre_hook(before_step),
             agent.policy.optimizer.register_step_post_hook(after_step)]
    minibatch_losses = []
    class LossAudit(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is torch.Tensor.backward:
                finite("minibatch_loss", args[0])
                loss_devices.add(str(args[0].device))
                assert args[0].device == device
                minibatch_losses.append(dict(environment_steps=agent.num_timesteps,
                                             loss=float(args[0].detach())))
            return func(*args, **(kwargs or {}))

    initial_policy_hash = policy_hash(agent)
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start, cpu_start = time.perf_counter(), time.process_time()
    with LossAudit():
        agent.learn(total_timesteps=4096, callback=callback, log_interval=1)
    synchronize()
    train_seconds, train_cpu_seconds = time.perf_counter()-train_start, time.process_time()-cpu_start
    cuda_peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    write_json(out / "parameter_devices_after_training.json", verify_placement(agent))
    agent.logger.dump(agent.num_timesteps)  # SB3 normally dumps before each train call.
    for hook in hooks:
        hook.remove()
    assert agent.num_timesteps == len(callback.transitions) == 4096
    assert callback.rollouts == 16
    assert agent._n_updates == 80
    assert len(optimizer_steps) == 320
    assert len(minibatch_losses) == 320
    assert initial_policy_hash != policy_hash(agent)
    assert len({m["environment_steps"] for m in metrics if m["key"] == "train/value_loss"}) == 16
    write_json(out / "optimizer_steps.json", optimizer_steps)
    write_json(out / "minibatch_losses.json", minibatch_losses)
    write_json(out / "training_transitions.json", callback.transitions)
    config["trained_policy_sha256"] = policy_hash(agent)
    write_json(out / "config.json", config)
    synchronize()
    save_start = time.perf_counter()
    agent.save(out / "agent.zip")
    synchronize()
    save_seconds = time.perf_counter()-save_start
    result, data = evaluate(agent, eval_factory)
    write_json(out / "before_reload.json", result)
    np.savez_compressed(out / "before_reload.npz", **data)
    agent.get_env().close()
    reload_start = time.perf_counter()
    with (out / "reload.log").open("x") as log:
        subprocess.run([sys.executable, "-m", "Deeplearning.server_smoke", "--output", str(out),
                        "--evaluate-only", "--device", args.device], cwd=ROOT.parent,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    reload_seconds = time.perf_counter()-reload_start
    evaluation = json.loads((out / "evaluation.json").read_text())
    assert evaluation["pid"] != os.getpid()
    assert evaluation["reload_passed"]
    previous = out / "original_files_sha256.json"
    preservation = {}
    if previous.exists():
        original = json.loads(previous.read_text())
        changed = [path for path, old_hash in original.items()
                   if not Path(path).is_file() or digest(Path(path)) != old_hash]
        preservation = dict(original_file_count=len(original), changed_original_files=changed)
        assert not changed, changed
    summary = dict(status="PASS", algorithm="PPO", seed=0, training_environment_steps=4096,
                   n_envs=agent.n_envs, completed_training_episodes=callback.completed_episodes,
                   partial_episode_steps=4096 % 35, rollout_training_calls=callback.rollouts,
                   sb3_n_updates=agent._n_updates,
                   sb3_n_updates_meaning="PPO epoch counter, not environment steps or optimizer steps",
                   epochs_per_rollout=agent.n_epochs, minibatch_optimizer_steps=len(optimizer_steps),
                   optimizer_step_meaning="Shared Adam step updates policy and value networks together",
                   evaluation_environment_steps=2*35, evaluation_before_save_reload_steps=35,
                   evaluation_after_reload_steps=35, training_plus_evaluation_environment_steps=4166,
                   finite_checks="All training rewards/observations/actions/values/log probabilities; all rollout returns/advantages; all 320 backward losses and every logged metric; gradients, parameters and Adam states at all 320 optimizer steps; evaluation/audio",
                   nonfinite_detected=0, trained_policy_changed=True,
                   model_reload_passed=True, reload_exact_matches=evaluation["exact_array_matches"],
                   final_acoustic_error=evaluation["error"],
                   cpu_model=cpu_model, device=str(agent.device), observed_cpu_ids=sorted(observed_cpu_ids),
                   policy_devices=environment["policy_devices"], gpu_used=device.type == "cuda",
                   parameter_devices=placements, gradient_devices=sorted(gradient_devices),
                   loss_devices=sorted(loss_devices),
                   gpu_logical_index=environment.get("gpu_logical_index"),
                   gpu_name=environment.get("gpu_name"), gpu_uuid=environment.get("gpu_uuid"),
                   cuda_visible_devices=environment["cuda_visible_devices"],
                   training_cuda_peak_allocated_bytes=cuda_peak_memory,
                   training_wall_includes_cuda_synchronization=device.type == "cuda",
                   reload_array_differences=evaluation["array_differences"],
                   reload_acoustic_error_abs_difference=evaluation["acoustic_error_abs_difference"],
                   training_wall_seconds=train_seconds, training_process_cpu_seconds=train_cpu_seconds,
                   model_save_seconds=save_seconds, reload_subprocess_wall_seconds=reload_seconds,
                   total_run_wall_seconds=time.perf_counter()-start,
                   max_parent_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   finished_at=now(), **preservation)
    write_json(out / "summary.json", summary)
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob("*")
                 if p.is_file() and not any(part in (".venv", "mpl-cache", "pycache")
                                           for part in p.relative_to(out).parts)
                 and p.name not in ("artifacts_sha256.json", "execution.log")}
    write_json(out / "artifacts_sha256.json", artifacts)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            failure_out = Path(sys.argv[sys.argv.index("--output")+1]).resolve()
            if failure_out.is_dir():
                filename = "reload_failure.json" if "--evaluate-only" in sys.argv else "failure.json"
                write_json(failure_out / filename, dict(status="ERROR", error_type=type(error).__name__,
                           error=str(error), traceback=traceback.format_exc(), finished_at=now(),
                           python=sys.executable, hostname=socket.gethostname(), pid=os.getpid()))
        raise
