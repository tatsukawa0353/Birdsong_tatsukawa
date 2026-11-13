"""Verify the independent RL runtime and optionally evaluate existing checkpoints."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import traceback
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", nargs=2, action="append", default=[],
                        metavar=("DEVICE", "CHECKPOINT"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(started_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
                  hostname=socket.gethostname(), python=sys.executable,
                  python_version=platform.python_version(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  training_environment_steps=0, optimizer_updates=0)
    try:
        with tempfile.TemporaryDirectory(prefix="birdsong-rl-verify-") as temporary:
            os.environ["MPLCONFIGDIR"] = temporary
            os.environ["MPLBACKEND"] = "Agg"
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[key] = "1"
            import numpy as np
            import scipy
            import matplotlib
            import scienceplots
            import gymnasium
            import torch
            from stable_baselines3 import PPO
            from Deeplearning.experiments import phase0_factories, rollout

            torch.set_num_threads(1)
            result["packages"] = {name: importlib.metadata.version(name) for name in
                                  ("torch", "stable-baselines3", "gymnasium", "numpy", "scipy",
                                   "matplotlib", "SciencePlots", "cloudpickle", "Farama-Notifications")}
            result["torch_path"] = torch.__file__
            result["prefix"] = sys.prefix
            result["cuda_available"] = torch.cuda.is_available()
            result["cuda_device_count"] = torch.cuda.device_count()
            query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
                                    "--format=csv,noheader"], capture_output=True, text=True)
            result["nvidia_smi"] = dict(returncode=query.returncode,
                                        stdout=query.stdout, stderr=query.stderr)
            if not result["cuda_available"]:
                raise RuntimeError("CUDA unavailable in this execution environment; no CPU fallback")
            gpu = torch.device("cuda:0")
            tensor = torch.arange(16, dtype=torch.float32, device=gpu).reshape(4, 4)
            product = tensor @ tensor.T
            torch.cuda.synchronize(gpu)
            expected = np.arange(16, dtype=np.float32).reshape(4, 4)
            expected = expected @ expected.T
            np.testing.assert_array_equal(product.cpu().numpy(), expected)
            assert product.device == gpu and torch.isfinite(product).all()
            result["gpu_tensor_test"] = dict(device=str(product.device),
                                              gpu_name=torch.cuda.get_device_name(gpu),
                                              max_abs_error=0.0, finite=True)
            result["models"] = []
            evaluation_steps = 0
            for device_name, filename in args.model:
                device = torch.device(device_name)
                path = Path(filename).resolve(strict=True)
                config = json.loads((path.parent / "config.json").read_text())
                agent = PPO.load(path, device=device)
                placements = {name: str(param.device) for name, param in agent.policy.named_parameters()}
                assert agent.device == device and set(placements.values()) == {str(device)}

                def policy_hash():
                    h = hashlib.sha256()
                    for key, value in agent.policy.state_dict().items():
                        assert torch.isfinite(value).all()
                        h.update(key.encode())
                        h.update(value.detach().cpu().numpy().tobytes())
                    return h.hexdigest()

                before = policy_hash()
                before_counters = (agent.num_timesteps, agent._n_updates)
                assert before == config["trained_policy_sha256"]
                _, factory = phase0_factories(config)
                env = factory()
                try:
                    metrics, audio, actions = rollout(
                        env, lambda obs, _: agent.predict(obs, deterministic=True)[0], seed=0)
                    assert not metrics["failed"] and metrics["env_steps"] == 35
                    assert np.isfinite(audio).all() and np.isfinite(actions).all()
                    with np.load(path.parent / "evaluation.npz") as reference:
                        differences = {
                            "audio_max_abs": float(np.max(np.abs(audio-reference["audio"]))),
                            "actions_max_abs": float(np.max(np.abs(actions-reference["actions"]))) }
                    recorded = json.loads((path.parent / "evaluation.json").read_text())
                    differences["acoustic_error_abs"] = abs(metrics["error"]-recorded["error"])
                    assert all(value == 0 for value in differences.values()), differences
                    assert before == policy_hash()
                    assert before_counters == (agent.num_timesteps, agent._n_updates)
                    evaluation_steps += metrics["env_steps"]
                    result["models"].append(dict(checkpoint=str(path), device=str(device),
                        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        parameter_devices=placements, policy_sha256=before,
                        saved_training_steps=agent.num_timesteps, saved_epoch_updates=agent._n_updates,
                        evaluation_environment_steps=metrics["env_steps"], error=metrics["error"],
                        differences_from_original_evaluation=differences, learner_unchanged=True))
                finally:
                    env.close()
            result.update(status="PASS", evaluation_environment_steps=evaluation_steps)
    except Exception as error:
        result.update(status="ERROR", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["finished_at"] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
