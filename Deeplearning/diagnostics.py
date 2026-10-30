"""Synthetic-only diagnostics and descriptors; recording-derived files are excluded."""
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from .acoustics import distance
from .evaluation import descriptors
from .experiments import rollout, write_json
from .model import ROOT
from .model_env import ModelConfig, ModelEnv
from .plotting import plt


def main():
    output = ROOT/"results/diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    features = []
    for path in sorted((ROOT/"results").rglob("*.npz")):
        with np.load(path) as data:
            if not {"target", "audio", "sample_rate"}.issubset(data.files):
                continue
            rate = float(data["sample_rate"])
            features.append(dict(artifact=str(path.relative_to(ROOT)),
                                 target=descriptors(data["target"], rate),
                                 generated=descriptors(data["audio"], rate)))
            if path.name == "rollout.npz":
                controls = data["actions"]
                if rate == 48000:
                    controls = controls.copy().astype(float)
                    controls[:, 0] = 500+(controls[:, 0]+1)/2*2500
                    controls[:, 1] = (controls[:, 1]+1)/2
                    labels = ("Frequency [Hz]", "Amplitude")
                else:
                    controls = data["physical_controls"][:, [0, 2]].copy()
                    labels = ("epsilon_l", "ps")
                with plt.style.context(["science", "ieee", "no-latex"]):
                    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
                    for i, ax in enumerate(axes):
                        ax.step(np.arange(len(controls))*.01, controls[:, i], where="post")
                        ax.set_ylabel(labels[i])
                    axes[-1].set_xlabel("Time [s]")
                    fig.tight_layout()
                    fig.savefig(path.parent/"controls.png", dpi=150)
                    plt.close(fig)
    write_json(output/"descriptors.json", features)
    # Evaluate one existing policy at several clocks/penalty weights. No retraining;
    # these quantify deployment sensitivity only, not training-hyperparameter sensitivity.
    agent = SAC.load(ROOT/"results/phase1_extended/SAC_0/agent.zip", device="cpu")
    sensitivity = []
    for control_s in (.005, .01, .0175):
        for weight in (0, .01, .1):
            env = ModelEnv(ModelConfig(control_s=control_s, smooth_weight=weight))
            try:
                result, _, _ = rollout(env, lambda obs, _: agent.predict(obs, deterministic=True)[0])
                sensitivity.append(dict(control_s=control_s, smooth_weight=weight,
                                        episode_return=float(sum(env.rewards)), **result))
            finally:
                env.close()
    write_json(output/"evaluation_sensitivity.json", sensitivity)
    glide = []
    for policy in ("constant", "random", "oracle"):
        env = ModelEnv(ModelConfig(target="glide"))
        rng = np.random.default_rng(0)
        policies = dict(constant=lambda obs, e: np.zeros(2),
                        random=lambda obs, e: rng.uniform(-1, 1, 2),
                        oracle=lambda obs, e: e.encode(e.target_controls[e.index]))
        try:
            result, audio, actions = rollout(env, policies[policy])
            glide.append(dict(policy=policy, **result))
            np.savez_compressed(output/f"model_glide_{policy}.npz", target=env.target_audio,
                                audio=audio, actions=actions, sample_rate=env.config.sample_rate)
        finally:
            env.close()
    write_json(output/"model_glide.json", glide)


if __name__ == "__main__":
    main()
