"""Headless, reproducible figures using the legacy SciencePlots conventions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 registers styles

from .acoustics import spectrum


def comparison(target, generated, rate, output, *, legacy_scale=False):
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        for ax, x, title in zip(axes, (target, generated), ("Target", "Generated")):
            f, t, p = spectrum(x, rate)
            db = 10*np.log10(p/max(float(p.max()), 1e-30)+1e-10)
            ax.pcolormesh(t, f, db, shading="auto", cmap="gray_r",
                          vmin=-38.002 if legacy_scale else -80,
                          vmax=-38 if legacy_scale else 0, rasterized=True)
            ax.set_ylim(0, min(10000, rate/2))
            ax.set_xlim(0, len(x)/rate)
            ax.set_ylabel("Frequency [Hz]", fontsize=46)
            ax.set_title(title, fontsize=22)
            ax.tick_params(labelsize=30, width=2, length=10, direction="in", top=False, right=False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        axes[-1].set_xlabel("Time [s]", fontsize=46)
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=120)
        plt.close(fig)


def learning_curves(curves, output):
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        for group, (name, entries) in enumerate(curves.items()):
            for i, (steps, errors) in enumerate(entries):
                ax.plot(steps, errors, color=colors[group % len(colors)],
                        linestyle=["-", "--", ":"][i % 3],
                        alpha=0.8, label=name if i == 0 else None)
        ax.set(xlabel="Training environment steps", ylabel="Evaluation acoustic error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
