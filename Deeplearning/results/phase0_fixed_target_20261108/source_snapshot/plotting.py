"""Headless, reproducible figures using the legacy SciencePlots conventions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scienceplots  # noqa: F401 registers styles
from scipy import signal

from .acoustics import waveform, resample


# Match the user's original recording-to-simulation display pipeline exactly.
# Upsampling is for display only; it does not add original high-frequency content.
TARGET_FS = 10_000_000
NPERSEG = 245760
NOVERLAP = 184320
WINDOW_TYPE = "blackmanharris"
CMAP = "gray_r"
VMIN = -50.0
VMAX = -20.0
MAX_HZ = 10000
PHASE0_MAX_HZ = 2500
CONTROL_LABELS = (r"$\varepsilon_l$", r"$\varepsilon_r$", r"$p_s$",
                  r"$f_{e,l}$", r"$f_{e,r}$")


def control_trajectories(controls, output, *, control_s=.01):
    """Five actual model controls in research notation and legacy plot styling."""
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 5 or not len(controls) or not np.isfinite(controls).all():
        raise ValueError("Expected a finite N-by-5 applied-control trajectory")
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)
        try:
            # Include the held final interval through the episode boundary.
            times = np.arange(len(controls)+1)*control_s
            for i, ax in enumerate(axes):
                ax.step(times, np.r_[controls[:, i], controls[-1, i]], where="post", linewidth=2)
                ax.set_ylabel(CONTROL_LABELS[i], fontsize=46)
                ax.tick_params(labelsize=38, width=2, length=10, direction="in", top=False, right=False)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
                ax.yaxis.get_offset_text().set_fontsize(30)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
            axes[-1].set_xlim(0, times[-1])
            axes[-1].set_xlabel("Time [s]", fontsize=46)
            fig.tight_layout()
            fig.savefig(output, bbox_inches="tight", pad_inches=.1, dpi=120)
        finally:
            plt.close(fig)


def display_spectrum(x, rate):
    """Display-only PSD: complete windows, no padding or boundary frames.

    Match the legacy spectrogram's mean detrending, density scaling and per-plot
    peak normalization. This function does not change the acoustic training loss.
    """
    x = waveform(x)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample rate must be finite and positive")
    if rate != TARGET_FS:
        x = resample(x, rate, TARGET_FS)
    if len(x) < NPERSEG:
        raise ValueError("Audio is shorter than one complete spectrogram window")
    f, t, p = signal.spectrogram(x, fs=TARGET_FS, window=WINDOW_TYPE,
                                nperseg=NPERSEG, noverlap=NOVERLAP,
                                detrend="constant", scaling="density", mode="psd")
    # Normalize before the frequency crop, as in the original plotting script.
    db = 10*np.log10(p/max(float(p.max()), 1e-30)+1e-10)
    keep = f <= min(MAX_HZ, rate/2)
    return f[keep], t, db[keep]


def draw_spectrogram(ax, x, rate, *, max_hz=MAX_HZ):
    x = waveform(x)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample rate must be finite and positive")
    if len(x)/rate < (2*NPERSEG-NOVERLAP)/TARGET_FS:
        # A failed episode can be shorter than the display window. Still save its
        # evaluation report without inventing padded boundary spectra.
        ax.set_axis_off()
        ax.text(.5, .5, "Insufficient audio for interior STFT frames",
                ha="center", va="center", transform=ax.transAxes)
        return None
    f, t, db = display_spectrum(x, rate)
    if len(t) < 2:
        raise ValueError("At least two complete spectrogram windows are required for display")
    mesh = ax.pcolormesh(t, f, db, shading="gouraud", cmap=CMAP,
                         vmin=VMIN, vmax=VMAX, rasterized=True)
    ax.set_ylim(0, min(max_hz, MAX_HZ, rate/2))
    # Gouraud shading spans the actual frame centers; do not extrapolate pixels
    # back to t=0 or out to the end of the recording.
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylabel("Frequency [Hz]", fontsize=46)
    ax.tick_params(labelsize=30, width=2, length=10, direction="in", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    return mesh


def comparison(target, generated, rate, output, *, max_hz=MAX_HZ):
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        try:
            for ax, x, title in zip(axes, (target, generated), ("Target", "Generated")):
                draw_spectrogram(ax, x, rate, max_hz=max_hz)
                ax.set_title(title, fontsize=22)
            axes[-1].set_xlabel("Time [s]", fontsize=46)
            fig.tight_layout()
            fig.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=120)
        finally:
            plt.close(fig)


def single_spectrogram(x, rate, output):
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, ax = plt.subplots(figsize=(12, 12))
        try:
            draw_spectrogram(ax, x, rate)
            ax.tick_params(labelsize=46)
            ax.set_xlabel("Time [s]", fontsize=46)
            fig.tight_layout()
            fig.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=120)
        finally:
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


def phase0_learning_curves(curves, output):
    """Plot the nine recorded Phase 0 evaluations for each method's run 0.

    Keep the general learning_curves renderer unchanged for other experiments.
    Colors match the existing Phase 0 PNG, independently of dictionary order.
    """
    colors = {"PPO": "#0072B2", "TD3": "#009E73", "SAC": "#D55E00"}
    evaluation_steps = np.arange(0, 4097, 512)
    for name in colors:
        steps, errors = curves[name]
        if not np.array_equal(steps, evaluation_steps):
            raise ValueError(f"{name}: expected recorded evaluations at 0, 512, ..., 4096 steps")
        if np.shape(errors) != evaluation_steps.shape or not np.isfinite(errors).all():
            raise ValueError(f"{name}: expected nine finite recorded acoustic errors")

    # Use the same font family and label/tick sizes as draw_spectrogram.
    # Disable rendering simplification so all recorded vertices are retained.
    with plt.style.context(["science", "ieee", "no-latex"]), \
            plt.rc_context({"path.simplify": False}):
        fig, ax = plt.subplots(figsize=(16, 10))
        try:
            for name, color in colors.items():
                steps, errors = curves[name]
                ax.plot(steps, errors, color=color, alpha=.8, linewidth=2.8,
                        linestyle="-", label=name)
            ax.set_xlabel("Training steps", fontsize=46)
            ax.set_ylabel("Acoustic error", fontsize=46)
            ax.set_xlim(0, 4096)
            ax.set_xticks(evaluation_steps)
            ax.tick_params(labelsize=30, width=2, length=10, direction="in",
                           top=False, right=False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            ax.legend(fontsize=30, loc="upper right", ncol=3)
            fig.tight_layout()
            fig.savefig(output, bbox_inches="tight", pad_inches=.1, dpi=120)
        finally:
            plt.close(fig)


def phase0_all_seed_learning_curves(curves, output):
    """Plot all nine recorded runs; color is method, marker is run 0/1/2.

    ``curves[method][seed]`` contains the original (steps, errors) arrays.
    Separate legend rows identify method colors and run marker shapes.
    """
    colors = {"PPO": "#0072B2", "TD3": "#009E73", "SAC": "#D55E00"}
    markers = {0: "o", 1: "s", 2: "^"}
    evaluation_steps = np.arange(0, 4097, 512)
    for name in colors:
        for seed in markers:
            steps, errors = curves[name][seed]
            if not np.array_equal(steps, evaluation_steps):
                raise ValueError(f"{name}, run {seed}: expected evaluations at 0, 512, ..., 4096 steps")
            if (np.shape(errors) != evaluation_steps.shape or
                    not np.isfinite(errors).all() or np.any(np.asarray(errors) < 0)):
                raise ValueError(f"{name}, run {seed}: expected nine finite nonnegative acoustic errors")

    maximum = max(max(curves[name][seed][1]) for name in colors for seed in markers)
    upper = max(.2, float(np.ceil(maximum * 1.2 * 10) / 10))
    with plt.style.context(["science", "ieee", "no-latex"]), \
            plt.rc_context({"path.simplify": False}):
        fig, ax = plt.subplots(figsize=(16, 10))
        try:
            for name, color in colors.items():
                for seed, marker in markers.items():
                    steps, errors = curves[name][seed]
                    ax.plot(steps, errors, color=color, alpha=.8, linewidth=2.8,
                            linestyle="-", solid_capstyle="round", solid_joinstyle="round",
                            marker=marker, markersize=14, markerfacecolor="white",
                            markeredgecolor=color, markeredgewidth=2.3, clip_on=False,
                            label=f"{name} ({seed})")
            ax.set_xlabel("Training steps", fontsize=46)
            ax.set_ylabel("Acoustic error", fontsize=46)
            ax.set_xlim(0, 4096)
            ax.set_xticks(evaluation_steps)
            ax.set_ylim(0, upper)
            ax.tick_params(labelsize=30, width=2, length=10, direction="in",
                           top=False, right=False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            method_handles = [Line2D([], [], color=color, alpha=.8, linewidth=2.8,
                                     linestyle="-", label=name)
                              for name, color in colors.items()]
            method_legend = ax.legend(handles=method_handles, fontsize=30,
                                      loc="upper right", ncol=3, frameon=False)
            ax.add_artist(method_legend)
            seed_handles = [Line2D([], [], color="black", linestyle="None", marker=marker,
                                   markersize=14, markerfacecolor="white", markeredgewidth=2.3,
                                   label=f"seed {seed}")
                            for seed, marker in markers.items()]
            ax.legend(handles=seed_handles, fontsize=30, loc="upper right",
                      bbox_to_anchor=(1, .91), ncol=3, frameon=False,
                      handlelength=1, handletextpad=.5, columnspacing=1.5)
            fig.tight_layout()
            fig.savefig(output, bbox_inches="tight", pad_inches=.1, dpi=120)
        finally:
            plt.close(fig)


def phase1_task_a_learning_curves(curves, output):
    """Draw only the saved run-0 Task A acoustic-error series, without aggregation."""
    colors = {"TD3": "#0072B2", "SAC": "#D55E00", "Random search": "#009E73",
              "Differential evolution": "#CC79A7"}
    evaluation_episodes = np.arange(0, 257, 32)
    for name in colors:
        episodes, errors = curves[name]
        if not np.array_equal(episodes, evaluation_episodes):
            raise ValueError(f"{name}: expected recorded evaluations at 0, 32, ..., 256 episodes")
        if (np.shape(errors) != evaluation_episodes.shape or
                not np.isfinite(errors).all() or np.any(np.asarray(errors) < 0)):
            raise ValueError(f"{name}: expected nine finite nonnegative recorded acoustic errors")

    # Reserve the upper part of the axes for the four-line legend. A small lower
    # margin keeps near-zero DE values visible above the bottom frame, unchanged.
    maximum = max(max(curves[name][1]) for name in colors)
    upper = max(.1, float(np.ceil(maximum * 1.5 * 10) / 10))
    with plt.style.context(["science", "ieee", "no-latex"]), \
            plt.rc_context({"path.simplify": False}):
        fig, ax = plt.subplots(figsize=(16, 10))
        try:
            for name, color in colors.items():
                episodes, errors = curves[name]
                ax.plot(episodes, errors, color=color, linewidth=2.8,
                        linestyle="-", label=name)
            ax.set_xlabel("Training episodes / candidates", fontsize=46)
            ax.set_ylabel("Full-episode acoustic error", fontsize=46)
            ax.set_xlim(0, 256)
            ax.set_xticks(evaluation_episodes)
            ax.set_ylim(-upper * .03, upper)
            ax.tick_params(labelsize=30, width=2, length=10, direction="in",
                           top=False, right=False)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            ax.legend(fontsize=30, loc="upper right")
            fig.tight_layout()
            fig.savefig(output, bbox_inches="tight", pad_inches=.1, dpi=120)
        finally:
            plt.close(fig)
