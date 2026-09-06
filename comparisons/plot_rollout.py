"""
Plot the per-step actions and policy states recorded in one or more rollout
JSONs (the {"actions": [...], "states": [...]} format written by
generate_rollout.py and replay_rollout.py), overlaid so rollouts can be
compared directly. Actions and states are saved as two separate PNG files,
since they're different shapes and not meaningfully overlaid on each other.

By default, plots every *.json file under comparisons/rollouts/. Pass
explicit paths to plot a different set instead.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from gnoci_gym.config import CONTROL_HZ

comparisons_dir = os.path.dirname(__file__)
default_rollouts_dir = os.path.join(comparisons_dir, "rollouts")

PALETTE = ["#2563EB", "#EA580C", "#16A34A", "#DB2777", "#7C3AED", "#0891B2", "#CA8A04"]

_J = ["L head\nyoke", "L yoke\nhip", "L hip\nupper", "L upper\nlower", "L lower\nfoot",
      "R head\nyoke", "R yoke\nhip", "R hip\nupper", "R upper\nlower", "R lower\nfoot"]

ACTION_LABELS = _J

STATE_LABELS = (
    [f"{j}\npos" for j in _J]
  + [f"{j}\nvel" for j in _J]
  + ["fwd L\ncontact", "bck L\ncontact", "fwd R\ncontact", "bck R\ncontact"]
  + ["gyro x", "gyro y", "gyro z"]
  + ["acc x",  "acc y",  "acc z"]
  + ["roll", "pitch"]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "axes.labelcolor": "#374151",
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.titlesize": 7.5,
    "axes.titlecolor": "#111827",
    "axes.titlepad": 6,
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
})

import math

def _load_rollout(path):
    with open(path) as f:
        d = json.load(f)
    states = d["states"][:50*3]
    if "target_actions" in d:
        actions = d["target_actions"][:50*3]
    else:
        actions = (np.array(d["actions"][:50*3]) * 0.75 * math.pi).tolist()
    return {
        "states": np.array(states, dtype=float),
        "actions": np.array(actions, dtype=float),
    }


def _plot_grid(rollouts, key, labels, cols, title, out_path):
    n_dims = rollouts[0][1][key].shape[1]
    rows = -(-n_dims // cols)  # ceil division

    fig, axs = plt.subplots(rows, cols, figsize=(2.75 * cols, 2.2 * rows), sharex=False)
    axs = np.atleast_2d(axs)
    fig.suptitle(title, fontsize=12, fontweight="bold", color="#111827")

    colors = list(itertools.islice(itertools.cycle(PALETTE), len(rollouts)))

    for i in range(rows * cols):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        if i >= n_dims:
            ax.axis("off")
            continue
        for (label, data), color in zip(rollouts, colors):
            arr = data[key]
            times = np.arange(len(arr)) / CONTROL_HZ
            ax.plot(times, arr[:, i], color=color, linewidth=1.0, label=label)
        ax.set_title(labels[i], pad=4)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.tick_params(length=2)
        if row == rows - 1:
            ax.set_xlabel("time (s)", labelpad=3)

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.6, label=label)
        for (label, _), color in zip(rollouts, colors)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(rollouts), 4),
               frameon=True, framealpha=0.9, edgecolor="#D1D5DB",
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rollouts", nargs="*",
        help=f"rollout JSON files to overlay (default: all *.json under {default_rollouts_dir})",
    )
    parser.add_argument("--out-dir", default=comparisons_dir)
    args = parser.parse_args()

    paths = [Path(p) for p in args.rollouts] or sorted(Path(default_rollouts_dir).glob("*.json"))
    if not paths:
        raise SystemExit(f"No rollout JSON files found under {default_rollouts_dir}")

    rollouts = [(p.stem, _load_rollout(p)) for p in paths]

    _plot_grid(
        rollouts, "states", STATE_LABELS, cols=8, title="Rollout States",
        out_path=os.path.join(args.out_dir, "rollout_states.png"),
    )
    _plot_grid(
        rollouts, "actions", ACTION_LABELS, cols=5, title="Rollout Actions",
        out_path=os.path.join(args.out_dir, "rollout_actions.png"),
    )


if __name__ == "__main__":
    main()
