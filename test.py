import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "src")
from gnoci_gym.filters import EMAFilter


def plot_ema_smoothing(n_steps=100, n_joints=10, alpha=0.4):
    rng = np.random.default_rng(42)
    actions = rng.uniform(-1, 1, size=(n_steps, n_joints))

    filters = [EMAFilter(alpha=alpha) for _ in range(n_joints)]
    smoothed = np.array([
        [f.update(actions[t, i]) for i, f in enumerate(filters)]
        for t in range(n_steps)
    ])

    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2 * n_joints), sharex=True)
    fig.suptitle(f"EMA action smoothing  (α={alpha})", fontsize=13)

    for i, ax in enumerate(axes):
        ax.plot(actions[:, i],   color="steelblue", alpha=0.5, linewidth=1, label="raw")
        ax.plot(smoothed[:, i],  color="tomato",    linewidth=1.5,           label="smoothed")
        ax.set_ylabel(f"joint {i}", fontsize=8)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, linewidth=0.4)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_ema_smoothing()
