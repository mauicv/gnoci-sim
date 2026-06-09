import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REAL_COLOR = "#2563EB"
SIM_COLOR  = "#EA580C"

N_STATES = 32
COLS = 8
ROWS = N_STATES // COLS  # 4

_J = ["L head\nyoke", "L yoke\nhip", "L hip\nupper", "L upper\nlower", "L lower\nfoot",
      "R head\nyoke", "R yoke\nhip", "R hip\nupper", "R upper\nlower", "R lower\nfoot"]

STATE_LABELS = (
    [f"{j}\npos" for j in _J]
  + [f"{j}\nvel" for j in _J]
  + ["fwd L\ncontact", "bck L\ncontact", "fwd R\ncontact", "bck R\ncontact"]
  + ["gyro x", "gyro y", "gyro z"]
  + ["acc x",  "acc y",  "acc z"]
  + ["roll", "pitch"] # NOTE: swapped for real data
)

with open("comparisons/real_state_data.json") as f:
    real_data = json.load(f)

with open("comparisons/simulated_state_data.json") as f:
    sim_data = json.load(f)

real_states = np.array(real_data["states"], dtype=float)
sim_states  = np.array(sim_data["states"],  dtype=float)
real_times  = np.array(real_data["times"],  dtype=float) - real_data["times"][0]
sim_times   = np.array(sim_data["times"],   dtype=float) - sim_data["times"][0]

real_states = real_states[15:]
sim_states  = sim_states[15:]
real_times  = real_times[15:]
sim_times   = sim_times[15:]

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

fig, axs = plt.subplots(ROWS, COLS, figsize=(22, 9), sharex=False)
fig.suptitle("State Comparison — Real vs Simulated", fontsize=12, fontweight="bold",
             color="#111827", y=0.999)

for i in range(N_STATES):
    row, col = divmod(i, COLS)
    ax = axs[row, col]

    ax.plot(real_times, real_states[:, i], color=REAL_COLOR, linewidth=1.0, zorder=3)
    ax.plot(sim_times,  sim_states[:, i],  color=SIM_COLOR,  linewidth=1.0,
            linestyle="--", zorder=2)

    ax.set_title(STATE_LABELS[i], pad=4)
    if i < 20:
        ax.set_ylim(-1, 1)
    elif 24 <= i <= 26:
        ax.set_ylim(-1.5, 1.5)
    elif 27 <= i <= 29:
        ax.set_ylim(-2, 2)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.tick_params(length=2)
    ax.spines["top"].set_visible(False)

    if row == ROWS - 1:
        ax.set_xlabel("time (s)", labelpad=3)

handles = [
    plt.Line2D([0], [0], color=REAL_COLOR, linewidth=1.6,                  label="Real"),
    plt.Line2D([0], [0], color=SIM_COLOR,  linewidth=1.6, linestyle="--",  label="Sim"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           frameon=True, framealpha=0.9, edgecolor="#D1D5DB",
           fontsize=9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.997])
plt.savefig("comparisons/state_curves.png", dpi=150, bbox_inches="tight")
plt.show()
